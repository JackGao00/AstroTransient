"""Step 4: LSTM v2 — 更简单的架构, 更多训练"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES

DEVICE = torch.device("cpu")

print("Loading data...")
train_df, _, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=1)

np.random.seed(42)
MAX_SEQ, SAMPLES_PER = 100, 500

all_objects = []
for cls_id in sorted(train_df["label"].unique()):
    obj_idx = train_df[train_df["label"] == cls_id].index.values
    n = min(SAMPLES_PER, len(obj_idx))
    chosen = np.random.choice(obj_idx, size=n, replace=False)
    for idx in chosen:
        all_objects.append((idx, cls_id))

print(f"Building {len(all_objects)} sequences...")
sequences, labels, skipped = [], [], 0
for idx, cls_id in all_objects:
    try:
        mjd, flux, flux_err, pbs = get_lightcurve(train_df.iloc[idx])
    except:
        skipped += 1; continue
    mask = flux_err > 0
    if mask.sum() < 5:
        skipped += 1; continue

    # Robust normalization
    f_median = np.median(flux[mask])
    f_mad = np.median(np.abs(flux[mask] - f_median))
    if f_mad < 1e-6:
        f_mad = 1.0
    flux_n = (flux - f_median) / f_mad
    flux_err_n = flux_err / f_mad

    n = min(len(mjd), MAX_SEQ)
    seq = np.zeros((MAX_SEQ, 4))  # +1 channel: passband
    seq[:n, 0] = (mjd[:n] - mjd[0]) / 100.0  # scaled time
    seq[:n, 1] = flux_n[:n]
    seq[:n, 2] = np.clip(flux_err_n[:n], 0, 10)
    seq[:n, 3] = pbs[:n] / 10000.0  # scaled wavelength
    sequences.append(seq)
    labels.append(cls_id)

print(f"Sequences: {len(sequences)}, skipped: {skipped}")

X_seq = np.array(sequences, dtype=np.float32)
y_seq = np.array(labels, dtype=np.int64)

le = LabelEncoder()
y_enc = le.fit_transform(y_seq)
num_classes = len(le.classes_)
print(f"Classes: {num_classes}")

# Split
X_tmp, X_test, y_tmp, y_test = train_test_split(X_seq, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=42)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

B = 256
tl = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)), batch_size=B, shuffle=True)
vl = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)), batch_size=B)
ts = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)), batch_size=B)

# Simple 1-layer LSTM with skip connection
class SimpleLSTM(nn.Module):
    def __init__(self, nc=14):
        super().__init__()
        self.lstm = nn.LSTM(4, 128, 2, batch_first=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(128)
        self.fc = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, nc)
        )

    def forward(self, x):
        out, (h, _) = self.lstm(x)
        # Take last hidden state + mean pool
        last = out[:, -1, :]  # (B, H)
        mean_pool = out.mean(dim=1)
        combined = last + mean_pool  # skip-like
        combined = self.bn(combined)
        return self.fc(combined)

model = SimpleLSTM(nc=num_classes).to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

crit = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

best_acc, patience = 0, 0
print("\nTraining (60 epochs)...")
for ep in range(60):
    model.train()
    train_loss = 0
    for xb, yb in tl:
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_loss += loss.item()
    sch.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in vl:
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    acc = correct / total

    if acc > best_acc:
        best_acc = acc; patience = 0
        torch.save(model.state_dict(), "models/lstm_classifier.pt")
    else:
        patience += 1

    if ep % 10 == 0 or ep == 59:
        print(f"  Epoch {ep+1:2d} | Train Loss: {train_loss/len(tl):.4f} | Val Acc: {acc:.4f} | Best: {best_acc:.4f} | LR: {opt.param_groups[0]['lr']:.6f}")

    if patience >= 15:
        print(f"  Early stop at epoch {ep+1}")
        break

# Test
model.load_state_dict(torch.load("models/lstm_classifier.pt", map_location=DEVICE))
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in ts:
        preds.append(model(xb).argmax(1).numpy())
        trues.append(yb.numpy())

yp, yt = np.concatenate(preds), np.concatenate(trues)
acc = accuracy_score(yt, yp)
f1 = f1_score(yt, yp, average="macro")

print(f"\n{'='*50}")
print(f"LSTM v2 Final: Accuracy={acc:.4f}, Macro-F1={f1:.4f}")
print(f"{'='*50}")

cnames = [CLASS_NAMES.get(c, f"class_{c}") for c in le.classes_]
print(f"\n{classification_report(yt, yp, target_names=cnames, digits=3, zero_division=0)}")
print("DONE!")
