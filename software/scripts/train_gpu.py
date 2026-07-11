"""GPU Training: LSTM + MLP vs XGBoost 完整对比"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES, make_fixed_length_sequence

DEVICE = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")

# ============================================================
# 1. LOAD DATA & BUILD SEQUENCES
# ============================================================
print("\n[1/4] Loading data & building sequences...")
train_df, val_df, _ = load_plasticc("data/raw/plasticc", load_test=False)
all_df = train_df  # 7066 objects

np.random.seed(42)
MAX_SEQ, SAMPLES_PER = 150, 500

all_objects = []
for cls_id in sorted(all_df["label"].unique()):
    obj_idx = all_df[all_df["label"] == cls_id].index.values
    n = min(SAMPLES_PER, len(obj_idx))
    chosen = np.random.choice(obj_idx, size=n, replace=False)
    for idx in chosen:
        all_objects.append((idx, cls_id))

print(f"Building {len(all_objects)} sequences...")
sequences, labels, skipped = [], [], 0
t0 = time.time()
for idx, cls_id in all_objects:
    try:
        mjd, flux, flux_err, pbs = get_lightcurve(all_df.iloc[idx])
    except:
        skipped += 1; continue
    mask = flux_err > 0
    if mask.sum() < 5:
        skipped += 1; continue

    # Robust normalization
    f_med = np.median(flux[mask])
    f_mad = np.median(np.abs(flux[mask] - f_med)) + 1e-8
    flux_n = (flux - f_med) / f_mad
    flux_err_n = np.clip(flux_err / f_mad, 0, 10)

    n = min(len(mjd), MAX_SEQ)
    seq = np.zeros((MAX_SEQ, 4), dtype=np.float32)
    seq[:n, 0] = (mjd[:n] - mjd[0]) / 100.0
    seq[:n, 1] = flux_n[:n]
    seq[:n, 2] = flux_err_n[:n]
    seq[:n, 3] = pbs[:n] / 10000.0
    sequences.append(seq)
    labels.append(cls_id)

print(f"Done: {len(sequences)} sequences ({time.time()-t0:.1f}s), skipped: {skipped}")

X_seq = np.array(sequences, dtype=np.float32)
y_seq = np.array(labels, dtype=np.int64)

le = LabelEncoder()
y_enc = le.fit_transform(y_seq)
num_classes = len(le.classes_)
cnames = [CLASS_NAMES.get(c, f"class_{c}") for c in le.classes_]
print(f"Classes: {num_classes}")

# Split
X_tmp, X_test, y_tmp, y_test = train_test_split(X_seq, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=42)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ============================================================
# 2. LSTM MODEL
# ============================================================
print("\n[2/4] Training LSTM (GPU, 200 epochs)...")

class DeepLSTM(nn.Module):
    def __init__(self, input_size=4, hidden=256, num_layers=3, num_classes=14, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden * 2)
        self.attn = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, L, 2H)
        out = self.ln(out)
        attn_w = torch.softmax(self.attn(out).squeeze(-1), dim=1)  # (B, L)
        context = torch.sum(out * attn_w.unsqueeze(-1), dim=1)  # (B, 2H)
        # Add mean pooling for stability
        mean_pool = out.mean(dim=1)
        combined = context + mean_pool
        return self.fc(combined)

BATCH = 256
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH, pin_memory=True)

lstm = DeepLSTM(input_size=4, hidden=256, num_layers=3, num_classes=num_classes, dropout=0.35).cuda()
print(f"LSTM params: {sum(p.numel() for p in lstm.parameters()):,}")

# Class weights for imbalance
class_counts = np.bincount(y_train)
class_weights = torch.tensor([len(y_train) / (num_classes * max(c, 1)) for c in class_counts], dtype=torch.float32).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(lstm.parameters(), lr=0.0015, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, epochs=200, steps_per_epoch=len(train_loader))
scaler = GradScaler()

best_acc, patience = 0, 0
t0 = time.time()
for ep in range(200):
    lstm.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.cuda(), yb.cuda()
        optimizer.zero_grad()
        with autocast():
            loss = criterion(lstm(xb), yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss.item()

    lstm.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.cuda(), yb.cuda()
            with autocast():
                logits = lstm(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    acc = correct / total

    if acc > best_acc:
        best_acc = acc; patience = 0
        torch.save(lstm.state_dict(), "models/lstm_gpu.pt")
    else:
        patience += 1

    if ep % 20 == 0 or ep == 199:
        elapsed = time.time() - t0
        print(f"  Epoch {ep+1:3d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f} | Best: {best_acc:.4f} | {elapsed:.0f}s")

    if patience >= 25:
        print(f"  Early stop at epoch {ep+1}")
        break

print(f"LSTM training done in {time.time()-t0:.0f}s")

# Evaluate LSTM
lstm.load_state_dict(torch.load("models/lstm_gpu.pt"))
lstm.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.cuda()
        with autocast():
            logits = lstm(xb)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.cpu().numpy())

yp_lstm, yt = np.concatenate(preds), np.concatenate(trues)
lstm_acc = accuracy_score(yt, yp_lstm)
lstm_f1 = f1_score(yt, yp_lstm, average="macro")

# ============================================================
# 3. MLP MODEL (on same features as XGBoost)
# ============================================================
print("\n[3/4] Training MLP on XGBoost features (GPU, 200 epochs)...")

# Load features from step3
test_data = np.load("data/processed/test.npz")
X_feat, y_feat = test_data["X"], test_data["y"]

X_ft, X_fte, y_ft, y_fte = train_test_split(X_feat, y_feat, test_size=0.3, stratify=y_feat, random_state=42)
X_ftr, X_fv, y_ftr, y_fv = train_test_split(X_ft, y_ft, test_size=0.2, stratify=y_ft, random_state=42)
print(f"MLP Train: {len(X_ftr)}, Val: {len(X_fv)}, Test: {len(X_fte)}")

B2 = 512
ft_train = DataLoader(TensorDataset(torch.tensor(X_ftr.astype(np.float32)), torch.tensor(y_ftr, dtype=torch.long)), batch_size=B2, shuffle=True, pin_memory=True)
ft_val = DataLoader(TensorDataset(torch.tensor(X_fv.astype(np.float32)), torch.tensor(y_fv, dtype=torch.long)), batch_size=B2, pin_memory=True)
ft_test = DataLoader(TensorDataset(torch.tensor(X_fte.astype(np.float32)), torch.tensor(y_fte, dtype=torch.long)), batch_size=B2, pin_memory=True)

n_feat = X_feat.shape[1]

class DeepMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_out)
        )

    def forward(self, x):
        return self.net(x)

mlp = DeepMLP(n_feat, num_classes).cuda()
print(f"MLP params: {sum(p.numel() for p in mlp.parameters()):,}")

mlp_crit = nn.CrossEntropyLoss()
mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=1e-4)
mlp_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(mlp_opt, mode='max', factor=0.5, patience=15)

best_mlp, patience = 0, 0
t0 = time.time()
for ep in range(200):
    mlp.train()
    for xb, yb in ft_train:
        xb, yb = xb.cuda(), yb.cuda()
        mlp_opt.zero_grad()
        mlp_crit(mlp(xb), yb).backward()
        mlp_opt.step()

    mlp.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in ft_val:
            xb, yb = xb.cuda(), yb.cuda()
            correct += (mlp(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    acc = correct / total
    mlp_sch.step(acc)

    if acc > best_mlp:
        best_mlp = acc; patience = 0
        torch.save(mlp.state_dict(), "models/mlp_gpu.pt")
    else:
        patience += 1

    if ep % 30 == 0:
        print(f"  Epoch {ep+1:3d} | Val Acc: {acc:.4f} | Best: {best_mlp:.4f} | LR: {mlp_opt.param_groups[0]['lr']:.6f}")

    if patience >= 25:
        print(f"  Early stop at epoch {ep+1}")
        break

mlp.load_state_dict(torch.load("models/mlp_gpu.pt"))
mlp.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in ft_test:
        xb, yb = xb.cuda(), yb.cuda()
        preds.append(mlp(xb).argmax(1).cpu().numpy())
        trues.append(yb.cpu().numpy())

yp_mlp, yt_mlp = np.concatenate(preds), np.concatenate(trues)
mlp_acc = accuracy_score(yt_mlp, yp_mlp)
mlp_f1 = f1_score(yt_mlp, yp_mlp, average="macro")

# ============================================================
# 4. XGBoost (already trained, load from disk)
# ============================================================
print("\n[4/4] Evaluating XGBoost baseline...")
xgb = joblib.load("models/xgboost_optimized.pkl")
xgb_preds = xgb.predict(X_fte)
xgb_acc = accuracy_score(y_fte, xgb_preds)
xgb_f1 = f1_score(y_fte, xgb_preds, average="macro")

# ============================================================
# FINAL COMPARISON
# ============================================================
print(f"\n{'='*70}")
print(f"FINAL GPU COMPARISON (RTX 4060 Ti)")
print(f"{'='*70}")
print(f"{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Train Time':>12}")
print(f"{'-'*57}")
print(f"{'XGBoost (feature-based)':<25} {xgb_acc:>10.4f} {xgb_f1:>10.4f} {'~2 min (CPU)':>12}")
print(f"{'MLP (same features, GPU)':<25} {mlp_acc:>10.4f} {mlp_f1:>10.4f} {f'{time.time()-t0:.0f}s':>12}")
print(f"{'LSTM (raw lightcurves, GPU)':<25} {lstm_acc:>10.4f} {lstm_f1:>10.4f} {f'{time.time()-t0:.0f}s':>12}")
print(f"{'='*70}")

# XGBoost full report
print(f"\n▸ XGBoost Classification Report:")
print(classification_report(y_fte, xgb_preds, target_names=cnames, digits=3, zero_division=0))

# MLP report
print(f"\n▸ MLP (GPU) Classification Report:")
print(classification_report(yt_mlp, yp_mlp, target_names=cnames, digits=3, zero_division=0))

# LSTM report
print(f"\n▸ LSTM (GPU) Classification Report:")
print(classification_report(yt, yp_lstm, target_names=cnames, digits=3, zero_division=0))

# Save all
torch.save(lstm.state_dict(), "models/lstm_gpu_final.pt")
torch.save(mlp.state_dict(), "models/mlp_gpu_final.pt")

print("\n" + "="*70)
print("ALL GPU TRAINING COMPLETE!")
print("Models saved: lstm_gpu.pt, mlp_gpu.pt, xgboost_optimized.pkl")
print("="*70)
