"""Evaluate all 3 models and produce final comparison"""
import sys, os, time
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES

DEVICE = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data
train_df, _, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=1)
np.random.seed(42)
MAX_SEQ, SAMPLES_PER = 150, 500

all_objects = []
for cls_id in sorted(train_df["label"].unique()):
    obj_idx = train_df[train_df["label"] == cls_id].index.values
    n = min(SAMPLES_PER, len(obj_idx))
    chosen = np.random.choice(obj_idx, size=n, replace=False)
    for idx in chosen:
        all_objects.append((idx, cls_id))

print(f"Building {len(all_objects)} sequences...")
sequences, labels = [], []
for idx, cls_id in all_objects:
    try:
        mjd, flux, flux_err, pbs = get_lightcurve(train_df.iloc[idx])
    except:
        continue
    mask = flux_err > 0
    if mask.sum() < 5: continue
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

X_seq = np.array(sequences, dtype=np.float32)
y_seq = np.array(labels, dtype=np.int64)
le = LabelEncoder()
y_enc = le.fit_transform(y_seq)
num_classes = len(le.classes_)
cnames = [CLASS_NAMES.get(c, f"class_{c}") for c in le.classes_]

_, X_test, _, y_test = train_test_split(X_seq, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
print(f"Test: {len(X_test)}")

# ---- LSTM ----
class DeepLSTM(nn.Module):
    def __init__(self, input_size=4, hidden=256, num_layers=3, num_classes=14, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden * 2)
        self.attn = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.fc = nn.Sequential(nn.Linear(hidden * 2, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln(out)
        attn_w = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        context = torch.sum(out * attn_w.unsqueeze(-1), dim=1)
        return self.fc(context + out.mean(dim=1))

lstm = DeepLSTM(input_size=4, hidden=256, num_layers=3, num_classes=num_classes, dropout=0.35).cuda()
lstm.load_state_dict(torch.load("models/lstm_gpu.pt"))
lstm.eval()

ts = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)), batch_size=256)
preds, trues = [], []
with torch.no_grad():
    for xb, yb in ts:
        logits = lstm(xb.cuda())
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.cpu().numpy())

yp_lstm, yt_seq = np.concatenate(preds), np.concatenate(trues)
lstm_acc = accuracy_score(yt_seq, yp_lstm)
lstm_f1 = f1_score(yt_seq, yp_lstm, average="macro")
lstm_wf1 = f1_score(yt_seq, yp_lstm, average="weighted")

# ---- MLP ----
class DeepMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_out))

    def forward(self, x): return self.net(x)

# Load features test
ft_data = np.load("data/processed/test.npz")
X_feat, y_feat = ft_data["X"], ft_data["y"]
_, X_fte, _, y_fte = train_test_split(X_feat, y_feat, test_size=0.3, stratify=y_feat, random_state=42)
print(f"Feature test set: {len(X_fte)}")

n_feat = X_feat.shape[1]
mlp = DeepMLP(n_feat, num_classes).cuda()
mlp.load_state_dict(torch.load("models/mlp_gpu.pt"))
mlp.eval()

ts2 = DataLoader(TensorDataset(torch.tensor(X_fte.astype(np.float32)), torch.tensor(y_fte, dtype=torch.long)), batch_size=512)
preds, trues = [], []
with torch.no_grad():
    for xb, yb in ts2:
        preds.append(mlp(xb.cuda()).argmax(1).cpu().numpy())
        trues.append(yb.cpu().numpy())

yp_mlp, yt_feat = np.concatenate(preds), np.concatenate(trues)
mlp_acc = accuracy_score(yt_feat, yp_mlp)
mlp_f1 = f1_score(yt_feat, yp_mlp, average="macro")
mlp_wf1 = f1_score(yt_feat, yp_mlp, average="weighted")

# ---- XGBoost ----
xgb = joblib.load("models/xgboost_optimized.pkl")
xgb_preds = xgb.predict(X_fte)
xgb_acc = accuracy_score(y_fte, xgb_preds)
xgb_f1 = f1_score(y_fte, xgb_preds, average="macro")
xgb_wf1 = f1_score(y_fte, xgb_preds, average="weighted")

# ---- COMPARISON ----
print(f"\n{'='*70}")
print(f"FINAL GPU MODEL COMPARISON")
print(f"{'='*70}")
print(f"{'Model':<30} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>10}")
print(f"{'-'*60}")
print(f"{'XGBoost (feature-based)':<30} {xgb_acc:>10.4f} {xgb_f1:>10.4f} {xgb_wf1:>10.4f}")
print(f"{'MLP GPU (same features)':<30} {mlp_acc:>10.4f} {mlp_f1:>10.4f} {mlp_wf1:>10.4f}")
print(f"{'LSTM GPU (raw lightcurves)':<30} {lstm_acc:>10.4f} {lstm_f1:>10.4f} {lstm_wf1:>10.4f}")
print(f"{'='*70}")

print(f"\n--- XGBoost (best overall) ---")
print(classification_report(y_fte, xgb_preds, target_names=cnames, digits=3, zero_division=0))

print(f"\n--- MLP GPU ---")
print(classification_report(yt_feat, yp_mlp, target_names=cnames, digits=3, zero_division=0))

print(f"\n--- LSTM GPU ---")
print(classification_report(yt_seq, yp_lstm, target_names=cnames, digits=3, zero_division=0))

print("EVALUATION COMPLETE!")
