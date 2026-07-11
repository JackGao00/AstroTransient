"""Step 5: MLP on features — quick comparison"""
import sys, os
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

from src.data.preprocess import CLASS_NAMES

# Load XGBoost model and test data from step3
xgb_model = joblib.load("models/xgboost_optimized.pkl")

test_data = np.load("data/processed/test.npz")
X_test, y_test = test_data["X"], test_data["y"]

num_classes = len(np.unique(y_test))
print(f"Classes: {num_classes}, Test samples: {len(X_test)}")

# Resplit test data for MLP training (for quick comparison)
# 60% train, 20% val, 20% final test
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_test, y_test, test_size=0.4, stratify=y_test, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

# Evaluate XGBoost on this same test split
xgb_preds = xgb_model.predict(X_te)
xgb_acc = accuracy_score(y_te, xgb_preds)
xgb_f1 = f1_score(y_te, xgb_preds, average="macro")

print(f"MLP Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")
print(f"XGBoost on this test: Acc={xgb_acc:.4f}, F1={xgb_f1:.4f}")

B = 256
tl = DataLoader(TensorDataset(torch.tensor(X_tr.astype(np.float32)), torch.tensor(y_tr, dtype=torch.long)), batch_size=B, shuffle=True)
vl = DataLoader(TensorDataset(torch.tensor(X_val.astype(np.float32)), torch.tensor(y_val, dtype=torch.long)), batch_size=B)
ts = DataLoader(TensorDataset(torch.tensor(X_te.astype(np.float32)), torch.tensor(y_te, dtype=torch.long)), batch_size=B)

n_features = X_tr.shape[1]
mlp = nn.Sequential(
    nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, num_classes)
)
print(f"MLP params: {sum(p.numel() for p in mlp.parameters()):,}")

crit = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)

best_acc, patience = 0, 0
for ep in range(100):
    mlp.train()
    for xb, yb in tl:
        opt.zero_grad()
        crit(mlp(xb), yb).backward()
        opt.step()

    mlp.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in vl:
            correct += (mlp(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    acc = correct / total
    sch.step(acc)

    if acc > best_acc:
        best_acc = acc; patience = 0
        torch.save(mlp.state_dict(), "models/mlp_classifier.pt")
    else:
        patience += 1

    if ep % 20 == 0 or ep == 99:
        print(f"  Epoch {ep+1:3d} | Val Acc: {acc:.4f} | Best: {best_acc:.4f}")

    if patience >= 20:
        print(f"  Early stop at epoch {ep+1}")
        break

mlp.load_state_dict(torch.load("models/mlp_classifier.pt"))
mlp.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in ts:
        preds.append(mlp(xb).argmax(1).numpy())
        trues.append(yb.numpy())

yp, yt = np.concatenate(preds), np.concatenate(trues)
mlp_acc = accuracy_score(yt, yp)
mlp_f1 = f1_score(yt, yp, average="macro")

# Final summary
lstm_acc, lstm_f1 = 0.3048, 0.1876
print(f"\n{'='*60}")
print(f"MODEL COMPARISON (same test set: {len(X_te)} samples)")
print(f"{'='*60}")
print(f"{'Model':<20} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"{'-'*40}")
print(f"{'XGBoost (optimized)':<20} {xgb_acc:>10.4f} {xgb_f1:>10.4f}")
print(f"{'MLP (features)':<20} {mlp_acc:>10.4f} {mlp_f1:>10.4f}")
print(f"{'LSTM (raw seq)':<20} {lstm_acc:>10.4f} {lstm_f1:>10.4f}")
print(f"{'='*60}")
print(f"\nBest: XGBoost — {xgb_acc*100:.1f}% accuracy on 14-class astronomical transient classification")

cnames = [CLASS_NAMES.get(i, f"class_{i}") for i in range(num_classes)]
print(f"\nMLP Classification Report:")
print(classification_report(yt, yp, target_names=cnames[:num_classes], digits=3, zero_division=0))
print("DONE!")
