"""Contrastive Learning V2 — 完整对比: 手写特征 vs 学习特征 vs 融合"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np, time, warnings
warnings.filterwarnings("ignore")

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb, joblib

from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES

DEV = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# ====== LOAD & BUILD SEQUENCES ======
print("=== Loading data ===")
train_df, _, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=1)

np.random.seed(42)
MAX_SEQ = 200
all_objs = []
for cid in sorted(train_df["label"].unique()):
    idxs = train_df[train_df["label"] == cid].index.values
    n = min(500, len(idxs))
    for idx in np.random.choice(idxs, n, replace=False):
        all_objs.append((idx, cid))

seqs, labs, skip = [], [], 0
for idx, cid in all_objs:
    try: mjd, flx, fle, pbs = get_lightcurve(train_df.iloc[idx])
    except: skip += 1; continue
    m = fle > 0
    if m.sum() < 5: skip += 1; continue
    fmed = np.median(flx[m]); fmad = np.median(np.abs(flx[m] - np.median(flx[m]))) + 1e-8
    fn = (flx - fmed) / fmad; fen = np.clip(fle / fmad, 0, 10)
    n = min(len(mjd), MAX_SEQ)
    s = np.zeros((MAX_SEQ, 4), dtype=np.float32)
    s[:n, 0] = (mjd[:n] - mjd[0]) / 100.0
    s[:n, 1] = fn[:n]; s[:n, 2] = fen[:n]; s[:n, 3] = pbs[:n] / 10000.0
    seqs.append(s); labs.append(cid)

X_seq = np.array(seqs, dtype=np.float32)
y_raw = np.array(labs, dtype=np.int64)
le = LabelEncoder(); y_enc = le.fit_transform(y_raw)
NC = len(le.classes_)
print(f"Sequences: {len(seqs)}, Classes: {NC}")

# Split sequences
idxs = np.arange(len(y_enc))
itmp, ite = train_test_split(idxs, test_size=0.15, stratify=y_enc, random_state=42)
itr, iva = train_test_split(itmp, test_size=0.15/0.85, stratify=y_enc[itmp], random_state=42)
print(f"Seq Train: {len(itr)} Val: {len(iva)} Test: {len(ite)}")

# ====== CONTRASTIVE PRETRAINING ======
print("\n=== Contrastive Pretraining (GPU, 150 epochs) ===")

class ConvEncoder(nn.Module):
    def __init__(self, in_c=4, out_d=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 5, 2, 2), nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, out_d, 3, 2, 1), nn.BatchNorm1d(out_d), nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return self.gap(x).squeeze(-1)

def augment(x):
    B, L, C = x.shape
    x2 = x.clone()
    x2[:, :, 1] += torch.randn_like(x2[:, :, 1]) * 0.05
    mask = torch.rand(B, L, 1, device=x.device) > 0.1
    x2 = x2 * mask.float()
    return x2

def contrastive_loss_fn(z1, z2, temp=0.07):
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    z = torch.cat([z1, z2], dim=0)
    B = z1.shape[0]
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).cuda()
    sim = torch.mm(z, z.T) / temp
    mask = torch.eye(2*B, device=DEV, dtype=torch.bool)
    sim = sim.float().masked_fill(mask, -1e4)
    return F.cross_entropy(sim, labels)

encoder = ConvEncoder(in_c=4, out_d=128).cuda()
opt = torch.optim.AdamW(encoder.parameters(), lr=0.001, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
scaler = GradScaler()

seq_all = torch.tensor(X_seq)
ds = TensorDataset(seq_all)
dl = DataLoader(ds, batch_size=256, shuffle=True)

t0 = time.time()
for ep in range(150):
    encoder.train()
    total_loss = 0
    for (xb,) in dl:
        xb = xb.cuda()
        x1, x2 = augment(xb), augment(xb)
        with autocast():
            z1, z2 = encoder(x1), encoder(x2)
            loss = contrastive_loss_fn(z1, z2)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total_loss += loss.item()
    sch.step()
    if ep % 30 == 0 or ep == 149:
        print(f"  E{ep+1:3d} | Loss: {total_loss/len(dl):.4f} | {time.time()-t0:.0f}s")

contrast_time = time.time() - t0
torch.save(encoder.state_dict(), "models/contrastive_encoder.pt")
print(f"Done in {contrast_time:.0f}s")

# ====== EXTRACT EMBEDDINGS ======
print("\n=== Extracting embeddings ===")
encoder.eval()
embs = []
edl = DataLoader(ds, batch_size=512)
with torch.no_grad():
    for (xb,) in edl:
        with autocast():
            embs.append(encoder(xb.cuda()).cpu().numpy())
E = np.concatenate(embs)
scaler_emb = StandardScaler()
E = scaler_emb.fit_transform(E)
print(f"Embeddings: {E.shape}")

# ====== LOAD HANDCRAFTED FEATURES ======
print("\n=== Loading handcrafted features ===")
ft = np.load("data/processed/train_full.npz")
Xf, yf = ft["X"], ft["y"]
le_f = LabelEncoder()
yf_enc = le_f.fit_transform(yf)
print(f"Features: {Xf.shape}")

# ====== 3-WAY COMPARISON ======
print("\n=== 3-way comparison ===")

def train_xgb(Xtr, ytr, Xte, yte, label=""):
    cc = np.bincount(ytr)
    sw = np.array([len(ytr)/(len(cc)*max(c,1)) for c in cc])[ytr]
    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.5,
        objective="multi:softprob", num_class=len(np.unique(ytr)),
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
    )
    m.fit(Xtr, ytr, sample_weight=sw, verbose=False)
    p = m.predict(Xte)
    acc = accuracy_score(yte, p)
    f1 = f1_score(yte, p, average="macro")
    print(f"  {label:30s} Acc={acc:.4f}  F1={f1:.4f}")
    return acc, f1, p

# 1. Learned embeddings only
Etr, Ete, y_etr, y_ete = train_test_split(E, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
acc_e, f1_e, p_e = train_xgb(Etr, y_etr, Ete, y_ete, "Learned Embeddings (128d)")

# 2. Handcrafted features only (use same split style)
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(Xf, yf_enc, test_size=0.15, stratify=yf_enc, random_state=42)
acc_f, f1_f, p_f = train_xgb(Xf_tr, yf_tr, Xf_te, yf_te, "Handcrafted Features (47d)")

# 3. Embeddings + Features (fused) — use intersection
# Take first N of each, fused
N = min(len(E), len(Xf))
E_fused = np.hstack([E[:N], Xf[:N]])
y_fused = y_enc[:N]
Ef_tr, Ef_te, yf_tr2, yf_te2 = train_test_split(E_fused, y_fused, test_size=0.15, stratify=y_fused, random_state=42)
acc_fused, f1_fused, p_fused = train_xgb(Ef_tr, yf_tr2, Ef_te, yf_te2, "Fused (175d)")

# ====== REPORT ======
cn = [CLASS_NAMES.get(c, f"cls_{c}") for c in le.classes_]
print(f"\n{'='*60}")
print(f"CONTRASTIVE LEARNING RESULTS")
print(f"{'='*60}")
print(f"Pretraining: {contrast_time:.0f}s on RTX 4060 Ti (GPU)")
print(f"{'='*60}")
print(f"{'Model':<30} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"{'-'*50}")
print(f"{'Features (47d)':<30} {acc_f:>10.4f} {f1_f:>10.4f}")
print(f"{'Embeddings (128d)':<30} {acc_e:>10.4f} {f1_e:>10.4f}")
print(f"{'Fused (175d)':<30} {acc_fused:>10.4f} {f1_fused:>10.4f}")
print(f"{'='*60}")

best = max([("Features", acc_f, f1_f, p_f), ("Embeddings", acc_e, f1_e, p_e),
             ("Fused", acc_fused, f1_fused, p_fused)], key=lambda x: x[1])

print(f"\nBest: {best[0]} ({best[1]*100:.1f}%)")

# Full report for best
if best[0] == "Features":
    print(f"\n--- Handcrafted Features Report ---")
    print(classification_report(yf_te, p_f, target_names=cn, digits=3, zero_division=0))
elif best[0] == "Embeddings":
    print(f"\n--- Learned Embeddings Report ---")
    print(classification_report(y_ete, p_e, target_names=cn, digits=3, zero_division=0))
else:
    print(f"\n--- Fused Report ---")
    print(classification_report(yf_te2, p_fused, target_names=cn, digits=3, zero_division=0))

print("\nDONE!")
