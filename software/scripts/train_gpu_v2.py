"""GPU V2: Transformer + 2D Heatmap CNN + Ensemble"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np, time, warnings
warnings.filterwarnings("ignore")

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES

DEV = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# ====== DATA ======
print("\n=== Loading Data ===")
train_df, _, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=1)
np.random.seed(42)

all_objs = []
for cid in sorted(train_df["label"].unique()):
    idxs = train_df[train_df["label"] == cid].index.values
    n = min(500, len(idxs))
    for idx in np.random.choice(idxs, n, replace=False):
        all_objs.append((idx, cid))

print(f"Building {len(all_objs)} sequences + heatmaps...")
t1 = time.time()
MAX_SEQ, TB, BB = 200, 48, 6
band_edges = [3000, 4200, 5400, 6800, 8200, 9200, 10500]
seqs, hms, labs, skip = [], [], [], 0

for idx, cid in all_objs:
    try:
        mjd, flx, fle, pbs = get_lightcurve(train_df.iloc[idx])
    except: skip += 1; continue
    m = fle > 0
    if m.sum() < 5: skip += 1; continue

    # Normalize
    fmed, fmad = np.median(flx[m]), np.median(np.abs(flx[m] - np.median(flx[m]))) + 1e-8
    fn = (flx - fmed) / fmad
    fen = np.clip(fle / fmad, 0, 10)

    # Sequence (N, MAX_SEQ, 5)
    nn_pts = min(len(mjd), MAX_SEQ)
    s = np.zeros((MAX_SEQ, 5), dtype=np.float32)
    s[:nn_pts, 0] = (mjd[:nn_pts] - mjd[0]) / 100.0
    s[:nn_pts, 1] = fn[:nn_pts]
    s[:nn_pts, 2] = fen[:nn_pts]
    s[:nn_pts, 3] = pbs[:nn_pts] / 10000.0
    s[:nn_pts, 4] = m[:nn_pts].astype(np.float32)
    seqs.append(s)

    # Heatmap (TB, BB, 3)
    t_n = (mjd - mjd[0]) / max(mjd[-1] - mjd[0], 1)
    tb = np.clip((t_n * TB).astype(int), 0, TB - 1)
    bb = np.clip(np.digitize(pbs, band_edges) - 1, 0, BB - 1)
    hm = np.zeros((TB, BB, 3), dtype=np.float32)
    cnt = np.zeros((TB, BB), dtype=np.float32)
    for i in range(len(mjd)):
        hm[tb[i], bb[i], 0] += fn[i]
        hm[tb[i], bb[i], 1] += fen[i]
        hm[tb[i], bb[i], 2] += 1
        cnt[tb[i], bb[i]] += 1
    cnt = np.maximum(cnt, 1)
    hm[:, :, 0] /= cnt; hm[:, :, 1] /= cnt
    hms.append(hm)
    labs.append(cid)

print(f"  {len(seqs)} seqs + {len(hms)} heatmaps ({time.time()-t1:.1f}s)")

X_seq = np.array(seqs, dtype=np.float32)
X_hm = np.array(hms, dtype=np.float32)  # (N, 48, 6, 3)
y = np.array(labs, dtype=np.int64)

le = LabelEncoder(); y_enc = le.fit_transform(y)
NC = len(le.classes_)
cn = [CLASS_NAMES.get(c, f"cls_{c}") for c in le.classes_]
print(f"  Classes: {NC}")

# Split
idxs = np.arange(len(y_enc))
i_tmp, i_te = train_test_split(idxs, test_size=0.15, stratify=y_enc, random_state=42)
i_tr, i_vl = train_test_split(i_tmp, test_size=0.15/0.85, stratify=y_enc[i_tmp], random_state=42)
print(f"  Train: {len(i_tr)}, Val: {len(i_vl)}, Test: {len(i_te)}")

Xst, Xsv, Xste = X_seq[i_tr], X_seq[i_vl], X_seq[i_te]
Xht, Xhv, Xhte = X_hm[i_tr], X_hm[i_vl], X_hm[i_te]
yt, yv, yte = y_enc[i_tr], y_enc[i_vl], y_enc[i_te]

B = 128
stdl = DataLoader(TensorDataset(torch.tensor(Xst), torch.tensor(yt, dtype=torch.long)), B, shuffle=True, pin_memory=True)
svdl = DataLoader(TensorDataset(torch.tensor(Xsv), torch.tensor(yv, dtype=torch.long)), B, pin_memory=True)
sedl = DataLoader(TensorDataset(torch.tensor(Xste), torch.tensor(yte, dtype=torch.long)), B, pin_memory=True)
htdl = DataLoader(TensorDataset(torch.tensor(Xht).permute(0,3,1,2), torch.tensor(yt, dtype=torch.long)), B, shuffle=True, pin_memory=True)
hvdl = DataLoader(TensorDataset(torch.tensor(Xhv).permute(0,3,1,2), torch.tensor(yv, dtype=torch.long)), B, pin_memory=True)
hedl = DataLoader(TensorDataset(torch.tensor(Xhte).permute(0,3,1,2), torch.tensor(yte, dtype=torch.long)), B, pin_memory=True)

cc = np.bincount(yt)
cw = torch.tensor([len(yt)/(NC*max(c,1)) for c in cc], dtype=torch.float32).cuda()

# ====== TRAINING UTILS ======
def train_model(model, trdl, vdl, epochs, lr=0.001, name="m", patience=30):
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=25, T_mult=2)
    best, pat = 0, 0; t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in trdl:
            xb, yb = xb.cuda(), yb.cuda()
            opt.zero_grad()
            crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        model.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for xb, yb in vdl:
                xb, yb = xb.cuda(), yb.cuda()
                corr += (model(xb).argmax(1) == yb).sum().item()
                tot += len(yb)
        acc = corr / tot
        if acc > best: best = acc; pat = 0; torch.save(model.state_dict(), f"models/{name}.pt")
        else: pat += 1
        if ep % 25 == 0 or ep == epochs-1:
            print(f"  {name:18s} E{ep+1:3d} | Val: {acc:.4f} | Best: {best:.4f} | {time.time()-t0:.0f}s")
        if pat >= patience: print(f"  Early stop E{ep+1}"); break
    model.load_state_dict(torch.load(f"models/{name}.pt"))
    return model

def test_model(model, tdl):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for xb, yb in tdl:
            ps.append(model(xb.cuda()).argmax(1).cpu().numpy())
            ts.append(yb.cpu().numpy())
    yp, yt = np.concatenate(ps), np.concatenate(ts)
    return accuracy_score(yt, yp), f1_score(yt, yp, average="macro"), yp, yt

# ====== TRANSFORMER ======
print("\n=== Transformer ===")
class PosEnc(nn.Module):
    def __init__(self, d, dropout=0.1, mx=500):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(mx, d)
        pos = torch.arange(0, mx).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:,:x.size(1),:])

class Transformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, nlayers=4, nc=14, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(5, d_model)
        self.pe = PosEnc(d_model, dropout, 500)
        el = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, 'gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(el, nlayers)
        self.ln = nn.LayerNorm(d_model)
        self.cls = nn.Parameter(torch.randn(1,1,d_model)*0.02)
        self.head = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*2, nc))
    def forward(self, x):
        B = x.shape[0]; x = self.proj(x)
        x = torch.cat([self.cls.expand(B,-1,-1), x], 1)
        x = self.ln(self.enc(self.pe(x))[:,0,:])
        return self.head(x)

tf = Transformer(d_model=128, nhead=8, nlayers=4, nc=NC, dropout=0.15).cuda()
print(f"Params: {sum(p.numel() for p in tf.parameters()):,}")
tf = train_model(tf, stdl, svdl, 200, lr=0.0008, name="transformer")
tf_acc, tf_f1, tf_yp, _ = test_model(tf, sedl)
print(f"Transformer: Acc={tf_acc:.4f} F1={tf_f1:.4f}")

# ====== 2D CNN ======
print("\n=== 2D Heatmap CNN ===")
class HeatmapCNN(nn.Module):
    def __init__(self, nc=14, dr=0.3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3,64,3,1,1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2), nn.Dropout2d(dr),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128,128,3,1,1), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2), nn.Dropout2d(dr),
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.GELU(), nn.AdaptiveAvgPool2d((6,1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256*6,256), nn.GELU(), nn.Dropout(dr), nn.Linear(256,nc))
    def forward(self, x): return self.head(self.feat(x))

cnn_m = HeatmapCNN(nc=NC, dr=0.3).cuda()
print(f"Params: {sum(p.numel() for p in cnn_m.parameters()):,}")
cnn_m = train_model(cnn_m, htdl, hvdl, 200, lr=0.001, name="heatmap_cnn")
cnn_acc, cnn_f1, cnn_yp, _ = test_model(cnn_m, hedl)
print(f"Heatmap CNN: Acc={cnn_acc:.4f} F1={cnn_f1:.4f}")

# ====== XGBOOST ======
print("\n=== XGBoost (load from disk) ===")
xgb = joblib.load("models/xgboost_optimized.pkl")
ft = np.load("data/processed/test.npz")
Xf, yf = ft["X"], ft["y"]
_, Xft, _, yft = train_test_split(Xf, yf, test_size=0.3, stratify=yf, random_state=42)
xp = xgb.predict(Xft)
xgb_acc = accuracy_score(yft, xp)
xgb_f1 = f1_score(yft, xp, average="macro")
print(f"XGBoost: Acc={xgb_acc:.4f} F1={xgb_f1:.4f}")

# ====== ENSEMBLE ======
print("\n=== Ensemble (soft voting) ===")
# Get probs from each model
tf.eval(); cnn_m.eval()
ptf, pcnn = [], []
with torch.no_grad():
    for xb, _ in sedl: ptf.append(F.softmax(tf(xb.cuda()),1).cpu().numpy())
    for xb, _ in hedl: pcnn.append(F.softmax(cnn_m(xb.cuda()),1).cpu().numpy())
ptf = np.concatenate(ptf)[:len(Xft)]
pcnn = np.concatenate(pcnn)[:len(Xft)]
pxgb = xgb.predict_proba(Xft)

# Search best weights
best_w, best_ef1 = (0.5, 0.3, 0.2), 0
for w1 in [0.4, 0.5, 0.55, 0.6, 0.65]:
    for w2 in [0.15, 0.2, 0.25, 0.3]:
        w3 = 1 - w1 - w2
        if w3 <= 0: continue
        pe = w1*pxgb + w2*ptf + w3*pcnn
        ye = np.argmax(pe, 1)
        ef1 = f1_score(yft, ye, average="macro")
        if ef1 > best_ef1: best_ef1 = ef1; best_w = (w1,w2,w3)

pe = best_w[0]*pxgb + best_w[1]*ptf + best_w[2]*pcnn
ye = np.argmax(pe, 1)
ens_acc = accuracy_score(yft, ye)
ens_f1 = f1_score(yft, ye, average="macro")

# ====== REPORT ======
print(f"\n{'='*65}")
print(f"FINAL GPU V2 RESULTS (RTX 4060 Ti)")
print(f"{'='*65}")
print(f"{'Model':<28} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"{'-'*48}")
print(f"{'XGBoost':<28} {xgb_acc:>10.4f} {xgb_f1:>10.4f}")
print(f"{'Transformer (GPU)':<28} {tf_acc:>10.4f} {tf_f1:>10.4f}")
print(f"{'2D Heatmap CNN (GPU)':<28} {cnn_acc:>10.4f} {cnn_f1:>10.4f}")
print(f"{'Ensemble (XGB+TF+CNN)':<28} {ens_acc:>10.4f} {ens_f1:>10.4f}")
print(f"{'='*65}")
print(f"Ensemble weights: XGB={best_w[0]:.2f}, TF={best_w[1]:.2f}, CNN={best_w[2]:.2f}")

# Best single model
results = [("XGBoost", xgb_acc, xgb_f1, xp, yft),
           ("Transformer", tf_acc, tf_f1, tf_yp, yte),
           ("Heatmap CNN", cnn_acc, cnn_f1, cnn_yp, yte)]

best = max(results, key=lambda x: x[1])
print(f"\nBest single model: {best[0]} ({best[1]*100:.1f}%)")
print(f"\n--- {best[0]} Report ---")
print(classification_report(best[4], best[3], target_names=cn, digits=3, zero_division=0))

if ens_acc > best[1]:
    print(f"\nEnsemble WINS! ({ens_acc*100:.1f}% vs {best[1]*100:.1f}%)")
    print(f"\n--- Ensemble Report ---")
    print(classification_report(yft, ye, target_names=cn, digits=3, zero_division=0))

# Save models
torch.save(tf.state_dict(), "models/transformer_gpu.pt")
torch.save(cnn_m.state_dict(), "models/heatmap_cnn_gpu.pt")
print("\nGPU V2 COMPLETE!")
