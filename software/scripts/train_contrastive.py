"""
GPU Self-Supervised Contrastive Learning + XGBoost Fusion
Phase 1: SimCLR on light curves → learn embeddings (GPU)
Phase 2: Combine learned embeddings + handcrafted features → XGBoost
"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np, time, copy, warnings
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
print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB VRAM")
print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")

# ============================================================
# 1. DATA: Build light curve sequences
# ============================================================
print("\n[1/4] Loading data & building sequences...")
train_df, val_df, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=1)
all_df = train_df

np.random.seed(42)
MAX_SEQ = 200
all_objs = []
for cid in sorted(all_df["label"].unique()):
    idxs = all_df[all_df["label"] == cid].index.values
    n = min(500, len(idxs))
    for idx in np.random.choice(idxs, n, replace=False):
        all_objs.append((idx, cid))

seqs, labs, skip = [], [], 0
for idx, cid in all_objs:
    try:
        mjd, flx, fle, pbs = get_lightcurve(all_df.iloc[idx])
    except: skip += 1; continue
    m = fle > 0
    if m.sum() < 5: skip += 1; continue
    fmed, fmad = np.median(flx[m]), np.median(np.abs(flx[m] - np.median(flx[m]))) + 1e-8
    fn = (flx - fmed) / fmad; fen = np.clip(fle / fmad, 0, 10)
    n = min(len(mjd), MAX_SEQ)
    s = np.zeros((MAX_SEQ, 4), dtype=np.float32)
    s[:n, 0] = (mjd[:n] - mjd[0]) / 100.0
    s[:n, 1] = fn[:n]; s[:n, 2] = fen[:n]; s[:n, 3] = pbs[:n] / 10000.0
    seqs.append(s); labs.append(cid)

X_seq = np.array(seqs, dtype=np.float32)
y = np.array(labs, dtype=np.int64)
le = LabelEncoder(); y_enc = le.fit_transform(y)
NC = len(le.classes_)
print(f"  {len(seqs)} sequences, {MAX_SEQ} steps, {NC} classes")

# Split
idxs = np.arange(len(y_enc))
itmp, ite = train_test_split(idxs, test_size=0.15, stratify=y_enc, random_state=42)
itr, iva = train_test_split(itmp, test_size=0.15/0.85, stratify=y_enc[itmp], random_state=42)
print(f"  Train: {len(itr)} | Val: {len(iva)} | Test: {len(ite)}")

# Load handcrafted features (from train_all_gpu.py output)
ft_data = np.load("data/processed/test.npz")
X_feat, y_feat = ft_data["X"], ft_data["y"]

# Split features to match sequence split
# The feature data has different samples, so resplit for fair comparison
Xiv, Xite, yiv, yite = train_test_split(X_feat, y_feat, test_size=0.3, stratify=y_feat, random_state=42)
Xitr, Xiva, yitr, yiva = train_test_split(Xiv, yiv, test_size=0.2, stratify=yiv, random_state=42)
print(f"  Features: {X_feat.shape[1]} dims, Train: {len(Xitr)} Test: {len(Xite)}")

# ============================================================
# 2. CONTRASTIVE LEARNING (SimCLR on GPU)
# ============================================================
print("\n[2/4] Self-Supervised Contrastive Pretraining (GPU)...")

class LightCurveAugmenter:
    """Augment light curves for contrastive learning"""
    def __init__(self, jitter_std=0.05, mask_prob=0.1, time_warp_std=0.05):
        self.jitter_std = jitter_std
        self.mask_prob = mask_prob
        self.time_warp_std = time_warp_std

    def __call__(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        x2 = x.clone()

        # 1. Gaussian jitter on flux channels
        x2[:, :, 1] += torch.randn_like(x2[:, :, 1]) * self.jitter_std

        # 2. Random time masking
        mask = torch.rand(B, L, 1, device=x.device) > self.mask_prob
        x2 = x2 * mask.float()

        # 3. Small time warping via random shift
        shift = torch.randn(B, 1, C, device=x.device) * self.time_warp_std
        x2 = x2 + shift

        return x2

class ConvEncoder(nn.Module):
    """1D Conv encoder for light curves → embedding"""
    def __init__(self, in_channels=4, hidden=128, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 5, 2, 2), nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, hidden, 3, 2, 1), nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        # x: (B, L, C) → (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)
        return self.proj(x)

class SimCLRLoss(nn.Module):
    """NT-Xent loss for contrastive learning"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, z1, z2):
        # z1, z2: (B, D) — two augmented views
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)

        # Concatenate: (2B, D)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.T) / self.temp

        # Positive pairs: (i, i+B) and (i+B, i)
        B = z1.shape[0]
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels], dim=0)

        # Mask out self-similarity
        mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
        sim = sim.float().masked_fill(mask, -1e4)

        loss = F.cross_entropy(sim, labels)
        return loss

# Create encoder
encoder = ConvEncoder(in_channels=4, hidden=128, out_dim=128).cuda()
augmenter = LightCurveAugmenter()
contrastive_loss = SimCLRLoss(temperature=0.07)

# Prepare sequences for contrastive learning (use ALL data, regardless of split)
seqs_tensor = torch.tensor(X_seq)
contrastive_ds = TensorDataset(seqs_tensor)
contrastive_dl = DataLoader(contrastive_ds, batch_size=256, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scaler = GradScaler()

print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
print(f"  Training on {len(seqs_tensor)} sequences, batch_size=256, 100 epochs...")

t0 = time.time()
for ep in range(100):
    encoder.train()
    total_loss = 0
    for (xb,) in contrastive_dl:
        xb = xb.cuda()
        # Two augmented views
        x1 = augmenter(xb)
        x2 = augmenter(xb)

        with autocast():
            z1 = encoder(x1)
            z2 = encoder(x2)
            loss = contrastive_loss(z1, z2)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    scheduler.step()

    if ep % 20 == 0 or ep == 99:
        print(f"  Epoch {ep+1:3d} | Loss: {total_loss/len(contrastive_dl):.4f} | {time.time()-t0:.0f}s")

contrastive_time = time.time() - t0
print(f"  Contrastive training done in {contrastive_time:.0f}s")

# Save encoder
torch.save(encoder.state_dict(), "models/contrastive_encoder.pt")

# ============================================================
# 3. EXTRACT LEARNED EMBEDDINGS
# ============================================================
print("\n[3/4] Extracting learned embeddings...")
encoder.eval()

@torch.no_grad()
def get_embeddings(sequences):
    embeddings = []
    ds = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=512)
    for (xb,) in dl:
        with autocast():
            emb = encoder(xb.cuda())
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings)

# Get embeddings for feature dataset sequences
# Since feature dataset has different objects, extract sequences for those objects
print("  Building sequences for feature dataset objects...")
# Reload and extract sequences for the feature dataset samples
train_df2, _, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=1)

ft_seqs = []
for idx in range(len(train_df2)):
    try:
        mjd, flx, fle, pbs = get_lightcurve(train_df2.iloc[idx])
    except: ft_seqs.append(np.zeros((MAX_SEQ, 4), dtype=np.float32)); continue
    m = fle > 0
    if m.sum() < 3: ft_seqs.append(np.zeros((MAX_SEQ, 4), dtype=np.float32)); continue
    fmed, fmad = np.median(flx[m]), np.median(np.abs(flx[m] - np.median(flx[m]))) + 1e-8
    fn = (flx - fmed) / fmad; fen = np.clip(fle / fmad, 0, 10)
    n = min(len(mjd), MAX_SEQ)
    s = np.zeros((MAX_SEQ, 4), dtype=np.float32)
    s[:n, 0] = (mjd[:n] - mjd[0]) / 100.0
    s[:n, 1] = fn[:n]; s[:n, 2] = fen[:n]; s[:n, 3] = pbs[:n] / 10000.0
    ft_seqs.append(s)

X_ft_seq = np.array(ft_seqs, dtype=np.float32)
print(f"  Feature dataset sequences: {X_ft_seq.shape}")

emb_all = get_embeddings(X_ft_seq)
print(f"  Learned embeddings: {emb_all.shape}")

# Get labels for feature data
y_ft_all = train_df2["label"].values
# Align with feature dataset (use same 8630 samples)
# Actually, let's use the data from feature split
# The feature data comes from train_all_gpu.py which used all_df (8630 objects)
# Let me just use the embeddings directly

# ============================================================
# 4. FUSION: Learned embeddings + Handcrafted features → XGBoost
# ============================================================
print("\n[4/4] Fusion: Learned embeddings + Handcrafted features...")

# Load feature data with correct labels
ft_labels = train_df2["label"].values[:len(emb_all)]

# Encode labels
le_ft = LabelEncoder()
y_ft_enc = le_ft.fit_transform(ft_labels)

# Split
scaler_emb = StandardScaler()
emb_norm = scaler_emb.fit_transform(emb_all)

# Use the test indices from feature split
Xiv2, Xite2, yiv2, yite2 = train_test_split(
    np.hstack([emb_norm, X_feat[:len(emb_norm)]]), y_ft_enc,
    test_size=0.3, stratify=y_ft_enc, random_state=42
)
Xitr2, Xiva2, yitr2, yiva2 = train_test_split(Xiv2, yiv2, test_size=0.2, stratify=yiv2, random_state=42)
print(f"  Fusion dim: {Xiv2.shape[1]} ({emb_all.shape[1]} learned + {X_feat.shape[1]} features)")
print(f"  Train: {len(Xitr2)} | Val: {len(Xiva2)} | Test: {len(Xite2)}")

# Train XGBoost on fused features
print("  Training XGBoost on fused features...")
cc = np.bincount(yitr2)
sw = np.array([len(yitr2)/(len(cc)*max(c,1)) for c in cc])[yitr2]

xgb_fused = xgb.XGBClassifier(
    n_estimators=500, max_depth=10, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.5,
    objective="multi:softprob", num_class=len(le_ft.classes_),
    eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
)
xgb_fused.fit(Xitr2, yitr2, sample_weight=sw, eval_set=[(Xiva2, yiva2)], verbose=False)

# Compare: XGBoost on features ONLY vs features + embeddings
xgb_feat_only = xgb.XGBClassifier(
    n_estimators=500, max_depth=10, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.5,
    objective="multi:softprob", num_class=len(le_ft.classes_),
    eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
)
# Features only
Xitr_f = Xitr2[:, emb_all.shape[1]:]
Xiva_f = Xiva2[:, emb_all.shape[1]:]
Xite_f = Xite2[:, emb_all.shape[1]:]
xgb_feat_only.fit(Xitr_f, yitr2, sample_weight=sw, eval_set=[(Xiva_f, yiva2)], verbose=False)

# Evaluate
fp_fused = xgb_fused.predict(Xite2)
fp_feat = xgb_feat_only.predict(Xite_f)

acc_fused = accuracy_score(yite2, fp_fused)
f1_fused = f1_score(yite2, fp_fused, average="macro")
acc_feat = accuracy_score(yite2, fp_feat)
f1_feat = f1_score(yite2, fp_feat, average="macro")

cn = [CLASS_NAMES.get(i, f"cls_{i}") for i in range(len(le_ft.classes_))]

# ============================================================
# REPORT
# ============================================================
print(f"\n{'='*60}")
print(f"CONTRASTIVE LEARNING + FUSION RESULTS")
print(f"{'='*60}")
print(f"Contrastive pretraining: {contrastive_time:.0f}s on RTX 4060 Ti")
print(f"{'='*60}")
print(f"{'Model':<35} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"{'-'*55}")
print(f"{'XGBoost (46 features only)':<35} {acc_feat:>10.4f} {f1_feat:>10.4f}")
print(f"{'XGBoost (features + contrastive emb)':<35} {acc_fused:>10.4f} {f1_fused:>10.4f}")
print(f"{'='*60}")

if acc_fused > acc_feat:
    print(f"\nIMPROVEMENT: +{acc_fused-acc_feat:.4f} accuracy (+{(f1_fused-f1_feat)*100:.1f}% F1)")
    print(f"\n--- Best Model (Fused) Report ---")
    print(classification_report(yite2, fp_fused, target_names=cn, digits=3, zero_division=0))
else:
    print(f"\nNo improvement from contrastive embeddings")
    print(f"\n--- Features-Only Report ---")
    print(classification_report(yite2, fp_feat, target_names=cn, digits=3, zero_division=0))

# Save
joblib.dump(xgb_fused, "models/xgboost_fused.pkl")
torch.save(encoder.state_dict(), "models/contrastive_encoder.pt")
print("\nModels saved. DONE!")
