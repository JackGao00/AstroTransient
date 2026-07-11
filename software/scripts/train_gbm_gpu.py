"""GPU GBM: LightGBM + CatBoost + Stacking — simple & clean"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data.preprocess import CLASS_NAMES

print("Loading features...")
d = np.load("data/processed/test.npz")
X, y = d["X"], d["y"]

# Resplit for proper training
Xiv, Xte, yiv, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
Xtr, Xva, ytr, yva = train_test_split(Xiv, yiv, test_size=0.2, stratify=yiv, random_state=42)

NC = len(np.unique(y))
cn = [CLASS_NAMES.get(i, f"cls_{i}") for i in range(NC)]
print(f"Train: {len(Xtr)} Val: {len(Xva)} Test: {len(Xte)} | Classes: {NC}")

# ====== XGBoost ======
xgb = joblib.load("models/xgboost_optimized.pkl")
xgb_p = xgb.predict(Xte)
xgb_a = accuracy_score(yte, xgb_p)
xgb_f = f1_score(yte, xgb_p, average="macro")
print(f"\nXGBoost:      Acc={xgb_a:.4f}  F1={xgb_f:.4f}")

# ====== LightGBM GPU ======
try:
    import lightgbm as lgb
    lgb_m = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=63, max_depth=10,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=1.5, min_child_samples=10,
        objective='multiclass', num_class=NC,
        device='gpu', verbose=-1, random_state=42,
    )
    lgb_m.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    lgb_p = lgb_m.predict(Xte)
    lgb_a = accuracy_score(yte, lgb_p)
    lgb_f = f1_score(yte, lgb_p, average="macro")
    print(f"LightGBM GPU: Acc={lgb_a:.4f}  F1={lgb_f:.4f}")
    has_lgb = True
except Exception as e:
    print(f"LightGBM failed: {e}")
    lgb_m, has_lgb = None, False

# ====== CatBoost GPU ======
try:
    from catboost import CatBoostClassifier
    cat_m = CatBoostClassifier(
        iterations=500, depth=8, learning_rate=0.03,
        l2_leaf_reg=3, task_type='GPU', devices='0',
        random_seed=42, verbose=False, allow_writing_files=False,
    )
    cat_m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=30)
    cat_p = cat_m.predict(Xte)
    cat_a = accuracy_score(yte, cat_p)
    cat_f = f1_score(yte, cat_p, average="macro")
    print(f"CatBoost GPU: Acc={cat_a:.4f}  F1={cat_f:.4f}")
    has_cat = True
except Exception as e:
    print(f"CatBoost failed: {e}")
    cat_m, has_cat = None, False

# ====== STACKING ENSEMBLE ======
print("\n=== Stacking Ensemble ===")
model_list = [("xgb", xgb)]
if has_lgb: model_list.append(("lgb", lgb_m))
if has_cat: model_list.append(("cat", cat_m))

NM = len(model_list)
meta_dim = NC * NM

# Build meta-features via 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_tr = np.zeros((len(Xtr), meta_dim), dtype=np.float32)
meta_te = np.zeros((len(Xte), meta_dim), dtype=np.float32)

for fold, (tidx, _) in enumerate(skf.split(Xtr, ytr)):
    Xf, yf = Xtr[tidx], ytr[tidx]
    for mi, (name, _) in enumerate(model_list):
        if name == "xgb":
            m = joblib.load("models/xgboost_optimized.pkl")
        elif name == "lgb":
            m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=8,
                                    learning_rate=0.05, objective='multiclass',
                                    num_class=NC, device='gpu', verbose=-1, random_state=42)
            m.fit(Xf, yf, verbose=False)
        else:
            m = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05,
                                    task_type='GPU', devices='0', random_seed=42,
                                    verbose=False, allow_writing_files=False)
            m.fit(Xf, yf, verbose=False)
        meta_tr[tr_idx, mi*NC:(mi+1)*NC] = m.predict_proba(Xtr[tr_idx])

# Test meta-features
for mi, (name, m) in enumerate(model_list):
    meta_te[:, mi*NC:(mi+1)*NC] = m.predict_proba(Xte)

# Meta-learner: MLP on GPU
DEV = torch.device("cuda")
meta_mlp = nn.Sequential(
    nn.Linear(meta_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
    nn.Linear(128, NC)
).cuda()

opt = torch.optim.Adam(meta_mlp.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()
tdl = DataLoader(TensorDataset(torch.tensor(meta_tr), torch.tensor(ytr, dtype=torch.long)), 128, shuffle=True)

best, pat = 0, 0
for ep in range(80):
    meta_mlp.train()
    for xb, yb in tdl:
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad(); crit(meta_mlp(xb), yb).backward(); opt.step()
    meta_mlp.eval()
    with torch.no_grad():
        mp = meta_mlp(torch.tensor(meta_tr).cuda()).argmax(1).cpu().numpy()
    acc = accuracy_score(ytr, mp)
    if acc > best: best = acc; pat = 0
    else: pat += 1
    if pat >= 15: break

meta_mlp.eval()
with torch.no_grad():
    mp = meta_mlp(torch.tensor(meta_te).cuda()).argmax(1).cpu().numpy()
stack_a = accuracy_score(yte, mp)
stack_f = f1_score(yte, mp, average="macro")

# ====== RESULTS ======
print(f"\n{'='*55}")
print(f"GPU GBM FINAL RESULTS")
print(f"{'='*55}")
print(f"{'Model':<22} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"{'-'*42}")
print(f"{'XGBoost':<22} {xgb_a:>10.4f} {xgb_f:>10.4f}")
if has_lgb: print(f"{'LightGBM GPU':<22} {lgb_a:>10.4f} {lgb_f:>10.4f}")
if has_cat: print(f"{'CatBoost GPU':<22} {cat_a:>10.4f} {cat_f:>10.4f}")
print(f"{'Stacking (GPU)':<22} {stack_a:>10.4f} {stack_f:>10.4f}")
print(f"{'='*55}")

best_m = max([("XGBoost", xgb_a, xgb_f, xgb_p)] +
             ([("LightGBM", lgb_a, lgb_f, lgb_p)] if has_lgb else []) +
             ([("CatBoost", cat_a, cat_f, cat_p)] if has_cat else []) +
             [("Stacking", stack_a, stack_f, mp)],
             key=lambda x: x[1])

print(f"\nBest: {best_m[0]} — {best_m[1]*100:.1f}%")
print(f"\n--- {best_m[0]} Report ---")
print(classification_report(yte, best_m[3], target_names=cn, digits=3, zero_division=0))
print("DONE!")
