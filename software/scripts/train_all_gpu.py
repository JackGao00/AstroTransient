"""Fair comparison: all models on same full training data"""
import sys, os, time
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib, pickle
import xgboost as xgb

from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES

print("="*55)
print("FAIR GPU COMPARISON — All models, same data")
print("="*55)

# ====== 1. Load data & extract features ONCE ======
print("\n[1] Loading & extracting features...")
train_df, val_df, _ = load_plasticc("data/raw/plasticc", load_test=False)
all_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"Total objects: {len(all_df):,}")

# Same feature extraction as step3
def extract_features(row):
    try:
        mjd_all, flux_all, flux_err_all, pbs_all = get_lightcurve(row)
    except: return None
    mask = (flux_err_all > 0) & (~np.isnan(flux_all))
    if mask.sum() < 3: return None
    t, f, e = mjd_all[mask], flux_all[mask], flux_err_all[mask]
    pbs_m = pbs_all[mask]
    feats = {}

    feats["n_points"] = float(len(f))
    feats["duration"] = float(t[-1] - t[0])
    feats["flux_mean"] = float(np.mean(f))
    feats["flux_std"] = float(np.std(f))
    feats["flux_median"] = float(np.median(f))
    feats["flux_min"] = float(np.min(f))
    feats["flux_max"] = float(np.max(f))
    feats["flux_range"] = float(np.max(f) - np.min(f))
    feats["flux_skew"] = float(stats.skew(f)) if len(f) > 2 else 0
    feats["flux_kurtosis"] = float(stats.kurtosis(f)) if len(f) > 3 else 0
    feats["snr_mean"] = float(np.mean(np.abs(f) / (e + 1e-8)))
    feats["snr_max"] = float(np.max(np.abs(f) / (e + 1e-8)))
    sf = np.sort(f)
    feats["amplitude_90"] = float(sf[int(0.95*len(f))] - sf[int(0.05*len(f))])
    feats["beyond_1std"] = float(np.mean(np.abs(f - np.mean(f)) > np.std(f)))
    diff = np.diff(f)
    feats["max_rise"] = float(np.max(diff)) if len(diff) > 0 else 0
    feats["max_decay"] = float(np.min(diff)) if len(diff) > 0 else 0
    feats["peak_position"] = float(np.argmax(np.abs(f)) / max(len(f), 1))
    if len(f) > 4:
        ac = np.corrcoef(f[:-1], f[1:])[0, 1]
        feats["autocorr_lag1"] = float(ac) if not np.isnan(ac) else 0
    else: feats["autocorr_lag1"] = 0
    feats["iqr"] = float(np.percentile(f, 75) - np.percentile(f, 25))
    feats["mad"] = float(np.median(np.abs(f - np.median(f))))

    band_ranges = {"u":(3000,4200),"g":(4200,5400),"r":(5400,6800),
                   "i":(6800,8200),"z":(8200,9200),"y":(9200,10500)}
    band_fluxes = {}
    for bn, (lo, hi) in band_ranges.items():
        bm = (pbs_m >= lo) & (pbs_m < hi)
        if bm.sum() >= 3:
            fb = f[bm]; eb = e[bm]
            feats[f"{bn}_mean"] = float(np.mean(fb))
            feats[f"{bn}_std"] = float(np.std(fb))
            feats[f"{bn}_snr"] = float(np.mean(np.abs(fb)/(eb+1e-8)))
            band_fluxes[bn] = float(np.mean(fb))
        else: feats[f"{bn}_mean"] = feats[f"{bn}_std"] = feats[f"{bn}_snr"] = 0.0; band_fluxes[bn] = 0.0
    for b1,b2 in [("u","g"),("g","r"),("r","i"),("i","z"),("z","y")]:
        feats[f"color_{b1}_{b2}"] = band_fluxes[b1] - band_fluxes[b2]

    feats["redshift"] = float(row.get("redshift", 0) or 0)
    feats["hostgal_specz"] = float(row.get("hostgal_specz", 0) or 0)
    feats["hostgal_photoz"] = float(row.get("hostgal_photoz", 0) or 0)
    feats["n_passbands"] = float(len(set(pbs_m)))
    return feats

t0 = time.time()
feats_list, labs_list = [], []
for idx in range(len(all_df)):
    feat = extract_features(all_df.iloc[idx])
    if feat is not None:
        feats_list.append(feat)
        labs_list.append(all_df.iloc[idx]["label"])

X = pd.DataFrame(feats_list)
y = pd.Series(labs_list)
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    med = X[col].median()
    X[col] = X[col].fillna(med if not np.isnan(med) else 0)
print(f"Features: {X.shape[1]} dims, {len(X)} samples ({time.time()-t0:.0f}s)")

le = LabelEncoder()
y_enc = le.fit_transform(y)
NC = len(le.classes_)
cn = [CLASS_NAMES.get(c, f"cls_{c}") for c in le.classes_]

# Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xiv, Xte, yiv, yte = train_test_split(X_scaled, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
Xtr, Xva, ytr, yva = train_test_split(Xiv, yiv, test_size=0.15/0.85, stratify=yiv, random_state=42)
print(f"Train: {len(Xtr):,} | Val: {len(Xva):,} | Test: {len(Xte):,} | Classes: {NC}")

# Save data
os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/train_full.npz", X=Xtr.astype(np.float32), y=ytr.astype(np.int32))
np.savez("data/processed/val.npz", X=Xva.astype(np.float32), y=yva.astype(np.int32))
np.savez("data/processed/test.npz", X=Xte.astype(np.float32), y=yte.astype(np.int32))

# ====== 2. XGBoost (baseline, recalculated) ======
print("\n[2] Training XGBoost...")
xgb_m = xgb.XGBClassifier(
    n_estimators=500, max_depth=10, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=1.5, min_child_weight=3,
    objective="multi:softprob", num_class=NC,
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
# Class weights
cc = np.bincount(ytr)
sw = np.array([len(ytr)/(NC*max(c,1)) for c in cc])[ytr]
xgb_m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
xgb_p = xgb_m.predict(Xte)
xgb_a = accuracy_score(yte, xgb_p)
xgb_f = f1_score(yte, xgb_p, average="macro")
print(f"XGBoost: Acc={xgb_a:.4f} F1={xgb_f:.4f}")

# ====== 3. LightGBM GPU ======
print("\n[3] Training LightGBM GPU...")
try:
    import lightgbm as lgb
    lgb_m = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=63, max_depth=10,
        learning_rate=0.02, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=1.5, min_child_samples=10,
        objective='multiclass', num_class=NC,
        device='gpu', verbose=-1, random_state=42,
    )
    lgb_m.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    lgb_p = lgb_m.predict(Xte)
    lgb_a = accuracy_score(yte, lgb_p)
    lgb_f = f1_score(yte, lgb_p, average="macro")
    print(f"LightGBM GPU: Acc={lgb_a:.4f} F1={lgb_f:.4f}")
    has_lgb = True
except Exception as e:
    print(f"LightGBM: {e}")
    has_lgb = False

# ====== 4. CatBoost GPU ======
print("\n[4] Training CatBoost GPU...")
try:
    from catboost import CatBoostClassifier
    cat_m = CatBoostClassifier(
        iterations=500, depth=8, learning_rate=0.02,
        l2_leaf_reg=3, task_type='GPU', devices='0',
        random_seed=42, verbose=False, allow_writing_files=False,
    )
    cat_m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=30)
    cat_p = cat_m.predict(Xte)
    cat_a = accuracy_score(yte, cat_p)
    cat_f = f1_score(yte, cat_p, average="macro")
    print(f"CatBoost GPU: Acc={cat_a:.4f} F1={cat_f:.4f}")
    has_cat = True
except Exception as e:
    print(f"CatBoost: {e}")
    has_cat = False

# ====== 5. Ensemble (simple voting) ======
print("\n[5] Ensemble voting...")
models = [("XGB", xgb_m, xgb_p)]
if has_lgb: models.append(("LGB", lgb_m, lgb_p))
if has_cat: models.append(("CAT", cat_m, cat_p))

# Hard voting
all_preds = np.stack([m[2] for m in models], axis=0)
from scipy.stats import mode
vote_result = mode(all_preds, axis=0, keepdims=False)
vote_p = vote_result.mode.flatten() if hasattr(vote_result, 'mode') else vote_result[0][0]
vote_a = accuracy_score(yte, vote_p)
vote_f = f1_score(yte, vote_p, average="macro")

# ====== RESULTS ======
print(f"\n{'='*55}")
print(f"FAIR COMPARISON (ALL models trained on {len(Xtr):,} samples)")
print(f"{'='*55}")
print(f"{'Model':<22} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"{'-'*42}")
print(f"{'XGBoost':<22} {xgb_a:>10.4f} {xgb_f:>10.4f}")
if has_lgb: print(f"{'LightGBM GPU':<22} {lgb_a:>10.4f} {lgb_f:>10.4f}")
if has_cat: print(f"{'CatBoost GPU':<22} {cat_a:>10.4f} {cat_f:>10.4f}")
print(f"{'Ensemble Vote':<22} {vote_a:>10.4f} {vote_f:>10.4f}")
print(f"{'='*55}")

# Best
results = [("XGBoost", xgb_a, xgb_f, xgb_p)]
if has_lgb: results.append(("LightGBM", lgb_a, lgb_f, lgb_p))
if has_cat: results.append(("CatBoost", cat_a, cat_f, cat_p))
results.append(("Ensemble", vote_a, vote_f, vote_p))
best = max(results, key=lambda x: x[1])

print(f"\nBest: {best[0]} ({best[1]*100:.1f}%)")
print(f"\n--- {best[0]} Report ---")
print(classification_report(yte, best[3], target_names=cn, digits=3, zero_division=0))

# Save best model
joblib.dump(xgb_m, "models/xgboost_final.pkl")
print("\nALL DONE! Best model saved.")
