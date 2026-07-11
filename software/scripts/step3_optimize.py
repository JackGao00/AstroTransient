"""Step 3: 模型优化 — 数据增强 + 超参调优 + 类别平衡"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np
import pandas as pd
import pickle, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy import stats
import xgboost as xgb

from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES

print("=" * 60)
print("STEP 3: MODEL OPTIMIZATION")
print("=" * 60)

# ---- 1. Load ALL available data ----
print("\n[1/5] Loading all data (train + validation)...")
train_df, val_df, _ = load_plasticc("data/raw/plasticc", load_test=False)
all_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"Combined: {len(all_df):,} objects ({len(train_df):,} train + {len(val_df):,} val)")

# ---- 2. Enhanced feature extraction ----
print("\n[2/5] Extracting features from all objects...")

def extract_full_features(row):
    """46 features: global + per-band + color + metadata"""
    try:
        mjd_all, flux_all, flux_err_all, pbs_all = get_lightcurve(row)
    except:
        return None

    mask = (flux_err_all > 0) & (~np.isnan(flux_all))
    if mask.sum() < 3:
        return None

    t, f, e = mjd_all[mask], flux_all[mask], flux_err_all[mask]
    pbs_masked = pbs_all[mask]
    feats = {}

    # ---- Global features ----
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
    p5, p95 = sf[int(0.05 * len(f))], sf[int(0.95 * len(f))]
    feats["amplitude_90"] = float(p95 - p5)
    feats["beyond_1std"] = float(np.mean(np.abs(f - np.mean(f)) > np.std(f)))

    diff = np.diff(f)
    feats["max_rise"] = float(np.max(diff)) if len(diff) > 0 else 0
    feats["max_decay"] = float(np.min(diff)) if len(diff) > 0 else 0
    feats["peak_position"] = float(np.argmax(np.abs(f)) / max(len(f), 1))

    if len(f) > 4:
        ac = np.corrcoef(f[:-1], f[1:])[0, 1]
        feats["autocorr_lag1"] = float(ac) if not np.isnan(ac) else 0
    else:
        feats["autocorr_lag1"] = 0

    # Robust statistics
    feats["iqr"] = float(np.percentile(f, 75) - np.percentile(f, 25))
    feats["mad"] = float(np.median(np.abs(f - np.median(f))))  # Median absolute deviation

    # ---- Per-band features ----
    band_ranges = {"u": (3000, 4200), "g": (4200, 5400), "r": (5400, 6800),
                   "i": (6800, 8200), "z": (8200, 9200), "y": (9200, 10500)}
    band_fluxes = {}

    for band_name, (lo, hi) in band_ranges.items():
        bm = (pbs_masked >= lo) & (pbs_masked < hi)
        if bm.sum() >= 3:
            fb = f[bm]
            eb = e[bm]
            feats[f"{band_name}_mean"] = float(np.mean(fb))
            feats[f"{band_name}_std"] = float(np.std(fb))
            feats[f"{band_name}_snr"] = float(np.mean(np.abs(fb) / (eb + 1e-8)))
            band_fluxes[band_name] = float(np.mean(fb))
        else:
            feats[f"{band_name}_mean"] = 0.0
            feats[f"{band_name}_std"] = 0.0
            feats[f"{band_name}_snr"] = 0.0
            band_fluxes[band_name] = 0.0

    # ---- Color features ----
    for b1, b2 in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("z", "y")]:
        feats[f"color_{b1}_{b2}"] = band_fluxes[b1] - band_fluxes[b2]

    # ---- Metadata features ----
    feats["redshift"] = float(row.get("redshift", 0) or 0)
    feats["hostgal_specz"] = float(row.get("hostgal_specz", 0) or 0)
    feats["hostgal_photoz"] = float(row.get("hostgal_photoz", 0) or 0)

    feats["n_passbands"] = float(len(set(pbs_masked)))

    return feats

# Extract for all objects
np.random.seed(42)
features_list, labels_list, skipped = [], [], 0
for idx in range(len(all_df)):
    row = all_df.iloc[idx]
    feat = extract_full_features(row)
    if feat is None:
        skipped += 1
        continue
    features_list.append(feat)
    labels_list.append(all_df.iloc[idx]["label"])

print(f"Extracted: {len(features_list)} features, {len(features_list[0])} dims, skipped: {skipped}")

X = pd.DataFrame(features_list)
y = pd.Series(labels_list)

# Clean
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    med = X[col].median()
    X[col] = X[col].fillna(med if not np.isnan(med) else 0)

zero_var = X.columns[X.std() == 0]
if len(zero_var) > 0:
    X = X.drop(columns=zero_var)
print(f"After cleaning: {X.shape[1]} features")

# Encode
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"Classes: {num_classes}")
for i, c in enumerate(label_encoder.classes_):
    count = (y == c).sum()
    name = CLASS_NAMES.get(c, f"class_{c}")
    print(f"  {i:2d}  {name[:40]:40s}  n={count}")

# Split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_temp)
X_test_s = scaler.transform(X_test)
y_train = y_temp
y_test_s = y_test
print(f"\nTrain: {X_train.shape[0]}, Test: {X_test_s.shape[0]}")

# ---- 3. Hyperparameter tuning (simplified grid) ----
print("\n[3/5] Hyperparameter tuning...")

# Compute scale_pos_weight for imbalanced classes
class_counts = np.bincount(y_train)
scale_weights = {i: len(y_train) / (num_classes * max(c, 1)) for i, c in enumerate(class_counts)}
sample_weights = np.array([scale_weights[y] for y in y_train])

param_sets = [
    {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03, "name": "baseline"},
    {"n_estimators": 500, "max_depth": 10, "learning_rate": 0.02, "name": "deeper"},
    {"n_estimators": 400, "max_depth": 12, "learning_rate": 0.02, "name": "deepest"},
]

best_model, best_f1 = None, 0
for params in param_sets:
    name = params.pop("name")
    model = xgb.XGBClassifier(
        **params,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.5, min_child_weight=3,
        objective="multi:softprob", num_class=num_classes,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
        y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]
        sw_tr = sample_weights[tr_idx]

        model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
        y_pred_vl = model.predict(X_vl)
        cv_scores.append(f1_score(y_vl, y_pred_vl, average="macro"))

    mean_f1 = np.mean(cv_scores)
    print(f"  {name}: CV Macro-F1 = {mean_f1:.4f} (+/- {np.std(cv_scores):.4f})")

    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_params = {**params, "name": name}

# ---- 4. Train final model with best params ----
print(f"\n[4/5] Training final model (best: {best_params['name']})...")
best_params.pop("name")

final_model = xgb.XGBClassifier(
    **best_params,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=1.5, min_child_weight=3,
    objective="multi:softprob", num_class=num_classes,
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
final_model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

y_pred = final_model.predict(X_test_s)
test_acc = accuracy_score(y_test_s, y_pred)
test_f1 = f1_score(y_test_s, y_pred, average="macro")
test_wf1 = f1_score(y_test_s, y_pred, average="weighted")

# ---- 5. Results ----
print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"Accuracy:    {test_acc:.4f}")
print(f"Macro-F1:    {test_f1:.4f}")
print(f"Weighted-F1: {test_wf1:.4f}")
print(f"{'='*60}")

class_names = [CLASS_NAMES.get(c, f"class_{c}") for c in label_encoder.classes_]
print(f"\nPer-class F1:")
report = classification_report(y_test_s, y_pred, target_names=class_names, digits=3, zero_division=0)
print(report)

# Top confusions
cm = confusion_matrix(y_test_s, y_pred)
print("\nTop Misclassifications:")
errors = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i][j] > 0:
            errors.append((cm[i][j], class_names[i], class_names[j]))
errors.sort(reverse=True)
for cnt, t, p in errors[:15]:
    print(f"  {cnt:3d}  {t[:45]} -> {p[:45]}")

# Save everything
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
joblib.dump(final_model, "models/xgboost_optimized.pkl")
joblib.dump(scaler, "data/processed/scaler.pkl")
joblib.dump(label_encoder, "data/processed/label_encoder.pkl")
np.savez("data/processed/train_full.npz", X=X_train.astype(np.float32), y=y_train.astype(np.int32))
np.savez("data/processed/test.npz", X=X_test_s.astype(np.float32), y=y_test_s.astype(np.int32))

# Save feature importances
importance = final_model.feature_importances_
feat_names = X.columns.tolist()
sorted_idx = np.argsort(importance)[::-1]
print("\nTop 10 Features:")
for i in range(10):
    idx = sorted_idx[i]
    print(f"  {i+1:2d}. {feat_names[idx]:30s} = {importance[idx]:.4f}")

print("\nALL DONE! Model saved to models/xgboost_optimized.pkl")
