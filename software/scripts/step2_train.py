"""Step 2: Feature extraction + XGBoost training (improved)"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pickle, joblib, warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES

print("Loading PLAsTiCC data...")
train_df, _, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=None)

if train_df is None:
    print("ERROR: No data!")
    exit(1)

# Use ALL objects (no sampling limit)
SAMPLES_PER_CLASS = 1000
np.random.seed(42)

all_objects = []
for cls_id in sorted(train_df["label"].unique()):
    obj_mask = train_df["label"] == cls_id
    obj_indices = train_df[obj_mask].index.values
    n_take = min(SAMPLES_PER_CLASS, len(obj_indices))
    chosen = np.random.choice(obj_indices, size=n_take, replace=False)
    for idx in chosen:
        all_objects.append((idx, cls_id))

print(f"Total objects: {len(all_objects)}")

# Enhanced feature extraction with band-specific features
def extract_enhanced_features(row):
    """Extract features from all passbands plus per-band features"""
    from scipy import stats

    try:
        mjd_all, flux_all, flux_err_all, pbs_all = get_lightcurve(row)
    except Exception:
        return None

    mask = (flux_err_all > 0) & (~np.isnan(flux_all))
    if mask.sum() < 3:
        return None

    t = mjd_all[mask]
    f = flux_all[mask]
    e = flux_err_all[mask]

    features = {}

    # Overall stats
    features["duration"] = float(t[-1] - t[0])
    features["n_points"] = float(len(f))
    features["flux_mean"] = float(np.mean(f))
    features["flux_std"] = float(np.std(f))
    features["flux_median"] = float(np.median(f))
    features["flux_min"] = float(np.min(f))
    features["flux_max"] = float(np.max(f))
    features["flux_range"] = float(np.max(f) - np.min(f))
    features["flux_skew"] = float(stats.skew(f)) if len(f) > 2 else 0.0
    features["flux_kurtosis"] = float(stats.kurtosis(f)) if len(f) > 2 else 0.0
    features["snr_mean"] = float(np.mean(np.abs(f) / (e + 1e-8)))
    features["snr_max"] = float(np.max(np.abs(f) / (e + 1e-8)))

    sorted_f = np.sort(f)
    p5 = sorted_f[int(0.05 * len(f))]
    p95 = sorted_f[int(0.95 * len(f))]
    features["amplitude_90"] = float(p95 - p5)
    features["beyond_1std"] = float(np.mean(np.abs(f - np.mean(f)) > np.std(f)))

    flux_diff = np.diff(f)
    features["max_rise"] = float(np.max(flux_diff)) if len(flux_diff) > 0 else 0.0
    features["max_decay"] = float(np.min(flux_diff)) if len(flux_diff) > 0 else 0.0

    if len(f) > 4:
        ac = np.corrcoef(f[:-1], f[1:])[0, 1]
        features["autocorr_lag1"] = float(ac) if not np.isnan(ac) else 0.0
    else:
        features["autocorr_lag1"] = 0.0

    features["peak_position"] = float(np.argmax(f) / max(len(f), 1))
    features["num_passbands"] = float(len(set(pbs_all[mask])))

    # Per-band features (key bands in PLAsTiCC: u~3671, g~4827, r~6222, i~7546, z~8691)
    band_centers = {"u": 3700, "g": 4800, "r": 6200, "i": 7500, "z": 8700, "y": 9700}
    for band_name, center in band_centers.items():
        band_mask = (np.abs(pbs_all - center) < 500) & mask
        if band_mask.sum() >= 3:
            f_band = flux_all[band_mask]
            features[f"{band_name}_mean"] = float(np.mean(f_band))
            features[f"{band_name}_std"] = float(np.std(f_band))
            features[f"{band_name}_max"] = float(np.max(f_band))
        else:
            features[f"{band_name}_mean"] = 0.0
            features[f"{band_name}_std"] = 0.0
            features[f"{band_name}_max"] = 0.0

    # Color features (differences between bands)
    for b1, b2 in [("g", "r"), ("r", "i"), ("u", "g"), ("i", "z")]:
        m1 = features.get(f"{b1}_mean", 0)
        m2 = features.get(f"{b2}_mean", 0)
        features[f"color_{b1}_{b2}"] = m1 - m2

    return features

# Extract features
print("Extracting enhanced features...")
features_list, labels_list, skipped = [], [], 0
for idx, cls_id in all_objects:
    row = train_df.loc[idx]
    feat = extract_enhanced_features(row)
    if feat is None:
        skipped += 1
        continue
    features_list.append(feat)
    labels_list.append(cls_id)

print(f"Extracted: {len(features_list)}, skipped: {skipped}, features dim: {len(features_list[0])}")

X = pd.DataFrame(features_list)
y = pd.Series(labels_list, name="target")

# Clean
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)

zero_var = X.columns[X.std() == 0]
if len(zero_var) > 0:
    X = X.drop(columns=zero_var)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, stratify=y_temp, random_state=42)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/train.npz", X=X_train.astype(np.float32), y=y_train.astype(np.int32))
np.savez("data/processed/val.npz", X=X_val.astype(np.float32), y=y_val.astype(np.int32))
np.savez("data/processed/test.npz", X=X_test.astype(np.float32), y=y_test.astype(np.int32))
with open("data/processed/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
with open("data/processed/label_encoder.pkl", "wb") as f: pickle.dump(label_encoder, f)

# Train XGBoost with hyperparameter search
num_classes = len(label_encoder.classes_)
print(f"\nTraining XGBoost ({num_classes} classes)...")

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=10, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    objective="multi:softprob", num_class=num_classes,
    eval_metric="mlogloss", early_stopping_rounds=30, random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average="macro")

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"XGBoost Test Accuracy:  {test_acc:.4f}")
print(f"XGBoost Test Macro-F1:  {test_f1:.4f}")
print(f"{'='*60}")

class_names = [CLASS_NAMES.get(c, f"class_{c}") for c in label_encoder.classes_]
print(f"\nClassification Report ({len(class_names)} classes):")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3, zero_division=0))

# Top confusions
cm = confusion_matrix(y_test, y_pred)
print("\nTop Misclassifications:")
errors = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i][j] > 0:
            errors.append((cm[i][j], class_names[i], class_names[j]))
errors.sort(reverse=True)
for cnt, t, p in errors[:12]:
    print(f"  {cnt:3d}  {t[:45]} -> {p[:45]}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_baseline.pkl")
print("\nModel saved. DONE!")
