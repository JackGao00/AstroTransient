"""AstroTransient 一键运行脚本: 安装依赖 → 下载数据 → 特征工程 → 训练模型"""

import subprocess
import sys
import os

os.chdir(r"D:\AITools\AstroTransient")

def run(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f">>> {desc or cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr[:500])
    if result.returncode != 0:
        print(f"WARNING: exit code {result.returncode}")
    return result.returncode == 0

# Step 1: Install dependencies
run(f"{sys.executable} -m pip install numpy pandas scipy pyyaml astropy scikit-learn xgboost matplotlib seaborn tqdm ipywidgets huggingface-hub h5py joblib torch --quiet", "Step 1: Installing dependencies...")

# Step 2: Download PLAsTiCC dataset
run(f"{sys.executable} -c \"from src.data.download import download_plasticc; download_plasticc('data/raw/plasticc')\"", "Step 2: Downloading PLAsTiCC dataset...")

# Step 3: Load data and do feature engineering
print("\n" + "="*60)
print(">>> Step 3: Loading data & extracting features...")
print("="*60)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

from src.data.download import load_plasticc
from src.data.preprocess import CLASS_NAMES
from src.features.extractor import extract_features

# Load data
train_df, test_df, meta = load_plasticc("data/raw/plasticc")

if meta is not None:
    SAMPLES_PER_CLASS = 200
    np.random.seed(42)

    all_objects = []
    for cls_id in sorted(meta["target"].unique()):
        obj_ids = meta[meta["target"] == cls_id]["object_id"].values
        n_take = min(SAMPLES_PER_CLASS, len(obj_ids))
        chosen = np.random.choice(obj_ids, size=n_take, replace=False)
        for oid in chosen:
            all_objects.append((oid, cls_id))

    print(f"Total objects selected: {len(all_objects)}")

    # Extract features
    features_list, labels_list, skipped = [], [], 0
    for obj_id, cls_id in all_objects:
        obj = train_df[train_df["object_id"] == obj_id]
        band = obj[obj["passband"] == 3]
        if len(band) < 5:
            skipped += 1
            continue
        feat = extract_features(band["mjd"].values, band["flux"].values, band["flux_err"].values)
        features_list.append(feat)
        labels_list.append(cls_id)

    print(f"Features extracted: {len(features_list)}, skipped: {skipped}")

    X = pd.DataFrame(features_list)
    y = pd.Series(labels_list, name="target")

    # Clean
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    zero_var = X.columns[X.std() == 0]
    if len(zero_var) > 0:
        X = X.drop(columns=zero_var)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, stratify=y_temp, random_state=42)

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Save
    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/train.npz", X=X_train.values.astype(np.float32), y=y_train.astype(np.int32))
    np.savez("data/processed/val.npz", X=X_val.values.astype(np.float32), y=y_val.astype(np.int32))
    np.savez("data/processed/test.npz", X=X_test.values.astype(np.float32), y=y_test.astype(np.int32))
    with open("data/processed/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open("data/processed/label_encoder.pkl", "wb") as f: pickle.dump(label_encoder, f)
    print("Data saved to data/processed/")

    # Step 4: Train XGBoost baseline
    print("\n" + "="*60)
    print(">>> Step 4: Training XGBoost baseline...")
    print("="*60)

    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(label_encoder.classes_),
        eval_metric="mlogloss", early_stopping_rounds=20, random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    test_f1 = f1_score(y_test, model.predict(X_test), average="macro")

    print(f"XGBoost Results:")
    print(f"  Train Accuracy:  {train_acc:.4f}")
    print(f"  Val Accuracy:    {val_acc:.4f}")
    print(f"  Test Accuracy:   {test_acc:.4f}")
    print(f"  Test Macro-F1:   {test_f1:.4f}")

    # Per-class performance
    print("\nPer-class F1 scores:")
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    class_names = [CLASS_NAMES.get(c, f"class_{c}") for c in label_encoder.classes_]
    report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
    print(report)

    # Save model
    os.makedirs("models", exist_ok=True)
    import joblib
    joblib.dump(model, "models/xgboost_baseline.pkl")
    print("Model saved to models/xgboost_baseline.pkl")

    # Step 5: Confusion matrix summary
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (top misclassifications):")
    errors_list = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                errors_list.append((cm[i][j], class_names[i], class_names[j]))
    errors_list.sort(reverse=True)
    for count, true_cls, pred_cls in errors_list[:10]:
        print(f"  {true_cls[:40]} -> {pred_cls[:40]}: {count}")

    print("\n" + "="*60)
    print(">>> ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f">>> Final Test Accuracy: {test_acc:.4f}, Macro-F1: {test_f1:.4f}")
    print("="*60)

else:
    print("ERROR: Could not load metadata. Please check PLAsTiCC dataset.")
