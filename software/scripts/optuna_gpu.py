"""Optuna + LightGBM GPU — 200 trials on RTX 4060 Ti"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np, time, json, warnings
warnings.filterwarnings("ignore")

import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import joblib
import xgboost as xgb

from src.data.preprocess import CLASS_NAMES

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Loading data...")
train = np.load("data/processed/train_full.npz")
val = np.load("data/processed/val.npz")
test = np.load("data/processed/test.npz")

Xtr, ytr = train["X"], train["y"]
Xva, yva = val["X"], val["y"]
Xte, yte = test["X"], test["y"]

NC = len(np.unique(ytr))
cn = [CLASS_NAMES.get(i, f"cls_{i}") for i in range(NC)]
print(f"Train: {len(Xtr):,} | Val: {len(Xva):,} | Test: {len(Xte):,} | Classes: {NC}")

# Baseline with LightGBM GPU
print("\nTraining LightGBM baseline...")
lgb_base = lgb.LGBMClassifier(
    n_estimators=300, num_leaves=31, max_depth=8, learning_rate=0.05,
    objective='multiclass', num_class=NC,
    device='gpu', verbose=-1, random_state=42,
)
lgb_base.fit(Xtr, ytr, eval_set=[(Xva, yva)])
base_p = lgb_base.predict(Xte)
base_acc = accuracy_score(yte, base_p)
base_f1 = f1_score(yte, base_p, average="macro")
print(f"LightGBM baseline: Acc={base_acc:.4f} F1={base_f1:.4f}")

# Class weights
cc = np.bincount(ytr)
class_weight = {i: len(ytr)/(NC*max(c,1)) for i, c in enumerate(cc)}

t_total = time.time()

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 1.0, log=True),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 10),
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "objective": "multiclass",
        "num_class": NC,
        "device": "gpu",
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, vl_idx in skf.split(Xtr, ytr):
        Xf, yf = Xtr[tr_idx], ytr[tr_idx]
        Xv, yv = Xtr[vl_idx], ytr[vl_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(Xf, yf, eval_set=[(Xv, yv)])
        scores.append(f1_score(yv, model.predict(Xv), average="macro"))

    return np.mean(scores)

# Phase 1
print("\n=== Phase 1: Broad GPU search (120 trials) ===")
t0 = time.time()
study1 = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    study_name="lgb_gpu_v1",
)
study1.optimize(objective, n_trials=120, show_progress_bar=True)
p1 = study1.best_params
f1_1 = study1.best_value
print(f"Phase 1 Best CV-F1: {f1_1:.4f} | {time.time()-t0:.0f}s")

# Phase 2: Fine
print("\n=== Phase 2: Fine GPU search (80 trials) ===")
def objective_fine(trial):
    boost = p1.get("boosting_type", "gbdt")
    params = {
        "n_estimators": trial.suggest_int("n_estimators", max(100, p1["n_estimators"]-300), p1["n_estimators"]+300, step=50),
        "num_leaves": trial.suggest_int("num_leaves", max(10, p1["num_leaves"]-30), min(127, p1["num_leaves"]+30)),
        "max_depth": trial.suggest_int("max_depth", max(3, p1["max_depth"]-4), min(15, p1["max_depth"]+4)),
        "learning_rate": trial.suggest_float("learning_rate", max(0.005, p1["learning_rate"]*0.3), min(0.2, p1["learning_rate"]*3), log=True),
        "subsample": trial.suggest_float("subsample", max(0.4, p1["subsample"]-0.2), min(1.0, p1["subsample"]+0.2)),
        "colsample_bytree": trial.suggest_float("colsample_bytree", max(0.3, p1["colsample_bytree"]-0.2), min(1.0, p1["colsample_bytree"]+0.2)),
        "reg_alpha": trial.suggest_float("reg_alpha", max(1e-5, p1["reg_alpha"]*0.05), p1["reg_alpha"]*5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", max(1e-4, p1["reg_lambda"]*0.05), p1["reg_lambda"]*5, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", max(2, p1["min_child_samples"]-20), p1["min_child_samples"]+20),
        "min_child_weight": trial.suggest_float("min_child_weight", max(1e-5, p1["min_child_weight"]*0.1), p1["min_child_weight"]*10, log=True),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 10),
        "extra_trees": p1["extra_trees"],
        "boosting_type": boost,
        "objective": "multiclass",
        "num_class": NC,
        "device": "gpu",
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, vl_idx in skf.split(Xtr, ytr):
        Xf, yf = Xtr[tr_idx], ytr[tr_idx]
        Xv, yv = Xtr[vl_idx], ytr[vl_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(Xf, yf, eval_set=[(Xv, yv)])
        scores.append(f1_score(yv, model.predict(Xv), average="macro"))
    return np.mean(scores)

study2 = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=43, multivariate=True),
    study_name="lgb_gpu_v2",
)
study2.optimize(objective_fine, n_trials=80, show_progress_bar=True)

# ====== Final LightGBM ======
print("\n=== Training final LightGBM model ===")
best_params = study2.best_params
best_params["objective"] = "multiclass"
best_params["num_class"] = NC
best_params["device"] = "gpu"
best_params["verbose"] = -1
best_params["random_state"] = 42

print(f"Best CV-F1: {study2.best_value:.4f}")

final_lgb = lgb.LGBMClassifier(**best_params)
final_lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)])

lgb_p = final_lgb.predict(Xte)
lgb_acc = accuracy_score(yte, lgb_p)
lgb_f1 = f1_score(yte, lgb_p, average="macro")

# ====== Also train XGBoost with best LightGBM-inspired params ======
print("\nTraining XGBoost comparison...")
xgb_m = xgb.XGBClassifier(
    n_estimators=best_params.get("n_estimators", 500),
    max_depth=best_params.get("max_depth", 8),
    learning_rate=best_params.get("learning_rate", 0.03),
    subsample=best_params.get("subsample", 0.8),
    colsample_bytree=best_params.get("colsample_bytree", 0.7),
    reg_alpha=best_params.get("reg_alpha", 0.5),
    reg_lambda=best_params.get("reg_lambda", 0.5),
    min_child_weight=int(best_params.get("min_child_weight", 1)*10),
    objective="multi:softprob", num_class=NC,
    eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
)
xgb_m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
xgb_p = xgb_m.predict(Xte)
xgb_acc = accuracy_score(yte, xgb_p)
xgb_f1 = f1_score(yte, xgb_p, average="macro")

# ====== REPORT ======
print(f"\n{'='*55}")
print(f"OPTUNA GPU RESULTS — LightGBM + XGBoost")
print(f"{'='*55}")
print(f"Total time: {time.time()-t_total:.0f}s | 200 trials")
print(f"{'='*55}")
print(f"{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"{'-'*45}")
print(f"{'LightGBM (baseline)':<25} {base_acc:>10.4f} {base_f1:>10.4f}")
print(f"{'LightGBM (Optuna GPU)':<25} {lgb_acc:>10.4f} {lgb_f1:>10.4f}")
print(f"{'XGBoost (Optuna params)':<25} {xgb_acc:>10.4f} {xgb_f1:>10.4f}")
print(f"{'='*55}")

# Best
if lgb_acc > xgb_acc:
    final_model, final_p, best_name = final_lgb, lgb_p, "LightGBM Optuna"
else:
    final_model, final_p, best_name = xgb_m, xgb_p, "XGBoost (LightGBM-tuned)"

print(f"\nBest: {best_name}")

print(f"\n--- {best_name} Report ---")
print(classification_report(yte, final_p, target_names=cn, digits=3, zero_division=0))

# Save best
joblib.dump(final_model, "models/best_gpu_model.pkl")
print(f"\nBest model saved -> models/best_gpu_model.pkl")
print(f"Best params: {json.dumps(best_params, indent=2)}")
print("DONE!")
