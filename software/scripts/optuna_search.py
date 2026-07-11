"""Optuna hyperparameter optimization for XGBoost"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import joblib

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

# Baseline: train a quick XGBoost with default-ish params on THIS data
print("Training baseline...")
xgb_base = xgb.XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.03,
    objective="multi:softprob", num_class=NC,
    eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
)
xgb_base.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
base_p = xgb_base.predict(Xte)
base_acc = (base_p == yte).mean()
base_f1 = f1_score(yte, base_p, average="macro")
print(f"Baseline: Acc={base_acc:.4f} F1={base_f1:.4f}")

# Class weights
cc = np.bincount(ytr)
cw = {i: len(ytr)/(NC*max(c,1)) for i, c in enumerate(cc)}
sw = np.array([cw[y] for y in ytr])

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 3.0),
        "objective": "multi:softprob",
        "num_class": NC,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    # 3-fold CV for robust evaluation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, vl_idx in skf.split(Xtr, ytr):
        Xf, yf = Xtr[tr_idx], ytr[tr_idx]
        Xv, yv = Xtr[vl_idx], ytr[vl_idx]
        sw_f = np.array([cw[y] for y in yf])

        model = xgb.XGBClassifier(**params)
        model.fit(Xf, yf, sample_weight=sw_f, eval_set=[(Xv, yv)], verbose=False)
        yp = model.predict(Xv)
        scores.append(f1_score(yv, yp, average="macro"))

    return np.mean(scores)

# ====== Phase 1: Broad search ======
print("\n=== Phase 1: Broad search (50 trials) ===")
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    study_name="xgb_optuna_v1",
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

best_p1 = study.best_params
best_v1 = study.best_value
print(f"\nPhase 1 Best: CV-F1={best_v1:.4f}")
print(f"Params: {best_p1}")

# ====== Phase 2: Fine search around best ======
print("\n=== Phase 2: Fine search (30 trials) ===")
study2 = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=43),
    study_name="xgb_optuna_v2",
)

def objective_fine(trial):
    # Narrow ranges around best params from Phase 1
    params = {
        "n_estimators": trial.suggest_int("n_estimators", max(200, best_p1["n_estimators"]-200), best_p1["n_estimators"]+200, step=50),
        "max_depth": trial.suggest_int("max_depth", max(4, best_p1["max_depth"]-3), best_p1["max_depth"]+3),
        "learning_rate": trial.suggest_float("learning_rate", max(0.005, best_p1["learning_rate"]*0.5), min(0.1, best_p1["learning_rate"]*2), log=True),
        "subsample": trial.suggest_float("subsample", max(0.5, best_p1["subsample"]-0.15), min(1.0, best_p1["subsample"]+0.15)),
        "colsample_bytree": trial.suggest_float("colsample_bytree", max(0.4, best_p1["colsample_bytree"]-0.15), min(0.95, best_p1["colsample_bytree"]+0.15)),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", max(0.4, best_p1["colsample_bylevel"]-0.15), min(0.95, best_p1["colsample_bylevel"]+0.15)),
        "reg_alpha": trial.suggest_float("reg_alpha", max(0.001, best_p1["reg_alpha"]*0.1), min(10.0, best_p1["reg_alpha"]*3), log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", max(0.01, best_p1["reg_lambda"]*0.1), min(20.0, best_p1["reg_lambda"]*3), log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", max(1, best_p1["min_child_weight"]-5), best_p1["min_child_weight"]+5),
        "gamma": trial.suggest_float("gamma", max(0.0, best_p1["gamma"]-1.0), best_p1["gamma"]+1.0),
        "max_delta_step": trial.suggest_float("max_delta_step", max(0.0, best_p1["max_delta_step"]-1.5), best_p1["max_delta_step"]+1.5),
        "objective": "multi:softprob",
        "num_class": NC,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, vl_idx in skf.split(Xtr, ytr):
        Xf, yf = Xtr[tr_idx], ytr[tr_idx]
        Xv, yv = Xtr[vl_idx], ytr[vl_idx]
        sw_f = np.array([cw[y] for y in yf])
        model = xgb.XGBClassifier(**params)
        model.fit(Xf, yf, sample_weight=sw_f, eval_set=[(Xv, yv)], verbose=False)
        scores.append(f1_score(yv, model.predict(Xv), average="macro"))
    return np.mean(scores)

study2.optimize(objective_fine, n_trials=30, show_progress_bar=True)

# ====== Train final model with best params ======
print("\n=== Training final model... ===")
best_params = study2.best_params
print(f"Best params: {best_params}")
print(f"Best CV-F1: {study2.best_value:.4f}")

final_params = {
    **best_params,
    "objective": "multi:softprob",
    "num_class": NC,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}

final_model = xgb.XGBClassifier(**final_params)
sw_all = np.array([cw[y] for y in ytr])
final_model.fit(Xtr, ytr, sample_weight=sw_all, eval_set=[(Xva, yva)], verbose=False)

final_p = final_model.predict(Xte)
final_acc = (final_p == yte).mean()
final_f1 = f1_score(yte, final_p, average="macro")

# ====== Report ======
from sklearn.metrics import classification_report

print(f"\n{'='*55}")
print(f"OPTUNA OPTIMIZATION RESULTS")
print(f"{'='*55}")
print(f"{'Model':<20} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"{'-'*40}")
print(f"{'XGBoost (baseline)':<20} {base_acc:>10.4f} {base_f1:>10.4f}")
print(f"{'XGBoost (Optuna)':<20} {final_acc:>10.4f} {final_f1:>10.4f}")
print(f"{'='*55}")

if final_acc > base_acc:
    print(f"\nIMPROVEMENT: +{final_acc-base_acc:.4f} accuracy, +{final_f1-base_f1:.4f} F1")
else:
    print(f"\nNo improvement (baseline already well-tuned)")

print(f"\n--- XGBoost Optuna Report ---")
print(classification_report(yte, final_p, target_names=cn, digits=3, zero_division=0))

# Save
joblib.dump(final_model, "models/xgboost_optuna.pkl")
print(f"\nTrial count: {len(study.trials) + len(study2.trials)}")
print("Model saved to models/xgboost_optuna.pkl")
print("DONE!")
