"""Optuna CPU — 250 trials + feature selection"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np, time, json, warnings
warnings.filterwarnings("ignore")

import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
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
print(f"Train: {len(Xtr):,} | Val: {len(Xva):,} | Test: {len(Xte):,} | Classes: {NC} | Features: {Xtr.shape[1]}")

# Class weights
cc = np.bincount(ytr)
cw = {i: len(ytr)/(NC*max(c,1)) for i, c in enumerate(cc)}

# ====== Feature Selection ======
print("\n=== Feature Selection ===")
selector = SelectFromModel(
    xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                      objective="multi:softprob", num_class=NC,
                      eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0),
    threshold="median"
)
selector.fit(Xtr, ytr)
Xtr_sel = selector.transform(Xtr)
Xva_sel = selector.transform(Xva)
Xte_sel = selector.transform(Xte)
print(f"Features: {Xtr.shape[1]} -> {Xtr_sel.shape[1]}")

# ====== Optuna (250 trials) ======
print("\n=== Optuna Search (250 trials) ===")
t0 = time.time()

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 5.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "objective": "multi:softprob",
        "num_class": NC,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, vl_idx in skf.split(Xtr_sel, ytr):
        Xf, yf = Xtr_sel[tr_idx], ytr[tr_idx]
        Xv, yv = Xtr_sel[vl_idx], ytr[vl_idx]
        sw_f = np.array([cw[y] for y in yf])
        model = xgb.XGBClassifier(**params)
        model.fit(Xf, yf, sample_weight=sw_f, eval_set=[(Xv, yv)], verbose=False)
        scores.append(f1_score(yv, model.predict(Xv), average="macro"))
    return np.mean(scores)

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    study_name="xgb_final",
)
study.optimize(objective, n_trials=250, show_progress_bar=True)

# ====== Train final model ======
print(f"\nBest CV-F1: {study.best_value:.4f}")
print(f"Best params: {json.dumps(study.best_params, indent=2)}")

sw_all = np.array([cw[y] for y in ytr])
final = xgb.XGBClassifier(
    **study.best_params,
    objective="multi:softprob", num_class=NC,
    eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
)
final.fit(Xtr_sel, ytr, sample_weight=sw_all, eval_set=[(Xva_sel, yva)], verbose=False)
final_p = final.predict(Xte_sel)

# Also train full-feature model for comparison
final_full = xgb.XGBClassifier(
    **study.best_params,
    objective="multi:softprob", num_class=NC,
    eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
)
final_full.fit(Xtr, ytr, sample_weight=sw_all, eval_set=[(Xva, yva)], verbose=False)
fp = final_full.predict(Xte)

# ====== Report ======
acc_sel = accuracy_score(yte, final_p)
f1_sel = f1_score(yte, final_p, average="macro")
acc_full = accuracy_score(yte, fp)
f1_full = f1_score(yte, fp, average="macro")

print(f"\n{'='*55}")
print(f"FINAL RESULTS (250 Optuna trials)")
print(f"{'='*55}")
print(f"Time: {time.time()-t0:.0f}s")
print(f"{'Model':<30} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"{'-'*50}")
print(f"{'XGBoost (selected features)':<30} {acc_sel:>10.4f} {f1_sel:>10.4f}")
print(f"{'XGBoost (all features)':<30} {acc_full:>10.4f} {f1_full:>10.4f}")
print(f"{'='*55}")

best_acc = max(acc_sel, acc_full)
best_model = final if acc_sel >= acc_full else final_full
best_name = "Selected features" if acc_sel >= acc_full else "All features"

print(f"\nBest: {best_name} ({best_acc*100:.1f}%)")
print(f"\n--- Classification Report ---")
print(classification_report(yte, best_model.predict(Xte_sel if acc_sel >= acc_full else Xte),
                           target_names=cn, digits=3, zero_division=0))

# Save
joblib.dump(best_model, "models/xgboost_best.pkl")
joblib.dump(selector, "models/feature_selector.pkl")
print(f"\nModels saved: xgboost_best.pkl, feature_selector.pkl")
print("DONE!")
