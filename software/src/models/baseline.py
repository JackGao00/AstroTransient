"""XGBoost 基线模型"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Tuple


def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int = 18,
) -> xgb.XGBClassifier:
    """训练 XGBoost 基线分类器"""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """评估模型, 返回指标字典"""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
    }
