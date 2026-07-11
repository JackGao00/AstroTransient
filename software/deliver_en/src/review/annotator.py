"""人工标注辅助工具 — 找出模型不确定的样本供人工复核"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def find_uncertain_samples(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    confidence_threshold: float = 0.6,
    max_samples: int = 100,
) -> pd.DataFrame:
    """
    找出模型预测置信度低的样本。

    返回 DataFrame, 包含: index, true_label, predicted, confidence
    """
    probs = model.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    # 置信度低于阈值的样本
    uncertain_mask = max_probs < confidence_threshold
    indices = np.where(uncertain_mask)[0]

    # 按置信度升序排列 (最不确定的排前面)
    sort_idx = np.argsort(max_probs[indices])
    indices = indices[sort_idx][:max_samples]

    return pd.DataFrame({
        "index": indices,
        "true_label": y_true[indices],
        "predicted": preds[indices],
        "confidence": max_probs[indices],
    }).sort_values("confidence")


def merge_reviewed_labels(
    original_labels: np.ndarray,
    reviewed_indices: np.ndarray,
    corrected_labels: np.ndarray,
) -> np.ndarray:
    """将人工修正后的标签合并回原始标签数组"""
    updated = original_labels.copy()
    updated[reviewed_indices] = corrected_labels
    return updated


def review_summary(
    original: np.ndarray,
    updated: np.ndarray,
) -> dict:
    """统计人工校正改动了多少标签"""
    changes = (original != updated).sum()
    return {
        "total_samples": len(original),
        "changed_labels": changes,
        "change_rate": changes / len(original),
    }
