"""可视化工具"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_light_curve(
    mjd: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    title: str = "",
    color: str = "#1f77b4",
    ax: Optional[plt.Axes] = None,
):
    """绘制单条光变曲线"""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if flux_err is not None:
        ax.errorbar(mjd, flux, yerr=flux_err, fmt="o", color=color,
                     markersize=3, alpha=0.7, capsize=0)
    else:
        ax.plot(mjd, flux, "o-", color=color, markersize=3, alpha=0.7)

    ax.set_xlabel("MJD (Modified Julian Date)")
    ax.set_ylabel("Flux")
    if title:
        ax.set_title(title)
    ax.invert_yaxis()  # 天文惯例: 亮度越高 → 星等越小
    return ax


def plot_class_examples(
    class_names: dict,
    get_lightcurve_fn,
    n_cols: int = 4,
    figsize: tuple = (16, 12),
):
    """绘制每个类别的示例光变曲线"""
    classes = list(class_names.keys())
    n_rows = (len(classes) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        ax = axes[i]
        mjd, flux, err = get_lightcurve_fn(cls)
        plot_light_curve(mjd, flux, err, title=f"{cls}: {class_names[cls]}", ax=ax)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    figsize: tuple = (12, 10),
):
    """绘制混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]:.2f}" if cm[i, j] < 1 else f"{cm[i, j]:.0f}",
                    ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig
