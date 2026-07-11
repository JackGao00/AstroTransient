"""光变曲线统计特征提取"""

import numpy as np
from scipy import stats
from typing import Dict


def extract_features(
    mjd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray
) -> Dict[str, float]:
    """
    从光变曲线提取 20 个统计特征。
    """
    mask = (flux_err > 0) & (~np.isnan(flux))
    if mask.sum() < 3:
        return {f"feat_{i}": 0.0 for i in range(20)}

    t = mjd[mask]
    f = flux[mask]
    e = flux_err[mask]

    features = {
        "duration": float(t[-1] - t[0]),
        "n_points": float(len(f)),
        "flux_mean": float(np.mean(f)),
        "flux_std": float(np.std(f)),
        "flux_median": float(np.median(f)),
        "flux_min": float(np.min(f)),
        "flux_max": float(np.max(f)),
        "flux_range": float(np.max(f) - np.min(f)),
        "flux_skew": float(stats.skew(f)) if len(f) > 2 else 0.0,
        "flux_kurtosis": float(stats.kurtosis(f)) if len(f) > 2 else 0.0,
    }

    sorted_f = np.sort(f)
    p5 = sorted_f[int(0.05 * len(f))]
    p95 = sorted_f[int(0.95 * len(f))]
    features["amplitude_90"] = float(p95 - p5)
    features["beyond_1std"] = float(np.mean(np.abs(f - np.mean(f)) > np.std(f)))

    flux_diff = np.diff(f)
    features["max_rise"] = float(np.max(flux_diff)) if len(flux_diff) > 0 else 0.0
    features["max_decay"] = float(np.min(flux_diff)) if len(flux_diff) > 0 else 0.0

    features["snr_mean"] = float(np.mean(np.abs(f) / (e + 1e-8)))
    features["snr_max"] = float(np.max(np.abs(f) / (e + 1e-8)))

    if len(f) > 4:
        ac = np.corrcoef(f[:-1], f[1:])[0, 1]
        features["autocorr_lag1"] = float(ac) if not np.isnan(ac) else 0.0
    else:
        features["autocorr_lag1"] = 0.0

    features["peak_position"] = float(np.argmax(f) / max(len(f), 1))
    features["missing_ratio"] = float(1.0 - mask.sum() / len(mjd))

    return features
