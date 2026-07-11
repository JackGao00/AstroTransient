"""光变曲线预处理: 归一化, 序列切片"""

import numpy as np
from typing import Tuple


def normalize_flux(flux: np.ndarray, flux_err: np.ndarray) -> np.ndarray:
    """对流量值做 Z-score 归一化"""
    mask = flux_err > 0
    if mask.sum() < 2:
        return flux
    median = np.median(flux[mask])
    std = np.std(flux[mask])
    if std < 1e-8:
        return flux
    return (flux - median) / std


def make_fixed_length_sequence(
    mjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    max_points: int = 100,
) -> np.ndarray:
    """
    将光变曲线转换为固定长度序列 (max_points, 3)
    3 个通道: [相对时间, 归一化流量, 流量误差]
    """
    result = np.zeros((max_points, 3))
    n = min(len(mjd), max_points)

    result[:n, 0] = mjd[:n] - mjd[0]  # 相对时间 (天)
    result[:n, 1] = normalize_flux(flux[:n], flux_err[:n])
    result[:n, 2] = flux_err[:n]

    return result


CLASS_NAMES = {
    6:  "microlens-single",
    15: "tidal disruption event (TDE)",
    16: "eclipsing binary (EB)",
    42: "type II supernova (SNII)",
    52: "peculiar SNIa (SNIax)",
    53: "Mira variable",
    62: "type Ibc supernova (SNIbc)",
    64: "kilonova (KN)",
    65: "M-dwarf",
    67: "peculiar SNIa (SNIa-91bg)",
    88: "active galactic nuclei (AGN)",
    90: "type Ia supernova (SNIa)",
    92: "RR-Lyrae (RRL)",
    95: "superluminous supernova (SLSN-I)",
    991: "microlens-binary",
    992: "ILOT",
    993: "calcium-rich transient (CaRT)",
    994: "pair instability supernova (PISN)",
    995: "microlens-string",
}
