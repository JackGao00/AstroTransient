"""PLAsTiCC 数据集下载与加载 (Parquet 格式)"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download


def download_plasticc(save_dir: str = "data/raw/plasticc") -> str:
    """Download PLAsTiCC dataset"""
    os.makedirs(save_dir, exist_ok=True)

    # Suppress HF warnings and progress bars
    import logging
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    snapshot_download(
        repo_id="BrachioLab/PLAsTiCC",
        repo_type="dataset",
        local_dir=save_dir,
        allow_patterns=["data/train-*", "data/validation-*"],
        token=False,
    )
    return save_dir


def load_plasticc(data_dir: str = "data/raw/plasticc",
                  load_test: bool = False,
                  max_train_files: int = None) -> tuple:
    """
    加载 PLAsTiCC 数据集.
    - load_test=False: 只加载 train + validation (避免内存爆炸)
    - max_train_files: 限制加载的 train parquet 文件数量
    """
    data_path = os.path.join(data_dir, "data")

    train_files = sorted(Path(data_path).glob("train-*.parquet"))
    val_files = sorted(Path(data_path).glob("validation-*.parquet"))
    test_files = sorted(Path(data_path).glob("test-*.parquet"))

    if max_train_files and max_train_files < len(train_files):
        train_files = train_files[:max_train_files]

    print(f"Loading {len(train_files)} train files...")
    train_dfs = []
    for f in train_files:
        train_dfs.append(pd.read_parquet(f))
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else None

    if val_files:
        print(f"Loading {len(val_files)} validation files...")
        val_dfs = [pd.read_parquet(f) for f in val_files]
        val_df = pd.concat(val_dfs, ignore_index=True)
    else:
        val_df = None

    test_df = None
    if load_test and test_files:
        print(f"Loading {len(test_files)} test files (warning: large)...")
        test_dfs = [pd.read_parquet(f) for f in test_files]
        test_df = pd.concat(test_dfs, ignore_index=True)

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if df is not None:
            print(f"{name}: {len(df):,} objects")

    return train_df, val_df, test_df


def get_lightcurve(obj_row, passband: int = None) -> tuple:
    """
    从 DataFrame 的一行中提取光变曲线.
    PLAsTiCC parquet 格式:
      - times_wv: 1D object array, 每个元素是 [mjd, passband]
      - lightcurve: 1D object array, 每个元素是 [flux, flux_err]
    返回: (mjd, flux, flux_err, passbands)
    """
    times_wv = obj_row["times_wv"]
    lc = obj_row["lightcurve"]

    # 转换为 2D numpy 数组
    tw = np.array([list(x) for x in times_wv])  # (N, 2)
    lc_arr = np.array([list(x) for x in lc])     # (N, 2)

    mjd = tw[:, 0]
    pbs = tw[:, 1].astype(int)
    flux = lc_arr[:, 0]
    flux_err = lc_arr[:, 1]

    if passband is not None:
        mask = pbs == passband
        mjd, flux, flux_err, pbs = mjd[mask], flux[mask], flux_err[mask], pbs[mask]

    sort_idx = np.argsort(mjd)
    return mjd[sort_idx], flux[sort_idx], flux_err[sort_idx], pbs[sort_idx]
