"""Debug: inspect PLAsTiCC data format"""
import os, sys, numpy as np, pandas as pd

os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

from src.data.download import load_plasticc

train_df, _, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=1)

print(f"\nColumns: {list(train_df.columns)}")
print(f"Dtypes:\n{train_df.dtypes}")
print(f"\nFirst row:")
row = train_df.iloc[0]
for col in train_df.columns:
    val = row[col]
    if isinstance(val, np.ndarray):
        print(f"  {col}: shape={val.shape}, dtype={val.dtype}")
        print(f"    first 3: {val[:3]}")
    elif isinstance(val, list):
        print(f"  {col}: list, len={len(val)}, first elem type={type(val[0]) if val else 'empty'}")
        if val:
            print(f"    first 3: {val[:3]}")
    else:
        print(f"  {col}: {val} (type={type(val).__name__})")

# Try get_lightcurve
print("\n\nTrying get_lightcurve on first row...")
from src.data.download import get_lightcurve
try:
    mjd, flux, flux_err, pbs = get_lightcurve(row, passband=3)
    print(f"  r-band: {len(mjd)} points")
except Exception as e:
    print(f"  ERROR: {e}")
