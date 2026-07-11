"""
AstroTransient CLI - Command-line inference tool
Usage:
  python predict.py --file my_lightcurve.csv     # Predict your own data
  python predict.py --demo                       # Random demo (PLAsTiCC)
  python predict.py --batch                      # Batch accuracy evaluation
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, joblib
from scipy import stats

from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES

# Load model
print("Loading model...")
model = joblib.load("models/xgboost_final.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
print(f"Model: XGBoost, {model.n_estimators} trees, {len(label_encoder.classes_)} classes")

# Feature extraction (must match training exactly)
def extract_features_from_row(row):
    try: mjd_all, flux_all, flux_err_all, pbs_all = get_lightcurve(row)
    except: return None
    mask = (flux_err_all > 0) & (~np.isnan(flux_all))
    if mask.sum() < 3: return None
    t, f, e = mjd_all[mask], flux_all[mask], flux_err_all[mask]; pbs_m = pbs_all[mask]
    feats = {}
    feats["n_points"]=float(len(f)); feats["duration"]=float(t[-1]-t[0])
    feats["flux_mean"]=float(np.mean(f)); feats["flux_std"]=float(np.std(f))
    feats["flux_median"]=float(np.median(f))
    feats["flux_min"]=float(np.min(f)); feats["flux_max"]=float(np.max(f))
    feats["flux_range"]=float(np.max(f)-np.min(f))
    feats["flux_skew"]=float(stats.skew(f)) if len(f)>2 else 0
    feats["flux_kurtosis"]=float(stats.kurtosis(f)) if len(f)>3 else 0
    feats["snr_mean"]=float(np.mean(np.abs(f)/(e+1e-8)))
    feats["snr_max"]=float(np.max(np.abs(f)/(e+1e-8)))
    sf=np.sort(f); feats["amplitude_90"]=float(sf[int(0.95*len(f))]-sf[int(0.05*len(f))])
    feats["beyond_1std"]=float(np.mean(np.abs(f-np.mean(f))>np.std(f)))
    diff=np.diff(f); feats["max_rise"]=float(np.max(diff)) if len(diff)>0 else 0
    feats["max_decay"]=float(np.min(diff)) if len(diff)>0 else 0
    feats["peak_position"]=float(np.argmax(np.abs(f))/max(len(f),1))
    ac=np.corrcoef(f[:-1],f[1:])[0,1] if len(f)>4 else np.nan
    feats["autocorr_lag1"]=float(ac) if not np.isnan(ac) else 0
    feats["iqr"]=float(np.percentile(f,75)-np.percentile(f,25))
    feats["mad"]=float(np.median(np.abs(f-np.median(f))))
    band_fluxes = {}
    for bn,(lo,hi) in {"u":(3000,4200),"g":(4200,5400),"r":(5400,6800),"i":(6800,8200),"z":(8200,9200),"y":(9200,10500)}.items():
        bm=(pbs_m>=lo)&(pbs_m<hi)
        if bm.sum()>=3:
            fb=f[bm]; eb=e[bm]; feats[f"{bn}_mean"]=float(np.mean(fb))
            feats[f"{bn}_std"]=float(np.std(fb)); feats[f"{bn}_snr"]=float(np.mean(np.abs(fb)/(eb+1e-8)))
            band_fluxes[bn]=float(np.mean(fb))
        else: feats[f"{bn}_mean"]=feats[f"{bn}_std"]=feats[f"{bn}_snr"]=0.0; band_fluxes[bn]=0.0
    for b1,b2 in [("u","g"),("g","r"),("r","i"),("i","z"),("z","y")]:
        feats[f"color_{b1}_{b2}"] = band_fluxes[b1] - band_fluxes[b2]
    feats["redshift"]=float(row.get("redshift",0)or 0)
    feats["hostgal_specz"]=float(row.get("hostgal_specz",0)or 0)
    feats["hostgal_photoz"]=float(row.get("hostgal_photoz",0)or 0)
    feats["n_passbands"]=float(len(set(pbs_m)))
    return feats

def predict(row, top_k=5):
    feat = extract_features_from_row(row)
    if feat is None: return None
    X = np.array([list(feat.values())], dtype=np.float32)
    probs = model.predict_proba(X)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_idx):
        cls_id = label_encoder.classes_[idx]
        name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        results.append({"rank": rank+1, "class_id": cls_id, "class_name": name, "probability": float(probs[idx])})
    return results

# ===== Main =====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AstroTransient Classifier")
    parser.add_argument("--file", type=str, help="Predict a custom CSV light curve file")
    parser.add_argument("--demo", action="store_true", help="Random demo (PLAsTiCC)")
    parser.add_argument("--batch", action="store_true", help="Batch accuracy evaluation")
    args = parser.parse_args()

    if args.file:
        df = pd.read_csv(args.file)
        req = ["mjd", "flux", "flux_err", "passband"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            print(f"ERROR: Missing columns: {missing}")
            sys.exit(1)
        row = {
            "times_wv": np.array([np.array([m, p]) for m, p in zip(df["mjd"], df["passband"])]),
            "lightcurve": np.array([np.array([f, e]) for f, e in zip(df["flux"], df["flux_err"])]),
            "redshift": float(df.get("redshift", [0]).iloc[0]) if "redshift" in df.columns else 0,
            "hostgal_specz": float(df.get("hostgal_specz", [0]).iloc[0]) if "hostgal_specz" in df.columns else 0,
            "hostgal_photoz": float(df.get("hostgal_photoz", [0]).iloc[0]) if "hostgal_photoz" in df.columns else 0,
        }
        result = predict(row, top_k=5)
        if result is None:
            print("Not enough valid data points.")
            sys.exit(1)
        print(f"\nInput: {len(df)} observations")
        print(f"Top-5 Predictions:")
        for r in result:
            bar = "#" * int(r["probability"] * 40)
            print(f"  {r['rank']}. {r['class_name'][:45]:45s} {r['probability']:.4f}  {bar}")
        sys.exit(0)

    # Load data for demo/batch
    print("Loading PLAsTiCC data...")
    train_df, _, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=1)

    if args.batch:
        from sklearn.metrics import accuracy_score, classification_report
        np.random.seed(42)
        n = min(200, len(train_df))
        idxs = np.random.choice(len(train_df), n, replace=False)
        preds, trues, cor = [], [], 0
        for idx in idxs:
            r = predict(train_df.iloc[idx], top_k=1)
            if r:
                preds.append(r[0]["class_id"]); trues.append(train_df.iloc[idx]["label"])
                if r[0]["class_id"] == train_df.iloc[idx]["label"]: cor += 1
        acc = accuracy_score(trues, preds)
        print(f"\nTested: {len(preds)} samples")
        print(f"Accuracy: {acc:.4f}")
        labels = sorted(set(trues+preds))
        cn = [CLASS_NAMES.get(l, f"Class {l}") for l in labels]
        print("\nClassification Report:")
        print(classification_report(trues, preds, target_names=cn, digits=3, zero_division=0))
        sys.exit(0)

    # Demo mode (default)
    np.random.seed()
    demo_ids = np.random.choice(len(train_df), 5, replace=False)
    correct = 0
    for i, idx in enumerate(demo_ids):
        row = train_df.iloc[idx]
        true_label = row["label"]
        true_name = CLASS_NAMES.get(true_label, f"Class {true_label}")
        result = predict(row, top_k=3)
        if result is None:
            print(f"\n[Sample {i+1}] Object {row['object_id']}: Not enough data")
            continue
        print(f"\n[Sample {i+1}] Object ID: {row['object_id']}")
        print(f"  True:  {true_name}")
        print(f"  Predictions:")
        for r in result:
            bar = "#" * int(r["probability"] * 30)
            tag = " <-- CORRECT" if r["class_id"] == true_label else ""
            print(f"    {r['rank']}. {r['class_name'][:45]:45s} {r['probability']:.3f}  {bar}{tag}")
        if result[0]["class_id"] == true_label: correct += 1
    print(f"\n{'='*50}")
    print(f"Demo accuracy: {correct}/{len(demo_ids)}")
    print(f"{'='*50}")
