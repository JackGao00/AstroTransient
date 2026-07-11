"""
AstroTransient 推理脚本

=== 输入格式 ===
CSV 文件, 包含列:
  mjd        - 观测时间 (Modified Julian Date, 浮点数)
  flux       - 流量/亮度 (浮点数, 任意单位)
  flux_err   - 流量误差 (浮点数)
  passband   - 波段波长, 单位埃 (浮点数, 如 6222 = r-band)

可选列:
  redshift, hostgal_specz, hostgal_photoz  - 宿主星系信息

=== 输出格式 ===
  JSON 或文本: Top-K 预测结果, 包含类别名/置信度/排名

=== 用法 ===
  python predict.py --file my_lightcurve.csv     # 预测你自己的光变曲线
  python predict.py --demo                       # 随机演示 (PLAsTiCC 数据)
  python predict.py --object-id 12345            # 预测指定 PLAsTiCC 天体
  python predict.py --batch                      # 批量预测 + 准确率报告

=== 输入 CSV 示例 ===
  mjd,flux,flux_err,passband,redshift
  59581.36,8.07,21.50,6222,0.05
  59582.35,-0.65,26.62,6222,0.05
  59584.17,23.21,20.62,4827,0.05
  ...(至少 5 行)...
"""
import sys, os
os.chdir(r"D:\AITools\AstroTransient")
sys.path.insert(0, r"D:\AITools\AstroTransient")

import numpy as np
import pandas as pd
import joblib
from scipy import stats

from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES

# ============================================================
# 1. 加载模型
# ============================================================
print("Loading model...")
# 加载模型和标签编码器
model = joblib.load("models/xgboost_final.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
print(f"Model: XGBoost, {model.n_estimators} trees, {len(label_encoder.classes_)} classes")
# ============================================================
# 2. 特征提取 (与训练时完全一致)
# ============================================================
def extract_features_from_row(row):
    """从一行 PLAsTiCC 数据中提取 46 个特征"""
    try:
        mjd_all, flux_all, flux_err_all, pbs_all = get_lightcurve(row)
    except:
        return None

    mask = (flux_err_all > 0) & (~np.isnan(flux_all))
    if mask.sum() < 3:
        return None

    t, f, e = mjd_all[mask], flux_all[mask], flux_err_all[mask]
    pbs_m = pbs_all[mask]
    feats = {}

    feats["n_points"] = float(len(f))
    feats["duration"] = float(t[-1] - t[0])
    feats["flux_mean"] = float(np.mean(f))
    feats["flux_std"] = float(np.std(f))
    feats["flux_median"] = float(np.median(f))
    feats["flux_min"] = float(np.min(f))
    feats["flux_max"] = float(np.max(f))
    feats["flux_range"] = float(np.max(f) - np.min(f))
    feats["flux_skew"] = float(stats.skew(f)) if len(f) > 2 else 0
    feats["flux_kurtosis"] = float(stats.kurtosis(f)) if len(f) > 3 else 0
    feats["snr_mean"] = float(np.mean(np.abs(f) / (e + 1e-8)))
    feats["snr_max"] = float(np.max(np.abs(f) / (e + 1e-8)))
    sf = np.sort(f)
    feats["amplitude_90"] = float(sf[int(0.95*len(f))] - sf[int(0.05*len(f))])
    feats["beyond_1std"] = float(np.mean(np.abs(f - np.mean(f)) > np.std(f)))
    diff = np.diff(f)
    feats["max_rise"] = float(np.max(diff)) if len(diff) > 0 else 0
    feats["max_decay"] = float(np.min(diff)) if len(diff) > 0 else 0
    feats["peak_position"] = float(np.argmax(np.abs(f)) / max(len(f), 1))
    if len(f) > 4:
        ac = np.corrcoef(f[:-1], f[1:])[0, 1]
        feats["autocorr_lag1"] = float(ac) if not np.isnan(ac) else 0
    else:
        feats["autocorr_lag1"] = 0
    feats["iqr"] = float(np.percentile(f, 75) - np.percentile(f, 25))
    feats["mad"] = float(np.median(np.abs(f - np.median(f))))

    band_ranges = {"u": (3000, 4200), "g": (4200, 5400), "r": (5400, 6800),
                   "i": (6800, 8200), "z": (8200, 9200), "y": (9200, 10500)}
    band_fluxes = {}
    for bn, (lo, hi) in band_ranges.items():
        bm = (pbs_m >= lo) & (pbs_m < hi)
        if bm.sum() >= 3:
            fb = f[bm]; eb = e[bm]
            feats[f"{bn}_mean"] = float(np.mean(fb))
            feats[f"{bn}_std"] = float(np.std(fb))
            feats[f"{bn}_snr"] = float(np.mean(np.abs(fb) / (eb + 1e-8)))
            band_fluxes[bn] = float(np.mean(fb))
        else:
            feats[f"{bn}_mean"] = feats[f"{bn}_std"] = feats[f"{bn}_snr"] = 0.0
            band_fluxes[bn] = 0.0

    for b1, b2 in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("z", "y")]:
        feats[f"color_{b1}_{b2}"] = band_fluxes[b1] - band_fluxes[b2]

    feats["redshift"] = float(row.get("redshift", 0) or 0)
    feats["hostgal_specz"] = float(row.get("hostgal_specz", 0) or 0)
    feats["hostgal_photoz"] = float(row.get("hostgal_photoz", 0) or 0)
    feats["n_passbands"] = float(len(set(pbs_m)))

    return feats


def predict(row, top_k=5):
    """预测单个天体, 返回 Top-K 结果"""
    feat = extract_features_from_row(row)
    if feat is None:
        return None

    X = np.array([list(feat.values())], dtype=np.float32)
    probs = model.predict_proba(X)[0]

    top_idx = np.argsort(probs)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_idx):
        encoded_label = idx  # model's internal label (0..13)
        original_cls_id = label_encoder.classes_[encoded_label]  # PLAsTiCC ID
        name = CLASS_NAMES.get(original_cls_id, f"Class {original_cls_id}")
        results.append({
            "rank": rank + 1,
            "encoded_label": encoded_label,
            "class_id": original_cls_id,
            "class_name": name,
            "probability": float(probs[idx]),
        })
    return results


def predict_batch(df, n_samples=100):
    """批量预测, 输出准确率报告"""
    from sklearn.metrics import accuracy_score, classification_report

    np.random.seed(42)
    sample_idx = np.random.choice(len(df), min(n_samples, len(df)), replace=False)

    preds, trues = [], []
    for idx in sample_idx:
        row = df.iloc[idx]
        result = predict(row, top_k=1)
        if result:
            preds.append(result[0]["encoded_label"])
            trues.append(label_encoder.transform([row["label"]])[0])  # encode to 0..13

    if len(preds) == 0:
        print("No valid predictions!")
        return

    acc = accuracy_score(trues, preds)
    print(f"\nBatch Prediction: {len(preds)} samples")
    print(f"Accuracy: {acc:.4f}")

    cn = [CLASS_NAMES.get(label_encoder.classes_[i], f"cls_{i}") for i in range(len(label_encoder.classes_))]
    print(f"\nClassification Report:")
    print(classification_report(trues, preds, target_names=cn, digits=3, zero_division=0))


# ============================================================
# 3. 主程序
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AstroTransient 天体分类器")
    parser.add_argument("--object-id", type=int, help="预测指定天体 ID")
    parser.add_argument("--batch", action="store_true", help="批量预测模式")
    parser.add_argument("--demo", action="store_true", help="演示模式: 随机展示预测结果")
    parser.add_argument("--file", type=str, help="预测你自己的 CSV 光变曲线文件")
    args = parser.parse_args()

    # ---- 用户自定义文件输入 ----
    if args.file:
        print(f"Loading: {args.file}")
        df = pd.read_csv(args.file)

        # 验证必要列
        required = ["mjd", "flux", "flux_err", "passband"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"ERROR: Missing columns: {missing}")
            print(f"Required: {required}")
            print(f"Optional: redshift, hostgal_specz, hostgal_photoz")
            sys.exit(1)

        # 构造 light curve 数组
        tw = np.column_stack([df["mjd"].values, df["passband"].values])
        lc = np.column_stack([df["flux"].values, df["flux_err"].values])

        # 构造 row (兼容特征提取器)
        row = {
            "times_wv": np.array([np.array([m, p]) for m, p in zip(df["mjd"], df["passband"])]),
            "lightcurve": np.array([np.array([f, e]) for f, e in zip(df["flux"], df["flux_err"])]),
            "redshift": df.get("redshift", [0])[0] if "redshift" in df.columns else 0,
            "hostgal_specz": df.get("hostgal_specz", [0])[0] if "hostgal_specz" in df.columns else 0,
            "hostgal_photoz": df.get("hostgal_photoz", [0])[0] if "hostgal_photoz" in df.columns else 0,
        }

        result = predict(row, top_k=5)

        if result is None:
            print("ERROR: Not enough valid data points (need >= 5)")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*60}")
        print(f"Input: {len(df)} observations, MJD range [{df['mjd'].min():.1f}, {df['mjd'].max():.1f}]")
        print(f"Passbands: {sorted(df['passband'].unique())}")
        print(f"\nPredicted class (Top-5):")
        for r in result:
            bar = "#" * int(r["probability"] * 40)
            print(f"  {r['rank']}. {r['class_name'][:50]:50s} {r['probability']:.4f}  {bar}")

        print(f"\nMost likely: {result[0]['class_name']}")
        print(f"Confidence:  {result[0]['probability']*100:.1f}%")
        print(f"{'='*60}")
        sys.exit(0)

    # ---- PLAsTiCC 内置模式 ----
    print("Loading PLAsTiCC data...")
    train_df, _, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=1)

    if args.batch:
        predict_batch(train_df, n_samples=200)
        sys.exit(0)

    if args.object_id:
        obj = train_df[train_df["object_id"] == args.object_id]
        if len(obj) == 0:
            print(f"Object {args.object_id} not found!")
            sys.exit(1)
        row = obj.iloc[0]
    else:
        # Demo: pick random objects and show predictions
        print("\n" + "=" * 60)
        print("ASTROTRANSIENT DEMO - Random Light Curve Predictions")
        print("=" * 60)

        np.random.seed(int(np.random.random() * 10000))
        demo_ids = np.random.choice(len(train_df), 5, replace=False)

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
                print(f"    {r['rank']}. {r['class_name'][:45]:45s} {r['probability']:.3f} {bar}{tag}")

        # Print summary accuracy
        correct = 0
        total_valid = 0
        for idx in demo_ids:
            row = train_df.iloc[idx]
            result = predict(row, top_k=1)
            if result:
                total_valid += 1
                if result[0]["class_id"] == row["label"]:
                    correct += 1

        print(f"\n{'='*60}")
        print(f"Demo accuracy: {correct}/{total_valid} ({correct/total_valid*100:.0f}%)")
        print(f"Model: XGBoost | Expected accuracy: ~82% on test set")
        print(f"{'='*60}")
