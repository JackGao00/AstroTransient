"""
AstroTransient GUI - English | python app_en.py | http://localhost:7860
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, joblib, gradio as gr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from scipy import stats
from PIL import Image

from src.data.download import load_plasticc, get_lightcurve
from src.data.preprocess import CLASS_NAMES

# ===== Load model & data =====
model = joblib.load("models/xgboost_final.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
print("Loading PLAsTiCC data...")
train_df, _, _ = load_plasticc("data/raw/plasticc", load_test=False, max_train_files=1)

# ===== Feature extraction =====
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

def predict_row(row, top_k=5):
    feat=extract_features_from_row(row)
    if feat is None: return None
    X=np.array([list(feat.values())],dtype=np.float32)
    probs=model.predict_proba(X)[0]; top_idx=np.argsort(probs)[::-1][:top_k]
    return [(CLASS_NAMES.get(label_encoder.classes_[idx],f"Class {label_encoder.classes_[idx]}"),
             float(probs[idx]),label_encoder.classes_[idx]) for idx in top_idx]

# ===== Plot =====
def plot_lightcurve(mjd, flux, flux_err, title="Light Curve"):
    fig,ax=plt.subplots(figsize=(10,4))
    ax.errorbar(mjd,flux,yerr=flux_err,fmt='o',color='#1f77b4',markersize=4,alpha=0.7,capsize=2)
    ax.set_xlabel("MJD",fontsize=12); ax.set_ylabel("Flux",fontsize=12)
    ax.set_title(title,fontsize=14,fontweight='bold')
    ax.invert_yaxis(); ax.grid(True,alpha=0.3); plt.tight_layout()
    buf=BytesIO(); fig.savefig(buf,format='png',dpi=100,bbox_inches='tight'); plt.close(fig)
    buf.seek(0); return np.array(Image.open(buf))

# ===== Callbacks =====
def predict_from_csv(file):
    if file is None: return None, "Please upload a CSV file.", ""
    try: df=pd.read_csv(file.name)
    except Exception as e: return None, f"CSV Error: {e}", ""
    req=["mjd","flux","flux_err","passband"]
    missing=[c for c in req if c not in df.columns]
    if missing: return None, f"Missing columns: {missing}", ""
    row={"times_wv":np.array([np.array([m,p])for m,p in zip(df["mjd"],df["passband"])]),
         "lightcurve":np.array([np.array([f,e])for f,e in zip(df["flux"],df["flux_err"])]),
         "redshift":float(df.get("redshift",[0]).iloc[0])if"redshift"in df.columns else 0,
         "hostgal_specz":float(df.get("hostgal_specz",[0]).iloc[0])if"hostgal_specz"in df.columns else 0,
         "hostgal_photoz":float(df.get("hostgal_photoz",[0]).iloc[0])if"hostgal_photoz"in df.columns else 0}
    result=predict_row(row,top_k=5)
    if result is None: return None, "Not enough valid data (need >= 5 observations).", ""
    img=plot_lightcurve(df["mjd"].values,df["flux"].values,df["flux_err"].values,
                        title=f"Light Curve ({len(df)} observations)")
    lines=[f"Observations: {len(df)}",
           f"MJD range: [{df['mjd'].min():.1f}, {df['mjd'].max():.1f}]",
           f"Passbands: {sorted(df['passband'].unique())}","",
           "Top-5 Predictions:"]
    for name,prob,_ in result:
        bar="#"*int(prob*40)
        tag=" <-- BEST" if prob==result[0][1] else ""
        lines.append(f"  {name:<45s} {prob:.4f}  {bar}{tag}")
    status=f"Most likely: {result[0][0]} (Confidence: {result[0][1]*100:.1f}%)"
    return img,"\n".join(lines),status

def random_demo_fn():
    idx=np.random.randint(0,len(train_df)); row=train_df.iloc[idx]
    true_label=row["label"]; true_name=CLASS_NAMES.get(true_label,f"Class {true_label}")
    try: mjd,flux,flux_err,_=get_lightcurve(row)
    except: return None,"Error",""
    result=predict_row(row,top_k=5)
    if result is None: return None,"Not enough data",""
    img=plot_lightcurve(mjd,flux,flux_err,title=f"Object {row['object_id']} - True: {true_name}")
    lines=[f"Object ID: {row['object_id']}",
           f"True label: {true_name}","",
           "Model predictions:"]
    correct=result[0][2]==true_label
    for name,prob,cls_id in result:
        bar="#"*int(prob*40)
        tag=" <-- CORRECT!" if cls_id==true_label else ""
        lines.append(f"  {name:<45s} {prob:.4f}  {bar}{tag}")
    status="CORRECT!" if correct else "WRONG"
    return img,"\n".join(lines),f"Prediction: {status}"

def batch_test_fn(n_samples):
    from sklearn.metrics import accuracy_score,classification_report
    np.random.seed(42); n=min(int(n_samples),len(train_df))
    idxs=np.random.choice(len(train_df),n,replace=False)
    preds,trues,cor=[],[],0
    for idx in idxs:
        r=predict_row(train_df.iloc[idx],top_k=1)
        if r: preds.append(r[0][2]); trues.append(train_df.iloc[idx]["label"])
        if r and r[0][2]==train_df.iloc[idx]["label"]: cor+=1
    if not preds: return "No valid predictions."
    acc=accuracy_score(trues,preds)
    labels=sorted(set(trues+preds)); cn=[CLASS_NAMES.get(l,f"Class {l}")for l in labels]
    report=classification_report(trues,preds,target_names=cn,digits=3,zero_division=0)
    return f"Tested: {len(preds)}\nAccuracy: {acc*100:.1f}%\nCorrect: {cor}/{len(preds)}\n\nPer-class Report:\n{report}"

# ===== UI =====
with gr.Blocks(title="AstroTransient") as app:
    gr.Markdown("""
    # AstroTransient - AI Astronomical Transient Classifier

    XGBoost-based 14-class light curve classifier. Upload observation data and AI identifies the type.
    Supports supernovae, AGN, eclipsing binaries, TDEs, kilonovae, variable stars, and more.
    """)

    with gr.Tabs():
        with gr.TabItem("Upload CSV"):
            gr.Markdown("""
            ### Input Format
            CSV file with columns: `mjd`, `flux`, `flux_err`, `passband`
            Optional: `redshift`, `hostgal_specz`, `hostgal_photoz`

            **At least 5 valid rows**
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                    with gr.Row():
                        btn_predict = gr.Button("Predict", variant="primary", size="lg")
                        btn_template = gr.DownloadButton("Download CSV Template",
                                                         value="lightcurve_template.csv",
                                                         variant="secondary", size="lg")
                with gr.Column(scale=2):
                    upload_plot = gr.Image(label="Light Curve")
                    upload_status = gr.Textbox(label="Result")
            upload_text = gr.Textbox(label="Details", lines=8)

        with gr.TabItem("Demo"):
            gr.Markdown("### Random PLAsTiCC Object with Model Predictions")
            btn_demo = gr.Button("Random Sample", variant="primary", size="lg")
            with gr.Row():
                demo_plot = gr.Image(label="Light Curve")
                demo_text = gr.Textbox(label="Predictions", lines=10)
            demo_status = gr.Textbox(label="Result")

        with gr.TabItem("Benchmark"):
            gr.Markdown("### Evaluate Model Accuracy on PLAsTiCC Test Set")
            n_slider = gr.Slider(50, 500, value=100, step=50, label="Number of Samples")
            btn_bench = gr.Button("Run Benchmark", variant="primary")
            bench_output = gr.Textbox(label="Report", lines=20)

    gr.Markdown("---\n*Model: XGBoost (500 trees) | Features: 47 handcrafted | Classes: 14 | Accuracy: ~82%*")

    # Events
    btn_predict.click(fn=predict_from_csv, inputs=[file_input],
                      outputs=[upload_plot, upload_text, upload_status])
    btn_demo.click(fn=random_demo_fn, inputs=[], outputs=[demo_plot, demo_text, demo_status])
    btn_bench.click(fn=batch_test_fn, inputs=[n_slider], outputs=[bench_output])

if __name__ == "__main__":
    import argparse; parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    print(f"AstroTransient starting on http://localhost:{args.port}")
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
               inbrowser=True, theme=gr.themes.Soft())
