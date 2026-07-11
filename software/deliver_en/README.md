# AstroTransient

**Astronomical Transient Classification via Light Curve Analysis**

---

## English

AstroTransient identifies 14 classes of astronomical transients and variable sources from multi-band photometric light curves. The classifier is trained on PLAsTiCC (Photometric LSST Astronomical Time-series Classification Challenge) data and achieves approximately 82% accuracy on the test set.

**Supported classes**: Type Ia, Type II, and Type Ibc supernovae; superluminous supernovae; tidal disruption events; kilonovae; active galactic nuclei; eclipsing binaries; RR Lyrae and Mira variables; M-dwarfs; microlensing events; calcium-rich transients; and related subtypes.

---

### Getting Started

#### Prerequisites

- Python 3.10 or later

#### Installation & Launch

```bash
pip install -r requirements.txt
python launcher.py
```

The launcher automatically checks dependencies, prepares the environment, and starts the web interface. On first run, you will be prompted whether to download the PLAsTiCC demonstration dataset (~35 MB). This dataset enables the Demo and Benchmark tabs (browsing example light curves and evaluating accuracy). If you choose to skip the download, these two tabs will be unavailable, but you may still use the Upload CSV tab with the included sample file (`sample_lightcurve.csv`) or your own data.

A sample light curve file (`sample_lightcurve.csv`) is included in this directory for testing purposes.

#### Manual Start

```bash
python app.py
```

#### Command-Line Interface

```bash
python predict.py --demo          # Predict random sample objects
python predict.py --file data.csv # Classify custom light curve data
python predict.py --batch         # Evaluate accuracy on test set
```

#### Input Format

CSV files with the following columns:

| Column     | Description                          |
|------------|--------------------------------------|
| `mjd`      | Observation time (Modified Julian Date) |
| `flux`     | Brightness measurement               |
| `flux_err` | Flux uncertainty                     |
| `passband` | Filter wavelength in Angstrom        |

Optional columns: `redshift`, `hostgal_specz`, `hostgal_photoz`. A minimum of five valid rows is required.

---

### Technical Summary

The classifier extracts 47 statistical and photometric features from each light curve: amplitude, skewness, kurtosis, per-band statistics, color indices, autocorrelation, and host galaxy properties. Classification is performed using gradient-boosted decision trees (XGBoost) with class-weighted training. The model was validated on 6,040 training samples across 14 classes, achieving 81.9% overall accuracy and 76.0% macro-averaged F1 score.

---

### Project Structure

```
├── app.py                  # Web interface (Gradio)
├── launcher.py             # Cross-platform launcher
├── predict.py              # Command-line inference
├── requirements.txt        # Python dependencies
├── lightcurve_template.csv # CSV template for custom data
├── models/                 # Trained classifier
└── src/                    # Core library
```

---

### Authors

Built by Daisy Shang and Nnll_Temp (2026), with assistance from DeepSeek.

---

### License

MIT
