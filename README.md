# AstroTransient

**Astronomical Transient Classification via Light Curve Analysis**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A 14-class astronomical transient and variable source classifier based on gradient-boosted decision trees (XGBoost). Trained on PLAsTiCC (Photometric LSST Astronomical Time-series Classification Challenge) data, achieving ~82% accuracy on the test set.

---

## Repository Structure

```
├── software/           # Complete application source code
│   ├── app.py          # Gradio web GUI (bilingual EN/ZH)
│   ├── launcher.py     # Cross-platform one-click launcher
│   ├── predict.py      # CLI inference tool
│   ├── deliver/        # Distribution package (bilingual)
│   ├── deliver_en/     # Distribution package (English only)
│   ├── models/         # All trained model files
│   ├── src/            # Core library
│   ├── scripts/        # Training scripts (development history)
│   └── notebooks/      # Jupyter tutorials
│
├── docs/               # Project documentation
│   ├── development_log_en.md   # Full development log (EN)
│   ├── development_log_zh.md   # Full development log (ZH)
│   ├── project_report_en.md    # Research report (EN)
│   └── project_report_zh.md    # Research report (ZH)
│
└── README.md           # This file
```

## Quick Start

```bash
cd software
pip install -r requirements.txt
python launcher.py
```

The launcher handles dependency installation, dataset download (optional ~35 MB), and server startup automatically.

## Supported Classes

Type Ia, II, Ibc supernovae; superluminous supernovae; tidal disruption events; kilonovae; active galactic nuclei; eclipsing binaries; RR Lyrae and Mira variables; M-dwarfs; microlensing events; calcium-rich transients; and related subtypes.

## Technical Summary

- **Features**: 47 statistical and photometric features (global stats, per-band, color indices, host galaxy)
- **Model**: XGBoost (500 trees, max_depth=10)
- **Training**: 6,040 labeled objects, class-weighted loss
- **Performance**: 81.9% accuracy, 76.0% macro-averaged F1 on held-out test set (1,295 objects, 14 classes)

## Authors

**Daisy Shang, NullTemp** — 2026

Developed with assistance from DeepSeek.

## License

MIT
