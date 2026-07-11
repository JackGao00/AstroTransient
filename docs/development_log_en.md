# AstroTransient — Development Log

**Daisy Shang, NullTemp · 2026**

---

## Phase 1: Project Conception & Research (Day 1)

### 1.1 Initial Idea

The project originated from an interest in applying machine learning to time-domain astronomy. The core question was straightforward: given a light curve (a sequence of brightness measurements over time), can we automatically identify what type of astronomical object produced it?

After surveying the landscape, we identified PLAsTiCC (Photometric LSST Astronomical Time-series Classification Challenge) as the ideal dataset. It contains simulated light curves for 18 classes of astronomical transients and variable sources, designed to mimic what the Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST) will observe. The dataset is publicly available and comes pre-labeled, making it well-suited for a supervised learning approach.

### 1.2 Technology Choices

We evaluated several approaches:

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| CNN on raw light curve images | Proven in literature (BTSbot, SCONE) | Needs large GPU and more data | Deferred |
| LSTM on raw sequences | Temporal modeling | Requires large training sets | Tested, underperformed |
| Feature extraction + XGBoost | Fast, interpretable, strong on tabular data | Requires domain knowledge for features | **Chosen as primary** |

The feature extraction route was selected because handcrafted features encode astronomical domain knowledge that a neural network would need millions of examples to learn from scratch. With only ~7,000 training samples available, gradient-boosted trees offered the best trade-off.

### 1.3 Tools

- Python 3.11, XGBoost, scikit-learn, PyTorch, Gradio
- PLAsTiCC dataset via public mirror

---

## Phase 2: Data Pipeline Construction (Day 1–2)

### 2.1 Dataset Acquisition

The PLAsTiCC dataset was downloaded in Parquet format. Initial attempts to load all 85 test partition files caused a memory overflow. The fix was to load only the training and validation partitions (one file each, ~7,000 + ~1,500 objects).

### 2.2 Data Format Challenges

The Parquet files use a nested object array format:

- `times_wv`: 1D array where each element is a 2-element array `[MJD, passband_wavelength]`
- `lightcurve`: 1D array where each element is a 2-element array `[flux, flux_err]`

This required converting object arrays to 2D numeric arrays before processing. Initial code assumed CSV format with flat columns; this was the first major debugging hurdle.

Additional issues encountered:
- Passband values are wavelengths in Angstroms (e.g., 6222 for r-band), not 0–5 indices as initially assumed
- Loading all partitions in one call exhausted system memory (~12 GB total)

### 2.3 Feature Engineering

Feature design evolved through three iterations:

**V1 (20 features)** — Basic statistics only:
- Duration, number of observations, mean/std/median/min/max/range of flux
- Skewness, kurtosis, signal-to-noise ratio (mean and max)
- Amplitude (P5–P95), beyond-1σ fraction
- Maximum rise/decay rate, peak position, autocorrelation

**V2 (41 features)** — Added per-band and color information:
- Per-band (u, g, r, i, z, y) mean, standard deviation, and SNR (18 features)
- Color indices: u−g, g−r, r−i, i−z, z−y (5 features)
- These features capture the spectral energy distribution, which differs significantly between object classes

**V3 (47 features)** — Added robust statistics and host galaxy context:
- Interquartile range (IQR), median absolute deviation (MAD)
- Redshift, host galaxy spectroscopic and photometric redshift
- Number of unique passbands observed

The final 47-feature set was used for all production models.

---

## Phase 3: Model Training (Day 2–3)

### 3.1 Baseline (V1)

- **Model**: XGBoost with default parameters
- **Training set**: 3,080 objects (200 per class, r-band only)
- **Features**: 20 statistical features
- **Result**: 54.3% accuracy, 51.6% macro F1

The low score was expected — 20 simple features and limited training data are insufficient for 14-class discrimination.

### 3.2 Improved Baseline (V2)

- **Training set**: 5,910 objects (expanded sampling, all passbands)
- **Features**: 41 features
- **Result**: 68.6% accuracy, 59.0% macro F1 (+14.3 percentage points)

The jump came from two changes: more data and per-band color features, which are highly discriminative for distinguishing supernova types and variable stars.

### 3.3 Optimized Model (V3)

- **Training set**: 7,335 objects (train + validation combined)
- **Features**: 47 features with missing-value handling
- **Model tuning**: 3-fold cross-validation over parameter grid, class-weighted loss
- **Result**: 82.8% accuracy, 76.8% macro F1

Key optimization decisions:
- `max_depth=10, learning_rate=0.02, n_estimators=500`
- Class weighting to handle imbalance (e.g., Mira variables: only 33 samples; Type Ia SNe: 2,544 samples)
- L2 regularization (`reg_lambda=1.5`) to prevent overfitting

### 3.4 Hyperparameter Search (Optuna)

An automated search over 11 hyperparameters was conducted with 80 trials using 3-fold cross-validation. An AI assistant was used to help design the search strategy and suggest appropriate parameter ranges. The search confirmed that the manually tuned parameters were near-optimal, yielding 82.2% accuracy (vs. 82.8% manual). The marginal difference suggests the model is near the performance ceiling for this feature set.

---

## Phase 4: Deep Learning Experiments (Day 3–4)

### 4.1 Motivation

With an RTX 4060 Ti available, we wanted to determine whether deep learning could outperform the feature-based approach. Five architectures were tested:

| Model | Input | Accuracy | Notes |
|-------|-------|----------|-------|
| BiLSTM + Attention | Raw light curve (200×4) | 27.0% | 87 epochs, GPU; failed to converge |
| Transformer Encoder | Raw light curve (200×5) | 26.0% | 86 epochs, GPU; similar failure |
| 2D Heatmap CNN | Time×Passband image (48×6×3) | 33.5% | 102 epochs, GPU; marginal improvement |
| MLP | 47 handcrafted features | 69.2% | 82 epochs, GPU; decent but below XGBoost |
| Contrastive pretraining (SimCLR) | Raw light curve → learned embeddings | 34.4% | 150 epochs, 15s; embeddings not discriminative enough |

### 4.2 Analysis

All deep learning models trained on raw light curves severely underperformed. The primary cause is data scarcity: ~4,000 training sequences are insufficient for a deep network to learn meaningful temporal representations from noisy, heterogeneous light curves. The handcrafted features implicitly encode decades of astronomical domain knowledge (e.g., color indices as temperature proxies, amplitude ratios as explosion energy indicators).

The contrastive learning experiment confirmed this diagnosis. While the SimCLR encoder converged quickly (loss dropped from 3.2 to 0.12 in 15 seconds on GPU), the learned embeddings achieved only 34.4% accuracy when used for classification. With millions of samples (as available from the Zwicky Transient Facility), this approach would likely become competitive.

### 4.3 GPU Utility Assessment

GPU acceleration was tested with LightGBM's GPU backend (`device='gpu'`). For our dataset size (~8,000 samples × 47 features), GPU training was actually slower than CPU (7.3s vs. 1.1s per trial). The data transfer overhead exceeded any computational gain. GPU acceleration would only become beneficial with significantly larger datasets.

---

## Phase 5: Graphical User Interface (Day 4–5)

### 5.1 Framework Selection

Gradio was chosen for its simplicity and built-in support for machine learning demos. The interface provides three tabs:

1. **Upload CSV**: Users upload their own light curve data for classification
2. **Demo**: Randomly samples PLAsTiCC objects and displays predictions vs. ground truth
3. **Benchmark**: Evaluates model accuracy on a user-selected number of test samples

### 5.2 Internationalization

A bilingual (English/Chinese) interface was implemented. Tab labels use static bilingual text; Markdown content and component labels switch dynamically via `gr.update()`. Buttons and the download component use static bilingual labels due to Gradio 6 API limitations.

### 5.3 Cross-Platform Launcher

A Python-based launcher (`launcher.py`) was developed to handle:

- Automatic dependency installation (pip)
- PLAsTiCC dataset download prompt (~35 MB, with user consent)
- Port conflict resolution (scans 38000–38020)
- Server health polling before opening browser
- Old instance cleanup on startup

The launcher supports Windows, macOS, and Linux with identical behavior.

### 5.4 Distribution

Two distribution packages were prepared:

- `deliver/` — bilingual (English/Chinese)
- `deliver_en/` — English only

Both contain the complete application, model files, and source code. Users only need Python 3.10+.

---

## Phase 6: Validation & Results

### 6.1 Final Model Performance

Evaluated on a held-out test set of 1,295 objects across 14 classes:

| Metric | Value |
|--------|-------|
| Accuracy | 81.93% |
| Macro-averaged F1 | 75.95% |
| Weighted F1 | 81.59% |

### 6.2 Per-Class Performance

| Class | F1 Score |
|-------|----------|
| Eclipsing binary | 0.987 |
| M-dwarf | 0.985 |
| Active galactic nucleus | 0.984 |
| RR Lyrae | 0.963 |
| Superluminous supernova (SLSN-I) | 0.915 |
| Microlens (single) | 0.902 |
| Type Ia supernova | 0.868 |
| Tidal disruption event | 0.759 |
| Type II supernova | 0.663 |
| Kilonova | 0.600 |
| Type Ibc supernova | 0.579 |
| Peculiar SNIa (91bg-like) | 0.438 |
| Peculiar SNIa (SNIax) | 0.360 |

Best-performing classes (eclipsing binaries, M-dwarfs, AGN) have highly distinctive, periodic light curves. Worst-performing classes are supernova subtypes with similar photometric evolution, which even professional astronomers struggle to distinguish without spectroscopy.

### 6.3 Confusion Analysis

The most common misclassifications:

- Type II SN ↔ Type Ia SN (61 mutual confusions in 1,295 samples)
- Type Ibc SN ↔ Type II SN
- Peculiar SNIa subtypes ↔ Type Ia SN

These confusions are physically meaningful — the light curve morphology of different core-collapse and thermonuclear supernovae overlap significantly in broadband photometry. Spectroscopic follow-up is typically required for definitive classification.

---

## Phase 7: Lessons Learned

1. **Feature engineering matters more than model architecture** for small tabular datasets. Forty-seven carefully designed features outperformed every deep learning approach by a wide margin.

2. **Data quantity is the bottleneck for deep learning in astronomy**, not compute. With only ~7,000 training samples, even a capable GPU (RTX 4060 Ti) could not salvage underperforming neural networks.

3. **Gradient-boosted trees remain state-of-the-art** for tabular data with hundreds to thousands of samples. XGBoost consistently outperformed MLP, LSTM, Transformer, and CNN in our experiments.

4. **GUI development requires as much care as model development**. Handling cross-platform compatibility, dependency management, missing data scenarios, and internationalization took significant effort.

5. **Domain knowledge is irreplaceable**. Features like color indices and host galaxy redshift — obvious to astronomers — would take a neural network orders of magnitude more data to discover independently.

---

*Document prepared with assistance from DeepSeek.*
