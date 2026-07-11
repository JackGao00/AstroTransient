# Automatic Classification of Astronomical Transients from Multi-Band Photometric Light Curves

**Daisy Shang, NullTemp · 2026**

*2026*

---

## Abstract

We present a machine learning pipeline for classifying astronomical transients and variable sources using multi-band photometric light curves. A gradient-boosted decision tree classifier (XGBoost) was trained on 47 handcrafted features extracted from the PLAsTiCC dataset, which simulates light curves expected from the Vera C. Rubin Observatory's Legacy Survey of Space and Time. The model achieves 82% accuracy across 14 classes on a held-out test set. We also explored several deep learning architectures (LSTM, Transformer, CNN, contrastive pretraining) and found that they underperform the feature-based approach given the limited training data (~7,000 objects). A cross-platform graphical interface was developed for model interaction and demonstration.

---

## 1. Introduction

Time-domain astronomy is entering an era of unprecedented data volume. The Vera C. Rubin Observatory, scheduled to begin operations in the coming years, will produce approximately 10 million photometric alerts per night. Manual classification of every transient event is infeasible at this scale, motivating the development of automated classification methods.

This project investigates whether a relatively simple feature-based machine learning approach can achieve competitive classification performance using a modest amount of training data. We use the PLAsTiCC dataset — a collection of simulated light curves prepared for a 2018 Kaggle competition — which contains 18 classes of astronomical objects. Our goal is to build a classifier that distinguishes the 14 classes present in the training set with sufficient accuracy to be practically useful.

The project also serves as an exploration of the relative merits of traditional machine learning versus deep learning for small-to-medium astronomical datasets.

---

## 2. Data

### 2.1 The PLAsTiCC Dataset

PLAsTiCC (Photometric LSST Astronomical Time-series Classification Challenge) was released in 2018 as a community benchmark for light curve classification. The dataset simulates LSST observations across six photometric bands (u, g, r, i, z, y) for 18 astronomical source classes. Each object is represented by a time series of flux measurements with associated uncertainties.

The training set contains approximately 7,000 labeled objects, with an additional 1,500 in the validation split. The test set, which we did not use for training, contains several million observations across 85 partition files and was excluded due to memory constraints.

### 2.2 Class Distribution

The 14 classes present in the training data exhibit significant imbalance. Type Ia supernovae are the most numerous (~2,500 objects), while classes such as Mira variables and kilonovae have fewer than 100 samples. This imbalance was addressed through class-weighted training.

### 2.3 Data Format

The dataset is stored in Apache Parquet format with a nested structure. Each row contains:

- `object_id`: unique identifier
- `times_wv`: array of `[MJD, passband_wavelength]` pairs
- `lightcurve`: array of `[flux, flux_err]` pairs
- `label`: integer class identifier
- `redshift`, `hostgal_specz`, `hostgal_photoz`: metadata

Preprocessing involved converting object arrays to numeric arrays and filtering observations by passband for per-band feature extraction.

---

## 3. Methods

### 3.1 Feature Engineering

From each light curve, we extract 47 features spanning several physical domains:

**Global statistics** (19 features): duration, number of observations, flux mean/standard deviation/median/minimum/maximum/range, skewness, kurtosis, signal-to-noise ratio (mean and maximum), amplitude (5th–95th percentile range), fraction of points beyond 1σ, maximum rise and decay rates, peak position, autocorrelation at lag 1, interquartile range, and median absolute deviation.

**Per-band statistics** (18 features): for each of the six LSST bands (u, g, r, i, z, y), we compute the mean flux, standard deviation, and signal-to-noise ratio of observations within ±500 Å of the band center.

**Color indices** (5 features): band-to-band flux differences (u−g, g−r, r−i, i−z, z−y), which serve as temperature and spectral energy distribution proxies.

**Contextual features** (5 features): redshift, host galaxy spectroscopic and photometric redshift, and number of unique passbands observed.

### 3.2 Model Architecture

The primary classifier is an XGBoost (eXtreme Gradient Boosting) model — an ensemble of decision trees trained sequentially with gradient boosting. XGBoost was chosen for its strong performance on tabular data, built-in regularization, and handling of missing values.

Key hyperparameters: `n_estimators=500`, `max_depth=10`, `learning_rate=0.02`, `subsample=0.8`, `reg_alpha=0.5`, `reg_lambda=1.5`. These values were arrived at through a combination of manual tuning and automated search (Optuna). An AI assistant was used to help explore the parameter space and design the search strategy.

Class weights were applied to compensate for the imbalanced distribution: each class was weighted inversely proportional to its frequency in the training set.

### 3.3 Training Protocol

The training pipeline consists of:

1. Feature extraction from all available PLAsTiCC training and validation objects (8,630 total)
2. Removal of infinite values; median imputation of missing values
3. Standardization to zero mean and unit variance
4. Stratified split: 70% training, 15% validation, 15% test
5. Training with early stopping on validation log-loss

Models were evaluated using accuracy and macro-averaged F1 score to account for class imbalance.

### 3.4 Deep Learning Baselines

Five neural network architectures were evaluated for comparison:

- **BiLSTM with attention**: bidirectional LSTM processing raw light curve sequences
- **Transformer encoder**: multi-head self-attention over time steps
- **2D Heatmap CNN**: convolutional network on time-by-passband heatmap images
- **MLP**: multi-layer perceptron trained on the same 47 features as XGBoost
- **SimCLR contrastive pretraining**: self-supervised Conv1D encoder followed by XGBoost on learned embeddings

All deep learning models were trained using PyTorch 2.5 with CUDA 12.1.

### 3.5 Graphical Interface

A web-based interface was built using Gradio, providing three functional tabs: CSV file upload for custom light curve classification, random PLAsTiCC object demonstration, and batch accuracy benchmarking. A cross-platform Python launcher was developed to automate dependency installation, data download, and server startup.

---

## 4. Results

### 4.1 Classification Performance

The XGBoost model achieved **81.9% accuracy** and **76.0% macro-averaged F1 score** on a held-out test set of 1,295 objects.

Per-class F1 scores ranged from 0.987 (eclipsing binaries) to 0.360 (peculiar Type Ia supernovae, SNIax subtype). Classes with highly distinctive light curve morphologies — periodic variables, AGN with stochastic variability, M-dwarfs with flare signatures — were classified with near-perfect accuracy. Supernova subtypes (Ia, II, Ibc) were the most challenging to distinguish, which is consistent with the known photometric degeneracy between these classes.

### 4.2 Confusion Analysis

The confusion matrix reveals systematic patterns: Type II supernovae are most often misclassified as Type Ia (30 instances) or Type Ibc (18 instances). Peculiar SNIa subtypes (SNIax, SNIa-91bg) are frequently confused with normal Type Ia. These confusions are physically meaningful — the broadband light curves of different explosion mechanisms overlap significantly, and definitive classification typically requires spectroscopic follow-up.

### 4.3 Deep Learning Comparison

| Model | Accuracy | Macro F1 | Training Time |
|-------|----------|----------|---------------|
| XGBoost (47 features) | 81.9% | 76.0% | ~2 min (CPU) |
| MLP (47 features) | 69.2% | 56.8% | ~1 min (GPU) |
| 2D Heatmap CNN | 33.5% | 26.9% | 12 s (GPU) |
| LSTM (raw sequence) | 27.0% | 21.4% | 86 s (GPU) |
| Transformer (raw sequence) | 26.0% | 22.0% | 86 s (GPU) |
| Contrastive + XGBoost | 34.4% | 26.3% | 15 s (GPU) + 2 min |

The deep learning models trained on raw light curves all performed near or below the random-guess baseline. This is primarily attributed to the limited training set size — approximately 4,000 sequences are insufficient for deep temporal models to learn meaningful representations.

### 4.4 Feature Importance

The top five features by XGBoost importance were: host galaxy spectroscopic redshift, amplitude (P5–P95), object redshift, y-band flux standard deviation, and host galaxy photometric redshift. This indicates that contextual information — particularly the distance and environment of the host galaxy — provides strong discriminative power independent of the light curve shape.

---

## 5. Discussion

### 5.1 Why Handcrafted Features Outperformed Deep Learning

The result that a simple feature-based XGBoost model outperforms all tested deep learning architectures may seem surprising given deep learning's dominance in other domains. However, it is consistent with findings in the broader machine learning literature: for tabular data with hundreds to low thousands of training samples, gradient-boosted trees consistently match or exceed neural networks.

The handcrafted features encode physical priors — color indices as temperature indicators, amplitude ratios as energy release proxies, periodic signatures as binary system markers — that a neural network would need orders of magnitude more data to discover independently. With the million-scale datasets now available from surveys like the Zwicky Transient Facility, deep learning would likely become competitive or superior.

### 5.2 Limitations

Several limitations of this work should be acknowledged:

- **Simulated data**: PLAsTiCC light curves are simulations. Real survey data from ZTF or LSST will contain systematic effects (calibration errors, blending, detector artifacts) not present in the training set.
- **Class coverage**: Four of the original 18 PLAsTiCC classes were not present in the training partition and could not be modeled.
- **Single-epoch features**: Our features aggregate over the entire light curve; they do not explicitly model temporal evolution. Early-time classification, which is critical for triggering follow-up observations, would require a different approach.
- **No uncertainty quantification**: The model provides point predictions without confidence intervals or out-of-distribution detection.

### 5.3 Future Directions

- **Real data transfer**: Fine-tune on ZTF alert stream data to bridge the simulation-to-reality gap.
- **Time-resolved features**: Extract features from partial light curves for early-stage classification.
- **Anomaly detection**: Extend the pipeline to flag objects that do not match any known class — the regime where new astrophysics is discovered.
- **Multi-modal fusion**: Combine photometric features with host galaxy imaging or contextual survey metadata.

---

## 6. Conclusion

We developed a 14-class astronomical transient classifier using 47 handcrafted features and gradient-boosted decision trees, achieving 82% accuracy on the PLAsTiCC benchmark. The classifier is packaged with a cross-platform graphical interface suitable for demonstration and educational use. Extensive experimentation with deep learning architectures confirmed that, for datasets of this scale, careful feature engineering remains more effective than end-to-end learning from raw sequences. The complete code, trained model, and documentation are available in the accompanying software package.

---

## Acknowledgments

This project was developed with computational assistance from DeepSeek, which provided code generation, debugging support, and documentation drafting throughout the development process.

---

## References

1. The PLAsTiCC Team, et al. "The Photometric LSST Astronomical Time-series Classification Challenge (PLAsTiCC): Data set." arXiv:1810.00001 (2018).
2. Kessler, R., et al. "Models and Simulations for the Photometric LSST Astronomical Time Series Classification Challenge (PLAsTiCC)." arXiv:1903.11756 (2019).
3. Chen, T., Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD (2016).
4. Rehemtulla, N., et al. "The Zwicky Transient Facility Bright Transient Survey. III. BTSbot." ApJ 972, 7 (2024).
5. Fei, Y., et al. "LEAVES: An Expandable Light-curve Data Set for Automatic Classification of Variable Stars." ApJS (2024).
