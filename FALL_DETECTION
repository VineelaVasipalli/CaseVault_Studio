<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-58A6FF?style=for-the-badge"/>






<br/><br/>

# 🏥 Gait Analysis Using IMU Time-Series for Fall Risk Detection

### A complete end-to-end biomedical machine learning pipeline for real-time fall detection using smartphone inertial measurement unit (IMU) signals.

<br/>

> **Dataset:** MobiFall v2.0 · **Signals:** Accelerometer + Gyroscope · **Sampling Rate:** 87 Hz  
> **Models:** Random Forest · XGBoost · SVM · LSTM · **Graphs:** 15 clinical visualizations

<br/>

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Clinical Motivation](#-clinical-motivation)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Pipeline Sections](#-pipeline-sections)
- [Features Extracted](#-features-extracted)
- [Models & Results](#-models--results)
- [Visualizations](#-visualizations)
- [Getting Started](#-getting-started)
- [Requirements](#-requirements)
- [File Structure](#-file-structure)
- [Scientific Interpretation](#-scientific-interpretation)
- [Limitations & Future Work](#-limitations--future-work)
- [Clinical Applications](#-clinical-applications)
- [References](#-references)

---

## 🔭 Overview

This project implements a **research-grade, fully reproducible pipeline** for detecting fall events from raw IMU time-series data collected via smartphone sensors. Using the **MobiFall v2.0** dataset — recorded from subjects performing 11 distinct activities including 4 fall types — the notebook delivers a complete journey from raw signal ingestion to clinical risk scoring.

Every stage of the pipeline is grounded in **biomedical signal processing standards** and validated with statistical tests used in peer-reviewed gait analysis literature.

```
Raw IMU Signal  →  Preprocessing  →  Windowing  →  Feature Extraction
       →  Time-Series Analysis  →  ML/DL Models  →  Fall Risk Dashboard
```

---

## 🩺 Clinical Motivation

Falls are the **leading cause of injury-related death** among adults aged 65 and older.

| Statistic | Value |
|-----------|-------|
| Annual falls in adults 65+ (USA) | ~36 million |
| Fall-related deaths per year | ~32,000 |
| Emergency visits from falls | ~3 million |
| Annual healthcare cost | ~$50 billion |

> *Source: Centers for Disease Control and Prevention (CDC), 2020*

Early, automatic fall detection via wearable sensors can trigger **emergency alerts within seconds**, dramatically improving survival outcomes. This project demonstrates a wearable-ready pipeline that can be deployed on a smartphone or smartwatch.

---

## 📦 Dataset

**MobiFall Dataset v2.0** — Recorded at the Information Technologies Institute (ITI), Greece.

| Property | Value |
|----------|-------|
| Sensor | Smartphone IMU (Samsung Galaxy S3) |
| Placement | Trouser pocket (thigh level) |
| Sampling Rate | **87 Hz** |
| Subjects | 24 volunteers |
| Signal Channels | acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z |

### Activity Labels

| Code | Activity | Fall? |
|------|----------|-------|
| `WAL` | Walking | ❌ |
| `JOG` | Jogging | ❌ |
| `STU` | Stairs Up | ❌ |
| `STN` | Stairs Down | ❌ |
| `STD` | Standing | ❌ |
| `SCH` | Sitting | ❌ |
| `JUM` | Jumping | ❌ |
| `FOL` | Fall — Forward on Floor | ✅ |
| `FKL` | Fall — Forward on Knees | ✅ |
| `BSC` | Fall — Backward Sitting to Chair | ✅ |
| `SDL` | Fall — Sideways | ✅ |

### How to Get the Dataset

Upload your MobiFall ZIP file directly in Colab when prompted by **Cell 3**:

```python
from google.colab import files
uploaded = files.upload()   # ← Click "Choose Files" → select your ZIP
```

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FALL_DETECTION.ipynb                            │
│                                                                     │
│  SECTION 1 ─ Environment Setup                                      │
│    └── Install packages, imports, dark theme, label maps            │
│                                                                     │
│  SECTION 2 ─ Data Loading                                           │
│    └── ZIP upload → robust .txt parser → DataFrame assembly         │
│                                                                     │
│  SECTION 3 ─ Signal Visualization     [Graph 1]                     │
│    └── Raw SMV per activity                                         │
│                                                                     │
│  SECTION 4 ─ Preprocessing            [Graphs 2, 3]                 │
│    └── Butterworth LPF → DC removal → Z-score → Sliding window      │
│                                                                     │
│  SECTION 5 ─ Feature Extraction       [27 clinical features]        │
│    └── Stride time, cadence, jerk, SMA, spectral entropy …          │
│                                                                     │
│  SECTION 6 ─ Time-Series Analysis     [Graphs 4–9]                  │
│    └── KDE distributions, ACF, ADF, decomposition, PSD              │
│                                                                     │
│  SECTION 7 ─ Machine Learning         [Graphs 10–13]                │
│    └── Random Forest · XGBoost · SVM                                │
│                                                                     │
│  SECTION 8 ─ Deep Learning            [Graphs 11–12]                │
│    └── LSTM · Dropout · BatchNorm · EarlyStopping                   │
│                                                                     │
│  SECTION 9 ─ Model Comparison         [Graph 12]                    │
│    └── Confusion matrices · ROC · PR curves · Metric bars           │
│                                                                     │
│  SECTION 10 ─ Clinical Dashboard      [Graphs 14–15]                │
│    └── Individual risk gauges · Final summary                       │
│                                                                     │
│  SECTION 11 ─ Scientific Interpretation                             │
│    └── Findings · Limitations · Future research                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Pipeline Sections

### Section 1 — Environment Setup
Installs `xgboost`, `statsmodels`, configures a **professional dark matplotlib theme**, and defines the MobiFall activity label dictionary (`LABEL_MAP`, `ACT_COLORS`, `FALL_CODES`).

---

### Section 2 — Data Loading
A robust multi-stage loader handles the MobiFall `.txt` format:
- Skips `%` comment lines and non-numeric headers automatically
- Supports both `;` and `,` delimited files
- Infers **activity label** and **fall/non-fall** status from the parent folder name
- Extracts subject ID from filenames (`sub1_acc_1.txt` → subject 1)
- Assembles all files into a single unified DataFrame

---

### Section 3 — Signal Visualization `[Graph 1]`
Plots the **Signal Magnitude Vector** (SMV = √(acc_x² + acc_y² + acc_z²)) for every activity:
- Walking/Jogging → periodic oscillations
- Sitting/Standing → near-flat gravity signal
- Fall events → sudden high-amplitude spike → silence

---

### Section 4 — Signal Preprocessing `[Graphs 2, 3]`

| Step | Method | Reason |
|------|--------|--------|
| DC removal | Subtract mean | Remove gravity bias |
| Low-pass filter | Butterworth 4th-order, 10 Hz | Remove EMG & sensor noise |
| Normalization | Z-score per subject | Equalize amplitude across subjects |
| Segmentation | Sliding window 3s / 50% overlap | Capture full gait cycles + falls |

**Window parameters:**
```
Window size : 261 samples = 3.0 seconds @ 87 Hz
Step size   : 130 samples = 1.5 seconds (50% overlap)
```

---

### Section 5 — Gait Feature Extraction

27 clinically validated features per window — see [Features Extracted](#-features-extracted) below.

---

### Section 6 — Time-Series Analysis `[Graphs 4–9]`

| Analysis | Tool | Clinical Insight |
|----------|------|-----------------|
| Feature KDE distributions | `scipy.stats.gaussian_kde` | Separation between fall / non-fall |
| Mann-Whitney U test | `scipy.stats.mannwhitneyu` | Statistical significance (p-value) |
| Autocorrelation (ACF) | Custom implementation | Gait periodicity vs. aperiodic falls |
| ADF Stationarity | `statsmodels.tsa.adfuller` | Predictability of gait signal |
| Seasonal Decomposition | `statsmodels.tsa.seasonal_decompose` | Isolate trend, cycle, residual |
| Power Spectral Density | `scipy.signal.welch` | Dominant stride frequency |

---

### Section 7 — Machine Learning Models `[Graphs 10–13]`

Three traditional ML models trained on the 27 extracted features:

```python
# Example: Random Forest pipeline
Pipeline([
    ('sc',  StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced'))
])
```

Each model evaluated with **5-fold stratified cross-validation** + held-out test set.

---

### Section 8 — Deep Learning: LSTM `[Graphs 11–12]`

```
Input (N, 27, 1)
    └── LSTM(128, return_sequences=True)
    └── Dropout(0.3)
    └── LSTM(64)
    └── Dropout(0.3)
    └── BatchNormalization()
    └── Dense(32, relu)
    └── Dropout(0.2)
    └── Dense(1, sigmoid)   ← Fall probability
```

Training features: `EarlyStopping(patience=8)`, `ReduceLROnPlateau`, class weighting.

---

## 📐 Features Extracted

27 features per 3-second window, organized by clinical category:

### Stride & Cadence
| Feature | Formula | Clinical Meaning |
|---------|---------|-----------------|
| `stride_time_mean` | mean(Δpeak) / fs | Average time between footfalls |
| `stride_time_std` | std(Δpeak) / fs | Stride time variability |
| `cadence` | 60 / stride_time | Steps per minute |
| `step_variance` | var(acc_y at peaks) | Vertical force variability |
| `symmetry_index` | std/mean × 100% | CoV of stride times (>15% = disorder) |
| `n_steps` | count(peaks) | Number of steps in window |
| `regularity` | autocorr(smv, lag) | Gait rhythm consistency |

### Energy & Dynamics
| Feature | Formula | Clinical Meaning |
|---------|---------|-----------------|
| `jerk` | mean(\|Δsmv\|) / dt | Rate of acceleration change — highest in falls |
| `signal_energy` | Σsmv² / N | Total kinetic energy in window |
| `signal_rms` | √(mean(smv²)) | RMS acceleration magnitude |
| `sma` | Σ(\|ax\| + \|ay\| + \|az\|) / N | Signal Magnitude Area |
| `smv_mean` | mean(smv) | Mean magnitude |
| `smv_std` | std(smv) | Magnitude variability |
| `smv_max` | max(smv) | Peak acceleration (spike in falls) |
| `smv_p95` | percentile(smv, 95) | 95th percentile magnitude |
| `smv_skew` | scipy.stats.skew | Distribution asymmetry |

### Axis-Specific
| Feature | Clinical Meaning |
|---------|-----------------|
| `acc_x/y/z_std` | Variability on each axis |
| `acc_y_mean` | Mean vertical acceleration |
| `gyro_x/y/z_std` | Rotational variability per axis |
| `gyro_rms` | Overall rotation intensity |

### Spectral
| Feature | Clinical Meaning |
|---------|-----------------|
| `dom_freq` | Dominant stride frequency (Hz) |
| `spectral_entropy` | Signal complexity — low in falls (impulsive) |

---

## 🤖 Models & Results

### Model Comparison Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | — | — | — | — | — |
| **XGBoost** | — | — | — | — | — |
| **SVM (RBF)** | — | — | — | — | — |
| **LSTM** | — | — | — | — | — |

> *Actual values populated when notebook is run — results vary with dataset version and random seed.*

### Key Hyperparameters

```python
# Random Forest
RandomForestClassifier(n_estimators=300, max_depth=10,
                       min_samples_split=5, class_weight='balanced')

# XGBoost
XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=ratio)

# SVM
SVC(kernel='rbf', C=10, gamma='scale',
    class_weight='balanced', probability=True)

# LSTM
Sequential: LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3)
          → BatchNorm → Dense(32) → Dropout(0.2) → Dense(1, sigmoid)
```

### Why These Models?

| Model | Strength for Fall Detection |
|-------|----------------------------|
| **Random Forest** | Interpretable feature importance; robust to noise |
| **XGBoost** | Best-in-class on tabular features; handles class imbalance |
| **SVM** | Strong generalization on small fall datasets; RBF captures non-linear boundaries |
| **LSTM** | Learns temporal fall signature: gait → impact spike → stillness |

> **Clinical priority metric: Recall (Sensitivity)**  
> A missed fall (False Negative) is far more dangerous than a false alarm.  
> Models are optimized to maximize recall on the fall class.

---

## 📊 Visualizations

All 15 graphs are generated with **matplotlib only** (no seaborn), saved as high-DPI PNG files.

| Graph | Title | Key Insight |
|-------|-------|-------------|
| **Graph 1** | Raw IMU Magnitude by Activity | Fall spike vs. walking oscillation |
| **Graph 2** | Preprocessing Pipeline | DC removal → filter → normalization stages |
| **Graph 3** | Sliding Window Segmentation | 50% overlap windowing strategy |
| **Graph 4** | Feature Distributions (KDE) | Fall vs. non-fall separation per feature |
| **Graph 5** | Cadence & Stride Symmetry | Clinical gait metrics comparison |
| **Graph 6** | Autocorrelation (ACF) | Periodic walking vs. aperiodic falls |
| **Graph 7** | ADF Stationarity Test | Signal predictability per activity |
| **Graph 8** | Seasonal Decomposition | Trend / cycle / residual breakdown |
| **Graph 9** | Power Spectral Density | Stride frequency vs. broadband fall energy |
| **Graph 10** | Feature Variance | Most discriminative features for ML |
| **Graph 11** | LSTM Training History | Loss / accuracy / AUC curves + early stopping |
| **Graph 12** | Model Comparison Dashboard | Confusion matrices + ROC + PR + metric bars |
| **Graph 13** | Feature Importance (RF + XGB) | Clinical drivers of fall risk prediction |
| **Graph 14** | Individual Risk Dashboard | Per-subject gauge meter + ensemble prediction |
| **Graph 15** | Final Summary Dashboard | Complete project overview in 6 panels |

---

## 🚀 Getting Started

### Option A — Google Colab (Recommended)

1. Upload `FALL_DETECTION.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run **Cell 1** to install packages
3. Run **Cell 2** to import libraries
4. Run **Cell 3** → click **"Choose Files"** → select your MobiFall ZIP
5. Run all remaining cells sequentially (`Runtime → Run all`)

### Option B — Local Jupyter

```bash
# 1. Clone or download the notebook
git clone https://github.com/your-username/fall-detection-imu.git
cd fall-detection-imu

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook FALL_DETECTION.ipynb
```

> **Note:** For local use, replace `from google.colab import files` in Cell 3 with:
> ```python
> import zipfile
> zip_filename = "path/to/your/mobifall-dataset-v20.zip"
> with zipfile.ZipFile(zip_filename, 'r') as zf:
>     zf.extractall("/tmp/mobifall_dataset")
> path = "/tmp/mobifall_dataset"
> ```

---

## 📋 Requirements

```
Python          >= 3.10
numpy           >= 1.24
pandas          >= 2.0
matplotlib      >= 3.7
scipy           >= 1.11
statsmodels     >= 0.14
scikit-learn    >= 1.3
xgboost         >= 2.0
tensorflow      >= 2.13
kagglehub       >= 0.2     (optional — for direct Kaggle download)
```

Install all at once:

```bash
pip install numpy pandas matplotlib scipy statsmodels scikit-learn xgboost tensorflow
```

---

## 📁 File Structure

```
fall-detection-imu/
│
├── FALL_DETECTION.ipynb       ← Main notebook (25 code cells, 61 total)
├── README.md                  ← This file
│
├── outputs/                   ← Auto-generated when notebook runs
│   ├── graph1_raw_signals.png
│   ├── graph2_preprocessing.png
│   ├── graph3_sliding_window.png
│   ├── graph4_feature_distributions.png
│   ├── graph5_cadence_symmetry.png
│   ├── graph6_acf.png
│   ├── graph7_adf_test.png
│   ├── graph8_decomposition.png
│   ├── graph9_psd.png
│   ├── graph10_feature_variance.png
│   ├── graph11_lstm_history.png
│   ├── graph12_model_dashboard.png
│   ├── graph13_feature_importance.png
│   ├── graph14_risk_dashboard.png
│   └── graph15_final_dashboard.png
│
└── data/                      ← Place extracted MobiFall dataset here
    └── mobifall_dataset/
        ├── WAL/               ← Walking .txt files
        ├── JOG/               ← Jogging .txt files
        ├── FOL/               ← Fall (forward) .txt files
        ├── BSC/               ← Fall (backward) .txt files
        └── ...
```

---

## 🧪 Scientific Interpretation

### Why Fall Signals Are Distinct

A fall event produces a **unique three-phase time-domain signature**:

```
Phase 1: Pre-fall   → Normal gait / balance perturbation  (0.3–1.0s)
Phase 2: Impact     → Explosive high-amplitude spike       (0.05–0.2s)
Phase 3: Post-fall  → Near-zero signal (lying still)       (rest of window)
```

This signature creates discriminative patterns in **every analytical domain**:

| Domain | Fall Signature |
|--------|---------------|
| Time domain | SMV spike > 3× normal walking amplitude |
| Frequency domain | Broadband energy spread (no dominant frequency) |
| ACF | Rapid decay to zero (no periodicity) |
| Seasonal decomp. | No seasonal component — all energy in residual |
| Features | Jerk → max, Cadence → 0, Symmetry index → 100%+ |

### Statistical Validation

All feature differences between fall and non-fall classes are validated using the **Mann-Whitney U test** (non-parametric, suitable for non-normally distributed gait signals). Features with `p < 0.001` (***) are considered highly significant clinical discriminators.

---

## ⚠️ Limitations & Future Work

### Current Limitations

| Limitation | Impact |
|-----------|--------|
| Controlled lab falls | May not generalize to spontaneous real-world falls |
| Single IMU placement | Misses upper-body fall types |
| Class imbalance | Fewer fall samples requires careful handling |
| Post-fall detection only | Cannot predict falls before they happen |
| Healthy young subjects | May not represent elderly gait patterns |

### Future Research Directions

- **Multi-modal fusion** — IMU + pressure insoles + RGB-D cameras
- **Transformer models** — Self-attention over longer temporal windows (>10s)
- **Transfer learning** — Cross-demographic adaptation (age, BMI, pathology)
- **Edge deployment** — TensorFlow Lite on smartwatch / Raspberry Pi
- **Pre-fall prediction** — Balance perturbation detection using micro-perturbation analysis
- **Federated learning** — Privacy-preserving training across hospital IoT devices

---

## 🏥 Clinical Applications

```
🔔 Real-time fall alert          → Wearable triggers emergency call within 2s of fall
🏃 Rehabilitation monitoring     → Track gait recovery after hip surgery
🧓 Elderly care homes            → Continuous background fall risk scoring
⚽ Sports injury prevention      → Detect dangerous loading patterns in athletes
🏥 Pre-operative assessment      → Objective gait score before orthopaedic surgery
🔬 Clinical gait lab automation  → Replace manual video annotation with ML
```

---

## 📚 References

1. Vavoulas, G. et al. (2016). *The MobiFall Dataset: Fall Detection and Classification with a Smartphone.* International Journal on Artificial Intelligence Tools.

2. Noury, N. et al. (2007). *Fall Detection — Principles and Methods.* 29th IEEE EMBS Annual Conference.

3. Igual, R. et al. (2013). *Challenges, Issues and Trends in Fall Detection Systems.* BioMedical Engineering OnLine, 12(1), 66.

4. Sucerquia, A. et al. (2017). *SisFall: A Fall and Movement Dataset.* Sensors, 17(1), 198.

5. Kwolek, B. & Kepski, M. (2014). *Human Fall Detection on Embedded Platform Using Depth Maps and Wireless Accelerometer.* Computer Methods and Programs in Biomedicine.

6. CDC (2020). *Falls Are Leading Cause of Injury and Death in Older Americans.* Centers for Disease Control and Prevention.

---

<div align="center">

### Built for biomedical research · Designed for clinical deployment · Ready for Google Colab

<br/>

**Dataset:** [MobiFall v2.0 on Kaggle](https://www.kaggle.com/datasets/kmknation/mobifall-dataset-v20) · **Framework:** TensorFlow 2.x + Scikit-Learn · **Visualization:** Matplotlib

<br/>

*If this project helped your research, consider citing the MobiFall dataset authors.*

</div>
