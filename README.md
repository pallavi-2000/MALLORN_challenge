# 🌟 MALLORN V5: Physics-Based TDE Classification

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A physics-informed machine learning approach for identifying **Tidal Disruption Events (TDEs)** in simulated LSST lightcurve data, achieving an F1 score of **0.6361** on the Kaggle MALLORN Astronomical Classification Challenge.

---

## 📋 Table of Contents

- [Overview](#overview)
- [The Challenge](#the-challenge)
- [Approach](#approach)
- [Golden Features](#golden-features)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Evolution](#project-evolution)
- [Key Lessons Learned](#key-lessons-learned)
- [Acknowledgements](#acknowledgements)

---

## Overview

The **MALLORN (Many Artificial LSST Lightcurves based on Observations of Real Nuclear transients) Challenge** prepares the astronomical community for the upcoming Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST). This competition focuses on photometrically identifying TDEs—rare events where stars are torn apart by supermassive black holes—from simulated lightcurve data.

### Why TDEs Matter

Tidal Disruption Events are scientifically valuable for studying supermassive black hole properties, but with only ~100 observed to date, our research is sample-limited. LSST will discover vastly more transients than we can spectroscopically follow up, making photometric classification essential.

---

## The Challenge

| Aspect | Details |
|--------|---------|
| **Task** | Binary classification: TDE (1) vs Non-TDE (0) |
| **Data** | Simulated LSST lightcurves based on real ZTF observations |
| **Class Imbalance** | Only ~0.6% of training samples are TDEs |
| **Metric** | F1 Score |
| **Bands** | u, g, r, i, z, y (LSST filters) |

---

## Approach

### Core Philosophy: Feature Engineering > Data Augmentation

Based on competition insights, this solution prioritizes **physics-informed feature engineering** over data augmentation strategies. Previous experiments showed that synthetic data generation (both 50× and 3× augmentation) degraded performance by introducing distribution shifts.

### Physical Discriminators

TDEs have distinct signatures compared to supernovae (SNe) and Active Galactic Nuclei (AGN):

| Feature | TDE | Supernova | AGN |
|---------|-----|-----------|-----|
| **Rise time** | Slow (weeks-months) | Fast (~2 weeks) | Stochastic |
| **Decay** | t^(−5/3) power law | Exponential | Random |
| **Duration** | 100-300+ days | 50-100 days | Years |
| **Color** | Very blue (u-band strong) | Variable | Redder |
| **Symmetry** | Asymmetric (decay >> rise) | More symmetric | N/A |
| **Smoothness** | Very smooth | Smooth | Stochastic/flickering |
| **Temperature** | ~30,000-50,000K | ~10,000K | Variable |

---

## Golden Features

The V5 approach extracts physics-motivated features across 8 categories:

### 1. 📈 Shape Features (TDE vs SNe)
- **Rise time**: Time from first observation to peak
- **Decay time**: Time from peak to last observation  
- **Asymmetry ratio**: `decay_time / rise_time` (TDEs have long tails)
- **FWHM**: Full Width at Half Maximum
- **t10-t90**: Time between 10% and 90% of peak flux

### 2. 📉 Power-Law Decay Index (TDE Signature)
- Fits `log(F) = log(A) + α·log(t)` to post-peak decay
- **tde_index_match**: Distance from theoretical t^(−5/3) ≈ −1.67
- **decay_power_r2**: Goodness of fit

### 3. 🔵 Color Features (TDEs are BLUE!)
- **u/r flux ratios**: TDEs have strong UV emission
- **g/r ratios**: Proxy for blackbody temperature
- **Color at peak**: g/r ratio within ±10 days of maximum
- **Color evolution**: g/r at +30, +60, +100 days post-peak

### 4. 📊 Stochasticity Metrics (TDE vs AGN)
- **flux_scatter**: Excess variance beyond photometric errors
- **chi2_smooth**: Residuals from smoothed lightcurve
- **n_local_peaks**: Count of local maxima (TDE = 1, AGN = many)
- **largest_dip_frac**: Post-peak variability
- **flux_ratio_std**: Consecutive flux ratio variance

### 5. 🌊 Bazin Function Fits
Fits the standard transient model:
```
F(t) = A × exp(−(t−t₀)/t_fall) / (1 + exp(−(t−t₀)/t_rise)) + c
```
Extracts: amplitude, t_fall, t_rise, **rise_fall_ratio**, fit χ²

### 6. 🎨 Per-Band Statistics
For each LSST filter (u, g, r, i, z, y):
- Observation count, flux mean/std/max, amplitude

### 7. 🌐 Redshift Corrections
- Rest-frame duration: `duration / (1 + z)`
- E(B-V) dust extinction corrections via `fitzpatrick99`

### 8. 🔧 Preprocessing
- Milky Way dust de-extinction applied per-band
- Proper time-ordering of observations

---

## Results

### Model Performance

| Model | OOF F1 | Optimal Threshold |
|-------|--------|-------------------|
| LightGBM | 0.62+ | ~0.10-0.15 |
| XGBoost | 0.62+ | ~0.10-0.15 |
| **Ensemble** | **0.6361** | **0.12** |

### Ensemble Configuration
- **LightGBM weight**: 0.7
- **XGBoost weight**: 0.3
- **Class weight**: ~165× (inverse class frequency)

### Version Comparison

| Version | Strategy | OOF F1 |
|---------|----------|--------|
| V2 | Baseline features | 0.63 |
| V3 | 50× GP augmentation | 0.40 ❌ |
| V4 | 3× conservative augmentation | 0.618 |
| **V5** | **Golden features (no aug)** | **0.6361** |

---

## Installation

### Requirements
```bash
pip install numpy pandas scipy matplotlib scikit-learn lightgbm xgboost extinction tqdm
```

### Kaggle API Setup
```bash
pip install kaggle
mkdir -p ~/.kaggle
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Usage

### Option 1: Google Colab (Recommended)
1. Upload `MALLORN_v5_golden_features.ipynb` to Colab
2. Run cells sequentially
3. Upload `kaggle.json` when prompted
4. Download submission CSV

### Option 2: Local Execution
```bash
# Download competition data
kaggle competitions download -c mallorn-astronomical-classification-challenge

# Unzip data
unzip mallorn-astronomical-classification-challenge.zip -d data/

# Update DATA_PATH in notebook and run
jupyter notebook MALLORN_v5_golden_features.ipynb
```

---

## Project Evolution

```
V1 → V2 → V3 → V4 → V5
 │     │     │     │     └── Physics-based golden features ✓
 │     │     │     └── Conservative 3× augmentation (failed)
 │     │     └── Heavy 50× GP augmentation (failed badly)
 │     └── Baseline with standard features
 └── Initial exploration
```

### Key Insight
> *"Focus on light curve features that distinguish TDE from SN and AGN, especially the shape."*
> — Competition discussion (24th place solution)

---

## Key Lessons Learned

### ✅ What Worked
1. **Domain knowledge** beats brute-force ML approaches
2. **Physics-motivated features** (power-law decay, color ratios) are highly discriminative
3. **Ensemble methods** with threshold optimization for imbalanced data
4. **De-extinction correction** for accurate color measurements

### ❌ What Didn't Work
1. **Data augmentation** introduced harmful distribution shifts
2. **Synthetic lightcurves** didn't match test set characteristics
3. **Complex models** without physics features underperformed

### 💡 Feature Importance Insights
Top discriminative features tend to be:
- **Color ratios** (u/r, g/r) → TDEs are hot and blue
- **Asymmetry metrics** → TDEs have extended decays
- **Stochasticity measures** → TDEs are smooth single flares

---

## Acknowledgements

- **Competition**: [MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge) by Magill et al.
- **Data Source**: [Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu/)
- **Tools**: [SNCosmo](https://sncosmo.readthedocs.io/), Rubin Survey Simulator
- **References**: 
  - [TDE Overview - NASA](https://science.nasa.gov/universe/black-holes/)
  - [PLAsTiCC Challenge](https://www.kaggle.com/c/PLAsTiCC-2018) for methodology inspiration
  - Kyle Boone's "Avocado" solution for feature engineering ideas

### Funding Acknowledgements
- DM: Leverhulme Interdisciplinary Network on Algorithmic Solutions
- MN: European Research Council (ERC) Grant No. 948381

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Prepared for the Vera C. Rubin Observatory's LSST era 🔭</i>
</p>
