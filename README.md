# Computer Price Estimation & Device Classification

## Overview
This project is an end-to-end **Machine Learning system for computer price estimation and device form factor classification**. It integrates **Exploratory Data Analysis (EDA)**, advanced **feature engineering**, **benchmark-based hardware scoring**, **ensemble learning**, and **hierarchical classification** to produce realistic, scalable, and explainable predictions.

The system supports:
- Computer **price prediction**
- **Benchmark-based CPU & GPU scoring**
- **Form factor classification** (Laptop & Desktop sub-types)
- Rich **visualization and analysis**

---

## Project Scope

### 1. Exploratory Data Analysis (EDA)
The EDA phase focuses on understanding price drivers, feature distributions, correlations, and structural issues in the dataset.

**Key steps:**
- **Missing Value Handling**
  - Numeric features: Median imputation
  - Categorical features: Mode imputation

- **Outlier Analysis**
  - IQR-based filtering for extreme price values
  - Error distribution analysis to evaluate prediction stability

- **Visual Analysis**
  - Correlation heatmaps for numeric features
  - Price distribution histograms
  - Pair plots for hardware–price relationships
  - Box plots to identify hardware outliers

- **Statistical Testing**
  - ANOVA tests to measure categorical feature impact
  - Example: WIFI type showed no statistically significant effect on price (p-value ≈ 0.72)

**EDA Conclusion:**
- Hardware specifications (CPU, RAM, Storage) are strong price predictors
- Brand perception introduces premium pricing effects
- Linear models struggle with high-end devices, motivating non-linear approaches

---

### 2. Feature Engineering
Feature engineering is a core strength of this project and significantly improves model performance.

#### Display Features
- Resolution parsing (e.g., `1920x1080` → width & height)
- Pixel Per Inch (PPI) calculation using screen size

#### Hardware Normalization (Benchmarking)
Raw CPU and GPU model names were weakly correlated with price. To address this, **custom benchmark scores** were created.

##### CPU Benchmark Score
A unified numeric score combining:
- **Family** (e.g., i3 / i5 / i7 / i9, Ryzen tiers, Apple Silicon)
- **Generation** (parsed from model number)
- **Tier** (remaining digits)

**Example CPUs:**
- Intel i9-14373
- AMD Ryzen 9 7512
- Apple M3 Pro

##### GPU Benchmark Score
Similar methodology with GPU-specific adjustments:
- Architectural generation extraction
- Performance tier mapping
- Premium adjustments for high-end models

**Example GPUs:**
- RTX 4080 Ti
- RX 7800 XT
- Arc B770 Limited

#### Encoding
- One-Hot Encoding for categorical features
- CatBoost Encoding within stacking pipelines

---

### 3. Price Prediction

#### Dataset Preparation
- Train / Test split: **80% / 20%**
- Separate preprocessing pipelines for numeric and categorical features
- Log transformation applied to skewed price distributions

#### Models Trained
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor
- **Stacking Regressor (Final Model)**

**Stacking Configuration:**
- Base learners: Random Forest, XGBoost, LightGBM, CatBoost
- Final estimator: Linear Regression (stability and variance reduction)
- Hyperparameter tuning via GridSearchCV

The best-performing model was exported for use in the price prediction interface.

---

### 4. Benchmark-Based Prediction Improvement
Benchmark features significantly improved:
- Accuracy on premium devices
- Cross-brand generalization
- Reduction of brand-driven bias

A clear performance gain was observed when comparing predictions **before vs after feature engineering**.

---

### 5. Device Form Factor Classification

#### Problem Definition
Classify devices into detailed **form factors**, not just Laptop or Desktop.

**Examples:**
- **Laptops:** Gaming, Mainstream, Workstation, 2-in-1
- **Desktops:** Mini-ITX, Micro-ATX, ATX, Full-Tower

#### Challenges
- Class imbalance
- Visually and structurally similar mid-range categories

---

### 6. Hierarchical Stacking Classification (Final Model)

To address complexity and imbalance, a **Hierarchical Stacking Architecture** was implemented.

#### Why Hierarchical?
- Form factors are structured sub-categories
- Flat multi-class models underperform on minority classes
- Device-specific sub-models improve precision

#### Architecture
1. **Base Models**
   - Binary Random Forest classifiers trained per form factor
   - Oversampling applied to minority classes

2. **Meta-Features**
   - Probability outputs from base models
   - Represent confidence scores for each form factor

3. **Final Stacking Classifier**
   - Meta-features combined with engineered features
   - Multi-class prediction

This architecture improves both interpretability and robustness.

---

### 7. Model Performance Highlights
- Near-perfect performance on distinct desktop form factors (ATX, Full-Tower)
- Strong classification of Gaming and Mainstream laptops
- Remaining challenges in mid-range and hybrid categories

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, CatBoost, LightGBM
- Matplotlib, Seaborn

---

## Project Structure
computer-price-estimation/
├── data/
│ ├── raw/ # Original unprocessed dataset
│ └── processed/ # Cleaned and feature-engineered data
│
├── notebooks/
│ ├── eda.ipynb # Exploratory Data Analysis
│ ├── feature_engineering.ipynb
│ └── visualization.ipynb # Plots and analysis visuals
│
├── models/
│ ├── price_prediction/ # Trained regression models
│ └── form_factor_classification/ # Classification models
│
├── src/
│ ├── preprocessing.py # Data cleaning & encoding
│ ├── benchmark.py # CPU & GPU benchmark scoring
│ ├── train_price_model.py
│ └── train_classifier.py
│
├── README.md
└── requirements.txt



## Future Improvements
- Integration of real-world benchmark datasets
- Improved minority class recall using advanced resampling
- SHAP-based explainability
- Full-stack deployment

---

## Authors
**Computer Price Estimation Project**  
Machine Learning & Data Engineering focused academic project

---

## License
This project is intended for academic and portfolio use.
