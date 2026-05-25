# Explainable Machine Learning for Identifying Determinants of Corporate Unaccountability in Emerging Markets

This repository contains the implementation of an explainable machine learning framework developed to identify the key determinants of corporate unaccountability among firms listed on the Tehran Stock Exchange (TSE).

The study focuses on analyzing the factors associated with firms’ failure to appropriately respond to regulatory announcements issued by the securities regulator, within an emerging market context characterized by information asymmetry and governance heterogeneity.

---

## Research Objective

This study examines the financial reporting and corporate governance factors associated with corporate unaccountability in listed firms within an emerging market setting.
The focus is on interpreting model outputs to understand the main drivers of regulatory non-compliance.

---

## Data

The empirical analysis is based on a panel dataset of listed companies on the Tehran Stock Exchange covering the period 2017–2022 (708 firm-year observations across 118 firms).

The explanatory variables include:

- Financial reporting quality measures (e.g., earnings persistence, liquidity, conservatism)
- Corporate governance attributes
- Firm-specific characteristics (e.g., size, leverage, institutional structure)

---

## Methodology

The methodological framework integrates machine learning with modern explainability techniques:

### Data Processing
- Log transformation of skewed variables
- Winsorization of extreme values
- Target encoding for categorical variables
- Feature scaling using RobustScaler

### Feature Engineering
- Interaction terms between governance and financial variables
- Ratio-based constructs to capture relative effects

### Model Development
Three tree-based machine learning models are employed:

- Random Forest
- LightGBM
- CatBoost

### Feature Selection
- Recursive Feature Elimination with Cross-Validation (RFECV)
- Optimization based on ROC-AUC

### Class Imbalance Handling
- SMOTE oversampling
- Random undersampling

### Model Optimization
- GridSearchCV for hyperparameter tuning
- Probability calibration for improved probabilistic interpretation

---

## Explainability Framework

To ensure interpretability, SHAP (SHapley Additive exPlanations) is used to:

- Identify globally important predictors of unaccountability
- Quantify the marginal contribution of each variable
- Provide local-level explanations for individual predictions
- Compare the relative importance of financial vs. governance factors

This enables a shift from purely predictive modeling toward economically interpretable machine learning.

---

## Key Contribution

This study contributes to the literature in three ways:

1. **Empirical Contribution**  
   Identifies key determinants of corporate unaccountability in an emerging market context.

2. **Methodological Contribution**  
   Demonstrates the use of explainable machine learning (SHAP) as an alternative to traditional regression-based approaches.

3. **Policy Implication**  
   Provides evidence that regulators can improve enforcement efficiency by targeting firms with weaker earnings quality and governance structures.

---

## Machine Learning Models

- Random Forest
- LightGBM
- CatBoost

---

## Evaluation Metrics

Model performance is evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

A custom weighted metric is used to prioritize recall for the minority class.

---

## Repository Structure

```text
unaccountability/
│
├── unaccountability_explainable_ml.py
├── README.md
├── requirements.txt
└── outputs/
```

---

## Requirements

Main dependencies:

- Python 3.9+
- pandas
- numpy
- scikit-learn
- lightgbm
- catboost
- shap
- imbalanced-learn
- matplotlib
- seaborn

Install via:

```bash
pip install -r requirements.txt
```

---

## Notes

- This repository contains the full modeling and explainability pipeline.
- The study emphasizes model interpretability to understand the financial and governance-related drivers of corporate unaccountability.

---

## Author

Bahar Abedimanesh
