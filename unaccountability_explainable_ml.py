# --- Section 0: Library Imports and Environment Setup ---

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import warnings
import logging

from scipy.stats import mstats

# Model training, validation, and evaluation tools
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, recall_score, auc, make_scorer,
    precision_score, roc_curve
)

# Preprocessing and feature selection methods
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV

# Machine learning models
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

# Probability calibration and imbalanced data handling
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Model interpretability
import shap


# Configure logging and suppress unnecessary warnings
logging.basicConfig(
    filename='model_training.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Visualization settings
plt.rcParams['font.family'] = 'DejaVu Sans'

# Temporary directory for CatBoost processing files
temp_dir = tempfile.mkdtemp()
os.environ["CATBOOST_TEMP_DIR"] = temp_dir
# --- Section 1: Data Loading and Initial Preparation ---

try:
    # Load the dataset from the local working directory
    file_path = "thesis dataset new.xlsx"
    df = pd.read_excel(file_path)
    print("✅ Dataset loaded successfully.")

except FileNotFoundError:
    error_msg = (
        "Dataset file not found. Please upload the dataset "
        "and verify the file path."
    )
    logging.error(error_msg)
    print(f"❌ ERROR: {error_msg}")
    exit()

# Create output directory for storing model results and visualizations
output_dir = "final_result_14-06"
os.makedirs(output_dir, exist_ok=True)

# Remove observations with missing values
df = df.dropna()

# Define target variable
TARGET = 'UNACC'


# --- Section 2: Feature Engineering and Data Transformation ---

# Main explanatory variables used for correlation analysis
main_features_list = [
    'CON', 'UNCON', 'EP', 'PV', 'AQ',
    'EL', 'SIZE', 'INST', 'LEV', 'IC'
]


# Log transformation and winsorization for skewed continuous variables
for col in ['CON', 'UNCON', 'EL', 'LEV', 'SIZE', 'EP', 'AQ', 'PV']:

    # Shift variables containing non-positive values before log transformation
    if df[col].min() <= 0:
        df[col] = df[col] - df[col].min() + 1e-6

    # Apply logarithmic transformation
    df[col] = np.log1p(df[col])

    # Winsorize extreme observations at the 5th and 95th percentiles
    df[col] = mstats.winsorize(df[col], limits=[0.05, 0.05])


# Target encoding for categorical variable 'BC'
bc_target_map = df.groupby('BC')[TARGET].mean()
df['BC_target_encoded'] = df['BC'].map(bc_target_map)

# Remove original categorical feature after encoding
df = df.drop('BC', axis=1)

main_features_list.append('BC_target_encoded')


# Separate predictors and target variable
X_pre_scaling = df.drop(TARGET, axis=1).select_dtypes(include=np.number).copy()
y = df[TARGET]


# Interaction features
X_pre_scaling['CON_EP_interaction'] = (
    X_pre_scaling['CON'] * X_pre_scaling['EP']
)

X_pre_scaling['IC_CON_interaction'] = (
    X_pre_scaling['IC'] * X_pre_scaling['CON']
)

X_pre_scaling['SIZE_LEV_interaction'] = (
    X_pre_scaling['SIZE'] * X_pre_scaling['LEV']
)

X_pre_scaling['INST_IC_interaction'] = (
    X_pre_scaling['INST'] * X_pre_scaling['IC']
)


# Ratio-based features
epsilon = 1e-6

X_pre_scaling['CON_div_SIZE'] = (
    X_pre_scaling['CON'] / (X_pre_scaling['SIZE'] + epsilon)
)

X_pre_scaling['LEV_div_SIZE'] = (
    X_pre_scaling['LEV'] / (X_pre_scaling['SIZE'] + epsilon)
)

X_pre_scaling['INST_div_SIZE'] = (
    X_pre_scaling['INST'] / (X_pre_scaling['SIZE'] + epsilon)
)

X_pre_scaling['EP_div_AQ'] = (
    X_pre_scaling['EP'] / (X_pre_scaling['AQ'] + epsilon)
)

X_pre_scaling['CON_div_UNCON'] = (
    X_pre_scaling['CON'] / (X_pre_scaling['UNCON'] + epsilon)
)

X_pre_scaling['PV_div_AQ'] = (
    X_pre_scaling['PV'] / (X_pre_scaling['AQ'] + epsilon)
)


# Replace infinite values and remaining missing values
X_pre_scaling.replace([np.inf, -np.inf], np.nan, inplace=True)
X_pre_scaling.fillna(0, inplace=True)

print("✅ Feature engineering completed successfully.")


# --- Correlation Analysis and Visualization ---

main_df_corr = pd.concat([X_pre_scaling[main_features_list], y], axis=1)

plt.figure(figsize=(12, 8))

main_df_corr.corr()[TARGET] \
    .drop(TARGET) \
    .sort_values(ascending=False) \
    .plot(kind='bar', color='skyblue')

plt.title(
    f'Correlation of Main Features with Target ({TARGET})',
    fontsize=16
)

plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "main_features_correlation.png"),
    dpi=300
)
# --- Section 3: Feature Scaling ---

# Scale features using RobustScaler to reduce sensitivity to outliers
scaler = RobustScaler()

X_scaled = scaler.fit_transform(X_pre_scaling)

X = pd.DataFrame(
    X_scaled,
    columns=X_pre_scaling.columns
)

print("✅ Feature scaling completed successfully.")


# --- Section 4: Train-Test Split ---

# Split the dataset while preserving class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --- Section 5: SHAP Analysis Before Feature Selection ---

# Train an initial Random Forest model for global SHAP analysis
print("🔍 Performing SHAP analysis on all engineered features...")

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

# Extract SHAP values for the positive class in binary classification
shap_values_class1 = shap_values[:, :, 1]

shap_df = pd.DataFrame(
    shap_values_class1,
    columns=X_test.columns
)

# Create output directory for SHAP visualizations
output_dir_all_features = os.path.join(
    output_dir,
    "All_Features_SHAP"
)

os.makedirs(output_dir_all_features, exist_ok=True)


# Generate global SHAP importance plot
plt.figure()

shap.summary_plot(
    shap_values_class1,
    X_test,
    plot_type="bar",
    show=False
)

plt.title(
    'SHAP Global Importance (All Features)',
    fontsize=12
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir_all_features,
        "shap_global_bar_all_features.png"
    ),
    dpi=300
)

plt.close()

print("✅ SHAP global importance plot saved successfully.")


# Compute mean absolute SHAP values for feature importance ranking
feature_importance = np.abs(shap_values_class1).mean(axis=0)

feature_importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean_SHAP': feature_importance
})

print(
    "Feature Importance (Mean SHAP Values):\n",
    feature_importance_df.sort_values(
        by='Mean_SHAP',
        ascending=False
    )
)


# --- Section 6: Feature Selection Using RFECV ---

print("🚀 Starting recursive feature elimination with cross-validation...")

rfe_estimator = LGBMClassifier(
    random_state=42,
    verbose=-1
)

rfecv = RFECV(
    estimator=rfe_estimator,
    step=1,
    cv=StratifiedKFold(3),
    scoring='roc_auc',
    min_features_to_select=10,
    n_jobs=-1
)

rfecv.fit(X_train, y_train)

selected_features = X_train.columns[
    rfecv.support_
].tolist()

print(
    f"✅ RFECV selected {len(selected_features)} optimal features:\n",
    selected_features
)

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]


# --- Section 7: Resampling Pipeline for Class Imbalance ---

resampling_pipeline = ImbPipeline([
    ('smote', SMOTE(
        sampling_strategy=0.5,
        random_state=42
    )),
    ('under', RandomUnderSampler(
        sampling_strategy=1.0,
        random_state=42
    ))
])

X_train_resampled, y_train_resampled = (
    resampling_pipeline.fit_resample(
        X_train_selected,
        y_train
    )
)

print(
    "Class distribution after resampling:\n",
    y_train_resampled.value_counts()
)


# --- Section 8: Custom Evaluation Metric ---

# Define a weighted scoring metric emphasizing recall performance
def combined_score(y_true, y_pred):

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    return 0.60 * recall + 0.40 * f1


combined_scorer = make_scorer(
    combined_score,
    greater_is_better=True
)


# --- Section 9: Model Definitions and Hyperparameter Grids ---

models = {

    'LightGBM': {
        'base_model': LGBMClassifier(
            n_estimators=2000,
            random_state=42,
            verbose=-1
        ),

        'params': {
            'learning_rate': [0.01, 0.05],
            'max_depth': [7, 10],
            'reg_lambda': [0.1, 1.0],
            'num_leaves': [20, 31, 40]
        }
    },

    'CatBoost': {
        'base_model': CatBoostClassifier(
            random_state=42,
            verbose=0
        ),

        'params': {
            'learning_rate': [0.01, 0.05],
            'depth': [6, 8],
            'l2_leaf_reg': [1, 3],
            'iterations': [1000, 2000]
        }
    },

    'RandomForest': {
        'base_model': RandomForestClassifier(
            random_state=42
        ),

        'params': {
            'n_estimators': [300, 500],
            'max_depth': [10, None],
            'min_samples_leaf': [2, 4],
            'criterion': ['entropy']
        }
    }
}
plt.close()
