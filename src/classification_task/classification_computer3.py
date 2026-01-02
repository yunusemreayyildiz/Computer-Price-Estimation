"""
Form factor classification notebook (single-file runnable script)

Place `computer_prices_all.csv` in the same folder (the dataset you uploaded is available at
`/mnt/data/computer_prices_all.csv` when running inside this environment).

This script does the following:
- Loads data
- Replaces/synthesizes weight_kg using a truncated-normal generator (improved over uniform)
- Performs simple feature engineering
- Encodes categorical features safely
- Trains RandomForest classifiers separately for Laptop and Desktop
- Prints classification reports, shows confusion matrices and feature importances
- Includes single-feature baseline tests (only weight)

Optional parts (SMOTE, SHAP) are guarded by try/except so script still runs if packages
are not installed.
"""

import os
import warnings

from metrics_exporter import exportMetrics
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DATA_PATH = 'computer_prices_all.csv'
if not os.path.exists(DATA_PATH):
    # fallback to path used in this environment
    DATA_PATH = '/mnt/data/computer_prices_all.csv'
# Load
print('Loading:', DATA_PATH)
df = pd.read_csv(DATA_PATH)
print('Rows, cols:', df.shape)
print(df[['device_type','form_factor','weight_kg']].head())

# keep original weight
if 'weight_kg' in df.columns:
    df['orig_weight_kg'] = df['weight_kg'].copy()
else:
    df['orig_weight_kg'] = np.nan

# Improved weight generator (truncated normal + small feature adjustments)
WEIGHT_PARAMS = {
    'Ultrabook':    (1.25, 0.18, 0.9, 1.6),
    '2-in-1':       (1.35, 0.20, 0.9, 1.8),
    'Mainstream':   (1.85, 0.35, 1.2, 2.8),
    'Gaming':       (3.00, 0.60, 2.0, 5.5),
    'Workstation':  (3.80, 0.80, 2.0, 7.0),
    'Mini-ITX':     (3.00, 1.00, 1.0, 7.0),
    'SFF':          (5.50, 1.50, 2.5, 10.0),
    'Micro-ATX':    (8.50, 2.00, 4.0, 13.0),
    'ATX':          (12.0, 3.00, 6.0, 22.0),
    'Full-Tower':   (18.0, 5.00, 10.0, 40.0),
}
def add_advanced_features(df):
    # Ultra-light notebook flag -> 2-in-1 and ultrabook
    df['is_ultra_light'] = (df['weight_kg'] < 1.4).astype(int)

    # GPU and weight interaction
    df['gpu_density'] = df['weight_kg'] * df['has_dedicated_gpu']

    # Weight per core
    df['density_per_core'] = df['weight_kg'] / df['cpu_cores'].replace(0, np.nan)

    # Estimated touch capability (synthetic)
    df['likely_touch'] = ((df['display_size_in'] <= 14) &
                          (df['weight_kg'] <= 1.6)).astype(int)

    # Likelihood of being a 2-in-1 -> low CPU tier + low weight
    df['likely_2in1'] = ((df['likely_touch'] == 1) &
                         (df['cpu_tier'] <= 3)).astype(int)

    df.fillna(0, inplace=True)
    return df

def truncated_normal(mean, std, low, high):
    val = np.random.normal(loc=mean, scale=std)
    return float(np.clip(val, low, high))


def infer_psu_type(psu_watts):
    try:
        pw = float(psu_watts)
    except Exception:
        return 'unknown'
    if pw >= 700:
        return 'heavy'
    elif pw >= 450:
        return 'medium'
    elif pw > 0:
        return 'light'
    else:
        return 'none'


def improved_fix_weight(row):
    ff = row.get('form_factor', None)
    if ff in WEIGHT_PARAMS:
        mean, std, low, high = WEIGHT_PARAMS[ff]
        w = truncated_normal(mean, std, low, high)
    else:
        w = float(np.clip(np.random.normal(2.0, 1.5), 0.3, 60.0))

    # gpu_tier based adjustment
    try:
        gpu_tier = row.get('gpu_tier', np.nan)
        if not pd.isna(gpu_tier):
            if float(gpu_tier) >= 3:
                w += 0.9
            elif float(gpu_tier) >= 2:
                w += 0.5
    except Exception:
        pass

    # large battery
    try:
        battery_wh = row.get('battery_wh', np.nan)
        if not pd.isna(battery_wh) and float(battery_wh) >= 60:
            w += 0.4
    except Exception:
        pass

    # psu effect
    try:
        psu_watts = row.get('psu_watts', np.nan)
        psu_type = infer_psu_type(psu_watts)
        if psu_type == 'heavy':
            w += 1.0
        elif psu_type == 'medium':
            w += 0.5
    except Exception:
        pass

    w = float(np.clip(w, 0.3, 100.0))
    return w

# apply reproducibly
np.random.seed(RANDOM_STATE)
df['weight_kg'] = df.apply(improved_fix_weight, axis=1)

# Simple feature engineering
# numeric conversions
for col in ['gpu_tier','battery_wh','psu_watts','display_size_in','refresh_hz','storage_gb']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# derived features
df['weight_log'] = np.log1p(df['weight_kg'])
df['has_dedicated_gpu'] = (pd.to_numeric(df['gpu_tier'], errors='coerce').fillna(0).astype(int) >= 3).astype(int)
df['large_battery'] = (pd.to_numeric(df['battery_wh'], errors='coerce').fillna(0) >= 60).astype(int)
df['psu_type'] = df['psu_watts'].apply(infer_psu_type) if 'psu_watts' in df.columns else 'unknown'

# interaction
if 'display_size_in' in df.columns:
    df['portability_index'] = df['weight_kg'] / df['display_size_in'].replace({0: np.nan})
else:
    df['portability_index'] = 0

# fillna for safe encoding
# drop some columns that are clearly non-features
drop_cols = ['price','model','brand','release_year','cpu_model','gpu_model']
drop_cols_existing = [c for c in drop_cols if c in df.columns]
df_clean = df.drop(columns=drop_cols_existing)
df_clean = add_advanced_features(df_clean)
# label encode non-target object columns (except form_factor, device_type)
le_dict = {}
for col in df_clean.select_dtypes(include='object').columns:
    if col in ['form_factor','device_type']:
        continue
    df_clean[col] = df_clean[col].fillna('__NA__').astype(str)
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    le_dict[col] = le

# ensure numeric NaNs replaced
num_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[num_cols] = df_clean[num_cols].fillna(0)

print('\nPrepared features sample:')
print(df_clean.head())

# function to train and report
from collections import namedtuple
Result = namedtuple('Result', ['clf','X_test','y_test','y_pred','fi'])


def train_and_evaluate(df_in, device_type_label='Laptop'):
    print(f"\n=== TRAIN FOR {device_type_label.upper()} ===")
    sub = df_in[df_in['device_type'] == device_type_label].copy()
    if sub.shape[0] < 50:
        print('Not enough samples')
        return None

    X = sub.drop(columns=['form_factor','device_type'])
    y = sub['form_factor']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print('\nTop features:')
    print(fi.head(12))

    return Result(clf=clf, X_test=X_test, y_test=y_test, y_pred=y_pred, fi=fi)

# train laptop and desktop
res_lap = train_and_evaluate(df_clean, 'Laptop')
res_desk = train_and_evaluate(df_clean, 'Desktop')

# export model metrics
exportMetrics(res_lap.y_test, res_lap.y_pred, res_lap.clf.classes_, "RandomForestModel2_Laptop")
exportMetrics(res_desk.y_test, res_desk.y_pred, res_desk.clf.classes_, "RandomForestModel2_Desktop")

# --- Confusion matrices
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

if res_lap is not None:
    cm = confusion_matrix(res_lap.y_test, res_lap.y_pred, labels=res_lap.clf.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=res_lap.clf.classes_, yticklabels=res_lap.clf.classes_, cmap='Blues')
    plt.title('Laptop confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if res_desk is not None:
    cm = confusion_matrix(res_desk.y_test, res_desk.y_pred, labels=res_desk.clf.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=res_desk.clf.classes_, yticklabels=res_desk.clf.classes_,cmap='Blues')
    plt.title('Desktop confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# --- Single feature baseline test (only weight)
def single_feature_test(df_in, device_type_label='Laptop', feature='weight_kg'):
    print(f"\n--- Single feature test ({device_type_label}) using '{feature}' ---")
    sub = df_in[df_in['device_type'] == device_type_label].copy()
    X = sub[[feature]].copy()
    y = sub['form_factor']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy (single feature):', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

single_feature_test(df_clean, 'Laptop', 'weight_kg')
single_feature_test(df_clean, 'Desktop', 'weight_kg')

# --- Optional: SMOTE (if imblearn available)
try:
    from imblearn.over_sampling import SMOTE
    print('\nSMOTE available: running a small over-sampling test for Laptop')
    sub = df_clean[df_clean['device_type']=='Laptop'].copy()
    X = sub.drop(columns=['form_factor','device_type'])
    y = sub['form_factor']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE,stratify=y)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    clf2 = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    clf2.fit(X_res, y_res)
    y_pred2 = clf2.predict(X_test)
    print('SMOTE trial accuracy:', accuracy_score(y_test, y_pred2))
    print(classification_report(y_test, y_pred2, zero_division=0))
except Exception as e:
    print('\nSMOTE not run (imblearn not installed?):', e)

# SHAP explainability 
try:
    import shap
    print('\nSHAP available: computing summary for laptop model')
    explainer = shap.TreeExplainer(res_lap.clf)
    shap_values = explainer.shap_values(res_lap.X_test)
    cls_idx = list(res_lap.clf.classes_).index('2-in-1') if '2-in-1' in res_lap.clf.classes_ else 0
    shap.summary_plot(shap_values[cls_idx], res_lap.X_test)
except Exception as e:
    print('\nSHAP summary not produced (shap might not be installed):', e)

print('\nAll done.')
