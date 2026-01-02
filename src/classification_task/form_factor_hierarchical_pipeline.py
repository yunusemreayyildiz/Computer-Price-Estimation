
import os, warnings

from metrics_exporter import exportMetrics
warnings.filterwarnings("ignore")
import seaborn as sns 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = 'computer_prices_all.csv'

print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Rows, cols:", df.shape)

# keep backup of original
df['orig_weight_kg'] = df.get('weight_kg', pd.Series([np.nan]*len(df)))

# --- Feature engineering: compute derived numeric features for modeling
def add_advanced_features(df):
    df = df.copy()
    for c in ['gpu_tier','battery_wh','psu_watts','charger_watts','display_size_in','refresh_hz','storage_gb','cpu_cores','cpu_boost_ghz','vram_gb']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    if 'resolution' in df.columns:
        def parse_res(r):
            try:
                w,h = str(r).split('x'); return int(w), int(h)
            except: return 0,0
        wh = df['resolution'].apply(parse_res)
        df['res_w'] = [t[0] for t in wh]; df['res_h'] = [t[1] for t in wh]
    else:
        df['res_w'] = 0; df['res_h'] = 0

    df['display_size_in'] = pd.to_numeric(df.get('display_size_in', 0)).fillna(0)
    mask = (df['display_size_in'] > 0) & (df['res_w'] > 0)
    df['ppi'] = 0.0
    df.loc[mask, 'ppi'] = ((df.loc[mask,'res_w']**2 + df.loc[mask,'res_h']**2)**0.5) / df.loc[mask,'display_size_in']

    df['weight_kg'] = pd.to_numeric(df.get('weight_kg', 0)).fillna(0)
    df['weight_log'] = np.log1p(df['weight_kg'])
    df['weight_per_inch'] = df['weight_kg'] / df['display_size_in'].replace({0:np.nan}).fillna(0)
    df['weight_per_cpu_core'] = df['weight_kg'] / df['cpu_cores'].replace({0:np.nan}).fillna(0)

    df['gpu_tier'] = pd.to_numeric(df.get('gpu_tier', 0)).fillna(0).astype(int)
    df['has_dedicated_gpu'] = (df['gpu_tier'] >= 3).astype(int)

    df['battery_wh'] = pd.to_numeric(df.get('battery_wh', 0)).fillna(0)
    df['battery_density'] = df['battery_wh'] / df['weight_kg'].replace({0:np.nan}).fillna(0)

    df['likely_touch'] = 0
    cond_touch = (df['display_size_in'] <= 14) & (df['weight_kg'] <= 1.6) & (df['device_type']=='Laptop')
    df.loc[cond_touch.fillna(False), 'likely_touch'] = 1

    if 'thickness_mm' not in df.columns:
        # proxy thickness to avoid zeros (estimate from weight and display size)
        df['thickness_mm'] = (df['weight_kg'] / (df['display_size_in'].replace(0,np.nan))).fillna(5.0)

    df['mobility_score'] = (1/(1+df['weight_kg'])) + (df['battery_wh']/1000) + df['likely_touch']
    df['performance_score'] = df['cpu_cores'] * df.get('cpu_boost_ghz',0) + df['gpu_tier'] * 8 + df.get('vram_gb',0) * 2

    for c in ['height_mm','width_mm','depth_mm']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    if all(c in df.columns for c in ['height_mm','width_mm','depth_mm']):
        df['volume_liters'] = (df['height_mm'] * df['width_mm'] * df['depth_mm']) / 1e6
    else:
        df['volume_liters'] = 0

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df

df = add_advanced_features(df)
print("Feature engineering done. Columns:", len(df.columns))

# drop identifiers
drop_cols = ['price','model','brand','release_year','cpu_model','gpu_model']
drop_cols_exist = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols_exist)

# label encode non-target object columns
for c in df.select_dtypes(include='object').columns:
    if c in ['form_factor','device_type']:
        continue
    df[c] = df[c].astype(str).fillna('__NA__')
    le = LabelEncoder(); df[c] = le.fit_transform(df[c])

# ensure numeric fill
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].fillna(0)

# Oversampler helper: try ADASYN, then SMOTE; return None if imblearn is unavailable
def get_oversampler():
    try:
        from imblearn.over_sampling import ADASYN
        return ADASYN(random_state=RANDOM_STATE)
    except Exception:
        try:
            from imblearn.over_sampling import SMOTE
            return SMOTE(random_state=RANDOM_STATE)
        except Exception:
            return None

oversampler = get_oversampler()
if oversampler is None:
    print("Warning: imblearn not installed. Oversampling disabled.")

# generic binary trainer
def train_binary(X, y, base_model=None, use_oversample=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
    )
    if use_oversample and oversampler is not None:
        try:
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
            print("Oversampling applied. New train shape:", X_train.shape)
        except Exception as e:
            print("Oversample failed:", e)
    if base_model is None:
        base_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    base_model.fit(X_train, y_train)
    y_pred = base_model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    return base_model, X_test, y_test, y_pred

# --- LAPTOP HIERARCHY
laptop_df = df[df['device_type']=='Laptop'].copy()
print("Laptop samples:", len(laptop_df))
feature_cols = [c for c in laptop_df.columns if c not in ['form_factor','device_type']]

print("\n=== Laptop Step A: 2-in-1 vs others ===")
y_A = (laptop_df['form_factor']=='2-in-1').astype(int)
X_A = laptop_df[feature_cols]
clf_2in1, XA_test, yA_test, yA_pred = train_binary(X_A, y_A, use_oversample=True)

print("\n=== Laptop Step B: Gaming vs others ===")
y_B = (laptop_df['form_factor']=='Gaming').astype(int)
X_B = laptop_df[feature_cols]
clf_gaming, XB_test, yB_test, yB_pred = train_binary(X_B, y_B, use_oversample=True)

print("\n=== Laptop Step C: Ultrabook vs others ===")
y_C = (laptop_df['form_factor']=='Ultrabook').astype(int)
clf_ultrabook, XC_test, yC_test, yC_pred = train_binary(X_B, y_C, use_oversample=True)

print("\n=== Laptop Step D: Workstation vs others ===")
y_D = (laptop_df['form_factor']=='Workstation').astype(int)
clf_work, XD_test, yD_test, yD_pred = train_binary(X_B, y_D, use_oversample=True)

# ---- Meta-feature builder for laptop (preserve original index then reset)
def meta_features_for_laptop(X_full):
    """
    X_full: laptop_df[feature_cols] (original index preserved)
    returns meta dataframe with SAME ROW COUNT (reset_index(drop=True) at end)
    """
    meta = pd.DataFrame(index=X_full.index)

    def get_prob_or_pred(clf, X):
        try:
            prob = clf.predict_proba(X)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                return pd.Series(prob[:,1], index=X.index)
            else:
                return pd.Series(clf.predict(X), index=X.index)
        except Exception:
            return pd.Series(clf.predict(X), index=X.index)

    meta['p_2in1'] = get_prob_or_pred(clf_2in1, X_full)
    meta['p_gaming'] = get_prob_or_pred(clf_gaming, X_full)
    meta['p_ultrabook'] = get_prob_or_pred(clf_ultrabook, X_full)
    meta['p_work'] = get_prob_or_pred(clf_work, X_full)

    extra_cols = ['weight_log','weight_kg','weight_per_inch','performance_score','ppi']
    extra_cols = [c for c in extra_cols if c in X_full.columns]
    extra = X_full[extra_cols].copy()

    # concat preserving original index alignment
    meta = pd.concat([meta, extra], axis=1)

    # drop duplicated index rows (safety) and then reset index to align with y later
    meta = meta.loc[~meta.index.duplicated(keep='first')].reset_index(drop=True)
    return meta

# build meta dataset and aligned y
X_meta = meta_features_for_laptop(laptop_df[feature_cols])
y_meta = laptop_df['form_factor'].reset_index(drop=True)

print("Shapes after fix -> X_meta:", X_meta.shape, "y_meta:", y_meta.shape)
if X_meta.shape[0] != y_meta.shape[0]:
    raise RuntimeError(f"Shape mismatch after meta creation: X_meta {X_meta.shape} vs y_meta {y_meta.shape}")

# Train stacking meta-classifier
print("Training stacking meta-classifier for laptop...")
base_learners = [('rf_meta', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))]
meta_clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_meta, y_meta, test_size=0.2, random_state=RANDOM_STATE, stratify=y_meta)
meta_clf.fit(Xm_train, ym_train)
ym_pred = meta_clf.predict(Xm_test)
print("Laptop final stacking accuracy:", accuracy_score(ym_test, ym_pred))
print(classification_report(ym_test, ym_pred, zero_division=0))
exportMetrics(ym_test, ym_pred, meta_clf.classes_, "StackingModel_Laptop")

# --- DESKTOP HIERARCHY
desk_df = df[df['device_type']=='Desktop'].copy()
print("\nDesktop samples:", len(desk_df))
feature_cols_d = [c for c in desk_df.columns if c not in ['form_factor','device_type']]

print("\n=== Desktop Step 1: ATX vs others ===")
y_d1 = (desk_df['form_factor']=='ATX').astype(int)
clf_atx, XD1_test, yD1_test, yD1_pred = train_binary(desk_df[feature_cols_d], y_d1, use_oversample=False)

print("\n=== Desktop Step 2: Full-Tower vs others ===")
y_d2 = (desk_df['form_factor']=='Full-Tower').astype(int)
clf_ft, XD2_test, yD2_test, yD2_pred = train_binary(desk_df[feature_cols_d], y_d2, use_oversample=True)
print("\n=== Desktop Step 3: Mini-ITX vs others (NEW) ===")
# Target: 1 for Mini-ITX, 0 for others
y_d3 = (desk_df['form_factor']=='Mini-ITX').astype(int)

# Special binary model for Mini-ITX (oversampling recommended due to low counts)
clf_mini, XD3_test, yD3_test, yD3_pred = train_binary(
    desk_df[feature_cols_d], 
    y_d3, 
    use_oversample=True # critical: oversample due to small class size
)
print("\n=== Desktop Step 4: SFF vs others (ADDED) ===")
# Target: 1 for SFF, 0 for others
y_d4 = (desk_df['form_factor']=='SFF').astype(int)

# Special binary model for SFF (oversampling required due to minority class)
clf_sff, XD4_test, yD4_test, yD4_pred = train_binary(
    desk_df[feature_cols_d], 
    y_d4, 
    use_oversample=True 
)
def meta_features_for_desktop(X_full):
    meta = pd.DataFrame(index=X_full.index)
    def gp(clf, X):
        try:
            p = clf.predict_proba(X)
            if p.ndim==2 and p.shape[1]>=2:
                return pd.Series(p[:,1], index=X.index)
            else:
                return pd.Series(clf.predict(X), index=X.index)
        except Exception:
            return pd.Series(clf.predict(X), index=X.index)
    meta['p_atx'] = gp(clf_atx, X_full)
    meta['p_ft'] = gp(clf_ft, X_full)
    meta['p_mini'] = gp(clf_mini, X_full)
    meta['p_sff'] = gp(clf_sff, X_full)
    extra_cols = ['weight_log','weight_kg','weight_per_inch','volume_liters','psu_watts']
    extra_cols = [c for c in extra_cols if c in X_full.columns]
    extra = X_full[extra_cols].copy()
    meta = pd.concat([meta, extra], axis=1)
    meta = meta.loc[~meta.index.duplicated(keep='first')].reset_index(drop=True)
    return meta

X_meta_d = meta_features_for_desktop(desk_df[feature_cols_d])
y_meta_d = desk_df['form_factor'].reset_index(drop=True)
print("Shapes after fix (desktop) ->", X_meta_d.shape, y_meta_d.shape)
if X_meta_d.shape[0] != y_meta_d.shape[0]:
    raise RuntimeError("Desktop meta / y mismatch")

print("Training stacking meta-classifier for desktop...")
meta_clf_d = StackingClassifier(estimators=[('rf_d', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))],
                                 final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1)
Xm_train_d, Xm_test_d, ym_train_d, ym_test_d = train_test_split(X_meta_d, y_meta_d, test_size=0.2, random_state=RANDOM_STATE, stratify=y_meta_d)
meta_clf_d.fit(Xm_train_d, ym_train_d)
ym_pred_d = meta_clf_d.predict(Xm_test_d)
print("Desktop final stacking accuracy:", accuracy_score(ym_test_d, ym_pred_d))
print(classification_report(ym_test_d, ym_pred_d, zero_division=0))
exportMetrics(ym_test_d, ym_pred_d, meta_clf_d.classes_, "StackingModel_Desktop")

# Save models optionally (saving block removed)
# The code block for persisting models (joblib.dump) has been removed intentionally.
# If you want to enable saving, re-add a try/except block that imports joblib
# and calls joblib.dump(...) for each trained model.
def plot_confusion_matrix(y_true, y_pred, labels, title, ax):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels, 
        yticklabels=labels, 
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Actual Value")

# --- Confusion matrix visualization ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
plt.subplots_adjust(wspace=0.3)

# 1. Laptop Final Stacking CM
plot_confusion_matrix(
    ym_test, 
    ym_pred, 
    labels=meta_clf.classes_, 
    title=f"Laptop Stacking CM (Acc: {accuracy_score(ym_test, ym_pred):.3f})", 
    ax=axes[0]
)

# 2. Desktop Final Stacking CM
plot_confusion_matrix(
    ym_test_d, 
    ym_pred_d, 
    labels=meta_clf_d.classes_, 
    title=f"Desktop Stacking CM (Acc: {accuracy_score(ym_test_d, ym_pred_d):.3f})", 
    ax=axes[1]
)

plt.show()
def plot_feature_importance(model, feature_names, title):
    # If the model is a StackingClassifier, focus on its primary estimator (e.g., 'rf_meta')
    if isinstance(model, StackingClassifier):
        # Take the first estimator inside the StackingClassifier (assumed to be a RandomForest)
        estimator = model.named_estimators_[list(model.named_estimators_.keys())[0]]
    else:
        # If a single RandomForest model is provided, use it directly
        estimator = model

    # Extract RandomForest feature importances
    importances = estimator.feature_importances_

    # Build a Series of feature names and their importance, keep top 20
    feature_importances = pd.Series(importances, index=feature_names)
    feature_importances = feature_importances.sort_values(ascending=False).head(20)

    # Visualize the top features as a horizontal bar chart
    plt.figure(figsize=(10, 6))
    feature_importances.sort_values().plot(kind='barh', color='skyblue')
    plt.title(title)
    plt.xlabel('Feature Importance Score')
    plt.show()

# ----------------------------------------------------
# A. Laptop Feature Importance
# ----------------------------------------------------
# X_meta'nın sütun isimleri meta-özellikler ve ek özellikleri içerir
laptop_feature_names = X_meta.columns.tolist() 

print("\n--- Laptop Stacking Model (RF_meta) Feature Importance ---")
plot_feature_importance(meta_clf, laptop_feature_names, 
                        'Laptop Stacking (RF Estimator) Feature Importance')

# ----------------------------------------------------
# B. Desktop Feature Importance
# ----------------------------------------------------
desktop_feature_names = X_meta_d.columns.tolist()

print("\n--- Desktop Stacking Model (RF_d) Feature Importance ---")
plot_feature_importance(meta_clf_d, desktop_feature_names, 
                        'Desktop Stacking (RF Estimator) Feature Importance')

print("\nPipeline finished. Models trained but not saved to disk.")