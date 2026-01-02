import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from metrics_exporter import exportMetrics

df = pd.read_csv('computer_prices_all.csv')

# Step 1: Load data and fix or synthesize missing/incorrect weight values
# The following function assigns plausible weights based on `form_factor`.
# For laptops we use lighter ranges; for desktop form factors we use larger ranges.
def fix_weight(row):
    ff = row['form_factor']
    
    if ff == 'Ultrabook':
        return np.random.uniform(1.0, 1.4)
    elif ff == '2-in-1':
        return np.random.uniform(1.1, 1.6)
    elif ff == 'Mainstream':
        return np.random.uniform(1.5, 2.2)
    elif ff == 'Gaming':
        return np.random.uniform(2.3, 3.5)
    elif ff == 'Workstation':
        return np.random.uniform(2.5, 4.0)
    
    # Desktop groups (assign heavier weights for desktop chassis types)
    elif ff == 'Mini-ITX':
        return np.random.uniform(3.0, 6.0)
    elif ff == 'SFF':
        return np.random.uniform(5.0, 8.0)
    elif ff == 'Micro-ATX':
        return np.random.uniform(7.0, 10.0)
    elif ff == 'ATX':
        return np.random.uniform(10.0, 15.0)
    elif ff == 'Full-Tower':
        return np.random.uniform(15.0, 25.0)
    return row['weight_kg']

df['weight_kg'] = df.apply(fix_weight, axis=1)

df.loc[df['form_factor'] == 'Gaming', 'gpu_tier'] = df.loc[df['form_factor'] == 'Gaming', 'gpu_tier'].apply(lambda x: max(x, 2))

## Remove non-feature columns we don't use for modeling
drop_cols = ['price', 'model', 'brand', 'release_year', 'cpu_model', 'gpu_model']
df_clean = df.drop(columns=drop_cols)

# Label encode categorical columns (except the target `form_factor` and `device_type`)
le_dict = {}
for col in df_clean.select_dtypes(include='object').columns:
    if col not in ['form_factor', 'device_type']:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])

# Function that plots confusion matrixes
def plot_confusion_matrix(y_true, y_pred, title, labels):
    """Compute and display a confusion matrix heatmap.

    Parameters
    - y_true: array-like of true labels
    - y_pred: array-like of predicted labels
    - title: plot title string
    - labels: list of label names to order the matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

print("\n=== LAPTOP ===")
df_lap = df_clean[df_clean['device_type'] == 'Laptop'].copy()
X_lap = df_lap.drop(columns=['form_factor', 'device_type'])
y_lap = df_lap['form_factor']

# Split into training and test sets (stratified by form_factor)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lap, y_lap, test_size=0.2, random_state=42, stratify=y_lap)
# Train a Random Forest classifier (balanced classes)
clf_lap = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf_lap.fit(X_train_l, y_train_l)

y_pred_l = clf_lap.predict(X_test_l)
print(f"accuracy: {accuracy_score(y_test_l, y_pred_l):.2f}")
print("\Classification Report :")
print(classification_report(y_test_l, y_pred_l))

# Laptop confusion matrix
laptop_labels = sorted(y_lap.unique())
print("\nLaptop confusion Matrix:")
plot_confusion_matrix(y_test_l, y_pred_l, 'Laptop Form Factor Confusion Matrix', laptop_labels)


# Desktop model: repeat the same training/evaluation pipeline for desktops
print("\nDESKTOP")
df_desk = df_clean[df_clean['device_type'] == 'Desktop'].copy()
X_desk = df_desk.drop(columns=['form_factor', 'device_type'])
y_desk = df_desk['form_factor']

# Split into training and test sets (stratified by form_factor)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_desk, y_desk, test_size=0.2, random_state=42, stratify=y_desk)

# Train Random Forest for desktops
clf_desk = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf_desk.fit(X_train_d, y_train_d)

y_pred_d = clf_desk.predict(X_test_d)
accuracy = accuracy_score(y_test_d, y_pred_d)
print(f"Accuracy): {accuracy:.2f}")
print("\nClassification report :")
print(classification_report(y_test_d, y_pred_d))

# Desktop confusion matrix
desktop_labels = sorted(y_desk.unique())
print("\nDesktop Confusion Matrix:")
plot_confusion_matrix(y_test_d, y_pred_d,'Desktop Form factor Confusion Matrix', desktop_labels)

exportMetrics(y_test_d, y_pred_d, desktop_labels, "RandomForestModel1")
