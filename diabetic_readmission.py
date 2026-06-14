# ==========================================
# 1. Imports and Setup
# ==========================================
!pip install ucimlrepo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             precision_recall_curve, classification_report)

# ==========================================
# 2. Data Loading
# ==========================================
print("⏳ Fetching data from the official source...")
dataset = fetch_ucirepo(id=296)

# Combine features and targets into a single DataFrame
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

print("✅ Data loaded successfully!")
print(f"Number of Rows: {df.shape[0]} | Number of Columns: {df.shape[1]}")

# ==========================================
# 3. Data Preprocessing & Feature Engineering
# ==========================================
# Fill missing numeric values with the column mean
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill missing categorical values with 'Unknown'
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna('Unknown')

# Feature Engineering: Create 'treatment_type' based on medication logic
oral_user = (df['diabetesMed'] == 'Yes') & (df['insulin'] == 'No')
insulin_user = (df['diabetesMed'] == 'Yes') & (df['insulin'] != 'No')
no_trt = (df['diabetesMed'] == 'No')

conditions = [oral_user, insulin_user, no_trt]
labels = ['oral_user', 'insulin_user', 'no_trt']

df['treatment_type'] = np.select(conditions, labels, default='other')

# Drop unnecessary columns (weight is mostly missing in this dataset)
df.drop(columns=['weight'], inplace=True, errors='ignore')

# ==========================================
# 4. Prepare Data for Modeling
# ==========================================
# Isolate the target variable and convert it to a binary integer (1 for <30 days, 0 otherwise)
y = (df['readmitted'] == '<30').astype(int)

# Prepare features (drop the target column and the 'race' column)
X_raw = df.drop(['readmitted', 'race'], axis=1, errors='ignore')

# Convert categorical text data into numeric format (One-Hot Encoding)
X = pd.get_dummies(X_raw, drop_first=True)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. Model Training
# ==========================================
# Build the Random Forest model with robust settings for imbalanced data
rf_fix = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced_subsample', # Balances classes by adjusting weights inversely proportional to class frequencies
    max_depth=10,                      # Limits tree depth to prevent overfitting (memorizing the data)
    random_state=42
)

# Train the model on the raw (unscaled) training data
rf_fix.fit(X_train, y_train)

# ==========================================
# 6. Evaluation & Visualizations
# ==========================================
y_pred_fix = rf_fix.predict(X_test)
print("--- Standard Threshold (0.5) ---")
print(classification_report(y_test, y_pred_fix))

# 1. Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_fix)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Readmit', 'Readmit <30'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix: Hospital Readmission Prediction')
plt.show()

# 2. Plot Feature Importance (Top 10 Predictors)
importances = rf_fix.feature_importances_
feature_names = X.columns 

feat_importances = pd.Series(importances, index=feature_names)
top_10_features = feat_importances.nlargest(10)

plt.figure(figsize=(10, 6))
top_10_features.plot(kind='barh', color='skyblue')
plt.title('Top 10 Predictors for Patient Readmission')
plt.xlabel('Relative Importance Score')
plt.ylabel('Clinical Features')
plt.gca().invert_yaxis() # Invert y-axis to display the most important feature at the top
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Plot the Precision-Recall Curve
plt.figure(figsize=(8, 6))
prec, rec, tre = precision_recall_curve(y_test, rf_fix.predict_proba(X_test)[:,1])
plt.plot(rec, prec, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Crucial for Imbalanced Data)')
plt.show()

# ==========================================
# 7. Clinical Optimization (Custom Threshold)
# ==========================================
# Get the predicted probabilities instead of the final binary prediction
y_probs = rf_fix.predict_proba(X_test)[:, 1]

# Lower the threshold to make the model more "cautious" and catch more true positives
custom_threshold = 0.45
y_pred_custom = (y_probs >= custom_threshold).astype(int)

print(f"\\n--- Results with Custom Threshold ({custom_threshold}) ---")
print(classification_report(y_test, y_pred_custom))

# Plot the new confusion matrix to visualize the reduction in false negatives
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_custom), annot=True, fmt='d', cmap='Greens')
plt.title('Optimized Confusion Matrix (Reduced False Negatives)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
