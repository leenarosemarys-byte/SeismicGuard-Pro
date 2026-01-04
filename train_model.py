import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the data
# Assuming your file is named 'earthquake_data.csv'
df = pd.read_csv('earthquake_data.csv')

# 2. Preprocessing
# Map colors to numbers
alert_mapping = {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}
df['alert'] = df['alert'].map(alert_mapping)

# Keep ONLY your requested 5 features
features = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']
X = df[features]
y = df['alert']

# Drop any rows with missing values
X = X.dropna()
y = y.loc[X.index]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scaling (Essential for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("--- TRAINING BASELINE (Logistic Regression) ---")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
print(f"Baseline Accuracy: {accuracy_score(y_test, lr.predict(X_test_scaled)):.4f}")

print("\n--- TRAINING ADVANCED (Random Forest) ---")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_
print(f"Advanced RF Accuracy: {accuracy_score(y_test, best_rf.predict(X_test_scaled)):.4f}")

# 4. SAVE RESOURCES FOR UI
if not os.path.exists('output'):
    os.makedirs('output')

with open('output/final_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

with open('output/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names to ensure order is preserved in UI
with open('output/feature_names.pkl', 'wb') as f:
    pickle.dump(features, f)

print("\nSuccess: Model and Scaler saved in /output folder.")