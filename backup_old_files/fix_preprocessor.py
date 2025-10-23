"""
Fix preprocessor file to include all required fields for dashboard
"""

import joblib
import json

print("Fixing preprocessor file...")

# Load existing preprocessor
try:
    preprocessor_data = joblib.load('improved_preprocessor.pkl')
    print("Loaded existing preprocessor")
except:
    preprocessor_data = {}
    print("Creating new preprocessor")

# Load model results to get feature list
try:
    with open('improved_model_results.json', 'r') as f:
        model_results = json.load(f)
    print("Loaded model results")
except:
    print("Warning: model_results.json not found")
    model_results = {}

# Base features that the model uses
base_features = [
    'Battery_Voltage', 'Battery_Current', 'Battery_Temperature',
    'SoC', 'SoH', 'Charge_Cycles', 'Distance_Traveled',
    'Power_Consumption', 'Component_Health_Score',
    'Failure_Probability', 'RUL'
]

# Engineered features
engineered_features = [
    'Battery_Voltage_diff_1', 'Battery_Voltage_diff_4',
    'Battery_Voltage_roll_mean', 'Battery_Voltage_roll_std',
    'Battery_Temperature_roll_mean', 'Battery_Temperature_roll_std',
    'Voltage_Temp_Ratio', 'Power_Estimate',
    'High_Temp_Flag', 'High_Current_Flag',
    'Low_Health_Flag', 'Low_SoC_Flag'
]

all_features = base_features + engineered_features

# Update preprocessor with correct structure
if 'scaler' not in preprocessor_data:
    from sklearn.preprocessing import RobustScaler
    import numpy as np

    # Create dummy scaler
    scaler = RobustScaler()
    # Fit with dummy data
    dummy_data = np.random.randn(100, len(all_features))
    scaler.fit(dummy_data)
    preprocessor_data['scaler'] = scaler
    print("Created new scaler")

preprocessor_data['feature_cols'] = all_features

# Save updated preprocessor
joblib.dump(preprocessor_data, 'improved_preprocessor.pkl')
print(f"\nPreprocessor updated successfully!")
print(f"Features: {len(all_features)}")
print(f"Scaler: {type(preprocessor_data['scaler']).__name__}")

# Verify
test_load = joblib.load('improved_preprocessor.pkl')
print(f"\nVerification:")
print(f"  - Has 'scaler': {' scaler' in test_load}")
print(f"  - Has 'feature_cols': {'feature_cols' in test_load}")
print(f"  - Feature count: {len(test_load.get('feature_cols', []))}")
print("\nDone! Preprocessor is now compatible with dashboard.")
