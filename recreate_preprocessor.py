"""
Recreate preprocessor with EXACT features from train_improved_model.py
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib

# Exact copy of engineer_advanced_features from train_improved_model.py
def engineer_advanced_features(df):
    """Create more sophisticated features to help model learn better."""
    print("Engineering advanced features...")

    df = df.copy()

    # 1. RATE OF CHANGE FEATURES (velocity)
    for col in ['Battery_Voltage', 'Battery_Temperature', 'SoC', 'SoH',
                'Component_Health_Score', 'Failure_Probability']:
        df[f'{col}_diff_1'] = df[col].diff(1).fillna(0)  # 15-min change
        df[f'{col}_diff_4'] = df[col].diff(4).fillna(0)  # 1-hour change

    # 2. ROLLING STATISTICS (12 steps = 3 hours)
    window = 12
    for col in ['Battery_Voltage', 'Battery_Current', 'Battery_Temperature', 'SoC']:
        df[f'{col}_roll_mean'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_roll_std'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'{col}_roll_max'] = df[col].rolling(window=window, min_periods=1).max()
        df[f'{col}_roll_min'] = df[col].rolling(window=window, min_periods=1).min()

    # 3. INTERACTION FEATURES (combinations that matter)
    df['Voltage_Temp_Ratio'] = df['Battery_Voltage'] / (df['Battery_Temperature'] + 1e-6)
    df['Current_Temp_Product'] = df['Battery_Current'] * df['Battery_Temperature']
    df['SoC_SoH_Product'] = df['SoC'] * df['SoH']
    df['Power_Estimate'] = df['Battery_Voltage'] * df['Battery_Current']

    # 4. HEALTH DEGRADATION FEATURES
    df['Health_Decline'] = df['Component_Health_Score'].diff(4).fillna(0)  # 1-hour decline
    df['SoH_Decline'] = df['SoH'].diff(4).fillna(0)
    df['Risk_Increase'] = df['Failure_Probability'].diff(4).fillna(0)

    # 5. STRESS INDICATORS
    df['High_Current_Flag'] = (df['Battery_Current'].abs() > df['Battery_Current'].abs().quantile(0.75)).astype(int)
    df['High_Temp_Flag'] = (df['Battery_Temperature'] > df['Battery_Temperature'].quantile(0.75)).astype(int)
    df['Low_Health_Flag'] = (df['Component_Health_Score'] < df['Component_Health_Score'].quantile(0.25)).astype(int)

    # 6. CUMULATIVE FEATURES
    df['Cumulative_Charge_Cycles'] = df['Charge_Cycles']
    df['Cumulative_Distance'] = df['Distance_Traveled']

    print(f"Created {len(df.columns)} total features")

    return df


# Load first 50000 records (same as training)
print("Loading data...")
df = pd.read_csv('EV_Predictive_Maintenance_Dataset_15min.csv',
                 parse_dates=['Timestamp'],
                 nrows=50000)
print(f"Loaded {len(df):,} records")

# Engineer features
df = engineer_advanced_features(df)

# Select features (exclude timestamp and target)
exclude_cols = ['Timestamp', 'Maintenance_Type']
feature_cols = [col for col in df.columns if col not in exclude_cols]
feature_cols = [col for col in feature_cols if df[col].dtype in ['float32', 'float64', 'int64', 'int32']]

print(f"\nUsing {len(feature_cols)} features")

# Prepare data for scaler (remove NaN)
X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

# Fit scaler
print("Fitting scaler...")
scaler = RobustScaler()
scaler.fit(X)

# Save preprocessor
preprocessor_data = {
    'scaler': scaler,
    'feature_cols': feature_cols,
    'sequence_length': 24,
    'prediction_horizon': 4,
    'best_threshold': 0.3
}

joblib.dump(preprocessor_data, 'improved_preprocessor.pkl')

print(f"\nSaved preprocessor with {len(feature_cols)} features")
print("\nFeature list:")
for i, f in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {f}")

print("\nDone!")
