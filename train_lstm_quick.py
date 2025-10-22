"""
Quick LSTM Training Script - Optimized for Hackathon
Trains model and saves it for deployment
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("EV FLEET PREDICTIVE MAINTENANCE - LSTM TRAINING")
print("="*70)

# 1. LOAD DATA
print("\n[1/6] Loading data...")
df = pd.read_csv('EV_Predictive_Maintenance_Dataset_15min.csv',
                 parse_dates=['Timestamp'],
                 nrows=30000)  # Use 30k for faster training
print(f"Loaded {len(df):,} records")

# 2. FEATURE ENGINEERING
print("\n[2/6] Engineering features...")

# Basic features (Professor's recommendations)
feature_cols = [
    'Battery_Voltage', 'Battery_Current', 'Battery_Temperature',
    'SoC', 'SoH', 'Charge_Cycles', 'Distance_Traveled',
    'Power_Consumption', 'Component_Health_Score',
    'Failure_Probability', 'RUL'
]

# Remove any NaN
df = df[feature_cols + ['Maintenance_Type']].dropna()

# Target: Binary classification (Maintenance needed?)
df['Maintenance_Needed'] = (df['Maintenance_Type'] > 0).astype(int)

print(f"Features: {len(feature_cols)}")
print(f"Target distribution: {df['Maintenance_Needed'].value_counts().to_dict()}")

# 3. CREATE SEQUENCES
print("\n[3/6] Creating sequences...")

sequence_length = 24  # 6 hours
prediction_horizon = 4  # 1 hour ahead

# Future target
df['Future_Maintenance'] = df['Maintenance_Needed'].shift(-prediction_horizon).fillna(0).astype(int)

data = df[feature_cols].values
targets = df['Future_Maintenance'].values

X, y = [], []
for i in range(len(data) - sequence_length):
    X.append(data[i:i + sequence_length])
    y.append(targets[i + sequence_length])

X = np.array(X)
y = np.array(y)

print(f"Sequences: {len(X):,}")
print(f"Shape: {X.shape}")
print(f"Positive samples: {y.sum():,} ({y.sum()/len(y)*100:.1f}%)")

# 4. SPLIT DATA (Temporal)
print("\n[4/6] Splitting data...")

n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"Train: {len(X_train):,} ({len(X_train)/n*100:.1f}%)")
print(f"Val:   {len(X_val):,} ({len(X_val)/n*100:.1f}%)")
print(f"Test:  {len(X_test):,} ({len(X_test)/n*100:.1f}%)")

# 5. SCALE DATA
print("\n[5/6] Scaling data...")

scaler = RobustScaler()
n_samples, n_timesteps, n_features = X_train.shape

X_train_2d = X_train.reshape(-1, n_features)
scaler.fit(X_train_2d)

X_train = scaler.transform(X_train_2d).reshape(n_samples, n_timesteps, n_features)
X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(-1, n_timesteps, n_features)
X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(-1, n_timesteps, n_features)

print("Scaling complete")

# 6. BUILD MODEL
print("\n[6/6] Building and training LSTM model...")

model = keras.Sequential([
    layers.Input(shape=(sequence_length, n_features)),
    layers.LSTM(128, activation='tanh', return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(64, activation='tanh'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
], name='LSTM_Predictive_Maintenance')

# Class weights (handle imbalance)
neg_samples = np.sum(y_train == 0)
pos_samples = np.sum(y_train == 1)
total = len(y_train)

class_weight = {
    0: total / (2 * neg_samples),
    1: total / (2 * pos_samples)
}

print(f"\nClass weights: {class_weight}")

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

print("\nModel architecture:")
model.summary()

# Train
print("\nTraining (30 epochs)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,  # Reduced for faster training
    batch_size=64,
    class_weight=class_weight,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ],
    verbose=1
)

# 7. EVALUATE
print("\n" + "="*70)
print("EVALUATION ON TEST SET")
print("="*70)

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nPerformance Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"              Predicted")
print(f"            No    Yes")
print(f"Actual No  {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Yes {cm[1,0]:4d}  {cm[1,1]:4d}")

print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['No Maintenance', 'Maintenance Needed']))

# 8. SAVE MODEL & PREPROCESSOR
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

model.save('best_lstm_model.keras')
print("Saved: best_lstm_model.keras")

joblib.dump({
    'scaler': scaler,
    'feature_columns': feature_cols,
    'sequence_length': sequence_length,
    'prediction_horizon': prediction_horizon
}, 'preprocessor.pkl')
print("Saved: preprocessor.pkl")

# Save results
results = {
    'model_type': 'stacked_lstm',
    'sequence_length': sequence_length,
    'prediction_horizon': prediction_horizon,
    'n_features': len(feature_cols),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auc': float(auc),
    'total_parameters': int(model.count_params()),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'timestamp': datetime.now().isoformat()
}

with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: model_results.json")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  - best_lstm_model.keras (trained model)")
print("  - preprocessor.pkl (scaler & config)")
print("  - model_results.json (performance metrics)")
print("\nModel is ready for deployment!")
print("="*70)
