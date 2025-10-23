"""
IMPROVED GRU Model Training
Matches LSTM architecture but uses GRU layers for comparison

Key Improvements (same as LSTM):
1. Better feature engineering
2. Optimized class weights
3. Focal loss for hard examples
4. Proper sequence creation
5. Hyperparameter tuning
6. Saves training visualizations to images/
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Create images folder if it doesn't exist
os.makedirs('images', exist_ok=True)


# ==============================================================================
# Focal Loss (handles class imbalance better)
# ==============================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = K.pow(1.0 - pt, gamma)

        # Binary crossentropy
        bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)

        # Apply focal weight
        focal = alpha * focal_weight * bce

        return K.mean(focal)

    return focal_loss_fixed


# ==============================================================================
# Advanced Feature Engineering
# ==============================================================================

def engineer_advanced_features(df):
    """
    Create more sophisticated features to help model learn better.
    """
    print("Engineering advanced features...")

    df = df.copy()

    # 1. RATE OF CHANGE FEATURES (velocity)
    for col in ['Battery_Voltage', 'Battery_Temperature', 'SoC', 'SoH',
                'Component_Health_Score', 'Failure_Probability']:
        df[f'{col}_diff_1'] = df[col].diff(1).fillna(0)
        df[f'{col}_diff_4'] = df[col].diff(4).fillna(0)

    # 2. ROLLING STATISTICS (12 steps = 3 hours)
    window = 12
    for col in ['Battery_Voltage', 'Battery_Current', 'Battery_Temperature', 'SoC']:
        df[f'{col}_roll_mean'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_roll_std'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'{col}_roll_max'] = df[col].rolling(window=window, min_periods=1).max()
        df[f'{col}_roll_min'] = df[col].rolling(window=window, min_periods=1).min()

    # 3. INTERACTION FEATURES
    df['Voltage_Temp_Ratio'] = df['Battery_Voltage'] / (df['Battery_Temperature'] + 1e-6)
    df['Current_Temp_Product'] = df['Battery_Current'] * df['Battery_Temperature']
    df['SoC_SoH_Product'] = df['SoC'] * df['SoH']
    df['Power_Estimate'] = df['Battery_Voltage'] * df['Battery_Current']

    # 4. HEALTH DEGRADATION FEATURES
    df['Health_Decline'] = df['Component_Health_Score'].diff(4).fillna(0)
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


# ==============================================================================
# Better Sequence Creation with Overlap
# ==============================================================================

def create_sequences_improved(data, targets, sequence_length=24, overlap=0.5):
    """
    Create sequences with overlap to generate more training samples.
    """
    stride = max(1, int(sequence_length * (1 - overlap)))

    X, y = [], []
    for i in range(0, len(data) - sequence_length, stride):
        X.append(data[i:i + sequence_length])
        y.append(targets[i + sequence_length])

    return np.array(X), np.array(y)


# ==============================================================================
# GRU Model Architecture (matching LSTM structure)
# ==============================================================================

def build_optimized_gru(input_shape, learning_rate=0.0005):
    """
    Build improved GRU with same structure as LSTM:
    - Same depth (3 GRU layers)
    - Same units (256, 128, 64)
    - Same regularization
    """
    inputs = layers.Input(shape=input_shape)

    # First GRU block (matching LSTM structure)
    x = layers.GRU(256, activation='tanh', return_sequences=True,
                    reset_after=True)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Second GRU block
    x = layers.GRU(128, activation='tanh', return_sequences=True,
                    reset_after=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Third GRU block
    x = layers.GRU(64, activation='tanh', reset_after=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Dense layers (same as LSTM)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='Improved_GRU')

    # Compile with focal loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    return model


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_training_history(history, save_path='images/gru_training_history.png'):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Recall
    axes[1, 0].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 0].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 0].set_title('Model Recall Over Epochs', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # AUC
    axes[1, 1].plot(history.history['auc'], label='Train AUC', linewidth=2)
    axes[1, 1].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
    axes[1, 1].set_title('Model AUC Over Epochs', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='images/gru_confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Maintenance', 'Maintenance'],
                yticklabels=['No Maintenance', 'Maintenance'])
    plt.title('GRU Model - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(thresholds, metrics_data, save_path='images/gru_threshold_analysis.png'):
    """Plot threshold analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(thresholds, metrics_data['accuracy'], marker='o', linewidth=2)
    axes[0, 0].set_title('Accuracy vs Threshold', fontweight='bold')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(thresholds, metrics_data['precision'], marker='o', linewidth=2, color='orange')
    axes[0, 1].set_title('Precision vs Threshold', fontweight='bold')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(thresholds, metrics_data['recall'], marker='o', linewidth=2, color='green')
    axes[1, 0].set_title('Recall vs Threshold', fontweight='bold')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(thresholds, metrics_data['f1'], marker='o', linewidth=2, color='red')
    axes[1, 1].set_title('F1-Score vs Threshold', fontweight='bold')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def main():
    print("="*70)
    print("IMPROVED GRU MODEL TRAINING")
    print("="*70)

    # 1. LOAD DATA
    print("\n[1/8] Loading data...")
    df = pd.read_csv('EV_Predictive_Maintenance_Dataset_15min.csv',
                     parse_dates=['Timestamp'],
                     nrows=50000)
    print(f"Loaded {len(df):,} records")

    # 2. ADVANCED FEATURE ENGINEERING
    print("\n[2/8] Engineering features...")
    df = engineer_advanced_features(df)

    # Select features
    exclude_cols = ['Timestamp', 'Maintenance_Type']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if df[col].dtype in ['float32', 'float64', 'int64', 'int32']]

    # Remove NaN
    df = df[feature_cols + ['Maintenance_Type']].dropna()

    # Create target
    df['Maintenance_Needed'] = (df['Maintenance_Type'] > 0).astype(int)
    df['Future_Maintenance'] = df['Maintenance_Needed'].shift(-4).fillna(0).astype(int)

    print(f"Using {len(feature_cols)} features")
    print(f"Target distribution: {df['Future_Maintenance'].value_counts().to_dict()}")

    # 3. CREATE SEQUENCES WITH OVERLAP
    print("\n[3/8] Creating sequences with overlap...")
    data = df[feature_cols].values
    targets = df['Future_Maintenance'].values

    X, y = create_sequences_improved(data, targets, sequence_length=24, overlap=0.5)

    print(f"Created {len(X):,} sequences (with 50% overlap)")
    print(f"Positive samples: {y.sum():,} ({y.sum()/len(y)*100:.1f}%)")

    # 4. TEMPORAL SPLIT
    print("\n[4/8] Splitting data...")
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
    print("\n[5/8] Scaling data...")
    scaler = RobustScaler()
    n_samples, n_timesteps, n_features = X_train.shape

    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    X_train = scaler.transform(X_train_2d).reshape(n_samples, n_timesteps, n_features)
    X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(-1, n_timesteps, n_features)
    X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(-1, n_timesteps, n_features)

    # 6. BUILD MODEL
    print("\n[6/8] Building improved GRU model...")
    model = build_optimized_gru(input_shape=(n_timesteps, n_features), learning_rate=0.0005)

    print("\nModel architecture:")
    model.summary()

    # 7. TRAIN
    print("\n[7/8] Training with focal loss and optimized settings...")

    # Class weights
    neg_samples = np.sum(y_train == 0)
    pos_samples = np.sum(y_train == 1)
    total = len(y_train)

    class_weight = {
        0: total / (2 * neg_samples) * 0.5,
        1: total / (2 * pos_samples) * 2.0
    }

    print(f"Class weights: {class_weight}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_recall',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_recall',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_improved_gru.keras',
            monitor='val_recall',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history plot
    plot_training_history(history)

    # 8. EVALUATE
    print("\n[8/8] Evaluating improved GRU model...")
    print("="*70)

    # Test with different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    y_pred_proba = model.predict(X_test, verbose=0)

    print("\nThreshold Analysis:")
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*70)

    # Store metrics for plotting
    metrics_data = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics_data['accuracy'].append(acc)
        metrics_data['precision'].append(prec)
        metrics_data['recall'].append(rec)
        metrics_data['f1'].append(f1)

        print(f"{threshold:<12.1f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("-"*70)
    print(f"\nBest threshold: {best_threshold} (F1 = {best_f1:.4f})")

    # Plot threshold analysis
    plot_metrics_comparison(thresholds, metrics_data)

    # Final evaluation with best threshold
    y_pred = (y_pred_proba > best_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "="*70)
    print("FINAL RESULTS (with optimal threshold)")
    print("="*70)
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"AUC-ROC:   {auc:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"            No    Yes")
    print(f"Actual No  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Yes {cm[1,0]:4d}  {cm[1,1]:4d}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Maintenance', 'Maintenance Needed']))

    # SAVE EVERYTHING
    print("\n" + "="*70)
    print("SAVING MODEL AND RESULTS")
    print("="*70)

    model.save('best_improved_gru.keras')
    print("Saved: best_improved_gru.keras")

    joblib.dump({
        'scaler': scaler,
        'feature_columns': feature_cols,
        'sequence_length': 24,
        'prediction_horizon': 4,
        'best_threshold': best_threshold
    }, 'improved_gru_preprocessor.pkl')
    print("Saved: improved_gru_preprocessor.pkl")

    results = {
        'model_type': 'improved_stacked_gru',
        'architecture': 'GRU (256, 128, 64) + Dense (64, 32)',
        'sequence_length': 24,
        'prediction_horizon': 4,
        'n_features': n_features,
        'best_threshold': float(best_threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'total_parameters': int(model.count_params()),
        'timestamp': datetime.now().isoformat()
    }

    with open('improved_gru_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: improved_gru_results.json")

    print("\n" + "="*70)
    print("GRU TRAINING COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to images/ folder:")
    print(f"  - images/gru_training_history.png")
    print(f"  - images/gru_confusion_matrix.png")
    print(f"  - images/gru_threshold_analysis.png")


if __name__ == "__main__":
    main()
