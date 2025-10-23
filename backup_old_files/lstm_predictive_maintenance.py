"""
Professional LSTM-based Predictive Maintenance System
Competition-Grade Implementation for EV Fleet Digital Twin

Features:
- Multi-variate time series LSTM
- Advanced feature engineering
- Proper train/validation/test splits (temporal)
- Class imbalance handling
- Comprehensive evaluation metrics
- Model interpretability tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Bidirectional, Input, Attention)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class EVFleetDataPreprocessor:
    """
    Professional data preprocessing pipeline for EV fleet time-series data.

    Handles:
    - Feature engineering
    - Sequence generation for LSTM
    - Scaling and normalization
    - Train/validation/test splitting (temporal)
    """

    def __init__(self, sequence_length=24, prediction_horizon=4):
        """
        Args:
            sequence_length: Number of past time steps (24 = 6 hours of 15-min data)
            prediction_horizon: How many steps ahead to predict (4 = 1 hour ahead)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_columns = []
        self.target_column = None

    def engineer_features(self, df):
        """
        Create advanced features from raw data.

        Professor's recommendations + advanced temporal features.
        """
        print("Engineering features...")

        df = df.copy()

        # 1. BASIC FEATURES (Professor's recommendations)
        basic_features = [
            'Battery_Voltage', 'Battery_Current', 'Battery_Temperature',
            'SoC', 'SoH', 'Charge_Cycles', 'Distance_Traveled',
            'Power_Consumption', 'Motor_Temperature',
            'Component_Health_Score', 'Failure_Probability', 'RUL'
        ]

        # 2. ENGINEERED FEATURES

        # Battery health indicators
        df['Battery_Power'] = df['Battery_Voltage'] * df['Battery_Current']
        df['Voltage_to_Temp_Ratio'] = df['Battery_Voltage'] / (df['Battery_Temperature'] + 1e-6)
        df['SoC_SoH_Product'] = df['SoC'] * df['SoH']

        # Rate of change features (velocity)
        for col in ['Battery_Voltage', 'Battery_Temperature', 'SoC', 'Component_Health_Score']:
            df[f'{col}_diff'] = df[col].diff().fillna(0)

        # Rolling statistics (6 hours = 24 steps)
        rolling_window = 24
        for col in ['Battery_Voltage', 'Battery_Current', 'Battery_Temperature', 'SoC']:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=rolling_window, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=rolling_window, min_periods=1).std().fillna(0)
            df[f'{col}_rolling_min'] = df[col].rolling(window=rolling_window, min_periods=1).min()
            df[f'{col}_rolling_max'] = df[col].rolling(window=rolling_window, min_periods=1).max()

        # Distance-based features
        df['Distance_per_Cycle'] = df['Distance_Traveled'] / (df['Charge_Cycles'] + 1)

        # Time-based features (if timestamp available)
        if 'Timestamp' in df.columns:
            df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
            df['DayOfWeek'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
            df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

        # 3. TARGET VARIABLE
        # Binary: Needs maintenance (Type 1, 2, or 3) vs No maintenance (Type 0)
        df['Maintenance_Needed'] = (df['Maintenance_Type'] > 0).astype(int)

        # Future target: Will need maintenance in next prediction_horizon steps?
        df['Future_Maintenance'] = df['Maintenance_Needed'].shift(-self.prediction_horizon).fillna(0).astype(int)

        print(f"✓ Engineered {len(df.columns)} total features")
        print(f"✓ Target: Future_Maintenance ({df['Future_Maintenance'].sum():,} positive samples)")

        return df

    def create_sequences(self, df, feature_cols, target_col='Future_Maintenance'):
        """
        Create LSTM-ready sequences from time-series data.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            target_col: Target variable column name

        Returns:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples,)
        """
        print(f"Creating sequences (length={self.sequence_length})...")

        self.feature_columns = feature_cols
        self.target_column = target_col

        data = df[feature_cols].values
        targets = df[target_col].values

        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(targets[i + self.sequence_length])

        X = np.array(X)
        y = np.array(y)

        print(f"✓ Created {len(X):,} sequences")
        print(f"  Shape: {X.shape} (samples, time_steps, features)")
        print(f"  Positive class: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")

        return X, y

    def temporal_train_test_split(self, X, y, train_size=0.7, val_size=0.15):
        """
        Split time-series data temporally (no shuffling!).

        Args:
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Remainder for testing
        """
        n = len(X)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        print(f"\nTemporal split:")
        print(f"  Train: {len(X_train):,} samples ({len(X_train)/n*100:.1f}%)")
        print(f"  Val:   {len(X_val):,} samples ({len(X_val)/n*100:.1f}%)")
        print(f"  Test:  {len(X_test):,} samples ({len(X_test)/n*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_scaler(self, X_train):
        """Fit scaler on training data only."""
        print("Fitting scaler on training data...")
        n_samples, n_timesteps, n_features = X_train.shape

        # Reshape to 2D for scaling
        X_train_2d = X_train.reshape(-1, n_features)
        self.scaler.fit(X_train_2d)

        print("✓ Scaler fitted")
        return self

    def transform(self, X):
        """Scale data using fitted scaler."""
        n_samples, n_timesteps, n_features = X.shape

        # Reshape, scale, reshape back
        X_2d = X.reshape(-1, n_features)
        X_scaled_2d = self.scaler.transform(X_2d)
        X_scaled = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        return X_scaled

    def save_preprocessor(self, path='preprocessor.pkl'):
        """Save preprocessor for deployment."""
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon
        }, path)
        print(f"✓ Preprocessor saved to {path}")


class LSTMPredictiveMaintenanceModel:
    """
    Professional LSTM model for predictive maintenance.

    Architecture options:
    - Vanilla LSTM
    - Stacked LSTM
    - Bidirectional LSTM
    - LSTM with Attention
    """

    def __init__(self, input_shape, model_type='stacked_lstm'):
        """
        Args:
            input_shape: (sequence_length, n_features)
            model_type: 'vanilla', 'stacked_lstm', 'bidirectional', 'attention'
        """
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        self.history = None

    def build_model(self, lstm_units=[128, 64], dropout_rate=0.3, learning_rate=0.001):
        """
        Build LSTM model with best practices.

        Args:
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        print(f"\nBuilding {self.model_type} model...")

        model = Sequential(name=f'LSTM_Predictive_Maintenance_{self.model_type}')

        if self.model_type == 'vanilla':
            # Simple single-layer LSTM
            model.add(Input(shape=self.input_shape))
            model.add(LSTM(lstm_units[0], activation='tanh'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(1, activation='sigmoid'))

        elif self.model_type == 'stacked_lstm':
            # Stacked LSTM (recommended for complex patterns)
            model.add(Input(shape=self.input_shape))
            for i, units in enumerate(lstm_units[:-1]):
                model.add(LSTM(units, activation='tanh', return_sequences=True))
                model.add(Dropout(dropout_rate))
            model.add(LSTM(lstm_units[-1], activation='tanh'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            model.add(Dense(1, activation='sigmoid'))

        elif self.model_type == 'bidirectional':
            # Bidirectional LSTM (considers past and future context)
            model.add(Input(shape=self.input_shape))
            for i, units in enumerate(lstm_units[:-1]):
                model.add(Bidirectional(LSTM(units, activation='tanh', return_sequences=True)))
                model.add(Dropout(dropout_rate))
            model.add(Bidirectional(LSTM(lstm_units[-1], activation='tanh')))
            model.add(Dropout(dropout_rate))
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            model.add(Dense(1, activation='sigmoid'))

        # Compile with appropriate loss for imbalanced classification
        optimizer = optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        self.model = model

        print("✓ Model architecture:")
        model.summary()

        return model

    def train(self, X_train, y_train, X_val, y_val,
              epochs=50, batch_size=64, class_weight=None):
        """
        Train LSTM model with best practices.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            class_weight: Dict for handling class imbalance
        """
        print(f"\nTraining model for {epochs} epochs...")

        # Calculate class weights if not provided (handle imbalance)
        if class_weight is None:
            neg_samples = np.sum(y_train == 0)
            pos_samples = np.sum(y_train == 1)
            total = len(y_train)

            class_weight = {
                0: total / (2 * neg_samples),
                1: total / (2 * pos_samples)
            }
            print(f"Computed class weights: {class_weight}")

        # Callbacks for training
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_lstm_model.keras',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callback_list,
            verbose=1
        )

        print("✓ Training complete")
        return self.history

    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Comprehensive model evaluation.

        Args:
            X_test, y_test: Test data
            threshold: Classification threshold
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)

        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > threshold).astype(int)

        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nClassification Metrics (threshold={threshold}):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f} (How many predicted failures were correct?)")
        print(f"  Recall:    {recall:.4f} (How many actual failures did we catch?)")
        print(f"  F1-Score:  {f1:.4f} (Harmonic mean of precision & recall)")
        print(f"  AUC-ROC:   {auc:.4f} (Overall discrimination ability)")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              No    Yes")
        print(f"Actual No   {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"       Yes  {cm[1,0]:5d}  {cm[1,1]:5d}")

        # Business metrics
        print(f"\nBusiness Impact Analysis:")
        true_positives = cm[1, 1]
        false_negatives = cm[1, 0]
        false_positives = cm[0, 1]

        print(f"  ✓ Prevented failures: {true_positives}")
        print(f"  ✗ Missed failures: {false_negatives}")
        print(f"  ⚠ False alarms: {false_positives}")

        if true_positives + false_negatives > 0:
            prevention_rate = true_positives / (true_positives + false_negatives) * 100
            print(f"  📊 Prevention rate: {prevention_rate:.1f}%")

        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Maintenance', 'Maintenance Needed']))

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }

    def save_model(self, path='lstm_predictive_maintenance.keras'):
        """Save trained model."""
        self.model.save(path)
        print(f"✓ Model saved to {path}")


class ModelVisualizer:
    """Visualization tools for model analysis."""

    @staticmethod
    def plot_training_history(history, save_path='training_history.png'):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # AUC
        axes[1, 0].plot(history.history['auc'], label='Train AUC')
        axes[1, 0].plot(history.history['val_auc'], label='Val AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Precision & Recall
        axes[1, 1].plot(history.history['precision'], label='Train Precision', linestyle='--')
        axes[1, 1].plot(history.history['val_precision'], label='Val Precision', linestyle='--')
        axes[1, 1].plot(history.history['recall'], label='Train Recall', linestyle=':')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linestyle=':')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Maintenance', 'Maintenance'],
                   yticklabels=['No Maintenance', 'Maintenance'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_roc_curve(y_test, y_pred_proba, save_path='roc_curve.png'):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'LSTM (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Predictive Maintenance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_precision_recall_curve(y_test, y_pred_proba, save_path='pr_curve.png'):
        """Plot Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'LSTM (AP = {ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curve saved to {save_path}")
        plt.close()


def main_training_pipeline():
    """
    Main training pipeline - Competition-ready implementation.
    """
    print("="*70)
    print("EV FLEET PREDICTIVE MAINTENANCE - LSTM MODEL TRAINING")
    print("="*70)

    # 1. Load data
    print("\n[1/7] Loading data...")
    df = pd.read_csv('EV_Predictive_Maintenance_Dataset_15min.csv',
                     parse_dates=['Timestamp'],
                     nrows=50000)  # Use full dataset for competition: remove nrows
    print(f"✓ Loaded {len(df):,} records")

    # 2. Preprocessing
    print("\n[2/7] Preprocessing...")
    preprocessor = EVFleetDataPreprocessor(sequence_length=24, prediction_horizon=4)
    df = preprocessor.engineer_features(df)

    # Select features
    feature_cols = [col for col in df.columns if col not in
                   ['Timestamp', 'Maintenance_Type', 'Maintenance_Needed', 'Future_Maintenance']]

    # Remove non-numeric columns if any
    feature_cols = [col for col in feature_cols if df[col].dtype in ['float32', 'float64', 'int64', 'int32']]

    print(f"Using {len(feature_cols)} features")

    # 3. Create sequences
    print("\n[3/7] Creating sequences...")
    X, y = preprocessor.create_sequences(df, feature_cols, target_col='Future_Maintenance')

    # 4. Split data
    print("\n[4/7] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.temporal_train_test_split(X, y)

    # 5. Scale data
    print("\n[5/7] Scaling data...")
    preprocessor.fit_scaler(X_train)
    X_train = preprocessor.transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    preprocessor.save_preprocessor()

    # 6. Build and train model
    print("\n[6/7] Building and training LSTM model...")
    model = LSTMPredictiveMaintenanceModel(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        model_type='stacked_lstm'  # Change to 'bidirectional' for comparison
    )

    model.build_model(lstm_units=[128, 64], dropout_rate=0.3, learning_rate=0.001)

    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,  # Increase for competition
        batch_size=64
    )

    # 7. Evaluate
    print("\n[7/7] Evaluating model...")
    results = model.evaluate(X_test, y_test, threshold=0.5)

    # Save model
    model.save_model()

    # 8. Visualizations
    print("\nGenerating visualizations...")
    visualizer = ModelVisualizer()
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(results['confusion_matrix'])
    visualizer.plot_roc_curve(y_test, results['y_pred_proba'])
    visualizer.plot_precision_recall_curve(y_test, results['y_pred_proba'])

    # Save results
    results_summary = {
        'model_type': 'stacked_lstm',
        'sequence_length': preprocessor.sequence_length,
        'prediction_horizon': preprocessor.prediction_horizon,
        'n_features': len(feature_cols),
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1']),
        'auc': float(results['auc']),
        'timestamp': datetime.now().isoformat()
    }

    with open('model_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"\nModel Performance Summary:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  AUC:       {results['auc']:.4f}")
    print(f"\nFiles generated:")
    print(f"  - best_lstm_model.keras")
    print(f"  - preprocessor.pkl")
    print(f"  - model_results.json")
    print(f"  - training_history.png")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - pr_curve.png")

    return model, preprocessor, results


if __name__ == "__main__":
    # Run complete training pipeline
    model, preprocessor, results = main_training_pipeline()
