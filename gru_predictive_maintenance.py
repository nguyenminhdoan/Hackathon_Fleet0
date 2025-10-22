"""
GRU-based Predictive Maintenance Model
For comparison with LSTM model

GRU (Gated Recurrent Unit) advantages:
- Fewer parameters than LSTM (faster training)
- Often similar performance to LSTM
- Better for smaller datasets
- Reduced risk of overfitting
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Bidirectional, Input
import json
from datetime import datetime

# Import from LSTM module for consistency
import sys
sys.path.append('.')
from lstm_predictive_maintenance import (EVFleetDataPreprocessor, ModelVisualizer)


class GRUPredictiveMaintenanceModel:
    """
    GRU model for predictive maintenance.

    GRU vs LSTM:
    - GRU has 2 gates (update, reset) vs LSTM's 3 gates (input, forget, output)
    - GRU is faster to train (~30% fewer parameters)
    - Often achieves similar performance to LSTM
    """

    def __init__(self, input_shape, model_type='stacked_gru'):
        """
        Args:
            input_shape: (sequence_length, n_features)
            model_type: 'vanilla', 'stacked_gru', 'bidirectional'
        """
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        self.history = None

    def build_model(self, gru_units=[128, 64], dropout_rate=0.3, learning_rate=0.001):
        """
        Build GRU model.

        Args:
            gru_units: List of units for each GRU layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        print(f"\nBuilding {self.model_type} model...")

        model = Sequential(name=f'GRU_Predictive_Maintenance_{self.model_type}')

        if self.model_type == 'vanilla':
            # Simple single-layer GRU
            model.add(Input(shape=self.input_shape))
            model.add(GRU(gru_units[0], activation='tanh'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(1, activation='sigmoid'))

        elif self.model_type == 'stacked_gru':
            # Stacked GRU (recommended)
            model.add(Input(shape=self.input_shape))
            for i, units in enumerate(gru_units[:-1]):
                model.add(GRU(units, activation='tanh', return_sequences=True))
                model.add(Dropout(dropout_rate))
            model.add(GRU(gru_units[-1], activation='tanh'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            model.add(Dense(1, activation='sigmoid'))

        elif self.model_type == 'bidirectional':
            # Bidirectional GRU
            model.add(Input(shape=self.input_shape))
            for i, units in enumerate(gru_units[:-1]):
                model.add(Bidirectional(GRU(units, activation='tanh', return_sequences=True)))
                model.add(Dropout(dropout_rate))
            model.add(Bidirectional(GRU(gru_units[-1], activation='tanh')))
            model.add(Dropout(dropout_rate))
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            model.add(Dense(1, activation='sigmoid'))

        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

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

        print("✓ GRU Model architecture:")
        model.summary()

        # Count parameters
        total_params = model.count_params()
        print(f"\n✓ Total parameters: {total_params:,}")

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, class_weight=None):
        """Train GRU model (same interface as LSTM)."""
        from tensorflow.keras import callbacks

        print(f"\nTraining GRU model for {epochs} epochs...")

        # Calculate class weights if not provided
        if class_weight is None:
            neg_samples = np.sum(y_train == 0)
            pos_samples = np.sum(y_train == 1)
            total = len(y_train)

            class_weight = {
                0: total / (2 * neg_samples),
                1: total / (2 * pos_samples)
            }
            print(f"Computed class weights: {class_weight}")

        # Callbacks
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
                'best_gru_model.keras',
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
        """Evaluate GRU model (same interface as LSTM)."""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     classification_report)

        print("\n" + "="*70)
        print("GRU MODEL EVALUATION RESULTS")
        print("="*70)

        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > threshold).astype(int)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nClassification Metrics (threshold={threshold}):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              No    Yes")
        print(f"Actual No   {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"       Yes  {cm[1,0]:5d}  {cm[1,1]:5d}")

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

    def save_model(self, path='gru_predictive_maintenance.keras'):
        """Save trained GRU model."""
        self.model.save(path)
        print(f"✓ GRU model saved to {path}")


def main_gru_training_pipeline():
    """
    GRU training pipeline for comparison with LSTM.
    """
    print("="*70)
    print("EV FLEET PREDICTIVE MAINTENANCE - GRU MODEL TRAINING")
    print("="*70)

    # 1. Load data
    print("\n[1/7] Loading data...")
    df = pd.read_csv('EV_Predictive_Maintenance_Dataset_15min.csv',
                     parse_dates=['Timestamp'],
                     nrows=50000)  # Use full dataset for competition
    print(f"✓ Loaded {len(df):,} records")

    # 2. Preprocessing (reuse LSTM preprocessor)
    print("\n[2/7] Preprocessing...")
    preprocessor = EVFleetDataPreprocessor(sequence_length=24, prediction_horizon=4)
    df = preprocessor.engineer_features(df)

    # Select features
    feature_cols = [col for col in df.columns if col not in
                   ['Timestamp', 'Maintenance_Type', 'Maintenance_Needed', 'Future_Maintenance']]
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

    # 6. Build and train GRU model
    print("\n[6/7] Building and training GRU model...")
    model = GRUPredictiveMaintenanceModel(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        model_type='stacked_gru'
    )

    model.build_model(gru_units=[128, 64], dropout_rate=0.3, learning_rate=0.001)

    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=64
    )

    # 7. Evaluate
    print("\n[7/7] Evaluating GRU model...")
    results = model.evaluate(X_test, y_test, threshold=0.5)

    # Save model
    model.save_model()

    # Visualizations
    print("\nGenerating visualizations...")
    visualizer = ModelVisualizer()
    visualizer.plot_training_history(history, save_path='gru_training_history.png')
    visualizer.plot_confusion_matrix(results['confusion_matrix'], save_path='gru_confusion_matrix.png')
    visualizer.plot_roc_curve(y_test, results['y_pred_proba'], save_path='gru_roc_curve.png')
    visualizer.plot_precision_recall_curve(y_test, results['y_pred_proba'], save_path='gru_pr_curve.png')

    # Save results
    results_summary = {
        'model_type': 'stacked_gru',
        'sequence_length': preprocessor.sequence_length,
        'prediction_horizon': preprocessor.prediction_horizon,
        'n_features': len(feature_cols),
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1']),
        'auc': float(results['auc']),
        'total_parameters': int(model.model.count_params()),
        'timestamp': datetime.now().isoformat()
    }

    with open('gru_model_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "="*70)
    print("✅ GRU TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"\nGRU Model Performance Summary:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  AUC:       {results['auc']:.4f}")
    print(f"  Parameters: {model.model.count_params():,}")

    return model, preprocessor, results


if __name__ == "__main__":
    # Run GRU training pipeline
    model, preprocessor, results = main_gru_training_pipeline()
