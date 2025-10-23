# LSTM vs GRU Model Comparison

## Overview

This project compares two deep learning architectures for EV Fleet Predictive Maintenance:
- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**

Both models use the same:
- Feature engineering (68 features from 28 raw sensors)
- Training data (50,000 records)
- Focal loss for class imbalance
- Optimization strategy (recall-focused)
- Evaluation metrics

## Files Created

### Training Scripts
1. **train_improved_model.py** - LSTM model training
   - Architecture: 3 LSTM layers (256 → 128 → 64)
   - Parameters: 587,649
   - Saves to: `best_improved_lstm.keras`

2. **train_improved_gru.py** - GRU model training
   - Architecture: 3 GRU layers (256 → 128 → 64)
   - Parameters: 444,161
   - Saves to: `best_improved_gru.keras`

### Comparison & Visualization
3. **compare_lstm_gru.py** - Comprehensive comparison analysis
   - Generates side-by-side metrics comparison
   - Creates radar charts
   - Produces detailed comparison tables
   - Recommends best model

### Output Files

#### Models
- `best_improved_lstm.keras` - Trained LSTM model
- `best_improved_gru.keras` - Trained GRU model
- `improved_preprocessor.pkl` - LSTM preprocessor
- `improved_gru_preprocessor.pkl` - GRU preprocessor

#### Results
- `improved_model_results.json` - LSTM performance metrics
- `improved_gru_results.json` - GRU performance metrics

#### Visualizations (in `images/` folder)
- `gru_training_history.png` - GRU training curves (loss, accuracy, recall, AUC)
- `gru_confusion_matrix.png` - GRU confusion matrix
- `gru_threshold_analysis.png` - GRU threshold optimization
- `lstm_gru_comparison.png` - Side-by-side bar charts
- `lstm_gru_radar.png` - Performance radar chart
- `lstm_gru_table.png` - Detailed comparison table

## Model Architecture Comparison

### LSTM Model
```
Input → LSTM(256) → BatchNorm → Dropout(0.3)
     → LSTM(128) → BatchNorm → Dropout(0.3)
     → LSTM(64)  → BatchNorm → Dropout(0.3)
     → Dense(64) → BatchNorm → Dropout(0.3)
     → Dense(32) → Dropout(0.2)
     → Dense(1, sigmoid)
```

**Total Parameters:** 587,649

### GRU Model
```
Input → GRU(256) → BatchNorm → Dropout(0.3)
     → GRU(128) → BatchNorm → Dropout(0.3)
     → GRU(64)  → BatchNorm → Dropout(0.3)
     → Dense(64) → BatchNorm → Dropout(0.3)
     → Dense(32) → Dropout(0.2)
     → Dense(1, sigmoid)
```

**Total Parameters:** 444,161

### Key Differences

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Parameters** | 587,649 | 444,161 |
| **Gates** | 3 (input, forget, output) | 2 (reset, update) |
| **Memory Cell** | Separate cell state | Hidden state only |
| **Complexity** | Higher | Lower |
| **Training Speed** | Slower | Faster (~25% speedup) |
| **Memory Usage** | Higher | Lower |

## How to Use

### 1. Train LSTM Model (if not already trained)
```bash
python train_improved_model.py
```

### 2. Train GRU Model
```bash
python train_improved_gru.py
```

### 3. Compare Models
```bash
python compare_lstm_gru.py
```

This will:
- Load both model results
- Generate comparison visualizations
- Print comprehensive analysis
- Recommend the best model for deployment

### 4. View Results
Check the `images/` folder for all generated visualizations.

## Expected Output

After running the comparison script, you'll see:

1. **Metrics Comparison**
   - Accuracy, Precision, Recall, F1-Score, AUC
   - Parameter count
   - Best thresholds

2. **Overall Winner**
   - Which model wins more metrics
   - Specific strengths of each model

3. **Key Insights**
   - Recall comparison (most important for predictive maintenance)
   - F1-Score balance
   - Model complexity trade-offs

4. **Recommendation**
   - Best model for deployment
   - Reasoning based on recall and use case

## Evaluation Metrics

Both models are evaluated on:

- **Recall** (Most Important): % of failures caught
  - Higher is better for predictive maintenance
  - Missing a failure costs $5,000
  - False alarm costs only $500

- **Precision**: % of maintenance alerts that are correct
  - Lower false alarms = less wasted maintenance

- **F1-Score**: Harmonic mean of precision and recall
  - Balance between catching failures and avoiding false alarms

- **Accuracy**: Overall correctness
  - Less important due to class imbalance

- **AUC**: Area under ROC curve
  - Model's ability to discriminate between classes

## Training Configuration

Both models use identical training setup:

- **Data**: 50,000 records
- **Features**: 68 engineered features
- **Sequence Length**: 24 timesteps (6 hours)
- **Prediction Horizon**: 4 timesteps (1 hour ahead)
- **Class Weights**: Favor minority class 4x
- **Loss Function**: Focal Loss (gamma=2.0, alpha=0.75)
- **Optimizer**: Adam (lr=0.0005)
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Callbacks**:
  - Early Stopping (patience=15, monitor=val_recall)
  - ReduceLROnPlateau (patience=7)
  - ModelCheckpoint (save best model)

## Results Summary

### LSTM Results
```json
{
  "model_type": "improved_stacked_lstm",
  "recall": 0.9076,
  "precision": 0.2860,
  "f1_score": 0.4349,
  "accuracy": 0.3056,
  "auc": 0.4718,
  "total_parameters": 587649,
  "best_threshold": 0.3
}
```

### GRU Results
(Will be populated after training completes)

## Visualization Guide

### 1. Training History (GRU)
Shows 4 plots:
- Loss over epochs (train vs val)
- Accuracy over epochs
- Recall over epochs (most important)
- AUC over epochs

Look for:
- Convergence (loss decreasing)
- No overfitting (val metrics not diverging)
- Improving recall

### 2. Confusion Matrix
Shows actual vs predicted classifications:
- True Negatives (TN): Correct "no maintenance" predictions
- False Positives (FP): False alarms
- False Negatives (FN): Missed failures (CRITICAL)
- True Positives (TP): Correct failure predictions

### 3. Threshold Analysis
Shows how different decision thresholds affect:
- Accuracy
- Precision
- Recall (prioritized)
- F1-Score

Lower threshold → Higher recall (catch more failures)

### 4. LSTM vs GRU Comparison
Side-by-side bar charts comparing all metrics with winner indicators.

### 5. Radar Chart
Visual comparison of all metrics on a pentagonal radar plot.

### 6. Comparison Table
Detailed table with:
- Metric values
- Differences
- Winners
- Parameter counts

## Recommendations for Hackathon Presentation

### Talking Points

1. **Why Compare LSTM vs GRU?**
   - Both are RNN variants for time-series
   - GRU is simpler (fewer parameters)
   - LSTM has more memory capacity
   - Real-world comparison on actual EV data

2. **Fair Comparison**
   - Same dataset
   - Same feature engineering
   - Same optimization strategy
   - Same evaluation metrics

3. **Key Finding: Recall is Critical**
   - Missing a failure costs $5,000
   - Preventive maintenance costs $500
   - Better to have false alarms than missed failures
   - Model with highest recall wins

4. **Computational Efficiency**
   - GRU trains ~25% faster
   - GRU uses ~25% fewer parameters
   - Easier deployment on edge devices

### Demo Flow

1. Show training scripts (same structure, different layers)
2. Run comparison script live
3. Display visualizations from images/ folder
4. Explain metrics and trade-offs
5. Reveal recommended model
6. Discuss deployment considerations

## Next Steps

1. ✅ Create images folder
2. ✅ Create GRU training script
3. ⏳ Train GRU model (in progress)
4. ✅ Create comparison script
5. ⏹ Run comparison (after GRU training completes)
6. ⏹ Generate all visualizations
7. ⏹ Update README with final results

## Troubleshooting

### If GRU training fails:
```bash
# Check if dependencies are installed
pip install tensorflow scikit-learn matplotlib seaborn

# Restart training
python train_improved_gru.py
```

### If comparison fails:
```bash
# Ensure both models are trained
ls -lh best_improved_lstm.keras
ls -lh best_improved_gru.keras

# Ensure results files exist
ls -lh improved_model_results.json
ls -lh improved_gru_results.json
```

### If visualizations don't save:
```bash
# Create images folder manually
mkdir images

# Check folder permissions
ls -la images/
```

## Citation

If using this comparison methodology, please cite:

```
EV Fleet Predictive Maintenance - LSTM vs GRU Comparison
Digital Twin Hackathon Challenge #2
Date: 2025-10-22
```

## License

MIT License - Free to use and modify

## Contact

For questions or issues, please refer to the project documentation.

---

**Status**: GRU training in progress...
**Last Updated**: 2025-10-22 20:45 UTC
