"""
Comprehensive Metrics for LSTM vs GRU Comparison

Includes both:
1. Classification metrics (for binary prediction)
2. Regression metrics (for continuous prediction)
3. Time-series specific metrics
4. Business metrics
"""

import numpy as np
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    # Regression metrics
    mean_absolute_error, mean_squared_error, r2_score
)
import json


class ComprehensiveMetricsCalculator:
    """
    Calculate all relevant metrics for model evaluation and comparison.
    """

    def __init__(self, model_name="Model"):
        self.model_name = model_name
        self.metrics = {}

    def calculate_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate classification metrics.

        Args:
            y_true: Actual labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_pred_proba: Predicted probabilities (0.0 to 1.0)
        """
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION METRICS - {self.model_name}")
        print(f"{'='*70}")

        # 1. ACCURACY
        accuracy = accuracy_score(y_true, y_pred)
        self.metrics['accuracy'] = accuracy
        print(f"\n1. ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   - What: Overall correctness")
        print(f"   - Formula: (TP + TN) / Total")
        print(f"   - Interpretation: {accuracy*100:.1f}% of predictions were correct")

        # 2. PRECISION
        precision = precision_score(y_true, y_pred, zero_division=0)
        self.metrics['precision'] = precision
        print(f"\n2. PRECISION: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   - What: Of predicted failures, how many were real?")
        print(f"   - Formula: TP / (TP + FP)")
        print(f"   - Interpretation: {precision*100:.1f}% of 'maintenance needed' predictions were correct")
        print(f"   - Business impact: Higher = fewer false alarms = less wasted money")

        # 3. RECALL (Sensitivity)
        recall = recall_score(y_true, y_pred, zero_division=0)
        self.metrics['recall'] = recall
        print(f"\n3. RECALL (Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
        print(f"   - What: Of actual failures, how many did we catch?")
        print(f"   - Formula: TP / (TP + FN)")
        print(f"   - Interpretation: Caught {recall*100:.1f}% of actual maintenance needs")
        print(f"   - Business impact: Higher = fewer missed failures = safer fleet")

        # 4. SPECIFICITY
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        self.metrics['specificity'] = specificity
        print(f"\n4. SPECIFICITY: {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"   - What: Of actual normal operations, how many did we identify correctly?")
        print(f"   - Formula: TN / (TN + FP)")
        print(f"   - Interpretation: {specificity*100:.1f}% of 'no maintenance' cases correctly identified")

        # 5. F1-SCORE
        f1 = f1_score(y_true, y_pred, zero_division=0)
        self.metrics['f1_score'] = f1
        print(f"\n5. F1-SCORE: {f1:.4f} ({f1*100:.2f}%)")
        print(f"   - What: Harmonic mean of precision and recall")
        print(f"   - Formula: 2 × (Precision × Recall) / (Precision + Recall)")
        print(f"   - Interpretation: Balanced measure considering both false alarms and missed failures")

        # 6. AUC-ROC
        if y_pred_proba is not None:
            auc = roc_auc_score(y_true, y_pred_proba)
            self.metrics['auc_roc'] = auc
            print(f"\n6. AUC-ROC: {auc:.4f}")
            print(f"   - What: Overall discrimination ability")
            print(f"   - Range: 0.5 (random) to 1.0 (perfect)")
            print(f"   - Interpretation: ", end="")
            if auc > 0.9:
                print("Excellent!")
            elif auc > 0.8:
                print("Good")
            elif auc > 0.7:
                print("Fair")
            elif auc > 0.6:
                print("Poor")
            else:
                print("Failing (barely better than random guessing)")

        # 7. CONFUSION MATRIX
        print(f"\n7. CONFUSION MATRIX:")
        print(f"                  Predicted")
        print(f"                No        Yes")
        print(f"   Actual No   {tn:5d}     {fp:5d}   (TN, FP)")
        print(f"   Actual Yes  {fn:5d}     {tp:5d}   (FN, TP)")
        print(f"\n   TP (True Positives):  {tp} - Correctly predicted maintenance")
        print(f"   TN (True Negatives):  {tn} - Correctly predicted no maintenance")
        print(f"   FP (False Positives): {fp} - False alarms (unnecessary maintenance)")
        print(f"   FN (False Negatives): {fn} - Missed failures (DANGEROUS!)")

        self.metrics['confusion_matrix'] = {
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
        }

        return self.metrics

    def calculate_regression_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics (for continuous values like RUL).

        Args:
            y_true: Actual values (continuous)
            y_pred: Predicted values (continuous)
        """
        print(f"\n{'='*70}")
        print(f"REGRESSION METRICS - {self.model_name}")
        print(f"{'='*70}")

        # 1. MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_true, y_pred)
        self.metrics['mae'] = mae
        print(f"\n1. MAE (Mean Absolute Error): {mae:.4f}")
        print(f"   - What: Average absolute difference between prediction and actual")
        print(f"   - Formula: (1/n) × Σ|y_true - y_pred|")
        print(f"   - Interpretation: On average, predictions are off by {mae:.2f} units")
        print(f"   - Good: Lower is better")

        # 2. MSE (Mean Squared Error)
        mse = mean_squared_error(y_true, y_pred)
        self.metrics['mse'] = mse
        print(f"\n2. MSE (Mean Squared Error): {mse:.4f}")
        print(f"   - What: Average squared difference (penalizes large errors more)")
        print(f"   - Formula: (1/n) × Σ(y_true - y_pred)²")
        print(f"   - Interpretation: Squared error = {mse:.2f}")
        print(f"   - Good: Lower is better")

        # 3. RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        self.metrics['rmse'] = rmse
        print(f"\n3. RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"   - What: Square root of MSE (same units as target)")
        print(f"   - Formula: √MSE")
        print(f"   - Interpretation: Typical prediction error = ±{rmse:.2f} units")
        print(f"   - Good: Lower is better")

        # 4. MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            self.metrics['mape'] = mape
            print(f"\n4. MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
            print(f"   - What: Average percentage error")
            print(f"   - Formula: (1/n) × Σ|(y_true - y_pred) / y_true| × 100")
            print(f"   - Interpretation: On average, {mape:.1f}% off from actual value")
            print(f"   - Good: Lower is better (< 10% excellent, < 20% good)")
        else:
            print(f"\n4. MAPE: Cannot calculate (division by zero)")

        # 5. R² (Coefficient of Determination)
        r2 = r2_score(y_true, y_pred)
        self.metrics['r2_score'] = r2
        print(f"\n5. R² (R-squared): {r2:.4f}")
        print(f"   - What: Proportion of variance explained by model")
        print(f"   - Range: -∞ to 1.0 (1.0 = perfect)")
        print(f"   - Interpretation: Model explains {r2*100:.1f}% of variance")
        print(f"   - Good: Closer to 1.0 is better")

        # 6. MAD (Median Absolute Deviation)
        mad = np.median(np.abs(y_true - y_pred))
        self.metrics['mad'] = mad
        print(f"\n6. MAD (Median Absolute Deviation): {mad:.4f}")
        print(f"   - What: Median of absolute errors (robust to outliers)")
        print(f"   - Formula: median(|y_true - y_pred|)")
        print(f"   - Interpretation: Typical error = {mad:.2f} units")
        print(f"   - Good: More robust than MAE to outliers")

        return self.metrics

    def calculate_business_metrics(self, y_true, y_pred):
        """
        Calculate business-relevant metrics.
        """
        print(f"\n{'='*70}")
        print(f"BUSINESS IMPACT METRICS - {self.model_name}")
        print(f"{'='*70}")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Cost assumptions (USD)
        COST_PREVENTIVE = 500        # Scheduled maintenance
        COST_CORRECTIVE = 2000       # Emergency repair
        COST_CRITICAL = 5000         # Complete breakdown
        COST_DOWNTIME_HOUR = 300     # Lost revenue per hour

        # Current costs (without prediction)
        baseline_failures = (y_true == 1).sum()
        baseline_cost = baseline_failures * (COST_CORRECTIVE + 4 * COST_DOWNTIME_HOUR)

        # Costs with prediction
        prevented_cost = tp * COST_PREVENTIVE  # Correct predictions → preventive
        missed_cost = fn * (COST_CORRECTIVE + 4 * COST_DOWNTIME_HOUR)  # Missed → breakdown
        false_alarm_cost = fp * COST_PREVENTIVE  # False alarms → unnecessary maintenance
        total_cost = prevented_cost + missed_cost + false_alarm_cost

        savings = baseline_cost - total_cost
        roi = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

        print(f"\n1. COST ANALYSIS:")
        print(f"   Baseline (without AI): ${baseline_cost:,.0f}")
        print(f"   With AI prediction:    ${total_cost:,.0f}")
        print(f"   Total savings:         ${savings:,.0f}")
        print(f"   ROI:                   {roi:.1f}%")

        print(f"\n2. FAILURE PREVENTION:")
        print(f"   Total failures:        {baseline_failures}")
        print(f"   Prevented:             {tp} ({tp/baseline_failures*100:.1f}%)")
        print(f"   Missed:                {fn} ({fn/baseline_failures*100:.1f}%)")

        print(f"\n3. OPERATIONAL EFFICIENCY:")
        print(f"   False alarms:          {fp}")
        print(f"   Correct dismissals:    {tn}")
        print(f"   Alert precision:       {tp/(tp+fp)*100:.1f}%")

        self.metrics['business'] = {
            'baseline_cost': float(baseline_cost),
            'predicted_cost': float(total_cost),
            'savings': float(savings),
            'roi_percent': float(roi),
            'failures_prevented': int(tp),
            'failures_missed': int(fn),
            'false_alarms': int(fp)
        }

        return self.metrics

    def save_metrics(self, filename):
        """Save all metrics to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\n✓ Metrics saved to {filename}")


def compare_models(lstm_results, gru_results):
    """
    Compare LSTM vs GRU models side-by-side.
    """
    print("\n" + "="*80)
    print("LSTM vs GRU - SIDE-BY-SIDE COMPARISON")
    print("="*80)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

    print(f"\n{'Metric':<20} {'LSTM':<15} {'GRU':<15} {'Winner':<15}")
    print("-"*80)

    for metric in metrics:
        lstm_val = lstm_results.get(metric, 0)
        gru_val = gru_results.get(metric, 0)

        if lstm_val > gru_val:
            winner = "LSTM ✓"
        elif gru_val > lstm_val:
            winner = "GRU ✓"
        else:
            winner = "Tie"

        print(f"{metric.upper():<20} {lstm_val:<15.4f} {gru_val:<15.4f} {winner:<15}")

    print("-"*80)

    # Overall assessment
    lstm_wins = sum([lstm_results.get(m, 0) > gru_results.get(m, 0) for m in metrics])
    gru_wins = sum([gru_results.get(m, 0) > lstm_results.get(m, 0) for m in metrics])

    print(f"\nOverall: LSTM wins {lstm_wins}/{len(metrics)} metrics")
    print(f"         GRU wins {gru_wins}/{len(metrics)} metrics")

    if lstm_wins > gru_wins:
        print(f"\n🏆 Winner: LSTM (better performance)")
    elif gru_wins > lstm_wins:
        print(f"\n🏆 Winner: GRU (better performance)")
    else:
        print(f"\n🏆 Tie (similar performance)")


# Example usage
if __name__ == "__main__":
    print("Comprehensive Metrics Calculator")
    print("Demonstrates all evaluation metrics for classification and regression\n")

    # Load actual results
    try:
        with open('model_results.json', 'r') as f:
            lstm_results = json.load(f)
        print("✓ Loaded LSTM results")
    except:
        lstm_results = {}
        print("⚠ LSTM results not found")

    try:
        with open('gru_model_results.json', 'r') as f:
            gru_results = json.load(f)
        print("✓ Loaded GRU results")
    except:
        gru_results = {}
        print("⚠ GRU results not found")

    if lstm_results and gru_results:
        compare_models(lstm_results, gru_results)

    print("\n" + "="*80)
    print("METRIC SELECTION GUIDE")
    print("="*80)
    print("""
FOR CLASSIFICATION (Yes/No prediction):
  - Accuracy:   Overall correctness
  - Precision:  Avoid false alarms (cost control)
  - Recall:     Catch all failures (safety)
  - F1-Score:   Balance precision & recall
  - AUC:        Overall discrimination ability
  ✓ USE: Accuracy, Precision, Recall, F1, AUC

FOR REGRESSION (Continuous values like RUL):
  - MAE:   Average error (easy to interpret)
  - MSE:   Squared error (penalizes large errors)
  - RMSE:  Root of MSE (same units as target)
  - MAPE:  Percentage error (scale-independent)
  - MAD:   Median error (robust to outliers)
  - R²:    Explained variance (goodness of fit)
  ✓ USE: MAE, RMSE, MAPE, R²

CURRENT PROJECT:
  Task: Binary classification (Maintenance: Yes/No)
  Using: Accuracy, Precision, Recall, F1, AUC ✓ CORRECT!

  If predicting RUL (days): Would use MAE, RMSE, MAPE instead.
""")
