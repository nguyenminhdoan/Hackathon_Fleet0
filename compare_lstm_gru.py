"""
LSTM vs GRU Comparison Analysis
Comprehensive comparison of both models with visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

# Create images folder
os.makedirs('images', exist_ok=True)

print("="*70)
print("LSTM vs GRU COMPARISON ANALYSIS")
print("="*70)

# Load results
print("\nLoading model results...")

try:
    with open('improved_model_results.json', 'r') as f:
        lstm_results = json.load(f)
    print("[OK] Loaded LSTM results")
except FileNotFoundError:
    print("[ERROR] LSTM results not found. Please run train_improved_model.py first")
    exit(1)

try:
    with open('improved_gru_results.json', 'r') as f:
        gru_results = json.load(f)
    print("[OK] Loaded GRU results")
except FileNotFoundError:
    print("[ERROR] GRU results not found. Please run train_improved_gru.py first")
    exit(1)

# Extract metrics
models = ['LSTM', 'GRU']
metrics = {
    'Accuracy': [
        lstm_results['accuracy'],
        gru_results['accuracy']
    ],
    'Precision': [
        lstm_results['precision'],
        gru_results['precision']
    ],
    'Recall': [
        lstm_results['recall'],
        gru_results['recall']
    ],
    'F1-Score': [
        lstm_results['f1_score'],
        gru_results['f1_score']
    ],
    'AUC': [
        lstm_results['auc'],
        gru_results['auc']
    ]
}

parameters = [
    lstm_results['total_parameters'],
    gru_results['total_parameters']
]

thresholds = [
    lstm_results['best_threshold'],
    gru_results['best_threshold']
]

# ==============================================================================
# VISUALIZATION 1: Metrics Comparison Bar Chart
# ==============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('LSTM vs GRU - Comprehensive Comparison', fontsize=16, fontweight='bold')

colors = ['#3498db', '#e74c3c']  # Blue for LSTM, Red for GRU

# Plot each metric
metric_names = list(metrics.keys())
for idx, (metric_name, values) in enumerate(metrics.items()):
    row = idx // 3
    col = idx % 3

    bars = axes[row, col].bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[row, col].set_title(f'{metric_name} Comparison', fontweight='bold', fontsize=12)
    axes[row, col].set_ylabel(metric_name, fontsize=11)
    axes[row, col].set_ylim([0, 1])
    axes[row, col].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}\n({value*100:.2f}%)',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight better model
    if values[0] > values[1]:
        axes[row, col].text(0, values[0] * 0.95, '★ WINNER', ha='center', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))
    elif values[1] > values[0]:
        axes[row, col].text(1, values[1] * 0.95, '★ WINNER', ha='center', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))

# Parameters comparison
axes[1, 2].bar(models, [p/1e6 for p in parameters], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 2].set_title('Model Parameters (Millions)', fontweight='bold', fontsize=12)
axes[1, 2].set_ylabel('Parameters (M)', fontsize=11)
axes[1, 2].grid(axis='y', alpha=0.3)

for i, (model, param) in enumerate(zip(models, parameters)):
    axes[1, 2].text(i, param/1e6, f'{param/1e6:.2f}M\n({param:,})',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('images/lstm_gru_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: images/lstm_gru_comparison.png")

# ==============================================================================
# VISUALIZATION 2: Radar Chart
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Metrics for radar
radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
lstm_values = [metrics[m][0] for m in radar_metrics]
gru_values = [metrics[m][1] for m in radar_metrics]

# Number of variables
N = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
lstm_values += lstm_values[:1]
gru_values += gru_values[:1]
angles += angles[:1]

# Plot
ax.plot(angles, lstm_values, 'o-', linewidth=2, label='LSTM', color='#3498db')
ax.fill(angles, lstm_values, alpha=0.25, color='#3498db')

ax.plot(angles, gru_values, 'o-', linewidth=2, label='GRU', color='#e74c3c')
ax.fill(angles, gru_values, alpha=0.25, color='#e74c3c')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=12)
ax.set_ylim(0, 1)
ax.set_title('LSTM vs GRU - Performance Radar', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True)

plt.tight_layout()
plt.savefig('images/lstm_gru_radar.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: images/lstm_gru_radar.png")

# ==============================================================================
# VISUALIZATION 3: Side-by-side Metrics Table
# ==============================================================================

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = []
table_data.append(['Metric', 'LSTM', 'GRU', 'Difference', 'Winner'])
table_data.append(['='*15, '='*12, '='*12, '='*12, '='*10])

for metric_name, values in metrics.items():
    lstm_val, gru_val = values
    diff = lstm_val - gru_val
    winner = 'LSTM ★' if diff > 0 else ('GRU ★' if diff < 0 else 'TIE')

    table_data.append([
        metric_name,
        f'{lstm_val:.4f} ({lstm_val*100:.2f}%)',
        f'{gru_val:.4f} ({gru_val*100:.2f}%)',
        f'{diff:+.4f} ({diff*100:+.2f}%)',
        winner
    ])

table_data.append(['='*15, '='*12, '='*12, '='*12, '='*10])
table_data.append([
    'Parameters',
    f'{parameters[0]:,}',
    f'{parameters[1]:,}',
    f'{parameters[0]-parameters[1]:+,}',
    'Lower is better'
])
table_data.append([
    'Best Threshold',
    f'{thresholds[0]:.2f}',
    f'{thresholds[1]:.2f}',
    f'{thresholds[0]-thresholds[1]:+.2f}',
    '-'
])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.2, 0.2, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(2, len(table_data)):
    for j in range(5):
        if table_data[i][4].startswith('LSTM'):
            table[(i, 1)].set_facecolor('#d6eaf8')  # Light blue
        elif table_data[i][4].startswith('GRU'):
            table[(i, 2)].set_facecolor('#fadbd8')  # Light red

plt.title('LSTM vs GRU - Detailed Comparison Table', fontsize=14, fontweight='bold', pad=20)
plt.savefig('images/lstm_gru_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: images/lstm_gru_table.png")

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

print("\nModel Performance:")
print(f"{'Metric':<15} {'LSTM':<15} {'GRU':<15} {'Winner':<10}")
print("-"*70)

for metric_name, values in metrics.items():
    lstm_val, gru_val = values
    winner = 'LSTM ★' if lstm_val > gru_val else ('GRU ★' if gru_val > lstm_val else 'TIE')
    print(f"{metric_name:<15} {lstm_val*100:>6.2f}%       {gru_val*100:>6.2f}%       {winner}")

print("-"*70)
print(f"{'Parameters':<15} {parameters[0]:>10,}   {parameters[1]:>10,}")
print(f"{'Threshold':<15} {thresholds[0]:>13.2f}   {thresholds[1]:>13.2f}")

# Overall winner
lstm_wins = sum(1 for values in metrics.values() if values[0] > values[1])
gru_wins = sum(1 for values in metrics.values() if values[1] > values[0])

print("\n" + "="*70)
print("OVERALL WINNER")
print("="*70)

if lstm_wins > gru_wins:
    print(f"\n★★★ LSTM WINS ★★★")
    print(f"LSTM wins {lstm_wins}/{len(metrics)} metrics")
elif gru_wins > lstm_wins:
    print(f"\n★★★ GRU WINS ★★★")
    print(f"GRU wins {gru_wins}/{len(metrics)} metrics")
else:
    print(f"\n★★★ TIE ★★★")
    print(f"Both models win {lstm_wins}/{len(metrics)} metrics each")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

# Most important metric for this use case
print(f"\n1. RECALL (Most Important for Predictive Maintenance):")
print(f"   - LSTM: {metrics['Recall'][0]*100:.2f}% - Catches {metrics['Recall'][0]*100:.0f} out of 100 failures")
print(f"   - GRU:  {metrics['Recall'][1]*100:.2f}% - Catches {metrics['Recall'][1]*100:.0f} out of 100 failures")
if metrics['Recall'][0] > metrics['Recall'][1]:
    print(f"   → LSTM is better at catching failures")
else:
    print(f"   → GRU is better at catching failures")

print(f"\n2. F1-SCORE (Balance of Precision and Recall):")
print(f"   - LSTM: {metrics['F1-Score'][0]*100:.2f}%")
print(f"   - GRU:  {metrics['F1-Score'][1]*100:.2f}%")

print(f"\n3. MODEL COMPLEXITY:")
print(f"   - LSTM: {parameters[0]:,} parameters ({parameters[0]/1e6:.2f}M)")
print(f"   - GRU:  {parameters[1]:,} parameters ({parameters[1]/1e6:.2f}M)")
param_diff_pct = ((parameters[0] - parameters[1]) / parameters[1]) * 100
if parameters[0] > parameters[1]:
    print(f"   → LSTM has {abs(param_diff_pct):.1f}% more parameters than GRU")
else:
    print(f"   → GRU has {abs(param_diff_pct):.1f}% more parameters than LSTM")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if metrics['Recall'][0] > metrics['Recall'][1]:
    print("\nRecommended Model: LSTM")
    print("Reason: Higher recall - better at catching failures before they occur")
    print("This is critical for predictive maintenance where missing a failure")
    print("can cost $5,000 vs $500 for preventive maintenance.")
else:
    print("\nRecommended Model: GRU")
    print("Reason: Higher recall - better at catching failures before they occur")
    print("This is critical for predictive maintenance where missing a failure")
    print("can cost $5,000 vs $500 for preventive maintenance.")

print("\n" + "="*70)
print("All comparison visualizations saved to images/ folder:")
print("  - images/lstm_gru_comparison.png")
print("  - images/lstm_gru_radar.png")
print("  - images/lstm_gru_table.png")
print("="*70)
