"""
Model Comparison Report Generator
Compares LSTM and GRU models with comprehensive visualizations

Generates:
1. Metrics comparison bar charts
2. Radar/Spider charts for multi-metric comparison
3. Confusion matrix heatmaps
4. Performance summary tables
5. Detailed HTML report
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==============================================================================
# Load Model Results
# ==============================================================================

def load_model_results():
    """Load results from both models."""
    print("Loading model results...")

    try:
        with open('improved_model_results.json', 'r') as f:
            lstm_results = json.load(f)
        print("✓ Loaded LSTM results")
    except FileNotFoundError:
        print("✗ LSTM results not found (improved_model_results.json)")
        lstm_results = None

    try:
        with open('gru_model_results.json', 'r') as f:
            gru_results = json.load(f)
        print("✓ Loaded GRU results")
    except FileNotFoundError:
        print("✗ GRU results not found (gru_model_results.json)")
        gru_results = None

    # Also try to load final model results
    try:
        with open('final_model_results.json', 'r') as f:
            final_results = json.load(f)
        print("✓ Loaded Final LSTM+Attention results")
    except FileNotFoundError:
        final_results = None

    return lstm_results, gru_results, final_results


# ==============================================================================
# Visualization 1: Metrics Comparison Bar Chart
# ==============================================================================

def plot_metrics_comparison(lstm_results, gru_results, final_results=None):
    """Create bar chart comparing key metrics."""
    print("\nGenerating metrics comparison chart...")

    models = []
    metrics_data = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'AUC-ROC': []
    }

    # Collect LSTM data
    if lstm_results:
        models.append('LSTM\n(Improved)')
        metrics_data['Accuracy'].append(lstm_results.get('accuracy', 0))
        metrics_data['Precision'].append(lstm_results.get('precision', 0))
        metrics_data['Recall'].append(lstm_results.get('recall', 0))
        metrics_data['F1-Score'].append(lstm_results.get('f1_score', 0))
        metrics_data['AUC-ROC'].append(lstm_results.get('auc', 0))

    # Collect GRU data
    if gru_results:
        models.append('GRU')
        if 'metrics' in gru_results:
            metrics_data['Accuracy'].append(gru_results['metrics'].get('accuracy', 0))
            metrics_data['Precision'].append(gru_results['metrics'].get('precision', 0))
            metrics_data['Recall'].append(gru_results['metrics'].get('recall', 0))
            metrics_data['F1-Score'].append(gru_results['metrics'].get('f1_score', 0))
            metrics_data['AUC-ROC'].append(gru_results['metrics'].get('auc_roc', 0))
        else:
            metrics_data['Accuracy'].append(gru_results.get('accuracy', 0))
            metrics_data['Precision'].append(gru_results.get('precision', 0))
            metrics_data['Recall'].append(gru_results.get('recall', 0))
            metrics_data['F1-Score'].append(gru_results.get('f1_score', 0))
            metrics_data['AUC-ROC'].append(gru_results.get('auc_roc', 0))

    # Collect Final model data
    if final_results:
        models.append('LSTM+\nAttention')
        if 'metrics' in final_results:
            metrics_data['Accuracy'].append(final_results['metrics'].get('accuracy', 0))
            metrics_data['Precision'].append(final_results['metrics'].get('precision', 0))
            metrics_data['Recall'].append(final_results['metrics'].get('recall', 0))
            metrics_data['F1-Score'].append(final_results['metrics'].get('f1_score', 0))
            metrics_data['AUC-ROC'].append(final_results['metrics'].get('auc_roc', 0))
        else:
            metrics_data['Accuracy'].append(final_results.get('accuracy', 0))
            metrics_data['Precision'].append(final_results.get('precision', 0))
            metrics_data['Recall'].append(final_results.get('recall', 0))
            metrics_data['F1-Score'].append(final_results.get('f1_score', 0))
            metrics_data['AUC-ROC'].append(final_results.get('auc_roc', 0))

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(models))
    width = 0.15
    multiplier = 0

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for (metric, values), color in zip(metrics_data.items(), colors):
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=metric, color=color, alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        multiplier += 1

    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison - Key Metrics',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random Baseline (0.5)')

    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: metrics_comparison.png")
    plt.close()


# ==============================================================================
# Visualization 2: Radar Chart
# ==============================================================================

def plot_radar_chart(lstm_results, gru_results, final_results=None):
    """Create radar chart for multi-metric comparison."""
    print("Generating radar chart...")

    # Prepare data
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    models_data = []
    model_names = []

    if lstm_results:
        model_names.append('LSTM (Improved)')
        models_data.append([
            lstm_results.get('accuracy', 0),
            lstm_results.get('precision', 0),
            lstm_results.get('recall', 0),
            lstm_results.get('f1_score', 0),
            lstm_results.get('auc', 0)
        ])

    if gru_results:
        model_names.append('GRU')
        if 'metrics' in gru_results:
            models_data.append([
                gru_results['metrics'].get('accuracy', 0),
                gru_results['metrics'].get('precision', 0),
                gru_results['metrics'].get('recall', 0),
                gru_results['metrics'].get('f1_score', 0),
                gru_results['metrics'].get('auc_roc', 0)
            ])
        else:
            models_data.append([
                gru_results.get('accuracy', 0),
                gru_results.get('precision', 0),
                gru_results.get('recall', 0),
                gru_results.get('f1_score', 0),
                gru_results.get('auc_roc', 0)
            ])

    if final_results:
        model_names.append('LSTM+Attention')
        if 'metrics' in final_results:
            models_data.append([
                final_results['metrics'].get('accuracy', 0),
                final_results['metrics'].get('precision', 0),
                final_results['metrics'].get('recall', 0),
                final_results['metrics'].get('f1_score', 0),
                final_results['metrics'].get('auc_roc', 0)
            ])

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, (data, name) in enumerate(zip(models_data, model_names)):
        data += data[:1]  # Complete the circle
        ax.plot(angles, data, 'o-', linewidth=2, label=name, color=colors[idx])
        ax.fill(angles, data, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Model Performance Radar Chart',
                 fontsize=16, fontweight='bold', pad=30, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig('radar_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: radar_comparison.png")
    plt.close()


# ==============================================================================
# Visualization 3: Confusion Matrix Comparison
# ==============================================================================

def plot_confusion_matrices(lstm_results, gru_results, final_results=None):
    """Create side-by-side confusion matrices."""
    print("Generating confusion matrix comparison...")

    models = []
    cms = []

    if lstm_results and 'confusion_matrix' not in lstm_results:
        # Calculate from metrics if available
        # Skip if not available
        pass

    # Collect confusion matrices
    if lstm_results:
        models.append('LSTM (Improved)')
        if 'confusion_matrix' in lstm_results:
            cm = lstm_results['confusion_matrix']
            cms.append(np.array([
                [cm.get('true_negatives', 0), cm.get('false_positives', 0)],
                [cm.get('false_negatives', 0), cm.get('true_positives', 0)]
            ]))

    if gru_results:
        models.append('GRU')
        if 'confusion_matrix' in gru_results:
            cm = gru_results['confusion_matrix']
            cms.append(np.array([
                [cm.get('true_negatives', 0), cm.get('false_positives', 0)],
                [cm.get('false_negatives', 0), cm.get('true_positives', 0)]
            ]))

    if final_results:
        models.append('LSTM+Attention')
        if 'confusion_matrix' in final_results:
            cm = final_results['confusion_matrix']
            cms.append(np.array([
                [cm.get('true_negatives', 0), cm.get('false_positives', 0)],
                [cm.get('false_negatives', 0), cm.get('true_positives', 0)]
            ]))

    if not cms:
        print("⚠ No confusion matrices available")
        return

    # Create subplots
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, cm, model_name in zip(axes, cms, models):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        ax.set_title(f'{model_name}\nConfusion Matrix',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['No Maint.', 'Maint.'], fontsize=11)
        ax.set_yticklabels(['No Maint.', 'Maint.'], fontsize=11, rotation=0)

    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrices_comparison.png")
    plt.close()


# ==============================================================================
# Visualization 4: Detailed Metrics Table
# ==============================================================================

def create_metrics_table(lstm_results, gru_results, final_results=None):
    """Create detailed comparison table."""
    print("Generating metrics table...")

    data = []

    # LSTM
    if lstm_results:
        data.append({
            'Model': 'LSTM (Improved)',
            'Accuracy': f"{lstm_results.get('accuracy', 0):.4f}",
            'Precision': f"{lstm_results.get('precision', 0):.4f}",
            'Recall': f"{lstm_results.get('recall', 0):.4f}",
            'F1-Score': f"{lstm_results.get('f1_score', 0):.4f}",
            'AUC-ROC': f"{lstm_results.get('auc', 0):.4f}",
            'Parameters': f"{lstm_results.get('total_parameters', 0):,}",
            'Threshold': f"{lstm_results.get('best_threshold', 0.5):.2f}"
        })

    # GRU
    if gru_results:
        if 'metrics' in gru_results:
            metrics = gru_results['metrics']
            params = gru_results.get('architecture', {}).get('total_parameters', 0)
        else:
            metrics = gru_results
            params = gru_results.get('total_parameters', 0)

        data.append({
            'Model': 'GRU',
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'Precision': f"{metrics.get('precision', 0):.4f}",
            'Recall': f"{metrics.get('recall', 0):.4f}",
            'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
            'AUC-ROC': f"{metrics.get('auc_roc', 0):.4f}",
            'Parameters': f"{params:,}",
            'Threshold': f"{gru_results.get('best_threshold', 0.5):.2f}"
        })

    # Final
    if final_results:
        if 'metrics' in final_results:
            metrics = final_results['metrics']
        else:
            metrics = final_results

        data.append({
            'Model': 'LSTM+Attention',
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'Precision': f"{metrics.get('precision', 0):.4f}",
            'Recall': f"{metrics.get('recall', 0):.4f}",
            'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
            'AUC-ROC': f"{metrics.get('auc_roc', 0):.4f}",
            'Parameters': f"{final_results.get('total_parameters', 0):,}",
            'Threshold': f"{final_results.get('best_threshold', 0.5):.2f}"
        })

    df = pd.DataFrame(data)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colColours=['#3498db']*len(df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

    # Style cells
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            table[(i, j)].set_text_props(fontsize=10)

    plt.title('Detailed Model Comparison Table',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('metrics_table.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: metrics_table.png")
    plt.close()

    # Also save as CSV
    df.to_csv('model_comparison_table.csv', index=False)
    print("✓ Saved: model_comparison_table.csv")

    return df


# ==============================================================================
# Visualization 5: Performance Heatmap
# ==============================================================================

def plot_performance_heatmap(lstm_results, gru_results, final_results=None):
    """Create heatmap showing all metrics for all models."""
    print("Generating performance heatmap...")

    models = []
    metrics_matrix = []

    if lstm_results:
        models.append('LSTM\n(Improved)')
        metrics_matrix.append([
            lstm_results.get('accuracy', 0),
            lstm_results.get('precision', 0),
            lstm_results.get('recall', 0),
            lstm_results.get('f1_score', 0),
            lstm_results.get('auc', 0)
        ])

    if gru_results:
        models.append('GRU')
        if 'metrics' in gru_results:
            metrics_matrix.append([
                gru_results['metrics'].get('accuracy', 0),
                gru_results['metrics'].get('precision', 0),
                gru_results['metrics'].get('recall', 0),
                gru_results['metrics'].get('f1_score', 0),
                gru_results['metrics'].get('auc_roc', 0)
            ])
        else:
            metrics_matrix.append([
                gru_results.get('accuracy', 0),
                gru_results.get('precision', 0),
                gru_results.get('recall', 0),
                gru_results.get('f1_score', 0),
                gru_results.get('auc_roc', 0)
            ])

    if final_results:
        models.append('LSTM+\nAttention')
        if 'metrics' in final_results:
            metrics_matrix.append([
                final_results['metrics'].get('accuracy', 0),
                final_results['metrics'].get('precision', 0),
                final_results['metrics'].get('recall', 0),
                final_results['metrics'].get('f1_score', 0),
                final_results['metrics'].get('auc_roc', 0)
            ])

    metrics_matrix = np.array(metrics_matrix)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics_names, fontsize=12, fontweight='bold')
    ax.set_yticklabels(models, fontsize=12, fontweight='bold')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add values
    for i in range(len(models)):
        for j in range(len(metrics_names)):
            text = ax.text(j, i, f'{metrics_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black",
                          fontsize=11, fontweight='bold')

    ax.set_title('Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_heatmap.png")
    plt.close()


# ==============================================================================
# Visualization 6: Model Complexity vs Performance
# ==============================================================================

def plot_complexity_vs_performance(lstm_results, gru_results, final_results=None):
    """Plot model parameters vs performance metrics."""
    print("Generating complexity vs performance chart...")

    models_data = []

    if lstm_results:
        models_data.append({
            'name': 'LSTM',
            'params': lstm_results.get('total_parameters', 0) / 1000,
            'f1': lstm_results.get('f1_score', 0),
            'auc': lstm_results.get('auc', 0)
        })

    if gru_results:
        if 'architecture' in gru_results:
            params = gru_results['architecture'].get('total_parameters', 0)
        else:
            params = gru_results.get('total_parameters', 0)

        if 'metrics' in gru_results:
            models_data.append({
                'name': 'GRU',
                'params': params / 1000,
                'f1': gru_results['metrics'].get('f1_score', 0),
                'auc': gru_results['metrics'].get('auc_roc', 0)
            })
        else:
            models_data.append({
                'name': 'GRU',
                'params': params / 1000,
                'f1': gru_results.get('f1_score', 0),
                'auc': gru_results.get('auc_roc', 0)
            })

    if final_results:
        if 'metrics' in final_results:
            models_data.append({
                'name': 'LSTM+Attn',
                'params': final_results.get('total_parameters', 0) / 1000,
                'f1': final_results['metrics'].get('f1_score', 0),
                'auc': final_results['metrics'].get('auc_roc', 0)
            })

    if len(models_data) < 2:
        print("⚠ Need at least 2 models for comparison")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # F1-Score vs Parameters
    for model in models_data:
        ax1.scatter(model['params'], model['f1'], s=200, alpha=0.6,
                   label=model['name'])
        ax1.text(model['params'], model['f1'], model['name'],
                fontsize=10, fontweight='bold', ha='center', va='bottom')

    ax1.set_xlabel('Model Parameters (thousands)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Complexity vs F1-Score', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # AUC-ROC vs Parameters
    for model in models_data:
        ax2.scatter(model['params'], model['auc'], s=200, alpha=0.6,
                   label=model['name'])
        ax2.text(model['params'], model['auc'], model['name'],
                fontsize=10, fontweight='bold', ha='center', va='bottom')

    ax2.set_xlabel('Model Parameters (thousands)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax2.set_title('Model Complexity vs AUC-ROC', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('complexity_vs_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: complexity_vs_performance.png")
    plt.close()


# ==============================================================================
# Generate HTML Report
# ==============================================================================

def generate_html_report(lstm_results, gru_results, final_results, df_table):
    """Generate comprehensive HTML report."""
    print("Generating HTML report...")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 40px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 10px;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 20px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .metric-card h3 {{
                margin: 0;
                font-size: 14px;
                opacity: 0.9;
            }}
            .metric-card .value {{
                font-size: 32px;
                font-weight: bold;
                margin: 10px 0;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .summary {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
            }}
            .badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
                margin: 5px;
            }}
            .badge-success {{ background-color: #2ecc71; color: white; }}
            .badge-warning {{ background-color: #f39c12; color: white; }}
            .badge-info {{ background-color: #3498db; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 EV Predictive Maintenance Model Comparison Report</h1>

            <div class="summary">
                <h2>📊 Executive Summary</h2>
                <p><strong>Report Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Models Compared:</strong>
                    {' LSTM (Improved) |' if lstm_results else ''}
                    {' GRU |' if gru_results else ''}
                    {' LSTM+Attention |' if final_results else ''}
                </p>
                <p><strong>Dataset:</strong> EV Predictive Maintenance Dataset (15-min intervals)</p>
                <p><strong>Task:</strong> Binary classification - Predict maintenance needs 1 hour ahead</p>
            </div>

            <h2>🎯 Key Metrics Comparison</h2>
            {df_table.to_html(index=False, classes='table')}

            <h2>📈 Visual Comparisons</h2>

            <h3>Overall Metrics Comparison</h3>
            <img src="metrics_comparison.png" alt="Metrics Comparison">

            <h3>Multi-Dimensional Performance</h3>
            <img src="radar_comparison.png" alt="Radar Comparison">

            <h3>Performance Heatmap</h3>
            <img src="performance_heatmap.png" alt="Performance Heatmap">

            <h3>Confusion Matrices</h3>
            <img src="confusion_matrices_comparison.png" alt="Confusion Matrices">

            <h3>Model Efficiency Analysis</h3>
            <img src="complexity_vs_performance.png" alt="Complexity vs Performance">

            <h2>🔍 Detailed Analysis</h2>
    """

    # Add model-specific analysis
    if lstm_results:
        html += f"""
            <h3>LSTM (Improved) Model</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Recall</h3>
                    <div class="value">{lstm_results.get('recall', 0):.1%}</div>
                    <span class="badge badge-{'success' if lstm_results.get('recall', 0) > 0.8 else 'warning'}">
                        {'High' if lstm_results.get('recall', 0) > 0.8 else 'Moderate'}
                    </span>
                </div>
                <div class="metric-card">
                    <h3>Precision</h3>
                    <div class="value">{lstm_results.get('precision', 0):.1%}</div>
                    <span class="badge badge-{'success' if lstm_results.get('precision', 0) > 0.5 else 'warning'}">
                        {'High' if lstm_results.get('precision', 0) > 0.5 else 'Low'}
                    </span>
                </div>
                <div class="metric-card">
                    <h3>AUC-ROC</h3>
                    <div class="value">{lstm_results.get('auc', 0):.3f}</div>
                    <span class="badge badge-{'success' if lstm_results.get('auc', 0) > 0.7 else 'warning'}">
                        {'Good' if lstm_results.get('auc', 0) > 0.7 else 'Moderate'}
                    </span>
                </div>
                <div class="metric-card">
                    <h3>F1-Score</h3>
                    <div class="value">{lstm_results.get('f1_score', 0):.1%}</div>
                </div>
            </div>
        """

    if gru_results:
        metrics = gru_results.get('metrics', gru_results)
        html += f"""
            <h3>GRU Model</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Recall</h3>
                    <div class="value">{metrics.get('recall', 0):.1%}</div>
                    <span class="badge badge-{'success' if metrics.get('recall', 0) > 0.8 else 'warning'}">
                        {'High' if metrics.get('recall', 0) > 0.8 else 'Moderate'}
                    </span>
                </div>
                <div class="metric-card">
                    <h3>Precision</h3>
                    <div class="value">{metrics.get('precision', 0):.1%}</div>
                    <span class="badge badge-{'success' if metrics.get('precision', 0) > 0.5 else 'warning'}">
                        {'High' if metrics.get('precision', 0) > 0.5 else 'Low'}
                    </span>
                </div>
                <div class="metric-card">
                    <h3>AUC-ROC</h3>
                    <div class="value">{metrics.get('auc_roc', 0):.3f}</div>
                    <span class="badge badge-{'success' if metrics.get('auc_roc', 0) > 0.7 else 'warning'}">
                        {'Good' if metrics.get('auc_roc', 0) > 0.7 else 'Moderate'}
                    </span>
                </div>
                <div class="metric-card">
                    <h3>F1-Score</h3>
                    <div class="value">{metrics.get('f1_score', 0):.1%}</div>
                </div>
            </div>
        """

    html += """
            <h2>💡 Recommendations</h2>
            <div class="summary">
                <ul>
                    <li><strong>For Safety-Critical Applications:</strong> Use the model with highest recall to minimize missed failures</li>
                    <li><strong>For Cost Optimization:</strong> Focus on the model with best precision to reduce false alarms</li>
                    <li><strong>For Balanced Performance:</strong> Choose the model with highest F1-score</li>
                    <li><strong>For Overall Discrimination:</strong> Select the model with highest AUC-ROC</li>
                </ul>
            </div>

            <div class="footer">
                <p>Generated by Model Comparison Report Tool</p>
                <p>EV Predictive Maintenance System | 2025</p>
            </div>
        </div>
    </body>
    </html>
    """

    with open('model_comparison_report.html', 'w') as f:
        f.write(html)

    print("✓ Saved: model_comparison_report.html")


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    print("="*80)
    print("MODEL COMPARISON REPORT GENERATOR")
    print("="*80)

    # Load results
    lstm_results, gru_results, final_results = load_model_results()

    if not lstm_results and not gru_results:
        print("\n❌ Error: No model results found!")
        print("Please ensure you have trained the models and generated result files.")
        return

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Generate all visualizations
    plot_metrics_comparison(lstm_results, gru_results, final_results)
    plot_radar_chart(lstm_results, gru_results, final_results)
    plot_confusion_matrices(lstm_results, gru_results, final_results)
    df_table = create_metrics_table(lstm_results, gru_results, final_results)
    plot_performance_heatmap(lstm_results, gru_results, final_results)
    plot_complexity_vs_performance(lstm_results, gru_results, final_results)

    # Generate HTML report
    generate_html_report(lstm_results, gru_results, final_results, df_table)

    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("   - metrics_comparison.png")
    print("   - radar_comparison.png")
    print("   - confusion_matrices_comparison.png")
    print("   - metrics_table.png")
    print("   - performance_heatmap.png")
    print("   - complexity_vs_performance.png")
    print("   - model_comparison_table.csv")
    print("   - model_comparison_report.html")
    print("\n🌐 Open 'model_comparison_report.html' in your browser to view the full report!")


if __name__ == "__main__":
    main()
