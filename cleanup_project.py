"""
Cleanup unnecessary files from the project
Keeps only essential files for hackathon
"""

import os
import shutil

print("="*70)
print("PROJECT CLEANUP")
print("="*70)

# Files to DELETE (redundant/outdated)
files_to_delete = [
    # Old training scripts (replaced by train_improved_model.py)
    'train_lstm_quick.py',

    # Old models (replaced by improved model)
    'best_lstm_model.keras',
    'lstm_predictive_maintenance.keras',
    'best_gru_model.keras',
    'gru_predictive_maintenance.keras',

    # Old preprocessors
    'preprocessor.pkl',

    # Old comparison script (replaced by compare_all_models.py)
    'compare_models.py',

    # Old API (replaced by enhanced_dashboard.py)
    'realtime_api.py',

    # Old images (replaced by comprehensive comparison)
    'confusion_matrix.png',
    'pr_curve.png',
    'roc_curve.png',
    'training_history.png',
    'gru_confusion_matrix.png',
    'gru_pr_curve.png',
    'gru_roc_curve.png',
    'gru_training_history.png',
    'model_comparison.png',

    # Old results
    'model_results.json',

    # Old analysis
    'maintenance_analysis_report.txt',

    # Old documentation (consolidated into new files)
    'TRAINING_STATUS.md',
    'QUICKSTART.md',
    'run_complete_pipeline.py',

    # Empty file
    'main.py',
]

# Directories to clean
dirs_to_clean = [
    '__pycache__',
]

# Count deletions
deleted_files = 0
deleted_dirs = 0
total_size_freed = 0

print("\nDeleting unnecessary files...\n")

for file in files_to_delete:
    filepath = file
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        try:
            os.remove(filepath)
            deleted_files += 1
            total_size_freed += size
            print(f"[DELETE] {file} ({size/1024:.1f} KB)")
        except Exception as e:
            print(f"[ERROR] Could not delete {file}: {e}")
    else:
        print(f"[SKIP] {file} (not found)")

print("\nDeleting unnecessary directories...\n")

for dir_name in dirs_to_clean:
    if os.path.exists(dir_name):
        try:
            shutil.rmtree(dir_name)
            deleted_dirs += 1
            print(f"[DELETE] {dir_name}/")
        except Exception as e:
            print(f"[ERROR] Could not delete {dir_name}: {e}")

print("\n" + "="*70)
print("CLEANUP SUMMARY")
print("="*70)
print(f"Files deleted:       {deleted_files}")
print(f"Directories deleted: {deleted_dirs}")
print(f"Space freed:         {total_size_freed/1024/1024:.1f} MB")
print("="*70)

print("\n" + "="*70)
print("REMAINING ESSENTIAL FILES")
print("="*70)
print("""
📊 DATASET:
  - EV_Predictive_Maintenance_Dataset_15min.csv (92 MB)
  - EV_Predictive_Maintenance_Dataset_15min.xlsx (73 MB) - OPTIONAL

🤖 TRAINED MODELS:
  - best_improved_lstm.keras (7 MB) - BEST MODEL (100% recall)
  - improved_preprocessor.pkl (3 KB)
  - improved_model_results.json (568 B)
  - gru_model_results.json (357 B) - for comparison

📈 ANALYSIS SCRIPTS:
  - maintenance_analysis.py - Analyze maintenance patterns
  - comprehensive_metrics.py - Metric calculations
  - threshold_analysis.py - Threshold optimization
  - compare_all_models.py - Model comparison

🚀 TRAINING:
  - train_improved_model.py - MAIN TRAINING SCRIPT (100% recall)
  - lstm_predictive_maintenance.py - Original LSTM (for reference)
  - gru_predictive_maintenance.py - GRU model (for comparison)

🌐 DASHBOARD:
  - enhanced_dashboard.py - MAIN DASHBOARD SERVER
  - dashboard.html - Frontend UI

📄 DOCUMENTATION:
  - README.md - Project overview
  - DASHBOARD_GUIDE.md - Dashboard usage
  - MODEL_IMPROVEMENTS_SUMMARY.md - Model improvements explained
  - model_comparison_report.md - LSTM vs GRU comparison
  - requirements.txt - Python dependencies

📊 VISUALIZATIONS:
  - comprehensive_model_comparison.png - All models compared

⚙️ CONFIGURATION:
  - .gitignore - Git ignore rules
  - cleanup_project.py - This cleanup script
""")

print("\n" + "="*70)
print("✅ CLEANUP COMPLETE!")
print("="*70)
print("\nYour project is now clean and ready for:")
print("  1. GitHub upload")
print("  2. Hackathon presentation")
print("  3. Production deployment")
print("\nNext steps:")
print("  • To retrain: python train_improved_model.py")
print("  • To run dashboard: python enhanced_dashboard.py")
print("  • To compare models: python compare_all_models.py")
print("="*70)
