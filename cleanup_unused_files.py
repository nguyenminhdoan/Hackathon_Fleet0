"""
Identify and clean up unused files in the project
"""
import os
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*70)
print("FILE CLEANUP ANALYSIS")
print("="*70)

# Files currently in use (KEEP THESE)
essential_files = {
    # Active model & data
    'best_improved_lstm.keras',           # Current production model
    'improved_preprocessor.pkl',          # Current preprocessor
    'improved_model_results.json',        # Current model metrics
    'EV_Predictive_Maintenance_Dataset_15min.csv',  # Training data

    # Active dashboard
    'enhanced_dashboard.py',              # Dashboard backend
    'dashboard.html',                     # Dashboard frontend

    # Training scripts
    'train_improved_model.py',            # Model training script

    # Utilities
    'recreate_preprocessor.py',           # Preprocessor regeneration
    'test_dashboard_setup.py',            # Validation script
    'RESTART_DASHBOARD.bat',              # Dashboard startup
    'START_DASHBOARD.bat',                # Alternative startup

    # Documentation
    'README.md',                          # Project documentation
    'requirements.txt',                   # Dependencies
    'project_update_email.txt',           # Project update email

    # Git & Config
    '.gitignore',
    '.git',
    '.claude',
    '__pycache__',
}

# Files that are SAFE TO DELETE (old/redundant versions)
unused_files = {
    # Old model versions
    'best_final_lstm.keras',              # Old model version
    'best_gru_model.keras',               # GRU model (not using)
    'best_lstm_model.keras',              # Old LSTM version
    'lstm_predictive_maintenance.keras',  # Old model file

    # Old preprocessors
    'final_preprocessor.pkl',             # Old preprocessor
    'gru_preprocessor.pkl',               # GRU preprocessor (not using)
    'preprocessor.pkl',                   # Old preprocessor

    # Old results/metrics
    'final_model_results.json',           # Old results
    'gru_model_results.json',             # GRU results (not using)
    'model_results.json',                 # Old results

    # Old training scripts
    'train_final_model.py',               # Old training script
    'train_gru_model.py',                 # GRU training (not using)
    'lstm_predictive_maintenance.py',     # Old LSTM script

    # Old analysis scripts
    'compare_all_models.py',              # One-time comparison done
    'comprehensive_metrics.py',           # One-time analysis done
    'threshold_analysis.py',              # One-time analysis done
    'maintenance_analysis.py',            # Old analysis

    # Old/redundant utilities
    'cleanup_project.py',                 # Old cleanup script
    'fix_preprocessor.py',                # One-time fix done

    # Old documentation
    'MODEL_COMPARISON.md',                # Old comparison report
    'model_comparison_report.md',         # Old report
    'MODEL_IMPROVEMENTS_SUMMARY.md',      # Old summary
    'PROJECT_STATUS.md',                  # Old status
    'DASHBOARD_GUIDE.md',                 # Redundant with README
    'TRAINING_GUIDE.md',                  # Redundant with README

    # Redundant data format
    'EV_Predictive_Maintenance_Dataset_15min.xlsx',  # CSV version is used

    # Old evaluation images (generated during training, can be regenerated)
    'confusion_matrix.png',
    'final_evaluation_results.png',
    'gru_evaluation.png',
    'gru_training_history.png',
    'pr_curve.png',
    'roc_curve.png',
    'training_history.png',
    'comprehensive_model_comparison.png',
}

print("\n[1] ESSENTIAL FILES (KEEP)")
print("-" * 70)
total_keep_size = 0
keep_count = 0
for f in sorted(essential_files):
    if os.path.exists(f) and os.path.isfile(f):
        size_mb = os.path.getsize(f) / 1024 / 1024
        total_keep_size += size_mb
        keep_count += 1
        print(f"  ✓ {f:<45} {size_mb:>8.2f} MB")

print(f"\nTotal: {keep_count} files, {total_keep_size:.2f} MB")

print("\n[2] UNUSED FILES (CAN DELETE)")
print("-" * 70)
total_delete_size = 0
delete_count = 0
for f in sorted(unused_files):
    if os.path.exists(f) and os.path.isfile(f):
        size_mb = os.path.getsize(f) / 1024 / 1024
        total_delete_size += size_mb
        delete_count += 1
        print(f"  × {f:<45} {size_mb:>8.2f} MB")

print(f"\nTotal: {delete_count} files, {total_delete_size:.2f} MB")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Essential files:  {keep_count} files ({total_keep_size:.2f} MB)")
print(f"Unused files:     {delete_count} files ({total_delete_size:.2f} MB)")
print(f"Space to free:    {total_delete_size:.2f} MB")

# Ask for confirmation
print("\n" + "="*70)
print("OPTIONS")
print("="*70)
print("\n1. Review only (this script)")
print("2. Delete unused files (run cleanup)")
print("3. Move to backup folder (safe option)")
print("\nTo DELETE unused files, run:")
print("  python cleanup_unused_files.py --delete")
print("\nTo MOVE to backup folder, run:")
print("  python cleanup_unused_files.py --backup")

import sys
if '--delete' in sys.argv:
    print("\n" + "="*70)
    print("DELETING UNUSED FILES...")
    print("="*70)
    deleted = 0
    for f in unused_files:
        if os.path.exists(f) and os.path.isfile(f):
            try:
                os.remove(f)
                print(f"  ✓ Deleted: {f}")
                deleted += 1
            except Exception as e:
                print(f"  ✗ Error deleting {f}: {e}")
    print(f"\n✓ Deleted {deleted} files, freed {total_delete_size:.2f} MB")

elif '--backup' in sys.argv:
    print("\n" + "="*70)
    print("MOVING TO BACKUP FOLDER...")
    print("="*70)

    backup_dir = 'backup_old_files'
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"Created backup directory: {backup_dir}/")

    moved = 0
    for f in unused_files:
        if os.path.exists(f) and os.path.isfile(f):
            try:
                import shutil
                shutil.move(f, os.path.join(backup_dir, f))
                print(f"  ✓ Moved: {f} -> {backup_dir}/")
                moved += 1
            except Exception as e:
                print(f"  ✗ Error moving {f}: {e}")
    print(f"\n✓ Moved {moved} files to {backup_dir}/")

print("\n" + "="*70)
