"""
Test if dashboard files and model are properly configured
"""

import os
import json
import joblib

print("="*70)
print("DASHBOARD SETUP VERIFICATION")
print("="*70)

errors = []
warnings = []
success = []

# Check 1: Model file exists
print("\n[1/5] Checking model file...")
if os.path.exists('best_improved_lstm.keras'):
    size_mb = os.path.getsize('best_improved_lstm.keras') / 1024 / 1024
    success.append(f"Model file found: {size_mb:.1f} MB")
    print(f"  [OK] best_improved_lstm.keras ({size_mb:.1f} MB)")
else:
    errors.append("Model file not found: best_improved_lstm.keras")
    print("  [ERROR] best_improved_lstm.keras not found")

# Check 2: Preprocessor file exists and is valid
print("\n[2/5] Checking preprocessor...")
if os.path.exists('improved_preprocessor.pkl'):
    try:
        prep_data = joblib.load('improved_preprocessor.pkl')
        if isinstance(prep_data, dict):
            has_scaler = 'scaler' in prep_data
            has_features = 'feature_cols' in prep_data
            feature_count = len(prep_data.get('feature_cols', []))

            print(f"  [OK] Preprocessor loaded")
            print(f"      - Has scaler: {has_scaler}")
            print(f"      - Has feature_cols: {has_features}")
            print(f"      - Feature count: {feature_count}")

            if not has_scaler:
                warnings.append("Preprocessor missing 'scaler' field")
            if not has_features:
                warnings.append("Preprocessor missing 'feature_cols' field")
            if feature_count == 0:
                warnings.append("Preprocessor has 0 features")

            success.append(f"Preprocessor valid: {feature_count} features")
        else:
            warnings.append("Preprocessor is not a dictionary")
            print(f"  [WARN] Preprocessor is {type(prep_data)}, expected dict")
    except Exception as e:
        errors.append(f"Cannot load preprocessor: {e}")
        print(f"  [ERROR] Cannot load: {e}")
else:
    errors.append("Preprocessor file not found")
    print("  [ERROR] improved_preprocessor.pkl not found")

# Check 3: Model results file
print("\n[3/5] Checking model results...")
if os.path.exists('improved_model_results.json'):
    try:
        with open('improved_model_results.json', 'r') as f:
            results = json.load(f)

        required_fields = ['model_type', 'recall', 'f1_score', 'accuracy', 'precision', 'auc']
        missing = [f for f in required_fields if f not in results]

        print(f"  [OK] Results file loaded")
        print(f"      - Model type: {results.get('model_type', 'unknown')}")
        print(f"      - Recall: {results.get('recall', 0)*100:.2f}%")
        print(f"      - F1-Score: {results.get('f1_score', 0)*100:.2f}%")
        print(f"      - Total parameters: {results.get('total_parameters', 0):,}")

        if missing:
            warnings.append(f"Results missing fields: {missing}")
        else:
            success.append(f"Results complete: Recall {results.get('recall', 0)*100:.2f}%")

    except Exception as e:
        errors.append(f"Cannot load results: {e}")
        print(f"  [ERROR] Cannot load: {e}")
else:
    errors.append("Results file not found")
    print("  [ERROR] improved_model_results.json not found")

# Check 4: Dashboard HTML
print("\n[4/5] Checking dashboard HTML...")
if os.path.exists('dashboard.html'):
    size_kb = os.path.getsize('dashboard.html') / 1024
    success.append(f"Dashboard HTML found: {size_kb:.1f} KB")
    print(f"  [OK] dashboard.html ({size_kb:.1f} KB)")
else:
    errors.append("dashboard.html not found")
    print("  [ERROR] dashboard.html not found")

# Check 5: Dashboard Python script
print("\n[5/5] Checking dashboard script...")
if os.path.exists('enhanced_dashboard.py'):
    size_kb = os.path.getsize('enhanced_dashboard.py') / 1024
    success.append(f"Dashboard script found: {size_kb:.1f} KB")
    print(f"  [OK] enhanced_dashboard.py ({size_kb:.1f} KB)")
else:
    errors.append("enhanced_dashboard.py not found")
    print("  [ERROR] enhanced_dashboard.py not found")

# Check 6: Dataset
print("\n[6/6] Checking dataset...")
if os.path.exists('EV_Predictive_Maintenance_Dataset_15min.csv'):
    size_mb = os.path.getsize('EV_Predictive_Maintenance_Dataset_15min.csv') / 1024 / 1024
    success.append(f"Dataset found: {size_mb:.1f} MB")
    print(f"  [OK] Dataset ({size_mb:.1f} MB)")
else:
    warnings.append("Dataset not found (dashboard will fail)")
    print("  [WARN] Dataset not found")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nSuccess ({len(success)}):")
for s in success:
    print(f"  + {s}")

if warnings:
    print(f"\nWarnings ({len(warnings)}):")
    for w in warnings:
        print(f"  ! {w}")

if errors:
    print(f"\nErrors ({len(errors)}):")
    for e in errors:
        print(f"  X {e}")
else:
    print("\nNo errors found!")

print("\n" + "="*70)

if errors:
    print("STATUS: ERRORS FOUND - Fix before running dashboard")
    print("\nTo fix:")
    print("  1. Run: python train_improved_model.py")
    print("  2. Then: python enhanced_dashboard.py")
elif warnings:
    print("STATUS: WARNINGS - Dashboard may not work correctly")
    print("\nRecommended:")
    print("  1. Run: python fix_preprocessor.py")
    print("  2. Then restart dashboard")
else:
    print("STATUS: ALL CHECKS PASSED - Ready to run!")
    print("\nStart dashboard:")
    print("  python enhanced_dashboard.py")
    print("  Then open: http://localhost:8001/dashboard")

print("="*70)
