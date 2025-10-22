"""
Complete Pipeline Runner
Execute all steps in sequence for hackathon demonstration

This script runs:
1. Maintenance pattern analysis
2. LSTM model training
3. GRU model training
4. Model comparison
5. Launches real-time dashboard

Use this for a complete end-to-end demonstration.
"""

import subprocess
import sys
import time
from datetime import datetime


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80 + "\n")


def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print_header(f"Step: {description}")
    print(f"⏳ Running {script_name}...")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )

        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed successfully!")
        print(f"⏱️  Time taken: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed!")
        print(f"⏱️  Time taken: {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False


def main():
    """Run complete pipeline."""
    print_header("EV FLEET PREDICTIVE MAINTENANCE - COMPLETE PIPELINE")
    print("Hackathon Challenge #2: Digital Twin Predictive Maintenance")
    print(f"\nPipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run the following steps:")
    print("  1. Maintenance Pattern Analysis (~2-5 minutes)")
    print("  2. LSTM Model Training (~10-30 minutes)")
    print("  3. GRU Model Training (~8-25 minutes)")
    print("  4. Model Comparison (~1 minute)")
    print("  5. Launch Real-time Dashboard (continuous)")
    print("\n⚠️  Note: Steps 2 & 3 can take 20-60 minutes total depending on hardware")
    print("\n" + "-"*80)

    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Pipeline cancelled by user")
        return

    # Track overall progress
    total_start = time.time()
    steps_completed = []
    steps_failed = []

    # Step 1: Maintenance Pattern Analysis
    if run_script('maintenance_analysis.py', 'Maintenance Pattern Analysis'):
        steps_completed.append('Maintenance Analysis')
    else:
        steps_failed.append('Maintenance Analysis')
        response = input("\n⚠️  Continue despite failure? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Pipeline stopped")
            return

    # Step 2: LSTM Model Training
    if run_script('lstm_predictive_maintenance.py', 'LSTM Model Training'):
        steps_completed.append('LSTM Training')
    else:
        steps_failed.append('LSTM Training')
        response = input("\n⚠️  Continue despite failure? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Pipeline stopped")
            return

    # Step 3: GRU Model Training
    if run_script('gru_predictive_maintenance.py', 'GRU Model Training'):
        steps_completed.append('GRU Training')
    else:
        steps_failed.append('GRU Training')
        response = input("\n⚠️  Continue despite failure? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Pipeline stopped")
            return

    # Step 4: Model Comparison
    if run_script('compare_models.py', 'Model Comparison'):
        steps_completed.append('Model Comparison')
    else:
        steps_failed.append('Model Comparison')

    # Summary
    total_elapsed = time.time() - total_start
    print_header("PIPELINE SUMMARY")

    print(f"✅ Completed Steps ({len(steps_completed)}):")
    for step in steps_completed:
        print(f"   ✓ {step}")

    if steps_failed:
        print(f"\n❌ Failed Steps ({len(steps_failed)}):")
        for step in steps_failed:
            print(f"   ✗ {step}")

    print(f"\n⏱️  Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 5: Launch Dashboard (optional)
    print("\n" + "="*80)
    print("All training steps complete!")
    print("="*80)

    print("\n📊 Generated Files:")
    print("   • maintenance_analysis_report.txt")
    print("   • best_lstm_model.keras")
    print("   • best_gru_model.keras")
    print("   • preprocessor.pkl")
    print("   • model_results.json")
    print("   • gru_model_results.json")
    print("   • model_comparison.png")
    print("   • model_comparison_report.md")
    print("   • Various visualization PNGs")

    print("\n" + "-"*80)
    response = input("\n🚀 Launch real-time dashboard now? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        print_header("Launching Real-time Dashboard")
        print("Dashboard will be available at:")
        print("   🌐 http://localhost:8000/dashboard")
        print("   📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the server")
        print("-"*80 + "\n")

        # Launch dashboard (this will run continuously)
        subprocess.run([sys.executable, 'realtime_api.py'])
    else:
        print("\n✅ Pipeline complete!")
        print("\nTo launch dashboard later, run:")
        print("   python realtime_api.py")
        print("\n" + "="*80)
        print("Ready for hackathon demonstration! 🏆")
        print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Pipeline error: {e}")
        sys.exit(1)
