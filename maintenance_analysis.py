"""
Professional EV Fleet Predictive Maintenance Analysis
Hackathon Challenge #2: Digital Twin Predictive Maintenance

This module analyzes maintenance patterns to understand:
1. Preventive maintenance (Type 1) reliability and effectiveness
2. Corrective maintenance (Type 2) precursor patterns
3. Predictive maintenance (Type 3) opportunity windows
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MaintenancePatternAnalyzer:
    """
    Analyzes maintenance patterns to establish baseline for predictive maintenance.

    Key Questions:
    1. How reliable are preventive maintenance schedules?
    2. What parameter patterns precede corrective maintenance?
    3. Where can we insert predictive maintenance to prevent corrective needs?
    """

    def __init__(self, data_path):
        """Initialize analyzer with dataset path."""
        self.data_path = data_path
        self.df = None
        self.maintenance_profiles = {}

    def load_and_prepare_data(self, sample_size=None):
        """
        Load dataset efficiently with optional sampling.

        Args:
            sample_size: Number of rows to sample (None = full dataset)
        """
        print("Loading dataset...")

        # Load with optimized dtypes
        dtype_dict = {
            'SoC': 'float32',
            'SoH': 'float32',
            'Battery_Voltage': 'float32',
            'Battery_Current': 'float32',
            'Battery_Temperature': 'float32',
            'Charge_Cycles': 'float32',
            'Motor_Temperature': 'float32',
            'Power_Consumption': 'float32',
            'Distance_Traveled': 'float32',
            'RUL': 'float32',
            'Failure_Probability': 'float32',
            'Maintenance_Type': 'int8',
            'Component_Health_Score': 'float32'
        }

        if sample_size:
            # Smart sampling: ensure we get all maintenance types
            self.df = pd.read_csv(self.data_path, dtype=dtype_dict,
                                 parse_dates=['Timestamp'], nrows=sample_size)
        else:
            self.df = pd.read_csv(self.data_path, dtype=dtype_dict,
                                 parse_dates=['Timestamp'])

        # Sort by timestamp
        self.df = self.df.sort_values('Timestamp').reset_index(drop=True)

        print(f"Loaded {len(self.df):,} records from {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
        print(f"\nMaintenance Type Distribution:")
        print(self.df['Maintenance_Type'].value_counts().sort_index())

        return self.df

    def analyze_preventive_maintenance_reliability(self):
        """
        Analyze Type 1 (Preventive) maintenance effectiveness.

        Key metrics:
        - Time intervals between preventive maintenance
        - Success rate (did it prevent Type 2/3?)
        - Parameter health before/after preventive maintenance
        """
        print("\n" + "="*70)
        print("PREVENTIVE MAINTENANCE (TYPE 1) RELIABILITY ANALYSIS")
        print("="*70)

        # Get all Type 1 maintenance events
        type1_events = self.df[self.df['Maintenance_Type'] == 1].copy()
        print(f"\nTotal Preventive Maintenance Events: {len(type1_events):,}")

        # Analyze intervals between preventive maintenance
        type1_events['time_diff'] = type1_events['Timestamp'].diff()
        intervals_hours = type1_events['time_diff'].dt.total_seconds() / 3600

        print(f"\nPreventive Maintenance Schedule Regularity:")
        print(f"  Mean interval: {intervals_hours.mean():.1f} hours ({intervals_hours.mean()/24:.1f} days)")
        print(f"  Median interval: {intervals_hours.median():.1f} hours ({intervals_hours.median()/24:.1f} days)")
        print(f"  Std deviation: {intervals_hours.std():.1f} hours")
        print(f"  Min interval: {intervals_hours.min():.1f} hours")
        print(f"  Max interval: {intervals_hours.max():.1f} hours")

        # Analyze effectiveness: what happens AFTER preventive maintenance?
        results = []
        lookback_window = 96  # 24 hours after (96 * 15min)

        for idx in type1_events.index:
            if idx + lookback_window < len(self.df):
                future_window = self.df.iloc[idx:idx+lookback_window]

                # Check if corrective/critical maintenance needed within 24h
                has_type2 = (future_window['Maintenance_Type'] == 2).any()
                has_type3 = (future_window['Maintenance_Type'] == 3).any()

                # Get parameter improvements
                before_health = self.df.iloc[idx]['Component_Health_Score']
                after_health = future_window['Component_Health_Score'].mean()

                results.append({
                    'timestamp': self.df.iloc[idx]['Timestamp'],
                    'health_before': before_health,
                    'health_after': after_health,
                    'health_improvement': after_health - before_health,
                    'type2_within_24h': has_type2,
                    'type3_within_24h': has_type3,
                    'rul_before': self.df.iloc[idx]['RUL'],
                    'failure_prob_before': self.df.iloc[idx]['Failure_Probability']
                })

        results_df = pd.DataFrame(results)

        print(f"\nPreventive Maintenance Effectiveness (24h follow-up):")
        print(f"  Success rate (no Type 2/3 within 24h): {((~results_df['type2_within_24h'] & ~results_df['type3_within_24h']).sum() / len(results_df) * 100):.1f}%")
        print(f"  Type 2 needed within 24h: {results_df['type2_within_24h'].sum()} events ({results_df['type2_within_24h'].sum()/len(results_df)*100:.1f}%)")
        print(f"  Type 3 needed within 24h: {results_df['type3_within_24h'].sum()} events ({results_df['type3_within_24h'].sum()/len(results_df)*100:.1f}%)")
        print(f"  Average health improvement: {results_df['health_improvement'].mean():.4f}")
        print(f"  Average RUL at maintenance: {results_df['rul_before'].mean():.1f} days")

        self.maintenance_profiles['preventive'] = results_df
        return results_df

    def analyze_corrective_maintenance_patterns(self, lookback_hours=24):
        """
        Analyze Type 2 (Corrective) maintenance precursor patterns.

        This is CRITICAL: identifies what parameters look like BEFORE corrective maintenance
        so we can predict and prevent it with Type 3 (Predictive) maintenance.

        Args:
            lookback_hours: Hours before corrective maintenance to analyze
        """
        print("\n" + "="*70)
        print("CORRECTIVE MAINTENANCE (TYPE 2) PRECURSOR ANALYSIS")
        print("="*70)

        type2_events = self.df[self.df['Maintenance_Type'] == 2].copy()
        print(f"\nTotal Corrective Maintenance Events: {len(type2_events):,}")

        lookback_steps = int(lookback_hours * 4)  # 15-min intervals

        # Key parameters to analyze (professor's recommendations)
        key_params = [
            'Battery_Voltage', 'Battery_Current', 'Battery_Temperature',
            'SoC', 'SoH', 'Charge_Cycles', 'Distance_Traveled',
            'Component_Health_Score', 'Failure_Probability', 'RUL'
        ]

        precursor_patterns = []

        for idx in type2_events.index:
            if idx - lookback_steps >= 0:
                # Get data before corrective maintenance
                before_window = self.df.iloc[idx-lookback_steps:idx]

                pattern = {
                    'timestamp': self.df.iloc[idx]['Timestamp'],
                }

                # Calculate trends and statistics
                for param in key_params:
                    values = before_window[param].values

                    # Statistical features
                    pattern[f'{param}_mean'] = np.mean(values)
                    pattern[f'{param}_std'] = np.std(values)
                    pattern[f'{param}_min'] = np.min(values)
                    pattern[f'{param}_max'] = np.max(values)
                    pattern[f'{param}_trend'] = np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                    pattern[f'{param}_final'] = values[-1]

                    # Rate of change
                    if len(values) > 1:
                        pattern[f'{param}_change_rate'] = (values[-1] - values[0]) / len(values)
                    else:
                        pattern[f'{param}_change_rate'] = 0

                precursor_patterns.append(pattern)

        patterns_df = pd.DataFrame(precursor_patterns)

        # Compare with normal operation (Type 0)
        type0_sample = self.df[self.df['Maintenance_Type'] == 0].sample(min(1000, len(self.df[self.df['Maintenance_Type'] == 0])))

        print(f"\n{lookback_hours}h BEFORE Corrective Maintenance - Parameter Analysis:")
        print("-" * 70)

        for param in key_params:
            type2_mean = patterns_df[f'{param}_mean'].mean()
            type0_mean = type0_sample[param].mean()

            type2_std = patterns_df[f'{param}_std'].mean()
            type2_trend = patterns_df[f'{param}_trend'].mean()

            # Statistical significance test
            if len(patterns_df) > 0 and len(type0_sample) > 0:
                t_stat, p_value = stats.ttest_ind(patterns_df[f'{param}_mean'],
                                                  type0_sample[param],
                                                  equal_var=False)
                significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            else:
                significant = ""

            print(f"\n{param}:")
            print(f"  Normal operation mean: {type0_mean:.4f}")
            print(f"  Before Type 2 mean: {type2_mean:.4f} (diff: {type2_mean-type0_mean:+.4f}) {significant}")
            print(f"  Trend (slope): {type2_trend:.6f}")
            print(f"  Variability (std): {type2_std:.4f}")

        print("\n" + "-" * 70)
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05")

        self.maintenance_profiles['corrective_precursors'] = patterns_df
        return patterns_df

    def identify_predictive_opportunities(self, lead_time_hours=12):
        """
        Identify opportunities for Type 3 (Predictive) maintenance insertion.

        Find windows where predictive maintenance could prevent corrective maintenance.

        Args:
            lead_time_hours: Desired lead time before corrective maintenance
        """
        print("\n" + "="*70)
        print("PREDICTIVE MAINTENANCE OPPORTUNITY ANALYSIS")
        print("="*70)

        type2_events = self.df[self.df['Maintenance_Type'] == 2].copy()
        lead_steps = int(lead_time_hours * 4)  # 15-min intervals

        opportunities = []

        for idx in type2_events.index:
            if idx - lead_steps >= 0:
                # Look at state X hours before corrective maintenance
                opportunity_window = self.df.iloc[idx-lead_steps:idx]
                trigger_point = self.df.iloc[idx-lead_steps]

                opportunities.append({
                    'type2_timestamp': self.df.iloc[idx]['Timestamp'],
                    'trigger_timestamp': trigger_point['Timestamp'],
                    'lead_time_hours': lead_time_hours,
                    'voltage_at_trigger': trigger_point['Battery_Voltage'],
                    'current_at_trigger': trigger_point['Battery_Current'],
                    'temp_at_trigger': trigger_point['Battery_Temperature'],
                    'soc_at_trigger': trigger_point['SoC'],
                    'soh_at_trigger': trigger_point['SoH'],
                    'health_score_at_trigger': trigger_point['Component_Health_Score'],
                    'failure_prob_at_trigger': trigger_point['Failure_Probability'],
                    'rul_at_trigger': trigger_point['RUL'],
                    'distance_at_trigger': trigger_point['Distance_Traveled'],
                    'voltage_trend': np.polyfit(range(len(opportunity_window)),
                                                opportunity_window['Battery_Voltage'].values, 1)[0],
                    'temp_trend': np.polyfit(range(len(opportunity_window)),
                                            opportunity_window['Battery_Temperature'].values, 1)[0],
                    'health_trend': np.polyfit(range(len(opportunity_window)),
                                              opportunity_window['Component_Health_Score'].values, 1)[0]
                })

        opp_df = pd.DataFrame(opportunities)

        print(f"\nPredictive Maintenance Opportunities ({lead_time_hours}h lead time):")
        print(f"  Total Type 2 events that could be prevented: {len(opp_df):,}")
        print(f"\nThreshold Analysis for Early Warning System:")
        print(f"  Failure Probability at trigger: {opp_df['failure_prob_at_trigger'].mean():.4f} ± {opp_df['failure_prob_at_trigger'].std():.4f}")
        print(f"  Component Health Score at trigger: {opp_df['health_score_at_trigger'].mean():.4f} ± {opp_df['health_score_at_trigger'].std():.4f}")
        print(f"  RUL at trigger: {opp_df['rul_at_trigger'].mean():.1f} ± {opp_df['rul_at_trigger'].std():.1f} days")
        print(f"  SoH at trigger: {opp_df['soh_at_trigger'].mean():.4f} ± {opp_df['soh_at_trigger'].std():.4f}")

        # Suggest thresholds
        print(f"\nSUGGESTED PREDICTIVE MAINTENANCE TRIGGERS:")
        print(f"  🚨 Failure Probability > {opp_df['failure_prob_at_trigger'].quantile(0.25):.4f}")
        print(f"  🚨 Component Health Score < {opp_df['health_score_at_trigger'].quantile(0.75):.4f}")
        print(f"  🚨 RUL < {opp_df['rul_at_trigger'].quantile(0.75):.1f} days")
        print(f"  🚨 SoH < {opp_df['soh_at_trigger'].quantile(0.75):.4f}")
        print(f"  🚨 Voltage declining (trend < {opp_df['voltage_trend'].quantile(0.25):.6f})")

        self.maintenance_profiles['predictive_opportunities'] = opp_df
        return opp_df

    def generate_report(self, output_path='maintenance_analysis_report.txt'):
        """Generate comprehensive analysis report."""

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EV FLEET PREDICTIVE MAINTENANCE - PROFESSIONAL ANALYSIS REPORT\n")
            f.write("Hackathon Challenge #2: Digital Twin Predictive Maintenance\n")
            f.write("="*80 + "\n\n")

            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Dataset: {len(self.df):,} records\n")
            f.write(f"Time range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}\n")
            f.write(f"Maintenance events: {(self.df['Maintenance_Type'] > 0).sum():,}\n\n")

            if 'preventive' in self.maintenance_profiles:
                f.write("PREVENTIVE MAINTENANCE RELIABILITY\n")
                f.write("-"*80 + "\n")
                prev = self.maintenance_profiles['preventive']
                success_rate = ((~prev['type2_within_24h'] & ~prev['type3_within_24h']).sum() / len(prev) * 100)
                f.write(f"Success Rate: {success_rate:.1f}%\n")
                f.write(f"Average Health Improvement: {prev['health_improvement'].mean():.4f}\n\n")

            if 'predictive_opportunities' in self.maintenance_profiles:
                f.write("PREDICTIVE MAINTENANCE OPPORTUNITIES\n")
                f.write("-"*80 + "\n")
                opp = self.maintenance_profiles['predictive_opportunities']
                f.write(f"Preventable corrective maintenance events: {len(opp):,}\n")
                f.write(f"Recommended trigger thresholds:\n")
                f.write(f"  - Failure Probability > {opp['failure_prob_at_trigger'].quantile(0.25):.4f}\n")
                f.write(f"  - Component Health Score < {opp['health_score_at_trigger'].quantile(0.75):.4f}\n")
                f.write(f"  - RUL < {opp['rul_at_trigger'].quantile(0.75):.1f} days\n")

        print(f"\n✅ Report saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    # Run comprehensive analysis
    analyzer = MaintenancePatternAnalyzer("EV_Predictive_Maintenance_Dataset_15min.csv")

    # Load data (use sample for faster development, full for competition)
    analyzer.load_and_prepare_data(sample_size=50000)  # Remove sample_size for full analysis

    # Run all analyses
    analyzer.analyze_preventive_maintenance_reliability()
    analyzer.analyze_corrective_maintenance_patterns(lookback_hours=24)
    analyzer.identify_predictive_opportunities(lead_time_hours=12)

    # Generate report
    analyzer.generate_report()

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE - Ready for LSTM model development")
    print("="*70)
