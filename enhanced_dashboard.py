"""
Enhanced Real-time Predictive Maintenance Dashboard
Features:
- Auto-refresh every 15 seconds
- Comprehensive metrics & analytics
- Manual prediction input form
- Model performance visualization
- Live simulation with real LSTM model
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import asyncio
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import json
from typing import List, Optional
import uvicorn
import io

app = FastAPI(title="EV Fleet Predictive Maintenance Dashboard",
              description="Enhanced real-time monitoring with analytics",
              version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class EnhancedFleetSimulator:
    def __init__(self):
        self.df = None
        self.current_index = 0
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.feature_cols = None
        self.prediction_history = []
        self.sequence_length = 24
        self.model_metrics = {}

    def load_data(self, csv_path='EV_Predictive_Maintenance_Dataset_15min.csv', nrows=50000):
        """Load fleet data for simulation."""
        print("Loading fleet data...")
        try:
            self.df = pd.read_csv(csv_path, parse_dates=['Timestamp'], nrows=nrows)
            self.current_index = self.sequence_length  # Start after one sequence
            print(f"[OK] Loaded {len(self.df):,} records")
            return True
        except Exception as e:
            print(f"[WARNING] Error loading data: {e}")
            return False

    def load_model(self, model_path='best_improved_lstm.keras', preprocessor_path='improved_preprocessor.pkl'):
        """Load trained LSTM model and preprocessor."""
        try:
            print("Loading improved model...")
            self.model = tf.keras.models.load_model(model_path, compile=False)

            # Load preprocessor
            preprocessor_data = joblib.load(preprocessor_path)

            # Handle different preprocessor structures
            if isinstance(preprocessor_data, dict):
                self.scaler = preprocessor_data.get('scaler')
                # Try both 'feature_columns' and 'feature_cols' for compatibility
                self.feature_cols = preprocessor_data.get('feature_columns', preprocessor_data.get('feature_cols', []))
            else:
                # Old format
                self.scaler = preprocessor_data
                self.feature_cols = []

            # Load model metrics
            try:
                with open('improved_model_results.json', 'r') as f:
                    self.model_metrics = json.load(f)
                    print(f"[OK] Loaded metrics: {self.model_metrics.get('model_type', 'unknown')}")
            except Exception as e:
                print(f"[WARNING] Model metrics not found: {e}")
                self.model_metrics = {}

            print(f"[OK] Model loaded: {len(self.feature_cols)} features")
            return True
        except Exception as e:
            print(f"[WARNING] Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            print("Trying original model...")
            try:
                self.model = tf.keras.models.load_model('best_lstm_model.keras', compile=False)
                preprocessor_data = joblib.load('preprocessor.pkl')
                if isinstance(preprocessor_data, dict):
                    self.scaler = preprocessor_data.get('scaler')
                    self.feature_cols = preprocessor_data.get('feature_columns', preprocessor_data.get('feature_cols', []))
                else:
                    self.scaler = preprocessor_data
                    self.feature_cols = []

                # Try to load original metrics
                try:
                    with open('model_results.json', 'r') as f:
                        self.model_metrics = json.load(f)
                except:
                    self.model_metrics = {}

                print("[OK] Original model loaded")
                return True
            except:
                print("[WARNING] No model found - using simulation mode")
                self.model_metrics = {}
                return False

    def engineer_features(self, df_subset):
        """Engineer features EXACTLY as training script."""
        df = df_subset.copy()

        # 1. RATE OF CHANGE FEATURES (velocity)
        for col in ['Battery_Voltage', 'Battery_Temperature', 'SoC', 'SoH',
                    'Component_Health_Score', 'Failure_Probability']:
            df[f'{col}_diff_1'] = df[col].diff(1).fillna(0)
            df[f'{col}_diff_4'] = df[col].diff(4).fillna(0)

        # 2. ROLLING STATISTICS (12 steps = 3 hours)
        window = 12
        for col in ['Battery_Voltage', 'Battery_Current', 'Battery_Temperature', 'SoC']:
            df[f'{col}_roll_mean'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_roll_std'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
            df[f'{col}_roll_max'] = df[col].rolling(window=window, min_periods=1).max()
            df[f'{col}_roll_min'] = df[col].rolling(window=window, min_periods=1).min()

        # 3. INTERACTION FEATURES
        df['Voltage_Temp_Ratio'] = df['Battery_Voltage'] / (df['Battery_Temperature'] + 1e-6)
        df['Current_Temp_Product'] = df['Battery_Current'] * df['Battery_Temperature']
        df['SoC_SoH_Product'] = df['SoC'] * df['SoH']
        df['Power_Estimate'] = df['Battery_Voltage'] * df['Battery_Current']

        # 4. HEALTH DEGRADATION FEATURES
        df['Health_Decline'] = df['Component_Health_Score'].diff(4).fillna(0)
        df['SoH_Decline'] = df['SoH'].diff(4).fillna(0)
        df['Risk_Increase'] = df['Failure_Probability'].diff(4).fillna(0)

        # 5. STRESS INDICATORS
        df['High_Current_Flag'] = (df['Battery_Current'].abs() > df['Battery_Current'].abs().quantile(0.75)).astype(int)
        df['High_Temp_Flag'] = (df['Battery_Temperature'] > df['Battery_Temperature'].quantile(0.75)).astype(int)
        df['Low_Health_Flag'] = (df['Component_Health_Score'] < df['Component_Health_Score'].quantile(0.25)).astype(int)

        # 6. CUMULATIVE FEATURES
        df['Cumulative_Charge_Cycles'] = df['Charge_Cycles']
        df['Cumulative_Distance'] = df['Distance_Traveled']

        return df

    def get_current_sequence(self):
        """Get current sequence for prediction."""
        if self.df is None or self.current_index < self.sequence_length:
            return None

        # Get sequence window
        start_idx = self.current_index - self.sequence_length
        end_idx = self.current_index

        sequence_df = self.df.iloc[start_idx:end_idx].copy()

        # Engineer features
        sequence_df = self.engineer_features(sequence_df)

        # Select feature columns
        available_cols = [col for col in self.feature_cols if col in sequence_df.columns]

        if len(available_cols) < len(self.feature_cols):
            # Use base features if engineered not available
            base_features = ['Battery_Voltage', 'Battery_Current', 'Battery_Temperature',
                           'SoC', 'SoH', 'Charge_Cycles', 'Distance_Traveled',
                           'Power_Consumption', 'Component_Health_Score',
                           'Failure_Probability', 'RUL']
            available_cols = [col for col in base_features if col in sequence_df.columns]

        sequence_data = sequence_df[available_cols].values

        # Scale
        if self.scaler is not None:
            try:
                sequence_data = self.scaler.transform(sequence_data)
            except:
                pass

        # Reshape for LSTM: (1, sequence_length, features)
        sequence_data = sequence_data.reshape(1, self.sequence_length, -1)

        return sequence_data

    def predict_maintenance(self):
        """Make real prediction using loaded model."""
        if self.model is None:
            # Fallback to simulation
            probability = float(np.random.random())
            return self._format_prediction(probability, simulated=True)

        try:
            # Get sequence
            sequence = self.get_current_sequence()
            if sequence is None:
                return self._format_prediction(0.5, simulated=True)

            # Make prediction
            prediction = self.model.predict(sequence, verbose=0)
            probability = float(prediction[0][0])

            # Store in history
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'probability': probability,
                'index': self.current_index
            })

            # Keep last 100
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]

            return self._format_prediction(probability, simulated=False)

        except Exception as e:
            print(f"Prediction error: {e}")
            return self._format_prediction(0.5, simulated=True)

    def _format_prediction(self, probability, simulated=False):
        """Format prediction result with cost analysis."""
        # Apply optimal threshold (0.3 from training)
        optimal_threshold = 0.3
        maintenance_needed = probability >= optimal_threshold

        # Cost assumptions (USD)
        COST_PREVENTIVE = 500        # Scheduled maintenance
        COST_CORRECTIVE = 2000       # Emergency repair
        COST_CRITICAL = 5000         # Complete breakdown
        COST_DOWNTIME_HOUR = 300     # Lost revenue per hour

        # Calculate expected costs based on probability and severity
        if probability > 0.7:
            severity = 'CRITICAL'
            recommendation = 'CRITICAL: Schedule immediate maintenance (within 4 hours)'
            confidence = 'high'
            # Expected cost if no action: probability of failure × (critical repair + downtime)
            expected_failure_cost = probability * (COST_CRITICAL + 8 * COST_DOWNTIME_HOUR)
            # Cost if we do preventive maintenance now
            preventive_cost = COST_PREVENTIVE
            # Savings = what we avoid - what we spend
            potential_savings = expected_failure_cost - preventive_cost
            action_recommendation = 'IMMEDIATE preventive maintenance required'

        elif probability > 0.4:
            severity = 'WARNING'
            recommendation = 'WARNING: Schedule maintenance within 24 hours'
            confidence = 'medium'
            expected_failure_cost = probability * (COST_CORRECTIVE + 4 * COST_DOWNTIME_HOUR)
            preventive_cost = COST_PREVENTIVE
            potential_savings = expected_failure_cost - preventive_cost
            action_recommendation = 'Schedule preventive maintenance soon' if potential_savings > 0 else 'Monitor closely'

        elif probability > 0.2:
            severity = 'MONITOR'
            recommendation = 'MONITOR: Continue normal operations, schedule routine check'
            confidence = 'medium'
            expected_failure_cost = probability * (COST_CORRECTIVE + 2 * COST_DOWNTIME_HOUR)
            preventive_cost = COST_PREVENTIVE
            potential_savings = expected_failure_cost - preventive_cost
            action_recommendation = 'Consider preventive maintenance' if potential_savings > 100 else 'Continue monitoring'

        else:
            severity = 'NORMAL'
            recommendation = 'NORMAL: No action required'
            confidence = 'high'
            expected_failure_cost = probability * COST_CORRECTIVE
            preventive_cost = 0  # No maintenance needed, so no cost
            potential_savings = 0  # No action needed
            action_recommendation = 'No maintenance required - continue normal operations'

        # Calculate cost efficiency ratio (how much we save per dollar spent)
        if preventive_cost > 0:
            cost_benefit_ratio = expected_failure_cost / preventive_cost
        else:
            cost_benefit_ratio = 0

        return {
            'maintenance_needed': maintenance_needed,
            'probability': probability,
            'severity': severity,
            'confidence': confidence,
            'recommendation': recommendation,
            'model_loaded': not simulated,
            'threshold': optimal_threshold,
            'cost_analysis': {
                'preventive_maintenance_cost': float(preventive_cost),
                'expected_cost_if_no_action': float(expected_failure_cost),
                'potential_savings': float(potential_savings),
                'cost_benefit_ratio': float(cost_benefit_ratio),
                'action_recommendation': action_recommendation
            }
        }

    def get_current_data(self):
        """Get current vehicle state."""
        if self.df is None or self.current_index >= len(self.df):
            return {}

        current_row = self.df.iloc[self.current_index]
        return current_row.to_dict()

    def advance_time(self):
        """Move to next timestep."""
        self.current_index += 1
        if self.current_index >= len(self.df) - 1:
            self.current_index = self.sequence_length  # Loop back

    def get_analytics(self):
        """Get comprehensive analytics."""
        if self.df is None:
            return {}

        # Recent data (last 1000 records)
        recent_df = self.df.iloc[max(0, self.current_index - 1000):self.current_index]

        analytics = {
            'total_records': len(self.df),
            'current_index': self.current_index,
            'progress_percent': (self.current_index / len(self.df)) * 100,

            'battery_stats': {
                'avg_voltage': float(recent_df['Battery_Voltage'].mean()),
                'avg_current': float(recent_df['Battery_Current'].mean()),
                'avg_temperature': float(recent_df['Battery_Temperature'].mean()),
                'avg_soc': float(recent_df['SoC'].mean()),
                'avg_soh': float(recent_df['SoH'].mean()),
            },

            'health_stats': {
                'avg_component_health': float(recent_df['Component_Health_Score'].mean()),
                'avg_failure_probability': float(recent_df['Failure_Probability'].mean()),
                'avg_rul': float(recent_df['RUL'].mean()),
            },

            'maintenance_events': {
                'total': int((recent_df['Maintenance_Type'] > 0).sum()),
                'preventive': int((recent_df['Maintenance_Type'] == 1).sum()),
                'corrective': int((recent_df['Maintenance_Type'] == 2).sum()),
                'predictive': int((recent_df['Maintenance_Type'] == 3).sum()),
            },

            'prediction_stats': {
                'total_predictions': len(self.prediction_history),
                'avg_probability': np.mean([p['probability'] for p in self.prediction_history]) if self.prediction_history else 0,
                'high_risk_count': sum(1 for p in self.prediction_history if p['probability'] > 0.7),
            }
        }

        return analytics


# Initialize simulator
simulator = EnhancedFleetSimulator()

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    simulator.load_data()
    simulator.load_model()


@app.get("/")
async def root():
    """API root."""
    return {
        "message": "Enhanced EV Fleet Predictive Maintenance Dashboard",
        "version": "2.0.0",
        "features": [
            "Auto-refresh every 15 seconds",
            "Real-time LSTM predictions",
            "Manual prediction testing",
            "Comprehensive analytics",
            "Model performance metrics"
        ],
        "endpoints": {
            "dashboard": "/dashboard",
            "status": "/api/vehicle/status",
            "predict": "/api/predict",
            "analytics": "/api/analytics",
            "model_info": "/api/model/info",
            "manual_predict": "/api/predict/manual",
            "csv_predict": "/api/predict/csv (POST with file upload - requires 24 rows)"
        }
    }


@app.get("/api/vehicle/status")
async def get_vehicle_status():
    """Get current vehicle status."""
    current_data = simulator.get_current_data()

    if not current_data:
        return {"error": "No data available"}

    return {
        "timestamp": str(current_data.get('Timestamp', datetime.now())),
        "battery": {
            "voltage": float(current_data.get('Battery_Voltage', 0)),
            "current": float(current_data.get('Battery_Current', 0)),
            "temperature": float(current_data.get('Battery_Temperature', 0)),
            "soc": float(current_data.get('SoC', 0)),
            "soh": float(current_data.get('SoH', 0)),
        },
        "vehicle": {
            "distance_traveled": float(current_data.get('Distance_Traveled', 0)),
            "power_consumption": float(current_data.get('Power_Consumption', 0)),
            "driving_speed": float(current_data.get('Driving_Speed', 0)),
            "charge_cycles": int(current_data.get('Charge_Cycles', 0)),
        },
        "health": {
            "component_health_score": float(current_data.get('Component_Health_Score', 0)),
            "failure_probability": float(current_data.get('Failure_Probability', 0)),
            "rul_days": float(current_data.get('RUL', 0)),
        },
        "maintenance_type": int(current_data.get('Maintenance_Type', 0))
    }


@app.get("/api/predict")
async def predict_maintenance():
    """Make AI prediction for current state."""
    prediction = simulator.predict_maintenance()
    current_data = simulator.get_current_data()

    return {
        "timestamp": str(datetime.now()),
        "prediction": prediction,
        "current_state": {
            "failure_probability": float(current_data.get('Failure_Probability', 0)),
            "rul_days": float(current_data.get('RUL', 0)),
            "component_health": float(current_data.get('Component_Health_Score', 0)),
        }
    }


@app.post("/api/predict/manual")
async def predict_manual(
    battery_voltage: float = Form(...),
    battery_current: float = Form(...),
    battery_temperature: float = Form(...),
    soc: float = Form(...),
    soh: float = Form(...),
    charge_cycles: int = Form(0),
    distance_traveled: float = Form(0),
    power_consumption: float = Form(0),
    component_health_score: float = Form(0.8),
    failure_probability: float = Form(0.1),
    rul: float = Form(30)
):
    """Make manual prediction with user input - creates realistic sequence with gradual degradation."""

    # Create a realistic sequence showing degradation over 24 timesteps
    # This simulates how the values would change over time leading to the current state
    sequence_rows = []

    # Add small random variations based on input values to make predictions unique
    np.random.seed(int(battery_voltage * 100 + battery_current * 10 + battery_temperature + soc * 1000 + soh * 1000) % 2**32)

    for i in range(simulator.sequence_length):
        # Calculate progress through sequence (0.0 to 1.0)
        progress = i / (simulator.sequence_length - 1)

        # Add slight random noise that varies based on the timestep
        noise = np.random.uniform(-0.02, 0.02)

        # The last few timesteps should be very close to actual input values
        if i >= simulator.sequence_length - 3:
            # Last 3 timesteps use actual input values with tiny variations
            blend = (i - (simulator.sequence_length - 3)) / 2.0  # 0.0, 0.5, 1.0 for last 3
            row = {
                'Battery_Voltage': battery_voltage * (0.98 + 0.02 * blend + noise * 0.5),
                'Battery_Current': battery_current * (0.98 + 0.02 * blend + noise * 0.5),
                'Battery_Temperature': battery_temperature * (0.98 + 0.02 * blend + noise * 0.5),
                'SoC': soc * (0.98 + 0.02 * blend + noise * 0.5),
                'SoH': soh * (0.98 + 0.02 * blend + noise * 0.5),
                'Charge_Cycles': int(charge_cycles * (0.95 + 0.05 * blend)),
                'Distance_Traveled': distance_traveled * (0.95 + 0.05 * blend),
                'Power_Consumption': power_consumption * (0.98 + 0.02 * blend + noise * 0.5),
                'Component_Health_Score': component_health_score * (0.98 + 0.02 * blend + noise * 0.5),
                'Failure_Probability': failure_probability * (0.95 + 0.05 * blend + noise * 0.3),
                'RUL': rul * (0.98 + 0.02 * blend + noise * 0.5),
            }
        else:
            # Earlier timesteps show progression toward current state
            # Create realistic trends based on input values
            health_factor = (component_health_score + soh) / 2.0  # Average health
            stress_factor = (failure_probability + (1 - health_factor)) / 2.0  # How stressed the system is

            # Better health = less variation in earlier timesteps
            variation = 1.0 + (stress_factor * 0.3 * (1 - progress)) + noise

            row = {
                # Battery metrics trend toward input values
                'Battery_Voltage': battery_voltage * (1.0 + (1 - progress) * 0.15 * (1 - health_factor)) * variation,
                'Battery_Current': battery_current * (0.85 + 0.15 * progress) * (1 + noise * stress_factor),
                'Battery_Temperature': battery_temperature * (0.90 + 0.10 * progress) * (1 + noise * stress_factor * 0.5),
                'SoC': min(100, soc * (1.0 + (1 - progress) * 0.25 * health_factor)),
                'SoH': min(100, soh * (1.0 + (1 - progress) * 0.15 * health_factor)),

                # Cumulative values scale with progress
                'Charge_Cycles': int(charge_cycles * (0.7 + 0.3 * progress)),
                'Distance_Traveled': distance_traveled * (0.7 + 0.3 * progress),
                'Power_Consumption': power_consumption * (0.85 + 0.15 * progress) * (1 + noise * 0.3),

                # Health scores degrade over time
                'Component_Health_Score': component_health_score * (1.0 + (1 - progress) * 0.2 * health_factor),
                'Failure_Probability': max(0, failure_probability * (0.3 + 0.7 * progress)),
                'RUL': rul * (1.0 + (1 - progress) * 0.3 * health_factor),
            }

        # Add other required features
        row.update({
            'Motor_Temperature': 20.0 + (battery_temperature - 20.0) * progress + noise * 3,
            'Motor_Vibration': 0.3 + stress_factor * 0.4 * progress,
            'Motor_Torque': 50.0 * (1 + noise * 0.1),
            'Motor_RPM': 1000.0 * (0.9 + 0.2 * progress),
            'Brake_Pad_Wear': min(1.0, stress_factor * 0.5 * progress),
            'Brake_Pressure': 50.0,
            'Reg_Brake_Efficiency': 0.90 - stress_factor * 0.1 * progress,
            'Tire_Pressure': 35.0 * (1.0 - stress_factor * 0.05 * progress),
            'Tire_Temperature': 20.0 + battery_temperature * 0.3 * progress,
            'Suspension_Load': 400.0 + power_consumption * 2,
            'Ambient_Temperature': 20.0,
            'Ambient_Humidity': 50.0,
            'Load_Weight': 400.0 + power_consumption * 2,
            'Driving_Speed': 40.0 + power_consumption * 0.2 * progress,
            'Idle_Time': max(0, 5.0 * (1 - progress) * (1 - stress_factor)),
            'Route_Roughness': 0.3 + stress_factor * 0.4,
            'TTF': rul * (1.0 + (1 - progress) * 0.3)
        })

        sequence_rows.append(row)

    # Create DataFrame with the sequence
    manual_sequence_df = pd.DataFrame(sequence_rows)

    # Engineer features on the entire sequence
    manual_sequence_df = simulator.engineer_features(manual_sequence_df)

    # Get features in exact order from training
    if len(simulator.feature_cols) > 0:
        # Use the exact feature list from training, in order
        available_cols = [col for col in simulator.feature_cols if col in manual_sequence_df.columns]

        # Log any missing features
        missing = [col for col in simulator.feature_cols if col not in manual_sequence_df.columns]
        if missing:
            print(f"Warning: {len(missing)} features missing from manual input: {missing[:5]}...")
    else:
        # Fallback to base features
        available_cols = ['Battery_Voltage', 'Battery_Current', 'Battery_Temperature',
                         'SoC', 'SoH', 'Charge_Cycles', 'Distance_Traveled',
                         'Power_Consumption', 'Component_Health_Score',
                         'Failure_Probability', 'RUL']
        available_cols = [col for col in available_cols if col in manual_sequence_df.columns]

    # Extract sequence data in exact order
    sequence_data = manual_sequence_df[available_cols].values

    # Scale
    if simulator.scaler is not None:
        try:
            sequence_data = simulator.scaler.transform(sequence_data)
        except Exception as e:
            print(f"Scaling warning: {e}")

    # Reshape for LSTM: (1, sequence_length, features)
    sequence_data = sequence_data.reshape(1, simulator.sequence_length, -1)

    # Predict
    if simulator.model is not None:
        try:
            prediction = simulator.model.predict(sequence_data, verbose=0)
            probability = float(prediction[0][0])
        except Exception as e:
            print(f"Manual prediction error: {e}")
            probability = 0.5
    else:
        probability = 0.5

    result = simulator._format_prediction(probability, simulated=(simulator.model is None))

    return {
        "timestamp": str(datetime.now()),
        "input_data": {
            "battery_voltage": battery_voltage,
            "battery_current": battery_current,
            "battery_temperature": battery_temperature,
            "soc": soc,
            "soh": soh,
            "charge_cycles": charge_cycles,
            "distance_traveled": distance_traveled,
            "power_consumption": power_consumption,
            "component_health_score": component_health_score,
            "failure_probability": failure_probability,
            "rul": rul,
        },
        "prediction": result,
        "note": "Prediction based on simulated sequence showing gradual degradation to your input values"
    }


@app.post("/api/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Make prediction from uploaded CSV file with 24 rows of historical data.

    Expected CSV format:
    - Must have exactly 24 rows (6 hours of data at 15-min intervals)
    - Required columns: Battery_Voltage, Battery_Current, Battery_Temperature,
      SoC, SoH, Charge_Cycles, Distance_Traveled, Power_Consumption,
      Component_Health_Score, Failure_Probability, RUL
    - Optional: all other sensor columns from the training dataset
    """
    try:
        # Read uploaded CSV
        contents = await file.read()
        csv_data = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validate row count
        if len(csv_data) != simulator.sequence_length:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Invalid CSV: Expected exactly {simulator.sequence_length} rows, got {len(csv_data)} rows",
                    "detail": f"The model requires {simulator.sequence_length} timesteps (6 hours of data at 15-min intervals)"
                }
            )

        # Required columns
        required_cols = [
            'Battery_Voltage', 'Battery_Current', 'Battery_Temperature',
            'SoC', 'SoH', 'Charge_Cycles', 'Distance_Traveled',
            'Power_Consumption', 'Component_Health_Score',
            'Failure_Probability', 'RUL'
        ]

        missing_cols = [col for col in required_cols if col not in csv_data.columns]
        if missing_cols:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Missing required columns: {', '.join(missing_cols)}",
                    "required_columns": required_cols,
                    "found_columns": list(csv_data.columns)
                }
            )

        # Add missing optional columns with default values
        optional_defaults = {
            'Motor_Temperature': 25.0,
            'Motor_Vibration': 0.5,
            'Motor_Torque': 50.0,
            'Motor_RPM': 1000.0,
            'Brake_Pad_Wear': 0.3,
            'Brake_Pressure': 50.0,
            'Reg_Brake_Efficiency': 0.85,
            'Tire_Pressure': 35.0,
            'Tire_Temperature': 25.0,
            'Suspension_Load': 500.0,
            'Ambient_Temperature': 20.0,
            'Ambient_Humidity': 50.0,
            'Load_Weight': 500.0,
            'Driving_Speed': 50.0,
            'Idle_Time': 0.0,
            'Route_Roughness': 0.5,
            'TTF': csv_data['RUL'].mean() if 'RUL' in csv_data.columns else 30.0
        }

        for col, default_val in optional_defaults.items():
            if col not in csv_data.columns:
                csv_data[col] = default_val

        # Engineer features
        csv_data = simulator.engineer_features(csv_data)

        # Get features in exact order from training
        if len(simulator.feature_cols) > 0:
            # Use the exact feature list from training
            available_cols = []
            for col in simulator.feature_cols:
                if col in csv_data.columns:
                    available_cols.append(col)
                else:
                    # Feature is missing - this is a problem
                    print(f"Warning: Feature '{col}' from training not found in CSV data")

            if len(available_cols) != len(simulator.feature_cols):
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Feature mismatch: Expected {len(simulator.feature_cols)} features, but only found {len(available_cols)}",
                        "missing_features": [col for col in simulator.feature_cols if col not in csv_data.columns],
                        "detail": "After feature engineering, some expected features are missing. This is likely a bug in the feature engineering code."
                    }
                )
        else:
            # Fallback to base features
            available_cols = required_cols
            available_cols = [col for col in available_cols if col in csv_data.columns]

        # Extract sequence data in the exact order
        sequence_data = csv_data[available_cols].values

        # Validate sequence length
        if sequence_data.shape[0] != simulator.sequence_length:
            return JSONResponse(
                status_code=400,
                content={"error": f"Sequence must have exactly {simulator.sequence_length} timesteps"}
            )

        # Scale
        if simulator.scaler is not None:
            try:
                sequence_data = simulator.scaler.transform(sequence_data)
            except Exception as e:
                print(f"Scaling warning: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to scale data: {str(e)}"}
                )

        # Reshape for LSTM: (1, sequence_length, features)
        sequence_data = sequence_data.reshape(1, simulator.sequence_length, -1)

        # Predict
        if simulator.model is not None:
            try:
                prediction = simulator.model.predict(sequence_data, verbose=0)
                probability = float(prediction[0][0])
            except Exception as e:
                print(f"CSV prediction error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Prediction failed: {str(e)}"}
                )
        else:
            return JSONResponse(
                status_code=503,
                content={"error": "Model not loaded"}
            )

        result = simulator._format_prediction(probability, simulated=False)

        # Get summary of uploaded data
        last_row = csv_data.iloc[-1]
        first_row = csv_data.iloc[0]

        return {
            "timestamp": str(datetime.now()),
            "data_summary": {
                "rows_processed": len(csv_data),
                "time_span": "6 hours (24 timesteps at 15-min intervals)",
                "first_timestep": {
                    "battery_voltage": float(first_row['Battery_Voltage']),
                    "soc": float(first_row['SoC']),
                    "soh": float(first_row['SoH']),
                    "component_health": float(first_row['Component_Health_Score']),
                },
                "last_timestep": {
                    "battery_voltage": float(last_row['Battery_Voltage']),
                    "soc": float(last_row['SoC']),
                    "soh": float(last_row['SoH']),
                    "component_health": float(last_row['Component_Health_Score']),
                },
                "trends": {
                    "voltage_change": float(last_row['Battery_Voltage'] - first_row['Battery_Voltage']),
                    "soc_change": float(last_row['SoC'] - first_row['SoC']),
                    "health_change": float(last_row['Component_Health_Score'] - first_row['Component_Health_Score']),
                }
            },
            "prediction": result,
            "note": "Prediction based on actual 6-hour historical data from uploaded CSV"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process CSV: {str(e)}"}
        )


@app.get("/api/download/csv-template")
async def download_csv_template():
    """Download sample CSV template for manual prediction."""
    return FileResponse(
        path="sample_24_rows_template.csv",
        filename="ev_prediction_template_24rows.csv",
        media_type="text/csv"
    )


@app.get("/api/analytics")
async def get_analytics():
    """Get comprehensive analytics."""
    analytics = simulator.get_analytics()

    return {
        "timestamp": str(datetime.now()),
        "analytics": analytics,
        "model_info": simulator.model_metrics
    }


@app.get("/api/model/info")
async def get_model_info():
    """Get model information and performance metrics."""
    return {
        "model_loaded": simulator.model is not None,
        "model_type": simulator.model_metrics.get('model_type', 'unknown'),
        "metrics": simulator.model_metrics,
        "feature_count": len(simulator.feature_cols) if simulator.feature_cols else 0,
        "sequence_length": simulator.sequence_length,
        "improvements": simulator.model_metrics.get('improvements', [])
    }


@app.post("/api/simulation/advance")
async def advance_simulation():
    """Manually advance simulation to next timestep."""
    simulator.advance_time()
    return {"current_index": simulator.current_index, "timestamp": str(datetime.now())}


# Continue in next message due to length...
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve enhanced dashboard."""
    with open('dashboard.html', 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    print("="*70)
    print("Enhanced EV Fleet Predictive Maintenance Dashboard")
    print("="*70)
    print("\nFeatures:")
    print("  + Auto-refresh every 15 seconds")
    print("  + Real-time LSTM predictions")
    print("  + Manual prediction testing")
    print("  + Comprehensive analytics")
    print("  + Model performance metrics")
    print("\nEndpoints:")
    print("  Dashboard:  http://localhost:8000/dashboard")
    print("  API Docs:   http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")
    print("="*70)

    uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed to 8001
