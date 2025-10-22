"""
Enhanced Real-time Predictive Maintenance Dashboard
Features:
- Auto-refresh every 15 seconds
- Comprehensive metrics & analytics
- Manual prediction input form
- Model performance visualization
- Live simulation with real LSTM model
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
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
import os

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
                self.feature_cols = preprocessor_data.get('feature_cols', [])
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
                    self.feature_cols = preprocessor_data.get('feature_cols', [])
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
            # If we do preventive now, cost is $500
            # If we don't and it fails, cost is $5000 + 8 hours downtime = $7400
            preventive_cost = COST_PREVENTIVE
            if_no_action_cost = probability * (COST_CRITICAL + 8 * COST_DOWNTIME_HOUR)
            potential_savings = if_no_action_cost - preventive_cost
        elif probability > 0.4:
            severity = 'WARNING'
            recommendation = 'WARNING: Schedule maintenance within 24 hours'
            confidence = 'medium'
            preventive_cost = COST_PREVENTIVE
            if_no_action_cost = probability * (COST_CORRECTIVE + 4 * COST_DOWNTIME_HOUR)
            potential_savings = if_no_action_cost - preventive_cost
        elif probability > 0.2:
            severity = 'MONITOR'
            recommendation = 'MONITOR: Continue normal operations, schedule routine check'
            confidence = 'medium'
            preventive_cost = COST_PREVENTIVE
            if_no_action_cost = probability * (COST_CORRECTIVE + 4 * COST_DOWNTIME_HOUR)
            potential_savings = if_no_action_cost - preventive_cost
        else:
            severity = 'NORMAL'
            recommendation = 'NORMAL: No action required'
            confidence = 'high'
            preventive_cost = COST_PREVENTIVE
            if_no_action_cost = probability * COST_CORRECTIVE
            potential_savings = if_no_action_cost - preventive_cost

        # Decide recommendation
        action_recommendation = 'Perform preventive maintenance' if potential_savings > 0 else 'Continue monitoring'

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
                'expected_cost_if_no_action': float(if_no_action_cost),
                'potential_savings': float(potential_savings),
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
            "manual_predict": "/api/predict/manual"
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
    """Make manual prediction with user input."""

    # Create manual data point with ALL required base features
    manual_data = pd.DataFrame([{
        'Battery_Voltage': battery_voltage,
        'Battery_Current': battery_current,
        'Battery_Temperature': battery_temperature,
        'SoC': soc,
        'SoH': soh,
        'Charge_Cycles': charge_cycles,
        'Distance_Traveled': distance_traveled,
        'Power_Consumption': power_consumption,
        'Component_Health_Score': component_health_score,
        'Failure_Probability': failure_probability,
        'RUL': rul,
        # Add missing base features with defaults
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
        'TTF': 30.0
    }])

    # Engineer features
    manual_data = simulator.engineer_features(manual_data)

    # Get available features
    available_cols = [col for col in simulator.feature_cols if col in manual_data.columns]

    if len(available_cols) == 0:
        # Fallback to base features
        available_cols = ['Battery_Voltage', 'Battery_Current', 'Battery_Temperature',
                         'SoC', 'SoH', 'Charge_Cycles', 'Distance_Traveled',
                         'Power_Consumption', 'Component_Health_Score',
                         'Failure_Probability', 'RUL']
        available_cols = [col for col in available_cols if col in manual_data.columns]

    # Create sequence by repeating the data point
    sequence_data = manual_data[available_cols].values
    sequence_data = np.tile(sequence_data, (simulator.sequence_length, 1))

    # Scale
    if simulator.scaler is not None:
        try:
            sequence_data = simulator.scaler.transform(sequence_data)
        except:
            pass

    # Reshape
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
        },
        "prediction": result
    }


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


@app.get("/api/model/comparison")
async def get_model_comparison():
    """Get model comparison data for all trained models."""
    models = []

    # Try to load LSTM (Improved) results
    try:
        with open('improved_model_results.json', 'r') as f:
            lstm_data = json.load(f)
            models.append({
                'name': 'LSTM (Improved)',
                'accuracy': lstm_data.get('accuracy', 0),
                'precision': lstm_data.get('precision', 0),
                'recall': lstm_data.get('recall', 0),
                'f1_score': lstm_data.get('f1_score', 0),
                'auc_roc': lstm_data.get('auc', 0),
                'parameters': lstm_data.get('total_parameters', 0),
                'threshold': lstm_data.get('best_threshold', 0.3)
            })
    except FileNotFoundError:
        pass

    # Try to load GRU results
    try:
        with open('gru_model_results.json', 'r') as f:
            gru_data = json.load(f)
            metrics = gru_data.get('metrics', gru_data)
            params = gru_data.get('architecture', {}).get('total_parameters',
                                                          gru_data.get('total_parameters', 0))
            models.append({
                'name': 'GRU',
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'auc_roc': metrics.get('auc_roc', metrics.get('auc', 0)),
                'parameters': params,
                'threshold': gru_data.get('best_threshold', 0.35)
            })
    except FileNotFoundError:
        pass

    # Try to load Final LSTM+Attention results
    try:
        with open('final_model_results.json', 'r') as f:
            final_data = json.load(f)
            metrics = final_data.get('metrics', final_data)
            models.append({
                'name': 'LSTM+Attention',
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'auc_roc': metrics.get('auc_roc', metrics.get('auc', 0)),
                'parameters': final_data.get('total_parameters', 0),
                'threshold': final_data.get('best_threshold', 0.4)
            })
    except FileNotFoundError:
        pass

    if not models:
        return {
            "status": "no_models",
            "message": "No trained models found. Please train models first.",
            "models": []
        }

    # Find best models
    best = {
        'recall': max(models, key=lambda x: x['recall']),
        'precision': max(models, key=lambda x: x['precision']),
        'f1_score': max(models, key=lambda x: x['f1_score']),
        'auc_roc': max(models, key=lambda x: x['auc_roc'])
    }

    return {
        "status": "success",
        "timestamp": str(datetime.now()),
        "models": models,
        "best_models": {
            'recall': best['recall']['name'],
            'precision': best['precision']['name'],
            'f1_score': best['f1_score']['name'],
            'auc_roc': best['auc_roc']['name']
        },
        "summary": {
            'recall': f"{best['recall']['name']} has the best Recall ({best['recall']['recall']*100:.2f}%) - catches the most failures.",
            'precision': f"{best['precision']['name']} has the best Precision ({best['precision']['precision']*100:.2f}%) - fewest false alarms.",
            'f1_score': f"{best['f1_score']['name']} has the best F1-Score ({best['f1_score']['f1_score']*100:.2f}%) - most balanced.",
            'auc_roc': f"{best['auc_roc']['name']} has the best AUC-ROC ({best['auc_roc']['auc_roc']:.4f}) - best discrimination ability."
        }
    }


# Serve PNG images for model comparison
@app.get("/{filename}.png")
async def serve_image(filename: str):
    """Serve PNG image files."""
    file_path = f"{filename}.png"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"Image {filename}.png not found. Please run generate_model_comparison_report.py"}
        )


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
