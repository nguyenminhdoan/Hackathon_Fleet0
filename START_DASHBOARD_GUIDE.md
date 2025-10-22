# 🚀 Complete Dashboard Startup Guide

## ✅ What I Fixed

1. **Added API endpoint** `/api/model/comparison` to `enhanced_dashboard.py`
2. **Updated dashboard.html** to fetch comparison data from API (not direct JSON files)
3. **Integrated with your existing backend** - consistent with other tabs

---

## 📋 **Steps to Start the Dashboard**

### **Step 1: Start the Backend API Server** ⚙️

```bash
python3 enhanced_dashboard.py
```

**You should see:**
```
======================================================================
Enhanced EV Fleet Predictive Maintenance Dashboard
======================================================================

Features:
  + Auto-refresh every 15 seconds
  + Real-time LSTM predictions
  + Manual prediction testing
  + Comprehensive analytics
  + Model performance metrics

Endpoints:
  Dashboard:  http://localhost:8000/dashboard
  API Docs:   http://localhost:8000/docs

Press Ctrl+C to stop
======================================================================
```

**Important:** The server runs on **port 8001** (not 8000 as shown in message)

---

### **Step 2: Open the Dashboard** 🌐

**Option A: Via API Server (Recommended)**
```bash
open http://localhost:8001/dashboard
```

**Option B: Direct HTML File**
```bash
open dashboard.html
```

> **Note:** Both work! The API serves the dashboard, but you can also open the HTML directly.

---

### **Step 3: Navigate to Model Comparison Tab** 📊

Click on **"📊 Model Comparison"** in the tab navigation.

---

## 🎯 **What Works Now**

### ✅ **ALL Tabs Work** (with backend running):

1. **🔴 LIVE: Real-Time Simulation**
   - Auto-refreshes every 15 seconds
   - Shows current vehicle data
   - AI predictions with cost analysis

2. **✍️ MANUAL: Test Prediction**
   - Enter custom vehicle parameters
   - Get instant AI prediction
   - Cost-benefit analysis

3. **📈 Analytics**
   - Dataset progress
   - Battery statistics
   - Health statistics
   - Maintenance events
   - Prediction statistics

4. **📊 Model Comparison** ⭐ **(NEW!)**
   - All 5 visualization charts
   - Live metrics table from API
   - Smart summary
   - Model selection guide

5. **🤖 Model Info**
   - Current model details
   - Performance metrics
   - Model improvements

---

## 📊 **Model Comparison Features**

### **What It Shows:**

1. **Visual Charts** (5 charts):
   - Metrics Comparison (bar chart)
   - Radar Chart (multi-dimensional)
   - Performance Heatmap
   - Confusion Matrices
   - Complexity vs Performance

2. **Live Metrics Table**:
   - Fetched from API in real-time
   - Shows all trained models
   - Auto-updates when you retrain

3. **Smart Summary**:
   - Automatically identifies best model for each metric
   - Example: "GRU has the best AUC-ROC (0.5649)"

4. **Selection Guide**:
   - For Maximum Safety (Best Recall)
   - For Cost Efficiency (Best Precision)
   - For Balanced Performance (Best F1)
   - For Best Discrimination (Best AUC-ROC)

---

## 🔧 **How It Works Now**

### **Backend API Flow:**

```
Dashboard (frontend)
    ↓
Fetch from: http://localhost:8001/api/model/comparison
    ↓
enhanced_dashboard.py (backend)
    ↓
Reads JSON files:
  - improved_model_results.json
  - gru_model_results.json
  - final_model_results.json
    ↓
Returns formatted data to dashboard
```

### **Key Changes:**

**Before:** Dashboard loaded JSON files directly
```javascript
fetch('improved_model_results.json')  // ❌ Direct file access
```

**Now:** Dashboard fetches from API
```javascript
fetch('http://localhost:8001/api/model/comparison')  // ✅ API endpoint
```

---

## 📁 **Required Files**

### **For Backend Server:**
- ✅ `enhanced_dashboard.py` (API server)
- ✅ `dashboard.html` (frontend)
- ✅ `EV_Predictive_Maintenance_Dataset_15min.csv` (data)
- ✅ `best_improved_lstm.keras` (model - optional)
- ✅ `improved_preprocessor.pkl` (preprocessor - optional)

### **For Model Comparison Tab:**
- ✅ `improved_model_results.json` (LSTM results)
- ✅ `gru_model_results.json` (GRU results - optional)
- ✅ `final_model_results.json` (LSTM+Attention - optional)
- ✅ `metrics_comparison.png` (visualization)
- ✅ `radar_comparison.png` (visualization)
- ✅ `performance_heatmap.png` (visualization)
- ✅ `confusion_matrices_comparison.png` (visualization)
- ✅ `complexity_vs_performance.png` (visualization)

---

## 🚀 **Complete Workflow**

### **1. First Time Setup:**

```bash
# Train models (if you haven't)
python3 train_improved_model.py    # LSTM
python3 train_gru_model.py         # GRU (optional)
python3 train_final_model.py       # LSTM+Attention (optional)

# Generate comparison visualizations
python3 generate_model_comparison_report.py
```

### **2. Start Dashboard:**

```bash
# Start the backend server
python3 enhanced_dashboard.py

# In a browser, go to:
# http://localhost:8001/dashboard
```

### **3. Use Model Comparison:**

1. Click **"📊 Model Comparison"** tab
2. See all your model comparisons
3. View charts, table, and recommendations

---

## 🔍 **Troubleshooting**

### **Problem: "Connection refused" or tabs show errors**

**Solution:** Make sure backend is running
```bash
python3 enhanced_dashboard.py
```

### **Problem: "No model results found"**

**Solution:** Train at least one model first
```bash
python3 train_improved_model.py
```

### **Problem: Charts show "Chart not available"**

**Solution:** Generate visualizations
```bash
python3 generate_model_comparison_report.py
```

### **Problem: Port 8001 already in use**

**Solution:** Kill existing process or change port
```bash
# Find and kill process
lsof -ti:8001 | xargs kill -9

# Or edit enhanced_dashboard.py line 597 to use different port
uvicorn.run(app, host="0.0.0.0", port=8002)  # Change to 8002
```

---

## 💡 **Pro Tips**

1. **Keep Backend Running**
   - Leave the terminal window open with the server running
   - Dashboard updates automatically every 15 seconds

2. **Retrain and Refresh**
   - After retraining models, the comparison data updates automatically
   - Just refresh the browser page

3. **View API Docs**
   - Go to: `http://localhost:8001/docs`
   - See all available endpoints
   - Test API directly

4. **Check Server Logs**
   - Watch the terminal where `enhanced_dashboard.py` is running
   - See real-time requests and any errors

---

## 🌐 **Available Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/dashboard` | GET | Serve dashboard HTML |
| `/api/vehicle/status` | GET | Current vehicle data |
| `/api/predict` | GET | AI prediction |
| `/api/predict/manual` | POST | Manual prediction |
| `/api/analytics` | GET | Analytics data |
| `/api/model/info` | GET | Current model info |
| `/api/model/comparison` | GET | **Model comparison (NEW!)** |
| `/api/simulation/advance` | POST | Advance simulation |
| `/docs` | GET | API documentation |

---

## 📊 **API Response Example**

### `GET /api/model/comparison`

```json
{
  "status": "success",
  "timestamp": "2025-10-22T19:30:00.123456",
  "models": [
    {
      "name": "LSTM (Improved)",
      "accuracy": 0.2944,
      "precision": 0.2944,
      "recall": 1.0000,
      "f1_score": 0.4549,
      "auc_roc": 0.5007,
      "parameters": 587649
    },
    {
      "name": "GRU",
      "accuracy": 0.3424,
      "precision": 0.3005,
      "recall": 0.9293,
      "f1_score": 0.4542,
      "auc_roc": 0.5649,
      "parameters": 444929
    }
  ],
  "best_models": {
    "recall": "LSTM (Improved)",
    "precision": "GRU",
    "f1_score": "LSTM (Improved)",
    "auc_roc": "GRU"
  },
  "summary": {
    "recall": "LSTM (Improved) has the best Recall (100.00%) - catches the most failures.",
    "precision": "GRU has the best Precision (30.05%) - fewest false alarms.",
    "f1_score": "LSTM (Improved) has the best F1-Score (45.49%) - most balanced.",
    "auc_roc": "GRU has the best AUC-ROC (0.5649) - best discrimination ability."
  }
}
```

---

## ✅ **Quick Start Checklist**

- [ ] Train at least one model
- [ ] Generate comparison visualizations
- [ ] Start backend server (`python3 enhanced_dashboard.py`)
- [ ] Open dashboard in browser (`http://localhost:8001/dashboard`)
- [ ] Click "📊 Model Comparison" tab
- [ ] Enjoy! 🎉

---

## 🎯 **Summary**

**To use JUST the Model Comparison tab:**
1. ✅ Start backend: `python3 enhanced_dashboard.py`
2. ✅ Open: `http://localhost:8001/dashboard`
3. ✅ Click: "📊 Model Comparison"

**All features work!** The Model Comparison tab is now fully integrated with your backend API and consistent with your other dashboard tabs!

---

**Created:** 2025-10-22
**Status:** ✅ Fully Integrated with Backend
**Server Port:** 8001
**API Endpoint:** `/api/model/comparison`
