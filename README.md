# Demand Forecasting System

A comprehensive demand forecasting system for warehouse and fulfillment center operations. Currently configured for FC8 (Fulfillment Center 8) shipment data, this system uses advanced time series models to predict future shipment volumes and identify top-performing SKUs.

## 🎯 Features

- **Multi-Model Forecasting**: Implements Moving Average, SARIMA, Prophet, and Ensemble models
- **SKU-Level Predictions**: Forecasts demand for individual products
- **Automated Data Processing**: Handles multiple Excel files with different structures
- **Visualization Suite**: Generates comprehensive charts and reports
- **Performance Metrics**: Compares models using MAE and RMSE

## 📁 Project Structure

```
Demand-Forecasting/
├── data/                    # Raw data and processed files
│   ├── FC8出貨明細(*.xlsx)  # Original shipment data
│   ├── daily_shipments.csv  # Aggregated daily data
│   └── sku_daily_shipments.csv # SKU-level daily data
├── models/                  # Trained model files
│   └── prophet_model.pkl    # Best performing model
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_processor.py    # Data loading and preprocessing
│   ├── forecasting_models.py # Model implementations
│   ├── visualizations.py    # Chart generation
│   └── main.py             # Main execution script
├── output/                  # Generated predictions and visualizations
│   ├── future_predictions.csv
│   ├── sku_predictions.csv
│   └── *.png               # Visualization files
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### Installation

1. Clone or navigate to the Demand-Forecasting directory:
```bash
cd /Users/user/claude-code/Demand-Forecasting
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Forecasting System

1. **Basic run with default settings:**
```bash
cd src
python main.py
```

2. **Custom forecast period:**
```bash
python main.py --forecast-days 60
```

3. **Specify custom directories:**
```bash
python main.py --data-dir ../data --output-dir ../output
```

## 📊 Data Requirements

The system expects Excel files with the following structure:
- Date column (e.g., 'Shippdat', 'shipment_datetime')
- SKU identifier column (e.g., 'internal_sku_id')
- One row per shipment

## 🔮 Models Implemented

### 1. Moving Average (MA)
- Simple baseline models with 7, 14, and 30-day windows
- Good for capturing recent trends

### 2. SARIMA
- Seasonal Autoregressive Integrated Moving Average
- Captures weekly seasonality patterns

### 3. Prophet
- Facebook's time series forecasting model
- Handles holidays and changepoints automatically
- **Currently the best performing model**

### 4. Ensemble
- Combines Prophet and best MA model
- Often provides more stable predictions

## 📈 Output Files

### Predictions
- `future_predictions.csv`: Daily shipment forecasts
- `sku_predictions.csv`: SKU-level demand predictions

### Visualizations
- `model_comparison.png`: Performance comparison of all models
- `forecast_with_history.png`: Historical data with future predictions
- `sku_analysis.png`: Top SKU predictions and growth rates
- `weekly_pattern.png`: Day-of-week shipment patterns

## 🎯 Key Insights from Analysis

Based on the current data (Jan-May 2025):
- Average daily shipments: 7,100 units
- Predicted 30-day average: 9,851 units (21.9% increase)
- Top growth SKUs: 168773 (+120%), 178801 (+78%), 808545 (+34%)

## 🛠️ Advanced Usage

### Custom Configuration

Create a `config.json` file:
```json
{
  "data_dir": "../data",
  "output_dir": "../output",
  "models_dir": "../models",
  "forecast_days": 30,
  "test_size": 0.2,
  "top_skus": 20
}
```

Run with configuration:
```bash
python main.py --config ../config.json
```

### Programmatic Usage

```python
from src.data_processor import DataProcessor
from src.forecasting_models import DemandForecaster

# Load data
processor = DataProcessor('../data')
data = processor.load_and_combine_data()
daily_data = processor.create_daily_aggregation()

# Train models
forecaster = DemandForecaster()
train_data, test_data = forecaster.prepare_data(daily_data)
forecaster.train_prophet()

# Get predictions
best_model = forecaster.select_best_model()
```

## 📝 Notes

- The system automatically handles missing dates and fills gaps
- Predictions are clipped to ensure non-negative values
- Models are retrained on each run for latest patterns
- Weekly seasonality is strong (Sundays show higher volumes)

## 🤝 Contributing

To add new features or models:
1. Add model implementation to `forecasting_models.py`
2. Update visualization methods in `visualizations.py`
3. Integrate into main pipeline in `main.py`

## 📄 License

Internal use only - Property of Analytics Team