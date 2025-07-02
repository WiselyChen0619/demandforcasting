# Demand Forecast Model Usage Guide

## Overview

The `DemandForecastModel` is a unified class that encapsulates all forecasting functionality, making it easy to train, save, load, and deploy demand forecasting models.

## Key Features

- **Multiple Models**: Supports Moving Average, SARIMA, Prophet, and Ensemble models
- **Easy Training**: Single method to train all or specific models
- **Model Persistence**: Save and load trained models
- **Flexible Predictions**: Generate forecasts for any time horizon
- **SKU-Level Forecasting**: Predict demand for individual products
- **API Ready**: Flask API for integration with other systems

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage

```python
from demand_forecast_model import DemandForecastModel

# Create model instance
model = DemandForecastModel()

# Load data
model.load_data('data/daily_shipments.csv')
model.load_sku_data('data/sku_daily_shipments.csv')

# Train all models
model.train()

# Generate 30-day forecast
predictions = model.predict(days_ahead=30)
print(predictions)

# Save model
model.save('models/my_forecast_model.pkl')
```

### 2. Custom Configuration

```python
# Custom configuration
config = {
    'test_size': 0.2,
    'ma_windows': [7, 14, 30],
    'forecast_days': 30,
    'prophet_params': {
        'weekly_seasonality': True,
        'changepoint_prior_scale': 0.05
    }
}

model = DemandForecastModel(config=config)
```

### 3. Train Specific Models

```python
# Train only specific models for faster execution
model.train(models=['ma', 'prophet'])  # Skip SARIMA and ensemble
```

### 4. Load and Use Saved Model

```python
# Load pre-trained model
model = DemandForecastModel.load('models/demand_forecast_model.pkl')

# Generate predictions without retraining
predictions = model.predict(days_ahead=60)
```

## API Usage

### Starting the API Server

```bash
cd src
python forecast_api.py
```

The API will run on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:5000/health
```

#### 2. Generate Forecast
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days_ahead": 30,
    "include_bounds": true
  }'
```

Response:
```json
{
  "forecast": [
    {
      "date": "2025-07-03",
      "predicted_shipments": 9500,
      "lower_bound": 8000,
      "upper_bound": 11000
    }
  ],
  "summary": {
    "total_shipments": 285000,
    "daily_average": 9500.0,
    "peak_day": "2025-07-15",
    "peak_value": 11500
  },
  "metadata": {
    "forecast_days": 30,
    "generated_at": "2025-07-02T15:30:00",
    "model_type": "Prophet"
  }
}
```

#### 3. SKU Predictions
```bash
curl -X POST http://localhost:5000/predict_sku \
  -H "Content-Type: application/json" \
  -d '{
    "top_n": 10,
    "days_ahead": 30
  }'
```

#### 4. Model Information
```bash
curl http://localhost:5000/model_info
```

#### 5. Retrain Model
```bash
curl -X POST http://localhost:5000/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "/path/to/new/daily_shipments.csv",
    "sku_data_path": "/path/to/new/sku_data.csv",
    "models": ["ma", "prophet", "ensemble"]
  }'
```

## Advanced Usage

### Batch Predictions

```python
# Generate predictions for multiple horizons
horizons = [7, 14, 30, 60, 90]
results = {}

for days in horizons:
    predictions = model.predict(days_ahead=days)
    results[f'{days}_days'] = {
        'total': predictions['predicted_shipments'].sum(),
        'avg': predictions['predicted_shipments'].mean()
    }
```

### Custom Data Integration

```python
# Use with custom DataFrame
import pandas as pd

custom_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', '2024-12-31'),
    'shipment_count': your_shipment_data
})

model = DemandForecastModel()
model.load_data(custom_data)  # Pass DataFrame directly
model.train()
```

### Model Performance Analysis

```python
# Get detailed performance metrics
performance = model.get_performance_summary()
print(performance)

# Output:
#              MAE     RMSE
# 14-day MA    1589.49  2152.56
# Prophet      1715.36  2149.62
# SARIMA       1648.17  2104.94
# Ensemble     1609.27  2112.71
```

## Model Architecture

### Class Structure

```
DemandForecastModel
├── __init__(config)
├── load_data(data_path)
├── load_sku_data(sku_path)
├── train(models)
│   ├── _train_moving_average()
│   ├── _train_sarima()
│   ├── _train_prophet()
│   └── _train_ensemble()
├── predict(days_ahead)
├── predict_skus(top_n, days_ahead)
├── get_performance_summary()
├── save(filepath)
└── load(filepath) [classmethod]
```

### Configuration Options

```python
{
    'test_size': 0.2,              # Train/test split ratio
    'ma_windows': [7, 14, 30],     # Moving average windows
    'sarima_order': (1, 1, 1),     # ARIMA parameters
    'sarima_seasonal_order': (1, 1, 1, 7),  # Seasonal parameters
    'prophet_params': {
        'yearly_seasonality': False,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'changepoint_prior_scale': 0.05
    },
    'forecast_days': 30,           # Default forecast horizon
    'confidence_interval': 0.95    # Prediction intervals
}
```

## Best Practices

1. **Data Quality**: Ensure your data has consistent date formatting and no missing dates
2. **Model Selection**: Start with Prophet for general use, add others for comparison
3. **Performance**: For faster training, use only necessary models
4. **Deployment**: Use the API for production integration
5. **Monitoring**: Regularly retrain with new data for best results

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce data size or train fewer models
2. **Prophet Installation**: Ensure cmdstanpy is properly installed
3. **API Timeout**: Increase timeout for large datasets

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

model = DemandForecastModel()
# Debug messages will now be shown
```

## Examples

See the `examples/` directory for complete working examples:
- `model_usage_example.py`: Basic usage examples
- `batch_predictions.py`: Bulk prediction scenarios
- `api_client_example.py`: API integration examples

## Support

For issues or questions, please refer to the main project documentation or create an issue in the repository.