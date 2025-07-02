"""
API interface for the Demand Forecast Model
Provides RESTful endpoints for predictions
"""

from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from demand_forecast_model import DemandForecastModel

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None
model_path = Path(__file__).parent.parent / 'models' / 'demand_forecast_model.pkl'


def load_model():
    """Load the trained model"""
    global model
    try:
        if model_path.exists():
            model = DemandForecastModel.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Generate demand predictions
    
    Request body:
    {
        "days_ahead": 30,
        "include_bounds": true
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        days_ahead = data.get('days_ahead', 30)
        include_bounds = data.get('include_bounds', True)
        
        # Generate predictions
        predictions = model.predict(days_ahead=days_ahead)
        
        # Format response
        response = {
            'forecast': [
                {
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'predicted_shipments': int(row['predicted_shipments']),
                    'lower_bound': int(row['lower_bound']) if include_bounds else None,
                    'upper_bound': int(row['upper_bound']) if include_bounds else None
                }
                for _, row in predictions.iterrows()
            ],
            'summary': {
                'total_shipments': int(predictions['predicted_shipments'].sum()),
                'daily_average': float(predictions['predicted_shipments'].mean()),
                'peak_day': predictions.loc[predictions['predicted_shipments'].idxmax(), 'date'].strftime('%Y-%m-%d'),
                'peak_value': int(predictions['predicted_shipments'].max())
            },
            'metadata': {
                'forecast_days': days_ahead,
                'generated_at': datetime.now().isoformat(),
                'model_type': model.best_model if model else 'unknown'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/predict_sku', methods=['POST'])
def predict_sku():
    """
    Generate SKU-level predictions
    
    Request body:
    {
        "top_n": 10,
        "days_ahead": 30
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if model.sku_data is None:
        return jsonify({'error': 'No SKU data available'}), 400
    
    try:
        data = request.get_json()
        top_n = data.get('top_n', 10)
        days_ahead = data.get('days_ahead', 30)
        
        # Generate SKU predictions
        sku_predictions = model.predict_skus(top_n=top_n, days_ahead=days_ahead)
        
        # Format response
        response = {
            'sku_forecast': [
                {
                    'sku_id': str(row['sku_id']),
                    'predicted_total': int(row['predicted_total']),
                    'historical_avg': int(row['historical_avg']),
                    'growth_rate': float(row['growth_rate']),
                    'historical_rank': int(row['historical_rank']),
                    'forecast_rank': idx + 1
                }
                for idx, (_, row) in enumerate(sku_predictions.iterrows())
            ],
            'summary': {
                'skus_analyzed': len(sku_predictions),
                'high_growth_skus': len(sku_predictions[sku_predictions['growth_rate'] > 20]),
                'declining_skus': len(sku_predictions[sku_predictions['growth_rate'] < -10])
            },
            'metadata': {
                'forecast_days': days_ahead,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"SKU prediction error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        performance = model.get_performance_summary()
        
        response = {
            'best_model': model.best_model,
            'is_trained': model.is_trained,
            'performance_metrics': performance.to_dict(),
            'available_models': list(model.models.keys()),
            'configuration': {
                'test_size': model.config.get('test_size'),
                'forecast_days': model.config.get('forecast_days'),
                'ma_windows': model.config.get('ma_windows')
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Retrain the model with new data
    
    Request body:
    {
        "data_path": "/path/to/daily_shipments.csv",
        "sku_data_path": "/path/to/sku_daily_shipments.csv",
        "models": ["ma", "prophet", "ensemble"]
    }
    """
    global model
    
    try:
        data = request.get_json()
        data_path = data.get('data_path')
        sku_data_path = data.get('sku_data_path')
        models_to_train = data.get('models', None)
        
        if not data_path:
            return jsonify({'error': 'data_path is required'}), 400
        
        # Create new model instance
        model = DemandForecastModel()
        
        # Load data
        model.load_data(data_path)
        if sku_data_path:
            model.load_sku_data(sku_data_path)
        
        # Train models
        model.train(models=models_to_train)
        
        # Save the new model
        model.save(model_path)
        
        response = {
            'status': 'success',
            'message': 'Model retrained successfully',
            'best_model': model.best_model,
            'performance': model.get_performance_summary().to_dict()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Retrain error: {e}")
        return jsonify({'error': str(e)}), 400


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# CLI for testing
def create_test_client():
    """Create a test client for the API"""
    app.config['TESTING'] = True
    return app.test_client()


# Load model on startup
load_model()

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)