"""
Production-ready API for Cloud Run deployment
Optimized for performance and reliability
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import pandas as pd
from pathlib import Path
from functools import lru_cache
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demand_forecast_model import DemandForecastModel

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/demand_forecast_model.pkl')
MAX_FORECAST_DAYS = 365
CACHE_SIZE = 128

# Global model instance
_model = None


@lru_cache(maxsize=1)
def get_model():
    """Load and cache the model (singleton pattern)"""
    global _model
    if _model is None:
        try:
            model_path = Path(MODEL_PATH)
            if model_path.exists():
                _model = DemandForecastModel.load(model_path)
                logger.info(f"Model loaded successfully from {MODEL_PATH}")
            else:
                logger.error(f"Model file not found at {MODEL_PATH}")
                # Initialize with default data if model not found
                _model = DemandForecastModel()
                _model.load_data('/app/data/daily_shipments.csv')
                _model.load_sku_data('/app/data/sku_daily_shipments.csv')
                _model.train(models=['ma', 'prophet'])
                logger.info("Model trained with default data")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    return _model


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Demand Forecasting API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Generate demand forecast',
            '/predict_sku': 'SKU-level predictions',
            '/model_info': 'Model information'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    try:
        model = get_model()
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'demand-forecast-api'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503


@app.route('/predict', methods=['POST'])
def predict():
    """Generate demand predictions"""
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        days_ahead = data.get('days_ahead', 30)
        include_bounds = data.get('include_bounds', True)
        
        # Validate input
        if not isinstance(days_ahead, int) or days_ahead < 1:
            return jsonify({'error': 'days_ahead must be a positive integer'}), 400
        
        if days_ahead > MAX_FORECAST_DAYS:
            return jsonify({'error': f'days_ahead cannot exceed {MAX_FORECAST_DAYS}'}), 400
        
        # Get model and generate predictions
        model = get_model()
        predictions = model.predict(days_ahead=days_ahead)
        
        # Format response
        forecast_data = []
        for _, row in predictions.iterrows():
            item = {
                'date': row['date'].strftime('%Y-%m-%d'),
                'predicted_shipments': int(row['predicted_shipments'])
            }
            if include_bounds:
                item['lower_bound'] = int(row['lower_bound'])
                item['upper_bound'] = int(row['upper_bound'])
            forecast_data.append(item)
        
        response = {
            'success': True,
            'forecast': forecast_data,
            'summary': {
                'total_shipments': int(predictions['predicted_shipments'].sum()),
                'daily_average': float(predictions['predicted_shipments'].mean()),
                'peak_day': predictions.loc[predictions['predicted_shipments'].idxmax(), 'date'].strftime('%Y-%m-%d'),
                'peak_value': int(predictions['predicted_shipments'].max()),
                'lowest_day': predictions.loc[predictions['predicted_shipments'].idxmin(), 'date'].strftime('%Y-%m-%d'),
                'lowest_value': int(predictions['predicted_shipments'].min())
            },
            'metadata': {
                'forecast_days': days_ahead,
                'generated_at': datetime.utcnow().isoformat(),
                'model_type': model.best_model if model else 'unknown',
                'include_bounds': include_bounds
            }
        }
        
        logger.info(f"Generated {days_ahead}-day forecast successfully")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@app.route('/predict_sku', methods=['POST'])
def predict_sku():
    """Generate SKU-level predictions"""
    try:
        # Check if SKU data is available
        model = get_model()
        if model.sku_data is None:
            return jsonify({
                'success': False,
                'error': 'SKU data not available in model'
            }), 400
        
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        top_n = data.get('top_n', 10)
        days_ahead = data.get('days_ahead', 30)
        
        # Validate input
        if not isinstance(top_n, int) or top_n < 1:
            return jsonify({'error': 'top_n must be a positive integer'}), 400
        
        if not isinstance(days_ahead, int) or days_ahead < 1:
            return jsonify({'error': 'days_ahead must be a positive integer'}), 400
        
        # Generate SKU predictions
        sku_predictions = model.predict_skus(top_n=top_n, days_ahead=days_ahead)
        
        # Format response
        sku_forecast = []
        for idx, (_, row) in enumerate(sku_predictions.iterrows()):
            sku_forecast.append({
                'rank': idx + 1,
                'sku_id': str(row['sku_id']),
                'predicted_total': int(row['predicted_total']),
                'historical_avg': int(row['historical_avg']),
                'growth_rate': round(float(row['growth_rate']), 2),
                'historical_rank': int(row['historical_rank'])
            })
        
        response = {
            'success': True,
            'sku_forecast': sku_forecast,
            'summary': {
                'skus_analyzed': len(sku_predictions),
                'high_growth_skus': len(sku_predictions[sku_predictions['growth_rate'] > 20]),
                'declining_skus': len(sku_predictions[sku_predictions['growth_rate'] < -10]),
                'avg_growth_rate': round(sku_predictions['growth_rate'].mean(), 2)
            },
            'metadata': {
                'top_n': top_n,
                'forecast_days': days_ahead,
                'generated_at': datetime.utcnow().isoformat()
            }
        }
        
        logger.info(f"Generated SKU predictions for top {top_n} SKUs")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"SKU prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        model = get_model()
        performance = model.get_performance_summary()
        
        response = {
            'success': True,
            'model_info': {
                'best_model': model.best_model,
                'is_trained': model.is_trained,
                'available_models': list(model.models.keys()),
                'performance_metrics': json.loads(performance.to_json())
            },
            'configuration': {
                'test_size': model.config.get('test_size'),
                'forecast_days': model.config.get('forecast_days'),
                'ma_windows': model.config.get('ma_windows'),
                'confidence_interval': model.config.get('confidence_interval')
            },
            'data_info': {
                'has_daily_data': model.daily_data is not None,
                'has_sku_data': model.sku_data is not None
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.utcnow().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.utcnow().isoformat()
    }), 500


@app.errorhandler(Exception)
def handle_exception(error):
    logger.error(f"Unhandled exception: {error}")
    return jsonify({
        'success': False,
        'error': 'An unexpected error occurred',
        'timestamp': datetime.utcnow().isoformat()
    }), 500


if __name__ == '__main__':
    # Get port from environment variable
    port = int(os.environ.get('PORT', 8080))
    
    # Log startup
    logger.info(f"Starting Demand Forecast API on port {port}")
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)