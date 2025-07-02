#!/usr/bin/env python3
"""
Test script for the unified DemandForecastModel
"""

import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from demand_forecast_model import DemandForecastModel
import pandas as pd

def test_model():
    print("=== Testing Unified Demand Forecast Model ===\n")
    
    try:
        # 1. Create model
        print("1. Creating model instance...")
        model = DemandForecastModel()
        print("✓ Model created successfully")
        
        # 2. Load data
        print("\n2. Loading data...")
        model.load_data('data/daily_shipments.csv')
        model.load_sku_data('data/sku_daily_shipments.csv')
        print("✓ Data loaded successfully")
        
        # 3. Train models (only fast ones for testing)
        print("\n3. Training models...")
        model.train(models=['ma', 'prophet'])
        print("✓ Models trained successfully")
        
        # 4. Check performance
        print("\n4. Model Performance:")
        performance = model.get_performance_summary()
        print(performance)
        print(f"\n✓ Best model: {model.best_model}")
        
        # 5. Generate predictions
        print("\n5. Generating 7-day forecast...")
        predictions = model.predict(days_ahead=7)
        print("\nForecast Preview:")
        print(predictions)
        print("✓ Predictions generated successfully")
        
        # 6. Save model
        print("\n6. Saving model...")
        model.save('models/test_unified_model.pkl')
        print("✓ Model saved successfully")
        
        # 7. Load model
        print("\n7. Loading saved model...")
        loaded_model = DemandForecastModel.load('models/test_unified_model.pkl')
        print("✓ Model loaded successfully")
        
        # 8. Test loaded model
        print("\n8. Testing loaded model...")
        new_predictions = loaded_model.predict(days_ahead=3)
        print("\nLoaded Model Predictions:")
        print(new_predictions)
        print("✓ Loaded model works correctly")
        
        print("\n=== All Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)