"""
Main script for demand forecasting system.
Orchestrates data processing, model training, and prediction generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse
import json

from data_processor import DataProcessor
from forecasting_models import DemandForecaster, SKUForecaster
from visualizations import ForecastVisualizer


def main(config_path=None):
    """Main execution function for demand forecasting."""
    
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'data_dir': '../data',
            'output_dir': '../output',
            'models_dir': '../models',
            'forecast_days': 30,
            'test_size': 0.2,
            'top_skus': 20
        }
    
    print("=== Demand Forecasting System ===")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Data Processing
    print("Step 1: Loading and processing data...")
    processor = DataProcessor(config['data_dir'])
    
    # Load data
    combined_data = processor.load_and_combine_data()
    
    # Create aggregations
    daily_shipments = processor.create_daily_aggregation()
    sku_daily = processor.create_sku_daily_aggregation()
    
    # Get summary
    summary = processor.get_data_summary()
    print(f"\nData Summary:")
    print(f"- Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"- Total shipments: {summary['total_shipments']:,}")
    print(f"- Unique SKUs: {summary['unique_skus']:,}")
    print(f"- Daily average: {summary['daily_stats']['mean']:.1f} ± {summary['daily_stats']['std']:.1f}")
    
    # Save processed data
    processor.save_processed_data(config['data_dir'])
    
    # Step 2: Model Training
    print("\n\nStep 2: Training forecasting models...")
    forecaster = DemandForecaster()
    
    # Prepare train/test split
    train_data, test_data = forecaster.prepare_data(daily_shipments, config['test_size'])
    
    # Train models
    print("\nTraining Moving Average models...")
    ma_results = forecaster.train_moving_average()
    
    print("Training SARIMA model...")
    sarima_results = forecaster.train_sarima()
    
    print("Training Prophet model...")
    prophet_results = forecaster.train_prophet()
    
    print("Creating Ensemble model...")
    ensemble_results = forecaster.create_ensemble()
    
    # Get performance summary
    performance_df = forecaster.get_performance_summary()
    print("\n=== Model Performance Summary ===")
    print(performance_df)
    
    # Select and save best model
    best_model = forecaster.select_best_model()
    forecaster.save_best_model(f"{config['models_dir']}/best_model.pkl")
    
    # Step 3: Generate Future Predictions
    print(f"\n\nStep 3: Generating {config['forecast_days']}-day forecast...")
    
    # Use Prophet model for future predictions
    if 'prophet' in forecaster.models:
        prophet_model = forecaster.models['prophet']['model']
        
        # Create future dates
        last_date = daily_shipments['date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=config['forecast_days'],
            freq='D'
        )
        
        # Generate predictions
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = prophet_model.predict(future_df)
        
        # Prepare results
        predictions = pd.DataFrame({
            'date': forecast['ds'],
            'predicted_shipments': forecast['yhat'].round().astype(int),
            'lower_bound': forecast['yhat_lower'].round().astype(int),
            'upper_bound': forecast['yhat_upper'].round().astype(int)
        })
        
        # Ensure non-negative
        predictions['predicted_shipments'] = predictions['predicted_shipments'].clip(lower=0)
        predictions['lower_bound'] = predictions['lower_bound'].clip(lower=0)
        
        # Save predictions
        predictions.to_csv(f"{config['output_dir']}/future_predictions.csv", index=False)
        print(f"Saved predictions to future_predictions.csv")
        
        # Print summary
        print(f"\nForecast Summary:")
        print(f"- Average daily shipments: {predictions['predicted_shipments'].mean():.0f}")
        print(f"- Total shipments: {predictions['predicted_shipments'].sum():,}")
        print(f"- Peak day: {predictions.loc[predictions['predicted_shipments'].idxmax(), 'date'].strftime('%Y-%m-%d')}")
    
    # Step 4: SKU-level Predictions
    if sku_daily is not None:
        print(f"\n\nStep 4: Generating SKU-level predictions...")
        
        # Calculate SKU summary
        sku_summary = sku_daily.groupby('sku_id').agg({
            'quantity': ['sum', 'mean', 'std', 'count']
        }).round(2)
        sku_summary.columns = ['total_quantity', 'avg_daily', 'std_daily', 'days_sold']
        sku_summary = sku_summary.sort_values('total_quantity', ascending=False)
        
        # Create SKU forecaster
        sku_forecaster = SKUForecaster()
        sku_predictions = sku_forecaster.predict_top_skus(
            sku_daily, sku_summary, 
            top_n=config['top_skus'], 
            days_ahead=config['forecast_days']
        )
        
        # Save SKU predictions
        sku_predictions.to_csv(f"{config['output_dir']}/sku_predictions.csv", index=False)
        print(f"Saved SKU predictions to sku_predictions.csv")
        
        # Print top 10 SKUs
        print("\n=== Top 10 Predicted SKUs ===")
        for i, row in sku_predictions.head(10).iterrows():
            print(f"{i+1}. SKU {row['sku_id']}: {row['predicted_total']:,} units "
                  f"(Growth: {row['growth_rate']:.1f}%)")
    
    # Step 5: Create Visualizations
    print("\n\nStep 5: Creating visualizations...")
    visualizer = ForecastVisualizer()
    
    # Prepare data for visualization
    viz_data = {
        'test_data': test_data,
        'model_results': {
            'Prophet': forecaster.models.get('prophet', {}),
            '14-day MA': forecaster.models.get('moving_average', {}).get(14, {}),
            'Ensemble': forecaster.models.get('ensemble', {})
        },
        'historical_data': daily_shipments,
        'forecast_data': predictions if 'predictions' in locals() else None,
        'confidence_intervals': {
            'lower': predictions['lower_bound'],
            'upper': predictions['upper_bound']
        } if 'predictions' in locals() else None,
        'sku_predictions': sku_predictions if 'sku_predictions' in locals() else None
    }
    
    # Create all plots
    plots = visualizer.create_forecast_report_plots(viz_data, config['output_dir'])
    
    print(f"\nCreated {len(plots)} visualizations")
    
    print("\n=== Forecasting Complete ===")
    print(f"Results saved to {config['output_dir']}")
    
    return {
        'summary': summary,
        'performance': performance_df,
        'best_model': best_model,
        'predictions': predictions if 'predictions' in locals() else None,
        'sku_predictions': sku_predictions if 'sku_predictions' in locals() else None
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demand Forecasting System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='../output', help='Output directory')
    parser.add_argument('--forecast-days', type=int, default=30, help='Days to forecast')
    
    args = parser.parse_args()
    
    # Create custom config if args provided
    if not args.config:
        config = {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'models_dir': '../models',
            'forecast_days': args.forecast_days,
            'test_size': 0.2,
            'top_skus': 20
        }
        
        # Save config
        config_path = '../config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        config_path = args.config
    
    # Run main
    results = main(config_path)