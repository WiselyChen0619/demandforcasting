"""
Example usage of the DemandForecastModel class
"""

import sys
sys.path.append('../src')

from demand_forecast_model import DemandForecastModel
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """
    Demonstrate how to use the DemandForecastModel
    """
    
    print("=== Demand Forecast Model Example ===\n")
    
    # 1. Create model instance with custom configuration
    config = {
        'test_size': 0.2,
        'ma_windows': [7, 14, 30],
        'forecast_days': 30,
        'prophet_params': {
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05
        }
    }
    
    model = DemandForecastModel(config=config)
    
    # 2. Load data
    print("Loading data...")
    model.load_data('../data/daily_shipments.csv')
    model.load_sku_data('../data/sku_daily_shipments.csv')
    
    # 3. Train models
    print("\nTraining models...")
    # Train specific models
    model.train(models=['ma', 'prophet', 'ensemble'])
    
    # Or train all available models
    # model.train()
    
    # 4. Get performance summary
    print("\nModel Performance Summary:")
    performance = model.get_performance_summary()
    print(performance)
    
    # 5. Generate future predictions
    print("\nGenerating 30-day forecast...")
    predictions = model.predict(days_ahead=30)
    
    print(f"\nForecast Summary:")
    print(f"- Average daily shipments: {predictions['predicted_shipments'].mean():.0f}")
    print(f"- Total shipments: {predictions['predicted_shipments'].sum():,}")
    print(f"- Peak day: {predictions.loc[predictions['predicted_shipments'].idxmax(), 'date'].strftime('%Y-%m-%d')}")
    
    # Save predictions
    predictions.to_csv('../output/model_predictions.csv', index=False)
    print("\nPredictions saved to ../output/model_predictions.csv")
    
    # 6. SKU-level predictions
    print("\nGenerating SKU predictions...")
    sku_predictions = model.predict_skus(top_n=10, days_ahead=30)
    
    print("\nTop 5 Growth SKUs:")
    for _, row in sku_predictions.head(5).iterrows():
        print(f"SKU {row['sku_id']}: {row['predicted_total']:,} units (Growth: {row['growth_rate']:.1f}%)")
    
    # 7. Save the trained model
    print("\nSaving model...")
    model.save('../models/trained_forecast_model.pkl')
    
    # 8. Load saved model (demonstration)
    print("\nLoading saved model...")
    loaded_model = DemandForecastModel.load('../models/trained_forecast_model.pkl')
    
    # Use loaded model for prediction
    new_predictions = loaded_model.predict(days_ahead=7)
    print(f"\n7-day forecast from loaded model:")
    print(new_predictions)
    
    # 9. Visualize predictions (optional)
    plt.figure(figsize=(12, 6))
    plt.plot(predictions['date'], predictions['predicted_shipments'], 'b-', label='Forecast')
    plt.fill_between(predictions['date'], 
                     predictions['lower_bound'], 
                     predictions['upper_bound'], 
                     alpha=0.3, label='95% CI')
    plt.title('30-Day Demand Forecast')
    plt.xlabel('Date')
    plt.ylabel('Shipments')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../output/forecast_example.png')
    print("\nForecast plot saved to ../output/forecast_example.png")
    
    print("\n=== Example Complete ===")


def batch_prediction_example():
    """
    Example of batch predictions for multiple scenarios
    """
    print("\n=== Batch Prediction Example ===\n")
    
    # Load trained model
    model = DemandForecastModel.load('../models/trained_forecast_model.pkl')
    
    # Generate predictions for different time horizons
    horizons = [7, 14, 30, 60, 90]
    results = {}
    
    for days in horizons:
        predictions = model.predict(days_ahead=days)
        results[f'{days}_days'] = {
            'total': predictions['predicted_shipments'].sum(),
            'daily_avg': predictions['predicted_shipments'].mean(),
            'peak': predictions['predicted_shipments'].max()
        }
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results).T
    print("Forecast Summary for Different Horizons:")
    print(summary_df.round(0))
    
    return summary_df


def custom_model_example():
    """
    Example of using the model with custom data
    """
    print("\n=== Custom Data Example ===\n")
    
    # Create synthetic daily data
    import numpy as np
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    # Generate synthetic shipment data with trend and seasonality
    trend = np.linspace(1000, 1500, len(dates))
    seasonality = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly pattern
    noise = np.random.normal(0, 50, len(dates))
    
    shipments = trend + seasonality + noise
    shipments = np.maximum(shipments, 0)  # Ensure non-negative
    
    # Create DataFrame
    custom_data = pd.DataFrame({
        'date': dates,
        'shipment_count': shipments.astype(int)
    })
    
    # Create and train model
    model = DemandForecastModel()
    model.load_data(custom_data)
    model.train(models=['prophet'])  # Train only Prophet for speed
    
    # Generate predictions
    predictions = model.predict(days_ahead=30)
    
    print(f"Custom Data Forecast Summary:")
    print(f"- Next 30 days average: {predictions['predicted_shipments'].mean():.0f}")
    print(f"- Growth trend detected: {'Yes' if predictions['predicted_shipments'].iloc[-1] > predictions['predicted_shipments'].iloc[0] else 'No'}")
    
    return model, predictions


if __name__ == "__main__":
    # Run main example
    main()
    
    # Run additional examples
    # batch_prediction_example()
    # custom_model_example()