"""
Visualization module for demand forecasting results.
Creates charts and plots for analysis and reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ForecastVisualizer:
    """Create visualizations for forecast results."""
    
    def __init__(self, figsize=(15, 10)):
        """Initialize visualizer with default figure size."""
        self.figsize = figsize
    
    def plot_model_comparison(self, test_data, model_results, save_path=None):
        """Plot actual vs predicted values for multiple models."""
        plt.figure(figsize=self.figsize)
        
        # Plot actual values
        plt.subplot(2, 2, 1)
        plt.plot(test_data['date'], test_data['shipment_count'], 
                'o-', label='Actual', color='black', alpha=0.7)
        
        # Plot model predictions
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (model_name, results) in enumerate(model_results.items()):
            if 'forecast' in results:
                plt.plot(test_data['date'], results['forecast'], 
                        's-', label=model_name, color=colors[i % len(colors)], alpha=0.7)
        
        plt.title('Model Predictions vs Actual', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Shipment Count', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot residuals for best model
        plt.subplot(2, 2, 2)
        if 'prophet' in model_results:
            residuals = test_data['shipment_count'] - model_results['prophet']['forecast']
            plt.scatter(test_data['date'], residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Model Residuals', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Residual', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # Plot performance metrics
        plt.subplot(2, 2, 3)
        metrics_data = []
        for model_name, results in model_results.items():
            if 'mae' in results:
                metrics_data.append({
                    'Model': model_name,
                    'MAE': results['mae'],
                    'RMSE': results['rmse']
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            x = range(len(metrics_df))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], metrics_df['MAE'], 
                   width, label='MAE', alpha=0.8)
            plt.bar([i + width/2 for i in x], metrics_df['RMSE'], 
                   width, label='RMSE', alpha=0.8)
            
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Error', fontsize=12)
            plt.title('Model Performance Comparison', fontsize=14)
            plt.xticks(x, metrics_df['Model'], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_forecast_with_history(self, historical_data, forecast_data, 
                                  confidence_intervals=None, save_path=None):
        """Plot historical data with future forecast."""
        plt.figure(figsize=(15, 8))
        
        # Plot historical data (last 60 days)
        recent_history = historical_data.tail(60)
        plt.plot(recent_history['date'], recent_history['shipment_count'], 
                'o-', label='Historical Data', color='blue', alpha=0.7)
        
        # Plot forecast
        plt.plot(forecast_data['date'], forecast_data['predicted_shipments'], 
                's-', label='Forecast', color='red', linewidth=2)
        
        # Add confidence interval if provided
        if confidence_intervals:
            plt.fill_between(forecast_data['date'], 
                           confidence_intervals['lower'], 
                           confidence_intervals['upper'], 
                           alpha=0.3, color='red', label='95% Confidence Interval')
        
        # Add vertical line at forecast start
        plt.axvline(x=historical_data['date'].max(), color='green', 
                   linestyle='--', alpha=0.7, label='Forecast Start')
        
        # Formatting
        plt.title('FC8 Shipment Forecast', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Shipments', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add annotations for max and min
        if len(forecast_data) > 0:
            max_idx = forecast_data['predicted_shipments'].idxmax()
            plt.annotate(f'Peak: {forecast_data.loc[max_idx, "predicted_shipments"]:,}',
                        xy=(forecast_data.loc[max_idx, 'date'], 
                            forecast_data.loc[max_idx, 'predicted_shipments']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_sku_analysis(self, sku_predictions, top_n=10, save_path=None):
        """Plot SKU-level analysis and predictions."""
        plt.figure(figsize=(14, 8))
        
        # Get top N SKUs
        top_skus = sku_predictions.head(top_n)
        
        # Plot 1: Historical vs Predicted
        plt.subplot(2, 1, 1)
        x = range(len(top_skus))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], top_skus['historical_avg'], 
               width, label='Historical 30-Day Avg', alpha=0.8, color='steelblue')
        plt.bar([i + width/2 for i in x], top_skus['predicted_total'], 
               width, label='Predicted 30-Day Total', alpha=0.8, color='darkorange')
        
        plt.xlabel('SKU', fontsize=12)
        plt.ylabel('Quantity', fontsize=12)
        plt.title(f'TOP {top_n} SKUs: Historical Average vs Future Prediction', fontsize=14)
        plt.xticks(x, [f"SKU\n{int(sku)}" for sku in top_skus['sku_id']], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Growth rates
        plt.subplot(2, 1, 2)
        growth_rates = top_skus['growth_rate'].values
        colors = ['green' if x > 0 else 'red' for x in growth_rates]
        bars = plt.bar(x, growth_rates, color=colors, alpha=0.7)
        
        plt.xlabel('SKU', fontsize=12)
        plt.ylabel('Growth Rate (%)', fontsize=12)
        plt.title(f'TOP {top_n} SKUs Expected Growth Rate', fontsize=14)
        plt.xticks(x, [f"SKU\n{int(sku)}" for sku in top_skus['sku_id']], rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, growth_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_weekly_pattern(self, data, save_path=None):
        """Plot weekly pattern analysis."""
        plt.figure(figsize=(10, 6))
        
        # Calculate weekday aggregation
        data['weekday'] = pd.to_datetime(data['date']).dt.dayofweek
        weekday_pattern = data.groupby('weekday')['shipment_count'].agg(['mean', 'std'])
        
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        x = range(7)
        
        # Plot with error bars
        plt.bar(x, weekday_pattern['mean'], 
               yerr=weekday_pattern['std'], 
               capsize=10, alpha=0.7, color='skyblue')
        
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('Average Shipments', fontsize=12)
        plt.title('Weekly Shipment Pattern', fontsize=14)
        plt.xticks(x, weekday_names)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(weekday_pattern['mean'], weekday_pattern['std'])):
            plt.text(i, mean + std + 50, f'{int(mean)}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def create_forecast_report_plots(self, data_dict, output_dir='../output'):
        """Create all standard plots for a forecast report."""
        plots_created = []
        
        # Model comparison
        if 'test_data' in data_dict and 'model_results' in data_dict:
            fig = self.plot_model_comparison(
                data_dict['test_data'], 
                data_dict['model_results'],
                save_path=f"{output_dir}/model_comparison.png"
            )
            plots_created.append('model_comparison.png')
            plt.close(fig)
        
        # Forecast with history
        if 'historical_data' in data_dict and 'forecast_data' in data_dict:
            fig = self.plot_forecast_with_history(
                data_dict['historical_data'],
                data_dict['forecast_data'],
                confidence_intervals=data_dict.get('confidence_intervals'),
                save_path=f"{output_dir}/forecast_with_history.png"
            )
            plots_created.append('forecast_with_history.png')
            plt.close(fig)
        
        # SKU analysis
        if 'sku_predictions' in data_dict:
            fig = self.plot_sku_analysis(
                data_dict['sku_predictions'],
                save_path=f"{output_dir}/sku_analysis.png"
            )
            plots_created.append('sku_analysis.png')
            plt.close(fig)
        
        # Weekly pattern
        if 'historical_data' in data_dict:
            fig = self.plot_weekly_pattern(
                data_dict['historical_data'],
                save_path=f"{output_dir}/weekly_pattern.png"
            )
            plots_created.append('weekly_pattern.png')
            plt.close(fig)
        
        print(f"Created {len(plots_created)} plots in {output_dir}")
        return plots_created