"""
Unified Demand Forecasting Model
A comprehensive model class that encapsulates all forecasting functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import required models
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet


class DemandForecastModel:
    """
    Unified demand forecasting model that combines data processing,
    multiple forecasting algorithms, and prediction capabilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the demand forecast model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or self._default_config()
        self.data = None
        self.daily_data = None
        self.sku_data = None
        self.models = {}
        self.best_model = None
        self.performance_metrics = {}
        self.is_trained = False
        
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'test_size': 0.2,
            'ma_windows': [7, 14, 30],
            'sarima_order': (1, 1, 1),
            'sarima_seasonal_order': (1, 1, 1, 7),
            'prophet_params': {
                'yearly_seasonality': False,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.05
            },
            'forecast_days': 30,
            'confidence_interval': 0.95
        }
    
    def load_data(self, data_path: Union[str, Path, pd.DataFrame]) -> 'DemandForecastModel':
        """
        Load data from file path or DataFrame.
        
        Args:
            data_path: Path to CSV file or DataFrame with shipment data
            
        Returns:
            self for method chaining
        """
        if isinstance(data_path, pd.DataFrame):
            self.daily_data = data_path
        else:
            self.daily_data = pd.read_csv(data_path)
            
        # Ensure date column is datetime
        if 'date' in self.daily_data.columns:
            self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
        
        self.daily_data = self.daily_data.sort_values('date')
        print(f"Loaded {len(self.daily_data)} days of data")
        print(f"Date range: {self.daily_data['date'].min()} to {self.daily_data['date'].max()}")
        
        return self
    
    def load_sku_data(self, sku_path: Union[str, Path, pd.DataFrame]) -> 'DemandForecastModel':
        """
        Load SKU-level data.
        
        Args:
            sku_path: Path to CSV file or DataFrame with SKU data
            
        Returns:
            self for method chaining
        """
        if isinstance(sku_path, pd.DataFrame):
            self.sku_data = sku_path
        else:
            self.sku_data = pd.read_csv(sku_path)
            
        if 'date' in self.sku_data.columns:
            self.sku_data['date'] = pd.to_datetime(self.sku_data['date'])
            
        print(f"Loaded SKU data with {self.sku_data['sku_id'].nunique()} unique SKUs")
        
        return self
    
    def train(self, models: Optional[List[str]] = None) -> 'DemandForecastModel':
        """
        Train specified models or all available models.
        
        Args:
            models: List of model names to train. If None, trains all models.
                   Options: ['ma', 'sarima', 'prophet', 'ensemble']
                   
        Returns:
            self for method chaining
        """
        if self.daily_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Split data
        self._split_data()
        
        # Define available models
        available_models = {
            'ma': self._train_moving_average,
            'sarima': self._train_sarima,
            'prophet': self._train_prophet,
            'ensemble': self._train_ensemble
        }
        
        # Select models to train
        if models is None:
            models = ['ma', 'sarima', 'prophet', 'ensemble']
        
        # Train selected models
        for model_name in models:
            if model_name in available_models:
                print(f"\nTraining {model_name.upper()} model...")
                try:
                    available_models[model_name]()
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
        
        # Select best model
        self._select_best_model()
        self.is_trained = True
        
        return self
    
    def _split_data(self):
        """Split data into train and test sets."""
        split_idx = int(len(self.daily_data) * (1 - self.config['test_size']))
        split_date = self.daily_data.iloc[split_idx]['date']
        
        self.train_data = self.daily_data[self.daily_data['date'] <= split_date]
        self.test_data = self.daily_data[self.daily_data['date'] > split_date]
        
        print(f"Train: {len(self.train_data)} days, Test: {len(self.test_data)} days")
    
    def _train_moving_average(self):
        """Train moving average models."""
        ma_results = {}
        
        for window in self.config['ma_windows']:
            train_ma = self.train_data.copy()
            train_ma[f'ma_{window}'] = train_ma['shipment_count'].rolling(
                window=window, min_periods=1
            ).mean()
            
            forecast_value = train_ma[f'ma_{window}'].iloc[-1]
            test_forecast = [forecast_value] * len(self.test_data)
            
            mae = mean_absolute_error(self.test_data['shipment_count'], test_forecast)
            rmse = np.sqrt(mean_squared_error(self.test_data['shipment_count'], test_forecast))
            
            ma_results[window] = {
                'forecast': test_forecast,
                'mae': mae,
                'rmse': rmse,
                'last_value': forecast_value
            }
            
            self.performance_metrics[f'{window}-day MA'] = {'MAE': mae, 'RMSE': rmse}
        
        self.models['ma'] = ma_results
    
    def _train_sarima(self):
        """Train SARIMA model."""
        try:
            train_ts = self.train_data.set_index('date')['shipment_count']
            
            model = SARIMAX(
                train_ts,
                order=self.config['sarima_order'],
                seasonal_order=self.config['sarima_seasonal_order'],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            sarima_fit = model.fit(disp=False)
            forecast = sarima_fit.forecast(steps=len(self.test_data))
            
            mae = mean_absolute_error(self.test_data['shipment_count'], forecast)
            rmse = np.sqrt(mean_squared_error(self.test_data['shipment_count'], forecast))
            
            self.models['sarima'] = {
                'model': sarima_fit,
                'forecast': forecast.values,
                'mae': mae,
                'rmse': rmse
            }
            
            self.performance_metrics['SARIMA'] = {'MAE': mae, 'RMSE': rmse}
            
        except Exception as e:
            print(f"SARIMA training failed: {e}")
    
    def _train_prophet(self):
        """Train Prophet model."""
        prophet_train = self.train_data[['date', 'shipment_count']].copy()
        prophet_train.columns = ['ds', 'y']
        
        model = Prophet(**self.config['prophet_params'])
        model.fit(prophet_train)
        
        future_dates = pd.DataFrame({'ds': self.test_data['date']})
        forecast = model.predict(future_dates)
        predictions = forecast['yhat'].values
        
        mae = mean_absolute_error(self.test_data['shipment_count'], predictions)
        rmse = np.sqrt(mean_squared_error(self.test_data['shipment_count'], predictions))
        
        self.models['prophet'] = {
            'model': model,
            'forecast': predictions,
            'mae': mae,
            'rmse': rmse
        }
        
        self.performance_metrics['Prophet'] = {'MAE': mae, 'RMSE': rmse}
    
    def _train_ensemble(self):
        """Train ensemble model."""
        if 'prophet' not in self.models or 'ma' not in self.models:
            print("Ensemble requires Prophet and MA models")
            return
        
        # Find best MA model
        best_ma_window = min(
            self.models['ma'].keys(),
            key=lambda x: self.models['ma'][x]['mae']
        )
        
        # Average predictions
        ensemble_forecast = (
            self.models['prophet']['forecast'] + 
            np.array(self.models['ma'][best_ma_window]['forecast'])
        ) / 2
        
        mae = mean_absolute_error(self.test_data['shipment_count'], ensemble_forecast)
        rmse = np.sqrt(mean_squared_error(self.test_data['shipment_count'], ensemble_forecast))
        
        self.models['ensemble'] = {
            'forecast': ensemble_forecast,
            'mae': mae,
            'rmse': rmse,
            'components': ['prophet', f'{best_ma_window}-day MA']
        }
        
        self.performance_metrics['Ensemble'] = {'MAE': mae, 'RMSE': rmse}
    
    def _select_best_model(self):
        """Select the best performing model based on MAE."""
        if not self.performance_metrics:
            raise ValueError("No models trained yet")
        
        self.best_model = min(
            self.performance_metrics.keys(),
            key=lambda x: self.performance_metrics[x]['MAE']
        )
        
        print(f"\nBest model: {self.best_model}")
        print(f"MAE: {self.performance_metrics[self.best_model]['MAE']:.2f}")
        print(f"RMSE: {self.performance_metrics[self.best_model]['RMSE']:.2f}")
    
    def predict(self, days_ahead: Optional[int] = None) -> pd.DataFrame:
        """
        Generate future predictions using the best model.
        
        Args:
            days_ahead: Number of days to forecast (default from config)
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        days_ahead = days_ahead or self.config['forecast_days']
        
        # Use Prophet for future predictions (most flexible)
        if 'prophet' in self.models:
            return self._predict_with_prophet(days_ahead)
        else:
            raise ValueError("Prophet model required for future predictions")
    
    def _predict_with_prophet(self, days_ahead: int) -> pd.DataFrame:
        """Generate predictions using Prophet model."""
        model = self.models['prophet']['model']
        
        # Create future dates
        last_date = self.daily_data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        # Generate predictions
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        
        # Prepare results
        predictions = pd.DataFrame({
            'date': forecast['ds'],
            'predicted_shipments': forecast['yhat'].round().astype(int),
            'lower_bound': forecast['yhat_lower'].round().astype(int),
            'upper_bound': forecast['yhat_upper'].round().astype(int)
        })
        
        # Ensure non-negative
        for col in ['predicted_shipments', 'lower_bound', 'upper_bound']:
            predictions[col] = predictions[col].clip(lower=0)
        
        return predictions
    
    def predict_skus(self, top_n: int = 20, days_ahead: Optional[int] = None) -> pd.DataFrame:
        """
        Predict demand for top SKUs.
        
        Args:
            top_n: Number of top SKUs to predict
            days_ahead: Forecast horizon
            
        Returns:
            DataFrame with SKU predictions
        """
        if self.sku_data is None:
            raise ValueError("No SKU data loaded. Call load_sku_data() first.")
        
        days_ahead = days_ahead or self.config['forecast_days']
        
        # Calculate SKU summary
        sku_summary = self.sku_data.groupby('sku_id').agg({
            'quantity': ['sum', 'mean', 'std', 'count']
        }).round(2)
        sku_summary.columns = ['total_quantity', 'avg_daily', 'std_daily', 'days_sold']
        sku_summary = sku_summary.sort_values('total_quantity', ascending=False)
        
        # Get top SKUs
        top_skus = sku_summary.head(top_n).index.tolist()
        predictions = []
        
        for i, sku_id in enumerate(top_skus, 1):
            print(f"Predicting SKU {sku_id} ({i}/{len(top_skus)})...", end='\r')
            
            try:
                # Get SKU data
                sku_df = self.sku_data[self.sku_data['sku_id'] == sku_id].copy()
                sku_df = sku_df.rename(columns={'date': 'ds', 'quantity': 'y'})
                
                # Fill missing dates
                date_range = pd.date_range(
                    start=sku_df['ds'].min(),
                    end=sku_df['ds'].max(),
                    freq='D'
                )
                full_dates = pd.DataFrame({'ds': date_range})
                sku_df = full_dates.merge(sku_df, on='ds', how='left')
                sku_df['y'] = sku_df['y'].fillna(0)
                
                # Create Prophet model
                model = Prophet(**self.config['prophet_params'])
                model.fit(sku_df)
                
                # Predict
                last_date = sku_df['ds'].max()
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=days_ahead,
                    freq='D'
                )
                future_df = pd.DataFrame({'ds': future_dates})
                
                forecast = model.predict(future_df)
                total_forecast = forecast['yhat'].clip(lower=0).sum()
                
                # Calculate growth
                historical_avg = sku_summary.loc[sku_id, 'avg_daily'] * days_ahead
                growth_rate = ((total_forecast - historical_avg) / historical_avg) * 100
                
                predictions.append({
                    'sku_id': sku_id,
                    'predicted_total': int(total_forecast),
                    'historical_avg': int(historical_avg),
                    'growth_rate': growth_rate,
                    'historical_rank': list(sku_summary.index).index(sku_id) + 1
                })
                
            except Exception as e:
                print(f"\nSKU {sku_id} prediction failed: {e}")
                continue
        
        print("\n")
        
        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        results_df = results_df.sort_values('predicted_total', ascending=False)
        
        return results_df
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of model performance metrics."""
        if not self.performance_metrics:
            raise ValueError("No models trained yet")
        
        return pd.DataFrame(self.performance_metrics).T.round(2)
    
    def save(self, filepath: Union[str, Path]):
        """
        Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        model_data = {
            'config': self.config,
            'models': self.models,
            'performance_metrics': self.performance_metrics,
            'best_model': self.best_model,
            'is_trained': self.is_trained,
            'data_info': {
                'n_days': len(self.daily_data) if self.daily_data is not None else 0,
                'date_range': [
                    str(self.daily_data['date'].min()) if self.daily_data is not None else None,
                    str(self.daily_data['date'].max()) if self.daily_data is not None else None
                ],
                'n_skus': self.sku_data['sku_id'].nunique() if self.sku_data is not None else 0
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'DemandForecastModel':
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded DemandForecastModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        model = cls(config=model_data['config'])
        model.models = model_data['models']
        model.performance_metrics = model_data['performance_metrics']
        model.best_model = model_data['best_model']
        model.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
        print(f"Best model: {model.best_model}")
        print(f"Data info: {model_data['data_info']}")
        
        return model


# Example usage
if __name__ == "__main__":
    # Create and train model
    model = DemandForecastModel()
    
    # Load data
    model.load_data('../data/daily_shipments.csv')
    model.load_sku_data('../data/sku_daily_shipments.csv')
    
    # Train all models
    model.train()
    
    # Generate predictions
    predictions = model.predict(days_ahead=30)
    print("\nFuture predictions:")
    print(predictions.head())
    
    # Predict top SKUs
    sku_predictions = model.predict_skus(top_n=10)
    print("\nTop SKU predictions:")
    print(sku_predictions)
    
    # Save model
    model.save('../models/demand_forecast_model.pkl')