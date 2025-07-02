"""
Forecasting models for demand prediction.
Includes Moving Average, SARIMA, Prophet, and Ensemble models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import pickle
import warnings
warnings.filterwarnings('ignore')


class DemandForecaster:
    """Main forecasting class for FCOrder demand prediction."""
    
    def __init__(self):
        """Initialize forecaster with empty models."""
        self.models = {}
        self.train_data = None
        self.test_data = None
        self.best_model = None
        self.performance_metrics = {}
    
    def prepare_data(self, daily_shipments, test_size=0.2):
        """Split data into train and test sets."""
        daily_shipments = daily_shipments.sort_values('date')
        
        # Calculate split point
        split_idx = int(len(daily_shipments) * (1 - test_size))
        split_date = daily_shipments.iloc[split_idx]['date']
        
        self.train_data = daily_shipments[daily_shipments['date'] <= split_date]
        self.test_data = daily_shipments[daily_shipments['date'] > split_date]
        
        print(f"Train data: {len(self.train_data)} days")
        print(f"Test data: {len(self.test_data)} days")
        
        return self.train_data, self.test_data
    
    def train_moving_average(self, windows=[7, 14, 30]):
        """Train moving average models with different window sizes."""
        ma_results = {}
        
        for window in windows:
            # Calculate moving average on training data
            train_ma = self.train_data.copy()
            train_ma[f'ma_{window}'] = train_ma['shipment_count'].rolling(window=window, min_periods=1).mean()
            
            # Use last value as forecast
            forecast_value = train_ma[f'ma_{window}'].iloc[-1]
            test_forecast = [forecast_value] * len(self.test_data)
            
            # Calculate metrics
            mae = mean_absolute_error(self.test_data['shipment_count'], test_forecast)
            rmse = np.sqrt(mean_squared_error(self.test_data['shipment_count'], test_forecast))
            
            ma_results[window] = {
                'model': f'{window}-day MA',
                'forecast': test_forecast,
                'mae': mae,
                'rmse': rmse
            }
            
            self.performance_metrics[f'{window}-day MA'] = {'MAE': mae, 'RMSE': rmse}
        
        self.models['moving_average'] = ma_results
        return ma_results
    
    def train_sarima(self, order=(1,1,1), seasonal_order=(1,1,1,7)):
        """Train SARIMA model with specified parameters."""
        try:
            # Prepare time series data
            train_ts = self.train_data.set_index('date')['shipment_count']
            
            # Fit SARIMA model
            model = SARIMAX(train_ts, 
                           order=order,
                           seasonal_order=seasonal_order,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            
            sarima_fit = model.fit(disp=False)
            
            # Make predictions
            forecast = sarima_fit.forecast(steps=len(self.test_data))
            
            # Calculate metrics
            mae = mean_absolute_error(self.test_data['shipment_count'], forecast)
            rmse = np.sqrt(mean_squared_error(self.test_data['shipment_count'], forecast))
            
            self.models['sarima'] = {
                'model': sarima_fit,
                'forecast': forecast.values,
                'mae': mae,
                'rmse': rmse
            }
            
            self.performance_metrics['SARIMA'] = {'MAE': mae, 'RMSE': rmse}
            
            return self.models['sarima']
            
        except Exception as e:
            print(f"SARIMA model failed: {e}")
            return None
    
    def train_prophet(self, **kwargs):
        """Train Prophet model with custom parameters."""
        # Prepare data for Prophet
        prophet_train = self.train_data[['date', 'shipment_count']].copy()
        prophet_train.columns = ['ds', 'y']
        
        # Set default parameters
        params = {
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05
        }
        params.update(kwargs)
        
        # Create and fit model
        model = Prophet(**params)
        model.fit(prophet_train)
        
        # Make predictions
        future_dates = pd.DataFrame({'ds': self.test_data['date']})
        forecast = model.predict(future_dates)
        predictions = forecast['yhat'].values
        
        # Calculate metrics
        mae = mean_absolute_error(self.test_data['shipment_count'], predictions)
        rmse = np.sqrt(mean_squared_error(self.test_data['shipment_count'], predictions))
        
        self.models['prophet'] = {
            'model': model,
            'forecast': predictions,
            'mae': mae,
            'rmse': rmse
        }
        
        self.performance_metrics['Prophet'] = {'MAE': mae, 'RMSE': rmse}
        
        return self.models['prophet']
    
    def create_ensemble(self, models_to_combine=['prophet', 'moving_average']):
        """Create ensemble model by averaging selected models."""
        ensemble_forecast = None
        model_count = 0
        
        # Combine Prophet forecast
        if 'prophet' in models_to_combine and 'prophet' in self.models:
            ensemble_forecast = self.models['prophet']['forecast'].copy()
            model_count += 1
        
        # Add best MA model
        if 'moving_average' in models_to_combine and 'moving_average' in self.models:
            # Find best MA model
            best_ma = min(self.models['moving_average'].keys(), 
                         key=lambda x: self.models['moving_average'][x]['mae'])
            
            if ensemble_forecast is None:
                ensemble_forecast = np.array(self.models['moving_average'][best_ma]['forecast'])
            else:
                ensemble_forecast += np.array(self.models['moving_average'][best_ma]['forecast'])
            model_count += 1
        
        # Average the forecasts
        if model_count > 0:
            ensemble_forecast = ensemble_forecast / model_count
            
            # Calculate metrics
            mae = mean_absolute_error(self.test_data['shipment_count'], ensemble_forecast)
            rmse = np.sqrt(mean_squared_error(self.test_data['shipment_count'], ensemble_forecast))
            
            self.models['ensemble'] = {
                'forecast': ensemble_forecast,
                'mae': mae,
                'rmse': rmse,
                'models_combined': model_count
            }
            
            self.performance_metrics['Ensemble'] = {'MAE': mae, 'RMSE': rmse}
            
            return self.models['ensemble']
        
        return None
    
    def select_best_model(self):
        """Select the best model based on MAE."""
        if not self.performance_metrics:
            raise ValueError("No models trained yet")
        
        self.best_model = min(self.performance_metrics.keys(), 
                             key=lambda x: self.performance_metrics[x]['MAE'])
        
        print(f"\nBest model: {self.best_model}")
        print(f"MAE: {self.performance_metrics[self.best_model]['MAE']:.2f}")
        print(f"RMSE: {self.performance_metrics[self.best_model]['RMSE']:.2f}")
        
        return self.best_model
    
    def get_performance_summary(self):
        """Get summary of all model performances."""
        return pd.DataFrame(self.performance_metrics).T.round(2)
    
    def save_best_model(self, filepath='../models/best_model.pkl'):
        """Save the best performing model."""
        if self.best_model is None:
            self.select_best_model()
        
        # Save Prophet model if it's the best
        if self.best_model == 'Prophet' and 'prophet' in self.models:
            with open(filepath, 'wb') as f:
                pickle.dump(self.models['prophet']['model'], f)
            print(f"Saved {self.best_model} model to {filepath}")
        else:
            print(f"Model {self.best_model} cannot be saved as pickle")


class SKUForecaster:
    """Forecaster for individual SKU demand prediction."""
    
    def __init__(self):
        """Initialize SKU forecaster."""
        self.sku_models = {}
        self.predictions = []
    
    def predict_sku_demand(self, sku_data, sku_id, days_ahead=30):
        """Predict demand for a single SKU."""
        # Prepare data
        sku_df = sku_data[sku_data['sku_id'] == sku_id].copy()
        sku_df = sku_df.rename(columns={'date': 'ds', 'quantity': 'y'})
        
        # Fill missing dates
        date_range = pd.date_range(start=sku_df['ds'].min(), 
                                  end=sku_df['ds'].max(), 
                                  freq='D')
        full_dates = pd.DataFrame({'ds': date_range})
        sku_df = full_dates.merge(sku_df, on='ds', how='left')
        sku_df['y'] = sku_df['y'].fillna(0)
        
        # Create Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0
        )
        
        # Fit model
        model.fit(sku_df)
        
        # Predict future
        last_date = sku_df['ds'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                    periods=days_ahead, 
                                    freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        
        forecast = model.predict(future_df)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        
        return forecast['yhat'].sum(), forecast, model
    
    def predict_top_skus(self, sku_data, sku_summary, top_n=20, days_ahead=30):
        """Predict demand for top N SKUs."""
        top_skus = sku_summary.head(top_n).index.tolist()
        
        for i, sku_id in enumerate(top_skus, 1):
            print(f"Predicting SKU {sku_id} ({i}/{len(top_skus)})...", end='\r')
            
            try:
                total_forecast, forecast_detail, model = self.predict_sku_demand(
                    sku_data, sku_id, days_ahead
                )
                
                # Calculate growth rate
                historical_avg = sku_summary.loc[sku_id, 'avg_daily'] * days_ahead
                growth_rate = ((total_forecast - historical_avg) / historical_avg) * 100
                
                self.predictions.append({
                    'sku_id': sku_id,
                    'predicted_total': int(total_forecast),
                    'historical_avg': int(historical_avg),
                    'growth_rate': growth_rate,
                    'historical_rank': list(sku_summary.index).index(sku_id) + 1
                })
                
                self.sku_models[sku_id] = model
                
            except Exception as e:
                print(f"\nSKU {sku_id} prediction failed: {e}")
                continue
        
        print("\n")
        
        # Create results dataframe
        results_df = pd.DataFrame(self.predictions)
        results_df = results_df.sort_values('predicted_total', ascending=False)
        
        return results_df