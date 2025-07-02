"""
Data processing module for demand forecasting.
Handles various data sources for warehouse and fulfillment center operations.
Handles data loading, cleaning, and preparation for modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Process shipment data for demand analysis and forecasting."""
    
    def __init__(self, data_dir='../data'):
        """Initialize data processor with data directory path."""
        self.data_dir = Path(data_dir)
        self.data_files = [
            'FC8出貨明細(0126-0225).xlsx',
            'FC8出貨明細(0226-0325).xlsx', 
            'FC8出貨明細(0326-0425).xlsx',
            'FC8出貨明細(0426-0525).xlsx'
        ]
        self.combined_data = None
        self.daily_shipments = None
        self.sku_daily = None
    
    def load_and_combine_data(self):
        """Load all Excel files and combine into single dataframe."""
        all_data = []
        
        for file in self.data_files:
            file_path = self.data_dir / file
            if file_path.exists():
                print(f"Loading {file}...")
                df = pd.read_excel(file_path)
                
                # Standardize date column
                date_col = self._find_date_column(df)
                if date_col:
                    df['shipment_date'] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Standardize SKU column
                sku_col = self._find_sku_column(df)
                if sku_col:
                    df['sku_id'] = df[sku_col]
                
                all_data.append(df)
                print(f"  - Loaded {len(df)} records")
        
        # Combine all dataframes
        self.combined_data = pd.concat(all_data, ignore_index=True)
        
        # Filter out invalid dates
        self.combined_data = self.combined_data[self.combined_data['shipment_date'].notna()]
        
        print(f"\nTotal records loaded: {len(self.combined_data)}")
        return self.combined_data
    
    def _find_date_column(self, df):
        """Find the date column in dataframe."""
        date_keywords = ['shipment_datetime', 'date', 'shippdat']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                return col
        return None
    
    def _find_sku_column(self, df):
        """Find the SKU column in dataframe."""
        sku_keywords = ['sku', 'internal_sku_id']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in sku_keywords):
                return col
        return None
    
    def create_daily_aggregation(self):
        """Create daily shipment aggregation."""
        if self.combined_data is None:
            raise ValueError("No data loaded. Call load_and_combine_data() first.")
        
        self.combined_data['date'] = self.combined_data['shipment_date'].dt.date
        self.daily_shipments = self.combined_data.groupby('date').size().reset_index(name='shipment_count')
        self.daily_shipments['date'] = pd.to_datetime(self.daily_shipments['date'])
        
        return self.daily_shipments
    
    def create_sku_daily_aggregation(self):
        """Create SKU-level daily aggregation."""
        if self.combined_data is None:
            raise ValueError("No data loaded. Call load_and_combine_data() first.")
        
        if 'sku_id' in self.combined_data.columns:
            self.sku_daily = self.combined_data.groupby(['date', 'sku_id']).size().reset_index(name='quantity')
            return self.sku_daily
        else:
            print("Warning: No SKU column found in data")
            return None
    
    def get_data_summary(self):
        """Get summary statistics of the data."""
        if self.combined_data is None:
            raise ValueError("No data loaded. Call load_and_combine_data() first.")
        
        summary = {
            'date_range': {
                'start': self.combined_data['shipment_date'].min(),
                'end': self.combined_data['shipment_date'].max()
            },
            'total_shipments': len(self.combined_data),
            'unique_skus': self.combined_data['sku_id'].nunique() if 'sku_id' in self.combined_data.columns else 0,
            'daily_stats': {
                'mean': self.daily_shipments['shipment_count'].mean() if self.daily_shipments is not None else 0,
                'std': self.daily_shipments['shipment_count'].std() if self.daily_shipments is not None else 0,
                'min': self.daily_shipments['shipment_count'].min() if self.daily_shipments is not None else 0,
                'max': self.daily_shipments['shipment_count'].max() if self.daily_shipments is not None else 0
            }
        }
        
        return summary
    
    def save_processed_data(self, output_dir='../data'):
        """Save processed data to CSV files."""
        output_path = Path(output_dir)
        
        if self.daily_shipments is not None:
            self.daily_shipments.to_csv(output_path / 'daily_shipments.csv', index=False)
            print(f"Saved daily_shipments.csv")
        
        if self.sku_daily is not None:
            self.sku_daily.to_csv(output_path / 'sku_daily_shipments.csv', index=False)
            print(f"Saved sku_daily_shipments.csv")