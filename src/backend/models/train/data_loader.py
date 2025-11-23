import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class StockDataLoader:    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds common technical indicators:
        - Return, Simple Moving Averages (SMA), MACD, Relative Strength Index (RSI)
        """
        # Calculate daily returns
        df['Return'] = df['Close'].pct_change()
        
        # Simple Moving Averages (SMA)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        # Exponential Moving Averages (EMA) for MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Relative Strength Index (RSI) - 14 days
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        
        # Handle division by zero for RSI (rare, but possible if loss is zero)
        rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Drop rows with NaN values resulting from indicator calculation
        df = df.dropna().reset_index(drop=True)
        
        return df

    def load_stock(
        self,
        ticker: str,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load stock data and split into train/val/test
        """
        filepath = self.data_dir / f'{ticker}.csv'
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.title()
        
        # Parse date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
        
        # Validate required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        
        # Clean data
        df = self._clean_data(df)
        
        # ADD TECHNICAL INDICATORS
        df = self._add_technical_indicators(df)
        
        # Time-series split
        n = len(df)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Loaded {ticker}: Total={n}, Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:        
        # Remove duplicates
        if 'Date' in df.columns:
            df = df.drop_duplicates(subset=['Date'], keep='last')
        
        # Handle missing values
        price_cols = ['Open', 'High', 'Low', 'Close']
        df[price_cols] = df[price_cols].ffill().bfill()
        df['Volume'] = df['Volume'].fillna(0)
        
        # Remove rows where all prices are zero
        df = df[~(df[price_cols] == 0).all(axis=1)]
        
        # Validate OHLC relationships
        mask = (df['High'] >= df['Low']) & \
               (df['High'] >= df['Open']) & \
               (df['High'] >= df['Close']) & \
               (df['Low'] <= df['Open']) & \
               (df['Low'] <= df['Close'])
        
        invalid_count = (~mask).sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid OHLC rows, fixing...")
            df.loc[~mask, 'High'] = df.loc[~mask, price_cols].max(axis=1)
            df.loc[~mask, 'Low'] = df.loc[~mask, price_cols].min(axis=1)
        
        # Remove extreme outliers (>10 standard deviations)
        for col in price_cols:
            mean = df[col].mean()
            std = df[col].std()
            df = df[np.abs(df[col] - mean) <= (10 * std)]
        
        return df.reset_index(drop=True)