import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class StockDataLoader:    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def _create_stationary_features(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Converts raw price data into stationary features.
        """
        # 1. Log Return (Daily)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Volatility
        df['range_pct'] = (df['High'] - df['Low']) / df['Close']
        df['close_loc'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)
        
        # 3. Volume
        df['volume_log_change'] = np.log(df['Volume'] / df['Volume'].shift(1).replace(0, 1) + 1e-9)
        
        # 4. Moving Averages
        sma_10 = df['Close'].rolling(window=10).mean()
        sma_30 = df['Close'].rolling(window=30).mean()
        df['dist_sma_10'] = (df['Close'] - sma_10) / sma_10
        df['dist_sma_30'] = (df['Close'] - sma_30) / sma_30
        
        # 5. MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_norm'] = macd / df['Close']
        df['macd_sig_norm'] = macd_signal / df['Close']
        
        # 6. RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()      
        rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        
        # Calculate Future Return over 'horizon' days
        future_close = df['Close'].shift(-horizon)
        df['target_log_return'] = np.log(future_close / df['Close'])
        
        # Direction: Did it go up over the next 'horizon' days?
        # ADDED: "Dead Zone" threshold. If move is < 0.5%, treat as noise (optional logic)
        df['target_direction'] = (df['target_log_return'] > 0).astype(np.int32)
        
        df = df.dropna().reset_index(drop=True)
        return df

    def load_stock(
        self,
        ticker: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        pred_horizon: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        filepath = self.data_dir / f'{ticker}.csv'
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.title()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
        df = self._clean_data(df)
        
        # Pass the horizon to feature creator
        df = self._create_stationary_features(df, horizon=pred_horizon)
        
        logger.info(f"Features created with horizon={pred_horizon}. Final shape: {df.shape}")
        
        # Split
        n = len(df)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:        
        price_cols = ['Open', 'High', 'Low', 'Close']
        df[price_cols] = df[price_cols].ffill().bfill()
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        
        mask = (df['High'] >= df['Low']) & (df['High'] >= df['Close']) & (df['Low'] <= df['Close'])
        if (~mask).sum() > 0:
            df.loc[~mask, 'High'] = df.loc[~mask, price_cols].max(axis=1)
            df.loc[~mask, 'Low'] = df.loc[~mask, price_cols].min(axis=1)
            
        return df.reset_index(drop=True)