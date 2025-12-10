import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from config import CONFIG

class UniversalDataProcessor:
    def __init__(self):
        self.data_dir = Path(CONFIG['data_dir'])
        self.tickers = CONFIG['tickers']
        self.stock_to_id = {t: i for i, t in enumerate(self.tickers)}
        self.feature_cols = [] 
        self.valid_tickers = []

    def load_and_process(self):
        print(f"Loading data from {self.data_dir}...")
        dfs = []
        start_cutoff = pd.to_datetime(CONFIG['force_start_date'])
        
        for ticker in self.tickers:
            fpath = self.data_dir / f"{ticker}.csv"
            if fpath.exists():
                try:
                    df = pd.read_csv(fpath)
                    df.columns = df.columns.str.strip().str.title()
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.set_index('Date').sort_index()
                    
                    df = df[df.index >= start_cutoff]
                    if len(df) < CONFIG['seq_len'] * 2: continue
                        
                    df = self._engineer_features(df)
                    df['Ticker'] = ticker
                    df = df.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if not df.empty: dfs.append(df)
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
        
        if not dfs: raise ValueError("No valid data loaded.")

        # Align Dates (Fixed Universe Intersection)
        big_df = pd.concat(dfs)
        date_counts = big_df.groupby(big_df.index)['Ticker'].count()
        valid_dates = date_counts[date_counts == len(dfs)].index # Only dates where ALL stocks exist
        
        universal_df = big_df[big_df.index.isin(valid_dates)].sort_index().reset_index()
        
        self.valid_tickers = sorted(universal_df['Ticker'].unique())
        # Ensure mapping is consistent
        self.stock_to_id = {t: i for i, t in enumerate(self.valid_tickers)}
        universal_df['Stock_ID'] = universal_df['Ticker'].map(self.stock_to_id)
        
        # Define Features
        self.feature_cols = [c for c in universal_df.columns 
                             if c not in ['Date', 'Ticker', 'Stock_ID', 'Target_5D', 'Log_Ret_Raw']]
        
        # Cross-Sectional Z-Score Normalization
        print("Applying Cross-Sectional Normalization...")
        universal_df[self.feature_cols] = universal_df.groupby('Date')[self.feature_cols].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        ).clip(-3, 3)
        
        return universal_df

    def _engineer_features(self, df):
        df = df.copy()
        # Log Returns
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Log_Ret_Raw'] = df['Log_Ret'] # Unnormalized for target
        
        # Target: 5-Day Future Cumulative Return
        df['Target_5D'] = df['Log_Ret_Raw'].rolling(CONFIG['pred_horizon']).sum().shift(-CONFIG['pred_horizon'])
        
        # Technical Indicators
        for p in [10, 20, 60]:
            sma = df['Close'].rolling(p).mean()
            df[f'Dist_SMA_{p}'] = (df['Close'] / sma - 1)
            
        df['Vol_20'] = df['Log_Ret'].rolling(20).std()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        df['RSI'] = (100 - (100 / (1 + rs))) / 100.0
        return df.dropna()

class UniversalDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.df = df.sort_values(['Date', 'Ticker'])
        self.feature_cols = feature_cols
        self.seq_len = CONFIG['seq_len']
        self.num_stocks = len(df['Ticker'].unique())
        
        self.dates = sorted(df['Date'].unique())
        self.date_to_idx = {date: i for i, date in enumerate(self.dates)}
        # Valid start date must allow for seq_len lookback
        self.valid_dates = self.dates[self.seq_len:]

    def __len__(self):
        return len(self.valid_dates)

    def __getitem__(self, idx):
        # Get specific date
        date = self.valid_dates[idx]
        
        # Get window [t-seq_len : t]
        window_start_idx = self.date_to_idx[date] - self.seq_len + 1
        window_dates = self.dates[window_start_idx : self.date_to_idx[date] + 1]
        
        window_df = self.df[self.df['Date'].isin(window_dates)]
        
        # Extract Features: [Seq_Len * Stocks, Features]
        flat_features = window_df[self.feature_cols].values.astype(np.float32)
        
        # Reshape to [Seq_Len, Stocks, Features]
        x = flat_features.reshape(self.seq_len, self.num_stocks, -1)
        
        # Transpose to [Stocks, Seq_Len, Features] for model processing
        x = x.transpose(1, 0, 2) 
        x = torch.FloatTensor(x)
        
        # Static IDs
        stock_ids = torch.arange(self.num_stocks)
        
        # Target for the current date t
        y = window_df[window_df['Date'] == date]['Target_5D'].values.astype(np.float32)
        
        return x, stock_ids, torch.FloatTensor(y)

def get_dataloaders(processor):
    df = processor.load_and_process()
    
    # Chronological Split
    dates = sorted(df['Date'].unique())
    split_idx = int(len(dates) * 0.8)
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])
    
    train_df = df[df['Date'].isin(train_dates)]
    test_df = df[df['Date'].isin(test_dates)]
    
    train_ds = UniversalDataset(train_df, processor.feature_cols)
    test_ds = UniversalDataset(test_df, processor.feature_cols)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    return train_loader, test_loader, test_df