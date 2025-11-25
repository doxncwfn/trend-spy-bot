import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from config import FEATURE_COLS

logger = logging.getLogger(__name__)

class CombinedLoss(nn.Module):
    def __init__(self, lambda_reg=1.0, lambda_cls=5.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_reg, pred_cls, target_reg, target_cls):
        loss_reg = self.mse(pred_reg, target_reg)
        loss_cls = self.bce(pred_cls, target_cls.float())
        return self.lambda_reg * loss_reg + self.lambda_cls * loss_cls

class StockDataset(Dataset):
    def __init__(self, df, seq_len=60, scaled_features=None):
        self.seq_len = seq_len
        
        if scaled_features is None:
            raise ValueError("StockDataset requires pre-scaled features.")
            
        self.features = scaled_features.astype(np.float32)
        self.target_log_return = df['target_log_return'].values.astype(np.float32)
        self.target_direction = df['target_direction'].values.astype(np.int32)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y_reg = self.target_log_return[idx + self.seq_len - 1]
        y_cls = self.target_direction[idx + self.seq_len - 1]
        return torch.tensor(x), torch.tensor(y_reg), torch.tensor(y_cls, dtype=torch.long)

class LSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=128, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # LSTM Block
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # Transformer Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size*2, nhead=n_heads, dropout=dropout,
            batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Attention Pooling
        self.attention_pool = nn.Linear(hidden_size*2, 1)

        # Heads
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size*2, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size*2, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        
        return self.reg_head(pooled).squeeze(-1), self.cls_head(pooled).squeeze(-1)

class StockForecaster:
    def __init__(self, config, device='cpu'):
        self.config = config
        self.seq_len = config['seq_length']
        self.device = device
        
        self.model = LSTMTransformer(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            n_heads=config['num_attention_heads'],
            n_layers=config['num_transformer_layers'],
            dropout=config['dropout']
        ).to(device)
        
        self.scaler = StandardScaler()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = CombinedLoss(config['lambda_reg'], config['lambda_cls'])

    def prepare_data(self, df, fit_scaler=False):
        try:
            raw_features = df[FEATURE_COLS].values.astype(np.float32)
        except KeyError as e:
            raise KeyError(f"Dataframe missing columns. Expected {FEATURE_COLS}. Missing: {e}")
        
        if fit_scaler:
            scaled_features = self.scaler.fit_transform(raw_features)
        else:
            scaled_features = self.scaler.transform(raw_features)
            
        dataset = StockDataset(df, self.seq_len, scaled_features)
        
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=fit_scaler, drop_last=True)

    def fit(self, train_df, val_df, epochs):
        train_loader = self.prepare_data(train_df, fit_scaler=True)
        val_loader = self.prepare_data(val_df, fit_scaler=False)

        history = {'train_loss':[], 'val_loss':[]}
        best_loss = float('inf')
        wait = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for x, y_reg, y_cls in train_loader:
                x, y_reg, y_cls = x.to(self.device), y_reg.to(self.device), y_cls.to(self.device)
                self.optimizer.zero_grad()
                pred_reg, pred_cls = self.model(x)
                loss = self.criterion(pred_reg, pred_cls, y_reg, y_cls)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y_reg, y_cls in val_loader:
                    x, y_reg, y_cls = x.to(self.device), y_reg.to(self.device), y_cls.to(self.device)
                    pred_reg, pred_cls = self.model(x)
                    loss = self.criterion(pred_reg, pred_cls, y_reg, y_cls)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                wait = 0
                torch.save(self.model.state_dict(), "temp_best.pth")
            else:
                wait += 1
                if wait >= self.config['early_stopping_patience']:
                    logger.info("Early stopping")
                    break
        
        if os.path.exists("temp_best.pth"):
            self.model.load_state_dict(torch.load("temp_best.pth"))
            os.remove("temp_best.pth")
        return history

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': {'seq_len': self.seq_len, 'input_size': self.config['input_size']}
        }, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.scaler = ckpt['scaler']