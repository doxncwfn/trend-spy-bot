import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import os

logger = logging.getLogger(__name__)

class CombinedLoss(nn.Module):
    """
    Combines MSE for Regression (Return Magnitude) and BCE for Classification (Direction).
    L = lambda_reg * MSE + lambda_cls * BCE
    """
    def __init__(self, lambda_reg=1.0, lambda_cls=3.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss() 

    def forward(self, pred_reg, pred_cls, target_reg, target_cls):
        # Regression Loss (Return Magnitude)
        loss_reg = self.mse(pred_reg, target_reg)

        # Classification Loss (Direction)
        loss_cls = self.bce(pred_cls, target_cls.float()) 
        
        return self.lambda_reg * loss_reg + self.lambda_cls * loss_cls

class StockDataset(Dataset):
    def __init__(self, data, seq_len=60):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        
        # 1. Regression Target (Return Value)
        close_next = self.data[idx + self.seq_len, 3]
        close_prev = self.data[idx + self.seq_len - 1, 3]
        y_reg = (close_next / close_prev) - 1
        
        # 2. Classification Target (Direction: 1 for Up, 0 for Down/No Change)
        y_cls = (y_reg > 0).astype(np.int32) 
        
        return (
            torch.tensor(x, dtype=torch.float32), 
            torch.tensor(y_reg, dtype=torch.float32), 
            torch.tensor(y_cls, dtype=torch.long)
        )


class LSTMTransformer(nn.Module):
    def __init__(self, input_size=11, seq_len=60, hidden_size=128, n_heads=8, n_layers=3, dropout=0.1): 
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # 1-4. Shared Feature Extractor and Attention Pooling
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, 
                            batch_first=True, dropout=0.2, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size*2, nhead=n_heads, dim_feedforward=512,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attention_pool = nn.Parameter(torch.randn(hidden_size*2, 1))
        nn.init.xavier_uniform_(self.attention_pool)
        
        # 5. Two Separate Prediction Heads
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Shared Encoder
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        
        # Attended Global Feature (Shared context vector)
        attn_scores = torch.matmul(x, self.attention_pool)
        attn_weights = torch.softmax(attn_scores, dim=1)
        shared_features = torch.sum(x * attn_weights, dim=1)
        
        # Two predictions
        pred_reg = self.regression_head(shared_features).squeeze(-1)
        pred_cls = self.classification_head(shared_features).squeeze(-1)
        
        return pred_reg, pred_cls


class StockForecaster:
    def __init__(self, seq_len=60, device='cpu'):
        self.seq_len = seq_len
        self.device = device
        
        self.scaler = StandardScaler()
        self.model = LSTMTransformer(seq_len=seq_len).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        
        # Use the Combined Loss for MTL with increased CLS weight
        self.criterion = CombinedLoss(lambda_reg=1.0, lambda_cls=10.0) 
        
    def prepare_data(self, df):
        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'SMA_10', 'SMA_30', 'MACD', 'MACD_Signal', 'RSI']
        data = df[cols].values
        
        scaled = self.scaler.fit_transform(data)
        
        dataset = StockDataset(scaled, seq_len=self.seq_len) 
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
        
        return loader, len(dataset)
    
    def fit(self, train_df, val_df, epochs=100):
        train_loader, train_n = self.prepare_data(train_df)
        val_loader, val_n = self.prepare_data(val_df)
        
        logger.info(f"Train: {train_n}, Val: {val_n}")
        
        patience = 20
        wait = 0
        # Early stopping based on the actual Combined Validation Loss
        best_loss = float('inf') 
        
        history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': []} 
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for x, y_reg, y_cls in train_loader: 
                x, y_reg, y_cls = x.to(self.device), y_reg.to(self.device), y_cls.to(self.device)
                self.optimizer.zero_grad()
                
                pred_reg, pred_cls = self.model(x)
                loss = self.criterion(pred_reg, pred_cls, y_reg, y_cls)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_preds_reg = [] 
            val_targets_reg = []
            standard_mse = nn.MSELoss() # Used only for reporting val RMSE/MAE
            
            with torch.no_grad():
                for x, y_reg, y_cls in val_loader: 
                    x, y_reg, y_cls = x.to(self.device), y_reg.to(self.device), y_cls.to(self.device)
                    pred_reg, pred_cls = self.model(x)

                    # Calculate the ACTUAL COMBINED LOSS for monitoring/early stopping
                    loss_combined = self.criterion(pred_reg, pred_cls, y_reg, y_cls) 
                    val_loss += loss_combined.item()
                    
                    # Store raw regression predictions for RMSE/MAE calculation
                    val_preds_reg.extend(pred_reg.cpu().numpy())
                    val_targets_reg.extend(y_reg.cpu().numpy())
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)

            val_preds_np = np.array(val_preds_reg)
            val_targets_np = np.array(val_targets_reg)
            
            # Calculate RMSE/MAE for reporting absolute error
            val_rmse = np.sqrt(mean_squared_error(val_targets_np, val_preds_np))
            val_mae = mean_absolute_error(val_targets_np, val_preds_np)
            
            history['val_rmse'].append(val_rmse)
            history['val_mae'].append(val_mae)
            
            # Early stopping based on the Combined Validation Loss (val_loss)
            if val_loss < best_loss - 1e-5: 
                best_loss = val_loss
                wait = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if epoch % 10 == 0 or wait == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.6f} | Val Loss (Combined) {val_loss:.6f} | Val RMSE {val_rmse:.4f}") 
        
        self.model.load_state_dict(torch.load("best_model.pth"))
        os.remove("best_model.pth")
        return history
    
    def predict(self, data_np):
        """
        Predicts the next closing price using the regression head output (pred_reg).
        """
        self.model.eval()
        with torch.no_grad():
            scaled = self.scaler.transform(data_np.reshape(-1, 11))
            x = torch.FloatTensor(scaled[-self.seq_len:]).unsqueeze(0).to(self.device)
            
            pred_reg, _ = self.model(x) 
            
            pred_return = pred_reg.item() if pred_reg.dim() == 0 else pred_reg.squeeze().item()
            
            last_close = data_np[-1, 3] 
            return last_close * (1 + pred_return)
        
    def save_model(self, path):
        """Save the model checkpoint for production use"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler
        }, path)
        logger.info(f"Model checkpoint saved to {path}")

    def load_model(self, path):
        """Load a saved checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        logger.info(f"Model loaded from {path}")