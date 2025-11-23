import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    def __init__(self, data, seq_len=60, predict_returns=False):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.predict_returns = predict_returns

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        
        if self.predict_returns:
            close_next = self.data[idx + self.seq_len, 3]
            close_prev = self.data[idx + self.seq_len - 1, 3]
            y = (close_next / close_prev) - 1
            y = np.float32(y)  # Force float32 to prevent promotion to float64
        else:
            y = self.data[idx + self.seq_len, 3]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# The Model: LSTM → Transformer Encoder → Attention Pooling → Ridge-like Head
class LSTMTransformer(nn.Module):
    def __init__(self, input_size=5, seq_len=60, hidden_size=128, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # 1. Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # 2. LSTM (bidirectional for richer features)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, 
                            batch_first=True, dropout=0.2, bidirectional=True)
        
        # 3. Transformer Encoder (captures long-term patterns)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size*2, nhead=n_heads, dim_feedforward=512,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Learnable attention pooling (better than last hidden state)
        self.attention_pool = nn.Parameter(torch.randn(hidden_size*2, 1))
        nn.init.xavier_uniform_(self.attention_pool)
        
        # 5. Final head (small MLP)
        self.head = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)                    # (B, S, H)
        x, _ = self.lstm(x)                       # (B, S, H*2)
        x = self.transformer(x)                   # (B, S, H*2)
        
        attn_scores = torch.matmul(x, self.attention_pool)      # (B, S, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)        # (B, S, 1)
        x = torch.sum(x * attn_weights, dim=1)                  # (B, H*2)
        
        return self.head(x).squeeze(-1)


# Forecaster Wrapper
class StockForecaster:
    def __init__(self, seq_len=60, predict_returns=True, device='cpu'):
        self.seq_len = seq_len
        self.predict_returns = predict_returns
        self.device = device
        
        self.scaler = StandardScaler()
        self.model = LSTMTransformer(seq_len=seq_len).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
    def prepare_data(self, df):
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = df[cols].values
        
        # Critical: scale AFTER computing returns if needed
        scaled = self.scaler.fit_transform(data)
        
        dataset = StockDataset(scaled, seq_len=self.seq_len, predict_returns=self.predict_returns)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
        
        return loader, len(dataset)
    
    def fit(self, train_df, val_df, epochs=100):
        train_loader, train_n = self.prepare_data(train_df)
        val_loader, val_n = self.prepare_data(val_df)
        
        logger.info(f"Train: {train_n}, Val: {val_n}")
        
        best_loss = float('inf')
        patience = 20
        wait = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x)
                    val_loss += self.criterion(pred, y).item()
            
            val_loss /= len(val_loader)
            
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
                logger.info(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.6f} | Val Loss {val_loss:.6f}")
        
        self.model.load_state_dict(torch.load("best_model.pth"))
        return best_loss
    
    def predict(self, data_np):
        self.model.eval()
        with torch.no_grad():
            scaled = self.scaler.transform(data_np.reshape(-1, 5))
            x = torch.FloatTensor(scaled[-self.seq_len:]).unsqueeze(0).to(self.device)
            pred_scaled = self.model(x)
            
            # Handle scalar output
            pred_return = pred_scaled.item() if pred_scaled.dim() == 0 else pred_scaled.squeeze().item()
            
            last_close = data_np[-1, 3]
            return last_close * (1 + pred_return)
        
    def save_model(self, path):
        """Save the model checkpoint for production use"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler  # Important for inference
        }, path)
        logger.info(f"Model checkpoint saved to {path}")

    def load_model(self, path):
        """Load a saved checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        logger.info(f"Model loaded from {path}")