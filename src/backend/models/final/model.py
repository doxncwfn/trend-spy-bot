import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedCrossSectionalModel(nn.Module):
    def __init__(self, num_stocks, input_dim, hidden_dim, embed_dim, num_heads, dropout):
        super().__init__()
        
        # 1. Stock Identity Embedding
        self.embedding = nn.Embedding(num_stocks, embed_dim)
        
        # 2. Temporal Encoder (LSTM)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout
        )
        
        # 3. Cross-Sectional Attention (Transformer)
        d_model = hidden_dim + embed_dim
        self.norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        
        # 4. Probabilistic Head (Mean & LogVar)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2) 
        )
        
    def forward(self, x, stock_ids):
        # Input: [Batch, Stocks, Seq_Len, Features]
        b, s, seq, f = x.shape
        
        # Flatten batch and stocks for LSTM: [B*S, Seq, F]
        x_flat = x.view(b * s, seq, f)
        lstm_out, _ = self.lstm(x_flat)
        last_step = lstm_out[:, -1, :] # [B*S, Hidden]
        
        # Embeddings
        if stock_ids.dim() == 1:
            stock_ids = stock_ids.repeat(b, 1)
        ids_flat = stock_ids.view(b * s)
        embeds = self.embedding(ids_flat)
        
        # Combine Features + Identity
        combined = torch.cat([last_step, embeds], dim=1).view(b, s, -1)
        
        # Cross-Sectional Attention
        # Normalize & Attend across stocks within the same day
        attended = self.transformer(self.norm(combined))
        
        # Output
        out = self.head(attended).view(b, s, 2)
        mu = out[:, :, 0]
        log_var = out[:, :, 1]
        
        # Return Mean and Sigma (exp(0.5*log_var))
        return mu, torch.exp(0.5 * log_var)

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7, margin=0.1):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.gnll = nn.GaussianNLLLoss()
        
    def forward(self, mu, log_var, target):
        # 1. Regression Loss (Gaussian NLL)
        # Flatten for element-wise loss
        gnll_loss = self.gnll(mu.flatten(), target.flatten(), torch.exp(log_var.flatten()))
        
        # 2. Ranking Loss (Pairwise Margin)
        # Expand dimensions for pairwise comparison [Batch, Stocks, Stocks]
        mu_diff = mu.unsqueeze(2) - mu.unsqueeze(1)
        target_diff = target.unsqueeze(2) - target.unsqueeze(1)
        target_sign = torch.sign(target_diff)
        
        # Hinge Loss
        loss_matrix = torch.relu(-target_sign * mu_diff + self.margin)
        
        # Mask out ties
        mask = (target_sign != 0)
        rank_loss = (loss_matrix * mask).sum() / mask.sum().clamp(min=1)
        
        return (1 - self.alpha) * gnll_loss + self.alpha * rank_loss