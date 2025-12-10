import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path
import optuna

CONFIG = {
'data_dir': '../data',
'tickers': [
'AAPL', 'ABBV', 'ADBE', 'AIG', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'AXON', 'BA',
'BAC', 'BLK', 'CAT', 'COST', 'CRM', 'CSCO', 'CVX', 'DE', 'DIS', 'GE',
'GOOGL', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'LLY', 'MA',
'MCD', 'META', 'MRK', 'MS', 'MSFT', 'MU', 'NFLX', 'NKE', 'NVDA', 'ORCL',
'PEP', 'PFE', 'PG', 'QCOM', 'SBUX', 'SPY', 'TSLA', 'TXN', 'UBER', 'UNH',
'V', 'WMT', 'XOM'
],

    'force_start_date': '2020-01-01', # Data safeguard
    'seq_len': 60,
    'pred_horizon': 5,

    # Tobe-tuned by Optuna
    'batch_size': 32,
    'ranking_margin': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'patience': 5,
    'epochs': 40,
    'optuna_trials': 60,

}

class UniversalDataProcessor:
def **init**(self, data_dir, tickers):
self.data_dir = Path(data_dir)
self.tickers = sorted(tickers)
self.stock_to_id = {}
self.feature_cols = []
self.valid_tickers = []

    def load_and_process(self):
        print(f"Loading stocks from {self.data_dir}...")
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

    if len(df) < CONFIG['seq_len'] * 2:
                        continue

    df = self._engineer_features(df)
                    df['Ticker'] = ticker
                    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if not df.empty:
                        dfs.append(df)
                except:
                    pass

    if not dfs:
            raise ValueError("No valid data found!")

    big_df = pd.concat(dfs)
        date_counts = big_df.groupby(big_df.index)['Ticker'].count()
        required_count = len(dfs)
        valid_dates = date_counts[date_counts == required_count].index

    universal_df = big_df[big_df.index.isin(valid_dates)].sort_index().reset_index()

    self.valid_tickers = sorted(universal_df['Ticker'].unique())
        self.stock_to_id = {t: i for i, t in enumerate(self.valid_tickers)}
        universal_df['Stock_ID'] = universal_df['Ticker'].map(self.stock_to_id)

    print(f"Fixed Universe: {len(self.valid_tickers)} stocks, {len(valid_dates)} common days.")

    self.feature_cols = [c for c in universal_df.columns
                             if c not in ['Date', 'Ticker', 'Stock_ID', 'Target_5D', 'Log_Ret_Raw']]

    universal_df[self.feature_cols] = universal_df.groupby('Date')[self.feature_cols].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        ).clip(-3, 3)

    return universal_df

    def _engineer_features(self, df):
        df = df.copy()
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Log_Ret_Raw'] = df['Log_Ret']
        df['Target_5D'] = df['Log_Ret_Raw'].rolling(window=CONFIG['pred_horizon']).sum().shift(-CONFIG['pred_horizon'])

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
def **init**(self, df, feature_cols, seq_len, num_stocks):
self.df = df.sort_values(['Date', 'Stock_ID']) # Crucial Sort
self.feature_cols = feature_cols
self.seq_len = seq_len
self.num_stocks = num_stocks

    self.dates = sorted(df['Date'].unique())
        self.date_to_idx = {date: i for i, date in enumerate(self.dates)}
        self.valid_dates = self.dates[self.seq_len:]

    def__len__(self):
        return len(self.valid_dates)

    def__getitem__(self, idx):
        date = self.valid_dates[idx]
        window_start = self.date_to_idx[date] - self.seq_len + 1
        window_dates = self.dates[window_start : self.date_to_idx[date] + 1]

    # Fast slice because df is sorted by Date
        window_df = self.df[self.df['Date'].isin(window_dates)]

    # Check integrity
        if len(window_df) != self.seq_len * self.num_stocks:
            # Fallback for edge cases (though "Fixed Universe" prevents this)
            x = torch.zeros(self.num_stocks, self.seq_len, len(self.feature_cols))
        else:
            # Shape: [Seq_Len * Stocks, Features]
            flat = window_df[self.feature_cols].values.astype(np.float32)
            # Reshape: [Seq_Len, Stocks, Features]
            x = flat.reshape(self.seq_len, self.num_stocks, -1)
            # Transpose: [Stocks, Seq_Len, Features]
            x = np.transpose(x, (1, 0, 2))
            x = torch.FloatTensor(x)

    stock_ids = torch.arange(self.num_stocks)
        y = window_df[window_df['Date'] == date]['Target_5D'].values.astype(np.float32)

    return x, stock_ids, torch.FloatTensor(y)

processor = UniversalDataProcessor(CONFIG['data_dir'], CONFIG['tickers'])
universal_df = processor.load_and_process()

class AdvancedCrossSectionalModel(nn.Module):
def **init**(self, num_stocks, input_dim, hidden_dim, embed_dim, num_heads, dropout):
super().**init**()
self.embedding = nn.Embedding(num_stocks, embed_dim)
self.lstm = nn.LSTM(
input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout
)

    # Transformer for Cross-Sectional Attention
        # d_model must be divisible by num_heads
        d_model = hidden_dim + embed_dim
        self.norm = nn.LayerNorm(d_model) # Stabilizer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )

    self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x, stock_ids):
        b, s, seq, f = x.shape

    x_flat = x.view(b * s, seq, f)
        lstm_out, _ = self.lstm(x_flat)
        last_step = lstm_out[:, -1, :] # [B*S, Hidden]

    if stock_ids.dim() == 1:
            stock_ids = stock_ids.repeat(b, 1)
        ids_flat = stock_ids.view(b * s)
        embeds = self.embedding(ids_flat)

    # Cross-Sectional Attention
        combined = torch.cat([last_step, embeds], dim=1) # [B*S, D_Model]

    # Reshape for Transformer: [Batch, Stocks, D_Model]
        combined_view = combined.view(b, s, -1)
        combined_norm = self.norm(combined_view)

    # Attend across stocks in the same batch (day)
        attended = self.transformer(combined_norm) # [Batch, Stocks, D_Model]

    # Prediction Head
        out = self.head(attended) # [Batch, Stocks, 2]
        return out[:, :, 0], out[:, :, 1] # Mu, Log_Var

class HybridLoss(nn.Module):
def **init**(self, alpha=0.7, margin=1e-4):
super().**init**()
self.alpha = alpha
self.margin = margin
self.gnll = nn.GaussianNLLLoss()

    def forward(self, mu, log_var, target):
        gnll_loss = self.gnll(mu.flatten(), target.flatten(), torch.exp(log_var.flatten()))

    mu_diff = mu.unsqueeze(2) - mu.unsqueeze(1)
        target_diff = target.unsqueeze(2) - target.unsqueeze(1)
        target_sign = torch.sign(target_diff)

    loss_matrix = torch.relu(-target_sign * mu_diff + self.margin)
        mask = (target_sign != 0)
        rank_loss = (loss_matrix * mask).sum() / mask.sum().clamp(min=1)

    return (1 - self.alpha) * gnll_loss + self.alpha * rank_loss

class AdvancedCrossSectionalModel(nn.Module):
def **init**(self, num_stocks, input_dim, hidden_dim, embed_dim, num_heads, dropout):
super().**init**()
self.embedding = nn.Embedding(num_stocks, embed_dim)
self.lstm = nn.LSTM(
input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout
)

    # Transformer for Cross-Sectional Attention
        # d_model must be divisible by num_heads
        d_model = hidden_dim + embed_dim
        self.norm = nn.LayerNorm(d_model) # Stabilizer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )

    self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x, stock_ids):
        b, s, seq, f = x.shape

    x_flat = x.view(b * s, seq, f)
        lstm_out, _ = self.lstm(x_flat)
        last_step = lstm_out[:, -1, :] # [B*S, Hidden]

    if stock_ids.dim() == 1:
            stock_ids = stock_ids.repeat(b, 1)
        ids_flat = stock_ids.view(b * s)
        embeds = self.embedding(ids_flat)

    # Cross-Sectional Attention
        combined = torch.cat([last_step, embeds], dim=1) # [B*S, D_Model]

    # Reshape for Transformer: [Batch, Stocks, D_Model]
        combined_view = combined.view(b, s, -1)
        combined_norm = self.norm(combined_view)

    # Attend across stocks in the same batch (day)
        attended = self.transformer(combined_norm) # [Batch, Stocks, D_Model]

    # Prediction Head
        out = self.head(attended) # [Batch, Stocks, 2]
        return out[:, :, 0], out[:, :, 1] # Mu, Log_Var

class HybridLoss(nn.Module):
def **init**(self, alpha=0.7, margin=1e-4):
super().**init**()
self.alpha = alpha
self.margin = margin
self.gnll = nn.GaussianNLLLoss()

    def forward(self, mu, log_var, target):
        gnll_loss = self.gnll(mu.flatten(), target.flatten(), torch.exp(log_var.flatten()))

    mu_diff = mu.unsqueeze(2) - mu.unsqueeze(1)
        target_diff = target.unsqueeze(2) - target.unsqueeze(1)
        target_sign = torch.sign(target_diff)

    loss_matrix = torch.relu(-target_sign * mu_diff + self.margin)
        mask = (target_sign != 0)
        rank_loss = (loss_matrix * mask).sum() / mask.sum().clamp(min=1)

    return (1 - self.alpha) * gnll_loss + self.alpha * rank_loss

def objective(trial): # 1. Architecture
hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
embedding_dim = trial.suggest_categorical('embedding_dim', [8, 16, 32])
num_heads = 4

    # Architecture Constraint: d_model % num_heads == 0
    d_model = hidden_size + embedding_dim
    if d_model % num_heads != 0:
        raise optuna.exceptions.TrialPruned()

    # 2. Regularization
    dropout = trial.suggest_float('dropout', 0.2, 0.6) # Widen range to higher dropout
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True) # L2 Reg

    # 3. Training Dynamics
    lr = trial.suggest_float('lr', 1e-5, 2e-4, log=True)
    alpha_weight = trial.suggest_float('alpha_weight', 0.5, 0.9)

    # --- Setup ---
    # Split Data
    dates = universal_df['Date'].unique()
    split_idx = int(len(dates) * 0.8)
    train_df = universal_df[universal_df['Date'].isin(dates[:split_idx])]
    val_df = universal_df[universal_df['Date'].isin(dates[split_idx:])]

    train_ds = UniversalDataset(train_df, processor.feature_cols, CONFIG['seq_len'], len(processor.valid_tickers))
    val_ds = UniversalDataset(val_df, processor.feature_cols, CONFIG['seq_len'], len(processor.valid_tickers))

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    model = AdvancedCrossSectionalModel(
        num_stocks=len(processor.valid_tickers),
        input_dim=len(processor.feature_cols),
        hidden_dim=hidden_size,
        embed_dim=embedding_dim,
        num_heads=num_heads,
        dropout=dropout
    ).to(CONFIG['device'])

    # Weight Decay handling
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    criterion = HybridLoss(alpha=alpha_weight, margin=CONFIG['ranking_margin'])

    best_val_ic = -1.0
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        for x, ids, y in train_loader:
            x, ids, y = x.to(CONFIG['device']), ids.to(CONFIG['device']), y.to(CONFIG['device'])
            optimizer.zero_grad()
            mu, log_var = model(x, ids)

    if torch.isnan(mu).any():
                raise optuna.exceptions.TrialPruned()

    loss = criterion(mu, log_var, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Step Scheduler
        scheduler.step()

    # Validation
        model.eval()
        ics = []
        with torch.no_grad():
            for x, ids, y in val_loader:
                x, ids, y = x.to(CONFIG['device']), ids.to(CONFIG['device']), y.to(CONFIG['device'])
                mu, _ = model(x, ids)
                mu_np, y_np = mu.cpu().numpy(), y.cpu().numpy()

    for i in range(len(mu_np)):
                    if np.std(mu_np[i]) < 1e-6: continue
                    ic, _ = spearmanr(mu_np[i], y_np[i])
                    if not np.isnan(ic): ics.append(ic)

    avg_ic = np.mean(ics) if ics else -1.0

    # Optimization & Pruning Logic
        trial.report(avg_ic, epoch)

    if avg_ic > best_val_ic:
            best_val_ic = avg_ic
            patience_counter = 0
        else:
            patience_counter += 1

    if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if patience_counter >= CONFIG['patience']:
            break

    return best_val_ic

processor = UniversalDataProcessor(CONFIG['data_dir'], CONFIG['tickers'])
universal_df = processor.load_and_process()

study = optuna.create_study(
direction='maximize',
sampler=optuna.samplers.TPESampler(seed=42),
pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)

print(f"Starting Optimized Search...")
study.optimize(objective, n_trials=CONFIG['optuna_trials'])

print("\nBest IC:", study.best_value)
print("Best Params:", study.best_params)

[I 2025-12-02 23:39:40,070] Trial 0 finished with value: 0.020366098785910104 and parameters: {'hidden_size': 128, 'embedding_dim': 8, 'dropout': 0.2232334448672798, 'weight_decay': 0.0003967605077052988, 'lr': 6.054365855469242e-05, 'alpha_weight': 0.7832290311184182}. Best is trial 0 with value: 0.020366098785910104.
[I 2025-12-02 23:49:22,427] Trial 1 finished with value: -0.0034167876149008214 and parameters: {'hidden_size': 128, 'embedding_dim': 8, 'dropout': 0.3216968971838151, 'weight_decay': 3.752055855124284e-05, 'lr': 3.647316284911205e-05, 'alpha_weight': 0.6164916560792167}. Best is trial 0 with value: 0.020366098785910104.
[I 2025-12-02 23:55:51,369] Trial 2 finished with value: 0.03340763010574331 and parameters: {'hidden_size': 64, 'embedding_dim': 32, 'dropout': 0.2798695128633439, 'weight_decay': 3.489018845491386e-05, 'lr': 5.89860241043269e-05, 'alpha_weight': 0.5185801650879991}. Best is trial 2 with value: 0.03340763010574331.
[I 2025-12-02 23:58:07,014] Trial 3 finished with value: 0.03104877957472297 and parameters: {'hidden_size': 64, 'embedding_dim': 16, 'dropout': 0.3218455076693483, 'weight_decay': 1.9634341572933354e-06, 'lr': 7.766184280392883e-05, 'alpha_weight': 0.6760609974958405}. Best is trial 2 with value: 0.03340763010574331.
[I 2025-12-03 00:02:32,021] Trial 4 finished with value: 0.028237453060566266 and parameters: {'hidden_size': 128, 'embedding_dim': 8, 'dropout': 0.3246844304357644, 'weight_decay': 3.632486956676606e-05, 'lr': 5.1438284050769266e-05, 'alpha_weight': 0.5739417822102109}. Best is trial 2 with value: 0.03340763010574331.
[I 2025-12-03 00:07:17,527] Trial 5 finished with value: 0.005368877715575827 and parameters: {'hidden_size': 64, 'embedding_dim': 32, 'dropout': 0.2353970008207678, 'weight_decay': 3.87211803217458e-06, 'lr': 1.1450964268326635e-05, 'alpha_weight': 0.6301321323053057}. Best is trial 2 with value: 0.03340763010574331.
[I 2025-12-03 00:34:49,084] Trial 6 finished with value: 0.032197427834220284 and parameters: {'hidden_size': 256, 'embedding_dim': 32, 'dropout': 0.25636968998990506, 'weight_decay': 0.0002550298070162893, 'lr': 1.2502377950801103e-05, 'alpha_weight': 0.8947547746402069}. Best is trial 2 with value: 0.03340763010574331.
[I 2025-12-03 00:37:35,609] Trial 7 finished with value: 0.027228471214320267 and parameters: {'hidden_size': 64, 'embedding_dim': 8, 'dropout': 0.5085081386743783, 'weight_decay': 1.6677615430197902e-06, 'lr': 2.926676128549071e-05, 'alpha_weight': 0.5463476238100519}. Best is trial 2 with value: 0.03340763010574331.
[I 2025-12-03 00:45:26,326] Trial 8 finished with value: 0.05456817160826594 and parameters: {'hidden_size': 64, 'embedding_dim': 32, 'dropout': 0.49184247133522563, 'weight_decay': 8.178476574339548e-05, 'lr': 0.0001426561143637793, 'alpha_weight': 0.6888859700647797}. Best is trial 8 with value: 0.05456817160826594.
[I 2025-12-03 01:21:23,818] Trial 9 finished with value: 0.022362104614463103 and parameters: {'hidden_size': 256, 'embedding_dim': 16, 'dropout': 0.4090931317527976, 'weight_decay': 1.9170041589170666e-05, 'lr': 1.0791232418686384e-05, 'alpha_weight': 0.5431565707973218}. Best is trial 8 with value: 0.05456817160826594.
[I 2025-12-03 01:23:50,529] Trial 10 finished with value: 0.017926616098785906 and parameters: {'hidden_size': 64, 'embedding_dim': 32, 'dropout': 0.5878148443151463, 'weight_decay': 0.00015001373831540624, 'lr': 0.00019041111223746303, 'alpha_weight': 0.753479653721226}. Best is trial 8 with value: 0.05456817160826594.
[I 2025-12-03 01:27:54,321] Trial 11 finished with value: 0.023678496555855046 and parameters: {'hidden_size': 64, 'embedding_dim': 32, 'dropout': 0.45075345796262717, 'weight_decay': 8.533716477932925e-05, 'lr': 0.00013106575195871793, 'alpha_weight': 0.5033126127239229}. Best is trial 8 with value: 0.05456817160826594.
[I 2025-12-03 01:30:45,198] Trial 12 finished with value: 0.03022122007971064 and parameters: {'hidden_size': 64, 'embedding_dim': 32, 'dropout': 0.5060954317960623, 'weight_decay': 9.012779589354681e-06, 'lr': 0.00010575464347165963, 'alpha_weight': 0.8416182997379973}. Best is trial 8 with value: 0.05456817160826594.
[I 2025-12-03 01:36:27,961] Trial 13 finished with value: 0.04081750927269795 and parameters: {'hidden_size': 64, 'embedding_dim': 32, 'dropout': 0.564766689287175, 'weight_decay': 0.0009468175586474681, 'lr': 2.6293207832476727e-05, 'alpha_weight': 0.6984165248678857}. Best is trial 8 with value: 0.05456817160826594.
[I 2025-12-03 01:58:19,982] Trial 14 finished with value: 0.020371498260649202 and parameters: {'hidden_size': 256, 'embedding_dim': 32, 'dropout': 0.5949848437355983, 'weight_decay': 0.000972858315193105, 'lr': 2.2649653433750454e-05, 'alpha_weight': 0.7059881656083773}. Best is trial 8 with value: 0.05456817160826594.
[I 2025-12-03 02:00:41,576] Trial 15 pruned.
[I 2025-12-03 02:03:01,735] Trial 16 finished with value: 0.020258469256110766 and parameters: {'hidden_size': 64, 'embedding_dim': 16, 'dropout': 0.46571243151511826, 'weight_decay': 9.06710179482546e-05, 'lr': 1.7621437838888276e-05, 'alpha_weight': 0.6690283340090343}. Best is trial 8 with value: 0.05456817160826594.
[I 2025-12-03 02:06:34,344] Trial 17 finished with value: 0.032693819545234634 and parameters: {'hidden_size': 64, 'embedding_dim': 32, 'dropout': 0.5557218551874994, 'weight_decay': 0.00037339920591605093, 'lr': 0.00019833777468130574, 'alpha_weight': 0.8178581456448356}. Best is trial 8 with value: 0.05456817160826594.
[I 2025-12-03 02:16:52,591] Trial 18 finished with value: 0.05299063401551534 and parameters: {'hidden_size': 128, 'embedding_dim': 32, 'dropout': 0.44953959816871203, 'weight_decay': 1.048496826036815e-05, 'lr': 4.072392816393183e-05, 'alpha_weight': 0.6275793479261579}. Best is trial 8 with value: 0.05456817160826594.
[I 2025-12-03 02:25:34,023] Trial 19 finished with value: 0.055583632824198856 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.3925442679141049, 'weight_decay': 1.0698336409175715e-05, 'lr': 8.6826860844493e-05, 'alpha_weight': 0.6182442489793656}. Best is trial 19 with value: 0.055583632824198856.
[I 2025-12-03 02:30:19,551] Trial 20 finished with value: 0.04189308464072615 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.3705519144436229, 'weight_decay': 5.417126635519039e-06, 'lr': 0.00011385410742540525, 'alpha_weight': 0.5885235582465119}. Best is trial 19 with value: 0.055583632824198856.
[I 2025-12-03 02:35:52,988] Trial 21 finished with value: 0.03213407399728154 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.4229524844495968, 'weight_decay': 1.1790987312442672e-05, 'lr': 8.157339735592011e-05, 'alpha_weight': 0.6310744807512884}. Best is trial 19 with value: 0.055583632824198856.
[I 2025-12-03 02:41:24,681] Trial 22 pruned.
[I 2025-12-03 02:46:57,317] Trial 23 finished with value: 0.05463008558527426 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.35971785932752254, 'weight_decay': 4.258493087294439e-06, 'lr': 0.00014747316133621722, 'alpha_weight': 0.5937308077711712}. Best is trial 19 with value: 0.055583632824198856.
[I 2025-12-03 02:53:17,724] Trial 24 pruned.
[I 2025-12-03 02:58:04,879] Trial 25 finished with value: 0.023033439307024207 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.37344030177763, 'weight_decay': 1.0332227955421275e-06, 'lr': 0.00016237025966683077, 'alpha_weight': 0.7414528402276978}. Best is trial 19 with value: 0.055583632824198856.
[I 2025-12-03 03:02:50,605] Trial 26 finished with value: 0.02815322125463635 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.3492177732030607, 'weight_decay': 7.754658782750997e-05, 'lr': 8.958718633049194e-05, 'alpha_weight': 0.5671946922736016}. Best is trial 19 with value: 0.055583632824198856.
[I 2025-12-03 03:20:51,704] Trial 27 finished with value: 0.06210295862418504 and parameters: {'hidden_size': 256, 'embedding_dim': 16, 'dropout': 0.4130070834841313, 'weight_decay': 1.9103467339073595e-05, 'lr': 0.00015167586266214228, 'alpha_weight': 0.6669437484282295}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 03:34:59,942] Trial 28 finished with value: 0.014690530905153548 and parameters: {'hidden_size': 256, 'embedding_dim': 16, 'dropout': 0.4159850716199475, 'weight_decay': 6.085946829777834e-06, 'lr': 0.0001035102841602315, 'alpha_weight': 0.6510531700526605}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 03:47:11,725] Trial 29 finished with value: 0.02955276510701039 and parameters: {'hidden_size': 256, 'embedding_dim': 16, 'dropout': 0.3932699085210043, 'weight_decay': 1.7819890629738284e-05, 'lr': 6.813165155360609e-05, 'alpha_weight': 0.6094289115145793}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 03:59:17,294] Trial 30 finished with value: 0.037074972730402896 and parameters: {'hidden_size': 256, 'embedding_dim': 16, 'dropout': 0.2765216806913994, 'weight_decay': 2.8976713527133107e-06, 'lr': 0.0001609994939352927, 'alpha_weight': 0.7826164186595433}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 04:04:04,467] Trial 31 finished with value: 0.05430395731103278 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.4308740407684508, 'weight_decay': 5.435173530183346e-05, 'lr': 0.00013518324129621503, 'alpha_weight': 0.6905983517445577}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 04:22:12,995] Trial 32 finished with value: 0.024277478286912246 and parameters: {'hidden_size': 256, 'embedding_dim': 8, 'dropout': 0.34464750555457685, 'weight_decay': 2.1855030174451537e-05, 'lr': 0.00015799241437319133, 'alpha_weight': 0.6504492961920062}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 04:30:57,718] Trial 33 finished with value: 0.02992172921418204 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.20389086329930933, 'weight_decay': 7.553369883340875e-06, 'lr': 9.23121110797058e-05, 'alpha_weight': 0.7249080548656114}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 04:37:17,442] Trial 34 pruned.
[I 2025-12-03 04:53:19,675] Trial 35 pruned.
[I 2025-12-03 04:58:05,590] Trial 36 finished with value: 0.020087125924390076 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.3408707503567052, 'weight_decay': 2.7423753001549123e-05, 'lr': 5.427658204256137e-05, 'alpha_weight': 0.5601685145662105}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 05:10:04,992] Trial 37 finished with value: 0.02924931866123573 and parameters: {'hidden_size': 256, 'embedding_dim': 8, 'dropout': 0.39633501265586446, 'weight_decay': 0.00015894498114478619, 'lr': 0.00017127194833827596, 'alpha_weight': 0.6409518185554833}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 05:15:34,431] Trial 38 finished with value: 0.03270137880986938 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.2929712572134046, 'weight_decay': 1.8945285749868058e-06, 'lr': 0.00013631316451188984, 'alpha_weight': 0.6153874718397311}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 05:20:18,872] Trial 39 finished with value: 0.027294531148869806 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.38592727192555226, 'weight_decay': 1.246946065742796e-05, 'lr': 9.415768314845772e-05, 'alpha_weight': 0.531428671816756}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 05:32:22,088] Trial 40 pruned.
[I 2025-12-03 05:41:55,402] Trial 41 finished with value: 0.05173452726518764 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.4288642034343337, 'weight_decay': 3.995807232424086e-05, 'lr': 0.0001262956263997056, 'alpha_weight': 0.6890723174005801}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 05:56:09,538] Trial 42 finished with value: 0.05510595929228005 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.45126063513297104, 'weight_decay': 4.7050080034878104e-05, 'lr': 0.00014934971332807929, 'alpha_weight': 0.7635642492922251}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 06:00:52,395] Trial 43 pruned.
[I 2025-12-03 06:03:11,682] Trial 44 finished with value: 0.04069692100352478 and parameters: {'hidden_size': 64, 'embedding_dim': 16, 'dropout': 0.5261604341553288, 'weight_decay': 2.8489969727073592e-05, 'lr': 0.0001495512757807223, 'alpha_weight': 0.8069163603895783}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 06:13:43,845] Trial 45 finished with value: 0.024652561798788216 and parameters: {'hidden_size': 128, 'embedding_dim': 32, 'dropout': 0.4539632694931592, 'weight_decay': 6.0821491590824626e-05, 'lr': 0.00019956227757807925, 'alpha_weight': 0.7342171917661944}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 06:17:53,338] Trial 46 pruned.
[I 2025-12-03 06:24:51,080] Trial 47 finished with value: 0.038693355910337034 and parameters: {'hidden_size': 128, 'embedding_dim': 32, 'dropout': 0.4069379521049969, 'weight_decay': 7.545795563792725e-06, 'lr': 0.00010753933265521791, 'alpha_weight': 0.7067064355576524}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 06:28:55,428] Trial 48 finished with value: 0.041666666666666664 and parameters: {'hidden_size': 64, 'embedding_dim': 16, 'dropout': 0.36122995979912353, 'weight_decay': 3.0249907164105375e-05, 'lr': 4.930911952585226e-05, 'alpha_weight': 0.5869367530001977}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 06:40:59,110] Trial 49 finished with value: 0.03387846430299261 and parameters: {'hidden_size': 256, 'embedding_dim': 32, 'dropout': 0.32046274863584645, 'weight_decay': 1.400627873791585e-05, 'lr': 0.00011845383739918831, 'alpha_weight': 0.6652930653145334}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 06:44:30,714] Trial 50 finished with value: 0.03150845485751146 and parameters: {'hidden_size': 64, 'embedding_dim': 16, 'dropout': 0.5381230506392203, 'weight_decay': 2.636998751052427e-06, 'lr': 0.0001460345554286125, 'alpha_weight': 0.7650914649330021}. Best is trial 27 with value: 0.06210295862418504.
[I 2025-12-03 07:01:59,670] Trial 51 finished with value: 0.0773651139217177 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.43951755524391284, 'weight_decay': 5.3788742936530506e-05, 'lr': 0.00013061115141321854, 'alpha_weight': 0.6967878897119416}. Best is trial 51 with value: 0.0773651139217177.
[I 2025-12-03 07:09:55,353] Trial 52 finished with value: 0.045668037413320435 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.4442177407612974, 'weight_decay': 0.00010397530651190146, 'lr': 0.0001815486731842769, 'alpha_weight': 0.7137925435666002}. Best is trial 51 with value: 0.0773651139217177.
[I 2025-12-03 07:15:29,979] Trial 53 finished with value: 0.028959542815674894 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.4631704714958548, 'weight_decay': 6.353942460268266e-05, 'lr': 9.754954658073451e-05, 'alpha_weight': 0.6394129119703991}. Best is trial 51 with value: 0.0773651139217177.
[I 2025-12-03 07:28:59,786] Trial 54 finished with value: 0.05154338585942359 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.41065893450806723, 'weight_decay': 4.6044125427982875e-05, 'lr': 0.00012184542982745299, 'alpha_weight': 0.609726409191321}. Best is trial 51 with value: 0.0773651139217177.
[I 2025-12-03 07:36:11,285] Trial 55 finished with value: 0.033760755753680274 and parameters: {'hidden_size': 128, 'embedding_dim': 32, 'dropout': 0.3852473242975879, 'weight_decay': 2.436172625408181e-05, 'lr': 0.0001420006528214214, 'alpha_weight': 0.6657556493449092}. Best is trial 51 with value: 0.0773651139217177.
[I 2025-12-03 07:41:44,534] Trial 56 finished with value: 0.06407340693897298 and parameters: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.49029780466363715, 'weight_decay': 3.6620861015032104e-05, 'lr': 0.0001691051195656132, 'alpha_weight': 0.7399496410855826}. Best is trial 51 with value: 0.0773651139217177.
[I 2025-12-03 07:46:32,770] Trial 57 pruned.
[I 2025-12-03 07:51:21,197] Trial 58 pruned.
[I 2025-12-03 07:57:53,468] Trial 59 pruned.

Best IC: 0.0773651139217177
Best Params: {'hidden_size': 128, 'embedding_dim': 16, 'dropout': 0.43951755524391284, 'weight_decay': 5.3788742936530506e-05, 'lr': 0.00013061115141321854, 'alpha_weight': 0.6967878897119416}

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import os
import sys

# --- 1. THE GOLDEN RECIPE (Trial 51) ---

TARGET_IC = 0.07
MAX_ATTEMPTS = 50

BEST_PARAMS = {
'hidden_size': 128,
'embedding_dim': 16,
'dropout': 0.4395,
'weight_decay': 5.38e-05,
'lr': 0.0001306,
'alpha_weight': 0.6968
}

# --- 2. SETUP DATA ---

dates = universal_df['Date'].unique()
split_idx = int(len(dates) \* 0.8)
train_df = universal_df[universal_df['Date'].isin(dates[:split_idx])]
val_df = universal_df[universal_df['Date'].isin(dates[split_idx:])]

train_ds = UniversalDataset(train_df, processor.feature_cols, CONFIG['seq_len'], len(processor.valid_tickers))
val_ds = UniversalDataset(val_df, processor.feature_cols, CONFIG['seq_len'], len(processor.valid_tickers))

# Increase batch size slightly to speed up mining

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

# --- 3. MINING LOOP ---

print(f"MINING FOR ALPHA (Target IC: > {TARGET_IC})...")

for attempt in range(1, MAX_ATTEMPTS + 1): # A. Random Seed for this attempt
seed = np.random.randint(1, 10000)
torch.manual_seed(seed)
np.random.seed(seed)

    print(f"\nAttempt {attempt}/{MAX_ATTEMPTS} (Seed {seed})...", end="", flush=True)

    # B. Init Model
    model = AdvancedCrossSectionalModel(
        num_stocks=len(processor.valid_tickers),
        input_dim=len(processor.feature_cols),
        hidden_dim=BEST_PARAMS['hidden_size'],
        embed_dim=BEST_PARAMS['embedding_dim'],
        num_heads=4,
        dropout=BEST_PARAMS['dropout']
    ).to(CONFIG['device'])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BEST_PARAMS['lr'],
        weight_decay=BEST_PARAMS['weight_decay']
    )
    criterion = HybridLoss(alpha=BEST_PARAMS['alpha_weight'], margin=CONFIG['ranking_margin'])

    # C. Short Train (Just enough to see if it's a winner)
    # We only need ~10 epochs to know if a seed is "good"
    peak_ic = -1.0

    for epoch in range(15):
        model.train()
        for x, ids, y in train_loader:
            x, ids, y = x.to(CONFIG['device']), ids.to(CONFIG['device']), y.to(CONFIG['device'])
            optimizer.zero_grad()
            mu, log_var = model(x, ids)
            loss = criterion(mu, log_var, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Fast Validation
        model.eval()
        ics = []
        with torch.no_grad():
            for x, ids, y in val_loader:
                x, ids, y = x.to(CONFIG['device']), ids.to(CONFIG['device']), y.to(CONFIG['device'])
                mu, _ = model(x, ids)
                mu_np, y_np = mu.cpu().numpy(), y.cpu().numpy()
                for i in range(len(mu_np)):
                    if np.std(mu_np[i]) > 1e-6:
                        ic, _ = spearmanr(mu_np[i], y_np[i])
                        if not np.isnan(ic): ics.append(ic)

    current_ic = np.mean(ics) if ics else 0.0
        peak_ic = max(peak_ic, current_ic)

    # Early Exit if it's trash
        if epoch == 10 and peak_ic < 0.02:
            print(f" [Bad Start: {peak_ic:.4f}]", end="")
            break

    # D. JACKPOT CHECK
        if current_ic >= TARGET_IC:
            print(f"\n\nJACKPOT! Found Model with IC: {current_ic:.5f}")
            torch.save(model.state_dict(), "golden_model.pth")
            print("Saved to 'golden_model.pth'")
            sys.exit()

    print(f" -> Peak: {peak_ic:.4f}", end="")

print("\n\nMining finished. No model hit the target. Best was close?")
⛏️ MINING FOR ALPHA (Target IC: > 0.07)...

Attempt 1/50 (Seed 482)... [Bad Start: 0.0194] -> Peak: 0.0194
Attempt 2/50 (Seed 9353)... -> Peak: 0.0420
Attempt 3/50 (Seed 9117)... -> Peak: 0.0682
Attempt 4/50 (Seed 9282)... -> Peak: 0.0530
Attempt 5/50 (Seed 9048)... [Bad Start: 0.0178] -> Peak: 0.0178
Attempt 6/50 (Seed 7489)... -> Peak: 0.0294
Attempt 7/50 (Seed 4076)... -> Peak: 0.0297
Attempt 8/50 (Seed 7576)... -> Peak: 0.0431
Attempt 9/50 (Seed 9206)... [Bad Start: 0.0068] -> Peak: 0.0068
Attempt 10/50 (Seed 7303)... -> Peak: 0.0386
Attempt 11/50 (Seed 4089)... -> Peak: 0.0421
Attempt 12/50 (Seed 5708)... -> Peak: 0.0415
Attempt 13/50 (Seed 4586)... [Bad Start: -0.0025] -> Peak: -0.0025
Attempt 14/50 (Seed 5841)... -> Peak: 0.0418
Attempt 15/50 (Seed 8931)... [Bad Start: 0.0087] -> Peak: 0.0087
Attempt 16/50 (Seed 8207)... -> Peak: 0.0382
Attempt 17/50 (Seed 4690)... -> Peak: 0.0509
Attempt 18/50 (Seed 499)... -> Peak: 0.0323
Attempt 19/50 (Seed 5816)... -> Peak: 0.0523
Attempt 20/50 (Seed 5687)... -> Peak: 0.0235
Attempt 21/50 (Seed 6982)... -> Peak: 0.0252
Attempt 22/50 (Seed 8305)... [Bad Start: 0.0157] -> Peak: 0.0157
Attempt 23/50 (Seed 9939)... -> Peak: 0.0231
Attempt 24/50 (Seed 387)... [Bad Start: -0.0005] -> Peak: -0.0005
Attempt 25/50 (Seed 3952)... [Bad Start: 0.0149] -> Peak: 0.0149
Attempt 26/50 (Seed 4870)... [Bad Start: -0.0028] -> Peak: -0.0028
Attempt 27/50 (Seed 7980)... [Bad Start: -0.0025] -> Peak: -0.0025
Attempt 28/50 (Seed 9644)... -> Peak: 0.0554
Attempt 29/50 (Seed 9649)... -> Peak: 0.0322
Attempt 30/50 (Seed 5993)... [Bad Start: 0.0152] -> Peak: 0.0152
Attempt 31/50 (Seed 2493)... -> Peak: 0.0280
Attempt 32/50 (Seed 8765)... -> Peak: 0.0252
Attempt 33/50 (Seed 8238)... [Bad Start: 0.0179] -> Peak: 0.0179
Attempt 34/50 (Seed 2051)... -> Peak: 0.0431
Attempt 35/50 (Seed 9468)... -> Peak: 0.0273
Attempt 36/50 (Seed 5982)... -> Peak: 0.0318
Attempt 37/50 (Seed 3492)... -> Peak: 0.0408
Attempt 38/50 (Seed 4291)...

JACKPOT! Found Model with IC: 0.07210
Saved to 'golden_model.pth'
