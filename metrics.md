import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, norm
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- CONFIGURATION ---

CONFIG = {
'data_dir': '../data',
'model_path': 'final_trendspy_model.pth',
'tickers': [
'AAPL', 'ABBV', 'ADBE', 'AIG', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'AXON', 'BA',
'BAC', 'BLK', 'CAT', 'COST', 'CRM', 'CSCO', 'CVX', 'DE', 'DIS', 'GE',
'GOOGL', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'LLY', 'MA',
'MCD', 'META', 'MRK', 'MS', 'MSFT', 'MU', 'NFLX', 'NKE', 'NVDA', 'ORCL',
'PEP', 'PFE', 'PG', 'QCOM', 'SBUX', 'SPY', 'TSLA', 'TXN', 'UBER', 'UNH',
'V', 'WMT', 'XOM'
],
'force_start_date': '2020-01-01',
'seq_len': 60,
'pred_horizon': 5,
'hidden_size': 128,
'embedding_dim': 16,
'num_heads': 4,
'dropout': 0.0,
'batch_size': 32,
'device': 'cuda' if torch.cuda.is_available() else 'cpu',
'top_k': 5,
'transaction_cost_bps': 10,
'confidence_threshold': 0.52
}

# --- 1. DATA PIPELINE (Fixed Universe) ---

class UniversalDataProcessor:
def **init**(self, data_dir, tickers):
self.data_dir = Path(data_dir)
self.tickers = sorted(tickers)
self.stock_to_id = {}
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
                except: pass

    big_df = pd.concat(dfs)
        date_counts = big_df.groupby(big_df.index)['Ticker'].count()
        valid_dates = date_counts[date_counts == len(dfs)].index

    universal_df = big_df[big_df.index.isin(valid_dates)].sort_index().reset_index()

    self.valid_tickers = sorted(universal_df['Ticker'].unique())
        self.stock_to_id = {t: i for i, t in enumerate(self.valid_tickers)}
        universal_df['Stock_ID'] = universal_df['Ticker'].map(self.stock_to_id)

    self.feature_cols = [c for c in universal_df.columns
                             if c not in ['Date', 'Ticker', 'Stock_ID', 'Target_5D', 'Log_Ret_Raw']]

    # Z-Score
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
self.seq_len = seq_len
self.num_stocks = num_stocks
df = df.sort_values(['Date', 'Ticker'])
unique_dates = df['Date'].unique()
self.num_days = len(unique_dates)

    self.X = np.zeros((self.num_days, num_stocks, len(feature_cols)), dtype=np.float32)
        self.Y = np.zeros((self.num_days, num_stocks), dtype=np.float32)
        self.IDs = np.arange(num_stocks)

    for i, feat in enumerate(feature_cols):
            self.X[:, :, i] = df.pivot(index='Date', columns='Ticker', values=feat).values
        self.Y = df.pivot(index='Date', columns='Ticker', values='Target_5D').values

    def__len__(self): return self.num_days - self.seq_len
    def__getitem__(self, idx):
        x = self.X[idx : idx + self.seq_len].transpose(1, 0, 2)
        return torch.FloatTensor(x), torch.LongTensor(self.IDs), torch.FloatTensor(self.Y[idx + self.seq_len - 1])

# --- 2. MODEL ---

class AdvancedCrossSectionalModel(nn.Module):
def **init**(self, num_stocks, input_dim, hidden_dim, embed_dim, num_heads, dropout):
super().**init**()
self.embedding = nn.Embedding(num_stocks, embed_dim)
self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
d_model = hidden_dim + embed_dim
self.norm = nn.LayerNorm(d_model)
self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x, stock_ids):
        b, s, seq, f = x.shape
        x_flat = x.view(b * s, seq, f)
        lstm_out, _ = self.lstm(x_flat)
        last_step = lstm_out[:, -1, :]
        if stock_ids.dim() == 1: stock_ids = stock_ids.repeat(b, 1)
        ids_flat = stock_ids.view(b * s)
        embeds = self.embedding(ids_flat)
        combined = torch.cat([last_step, embeds], dim=1).view(b, s, -1)
        attended = self.transformer(self.norm(combined))
        out = self.head(attended).view(b, s, 2)
        return out[:, :, 0], out[:, :, 1]

# 1. Load Data & Model

processor = UniversalDataProcessor(CONFIG['data_dir'], CONFIG['tickers'])
universal_df = processor.load_and_process()

# Split: Use last 20% as test set (Standard)

dates = universal_df['Date'].unique()
split_idx = int(len(dates) \* 0.8)
test_df = universal_df[universal_df['Date'].isin(dates[split_idx:])]

print(f"Evaluation Period: {test_df['Date'].min()} to {test_df['Date'].max()}")
print(f"Valid Test Days: {len(test_df['Date'].unique())} (Should be >700)")

test_ds = UniversalDataset(test_df, processor.feature_cols, CONFIG['seq_len'], len(processor.valid_tickers))
test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

model = AdvancedCrossSectionalModel(
len(processor.valid_tickers), len(processor.feature_cols), CONFIG['hidden_size'],
CONFIG['embedding_dim'], CONFIG['num_heads'], CONFIG['dropout']
).to(CONFIG['device'])

model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
model.eval()

# 2. Inference Loop

print("Generating predictions...")
results = []
valid_dates = test_df['Date'].unique()[CONFIG['seq_len']:]
curr_idx = 0

with torch.no_grad():
for x, ids, y in test_loader:
x, ids, y = x.to(CONFIG['device']), ids.to(CONFIG['device']), y.to(CONFIG['device'])
mu, log_var = model(x, ids)
sigma = torch.exp(0.5 \* log_var)

    mu_np, sigma_np, y_np = mu.cpu().numpy(), sigma.cpu().numpy(), y.cpu().numpy()

    for i in range(len(mu_np)):
            if curr_idx >= len(valid_dates): break
            date = valid_dates[curr_idx]
            for j, ticker in enumerate(processor.valid_tickers):
                results.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Pred': mu_np[i, j],
                    'Sigma': sigma_np[i, j],
                    'Actual': y_np[i, j]
                })
            curr_idx += 1

signal_df = pd.DataFrame(results)
print("Done.")

# 1. Purged IC (Alpha Strength)

def analyze_purged_ic(df, horizon=5):
preds = df.pivot(index='Date', columns='Ticker', values='Pred')
acts = df.pivot(index='Date', columns='Ticker', values='Actual')

    # Purge overlap (Weekly)
    purged_preds = preds.iloc[::horizon]
    purged_acts = acts.iloc[::horizon]

    ics = []
    for d in purged_preds.index:
        ic, _ = spearmanr(purged_preds.loc[d], purged_acts.loc[d])Ftot
        ics.append(ic)
    return np.nanmean(ics), ics

mean_ic, ic_series = analyze_purged_ic(signal_df)
print(f"Purged Weekly IC: {mean_ic}")

# 2. Long-Only Backtest (Economic Value)

mu_matrix = signal_df.pivot(index='Date', columns='Ticker', values='Pred')
sigma_matrix = signal_df.pivot(index='Date', columns='Ticker', values='Sigma')
ret_matrix = signal_df.pivot(index='Date', columns='Ticker', values='Actual')

equity = [100000]
trade_dates = mu_matrix.index[::5]

for d in trade_dates:
p = mu_matrix.loc[d] # Confidence Filter
probs = 1 - norm.cdf(0, p, sigma_matrix.loc[d])
mask = probs > CONFIG['confidence_threshold']

    ranked = p[mask].sort_values(ascending=False)

    if len(ranked) >= CONFIG['top_k']:
        # Long Top K
        picks = ranked.head(CONFIG['top_k']).index
        step_ret = ret_matrix.loc[d, picks].mean()
        net = step_ret - (CONFIG['transaction_cost_bps']/10000)
        equity.append(equity[-1] * (1 + net))
    else:
        equity.append(equity[-1])

total_ret = (equity[-1]/equity[0]) - 1
print(f"Total Return (Long-Only): {total_ret\*100:.2f}%")

# Plot

plt.figure(figsize=(12, 5))
plt.plot(trade_dates, equity[1:], label='Equity Curve')
plt.title(f"Validated Performance (IC: {mean_ic:.4f})")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

Purged Weekly IC: 0.07972907595549104
Total Return (Long-Only): 31.03%

# --- 1. Generate Signals for Backtesting ---

# Use the test loader (full history)

# Ensure test_loader is set up with shuffle=False

with torch.no_grad():
for x, ids, y in test_loader:
x, ids, y = x.to(CONFIG['device']), ids.to(CONFIG['device']), y.to(CONFIG['device'])
mu, log_var = model(x, ids)
sigma = torch.exp(0.5 \* log_var)

    # Convert to numpy
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()
        y_np = y.cpu().numpy()

    # Map back to dates/tickers is tricky with batched loader if shuffle=True.
        # Assuming test_loader is shuffle=False and we track index:
        # Ideally, use the 'generate_signals' function from Phase 4 if available.
        # Here is a simplified reconstruction assuming sequential batches:
        pass # Placeholder: Use the existing generate_signals function

# Use the robust generation function

def get_test_signals(model, loader, processor):
model.eval()
all_res = [] # Reconstruct date mapping from dataset # This relies on the loader being sequential and matching the test_df order # Best approach: Iterate the test_df directly and predict batch-wise

    # Extract unique dates from test set
    test_dates = sorted(test_df['Date'].unique())[CONFIG['seq_len']:]

    print(f"Predicting over {len(test_dates)} days...")
    current_idx = 0

    with torch.no_grad():
        for x, ids, y in loader:
            x, ids = x.to(CONFIG['device']), ids.to(CONFIG['device'])
            mu, log_var = model(x, ids)
            sigma = torch.exp(0.5 * log_var)

    mu_np = mu.cpu().numpy()
            sigma_np = sigma.cpu().numpy()
            y_np = y.cpu().numpy()

    for i in range(len(mu_np)):
                if current_idx >= len(test_dates): break
                date = test_dates[current_idx]

    for j, ticker in enumerate(processor.valid_tickers):
                    all_res.append({
                        'Date': date,
                        'Ticker': ticker,
                        'Pred_Mu': mu_np[i, j],
                        'Pred_Sigma': sigma_np[i, j],
                        'Actual_Return': y_np[i, j] # 5-day forward return
                    })
                current_idx += 1

    return pd.DataFrame(all_res)

signal_df = get_test_signals(model, test_loader, processor)
print(f"Signals generated: {len(signal_df)} rows")
signal_df.head()

Generating full signal history for backtesting...
Predicting over 224 days...
Signals generated: 11872 rows
Date Ticker Pred_Mu Pred_Sigma Actual_Return
0 2024-12-27 AAPL 0.032402 0.319673 -0.062363
1 2024-12-27 ABBV 0.034049 0.311389 0.011209
2 2024-12-27 ADBE 0.022492 0.311808 -0.044493
3 2024-12-27 AIG 0.024598 0.308843 -0.007253
4 2024-12-27 AMAT 0.035754 0.315155 0.014869

import numpy as np
import pandas as pd
from scipy.stats import norm

class ComprehensiveBacktester:
def **init**(self, signal_df, config):
self.signals = signal_df.copy()
self.config = config
self.cost_bps = config.get('transaction_cost_bps', 10)
self.top_k = config.get('top_k', 3)

    def run_strategy(self, strategy_type='long_only'):
        """
        Runs backtest for a specific strategy type.
        strategy_type: 'long_only' (Top K) or 'long_short' (Top K - Bottom K)
        """
        # 1. Pivot Data
        preds = self.signals.pivot(index='Date', columns='Ticker', values='Pred_Mu')
        sigmas = self.signals.pivot(index='Date', columns='Ticker', values='Pred_Sigma')
        returns = self.signals.pivot(index='Date', columns='Ticker', values='Actual_Return')

    # 2. Trading Schedule (Purged: Trade every 5 days)
        # We assume we hold for 5 days, then rebalance.
        trade_dates = preds.index[::self.config['pred_horizon']]

    # 3. Calculate Weights
        weights = pd.DataFrame(0.0, index=trade_dates, columns=preds.columns)

    for date in trade_dates:
            row_p = preds.loc[date]
            row_s = sigmas.loc[date]

    # Confidence Filter
            probs = 1 - norm.cdf(0, loc=row_p, scale=row_s)
            valid_mask = probs > self.config['confidence_threshold']

    # Rank valid stocks
            valid_preds = row_p[valid_mask]
            if len(valid_preds) < self.top_k:
                continue # Cash

    ranked = valid_preds.sort_values(ascending=False)

    if strategy_type == 'long_only':
                longs = ranked.head(self.top_k).index
                weights.loc[date, longs] = 1.0 / len(longs)

    elif strategy_type == 'long_short':
                longs = ranked.head(self.top_k).index
                shorts = ranked.tail(self.top_k).index
                if len(shorts) >= self.top_k:
                    weights.loc[date, longs] = 0.5 / len(longs)
                    weights.loc[date, shorts] = -0.5 / len(shorts)

    # 4. Calculate Returns & Turnover
        # Forward returns: Weight[t] * Return[t -> t+5]
        # Note: returns dataframe should already be forward looking 5-day returns
        rebalanced_returns = returns.loc[trade_dates]
        gross_ret = (weights * rebalanced_returns).sum(axis=1)

    # Turnover: Sum of absolute weight changes
        # We compare current desired weights vs previous weights
        # (Assuming we hold exactly the previous weights until rebalance)
        weight_delta = weights.diff().fillna(weights).abs().sum(axis=1)

    # Transaction Costs
        # Cost = Turnover * bps
        costs = weight_delta * (self.cost_bps / 10000.0)

    net_ret = gross_ret - costs

    # Equity Curve
        equity = (1 + net_ret).cumprod()

    return {
            'equity': equity,
            'net_ret': net_ret,
            'turnover': weight_delta,
            'weights': weights
        }

    def calculate_risk_metrics(self, results, name="Strategy"):
        returns = results['net_ret']
        equity = results['equity']
        turnover = results['turnover']

    # Annualization Factor (approx 50 trading periods per year)
        ann = 252 / self.config['pred_horizon']

    # 1. Returns
        total_ret = equity.iloc[-1] - 1
        cagr = (equity.iloc[-1])**(ann/len(returns)) - 1

    # 2. Risk
        vol = returns.std() * np.sqrt(ann)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(ann) if vol > 0 else 0

    # 3. Sortino (Downside Risk)
        neg_ret = returns[returns < 0]
        downside_dev = neg_ret.std() * np.sqrt(ann)
        sortino = (returns.mean() * ann) / downside_dev if downside_dev > 0 else 0

    # 4. Drawdown & Calmar
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # 5. Turnover
        avg_turnover = turnover.mean()

    return {
            "Strategy": name,
            "Total Return": f"{total_ret:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Sharpe": f"{sharpe:.2f}",
            "Sortino": f"{sortino:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Calmar": f"{calmar:.2f}",
            "Avg Turnover": f"{avg_turnover:.2%}"
        }

print("Backtester Ready.")

# Initialize

backtester = ComprehensiveBacktester(signal_df, CONFIG)

# Run Long-Only

res_long = backtester.run_strategy('long_only')
metrics_long = backtester.calculate_risk_metrics(res_long, "Long-Only Top 3")

# Run Long-Short

res_ls = backtester.run_strategy('long_short')
metrics_ls = backtester.calculate_risk_metrics(res_ls, "Long-Short (Neutral)")

# Display Metrics Table

metrics_df = pd.DataFrame([metrics_long, metrics_ls]).set_index("Strategy")
print("\n=== Performance Report (Net of 10bps Cost) ===")
display(metrics_df)

# Plot Equity Curves

plt.figure(figsize=(12, 6))
plt.plot(res_long['equity'], label='Long-Only', color='green', linewidth=2)
plt.plot(res_ls['equity'], label='Long-Short', color='grey', alpha=0.7)
plt.axhline(1.0, color='black', linestyle='--', alpha=0.3)
plt.title("Cumulative Returns: Long-Only vs Long-Short")
plt.ylabel("Wealth Index (Start=1.0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot Drawdowns

plt.figure(figsize=(12, 3))
roll_max_l = res_long['equity'].cummax()
dd_l = (res_long['equity'] - roll_max_l) / roll_max_l
plt.fill_between(dd_l.index, dd_l, 0, color='red', alpha=0.3, label='Long-Only Drawdown')
plt.title("Long-Only Drawdown Profile")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

Strategy Total Return CAGR Sharpe Sortino Max Drawdown Calmar Avg Turnover
Long-Only Top 5 35.86% 40.95% 1.49 2.75 -19.23% 2.13 19.11%
Long-Short (Neutral) 20.90% 23.68% 1.96 3.60 -4.26% 5.56 25.78%

# --- 4. ADVANCED METRICS & STABILITY ANALYSIS ---

def calculate*turnover(weights):
"""
Calculates portfolio turnover: sum(|w_t - w*{t-1}|)
Assumes rebalancing every 'horizon' days.
"""
delta = weights.diff().fillna(weights)
daily_turnover = delta.abs().sum(axis=1)
return daily_turnover.mean()

def plot_regime_stability(ic_series, dates, horizon=5):
cum_ic = np.cumsum(ic_series)

    plt.figure(figsize=(12, 5))
    plt.plot(dates, cum_ic, label='Cumulative IC', color='purple', linewidth=2)
    plt.title(f"Signal Stability Analysis (Purged {horizon}-Day IC)", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Information Coefficient")
    plt.grid(True, alpha=0.3)

    z = np.polyfit(range(len(cum_ic)), cum_ic, 1)
    p = np.poly1d(z)
    plt.plot(dates, p(range(len(cum_ic))), "r--", alpha=0.6, label="Trend (Stability)")

    plt.legend()
    plt.show()

# --- RUN DIAGNOSTICS ---

# 1. Re-Run Backtest to get Weights for Turnover

# Use CORRECT column names: 'Pred_Mu' and 'Pred_Sigma'

mu_matrix = signal_df.pivot(index='Date', columns='Ticker', values='Pred_Mu')
sigma_matrix = signal_df.pivot(index='Date', columns='Ticker', values='Pred_Sigma')
trade_dates = mu_matrix.index[::CONFIG['pred_horizon']]

# Reconstruct Weights DataFrame

weights = pd.DataFrame(0.0, index=trade_dates, columns=mu_matrix.columns)

for d in trade_dates:
p = mu_matrix.loc[d]
probs = 1 - norm.cdf(0, p, sigma_matrix.loc[d])
mask = probs > CONFIG['confidence_threshold']
ranked = p[mask].sort_values(ascending=False)

    if len(ranked) >= CONFIG['top_k']:
        picks = ranked.head(CONFIG['top_k']).index
        weights.loc[d, picks] = 1.0 / len(picks)

# 2. Calculate Turnover

avg_turnover = calculate_turnover(weights)
print(f"Average Turnover per Rebalance: {avg_turnover:.2%}")

# 3. Calculate Risk Ratios (Sortino & Calmar)

if 'equity' in locals():
returns = pd.Series(equity).pct_change().dropna()
downside = returns[returns < 0]
annual_factor = 252 / CONFIG['pred_horizon']

    sortino = (returns.mean() * annual_factor) / (downside.std() * np.sqrt(annual_factor) + 1e-9)
    calmar = total_ret / abs((pd.Series(equity) / pd.Series(equity).cummax() - 1).min())

    print(f"Sortino Ratio: {sortino:.2f} (Target > 1.0)")
    print(f"Calmar Ratio:  {calmar:.2f} (Target > 0.5)")

else:
print("Equity curve not found. Run Cell 3 first.")

# 4. Plot Regime Stability

ic_dates = mu_matrix.index[::CONFIG['pred_horizon']]
min_len = min(len(ic_dates), len(ic_series))
plot_regime_stability(ic_series[:min_len], ic_dates[:min_len])

Average Turnover per Rebalance: 19.11%
Sortino Ratio: 2.41 (Target > 1.0)
Calmar Ratio: 1.56 (Target > 0.5)

# --- 5. PUBLICATION-GRADE PLOTS ---

import matplotlib.dates as mdates

# Prepare Data

equity_series = pd.Series(equity, index=[mu_matrix.index[0]] + list(trade_dates))
returns_series = equity_series.pct_change().dropna()

# 1. Rolling Sharpe (6-Month Window)

rolling_sharpe = returns_series.rolling(window=25).mean() / returns_series.rolling(window=25).std() \* np.sqrt(50)

# 2. Drawdown Series

roll_max = equity_series.cummax()
drawdown = (equity_series - roll_max) / roll_max

# PLOTTING

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Panel A: Equity Curve

axes[0].plot(equity_series.index, equity_series.values, color='green', linewidth=2, label='Strategy (Long-Only)')
axes[0].set_title('A. Cumulative Wealth (Log Scale)', fontsize=12, fontweight='bold', loc='left')
axes[0].set_yscale('log')
axes[0].grid(True, which="both", ls="-", alpha=0.2)
axes[0].legend(loc='upper left')

# Panel B: Underwater Drawdown

axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3, label='Drawdown')
axes[1].set_title('B. Drawdown Profile', fontsize=12, fontweight='bold', loc='left')
axes[1].set_ylabel('Drawdown (%)')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='lower left')

# Panel C: Rolling Sharpe

axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, color='blue', linewidth=1.5, label='6-Month Rolling Sharpe')
axes[2].axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Threshold (1.0)')
axes[2].set_title('C. Consistency Analysis (Rolling Sharpe)', fontsize=12, fontweight='bold', loc='left')
axes[2].set_ylabel('Sharpe Ratio')
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc='upper left')

# Formatting

plt.xlabel('Date')
plt.tight_layout()
plt.show()

# --- 5. STATISTICAL VALIDATION & ROBUSTNESS METRICS ---

import scipy.stats as stats

# 1. Calculate Time-Series of IC (Information Coefficient)

# We need the daily/weekly IC values to compute stability metrics

def get_ic_series(df, horizon=5):
preds = df.pivot(index='Date', columns='Ticker', values='Pred_Mu')
acts = df.pivot(index='Date', columns='Ticker', values='Actual_Return')

    # Purge overlap to ensure independence for statistical tests
    purged_preds = preds.iloc[::horizon]
    purged_acts = acts.iloc[::horizon]

    ic_values = []
    for d in purged_preds.index:
        # Spearman Rank Correlation
        ic, _ = spearmanr(purged_preds.loc[d], purged_acts.loc[d])
        if not np.isnan(ic):
            ic_values.append(ic)
    return np.array(ic_values)

# Get the IC series

ic_series = get_ic_series(signal_df, horizon=CONFIG['pred_horizon'])

# 2. Compute Metrics

mean_ic = np.mean(ic_series)
std_ic = np.std(ic_series)
n_samples = len(ic_series)

# A. IC Information Ratio (IC-IR)

ic_ir = mean_ic / (std_ic + 1e-9)

# B. Win Rate (Positive vs Negative Periods)

n_positive = np.sum(ic_series > 0)
n_negative = np.sum(ic_series < 0)
win_rate = n_positive / n_samples

# C. Bootstrap Confidence Interval (95%)

def bootstrap*ci(data, n_boot=10000, ci=0.95):
boot_means = []
for * in range(n_boot):
sample = np.random.choice(data, size=len(data), replace=True)
boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (1 - ci)/2 * 100)
    upper = np.percentile(boot_means, (1 + ci)/2 * 100)
    return lower, upper

ci_lower, ci_upper = bootstrap_ci(ic_series)

# D. Empirical p-value (One-sided t-test against 0)

# H0: Mean IC <= 0, H1: Mean IC > 0

t_stat, p_value = stats.ttest_1samp(ic_series, 0, alternative='greater')

# --- PRINT REPORT ---

print("\n=== STATISTICAL ROBUSTNESS REPORT ===")
print(f"1. IC Information Ratio (IC-IR): {ic_ir:.3f}")
print(f"\n2. Win Rate (Positive IC Periods): {win_rate:.1%} ({n_positive}/{n_samples})")
print(f"\n3. Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"\n4. Empirical p-value: {p_value:.4f}")

# --- VISUALIZATION ---

median_ic = np.median(ic_series)

plt.figure(figsize=(10, 5))
plt.hist(ic_series, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(mean_ic, color='red', linestyle='--', linewidth=2, label=f'Mean IC: {mean_ic:.3f}')
plt.axvline(median_ic, color='green', linestyle='-.', linewidth=2, label=f'Median IC: {median_ic:.3f}')
plt.title("Distribution of Information Coefficients (IC)")
plt.xlabel("IC Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

=== STATISTICAL ROBUSTNESS REPORT ===

1. IC Information Ratio (IC-IR): 0.487

2. Win Rate (Positive IC Periods): 68.9% (31/45)

3. Bootstrap 95% CI: [0.0323, 0.1264]

4. Empirical p-value: 0.0012
