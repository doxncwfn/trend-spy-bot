import torch
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'

# Ensure directories exist
for d in [MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True, parents=True)

CONFIG = {
    # Paths
    'data_dir': str(DATA_DIR),
    'model_path': str(MODEL_DIR / 'trendspy_best.pth'),
    'inference_output': str(OUTPUT_DIR / 'latest_inference.json'),

    # Universe (Fixed 53)
    'tickers': sorted([
        'AAPL', 'ABBV', 'ADBE', 'AIG', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'AXON', 'BA', 
        'BAC', 'BLK', 'CAT', 'COST', 'CRM', 'CSCO', 'CVX', 'DE', 'DIS', 'GE', 
        'GOOGL', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'LLY', 'MA', 
        'MCD', 'META', 'MRK', 'MS', 'MSFT', 'MU', 'NFLX', 'NKE', 'NVDA', 'ORCL', 
        'PEP', 'PFE', 'PG', 'QCOM', 'SBUX', 'SPY', 'TSLA', 'TXN', 'UBER', 'UNH', 
        'V', 'WMT', 'XOM'
    ]),

    # Data Parameters
    'force_start_date': '2020-01-01',
    'seq_len': 60,
    'pred_horizon': 5,
    
    # Model Hyperparameters (Golden Config)
    'hidden_size': 128,
    'embedding_dim': 16,
    'num_heads': 4,
    'dropout': 0.43951755524391284,
    'weight_decay': 5.3788742936530506e-05, # L2 Regularization
    'learning_rate': 0.00013061115141321854,
    
    # Training Settings
    'batch_size': 32,      # 32 Days per batch
    'epochs': 50,
    'patience': 8,
    'ranking_margin': 0.1,
    'alpha_weight': 0.6967878897119416, # Ranking loss
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 4291,            # Deterministic seed
    
    # Trading / Backtest Strategy
    'initial_capital': 100000,
    'transaction_cost_bps': 10,
    'top_k': 3,
    'confidence_threshold': 0.50, # Probability > 50%
    'rebalance_freq': 5           # Trade every 5 days
}