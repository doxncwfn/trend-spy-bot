CONFIG = {
    # Data parameters
    'seq_length': 60,           # Number of days to look back
    'pred_horizon': 1,          # Number of days to predict ahead
    'test_size': 0.2,           # 20% for testing
    'val_size': 0.1,            # 10% for validation
    
    # Model architecture
    'hidden_size': 128,         # Hidden layer size
    'num_lstm_layers': 2,       # Number of LSTM layers
    'num_attention_heads': 8,   # Number of attention heads
    'num_transformer_layers': 2,# Number of transformer blocks
    'dropout': 0.2,             # Dropout rate
    
    # Training parameters
    'learning_rate': 0.0003,    # Initial learning rate
    'batch_size': 64,           # Batch size
    'epochs': 100,              # Maximum epochs
    'early_stopping_patience': 15,  # Early stopping patience
    
    # Device
    'device': 'cpu',           # 'cuda' or 'cpu'
}


AVAILABLE_STOCKS = [
    'AAPL',   # Apple
    'AMD',    # AMD
    'AMZN',   # Amazon
    'DIS',    # Disney
    'GOOGL',  # Google
    'META',   # Meta
    'MSFT',   # Microsoft
    'NFLX',   # Netflix
    'NVDA',   # NVIDIA
    'TSLA',   # Tesla
]