FEATURE_COLS = [
    'log_return', 
    'range_pct', 
    'close_loc', 
    'volume_log_change', 
    'dist_sma_10', 
    'dist_sma_30', 
    'macd_norm', 
    'macd_sig_norm', 
    'rsi'
]

CONFIG = {
    # Data parameters
    'seq_length': 60,
    'pred_horizon': 5,
    'test_size': 0.2,
    'val_size': 0.1,
    
    # Model architecture
    'input_size': len(FEATURE_COLS),
    'hidden_size': 128,
    'num_lstm_layers': 2,
    'num_attention_heads': 4,
    'num_transformer_layers': 2,
    'dropout': 0.2,
    
    # Training parameters
    'learning_rate': 0.0003,
    'batch_size': 64,
    'epochs': 100,
    'early_stopping_patience': 15,
    
    # Hybrid Loss Weights
    'lambda_reg': 1.0,
    'lambda_cls': 5.0,
    
    'device': 'cpu',
}