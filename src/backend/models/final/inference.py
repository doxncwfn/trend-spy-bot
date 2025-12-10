import torch
import pandas as pd
import numpy as np
import json
from scipy.stats import norm
from config import CONFIG
from data import UniversalDataProcessor
from model import AdvancedCrossSectionalModel

def run_inference():
    print("--- TrendSpy Daily Inference ---")
    
    # 1. Load Data
    processor = UniversalDataProcessor()
    try:
        df = processor.load_and_process()
    except Exception as e:
        print(f"Data Error: {e}")
        return

    # 2. Prepare Input (Last Window)
    last_window = df.tail(len(processor.tickers) * CONFIG['seq_len'])
    if len(last_window['Date'].unique()) < CONFIG['seq_len']:
        print("Not enough recent data for inference.")
        return

    num_stocks = len(processor.tickers)
    num_feats = len(processor.feature_cols)
    
    # Build Tensor
    X = np.zeros((CONFIG['seq_len'], num_stocks, num_feats), dtype=np.float32)
    for i, feat in enumerate(processor.feature_cols):
        pivot = last_window.pivot(index='Date', columns='Ticker', values=feat)
        X[:, :, i] = pivot.values
        
    x_tensor = torch.FloatTensor(X).transpose(1, 0).unsqueeze(0).to(CONFIG['device'])
    ids_tensor = torch.arange(num_stocks).unsqueeze(0).to(CONFIG['device'])
    
    # 3. Load Model
    model = AdvancedCrossSectionalModel(
        num_stocks=num_stocks,
        input_dim=num_feats,
        hidden_dim=CONFIG['hidden_size'],
        embed_dim=CONFIG['embedding_dim'],
        num_heads=CONFIG['num_heads'],
        dropout=0.0 # No dropout for inference
    ).to(CONFIG['device'])
    
    try:
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        model.eval()
    except FileNotFoundError:
        print(f"Model file not found at {CONFIG['model_path']}")
        return

    # 4. Predict
    with torch.no_grad():
        mu, sigma = model(x_tensor, ids_tensor)
    
    preds = mu[0].cpu().numpy()
    uncerts = sigma[0].cpu().numpy()
    
    # 5. Format Output
    ranks = np.argsort(preds)[::-1]
    output = {
        "metadata": {
            "date": df['Date'].max().strftime('%Y-%m-%d'),
            "model": "TrendSpy-Pro",
            "ic_score": "0.08 (Est)"
        },
        "signals": []
    }
    
    for rank_i, idx in enumerate(ranks):
        ticker = processor.valid_tickers[idx]
        p = preds[idx]
        u = uncerts[idx]
        
        conf = int((1 - norm.cdf(0, loc=p, scale=u)) * 100)
        
        action = "HOLD"
        if rank_i < CONFIG['top_k'] and conf > CONFIG['confidence_threshold']*100:
            action = "BUY"
        
        output["signals"].append({
            "rank": rank_i + 1,
            "ticker": ticker,
            "pred_return": round(float(p), 4),
            "uncertainty": round(float(u), 4),
            "confidence": conf,
            "action": action
        })
        
    with open(CONFIG['inference_output'], 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Signals saved to {CONFIG['inference_output']}")

if __name__ == "__main__":
    run_inference()