import torch
import numpy as np
import random
from scipy.stats import spearmanr
from config import CONFIG
from data import UniversalDataProcessor, get_dataloaders
from model import AdvancedCrossSectionalModel, HybridLoss

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("--- TrendSpy Training Pipeline ---")
    set_seed(CONFIG['seed'])
    
    # 1. Data
    processor = UniversalDataProcessor()
    train_loader, test_loader, _ = get_dataloaders(processor)
    
    # 2. Model
    model = AdvancedCrossSectionalModel(
        num_stocks=len(processor.valid_tickers),
        input_dim=len(processor.feature_cols),
        hidden_dim=CONFIG['hidden_size'],
        embed_dim=CONFIG['embedding_dim'],
        num_heads=CONFIG['num_heads'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = HybridLoss(alpha=CONFIG['alpha_weight'], margin=CONFIG['ranking_margin'])
    
    best_ic = -1.0
    
    # 3. Loop
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        
        for x, ids, y in train_loader:
            x, ids, y = x.to(CONFIG['device']), ids.to(CONFIG['device']), y.to(CONFIG['device'])
            optimizer.zero_grad()
            mu, log_var = model(x, ids)
            loss = criterion(mu, log_var, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        scheduler.step()
        
        # Validation
        model.eval()
        ics = []
        with torch.no_grad():
            for x, ids, y in test_loader:
                x, ids, y = x.to(CONFIG['device']), ids.to(CONFIG['device']), y.to(CONFIG['device'])
                mu, _ = model(x, ids)
                
                mu_np = mu.cpu().numpy()
                y_np = y.cpu().numpy()
                
                for i in range(len(mu_np)):
                    if np.std(mu_np[i]) > 1e-6:
                        ic, _ = spearmanr(mu_np[i], y_np[i])
                        if not np.isnan(ic): ics.append(ic)
        
        avg_ic = np.mean(ics) if ics else 0.0
        print(f"Epoch {epoch+1} | Val IC: {avg_ic:.4f}")
        
        if avg_ic > best_ic:
            best_ic = avg_ic
            torch.save(model.state_dict(), CONFIG['model_path'])
            print(f"  --> New Best! Saved to {CONFIG['model_path']}")
            
    print(f"\nTraining Complete. Peak IC: {best_ic:.4f}")

if __name__ == "__main__":
    main()