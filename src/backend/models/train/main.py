import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from model import StockForecaster
from data_loader import StockDataLoader
from evaluator import ModelEvaluator
from config import CONFIG, FEATURE_COLS

# Setup logging...
log_dir = Path(__file__).resolve().parent.parent / 'logs'
log_dir.mkdir(exist_ok=True, parents=True)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class ProductionPipeline:
    def __init__(self, config):
        self.config = config
        self.root = Path(__file__).resolve().parent.parent
        self.loader = StockDataLoader(str(self.root / 'data'))
        self.evaluator = ModelEvaluator(str(self.root / 'metrics'))

    def train_single_stock(self, ticker):
        logger.info(f"Training {ticker}...")
        
        # 1. Load Data
        train, val, test = self.loader.load_stock(ticker, self.config['test_size'], self.config['val_size'])
        
        # 2. Init Model
        model = StockForecaster(self.config, device='cpu')
        
        # 3. Train
        start = datetime.now()
        history = model.fit(train, val, self.config['epochs'])
        duration = (datetime.now() - start).total_seconds()
        
        # 4. Evaluate
        preds, actuals, cls_probs = self._evaluate_model(model, test)
        
        # 5. Metrics
        metrics = self.evaluator.calculate_metrics(actuals, preds, np.array(cls_probs))
        
        logger.info(f"Results {ticker}: RMSE={metrics['RMSE']:.4f}, DirAcc={metrics['Directional_Accuracy']:.2f}%")
        
        # 6. Save
        model.save_model(str(self.root / 'train' / 'checkpoints' / f'{ticker}_model.pth'))
        return metrics

    def _evaluate_model(self, model, test_df):
        """
        Manually transforms features using the trained scaler and reconstructs prices.
        """
        # Get raw feature values
        raw_x = test_df[FEATURE_COLS].values.astype(np.float32)
        # Transform using the scaler fitted on TRAIN data (Stored in model.scaler)
        scaled_x = model.scaler.transform(raw_x)
        
        # Get actual close prices for reconstruction
        # We need the Close price at time T to predict Price at T+1
        close_prices = test_df['Close'].values
        
        predictions = []
        actuals = []
        cls_probs = []
        
        model.model.eval()
        with torch.no_grad():
            for i in range(len(scaled_x) - model.seq_len):
                # Input Sequence
                seq = scaled_x[i : i + model.seq_len]
                seq_t = torch.FloatTensor(seq).unsqueeze(0).to(model.device)
                
                # Predict
                pred_log_ret, pred_cls = model.model(seq_t)
                
                # Reconstruct Price
                # The input sequence ends at index (i + seq_len - 1). This is "Today".
                last_known_price = close_prices[i + model.seq_len - 1]
                
                # We predicted the Log Return for "Tomorrow"
                predicted_price = last_known_price * np.exp(pred_log_ret.item())
                
                # The actual price for "Tomorrow" is at index (i + seq_len)
                actual_price = close_prices[i + model.seq_len]
                
                predictions.append(predicted_price)
                actuals.append(actual_price)
                cls_probs.append(torch.sigmoid(pred_cls).item())
                
        return np.array(predictions), np.array(actuals), cls_probs

if __name__ == "__main__":
    pipeline = ProductionPipeline(CONFIG)
    pipeline.train_single_stock("AXON")