import os
import sys
from pathlib import Path
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from itertools import product

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from model import StockForecaster
from data_loader import StockDataLoader
from evaluator import ModelEvaluator
from config import CONFIG

log_dir = Path(__file__).resolve().parent.parent / 'logs'
tuning_log_path = log_dir / f'tuning_run_{datetime.now():%Y%m%d_%H%M%S}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(tuning_log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LambdaTuner(ModelEvaluator):
    def __init__(self, config: dict, target_stock: str = 'AMZN'):
        super().__init__('tuning_metrics')
        
        self.config = config
        self.target_stock = target_stock
        self.data_dir = Path(__file__).resolve().parent.parent / 'data'
        self.loader = StockDataLoader(str(self.data_dir))
        
        self.train_df, self.val_df, self.test_df = self.loader.load_stock(
            self.target_stock,
            test_size=self.config['test_size'],
            val_size=self.config['val_size']
        )
        
        self.all_results = []

    def _evaluate_model_on_test_set(self, model):
        """
        Replicates evaluation logic from main.py but using log returns.
        Reconstructs prices using exponential of predicted log returns.
        """
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'Return', 'SMA_10', 'SMA_30', 'MACD', 'MACD_Signal', 'RSI'
        ]
        
        test_features = self.test_df[feature_cols].values.astype(np.float32)
        test_close = self.test_df['Close'].values.astype(np.float32)
        
        predictions = []
        actuals = []
        cls_probs = []

        model.model.eval()
        with torch.no_grad():
            for i in range(len(test_features) - model.seq_len):
                # Get feature sequence
                seq = test_features[i:i + model.seq_len]
                scaled_seq = torch.FloatTensor(model.scaler.transform(seq)).unsqueeze(0).to(model.device)
                
                # Get prediction
                pred_log_return, pred_cls = model.model(scaled_seq)
                pred_log_return = pred_log_return.item() if pred_log_return.dim() == 0 else pred_log_return.squeeze().item()
                pred_prob = torch.sigmoid(pred_cls).item()

                last_close = test_close[i + model.seq_len - 1]
                pred_price = last_close * np.exp(pred_log_return)
                actual_price = test_close[i + model.seq_len]
                
                predictions.append(pred_price)
                actuals.append(actual_price)
                cls_probs.append(pred_prob)
        
        return np.array(predictions), np.array(actuals), np.array(cls_probs)


    def run_grid_search(self, lambda_reg_values: list, lambda_cls_values: list):
        """Iterates through all lambda combinations and evaluates model performance."""

        logger.info("="*80)
        logger.info(f"Starting Hyperparameter Grid Search for {self.target_stock}")
        logger.info(f"Lambda Reg Search Space: {lambda_reg_values}")
        logger.info(f"Lambda Cls Search Space: {lambda_cls_values}")
        logger.info("="*80)

        best_score = float('-inf')
        best_params = {}
        
        # Grid Search iteration
        for lambda_reg, lambda_cls in product(lambda_reg_values, lambda_cls_values):
            # 1. Initialize and Configure Model
            model = StockForecaster(
                seq_len=self.config['seq_length'],
                device=self.config['device']
            )
            model.criterion = model.criterion.__class__(
                lambda_reg=lambda_reg, 
                lambda_cls=lambda_cls
            )
            logger.info(f"\n---> Tuning: Reg={lambda_reg:.2f}, Cls={lambda_cls:.2f} <---")
            
            try:
                # 2. Train Model
                start_time = datetime.now()
                history = model.fit(self.train_df, self.val_df, epochs=self.config['epochs'])
                training_time = (datetime.now() - start_time).total_seconds()

                # 3. Evaluate on Test Set
                test_predictions, test_actuals, cls_probs = self._evaluate_model_on_test_set(model)
                metrics = self.calculate_metrics(test_actuals, test_predictions, y_cls_prob=cls_probs)

                # 4. Define Scoring Metric (Prioritizing Directional Accuracy)
                # Use a custom score that rewards high directional accuracy but penalizes high RMSE
                # Formula: (Directional Accuracy) - (1 * RMSE)
                score = metrics['Directional_Accuracy'] - (1 * metrics['RMSE'])

                self.all_results.append({
                    'lambda_reg': lambda_reg,
                    'lambda_cls': lambda_cls,
                    'score': score,
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2'],
                    'Dir_Acc_%': metrics['Directional_Accuracy'],
                    'Dir_Acc_Source': metrics.get('Dir_Acc_Source', 'N/A'),
                    'train_time_s': training_time
                })
                logger.info(f"  Result: RMSE={metrics['RMSE']:.4f}, Dir Acc={metrics['Directional_Accuracy']:.2f}%, Score={score:.2f}")

                # 5. Check Best Model
                if score > best_score:
                    best_score = score
                    best_params = {'lambda_reg': lambda_reg, 'lambda_cls': lambda_cls, 'score': score}
                    logger.info("  *** NEW BEST MODEL FOUND ***")

            except Exception as e:
                logger.error(f"  Failed for lambda_reg={lambda_reg}, lambda_cls={lambda_cls}: {e}")
                continue

        logger.info("\n" + "="*80)
        logger.info("GRID SEARCH COMPLETE")
        logger.info(f"BEST PARAMS: {best_params}")
        logger.info("="*80)
        
        # Save summary of all runs
        summary_df = pd.DataFrame(self.all_results).sort_values('score', ascending=False)
        summary_path = self.output_dir / f'{self.target_stock}_lambda_tuning_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Full results saved to {summary_path}")
        
        return best_params

if __name__ == "__main__":
    LAMBDA_REG_VALUES = [0.5, 1.0, 2.0]
    LAMBDA_CLS_VALUES = [1.0, 3.0, 5.0, 7.0]

    tuner = LambdaTuner(config=CONFIG, target_stock='AMZN') 
    best_lambdas = tuner.run_grid_search(LAMBDA_REG_VALUES, LAMBDA_CLS_VALUES)

    print("\nOptimal Lambda Parameters Found:")
    print(json.dumps(best_lambdas, indent=4))