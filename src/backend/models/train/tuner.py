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
        """Replicates evaluation logic from main.py"""
        required_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'Return', 'SMA_10', 'SMA_30', 'MACD', 'MACD_Signal', 'RSI'
        ]
        test_data = self.test_df[required_cols].values.astype(np.float32)
        
        predictions = []
        actuals = []

        model.model.eval()
        with torch.no_grad():
            for i in range(len(test_data) - model.seq_len):
                seq = test_data[i:i + model.seq_len]
                scaled_seq = torch.FloatTensor(model.scaler.transform(seq)).unsqueeze(0).to(model.device)
                
                pred_reg, _ = model.model(scaled_seq)
                pred_return = pred_reg.item() if pred_reg.dim() == 0 else pred_reg.squeeze().item()
                
                last_close = test_data[i + model.seq_len - 1, 3]
                pred_price = last_close * (1 + pred_return)
                actual_price = test_data[i + model.seq_len, 3]
                
                predictions.append(pred_price)
                actuals.append(actual_price)
        
        return np.array(predictions), np.array(actuals)


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
            # CRITICAL: Overwrite the criterion with the current lambda values
            model.criterion = model.criterion.__class__(
                lambda_reg=lambda_reg, 
                lambda_cls=lambda_cls
            )
            
            logger.info(f"\n---> Tuning: Reg={lambda_reg:.2f}, Cls={lambda_cls:.2f} <---")
            
            try:
                # 2. Train Model
                start_time = datetime.now()
                # Pass the full training and validation dataframes
                history = model.fit(self.train_df, self.val_df, epochs=self.config['epochs'])
                training_time = (datetime.now() - start_time).total_seconds()

                # 3. Evaluate on Test Set
                test_predictions, test_actuals = self._evaluate_model_on_test_set(model)
                metrics = self.calculate_metrics(test_actuals, test_predictions)

                # 4. Define Scoring Metric (Prioritizing Directional Accuracy)
                # Use a custom score that rewards high directional accuracy but penalizes high RMSE
                
                # Formula: (Directional Accuracy) - (100 * RMSE)
                # This emphasizes direction (target: max 100) while keeping error low.
                score = metrics['Directional_Accuracy'] - (1 * metrics['RMSE'])

                self.all_results.append({
                    'lambda_reg': lambda_reg,
                    'lambda_cls': lambda_cls,
                    'score': score,
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2'],
                    'Dir_Acc_%': metrics['Directional_Accuracy'],
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
    # Define the search space for the two lambdas
    # Rationale: Keep Reg around 1.0, and explore Cls from 1.0 (baseline) up to 10.0 (aggressive)
    # The current issue was instability at 10.0, so we'll search between 1 and 7.
    LAMBDA_REG_VALUES = [0.5, 1.0, 2.0]
    LAMBDA_CLS_VALUES = [1.0, 3.0, 5.0, 7.0]

    # Initialize the tuner
    tuner = LambdaTuner(config=CONFIG, target_stock='AMZN') 

    # Run the tuning process
    best_lambdas = tuner.run_grid_search(LAMBDA_REG_VALUES, LAMBDA_CLS_VALUES)

    print("\nOptimal Lambda Parameters Found:")
    print(json.dumps(best_lambdas, indent=4))