import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime
import json

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from model import StockForecaster
from data_loader import StockDataLoader
from evaluator import ModelEvaluator
from config import CONFIG

# Setup logging
log_dir = Path(__file__).resolve().parent.parent / 'logs'
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config: dict):
        self.config = config
        project_root = Path(__file__).resolve().parent.parent
        self.data_dir = project_root / 'data'
        self.output_dir = project_root / 'train' / 'checkpoints'
        self.metrics_dir = project_root / 'metrics'
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.data_loader = StockDataLoader(str(self.data_dir))
        self.evaluator = ModelEvaluator(str(self.metrics_dir))
        
        self.models = {}
        self.results = {}
    
    def train_single_stock(self, ticker: str) -> dict:
        """
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"{'='*80}")
        logger.info(f"Training model for {ticker}")
        logger.info(f"{'='*80}")
        
        try:
            # Load and prepare data
            train_df, val_df, test_df = self.data_loader.load_stock(
                ticker,
                test_size=self.config['test_size'],
                val_size=self.config['val_size']
            )
            
            logger.info(f"Data loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            # Initialize model
            model = StockForecaster(
                seq_len=self.config['seq_length'],
                device='cpu'
            )

            # Prepare data loaders
            train_loader, _ = model.prepare_data(train_df)
            val_loader,   _ = model.prepare_data(val_df)

            logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

            # Train
            logger.info("Starting training...")
            start_time = datetime.now()

            # history now contains loss based on the weighted CombinedLoss
            history = model.fit(train_df, val_df, epochs=self.config['epochs'])

            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_predictions, test_actuals = self._evaluate_model(model, test_df)
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(test_actuals, test_predictions)
            
            logger.info(f"Test Metrics:")
            logger.info(f"  RMSE: {metrics['RMSE']:.4f}")
            logger.info(f"  MAE: {metrics['MAE']:.4f}")
            logger.info(f"  MAPE: {metrics['MAPE']:.2f}%")
            logger.info(f"  R²: {metrics['R2']:.4f}")
            logger.info(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
            
            # Save model
            model_path = self.output_dir / f'{ticker}_model.pth'
            model.save_model(str(model_path))
            logger.info(f"Model saved to {model_path}")
            
            # Generate visualizations
            self.evaluator.plot_predictions(
                test_actuals,
                test_predictions,
                dates=test_df['Date'].iloc[self.config['seq_length']:].reset_index(drop=True),
                ticker=ticker
            )
            
            self.evaluator.plot_training_history(history, ticker=ticker)
            
            # Compile results
            results = {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'training_time_seconds': training_time,
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'config': self.config,
                'model_path': str(model_path)
            }
            
            # Save results
            results_path = self.metrics_dir / f'{ticker}_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            self.models[ticker] = model
            self.results[ticker] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error training {ticker}: {str(e)}", exc_info=True)
            raise
    
    def _evaluate_model(self, model, test_df):
        """Helper to evaluate model on test set"""
        required_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'Return', 'SMA_10', 'SMA_30', 'MACD', 'MACD_Signal', 'RSI'
        ]
        test_data = test_df[required_cols].values.astype(np.float32)
        
        predictions = []
        actuals = []

        model.model.eval()
        with torch.no_grad():
            for i in range(len(test_data) - model.seq_len):
                seq = test_data[i:i + model.seq_len]
                
                # Forward pass
                scaled_seq = torch.FloatTensor(model.scaler.transform(seq)).unsqueeze(0).to(model.device)
                
                # --- FIX: Unpack the two model outputs (Regression and Classification) ---
                pred_reg, pred_cls = model.model(scaled_seq)
                
                # We use the Regression prediction (pred_reg) for price calculation
                pred_return = pred_reg.item() if pred_reg.dim() == 0 else pred_reg.squeeze().item()
                
                # Convert return back to price
                last_close = test_data[i + model.seq_len - 1, 3]  # previous close
                pred_price = last_close * (1 + pred_return)
                
                actual_price = test_data[i + model.seq_len, 3]    # true next close
                
                predictions.append(pred_price)
                actuals.append(actual_price)
        
        return np.array(predictions), np.array(actuals)
    
    def train_all_stocks(self):
        """Train models for all stocks in data directory"""
        
        # Get all CSV files
        csv_files = list(self.data_dir.glob('*.csv'))
        tickers = [f.stem for f in csv_files]
        
        logger.info(f"Found {len(tickers)} stocks to train: {', '.join(tickers)}")
        
        all_results = {}
        failed = []
        
        for ticker in tickers:
            try:
                results = self.train_single_stock(ticker)
                all_results[ticker] = results
            except Exception as e:
                logger.error(f"Failed to train {ticker}: {str(e)}", exc_info=True)
                failed.append(ticker)
                continue
        
        # Generate summary report
        self._generate_summary_report(all_results, failed)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Training Complete!")
        logger.info(f"  Successful: {len(all_results)}/{len(tickers)}")
        if failed:
            logger.info(f"  Failed: {', '.join(failed)}")
        logger.info(f"{'='*80}\n")
        
        return all_results
    
    def _generate_summary_report(self, results: dict, failed: list):
        """Generate comprehensive summary report"""
        
        if not results:
            logger.warning("No results to generate summary")
            return
        
        # Create summary DataFrame
        summary_data = []
        for ticker, res in results.items():
            summary_data.append({
                'Ticker': ticker,
                'RMSE': res['metrics']['RMSE'],
                'MAE': res['metrics']['MAE'],
                'MAPE': res['metrics']['MAPE'],
                'R2': res['metrics']['R2'],
                'Dir_Acc_%': res['metrics']['Directional_Accuracy'],
                'Train_Time_s': res['training_time_seconds'],
                'Train_Samples': res['train_samples'],
                'Test_Samples': res['test_samples']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('RMSE')
        
        # Save CSV
        summary_path = self.metrics_dir / f'training_summary_{datetime.now():%Y%m%d_%H%M%S}.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Save text report
        report_path = self.metrics_dir / f'training_report_{datetime.now():%Y%m%d_%H%M%S}.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STOCK FORECASTING MODEL - TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Training Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Total Stocks: {len(results)}\n")
            if failed:
                f.write(f"Failed: {len(failed)} ({', '.join(failed)})\n")
            f.write("\n" + "="*80 + "\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n")
            f.write("STATISTICS\n")
            f.write("="*80 + "\n\n")
            f.write(summary_df.describe().to_string())
            f.write("\n\n" + "="*80 + "\n")
            f.write("BEST MODELS (by RMSE)\n")
            f.write("="*80 + "\n\n")
            f.write(summary_df.head(5).to_string(index=False))
            f.write("\n\n")
        
        logger.info(f"\nSummary saved to {summary_path}")
        logger.info(f"Report saved to {report_path}")
        
        # Print summary to console
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80 + "\n")
        logger.info(summary_df.to_string(index=False))
        logger.info("\n" + "="*80 + "\n")


def main():
    """Main entry point"""
    
    logger.info("="*80)
    logger.info("LSTM-TRANSFORMER STOCK FORECASTING - PRODUCTION TRAINING")
    logger.info("="*80)
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    logger.info("="*80 + "\n")
    
    pipeline = ProductionPipeline(CONFIG)
    
    # Train all stocks
    # all_results = pipeline.train_all_stocks()
    
    result = pipeline.train_single_stock("META")
    
    logger.info("\nTraining pipeline completed successfully!")
    return result


if __name__ == "__main__":
    results = main()