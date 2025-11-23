from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (15, 10)

import logging
logger = logging.getLogger(__name__)

class ModelEvaluator:    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:        
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        }
        
        # Directional accuracy
        if len(y_true) > 1:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            metrics['Directional_Accuracy'] = np.mean(direction_true == direction_pred) * 100
        else:
            metrics['Directional_Accuracy'] = 0.0
        
        return metrics
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.Series = None,
        ticker: str = "Stock"
    ):
        """Plot predictions vs actual"""
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Time series
        if dates is not None:
            x = dates.values
        else:
            x = np.arange(len(y_true))
        
        axes[0, 0].plot(x, y_true, label='Actual', linewidth=2, alpha=0.8)
        axes[0, 0].plot(x, y_pred, label='Predicted', linewidth=2, alpha=0.8)
        axes[0, 0].set_title(f'{ticker} - Price Predictions', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 1].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Actual Price ($)')
        axes[0, 1].set_ylabel('Predicted Price ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_true - y_pred
        axes[1, 0].scatter(x, residuals, alpha=0.5, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Prediction Residuals', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Residual ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Residual ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{ticker}_predictions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Predictions plot saved: {save_path}")
    
    def plot_training_history(self, history: dict, ticker: str = "Stock"):        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title(f'{ticker} - Training Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metrics
        axes[1].plot(epochs, history['val_rmse'], label='RMSE', linewidth=2)
        axes[1].plot(epochs, history['val_mae'], label='MAE', linewidth=2)
        axes[1].set_title(f'{ticker} - Validation Metrics', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{ticker}_training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history saved: {save_path}")