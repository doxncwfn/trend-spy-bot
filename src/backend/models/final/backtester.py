import pandas as pd
import numpy as np
from scipy.stats import norm
from config import CONFIG

class ComprehensiveBacktester:
    def __init__(self, signal_df):
        self.signals = signal_df.copy()
        self.cost_bps = CONFIG['transaction_cost_bps']
        self.top_k = CONFIG['top_k']
        
    def run_strategy(self, strategy_type='long_only'):
        # Pivot Data
        preds = self.signals.pivot(index='Date', columns='Ticker', values='Pred_Mu')
        sigmas = self.signals.pivot(index='Date', columns='Ticker', values='Pred_Sigma')
        returns = self.signals.pivot(index='Date', columns='Ticker', values='Actual_Return')
        
        # Trade Schedule (Purged)
        trade_dates = preds.index[::CONFIG['pred_horizon']]
        
        weights = pd.DataFrame(0.0, index=trade_dates, columns=preds.columns)
        
        for date in trade_dates:
            row_p = preds.loc[date]
            row_s = sigmas.loc[date]
            
            # Confidence Filter
            probs = 1 - norm.cdf(0, loc=row_p, scale=row_s)
            valid_mask = probs > CONFIG['confidence_threshold']
            
            valid_preds = row_p[valid_mask]
            if len(valid_preds) < self.top_k: continue 
            
            ranked = valid_preds.sort_values(ascending=False)
            
            if strategy_type == 'long_only':
                longs = ranked.head(self.top_k).index
                weights.loc[date, longs] = 1.0 / len(longs)
                
            elif strategy_type == 'long_short':
                longs = ranked.head(self.top_k).index
                shorts = ranked.tail(self.top_k).index
                if len(shorts) >= self.top_k:
                    weights.loc[date, longs] = 0.5 / len(longs)
                    weights.loc[date, shorts] = -0.5 / len(shorts)
        
        # Returns & Turnover
        rebalanced_returns = returns.loc[trade_dates]
        gross_ret = (weights * rebalanced_returns).sum(axis=1)
        
        # Turnover: Sum of absolute weight changes
        weight_delta = weights.diff().fillna(weights).abs().sum(axis=1)
        costs = weight_delta * (self.cost_bps / 10000.0)
        
        net_ret = gross_ret - costs
        equity = (1 + net_ret).cumprod() * CONFIG['initial_capital']
        
        return equity, net_ret, weight_delta

    def calculate_metrics(self, equity, net_ret, turnover):
        ann = 252 / CONFIG['pred_horizon']
        
        total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
        cagr = (equity.iloc[-1] / equity.iloc[0])**(ann/len(net_ret)) - 1
        vol = net_ret.std() * np.sqrt(ann)
        sharpe = (net_ret.mean() / net_ret.std()) * np.sqrt(ann) if vol > 0 else 0
        
        neg_ret = net_ret[net_ret < 0]
        sortino = (net_ret.mean() * ann) / (neg_ret.std() * np.sqrt(ann) + 1e-9)
        
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return {
            "Total Return": f"{total_ret:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Sharpe": f"{sharpe:.2f}",
            "Sortino": f"{sortino:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Calmar": f"{calmar:.2f}",
            "Avg Turnover": f"{turnover.mean():.2%}"
        }