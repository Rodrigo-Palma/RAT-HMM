# src/markov/backtester.py

import torch
import pandas as pd
from src.markov.utils import create_dataloaders
from src.markov.visualizer import plot_backtest

def run_backtest(model, X_val, y_val, device, batch_size=32):
    model.eval()
    val_loader = create_dataloaders(X_val, y_val, X_val, y_val, batch_size=batch_size)[1]

    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).squeeze()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    df_backtest = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

    # Save CSV
    df_backtest.to_csv("data/results/backtest_baseline.csv", index=False)
    print("💾 Backtest salvo em data/results/backtest_baseline.csv")

    # Save plot
    plot_backtest(df_backtest, save_path="data/results/backtest_baseline.png")

    return df_backtest
