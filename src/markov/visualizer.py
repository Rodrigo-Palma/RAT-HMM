# src/markov/visualizer.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_losses(losses, save_path=None, title="Training Loss por Epoch", early_stop_epoch=None):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', label='Train Loss', linewidth=2)
    if early_stop_epoch:
        plt.axvline(early_stop_epoch, color='r', linestyle='--', label='Early Stop')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Loss plot salvo em {save_path}")
    plt.close()

def plot_backtest(df, save_path=None, title="Backtest: Previsão vs Real"):
    plt.figure(figsize=(14, 6))
    plt.plot(df['y_true'], label='Real', linewidth=2, alpha=0.7)
    plt.plot(df['y_pred'], label='Previsto', linewidth=2, alpha=0.7)
    plt.title(title)
    plt.xlabel("Observações (Val)")
    plt.ylabel("Retorno (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Backtest plot salvo em {save_path}")
    plt.close()

def plot_backtest_with_regimes(df, regimes, save_path=None, title="Backtest com Regimes (highlight regime 1)"):
    plt.figure(figsize=(14, 6))
    plt.plot(df['y_true'], label='Real', linewidth=2)
    plt.plot(df['y_pred'], label='Previsto', linewidth=2)
    
    for i in range(len(regimes)):
        if regimes[i] == 1:
            plt.axvspan(i-0.5, i+0.5, color='red', alpha=0.1)

    plt.title(title)
    plt.xlabel("Observações (Val)")
    plt.ylabel("Retorno (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Backtest + Regimes plot salvo em {save_path}")
    plt.close()
