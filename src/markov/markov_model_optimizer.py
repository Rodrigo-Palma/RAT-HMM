import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from src.markov.regime_inference import detect_regimes
from src.markov.data_loader import load_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def optimize_markov_model(df_returns, target_col, ativo, horizonte, n_regimes_list=[3, 4, 5, 6], use_exog_list=[True, False]):
    results = []

    for n_regimes in n_regimes_list:
        for use_exog in use_exog_list:

            if target_col not in df_returns.columns:
                logger.error(f"❌ TARGET_COL '{target_col}' não encontrado em df_returns.columns: {list(df_returns.columns)}")
                continue

            logger.info(f"📄 Carregando dados para {target_col}...")
            X_train, y_train, X_val, y_val = load_data(
                seq_length=10,
                target_horizon=1,
                target_col=target_col,
                add_regime=False,
                n_regimes=n_regimes
            )

            logger.info(f"🧭 Ajustando modelo Markov com {n_regimes} regimes (exógenas={use_exog})...")
            regimes_series, result = detect_regimes(
                df_returns,
                n_regimes=n_regimes,
                use_exog=use_exog,
                target_col=target_col
            )

            bic = result.bic
            aic = result.aic
            loglik = result.llf

            results.append({
                "ativo": ativo,
                "horizonte": horizonte,
                "n_regimes": n_regimes,
                "use_exog": use_exog,
                "BIC": bic,
                "AIC": aic,
                "loglikelihood": loglik,
                "regimes_detectados": regimes_series.value_counts().to_dict()
            })

            logger.info(f"✅ Modelo com {n_regimes} regimes e exógenas={use_exog} processado.")

    return pd.DataFrame(results)

def plot_regimes_comparison(results_df):
    plt.figure(figsize=(14, 8))
    for n_regimes in results_df["n_regimes"].unique():
        df_plot = results_df[results_df["n_regimes"] == n_regimes]
        plt.plot(df_plot["use_exog"], df_plot["AIC"], label=f"AIC - {n_regimes} regimes")
        plt.plot(df_plot["use_exog"], df_plot["BIC"], label=f"BIC - {n_regimes} regimes")
        plt.plot(df_plot["use_exog"], df_plot["loglikelihood"], label=f"Loglik - {n_regimes} regimes")

    plt.legend()
    plt.title("Comparação de AIC, BIC e Loglikelihood para diferentes modelos de Markov")
    plt.xlabel("Uso de Exógenas")
    plt.ylabel("Métricas de Ajuste")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/analysis/markov_model_comparison.png")
    plt.close()

def main():
    logger.info("✅ Iniciando otimização dos modelos de Markov...")

    ativos = ["BTC", "ETH", "SPX", "GOLD", "DXY"]
    horizontes = ["r5d", "r10d", "r15d"]

    TICKER_MAP = {
        "BTC": "('BTC', 'BTC-USD')",
        "ETH": "('ETH', 'ETH-USD')",
        "SPX": "('SPX', '^GSPC')",
        "GOLD": "('GOLD', 'GC=F')",
        "DXY": "('DXY', 'DX-Y.NYB')"
    }

    all_results = []

    data_path = "data/processed/market_data.parquet"
    if not os.path.exists(data_path):
        from src.markov.engineering import download_market_data
        logger.warning("⚠️  Dados não encontrados. Fazendo download...")
        download_market_data(save_path=data_path)

    df_prices = pd.read_parquet(data_path)
    from src.markov.engineering import compute_log_returns
    df_returns = compute_log_returns(df_prices, horizons=[5, 10, 15])
    logger.info(f"📊 Retornos shape: {df_returns.shape}")

    for ativo in ativos:
        for horizonte in horizontes:

            target_col = f"{TICKER_MAP[ativo]}_{horizonte}"
            logger.info(f"=== Iniciando otimização de Markov para {ativo}_{horizonte} ===")

            results = optimize_markov_model(
                df_returns, target_col=target_col, ativo=ativo, horizonte=horizonte
            )

            if not results.empty:
                all_results.append(results)

    if all_results:
        final_results_df = pd.concat(all_results)
        final_results_df.to_csv("data/analysis/markov_model_optimization_results.csv", index=False)
        logger.info("✅ Resultados de otimização salvos em data/analysis/markov_model_optimization_results.csv")

        plot_regimes_comparison(final_results_df)
        logger.info("✅ Gráfico de comparação de modelos de Markov gerado.")
    else:
        logger.warning("⚠️  Nenhum resultado gerado. Verifique os parâmetros de entrada.")

if __name__ == "__main__":
    main()
