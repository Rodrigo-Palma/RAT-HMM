from src.markov.training_pipeline import run_training

if __name__ == "__main__":

    print("\n✅ Usando mps")

    ativos = [
        "BTC",
        "ETH",
        "SPX",
        "GOLD",
        "DXY"
    ]

    TICKER_MAP = {
        "BTC": "('BTC', 'BTC-USD')",
        "ETH": "('ETH', 'ETH-USD')",
        "SPX": "('SPX', '^GSPC')",
        "GOLD": "('GOLD', 'GC=F')",
        "DXY": "('DXY', 'DX-Y.NYB')"
    }

    horizontes = ["r5d", "r10d", "r15d"]

    modelos = [
        "transformer",
        "gru",
        "lstm"
    ]

    # Hyperparams regime com base nos melhores resultados da análise
    n_regimes = 6
    seq_length_regime = 10
    rolling_window = None   # global fit foi o melhor

    for model_type in modelos:
        for ativo in ativos:
            for h in horizontes:

                target_col = f"{TICKER_MAP[ativo]}_{h}"

                model_name_baseline = f"{model_type}_baseline_{ativo}_{h}"
                model_name_regime    = f"{model_type}_regime_{ativo}_{h}_r{n_regimes}_w{rolling_window}"

                print(f"\n=== 🚀 Rodando BASELINE [{model_type}] para {target_col} ===\n")
                run_training(
                    model_name=model_name_baseline,
                    model_type=model_type,
                    add_regime=False,
                    add_regime_sequence=False,
                    n_regimes=1,
                    target_col=target_col,
                    seq_length=10,
                    target_horizon=1,
                    save_backtest_plot=True,
                    save_regime_plot=False
                )

                print(f"\n=== 🚀 Rodando REGIME [{model_type}] para {target_col} ===")
                print(f"    EXPERIMENT_CONFIG: n_regimes={n_regimes}, seq_length_regime={seq_length_regime}, rolling_window={rolling_window}\n")

                run_training(
                    model_name=model_name_regime,
                    model_type=model_type,
                    add_regime=True,
                    add_regime_sequence=False,
                    n_regimes=n_regimes,
                    target_col=target_col,
                    seq_length=10,
                    seq_length_regime=seq_length_regime,
                    rolling_window=rolling_window,
                    target_horizon=1,
                    save_backtest_plot=True,
                    save_regime_plot=True
                )
