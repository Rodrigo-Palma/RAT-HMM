import pandas as pd
import numpy as np
import os
import logging
from src.markov.engineering import compute_log_returns, create_supervised_data, download_market_data
from src.markov.regime_inference import detect_regimes, detect_regimes_sequence
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def load_data(
    seq_length=10,
    target_horizon=1,
    target_col=None,
    add_regime=False,
    add_regime_sequence=False,
    n_regimes=2,
    seq_length_regime=10,
    rolling_window=None,
    test_size=0.2
):
    # 1. Load price data
    data_path = "data/processed/market_data.parquet"
    if not os.path.exists(data_path):
        logger.warning("⚠️  Dados não encontrados. Fazendo download...")
        download_market_data(save_path=data_path)

    logger.info(f"📄 Lendo dados de cache {data_path} ...")
    df_prices = pd.read_parquet(data_path)
    logger.info(f"📄 Carregado {len(df_prices)} linhas, {df_prices.shape[1]} colunas.")

    # 2. Compute log returns
    df_ret = compute_log_returns(df_prices, horizons=[5, 10, 15])

    logger.info(f"📊 Retornos shape: {df_ret.shape}")

    if target_col not in df_ret.columns:
        raise ValueError(f"❌ TARGET_COL '{target_col}' não existe em df_ret.columns: {list(df_ret.columns)}")

    # 3. Create supervised data (X, y)
    X, y = create_supervised_data(
        df_ret,
        target_col=target_col,
        seq_length=seq_length,
        target_horizon=target_horizon
    )

    logger.info(f"📊 Supervised X shape: {X.shape}, y shape: {y.shape}")

    # 4. Optionally add regimes
    if add_regime_sequence:
        logger.info(f"🧭 Detectando regimes como SEQUÊNCIA (n_regimes={n_regimes}, rolling_window={rolling_window}) ...")
        X_regimes = detect_regimes_sequence(
            df_ret,
            n_regimes=n_regimes,
            seq_length=seq_length_regime,
            use_exog=True,
            target_col=target_col,
            rolling_window=rolling_window
        )
        n_samples = min(X.shape[0], X_regimes.shape[0])
        X = X[-n_samples:]
        y = y[-n_samples:]
        X_regimes = X_regimes[-n_samples:]

        X = np.concatenate([X, X_regimes[:, :, np.newaxis]], axis=-1)

        logger.info(f"✅ X com regimes (sequência): {X.shape}")

    elif add_regime:
        logger.info(f"🧭 Detectando regimes (n_regimes={n_regimes}, rolling_window={rolling_window}) como FEATURE fixa ...")
        
        # ⚠️ Correção: Extrai apenas a série de regimes
        regimes_series, _ = detect_regimes(
            df_ret,
            n_regimes=n_regimes,
            use_exog=True,
            target_col=target_col,
            rolling_window=rolling_window
        )
        regimes = regimes_series.values

        regimes = regimes[seq_length + target_horizon - 1:]
        regimes = regimes[-len(y):]

        regimes = regimes.reshape(-1, 1, 1).repeat(seq_length, axis=1)

        X = np.concatenate([X, regimes], axis=-1)

        logger.info(f"✅ X com regime atual: {X.shape}")

    # 5. Split train/test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    logger.info(f"📊 Train: {X_train.shape}, Val: {X_val.shape}")

    return X_train, y_train, X_val, y_val
