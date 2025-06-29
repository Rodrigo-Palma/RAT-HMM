# src/markov/engineering.py

import pandas as pd
import numpy as np
import yfinance as yf
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Mapeamento dos tickers
TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SPX": "^GSPC",
    "GOLD": "GC=F",
    "DXY": "DX-Y.NYB"
}

def download_market_data(save_path="data/processed/market_data.parquet"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_data = []

    logger.info("⬇️ Baixando dados de mercado via yfinance...")

    for nome, ticker in TICKERS.items():
        logger.info(f"⬇️ {nome} ({ticker}) ...")
        df = yf.download(ticker, start="2010-01-01", auto_adjust=True)
        df = df[["Close"]].rename(columns={"Close": nome})
        all_data.append(df)

    df_final = pd.concat(all_data, axis=1)
    df_final.dropna(inplace=True)

    df_final.to_parquet(save_path)
    logger.info(f"✅ Dados de mercado salvos em {save_path}")

    return df_final

def compute_log_returns(df_prices, horizons=[1, 5, 10]):
    df_ret = {}
    for h in horizons:
        df_h = np.log(df_prices / df_prices.shift(h))
        df_h.columns = [f"{col}_r{h}d" for col in df_h.columns]
        df_ret.update(df_h.to_dict(orient='series'))
    df_ret = pd.DataFrame(df_ret)
    df_ret.dropna(inplace=True)
    return df_ret

def create_supervised_data(df_ret, target_col, seq_length=10, target_horizon=1):
    X = []
    y = []
    for i in range(seq_length, len(df_ret) - target_horizon + 1):
        X.append(df_ret.iloc[i-seq_length:i].values)
        y.append(df_ret.iloc[i + target_horizon - 1][target_col])
    X = np.array(X)
    y = np.array(y)
    return X, y
