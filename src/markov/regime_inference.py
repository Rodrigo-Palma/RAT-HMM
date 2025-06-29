import pandas as pd
import numpy as np
import logging
import warnings
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Silenciar warnings do Statsmodels
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def detect_regimes(
    df_returns,
    n_regimes=2,
    use_exog=True,
    target_col=None,
    rolling_window=None,
    max_iter=1000
):
    """
    Ajusta um modelo de regime de Markov com `n_regimes` sobre os retornos.
    Se use_exog=True, usa colunas exógenas no modelo.
    Retorna vetor de regimes por data e o objeto `result` com métricas.
    """
    logger.info(f"🧭 Ajustando Markov Switching (regimes={n_regimes}, use_exog={use_exog}, rolling_window={rolling_window})...")

    if target_col is None or target_col not in df_returns.columns:
        raise ValueError(f"❌ target_col inválido ou ausente: {target_col}")

    # Seleciona a série alvo e remove NaNs
    target = df_returns[target_col].dropna()

    # Seleciona exógenas alinhadas à série alvo
    if use_exog:
        X_exog = df_returns.drop(columns=[target_col]).loc[target.index]
    else:
        X_exog = None

    # Cria e ajusta o modelo
    model = MarkovRegression(
        target,
        exog=X_exog,
        k_regimes=n_regimes,
        trend="c",
        switching_variance=True
    )

    try:
        result = model.fit(disp=False, maxiter=max_iter)
    except Exception as e:
        logger.warning(f"⚠️ Falha com 'lbfgs'. Tentando 'nm'... ({e})")
        result = model.fit(disp=False, maxiter=max_iter, method="nm")

    # Extrai a série de regimes mais prováveis
    regimes_series = pd.Series(
        np.argmax(result.smoothed_marginal_probabilities, axis=1),
        index=target.index
    )

    logger.info(f"✅ Regimes detectados: {regimes_series.value_counts().to_dict()}")

    return regimes_series, result


def detect_regimes_sequence(
    df_returns,
    n_regimes=3,
    seq_length=10,
    use_exog=True,
    target_col=None,
    rolling_window=None
):
    """
    Detecta regimes e gera sequência de regimes por janela (seq_length).
    Se rolling_window for definido, usa rolling estimation.
    Retorna: np.array (n_amostras, seq_length)
    """
    regimes, _ = detect_regimes(
        df_returns=df_returns,
        n_regimes=n_regimes,
        use_exog=use_exog,
        target_col=target_col,
        rolling_window=rolling_window
    )

    regimes_seq = []
    for i in range(seq_length, len(regimes)):
        seq = regimes.iloc[i - seq_length:i].values
        regimes_seq.append(seq)

    regimes_seq = np.array(regimes_seq).astype(np.float32)

    logger.info(f"✅ Regimes como sequência shape: {regimes_seq.shape}")

    return regimes_seq


def save_regime_labels(regimes, path="data/processed/regimes_labels.parquet"):
    """
    Salva os rótulos de regime em formato parquet.
    """
    df = pd.DataFrame({"regime": regimes})
    df.to_parquet(path)
    logger.info(f"💾 Regimes salvos em {path}")
