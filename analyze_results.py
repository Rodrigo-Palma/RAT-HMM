import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

RESULTS_PATH = "data/results"
SUMMARY_CSV = os.path.join(RESULTS_PATH, "summary_metrics.csv")
COMPARISON_CSV = os.path.join(RESULTS_PATH, "summary_comparison.csv")

def parse_model_filename(filename):
    name = os.path.basename(filename).replace(".csv", "").replace("backtest_", "")
    parts = name.split("_", 3)
    if len(parts) >= 3:
        model_type = parts[0]
        regime_type = parts[1]
        target_col = "_".join(parts[2:])
    else:
        model_type = "unknown"
        regime_type = "unknown"
        target_col = "unknown"
    return name, model_type, regime_type, target_col

print(f"\n✅ Analisando resultados em {RESULTS_PATH}\n")

csv_files = glob.glob(os.path.join(RESULTS_PATH, "backtest_*.csv"))
print(f"📂 Encontrados {len(csv_files)} arquivos de backtest.")

results = []

for file in csv_files:
    model_name, model_type, regime_type, target_col = parse_model_filename(file)

    try:
        df = pd.read_csv(file)
        if not {"y_true", "y_pred"}.issubset(df.columns):
            print(f"⚠️ Ignorando {model_name}: colunas 'y_true' e 'y_pred' não encontradas.")
            continue

        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        if len(y_true) < 5:
            print(f"⚠️ Ignorando {model_name}: menos de 5 observações.")
            continue

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        n_obs = len(y_true)

        std_y = np.std(y_true)
        rmse_norm = rmse / std_y if std_y != 0 else np.nan
        mae_norm = mae / std_y if std_y != 0 else np.nan

        results.append({
            "model_name": model_name,
            "model_type": model_type,
            "regime_type": regime_type,
            "target_col": target_col,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "rmse_norm": rmse_norm,
            "mae_norm": mae_norm,
            "r2": r2,
            "n_obs": n_obs
        })

    except Exception as e:
        print(f"⚠️ Erro ao processar {file}: {e}")

# Summary
df_summary = pd.DataFrame(results)
df_summary["model_full_name"] = df_summary["model_type"] + "_" + df_summary["regime_type"]
df_summary = df_summary.sort_values(by="rmse_norm")
df_summary.to_csv(SUMMARY_CSV, index=False)

print(f"\n✅ Summary salvo em {SUMMARY_CSV} ({len(df_summary)} experimentos)\n")

# Comparação baseline vs regime
df_baseline = df_summary[df_summary["regime_type"] == "baseline"].copy()
df_regime   = df_summary[df_summary["regime_type"] == "regime"].copy()

# Garante que só compara pares válidos
common_keys = pd.merge(
    df_baseline[["model_type", "target_col"]],
    df_regime[["model_type", "target_col"]],
    on=["model_type", "target_col"]
)

df_baseline = df_baseline.merge(common_keys, on=["model_type", "target_col"])
df_regime   = df_regime.merge(common_keys, on=["model_type", "target_col"])

df_compare = pd.merge(
    df_baseline[["model_type", "target_col", "rmse_norm", "mae_norm", "r2"]],
    df_regime[["model_type", "target_col", "rmse_norm", "mae_norm", "r2"]],
    on=["model_type", "target_col"],
    suffixes=("_baseline", "_regime")
)

df_compare["rmse_gain_%"] = 100 * (df_compare["rmse_norm_baseline"] - df_compare["rmse_norm_regime"]) / df_compare["rmse_norm_baseline"]
df_compare["mae_gain_%"]  = 100 * (df_compare["mae_norm_baseline"]  - df_compare["mae_norm_regime"])  / df_compare["mae_norm_baseline"]

df_compare.to_csv(COMPARISON_CSV, index=False)
print(f"✅ Comparação baseline vs regime salva em {COMPARISON_CSV} ({len(df_compare)} pares)\n")

# Gráficos por ativo + modelo
df_compare["ativo"] = df_compare["target_col"].apply(lambda x: x.split("_")[0])
df_compare["horizonte"] = df_compare["target_col"].apply(lambda x: x.split("_")[1])

ativos = df_compare["ativo"].unique()
modelos = df_compare["model_type"].unique()

for ativo in ativos:
    df_plot = df_compare[df_compare["ativo"] == ativo]

    plt.figure(figsize=(12, 6))
    for model in modelos:
        df_model = df_plot[df_plot["model_type"] == model]
        if len(df_model) == 0:
            continue

        plt.bar(df_model["horizonte"] + "_" + model, df_model["rmse_gain_%"], alpha=0.8, label=model)

    plt.axhline(0, color="gray", linestyle="--")
    plt.title(f"Ganho % de RMSE com Regimes - {ativo}")
    plt.ylabel("Ganho % RMSE vs Baseline")
    plt.xlabel("Horizonte + Modelo")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(RESULTS_PATH, f"summary_comparison_{ativo}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Gráfico de comparação salvo em {save_path}")

# Gráfico geral ALL
if not df_compare.empty:
    plt.figure(figsize=(16, 7))
    df_compare["h_label"] = df_compare["ativo"] + "_" + df_compare["horizonte"] + "_" + df_compare["model_type"]
    df_plot_all = df_compare.sort_values(by=["ativo", "horizonte", "model_type"])

    plt.bar(df_plot_all["h_label"], df_plot_all["rmse_gain_%"], color="blue", alpha=0.75)
    plt.axhline(0, color="gray", linestyle="--")
    plt.title("Ganho % de RMSE com Regimes - TODOS OS ATIVOS e MODELOS")
    plt.ylabel("Ganho % RMSE vs Baseline")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path_all = os.path.join(RESULTS_PATH, "summary_comparison_ALL.png")
    plt.savefig(save_path_all)
    plt.close()

    print(f"\n✅ Gráfico geral salvo em {save_path_all}")
else:
    print("⚠️ Nenhum par baseline-regime válido encontrado para plot geral.")

print("\n✅ Análise finalizada!")
