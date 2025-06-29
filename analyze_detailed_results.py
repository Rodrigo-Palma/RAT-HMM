# analyze_detailed_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

RESULTS_PATH = "data/results"
COMPARISON_CSV = os.path.join(RESULTS_PATH, "summary_comparison.csv")

# Carregar dados
df_compare = pd.read_csv(COMPARISON_CSV)

# Normalizar '_wNone' → '_wglobal' para facilitar agrupamentos
df_compare['target_col'] = df_compare['target_col'].str.replace('_wNone', '_wglobal')

# Colunas auxiliares
df_compare['ativo'] = df_compare['target_col'].apply(lambda x: x.split('_')[0])
df_compare['horizonte'] = df_compare['target_col'].apply(lambda x: x.split('_')[1])
df_compare['regime_scope'] = df_compare['target_col'].apply(lambda x: x.split('_w')[-1] if '_w' in x else 'baseline')

# Resumo executivo
print("\n=== 📈 Resumo Executivo ===")
mean_rmse_gain = df_compare['rmse_gain_%'].mean()
mean_mae_gain = df_compare['mae_gain_%'].mean()
print(f"🔸 Média geral de ganho em RMSE: {mean_rmse_gain:.2f}%")
print(f"🔸 Média geral de ganho em MAE: {mean_mae_gain:.2f}%")

# Top e worst
best_cases = df_compare.sort_values('rmse_gain_%', ascending=False).head(5)
worst_cases = df_compare.sort_values('rmse_gain_%').head(5)

print("\n🔹 Melhores casos (mais ganhos com regimes):")
print(best_cases[['model_type', 'target_col', 'rmse_gain_%', 'r2_regime']])

print("\n🔻 Piores casos (regimes prejudicaram):")
print(worst_cases[['model_type', 'target_col', 'rmse_gain_%', 'r2_regime']])

# Salvar como CSVs
df_compare.to_csv(os.path.join(RESULTS_PATH, "summary_comparison_cleaned.csv"), index=False)
best_cases.to_csv(os.path.join(RESULTS_PATH, "top_regimes.csv"), index=False)
worst_cases.to_csv(os.path.join(RESULTS_PATH, "worst_regimes.csv"), index=False)

# Gráficos
os.makedirs("data/analysis", exist_ok=True)
sns.set(style="whitegrid")

# Gráfico geral de RMSE
plt.figure(figsize=(14, 7))
df_compare["h_label"] = df_compare["ativo"] + "_" + df_compare["horizonte"] + "_" + df_compare["model_type"]
df_plot_all = df_compare.sort_values(by=["ativo", "horizonte", "model_type"])
sns.barplot(x='h_label', y='rmse_gain_%', data=df_plot_all, palette="coolwarm")
plt.axhline(0, color='black', linestyle='--')
plt.title("Ganho % RMSE com Regimes - TODOS OS ATIVOS e MODELOS")
plt.ylabel("Ganho % RMSE vs Baseline")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("data/analysis/general_rmse_gain.png")
plt.close()

# Gráficos por ativo
for ativo in df_compare['ativo'].unique():
    df_ativo = df_compare[df_compare['ativo'] == ativo]
    plt.figure(figsize=(10, 6))
    sns.barplot(x='horizonte', y='rmse_gain_%', hue='model_type', data=df_ativo, palette="viridis")
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Ganho % RMSE com Regimes - {ativo}")
    plt.ylabel("Ganho % RMSE vs Baseline")
    plt.xlabel("Horizonte")
    plt.legend(title="Modelo")
    plt.tight_layout()
    plt.savefig(f"data/analysis/rmse_gain_{ativo}.png")
    plt.close()

# Gráfico extra: R² com regime por ativo
for ativo in df_compare['ativo'].unique():
    df_ativo = df_compare[df_compare['ativo'] == ativo]
    plt.figure(figsize=(10, 6))
    sns.barplot(x='horizonte', y='r2_regime', hue='model_type', data=df_ativo, palette="crest")
    plt.title(f"R² com Regimes - {ativo}")
    plt.ylabel("R² com regime")
    plt.xlabel("Horizonte")
    plt.legend(title="Modelo")
    plt.tight_layout()
    plt.savefig(f"data/analysis/r2_regime_{ativo}.png")
    plt.close()

# Interpretação automática
print("\n=== 🧠 Interpretação Automatizada ===")
for _, row in df_compare.iterrows():
    msg = f"{row['model_type']} - {row['target_col']}: (RMSE {row['rmse_gain_%']:.2f}%)"
    if row['rmse_gain_%'] > 5:
        print(f"✅ Regimes trouxeram forte ganho para {msg}.")
    elif row['rmse_gain_%'] < -5:
        print(f"⚠️ Regimes prejudicaram significativamente para {msg}.")
    else:
        print(f"🔄 Regimes tiveram impacto marginal em {msg}.")

print("\n✅ Análise completa salva em data/analysis.")
