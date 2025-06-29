# analyze_model_ranking.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

RESULTS_PATH = "data/results"
ANALYSIS_PATH = "data/analysis"
SUMMARY_CSV = os.path.join(RESULTS_PATH, "summary_metrics.csv")

# Cria pasta se não existir
os.makedirs(ANALYSIS_PATH, exist_ok=True)

print(f"\n✅ Analisando ranking de modelos em {SUMMARY_CSV}\n")

# Carrega dados
df = pd.read_csv(SUMMARY_CSV)

# ⚠️ Filtra linhas com target_col válida
df_valid = df[df['target_col'].str.contains("_")].copy()

# Deriva auxiliar para horizonte e ativo
df_valid['ativo'] = df_valid['target_col'].apply(lambda x: x.split('_')[0])
df_valid['horizonte'] = df_valid['target_col'].apply(lambda x: x.split('_')[1])

# Resumo geral por modelo
df_ranking = df_valid.groupby('model_type').agg({
    'rmse_norm': 'mean',
    'mae_norm': 'mean',
    'r2': 'mean',
    'target_col': 'count'
}).rename(columns={'target_col': 'n_experimentos'}).reset_index()

df_ranking = df_ranking.sort_values('rmse_norm')

print("\n=== 🏆 Ranking Geral de Modelos ===")
print(df_ranking)

# Salva CSV
df_ranking.to_csv(os.path.join(ANALYSIS_PATH, "model_ranking.csv"), index=False)
print(f"\n✅ Ranking salvo em {ANALYSIS_PATH}/model_ranking.csv")

# Gráfico geral
plt.figure(figsize=(10, 6))
sns.barplot(x='model_type', y='rmse_norm', data=df_ranking, palette='Set2')
plt.title("Ranking de Modelos - Média RMSE Normalizado")
plt.ylabel("Média RMSE Normalizado")
plt.xlabel("Modelo")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_PATH, "model_ranking_rmse.png"))
plt.close()
print(f"✅ Gráfico de ranking salvo em {ANALYSIS_PATH}/model_ranking_rmse.png")

# Gráfico por horizonte
plt.figure(figsize=(12, 6))
sns.barplot(x='horizonte', y='rmse_norm', hue='model_type', data=df_valid, palette='tab10')
plt.title("Ranking de Modelos por Horizonte")
plt.ylabel("Média RMSE Normalizado")
plt.xlabel("Horizonte")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_PATH, "model_ranking_rmse_by_horizon.png"))
plt.close()
print(f"✅ Gráfico por horizonte salvo em {ANALYSIS_PATH}/model_ranking_rmse_by_horizon.png")

# Gráfico por ativo
plt.figure(figsize=(12, 6))
sns.barplot(x='ativo', y='rmse_norm', hue='model_type', data=df_valid, palette='tab10')
plt.title("Ranking de Modelos por Ativo")
plt.ylabel("Média RMSE Normalizado")
plt.xlabel("Ativo")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_PATH, "model_ranking_rmse_by_ativo.png"))
plt.close()
print(f"✅ Gráfico por ativo salvo em {ANALYSIS_PATH}/model_ranking_rmse_by_ativo.png")

print("\n✅ Análise de ranking finalizada!")
