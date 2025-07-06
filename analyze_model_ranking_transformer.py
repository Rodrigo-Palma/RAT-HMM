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

os.makedirs(ANALYSIS_PATH, exist_ok=True)

df = pd.read_csv(SUMMARY_CSV)
df = df[df['model_type'] == 'transformer']

if df.empty:
    print("Nenhum resultado para modelo transformer.")
else:
    df_valid = df[df['target_col'].str.contains("_")].copy()
    df_valid['ativo'] = df_valid['target_col'].apply(lambda x: x.split('_')[0])
    df_valid['horizonte'] = df_valid['target_col'].apply(lambda x: x.split('_')[1])

    df_ranking = df_valid.groupby('model_type').agg({
        'rmse_norm': 'mean',
        'mae_norm': 'mean',
        'r2': 'mean',
        'target_col': 'count'
    }).rename(columns={'target_col': 'n_experimentos'}).reset_index()

    print("\n=== 🏆 Ranking Transformer ===")
    print(df_ranking)

    df_ranking.to_csv(os.path.join(ANALYSIS_PATH, "model_ranking_transformer.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='model_type', y='rmse_norm', data=df_ranking, palette='Set2')
    plt.title("Transformer - Média RMSE Normalizado")
    plt.ylabel("Média RMSE Normalizado")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_PATH, "model_ranking_rmse_transformer.png"))
    plt.close()