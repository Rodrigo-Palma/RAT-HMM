import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

RESULTS_PATH = "data/results"
COMPARISON_CSV = os.path.join(RESULTS_PATH, "summary_comparison.csv")

df_compare = pd.read_csv(COMPARISON_CSV)
df_compare = df_compare[df_compare['model_type'] == 'transformer']

if df_compare.empty:
    print("Nenhum resultado para modelo transformer.")
else:
    df_compare['target_col'] = df_compare['target_col'].str.replace('_wNone', '_wglobal')
    df_compare['ativo'] = df_compare['target_col'].apply(lambda x: x.split('_')[0])
    df_compare['horizonte'] = df_compare['target_col'].apply(lambda x: x.split('_')[1])

    print("\n=== 📈 Resumo Transformer ===")
    print(df_compare[['model_type', 'target_col', 'rmse_gain_%', 'r2_regime']])

    df_compare.to_csv(os.path.join(RESULTS_PATH, "summary_comparison_transformer_cleaned.csv"), index=False)

    os.makedirs("data/analysis", exist_ok=True)
    plt.figure(figsize=(14, 7))
    df_compare["h_label"] = df_compare["ativo"] + "_" + df_compare["horizonte"]
    sns.barplot(x='h_label', y='rmse_gain_%', data=df_compare, palette="coolwarm")
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Transformer - Ganho % RMSE com Regimes")
    plt.ylabel("Ganho % RMSE vs Baseline")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("data/analysis/general_rmse_gain_transformer.png")
    plt.close()