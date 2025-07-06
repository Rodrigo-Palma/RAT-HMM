import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

RESULTS_PATH = "data/results"
SUMMARY_CSV = os.path.join(RESULTS_PATH, "summary_metrics.csv")
COMPARISON_CSV = os.path.join(RESULTS_PATH, "summary_comparison.csv")

df_summary = pd.read_csv(SUMMARY_CSV)
df_summary = df_summary[df_summary['model_type'] == 'transformer']

if df_summary.empty:
    print("Nenhum resultado para modelo transformer.")
else:
    df_summary.to_csv(os.path.join(RESULTS_PATH, "summary_metrics_transformer.csv"), index=False)
    print(f"Resumo transformer salvo em {RESULTS_PATH}/summary_metrics_transformer.csv")

    # Se quiser comparar baseline vs regime só para transformer:
    if os.path.exists(COMPARISON_CSV):
        df_compare = pd.read_csv(COMPARISON_CSV)
        df_compare = df_compare[df_compare['model_type'] == 'transformer']
        df_compare.to_csv(os.path.join(RESULTS_PATH, "summary_comparison_transformer.csv"), index=False)
        print(f"Comparação transformer salva em {RESULTS_PATH}/summary_comparison_transformer.csv")