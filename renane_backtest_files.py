import os
import re

RESULTS_DIR = "data/results"

def rename_regime_files():
    files = os.listdir(RESULTS_DIR)
    renamed = []

    for filename in files:
        if filename.startswith("backtest_") and "_regime_" in filename:
            # Ex: backtest_transformer_regime_BTC_r5d_r4_wNone.csv
            name_part = filename.replace("backtest_", "")
            parts = name_part.split("_")

            if len(parts) < 4:
                continue  # formato inesperado

            model_type = parts[0]
            regime_type = parts[1]
            asset = parts[2]
            horizon = parts[3]

            target_col = f"{asset}_{horizon}"
            extension = filename.split(".")[-1]

            new_name = f"backtest_{model_type}_regime_{target_col}.{extension}"
            old_path = os.path.join(RESULTS_DIR, filename)
            new_path = os.path.join(RESULTS_DIR, new_name)

            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                renamed.append((filename, new_name))

    if renamed:
        print("✅ Arquivos renomeados:")
        for old, new in renamed:
            print(f"  - {old} → {new}")
    else:
        print("ℹ️ Nenhum arquivo foi renomeado (ou já estavam corretos).")

if __name__ == "__main__":
    rename_regime_files()
