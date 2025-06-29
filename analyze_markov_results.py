import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_PATH = "data/analysis/markov_model_optimization_results.csv"

def load_results(path=RESULTS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    print(f"✅ Resultados carregados: {df.shape[0]} linhas")
    return df

def summarize_metrics(df):
    print("\n📊 Estatísticas Descritivas (AIC, BIC, loglikelihood):")
    print(df[["AIC", "BIC", "loglikelihood"]].describe())

def best_models(df, metric="AIC"):
    """
    Retorna os melhores modelos por ativo e horizonte com base em menor AIC ou BIC
    """
    best = df.loc[df.groupby(["ativo", "horizonte"])[metric].idxmin()]
    print(f"\n🏆 Melhores modelos por {metric}:")
    print(best[["ativo", "horizonte", "n_regimes", "use_exog", metric]])
    return best

def plot_heatmaps(df):
    for metric in ["AIC", "BIC", "loglikelihood"]:
        pivot = df.pivot_table(index="n_regimes", columns="use_exog", values=metric, aggfunc="mean")
        plt.figure(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm")
        plt.title(f"{metric} médio por configuração")
        plt.ylabel("n_regimes")
        plt.xlabel("use_exog")
        plt.tight_layout()
        plt.savefig(f"data/analysis/heatmap_{metric}.png")
        plt.close()
        print(f"🖼️ Heatmap salvo: data/analysis/heatmap_{metric}.png")

def main():
    df = load_results()
    summarize_metrics(df)

    best_aic = best_models(df, metric="AIC")
    best_bic = best_models(df, metric="BIC")

    plot_heatmaps(df)

    # Salva os melhores modelos em CSVs separados
    best_aic.to_csv("data/analysis/best_models_AIC.csv", index=False)
    best_bic.to_csv("data/analysis/best_models_BIC.csv", index=False)
    print("✅ CSVs de melhores modelos salvos.")

if __name__ == "__main__":
    main()
