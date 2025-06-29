# run_optimizer.py

from src.markov.markov_model_optimizer import main

if __name__ == "__main__":
    print("\n✅ Iniciando otimização dos modelos de Markov...\n")
    main()
    print("\n✅ Otimização finalizada!\nResultados em: data/analysis/markov_model_optimization_results.csv\n")
