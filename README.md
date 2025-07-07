🧠📈 RAT-HMM: Regime-Aware Transformer Hidden Markov Model
Previsão Financeira Ciente de Regimes com Transformers e Modelos de Markov Ocultos

✨ Visão Geral
Este repositório implementa um modelo híbrido para previsão de séries temporais financeiras não estacionárias, combinando o poder dos Transformers com a inferência sequencial de Modelos Ocultos de Markov (HMM). A proposta, chamada RAT-HMM (Regime-Aware Transformer-HMM), é aplicada à previsão de risco/retorno de múltiplos ativos financeiros: Bitcoin (BTC), Ethereum (ETH), S&P 500 (SPX) e Ouro (GOLD), com horizontes de 5, 10 e 15 dias.

📄 Artigo completo: “Previsão Financeira Ciente de Regimes: Um Modelo Híbrido Markov-Transformer para Predição Multi-Ativos de Risco e Retorno” (2025)

🧩 Arquitetura do Modelo
scss

Dados de Mercado → Feature Engineering → HMM (Inferência de Regimes) → Transformer Condicional → Previsão (r5d, r10d, r15d)
O HMM segmenta a série em regimes latentes (voláteis/estáveis/etc).

Os regimes inferidos são incorporados como atributos ao Transformer.

O modelo é treinado para prever retornos acumulados futuros.

📊 Resultados Empíricos
✅ Os melhores ganhos de performance com o RAT-HMM foram observados em:

Ativo	Horizonte	Modelo	Ganho RMSE (%)	R²
GOLD	r15d	Transformer	+30.38%	0.78
GOLD	r10d	Transformer	+16.73%	0.66
ETH	r10d	Transformer	+15.83%	0.81

📉 Em alguns casos, regimes introduziram ruído. Consulte data/results/*.csv para detalhes completos.

📌 Figura 2 demonstra graficamente o melhor cenário (GOLD_r15d).

📁 Estrutura do Repositório

bash
📦 RAT-HMM/
├── data/                   # Dados
│   ├── analysis/           # Resultados de análise final (ex: model_ranking.csv)
│   ├── processed/          # Dados prontos para treino/teste
│   └── results/            # Métricas, comparações, backtests, top_regimes.csv etc.
├── src/
│   └── markov/             # Implementação dos modelos e pipeline
│       ├── model_transformer.py         # Arquitetura Transformer
│       ├── model_rnn.py                 # LSTM/GRU
│       ├── regime_inference.py          # Inferência com HMM
│       ├── engineering.py               # Feature engineering
│       ├── training_pipeline.py         # Pipeline de treinamento
│       ├── backtester.py                # Funções de backtest
│       └── visualizer.py                # Geração de gráficos e visualizações
├── *.py                   # Scripts de análise, avaliação e comparação de modelos
├── graficos.ipynb         # Notebook para visualizações interativas
├── Artigo-RAT-HMM.pdf     # Versão PDF do artigo científico
├── create_env.txt         # Dependências do projeto (use com `pip install -r`)
├── LICENSE
└── README.md              # Este arquivo


🚀 Como Reproduzir os Experimentos
1. Clonar o Repositório
bash

git clone https://github.com/Rodrigo-Palma/RAT-HMM.git
cd RAT-HMM
2. Criar Ambiente com Conda ou Virtualenv
bash

conda create -n markov-env python=3.10
conda activate markov-env
pip install -r requirements.txt

3. Rodar os Experimentos
bash

python run_experiments.py
Configure os ativos, horizontes e modelos diretamente no script run_experiments.py.

🧠 Modelos Implementados
✅ Transformer (com e sem regimes)
✅ LSTM
✅ GRU
✅ HMM (via hmmlearn) para inferência de regimes ocultos

📈 Visualização
Gráficos e análises são gerados via visualizer.py, incluindo:
Comparações de RMSE com/sem regimes
Ranking de modelos por ativo e horizonte
R² por regime

Previsões reais vs preditas

🔍 Requisitos
Python 3.10+
PyTorch
scikit-learn
hmmlearn
yfinance
seaborn / matplotlib / pandas

Instale via:

pip install -r requirements.txt

🤝 Contribuição
Este projeto está aberto a colaborações acadêmicas e extensões futuras. Sinta-se à vontade para abrir um Pull Request ou Issue.

📜 Licença
Distribuído sob a Apache License. Veja LICENSE para mais detalhes.

