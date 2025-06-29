# setup_project.py
import os

dirs = [
    "data/raw",
    "data/processed",
    "data/regimes_cache",
    "data/results",
    "notebooks",
    "src/markov",
]

files = {
    "src/markov/__init__.py": "",
    "src/markov/data_loader.py": "",
    "src/markov/regime_inference.py": "",
    "src/markov/model_transformer.py": "",
    "src/markov/baselines.py": "",
    "src/markov/training_pipeline.py": "",
    "src/markov/utils.py": "",
    "run_training.py": "",
    "README.md": "# Markov Project\n",
    ".gitignore": ".DS_Store\n__pycache__\n*.pyc\n*.pkl\n*.npz\n*.npy\n",
}

# Cria diretórios
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"📁 Criado diretório: {d}")

# Cria arquivos
for fpath, content in files.items():
    with open(fpath, "w") as f:
        f.write(content)
    print(f"📄 Criado arquivo: {fpath}")

print("\n✅ Projeto criado com sucesso!")
print("👉 Para começar:")
print("    python -m venv markov-env")
print("    source markov-env/bin/activate")
print("    pip install -r requirements.txt")
print("    python run_training.py")
