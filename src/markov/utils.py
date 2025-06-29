import torch
from torch.utils.data import DataLoader, TensorDataset
import os

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Usando mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Usando cuda")
    else:
        device = torch.device("cpu")
        print("✅ Usando cpu")
    return device

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def save_model(model, path="data/results/transformer_model.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾 Modelo salvo em {path}")

def load_model(model, path="data/results/transformer_model.pt", device=None):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"📦 Modelo carregado de {path}")
    return model
