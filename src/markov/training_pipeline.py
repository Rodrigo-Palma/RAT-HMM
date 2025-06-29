import torch
import logging
from src.markov.data_loader import load_data
from src.markov.model_transformer import TimeSeriesTransformer
from src.markov.model_rnn import GRUModel, LSTMModel
from src.markov.utils import get_device, create_dataloaders, save_model
from src.markov.visualizer import plot_losses, plot_backtest, plot_backtest_with_regimes
from src.markov.backtester import run_backtest

# Hyperparameters
SEQ_LENGTH = 10
MODEL_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
OUTPUT_DIM = 1
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def build_model(model_type, input_dim, device):
    if model_type == "transformer":
        model = TimeSeriesTransformer(input_dim, model_dim=MODEL_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM)
    elif model_type == "gru":
        model = GRUModel(input_dim, hidden_dim=MODEL_DIM, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM)
    elif model_type == "lstm":
        model = LSTMModel(input_dim, hidden_dim=MODEL_DIM, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM)
    else:
        raise ValueError(f"❌ Tipo de modelo inválido: {model_type} (esperado: transformer, gru ou lstm)")
    return model.to(device)

def run_training(model_name, model_type, add_regime, add_regime_sequence, target_col, seq_length, target_horizon, 
                 n_regimes=2, seq_length_regime=10, rolling_window=None,
                 save_backtest_plot=True, save_regime_plot=True):

    device = get_device()
    logger.info(f"✅ Usando {device}")
    logger.info(f"🏁 Treinando modelo {model_name} ({model_type})...")

    X_train, y_train, X_val, y_val = load_data(
        seq_length=seq_length,
        target_horizon=target_horizon,
        target_col=target_col,
        add_regime=add_regime,
        add_regime_sequence=add_regime_sequence,
        n_regimes=n_regimes,
        seq_length_regime=seq_length_regime,
        rolling_window=rolling_window
    )

    model = build_model(model_type, input_dim=X_train.shape[-1], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE)

    best_loss = float('inf')
    patience_counter = 0
    train_losses = []

    logger.info(f"🚀 Começando treinamento por {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        logger.info(f"📉 Epoch [{epoch}/{EPOCHS}] Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            save_model(model, path=f"data/results/{model_name}.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("⏹️ Early stopping triggered!")
                break

    logger.info("✅ Treinamento concluído!")

    # Salva gráfico de loss
    plot_losses(
        train_losses,
        save_path=f"data/results/train_losses_{model_name}.png",
        title=f"Training Loss - {model_name}"
    )
    logger.info(f"💾 Gráfico de loss salvo em data/results/train_losses_{model_name}.png")

    # Backtest
    df_backtest = run_backtest(model, X_val, y_val, device, batch_size=BATCH_SIZE)
    df_backtest.to_csv(f"data/results/backtest_{model_name}.csv", index=False)
    logger.info(f"💾 Backtest salvo em data/results/backtest_{model_name}.csv")

    if save_backtest_plot:
        plot_backtest(
            df_backtest,
            save_path=f"data/results/backtest_{model_name}.png",
            title=f"Backtest - {model_name}"
        )
        logger.info(f"✅ Backtest plot salvo em data/results/backtest_{model_name}.png")

    if add_regime and save_regime_plot and "regime" in df_backtest.columns:
        plot_backtest_with_regimes(
            df_backtest,
            regimes=df_backtest["regime"].values,
            save_path=f"data/results/backtest_with_regimes_{model_name}.png",
            title=f"Backtest c/ Regimes - {model_name}"
        )
        logger.info(f"✅ Backtest w/ regimes plot salvo em data/results/backtest_with_regimes_{model_name}.png")
