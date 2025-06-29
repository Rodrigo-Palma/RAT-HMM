import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TimeSeriesTransformer, self).__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = x.to(self.input_proj.weight.device)
        x_proj = self.input_proj(x)
        x_encoded = self.transformer(x_proj)
        out = self.fc_out(x_encoded[:, -1, :])
        return out
