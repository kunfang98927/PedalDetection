import torch.nn as nn
import torch.nn.functional as F


class PedalDetectionModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        ff_dim,
        num_layers,
        num_classes,
        dropout=0.15,
    ):
        super(PedalDetectionModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ffn_layer = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim)
        )  # For latent representations
        self.classification_output_layer = nn.Linear(
            hidden_dim, num_classes
        )  # For frame-wise classification

    def forward(self, x):
        x = self.input_proj(x)  # Project input to hidden dimension
        x = self.transformer(x)  # Apply Transformer encoder

        latent_repr = F.normalize(
            self.ffn_layer(x), p=2, dim=-1
        )  # Latent representations
        class_logits = self.classification_output_layer(
            latent_repr
        )  # Classification using latent reps

        return class_logits, latent_repr
