import copy
import torch.nn as nn
import torch.nn.functional as F
from .transformer import (
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    EncoderLayer,
)


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
        self.positional_encoding = PositionalEncoding(hidden_dim)

        # Use custom transformer encoder layer
        attn = MultiHeadedAttention(num_heads, hidden_dim)
        ff = PositionwiseFeedForward(hidden_dim, ff_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim, copy.deepcopy(attn), copy.deepcopy(ff), dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.classification_output_layer = nn.Linear(
            hidden_dim, num_classes
        )  # For frame-wise classification

    def forward(self, x, src_mask=None):
        x = self.input_proj(x)  # Project input to hidden dimension
        x = self.positional_encoding(x)  # Add positional encoding
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        latent_repr = F.normalize(x, p=2, dim=-1)  # Latent representations
        class_logits = self.classification_output_layer(
            latent_repr
        )  # Classification using latent reps
        return class_logits, latent_repr
