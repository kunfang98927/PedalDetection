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
        # Classification output layer
        self.classification_output_layer = nn.Linear(
            hidden_dim, num_classes
        )
        # Room classification head
        self.room_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Output logits for 0 or 1
        )

    def forward(self, x, loss_mask=None, src_mask=None):
        x = self.input_proj(x)  # Project input to hidden dimension
        x = self.positional_encoding(x)  # Add positional encoding
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        latent_repr = F.normalize(x, p=2, dim=-1)  # Latent representations, (bs, seq_len, hidden_dim)
        class_logits = self.classification_output_layer(
            latent_repr
        )  # Classification using latent reps, (bs, seq_len, num_classes)

        # calculate mean of latent repr, for all non-masked frames
        latent_repr_ = latent_repr.clone()
        if loss_mask is not None:
            latent_repr_[loss_mask == 0] = 0
        mean_latent_repr = latent_repr_.sum(dim=1) / loss_mask.sum(dim=1).unsqueeze(-1)
        
        # Room classification logits
        room_logits = self.room_head(mean_latent_repr)  # (bs, 2)

        return class_logits, room_logits, latent_repr
