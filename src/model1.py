import copy
import torch.nn as nn
import torch.nn.functional as F
from .transformer import (
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    EncoderLayer,
)
from .cnn_block import CNNBlock


class PedalDetectionModelwithCNN(nn.Module):
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
        super(PedalDetectionModelwithCNN, self).__init__()

        # 1. CNN Block: [batch_size, seq_len, freq_dim] -> [batch_size, seq_len, hidden_dim]
        self.cnn = CNNBlock(hidden_dim=hidden_dim, dropout=dropout)

        # 2️. Transformer Encoder
        self.positional_encoding = PositionalEncoding(hidden_dim)
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

        # 3. Classification Heads
        self.low_res_pedal_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.room_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )  # 4 classes for room type: 0 for real, 1, 2, 3 for synthetic rooms

        self.pedal_value_output_layer = nn.Linear(hidden_dim, num_classes)
        self.pedal_onset_output_layer = nn.Linear(hidden_dim, num_classes)
        self.pedal_offset_output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, loss_mask=None, src_mask=None):

        # Apply CNN layers
        x = self.cnn(x)  # [batch, seq_len, hidden_dim]

        # Transformer Encoder
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        latent_repr = F.normalize(x, p=2, dim=-1)

        # Heads for Pedal Detection
        p_on_logits = self.pedal_onset_output_layer(latent_repr)
        p_off_logits = self.pedal_offset_output_layer(latent_repr)
        p_v_logits = self.pedal_value_output_layer(latent_repr)

        # Mean Latent Representation for Global Prediction
        mean_latent_repr = x.sum(dim=1) / x.shape[1]

        # Global Pedal & Room Predictions
        room_logits = self.room_head(mean_latent_repr)
        low_res_p_v_logits = self.low_res_pedal_value_head(mean_latent_repr)

        return (
            low_res_p_v_logits,
            p_v_logits,
            p_on_logits,
            p_off_logits,
            room_logits,
            latent_repr,
        )
