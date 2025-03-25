import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import (
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    EncoderLayer,
)
from .cnn_block import CNNBlock


class PedalDetectionModelContrastive(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        ff_dim,
        num_layers,
        num_classes,
        dropout=0.15,
        predict_room=False,
        predict_pedal_onset=False,
        predict_pedal_offset=False,
        predict_global_pedal=True,
    ):
        super().__init__()

        # CNN Block: [batch_size, seq_len, freq_dim] -> [batch_size, seq_len, hidden_dim]
        self.cnn = CNNBlock(hidden_dim=hidden_dim, dropout=dropout)

        # Transformer Encoder
        self.positional_encoding = PositionalEncoding(hidden_dim)
        attn = MultiHeadedAttention(num_heads, hidden_dim)
        ff = PositionwiseFeedForward(hidden_dim, ff_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, copy.deepcopy(attn), copy.deepcopy(ff), dropout)
                for _ in range(num_layers)
            ]
        )

        # Attribute Prediction MLPs
        self.pedal_value_output_layer = self._build_mlp(hidden_dim, 1)
        
        # Optional MLPs for additional predictions
        if predict_global_pedal:
            self.global_pedal_value_head = self._build_mlp(hidden_dim, 1)
        if predict_pedal_onset:
            self.pedal_onset_output_layer = self._build_mlp(hidden_dim, 1)
        if predict_pedal_offset:
            self.pedal_offset_output_layer = self._build_mlp(hidden_dim, 1)
        if predict_room:
            self.room_head = self._build_mlp(hidden_dim, 4)

        # # Projections for disentanglement
        # self.room_projection = self._build_mlp(hidden_dim, hidden_dim)
        # self.pedal_projection = self._build_mlp(hidden_dim, hidden_dim)

        print("Optional predictions:")
        print(f"Predict Global Pedal: {predict_global_pedal}")
        print(f"Predict Pedal Onset: {predict_pedal_onset}")
        print(f"Predict Pedal Offset: {predict_pedal_offset}")
        print(f"Predict Room: {predict_room}")

    def _build_mlp(self, input_dim, output_dim, dropout=0.1):
        """Helper function to build a two-layer MLP with ReLU and dropout."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, output_dim),
        )

    def forward(self, x, loss_mask=None, src_mask=None):
        # Apply CNN layers
        x = self.cnn(x)  # [batch, seq_len, hidden_dim]

        # Transformer Encoder
        x = self.positional_encoding(x)
        latent_reprs = []
        for layer in self.layers:
            x = layer(x, mask=src_mask)
            latent_reprs.append(x)

        # Latent Representation for each layer x
        latent_reprs[-1] = F.normalize(latent_reprs[-1], p=2, dim=-1)
        last_latent_repr = latent_reprs[-1]

        # Frame-wise Predictions
        p_v_logits = self.pedal_value_output_layer(last_latent_repr)
        p_on_logits = getattr(self, "pedal_onset_output_layer", None)
        p_off_logits = getattr(self, "pedal_offset_output_layer", None)

        if p_on_logits is not None:
            p_on_logits = p_on_logits(last_latent_repr)
        else:
            p_on_logits = torch.zeros_like(p_v_logits)  # Placeholder tensor

        if p_off_logits is not None:
            p_off_logits = p_off_logits(last_latent_repr)
        else:
            p_off_logits = torch.zeros_like(p_v_logits)  # Placeholder tensor

        # Apply Loss Mask
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            p_v_logits = p_v_logits * loss_mask
            p_on_logits = p_on_logits * loss_mask
            p_off_logits = p_off_logits * loss_mask
            latent_reprs = [layer_out * loss_mask for layer_out in latent_reprs]

        # Mean Latent Representation for Global Predictions
        mean_latent_reprs = [x.sum(dim=1) / loss_mask.sum(dim=1) for x in latent_reprs]
        last_mean_latent_repr = mean_latent_reprs[-1]
        # mean_latent_repr = latent_repr.sum(dim=1) / loss_mask.sum(dim=1)

        # Room Prediction uses only room_repr
        room_logits = getattr(self, "room_head", None)
        if room_logits is not None:
            room_logits = self.room_head(last_mean_latent_repr)
        else:
            room_logits = torch.zeros_like(p_v_logits[:, :4])  # Placeholder tensor

        # Global pedal prediction uses only pedal_repr
        global_p_v_logits = getattr(self, "global_pedal_value_head", None)
        if global_p_v_logits is not None:
            global_p_v_logits = self.global_pedal_value_head(last_mean_latent_repr)
        else:
            global_p_v_logits = torch.zeros_like(p_v_logits[:, :1])  # Placeholder tensor

        return global_p_v_logits, p_v_logits, p_on_logits, p_off_logits, \
            room_logits, last_latent_repr, mean_latent_reprs
