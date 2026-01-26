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

from .input_layer import InputFusion
torch.autograd.set_detect_anomaly(True)



class PedalDetectionModel(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_layers,
        dropout=0.15,
        predict_pedal_onset=False,
        predict_pedal_offset=False,
        predict_global_pedal=True,
        use_midi=False,
        use_pred_pedal=False, # placeholder for compatibility
        pedal_latent=False,
        cnn_dim=256,
        mfcc_dim=128,
        midi_dim=128,
        pedal_dim=128
    ):
        super().__init__()
        print(f"Using input fusion layer")

        # Input Fusion Layer
        self.fusion_block = InputFusion(
            hidden_dim=hidden_dim, 
            dropout=dropout, 
            use_midi=use_midi, 
            use_pred_pedal=use_pred_pedal,
            pedal_latent=pedal_latent,
            cnn_dim=cnn_dim,
            mfcc_dim=mfcc_dim,
            midi_dim=midi_dim,
            pedal_dim=pedal_dim
        )
    

        # update hidden dim
        hidden_dim = self.fusion_block.hidden_dim
        ff_dim = hidden_dim * 4  # Typically FFN dim is 4x hidden dim
        print(f"Updated hidden dim after fusion: {hidden_dim}")
        print(f"Updated FFN dim after fusion: {ff_dim}")
    
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

        print("Optional predictions:")
        print(f"Predict Global Pedal: {predict_global_pedal}")
        print(f"Predict Pedal Onset: {predict_pedal_onset}")
        print(f"Predict Pedal Offset: {predict_pedal_offset}")

    def _build_mlp(self, input_dim, output_dim, dropout=0.1):
        """Helper function to build a two-layer MLP with ReLU and dropout."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, output_dim),
        )

    def forward(self, x, midi_inputs=None, pred_pedal_inputs=None, loss_mask=None, src_mask=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # CNN preprocessing with optional MIDI
        x = self.fusion_block(x, midi_inputs=midi_inputs, pred_pedal_inputs=pred_pedal_inputs)
        # Transformer Encoder
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask=src_mask)

        latent_repr = F.normalize(x, p=2, dim=-1)

        # Frame-wise Predictions
        p_v_logits = self.pedal_value_output_layer(latent_repr)
        p_on_logits = getattr(self, "pedal_onset_output_layer", None)
        p_off_logits = getattr(self, "pedal_offset_output_layer", None)

        if p_on_logits is not None:
            p_on_logits = p_on_logits(latent_repr)
        else:
            p_on_logits = torch.zeros_like(p_v_logits)  # Placeholder tensor

        if p_off_logits is not None:
            p_off_logits = p_off_logits(latent_repr)
        else:
            p_off_logits = torch.zeros_like(p_v_logits)  # Placeholder tensor

        # Apply Loss Mask
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)  # Ensure correct shape [batch, seq_len, 1]
            p_v_logits = p_v_logits * loss_mask
            p_on_logits = p_on_logits * loss_mask
            p_off_logits = p_off_logits * loss_mask
            latent_repr = latent_repr * loss_mask

        # Mean Latent Representation for Global Predictions
        mean_latent_repr = latent_repr.sum(dim=1) / loss_mask.sum(dim=1)

        # Global Predictions
        global_p_v_logits = getattr(self, "global_pedal_value_head", None)

        if global_p_v_logits is not None:
            global_p_v_logits = global_p_v_logits(mean_latent_repr)
        else:
            global_p_v_logits = torch.zeros_like(p_v_logits[:, :1]) # Placeholder tensor

        return global_p_v_logits, p_v_logits, p_on_logits, p_off_logits
