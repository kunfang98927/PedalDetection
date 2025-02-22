import copy
import torch.nn as nn
import torch.nn.functional as F
from .transformer import (
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    EncoderLayer,
)

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
        self.cnn = nn.Sequential(
            # Conv1: Reduce frequency dimension while keeping time intact
            nn.Conv2d(1, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Conv2: Further frequency compression
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Conv3: Map to hidden_dim for Transformer
            nn.Conv2d(128, hidden_dim, kernel_size=(1, 1), stride=1, padding=0),  
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Optional: Adaptive Pooling to ensure final freq_dim = 1
            nn.AdaptiveAvgPool2d((None, 1))  # [batch_size, hidden_dim, seq_len, 1]
        )

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
        )
        
        self.pedal_value_output_layer = nn.Linear(hidden_dim, num_classes)
        self.pedal_onset_output_layer = nn.Linear(hidden_dim, num_classes)
        self.pedal_offset_output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, loss_mask=None, src_mask=None):
        # 1. CNN Input: Expected shape [batch_size, seq_len, freq_dim]
        batch_size, seq_len, freq_dim = x.shape
        x = x.view(batch_size, 1, seq_len, freq_dim)  # Add channel for Conv2D

        # 2. Apply CNN layers
        x = self.cnn(x)  # Shape: [batch_size, hidden_dim, time, freq]

        # 3. Flatten for Transformer input [64, 256, 100, 6] -> [64, 100, 256]
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # 4. Transformer Encoder
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        latent_repr = F.normalize(x, p=2, dim=-1)

        # 5. Heads for Pedal Detection
        p_on_logits = self.pedal_onset_output_layer(latent_repr)
        p_off_logits = self.pedal_offset_output_layer(latent_repr)
        p_v_logits = self.pedal_value_output_layer(latent_repr)

        # 6. Mean Latent Representation for Global Prediction
        mean_latent_repr = x.sum(dim=1) / x.shape[1]

        # 7. Global Pedal & Room Predictions
        room_logits = self.room_head(mean_latent_repr)
        low_res_p_v_logits = self.low_res_pedal_value_head(mean_latent_repr)

        return low_res_p_v_logits, p_v_logits, p_on_logits, p_off_logits, room_logits, latent_repr
