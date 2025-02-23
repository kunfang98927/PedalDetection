import copy
import torch.nn as nn
import torch.nn.functional as F
from .transformer import (
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    EncoderLayer,
)


class CNNBlock(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.15):
        super(CNNBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), stride=1, padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),  # Freq 249 → 83
            # nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 12), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),  # Freq 83 → 24
            # nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, hidden_dim, kernel_size=(3, 6), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),  # Freq 24 → 6
            # nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((None, 1))  # Collapse frequency

    def forward(self, x):
        batch_size, seq_len, freq_dim = x.shape
        x = x.view(batch_size, 1, seq_len, freq_dim)  # [batch, 1, time, freq]

        # CNN layers
        x = self.conv1(x)  # [batch, 32, time, 83]
        x = self.conv2(x)  # [batch, 64, time, 24]
        x = self.conv3(x)  # [batch, 256, time, 6]

        # Global pooling to collapse frequency
        x = self.global_pool(x)  # [batch, 256, time, 1]

        # Reshape for Transformer input
        x = x.squeeze(-1).permute(0, 2, 1)  # [batch, time, hidden_dim]

        return x

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
        )
        
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

        return low_res_p_v_logits, p_v_logits, p_on_logits, p_off_logits, room_logits, latent_repr
