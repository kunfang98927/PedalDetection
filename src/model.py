import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2, attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attn_weights


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
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.classification_output_layer = nn.Linear(
            hidden_dim, num_classes
        )  # For frame-wise classification

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.input_proj(x)  # Project input to hidden dimension
        x = self.positional_encoding(x)  # Add positional encoding

        attn_weights_list = []  # To collect attention weights
        for layer in self.layers:
            x, attn_weights = layer(
                x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )
            attn_weights_list.append(attn_weights)
        latent_repr = F.normalize(x, p=2, dim=-1)  # Latent representations
        class_logits = self.classification_output_layer(
            latent_repr
        )  # Classification using latent reps

        return class_logits, latent_repr, attn_weights_list
