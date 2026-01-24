import torch
import torch.nn as nn
import torch.nn.functional as F

class InputFusion(nn.Module):
    """Process mel, MFCC, and optionally MIDI features separately with optimized MLPs, then fuse"""
    def __init__(self, mel_bins=229, mfcc_dims=20, hidden_dim=256, dropout=0.15, 
                 use_midi=True, use_pred_pedal=False, pedal_latent=False, num_heads=8,
                 cnn_dim=128, mfcc_dim=128, midi_dim=0, pedal_dim=0):
        super().__init__()
        
        self.mel_bins = mel_bins
        self.mfcc_dims = mfcc_dims
        self.dropout = dropout
        self.use_midi = use_midi
        self.use_pred_pedal = use_pred_pedal
        self.use_pedal_latent = pedal_latent
        self.hidden_dim = hidden_dim
        
        self.mel_dim = cnn_dim
        self.mfcc_dim = mfcc_dim
        self.midi_dim = midi_dim if use_midi else 0
        self.pedal_dim = pedal_dim if use_pred_pedal else 0
        raw_hidden_dim = self.mel_dim + self.mfcc_dim + self.midi_dim + self.pedal_dim

        # Calculate modality dimensions based on MIDI usage
        self.num_modalities = 2 + int(self.use_midi) + int(self.use_pred_pedal)
        
        print(f"Fusion Configuration:")
        print(f"  Num modalities: {self.num_modalities}")
        print(f"  Raw hidden dim: {raw_hidden_dim}")
        print(f"  Transformer heads: {num_heads} ({self.hidden_dim // num_heads} dims per head)")
        
        # cnn for mel spectrogram (229 bins)
        self.mel_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=1, padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),  # freq ↓
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),  # freq ↓            
            nn.Dropout(dropout),
            nn.Conv2d(64, self.mel_dim, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.BatchNorm2d(self.mel_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Collapse frequency
        )
        
        # Optimized MLP for MFCC features (20 features) - Direct mapping with processing
        self.mfcc_mlp = nn.Sequential(
            nn.Linear(self.mfcc_dims, self.mfcc_dim * 2),  # 20 -> 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.mfcc_dim * 2, self.mfcc_dim),
            nn.ReLU(),
            nn.LayerNorm(self.mfcc_dim)
        )
        
        # MLP for MIDI features (88 features - standard piano range)
        if self.use_midi:
            self.midi_mlp = nn.Sequential(
                nn.Linear(88, self.midi_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.midi_dim, self.midi_dim),
                nn.ReLU(),
                nn.LayerNorm(self.midi_dim)
            )
        else:
            self.midi_mlp = None

        # MLP for predicted pedal features 
        if self.use_pred_pedal:
            if self.use_pedal_latent:
                print("  Using latent pedal feature as input (256 dims)")
                # BERT-style processing for pre-trained latent features
                self.pedal_mlp = nn.Sequential(
                    nn.LayerNorm(256),  # Stabilize pre-trained features
                    nn.Linear(256, 64),  # Bottleneck compression
                    nn.GELU(),  # GELU for pre-trained features
                    nn.Dropout(dropout),
                    nn.Linear(64, self.pedal_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.pedal_dim)
                )
            else:
                print("  Using predicted pedal value as input (1 dim)")
                # Simple expansion for scalar pedal value
                self.pedal_mlp = nn.Sequential(
                    nn.Linear(1, 16),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(16, self.pedal_dim),
                    nn.ReLU(),
                    nn.LayerNorm(self.pedal_dim)
                )
        else:
            self.pedal_mlp = None
        
        self.fusion = nn.Sequential(
            nn.Linear(raw_hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Print parameter summary
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"  Total parameters: ~{total_params:.2f}M")
        
    def forward(self, x, midi_inputs=None, pred_pedal_inputs=None):
        batch_size, seq_len, total_features = x.shape
        assert total_features == self.mel_bins + self.mfcc_dims, (
            f"Expected input with {self.mel_bins + self.mfcc_dims} features, but got {total_features}"
        )
        
        # Split audio features
        mel_features = x[:, :, :self.mel_bins]  # [batch, seq, 229]
        mfcc_features = x[:, :, self.mel_bins:]  # [batch, seq, 20]
        
        # Process mel spectrogram
        mel_features = mel_features.unsqueeze(1)  # [batch, 1, seq, 229]
        mel_features = mel_features.transpose(2, 3)  # [batch, 1, 229, seq]
        
        mel_out = self.mel_cnn(mel_features)  # [batch, modality_dim, 1, seq]
        
        # Properly reshape mel output
        mel_out = mel_out.squeeze(2)  # [batch, modality_dim, seq]
        mel_out = mel_out.transpose(1, 2)  # [batch, seq, modality_dim]
        
        # Process MFCC
        mfcc_out = self.mfcc_mlp(mfcc_features)  # [batch, seq, modality_dim]

        midi_out = None
        pedal_out = None

        # Handle MIDI based on initialization flag
        if self.use_midi:
            if midi_inputs is None:
                raise ValueError("Model was initialized with use_midi=True but no MIDI inputs provided")
            
            midi_out = self.midi_mlp(midi_inputs)  # [batch, seq, modality_dim]
        else:
            if midi_inputs is not None:
                print("Warning: MIDI inputs provided but model was initialized with use_midi=False. Ignoring MIDI.")

        # Handle predicted pedal if specified
        if self.use_pred_pedal:
            if pred_pedal_inputs is None:
                raise ValueError("Model was initialized with use_pred_pedal=True but no predicted pedal inputs provided")
                        
            pedal_out = self.pedal_mlp(pred_pedal_inputs)  # [batch, seq, modality_dim]


        tensors_to_cat = [t for t in [mel_out, mfcc_out, midi_out, pedal_out] if t is not None]

        fused = torch.cat(tensors_to_cat, dim=-1)  # [batch, seq, hidden_dim]
        
        # Apply fusion layer
        output = self.fusion(fused)  # [batch, seq, hidden_dim]
        
        return output