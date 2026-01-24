import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparateProcessingFusion(nn.Module):
    """Process mel, MFCC, and optionally MIDI features separately, then fuse"""
    def __init__(self, mel_bins=229, mfcc_dims=20, hidden_dim=128, dropout=0.15, use_midi=True, use_pred_pedal=False, pedal_latent=False):
        super().__init__()
        
        self.mel_bins = mel_bins
        self.mfcc_dims = mfcc_dims
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_midi = use_midi
        self.use_pred_pedal=use_pred_pedal # if include predicted pedal as input feature as well
        self.use_pedal_latent=pedal_latent # if include latent pedal feature as input feature instead of pedal value

        # Validate that pred_pedal can only be used with MIDI
        if self.use_pred_pedal and not self.use_midi:
            raise ValueError("use_pred_pedal=True requires use_midi=True. Predicted pedal can only be used when MIDI is available.")
        
        # Calculate modality dimensions based on MIDI usage
        if self.use_midi and self.use_pred_pedal:
            self.num_modalities = 4
        elif self.use_midi:
            self.num_modalities = 3
        else:
            self.num_modalities = 2

        self.modality_dim = self.hidden_dim 
        self.hidden_dim = self.modality_dim * self.num_modalities

        print(f"Modality dim: {self.modality_dim}, Num modalities: {self.num_modalities}, Total hidden dim: {self.hidden_dim}")
        
        # Separate CNN for mel spectrogram (229 bins)
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
            nn.Conv2d(64, self.modality_dim, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.BatchNorm2d(self.modality_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Collapse frequency
        )
        
        # MLP for MFCC features (20 features)
        self.mfcc_mlp = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.modality_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # MLP for MIDI features (88 features - standard piano range)
        if self.use_midi:
            self.midi_mlp = nn.Sequential(
                nn.Linear(88, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, self.modality_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.midi_mlp = None

        # MLP for predicted pedal features (single 0-1 value per frame)
        if self.use_pred_pedal:
            if self.use_pedal_latent:
                print("Using latent pedal feature as input")
                self.pedal_mlp = nn.Sequential(
                    nn.Linear(256, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, self.modality_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            else:
                print("Using predicted pedal value as input")
                self.pedal_mlp = nn.Sequential(
                    nn.Linear(1, 16),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(16, self.modality_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
        else:
            self.pedal_mlp = None
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
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
        
        # Handle MIDI based on initialization flag
        if self.use_midi:
            if midi_inputs is None:
                raise ValueError("Model was initialized with use_midi=True but no MIDI inputs provided")
            
            midi_out = self.midi_mlp(midi_inputs)  # [batch, seq, modality_dim]

            if self.use_pred_pedal:
                if pred_pedal_inputs is None:
                    raise ValueError("Model was initialized with use_pred_pedal=True but no predicted pedal inputs provided")
                
                pedal_out = self.pedal_mlp(pred_pedal_inputs)  # [batch, seq, modality_dim]
                fused = torch.cat([mel_out, mfcc_out, midi_out, pedal_out], dim=-1)  # [batch, seq, hidden_dim]
            else:
                fused = torch.cat([mel_out, mfcc_out, midi_out], dim=-1)  # [batch, seq, hidden_dim]
        else:
            if midi_inputs is not None:
                print("Warning: MIDI inputs provided but model was initialized with use_midi=False. Ignoring MIDI.")
            fused = torch.cat([mel_out, mfcc_out], dim=-1)  # [batch, seq, hidden_dim]
        

        output = self.fusion(fused)  # [B, T, hidden_dim]
        
        return output

class SimplePaddedFusion(nn.Module):
    """Concatenate and pad to nearest multiple of num_heads"""
    def __init__(self, mel_bins=229, mfcc_dims=20, use_midi=True, use_pred_pedal=False, num_heads=8, **kwargs):
        super().__init__()
        
        self.mel_bins = mel_bins
        self.mfcc_dims = mfcc_dims
        self.use_midi = use_midi
        self.use_pred_pedal = use_pred_pedal
        self.num_heads = num_heads
        # hidden_dim passed in will not be used, just for placeholder compatibility
        if self.use_pred_pedal and not self.use_midi:
            raise ValueError("use_pred_pedal=True requires use_midi=True")
        
        # Calculate raw concatenated dimension
        raw_dim = mel_bins + mfcc_dims  # 249
        if use_midi:
            raw_dim += 88  # 337
        if use_pred_pedal:
            raw_dim += 1   # 338
            
        # Round up to nearest multiple of num_heads
        self.hidden_dim = ((raw_dim + num_heads - 1) // num_heads) * num_heads
        self.padding_size = self.hidden_dim - raw_dim
        
        print(f"Padded fusion: {raw_dim} -> {self.hidden_dim} (padded by {self.padding_size})")
        print(f"Divides evenly by {num_heads} heads: {self.hidden_dim // num_heads} dims per head")
        
        # Optional: learnable padding instead of zeros
        if self.padding_size > 0:
            self.padding = nn.Parameter(torch.randn(self.padding_size) * 0.01)
        
    def forward(self, x, midi_inputs=None, pred_pedal_inputs=None):
        batch_size, seq_len = x.shape[:2]
        
        # Concatenate all features
        features = [x]
        
        if self.use_midi and midi_inputs is not None:
            features.append(midi_inputs)
            
        if self.use_pred_pedal and pred_pedal_inputs is not None:
            if pred_pedal_inputs.dim() == 2:
                pred_pedal_inputs = pred_pedal_inputs.unsqueeze(-1)
            features.append(pred_pedal_inputs)
        
        concatenated = torch.cat(features, dim=-1)  # [batch, seq, raw_dim]
        
        # Add padding if needed
        if self.padding_size > 0:
            # Expand learnable padding to batch/sequence dimensions
            padding_expanded = self.padding.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_len, -1
            )
            concatenated = torch.cat([concatenated, padding_expanded], dim=-1)
        
        return concatenated 
    
class GatedFusion(nn.Module):
    """Process mel, MFCC, and optionally MIDI features separately, then use gated fusion"""
    def __init__(self, mel_bins=229, mfcc_dims=20, hidden_dim=256, dropout=0.15, use_midi=True):
        super().__init__()
        
        self.mel_bins = mel_bins
        self.mfcc_dims = mfcc_dims
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_midi = use_midi
        
        # Calculate modality dimensions based on MIDI usage
        if self.use_midi:
            self.modality_dim = hidden_dim // 3  # 3-way split
            self.num_modalities = 3
        else:
            self.modality_dim = hidden_dim // 2  # 2-way split
            self.num_modalities = 2

        # Separate CNN for mel spectrogram (229 bins)
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
            nn.Conv2d(64, self.modality_dim, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.BatchNorm2d(self.modality_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Collapse frequency
        )
        
        # MLP for MFCC features (20 features)
        self.mfcc_mlp = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.modality_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # MLP for MIDI features (88 features - standard piano range)
        if self.use_midi:
            self.midi_mlp = nn.Sequential(
                nn.Linear(88, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, self.modality_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.midi_mlp = None
        
        # Gating mechanism - learns importance weights for each modality
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_modalities),  # Dynamic size!
            nn.Softmax(dim=-1)  # Ensure gates sum to 1
        )
        
        # Final fusion layer after gating
        self.fusion = nn.Sequential(
            nn.Linear(self.modality_dim, hidden_dim),  # Input is now modality_dim, not hidden_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, midi_inputs=None):
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
        
        # Handle MIDI presence/absence
        if self.use_midi:
            if midi_inputs is None:
                raise ValueError("Model was initialized with use_midi=True but no MIDI inputs provided")
            midi_out = self.midi_mlp(midi_inputs)  # [batch, seq, modality_dim]
            modalities = torch.stack([mel_out, mfcc_out, midi_out], dim=-1)  # [batch, seq, modality_dim, 3]
            all_features = torch.cat([mel_out, mfcc_out, midi_out], dim=-1)  # [batch, seq, hidden_dim]
        else:
            if midi_inputs is not None:
                print("Warning: MIDI inputs provided but model was initialized with use_midi=False. Ignoring MIDI.")
            modalities = torch.stack([mel_out, mfcc_out], dim=-1)  # [batch, seq, modality_dim, 2]
            all_features = torch.cat([mel_out, mfcc_out], dim=-1)  # [batch, seq, hidden_dim]
        
        # Apply gates: weighted combination of modalities
        gates = self.gate_network(all_features)
        gates_expanded = gates.unsqueeze(-2)  # [batch, seq, 1, num_modalities]
        
        # Weighted sum across modalities
        gated_features = torch.sum(modalities * gates_expanded, dim=-1)  # [batch, seq, modality_dim]
        
        # Final fusion - no need to artificially expand to hidden_dim
        output = self.fusion(gated_features)  # [batch, seq, hidden_dim]

        # Store for easy debugging
        self.last_gates = gates.mean(dim=[0,1]).detach()  # Average gates across batch/time
        return output