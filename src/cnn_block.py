import torch.nn as nn

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