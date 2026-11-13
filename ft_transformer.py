"""FT-Transformer (Gorishniy et al. 2021 style): each numeric FEATURE is its own
token (not each timestep, unlike the windowed Transformer benchmark already used in
SOURCE_METHOD_REPRODUCTION) -- a genuinely different mechanism, included as the
strongest modern tabular-Transformer competitor (Section 14)."""
import torch
import torch.nn as nn


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=32, n_layers=3, n_heads=4, dropout=0.1, n_classes=15):
        super().__init__()
        self.feature_tokenizer = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                                                dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, d_token), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(d_token, n_classes))

    def forward(self, x):  # x: [B,F]
        tok = x.unsqueeze(-1) * self.feature_tokenizer.unsqueeze(0) + self.bias.unsqueeze(0)  # [B,F,D]
        cls = self.cls.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        enc = self.encoder(seq)
        return self.head(enc[:, 0])
