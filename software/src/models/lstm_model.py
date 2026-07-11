"""LSTM 光变曲线分类器"""

import torch
import torch.nn as nn


class LightCurveLSTM(nn.Module):
    """用于光变曲线分类的双向 LSTM"""

    def __init__(
        self,
        input_size: int = 3,       # mjd, flux, flux_err
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 18,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        返回: (batch, num_classes)
        """
        lstm_out, _ = self.lstm(x)  # (B, L, 2*H)

        # 注意力加权池化
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (B, L)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (B, 2*H)

        return self.classifier(context)
