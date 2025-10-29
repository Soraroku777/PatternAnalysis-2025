from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual convolutional block with two convolutions and optional squeeze-excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        mid_channels: int | None = None,
        dropout: float = 0.0,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.use_se = use_se
        self.se = SqueezeExcite(out_channels) if use_se else nn.Identity()

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_se:
            x = self.se(x)
        x = x + residual
        x = self.act(x)
        return x


class SqueezeExcite(nn.Module):
    """Channel-wise attention via squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale