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


class DownBlock(nn.Module):
    """Down-sampling block that halves spatial dimensions."""

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0, use_se: bool = True) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout, use_se=use_se)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class AttentionGate(nn.Module):
    """Attention gate for skip connections, guiding the decoder to relevant regions."""

    def __init__(self, skip_channels: int, gating_channels: int) -> None:
        super().__init__()
        inter_channels = max(skip_channels // 2, 1)
        self.theta = nn.Conv2d(skip_channels, inter_channels, kernel_size=2, stride=2, bias=False)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        theta_x = self.theta(skip)
        phi_g = self.phi(gating)
        if theta_x.shape[-2:] != phi_g.shape[-2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[-2:], mode="bilinear", align_corners=True)
        f = F.relu(self.bn(theta_x + phi_g), inplace=True)
        psi = self.sigmoid(self.psi(f))
        psi = F.interpolate(psi, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return skip * psi


class UpBlock(nn.Module):
    """Up-sampling block with attention-gated skip fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        use_bilinear: bool = False,
        dropout: float = 0.0,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        if use_bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(skip_channels=skip_channels, gating_channels=out_channels)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout, use_se=use_se)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(
                x,
                [
                    0,
                    skip.shape[-1] - x.shape[-1],
                    0,
                    skip.shape[-2] - x.shape[-2],
                ],
            )
        skip = self.attention(skip, x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class Bottleneck(nn.Module):
    """Bottleneck block with dilated convolutions for larger receptive fields."""

    def __init__(self, channels: int, dilation: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.act(x)
        return x
