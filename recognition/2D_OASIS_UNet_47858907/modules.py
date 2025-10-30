"""
U-Net architecture components for medical image segmentation.

This module implements an enhanced U-Net with residual connections, squeeze-and-excitation
blocks, and attention gates for improved segmentation performance on brain MRI data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Residual convolutional block with two convolutions and optional squeeze-excitation.
    
    Implements a residual connection around two 3x3 convolutions with batch normalisation
    and optional squeeze-excitation attention mechanism.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        mid_channels: Number of intermediate channels (defaults to out_channels)
        dropout: Dropout probability for regularisation
        use_se: Whether to apply squeeze-excitation attention
    """

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

        # First convolution path
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Second convolution path
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shared components
        self.act = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.use_se = use_se
        self.se = SqueezeExcite(out_channels) if use_se else nn.Identity()

        # Residual connection (adjust channels if needed)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = self.shortcut(x)  # Prepare residual connection
        
        # Main convolution path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Apply squeeze-excitation if enabled
        if self.use_se:
            x = self.se(x)
        
        # Add residual and final activation
        x = x + residual
        x = self.act(x)
        return x


class SqueezeExcite(nn.Module):
    """
    Channel-wise attention via squeeze-and-excitation mechanism.
    
    Learns to weight feature channels based on their importance using
    global average pooling followed by two fully connected layers.
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for the bottleneck
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)  # Ensure minimum channels
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)  # Squeeze
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)  # Excite
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-excitation attention."""
        # Squeeze: Global average pooling
        scale = F.adaptive_avg_pool2d(x, 1)  # (B, C, H, W) -> (B, C, 1, 1)
        
        # Excitation: FC layers with activation
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)  # Attention weights in [0, 1]
        
        # Apply attention weights
        return x * scale


class DownBlock(nn.Module):
    """
    Down-sampling block that halves spatial dimensions.
    
    Combines max pooling for downsampling with a convolutional block
    for feature extraction at the new resolution.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout: Dropout probability for regularisation
        use_se: Whether to apply squeeze-excitation attention
    """

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0, use_se: bool = True) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)  # 2x2 max pooling with stride 2
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout, use_se=use_se)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling followed by convolution."""
        x = self.pool(x)    # Reduce spatial dimensions by factor of 2
        x = self.conv(x)    # Extract features at new resolution
        return x


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections in U-Net decoder.
    
    Guides the decoder to focus on relevant regions by learning spatial attention
    weights based on both skip connection features and gating signal from decoder.
    
    Args:
        skip_channels: Number of channels in skip connection
        gating_channels: Number of channels in gating signal
    """

    def __init__(self, skip_channels: int, gating_channels: int) -> None:
        super().__init__()
        inter_channels = max(skip_channels // 2, 1)  # Intermediate channel reduction
        
        # Transform skip features (with downsampling)
        self.theta = nn.Conv2d(skip_channels, inter_channels, kernel_size=2, stride=2, bias=False)
        
        # Transform gating signal
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=True)
        
        # Generate attention weights
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        
        self.bn = nn.BatchNorm2d(inter_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-weighted skip features.
        
        Args:
            skip: Skip connection features from encoder
            gating: Gating signal from decoder
            
        Returns:
            Attention-weighted skip features
        """
        # Transform inputs
        theta_x = self.theta(skip)      # Downsample skip features
        phi_g = self.phi(gating)        # Transform gating signal
        
        # Ensure spatial dimensions match
        if theta_x.shape[-2:] != phi_g.shape[-2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[-2:], mode="bilinear", align_corners=True)
        
        # Compute attention weights
        f = F.relu(self.bn(theta_x + phi_g), inplace=True)  # Combine features
        psi = self.sigmoid(self.psi(f))                     # Generate attention map
        
        # Upsample attention to match skip resolution
        psi = F.interpolate(psi, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        
        # Apply attention weights to skip features
        return skip * psi


class UpBlock(nn.Module):
    """
    Up-sampling block with attention-gated skip fusion.
    
    Performs upsampling, applies attention gating to skip connections,
    and fuses features through concatenation and convolution.
    
    Args:
        in_channels: Number of channels from lower resolution
        skip_channels: Number of channels in skip connection
        out_channels: Number of output channels
        use_bilinear: Whether to use bilinear upsampling vs transposed convolution
        dropout: Dropout probability for regularisation
        use_se: Whether to apply squeeze-excitation attention
    """

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
        
        # Upsampling method: bilinear interpolation or transposed convolution
        if use_bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Attention gate for skip connections
        self.attention = AttentionGate(skip_channels=skip_channels, gating_channels=out_channels)
        
        # Feature fusion and processing
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout, use_se=use_se)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Upsample features and fuse with attention-gated skip connection.
        
        Args:
            x: Features from lower resolution
            skip: Skip connection from encoder
            
        Returns:
            Fused and processed features
        """
        # Upsample low-resolution features
        x = self.up(x)
        
        # Handle potential size mismatches from pooling operations
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(
                x,
                [
                    0,
                    skip.shape[-1] - x.shape[-1],  # Pad width
                    0,
                    skip.shape[-2] - x.shape[-2],  # Pad height
                ],
            )
        
        # Apply attention gating to skip connection
        skip = self.attention(skip, x)
        
        # Concatenate and process combined features
        x = torch.cat([skip, x], dim=1)  # Combine along channel dimension
        x = self.conv(x)                 # Process fused features
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck block with dilated convolutions for larger receptive fields.
    
    Uses dilated convolutions to capture multi-scale context without
    increasing the number of parameters significantly.
    
    Args:
        channels: Number of input/output channels
        dilation: Dilation rate for the first convolution
        dropout: Dropout probability for regularisation
    """

    def __init__(self, channels: int, dilation: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        # Dilated convolution for expanded receptive field
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Standard convolution
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.act = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x  # Save input for residual connection
        
        # First dilated convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        # Second standard convolution
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Add residual and final activation
        x = x + residual
        x = self.act(x)
        return x


class UNet(nn.Module):
    """
    Enhanced U-Net architecture for medical image segmentation.
    
    Integrates residual blocks, squeeze-excitation attention, and attention gates
    for improved performance on brain tissue segmentation tasks like OASIS.
    
    Features:
    - Residual connections for better gradient flow
    - Squeeze-excitation for channel attention
    - Attention gates for spatial attention in skip connections
    - Dilated convolutions in bottleneck for multi-scale context
    
    Args:
        in_channels: Number of input image channels (1 for greyscale)
        num_classes: Number of segmentation classes
        base_channels: Number of channels in first layer
        depth: Number of encoder/decoder levels
        dropout: Dropout probability for regularisation
        use_bilinear: Use bilinear upsampling vs transposed convolution
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
        use_bilinear: bool = False,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("UNet depth must be at least 2.")

        self.depth = depth
        
        # Initial convolution (stem)
        self.stem = ConvBlock(in_channels, base_channels, dropout=dropout, use_se=True)

        # Build encoder path
        encoder_blocks: list[nn.Module] = []
        channel_progression = [base_channels]  # Track channels at each level
        current_channels = base_channels
        
        for _ in range(depth - 1):
            next_channels = current_channels * 2  # Double channels at each level
            encoder_blocks.append(DownBlock(current_channels, next_channels, dropout=dropout, use_se=True))
            channel_progression.append(next_channels)
            current_channels = next_channels
        
        self.encoder = nn.ModuleList(encoder_blocks)

        # Bottleneck with dilated convolutions for multi-scale context
        self.bottleneck = Bottleneck(current_channels, dilation=2, dropout=dropout)

        # Build decoder path
        decoder_blocks: list[nn.Module] = []
        skip_channels_list = list(reversed(channel_progression[:-1]))  # Skip channels from encoder
        
        for skip_channels in skip_channels_list:
            decoder_blocks.append(
                UpBlock(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=skip_channels,      # Output matches skip connection
                    use_bilinear=use_bilinear,
                    dropout=dropout,
                    use_se=True,
                )
            )
            current_channels = skip_channels
        
        self.decoder = nn.ModuleList(decoder_blocks)

        # Final classification layer
        self.output_conv = nn.Conv2d(current_channels, num_classes, kernel_size=1)

        # Initialise weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise network weights using appropriate schemes."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Kaiming initialisation for ReLU-like activations
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard batch norm initialisation
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input image tensor with shape (B, C, H, W)
            
        Returns:
            Logits tensor with shape (B, num_classes, H, W)
        """
        # Collect skip connections during encoder pass
        skips: list[torch.Tensor] = []
        
        # Initial convolution
        x = self.stem(x)
        skips.append(x)
        
        # Encoder path: downsample and extract features
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        
        # Bottleneck: process at lowest resolution
        x = self.bottleneck(x)
        
        # Decoder path: upsample and fuse with skip connections
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = block(x, skip)
        
        # Final classification
        logits = self.output_conv(x)
        return logits


__all__ = [
    "UNet",
    "ConvBlock",
    "DownBlock",
    "UpBlock",
    "Bottleneck",
    "SqueezeExcite",
    "AttentionGate",
]