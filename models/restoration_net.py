"""
Semiconductor Inspection Image Restoration Network
===================================================
Architecture: Lightweight Residual U-Net with Residual Channel Attention (RCAB)
              and Sub-Pixel Convolution (PixelShuffle) Upsampling Head.

Key Design Decisions:
1. Joint Task: Performs speckle denoising and 2x spatial super-resolution in
   a single unified forward pass.
2. Unclipped Input Feeding: Degraded input images are passed unclipped to the
   initial convolution. This preserves the overshoot (>1.0) and undershoot (<0.0)
   magnitudes caused by speckle noise, providing rich localized variance cues
   for the denoising layers.
3. Residual Channel Attention (RCAB): Squeeze-and-Excitation mechanisms
   adaptively recalibrate feature channels, allowing the network to prioritize
   high-frequency semiconductor line edges over uniform noise components.
4. Sub-Pixel Convolution Head: Uses PixelShuffle upsampling (2x) to expand
   feature maps from 128x128 to 256x256 without introducing checkerboard
   deconvolution artifacts.
5. Global Bicubic Residual Skip: Learns the high-frequency residual correction
   on top of a smooth bicubic upsampled base.
6. Output Clamping: Output is clamped to [0.0, 1.0] matching the physical
   reflectance domain of clean ground-truth semiconductor images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Layer (Squeeze-and-Excitation).
    Exploits inter-channel dependencies to adaptively rescale feature maps.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 8), kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(channels // reduction, 8), channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x))
        return x * scale


class RCABlock(nn.Module):
    """
    Residual Channel Attention Block (RCAB).
    Combines 3x3 convolutions with GELU non-linearity and channel attention.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            ChannelAttention(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SemiconductorRestorationNet(nn.Module):
    """
    Unified Denoising and Super-Resolution Network for Semiconductor Inspection.

    Input:  (B, 1, 128, 128) - Single-channel noisy low-resolution image
    Output: (B, 1, 256, 256) - Single-channel restored high-resolution image [0, 1]
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 3   # 96

        # 1. Shallow feature extraction from raw unclipped degraded input
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.GELU()
        )

        # 2. Multi-scale Encoder
        self.enc1 = RCABlock(c1)  # 128x128
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)  # 64x64

        self.enc2 = RCABlock(c2)  # 64x64
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)  # 32x32

        # 3. Bottleneck
        self.bottleneck = nn.Sequential(
            RCABlock(c3),
            RCABlock(c3)
        )  # 32x32

        # 4. Multi-scale Decoder with Skip Connections
        self.up2 = nn.Sequential(
            nn.Conv2d(c3, c2 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU()
        )  # 64x64
        self.dec_conv2 = nn.Conv2d(c2 + c2, c2, kernel_size=1)
        self.dec2 = RCABlock(c2)

        self.up1 = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU()
        )  # 128x128
        self.dec_conv1 = nn.Conv2d(c1 + c1, c1, kernel_size=1)
        self.dec1 = RCABlock(c1)

        # 5. Sub-Pixel Convolution (PixelShuffle) Super-Resolution Head (128x128 -> 256x256)
        self.sr_upsample = nn.Sequential(
            nn.Conv2d(c1, c1 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.GELU()
        )

        # 6. High-Resolution Refinement
        self.hr_refinement = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1 // 2, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global bicubic baseline skip connection
        x_base = F.interpolate(
            torch.clamp(x, 0.0, 1.0),
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )

        # Encoder path
        h0 = self.head(x)          # (B, 32, 128, 128)
        e1 = self.enc1(h0)         # (B, 32, 128, 128)
        
        d1 = self.down1(e1)        # (B, 64, 64, 64)
        e2 = self.enc2(d1)         # (B, 64, 64, 64)

        d2 = self.down2(e2)        # (B, 96, 32, 32)
        b = self.bottleneck(d2)    # (B, 96, 32, 32)

        # Decoder path with skip connections
        u2 = self.up2(b)           # (B, 64, 64, 64)
        cat2 = torch.cat([u2, e2], dim=1) # (B, 128, 64, 64)
        dec2_feat = self.dec2(self.dec_conv2(cat2)) # (B, 64, 64, 64)

        u1 = self.up1(dec2_feat)   # (B, 32, 128, 128)
        cat1 = torch.cat([u1, e1], dim=1) # (B, 64, 128, 128)
        dec1_feat = self.dec1(self.dec_conv1(cat1)) # (B, 32, 128, 128)

        # Super-resolution upsampling to 256x256
        sr_feat = self.sr_upsample(dec1_feat) # (B, 32, 256, 256)
        residual = self.hr_refinement(sr_feat) # (B, 1, 256, 256)

        # Global residual addition and physical dynamic range clamping
        out = torch.clamp(x_base + residual, 0.0, 1.0)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
