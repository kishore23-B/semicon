"""
Fast and numerically stable Loss functions for Semiconductor Inspection Image Restoration.
Combines Charbonnier (smooth L1) pixel loss with differentiable Structural Similarity (SSIM) loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_window(window_size: int = 7, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    """Creates a normalized 2D Gaussian kernel for SSIM calculation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    w = g_2d.unsqueeze(0).repeat(channels, 1, 1, 1)  # (C, 1, H, W)
    return w


class FastSSIMLoss(nn.Module):
    """
    Fast Differentiable Structural Similarity (SSIM) Loss: L_ssim = 1 - SSIM(x, y).
    Optimized for high CPU/GPU throughput using 7x7 Gaussian kernel.
    """
    def __init__(self, window_size: int = 7, sigma: float = 1.5, channels: int = 1, val_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.val_range = val_range
        self.register_buffer("window", create_window(window_size, sigma, channels))

        self.c1 = (0.01 * val_range) ** 2
        self.c2 = (0.03 * val_range) ** 2

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        window = self.window.to(dtype=img1.dtype, device=img1.device)
        pad = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=pad, groups=self.channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=self.channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)) / (
            (mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2) + 1e-8
        )

        return 1.0 - ssim_map.mean()


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (differentiable smooth L1 variant):
    L(x, y) = sqrt((x - y)^2 + eps^2)
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return loss.mean()


class SobelGradientLoss(nn.Module):
    """
    Computes L1 loss on horizontal and vertical Sobel filtered gradients.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reflect pad to avoid border artifacts
        p_pred = F.pad(pred, (1, 1, 1, 1), mode="reflect")
        p_target = F.pad(target, (1, 1, 1, 1), mode="reflect")

        grad_pred_x = F.conv2d(p_pred, self.sobel_x)
        grad_pred_y = F.conv2d(p_pred, self.sobel_y)

        grad_target_x = F.conv2d(p_target, self.sobel_x)
        grad_target_y = F.conv2d(p_target, self.sobel_y)

        loss_x = F.l1_loss(grad_pred_x, grad_target_x)
        loss_y = F.l1_loss(grad_pred_y, grad_target_y)

        return loss_x + loss_y


class CompoundRestorationLoss(nn.Module):
    """
    Compound Loss balancing pixel-level fidelity, structural accuracy, and edge gradients:
        Loss_total = w_charb * Loss_charbonnier + w_ssim * Loss_ssim + w_sobel * Loss_sobel
    """
    def __init__(self, lambda_ssim: float = 0.2, lambda_sobel: float = 0.0, eps: float = 1e-3):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_sobel = lambda_sobel
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.ssim_loss = FastSSIMLoss(window_size=7, sigma=1.5, channels=1, val_range=1.0)
        self.sobel_loss = SobelGradientLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l_charb = self.charbonnier(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_sobel = self.sobel_loss(pred, target)
        
        # Calculate weights to sum to 1.0 (or fall back to standard formulation if lambda_sobel is 0)
        if self.lambda_sobel > 0:
            w_ssim = self.lambda_ssim
            w_sobel = self.lambda_sobel
            w_charb = 1.0 - w_ssim - w_sobel
            total = w_charb * l_charb + w_ssim * l_ssim + w_sobel * l_sobel
        else:
            total = l_charb + self.lambda_ssim * l_ssim
            
        return total, l_charb, l_ssim
