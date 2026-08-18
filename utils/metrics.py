"""
Evaluation metrics for Semiconductor Image Restoration: PSNR, SSIM, and LPIPS.
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Optional, Union


def calculate_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        pred: Predicted restored image array or tensor.
        target: Clean ground truth image array or tensor.
        data_range: Maximum possible pixel value (1.0 for normalized float32).

    Returns:
        PSNR value in decibels (dB).
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred)

    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = np.asarray(target)

    # Squeeze extra single dimensions
    pred_np = np.squeeze(pred_np)
    target_np = np.squeeze(target_np)

    # Compute PSNR
    mse = np.mean((pred_np - target_np) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(data_range / np.sqrt(mse)))


def calculate_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0
) -> float:
    """
    Computes Structural Similarity Index (SSIM).

    Args:
        pred: Predicted restored image array or tensor.
        target: Clean ground truth image array or tensor.
        data_range: Dynamic range of the images (1.0).

    Returns:
        SSIM scalar in [-1, 1] (higher is better).
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred)

    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = np.asarray(target)

    pred_np = np.squeeze(pred_np).astype(np.float64)
    target_np = np.squeeze(target_np).astype(np.float64)

    return float(structural_similarity(target_np, pred_np, data_range=data_range))


class LPIPSCalculator:
    """
    Computes Learned Perceptual Image Patch Similarity (LPIPS) using AlexNet backbone.
    Grayscale images are converted to 3-channel tensors and rescaled to [-1, 1].
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.loss_fn = None

    def _init_model(self):
        if self.loss_fn is None:
            import lpips
            self.loss_fn = lpips.LPIPS(net="alex", verbose=False).to(self.device)
            self.loss_fn.eval()

    def calculate(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Args:
            pred: Restored image in [0, 1].
            target: Ground truth image in [0, 1].

        Returns:
            LPIPS distance (lower is better).
        """
        self._init_model()

        if isinstance(pred, np.ndarray):
            pred_t = torch.from_numpy(pred).float()
        else:
            pred_t = pred.detach().clone().float()

        if isinstance(target, np.ndarray):
            target_t = torch.from_numpy(target).float()
        else:
            target_t = target.detach().clone().float()

        # Reshape to (1, 1, H, W)
        if pred_t.ndim == 2:
            pred_t = pred_t.unsqueeze(0).unsqueeze(0)
            target_t = target_t.unsqueeze(0).unsqueeze(0)
        elif pred_t.ndim == 3:
            pred_t = pred_t.unsqueeze(0)
            target_t = target_t.unsqueeze(0)

        # Expand single grayscale channel to 3 channels: (1, 3, H, W)
        if pred_t.shape[1] == 1:
            pred_t = pred_t.repeat(1, 3, 1, 1)
            target_t = target_t.repeat(1, 3, 1, 1)

        # Normalize from [0, 1] to [-1, 1] for LPIPS network
        pred_norm = (pred_t * 2.0 - 1.0).to(self.device)
        target_norm = (target_t * 2.0 - 1.0).to(self.device)

        with torch.no_grad():
            dist = self.loss_fn(pred_norm, target_norm)
        return float(dist.mean().item())


def calculate_lpips(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    calculator: Optional[LPIPSCalculator] = None,
    device: str = "cpu"
) -> float:
    """Convenience helper for LPIPS metric."""
    if calculator is None:
        calculator = LPIPSCalculator(device=device)
    return calculator.calculate(pred, target)
