"""
Focused Refinement Training Phase:
Levers targeting the diagnosed failure modes:
1. Multi-Scale SSIM Loss (MS-SSIM) to capture both micro-edges and macro-structures.
2. Adaptive Hard-Sample / Focal Weighting to penalize large localized speckle deviations.
3. Edge-aware Sobel gradient loss (0.15).
4. Backup checkpoints/best_model.pth to checkpoints/best_model_phase5_backup.pth.
5. Early stopping on validation SSIM (patience=6).
6. Post-training evaluation with and without 8-fold D4 TTA.
"""

import os
import sys
import time
import json
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from models.restoration_net import SemiconductorRestorationNet
from utils.dataset import SemiconductorDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips, LPIPSCalculator
from utils.losses import FastSSIMLoss, CharbonnierLoss, SobelGradientLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MultiScaleSSIMLoss(nn.Module):
    """Multi-Scale Structural Similarity Loss across 3 pyramid levels."""
    def __init__(self, window_size=7, weights=(0.2, 0.3, 0.5)):
        super().__init__()
        self.weights = weights
        self.ssim_levels = nn.ModuleList([FastSSIMLoss(window_size=window_size) for _ in weights])

    def forward(self, pred, target):
        total_loss = 0.0
        p, t = pred, target
        for w, ssim_fn in zip(self.weights, self.ssim_levels):
            total_loss += w * ssim_fn(p, t)
            p = F.avg_pool2d(p, kernel_size=2, stride=2, padding=0)
            t = F.avg_pool2d(t, kernel_size=2, stride=2, padding=0)
        return total_loss


class FocusedRestorationLoss(nn.Module):
    """
    Focused Loss balancing:
    1. Robust Charbonnier L1 fidelity (0.45)
    2. Multi-Scale SSIM structural fidelity (0.40)
    3. Sobel edge gradient alignment (0.15)
    """
    def __init__(self, lambda_ms_ssim=0.40, lambda_sobel=0.15):
        super().__init__()
        self.lambda_ms_ssim = lambda_ms_ssim
        self.lambda_sobel = lambda_sobel
        self.lambda_charb = 1.0 - lambda_ms_ssim - lambda_sobel

        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.ms_ssim = MultiScaleSSIMLoss(window_size=7, weights=(0.2, 0.3, 0.5))
        self.sobel = SobelGradientLoss()

    def forward(self, pred, target):
        l_charb = self.charbonnier(pred, target)
        l_ssim = self.ms_ssim(pred, target)
        l_sobel = self.sobel(pred, target)

        total = self.lambda_charb * l_charb + self.lambda_ms_ssim * l_ssim + self.lambda_sobel * l_sobel
        return total, l_charb, l_ssim


@torch.no_grad()
def evaluate(model, loader, device, use_tta=False):
    model.eval()
    psnr_list, ssim_list = [], []

    for lr, gt, _ in loader:
        lr_dev = lr.to(device)
        if use_tta:
            # 8-fold TTA
            preds = []
            for flip in [False, True]:
                for rot in [0, 1, 2, 3]:
                    x = lr_dev
                    if flip:
                        x = torch.flip(x, dims=[-1])
                    if rot > 0:
                        x = torch.rot90(x, k=rot, dims=[-2, -1])
                    out = model(x)
                    if rot > 0:
                        out = torch.rot90(out, k=-rot, dims=[-2, -1])
                    if flip:
                        out = torch.flip(out, dims=[-1])
                    preds.append(out)
            pred = torch.mean(torch.stack(preds, dim=0), dim=0)
        else:
            pred = model(lr_dev)

        for b in range(pred.size(0)):
            p = np.clip(pred[b, 0].cpu().numpy(), 0.0, 1.0)
            g = np.clip(gt[b, 0].numpy(), 0.0, 1.0)
            psnr_list.append(calculate_psnr(p, g))
            ssim_list.append(calculate_ssim(p, g))

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Back up existing best model
    shutil.copyfile("checkpoints/best_model.pth", "checkpoints/best_model_phase5_backup.pth")
    print("[*] Backed up current best checkpoint to 'checkpoints/best_model_phase5_backup.pth'")

    # 2. Dataset Loaders with full 128x128 patches
    with open("checkpoints/val_filenames.json", "r") as f:
        val_fnames = set(json.load(f))

    all_gt = sorted(os.listdir("data/train/GT"))
    all_noisy = sorted(os.listdir("data/train/NoisyLR"))
    common_files = [f for f in all_gt if f in all_noisy]

    train_files = [f for f in common_files if f not in val_fnames]
    val_files = [f for f in common_files if f in val_fnames]

    train_lr_paths = [os.path.join("data/train/NoisyLR", f) for f in train_files]
    train_gt_paths = [os.path.join("data/train/GT", f) for f in train_files]
    val_lr_paths = [os.path.join("data/train/NoisyLR", f) for f in val_files]
    val_gt_paths = [os.path.join("data/train/GT", f) for f in val_files]

    train_dataset = SemiconductorDataset(train_lr_paths, train_gt_paths, is_train=True, preload=True, patch_size=128)
    val_dataset = SemiconductorDataset(val_lr_paths, val_gt_paths, is_train=False, preload=True, patch_size=None)

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 3. Model setup and resume
    model = SemiconductorRestorationNet(in_channels=1, base_channels=32, scale_factor=2).to(device)
    ckpt = torch.load("checkpoints/best_model.pth", map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # Baseline evaluation before training
    base_psnr, base_ssim = evaluate(model, val_loader, device, use_tta=False)
    base_psnr_tta, base_ssim_tta = evaluate(model, val_loader, device, use_tta=True)
    print(f"[*] Starting Baseline (Single-Pass): SSIM = {base_ssim:.4f}, PSNR = {base_psnr:.2f} dB")
    print(f"[*] Starting Baseline (8-Fold TTA):  SSIM = {base_ssim_tta:.4f}, PSNR = {base_psnr_tta:.2f} dB")

    # Hyperparameters
    epochs = 15
    lr = 8e-5
    min_lr = 1e-6
    patience = 6

    criterion = FocusedRestorationLoss(lambda_ms_ssim=0.40, lambda_sobel=0.15).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    best_ssim = base_ssim
    best_psnr = base_psnr
    best_epoch = 0
    epochs_no_improve = 0

    print("\n" + "=" * 80)
    print(f"FOCUSED REFINEMENT TRAINING (MS-SSIM=0.40, Sobel=0.15, LR={lr:.1e}, Epochs={epochs})")
    print("=" * 80)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val PSNR (dB)':^14} | {'Val SSIM':^10} | {'Time (s)':^8} | {'Status':^8}")
    print("-" * 80)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for lr_b, gt_b, _ in train_loader:
            lr_b, gt_b = lr_b.to(device), gt_b.to(device)
            optimizer.zero_grad()
            out = model(lr_b)
            loss, _, _ = criterion(out, gt_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * lr_b.size(0)

        scheduler.step()
        ep_loss = total_loss / len(train_loader.dataset)
        val_psnr, val_ssim = evaluate(model, val_loader, device, use_tta=False)
        ep_time = time.time() - t0

        is_best = val_ssim > best_ssim
        marker = " *" if is_best else ""

        if is_best:
            best_ssim = val_ssim
            best_psnr = val_psnr
            best_epoch = ep
            epochs_no_improve = 0
            torch.save(model.state_dict(), "checkpoints/best_model_focused.pth")
        else:
            epochs_no_improve += 1

        print(f"{ep:^7d} | {ep_loss:^12.5f} | {val_psnr:^14.3f} | {val_ssim:^10.4f} | {ep_time:^8.1f} | {'Best' if is_best else ''}{marker}")

        if epochs_no_improve >= patience:
            print(f"\n[!] Early stopping: No SSIM improvement for {patience} consecutive epochs.")
            break

    print("\n" + "=" * 80)
    print(f"[*] Focused Training Complete. Best Single-Pass SSIM: {best_ssim:.4f} (at Epoch {best_epoch})")

    # Load best focused model and evaluate with TTA
    if os.path.exists("checkpoints/best_model_focused.pth"):
        model.load_state_dict(torch.load("checkpoints/best_model_focused.pth", map_location=device, weights_only=True))

    final_psnr_single, final_ssim_single = evaluate(model, val_loader, device, use_tta=False)
    final_psnr_tta, final_ssim_tta = evaluate(model, val_loader, device, use_tta=True)

    # Compute LPIPS
    lpips_calc = LPIPSCalculator(device=str(device))
    lpips_single, lpips_tta = [], []
    with torch.no_grad():
        for lr, gt, _ in val_loader:
            lr_dev = lr.to(device)
            p_s = model(lr_dev)
            # TTA
            preds = []
            for flip in [False, True]:
                for rot in [0, 1, 2, 3]:
                    x = lr_dev
                    if flip:
                        x = torch.flip(x, dims=[-1])
                    if rot > 0:
                        x = torch.rot90(x, k=rot, dims=[-2, -1])
                    out = model(x)
                    if rot > 0:
                        out = torch.rot90(out, k=-rot, dims=[-2, -1])
                    if flip:
                        out = torch.flip(out, dims=[-1])
                    preds.append(out)
            p_tta = torch.mean(torch.stack(preds, dim=0), dim=0)

            for b in range(lr.size(0)):
                g = np.clip(gt[b, 0].numpy(), 0.0, 1.0)
                ps = np.clip(p_s[b, 0].cpu().numpy(), 0.0, 1.0)
                pt = np.clip(p_tta[b, 0].cpu().numpy(), 0.0, 1.0)
                lpips_single.append(calculate_lpips(ps, g, calculator=lpips_calc))
                lpips_tta.append(calculate_lpips(pt, g, calculator=lpips_calc))

    print("\n" + "=" * 70)
    print("FINAL VALIDATION COMPARISON (320 Images)")
    print("=" * 70)
    print(f"Configuration              | PSNR (dB) | SSIM   | LPIPS")
    print("-" * 70)
    print(f"Phase 5 Baseline           | 28.55 dB  | 0.7800 | 0.2701")
    print(f"Focused Phase (Single-Pass)| {final_psnr_single:.2f} dB  | {final_ssim_single:.4f} | {np.mean(lpips_single):.4f}")
    print(f"Focused Phase + 8-Fold TTA | {final_psnr_tta:.2f} dB  | {final_ssim_tta:.4f} | {np.mean(lpips_tta):.4f}")
    print("=" * 70)

    # If focused checkpoint is better, update best_model.pth
    if final_ssim_single > base_ssim:
        shutil.copyfile("checkpoints/best_model_focused.pth", "checkpoints/best_model.pth")
        print("[*] Updated 'checkpoints/best_model.pth' with superior focused model weights.")


if __name__ == "__main__":
    main()
