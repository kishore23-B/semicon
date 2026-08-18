"""
Step 3: Hard-Example-Weighted Fine-Tuning Script
- Analyzes training set sample difficulty (speckle noise variance + SSIM loss)
- Implements WeightedRandomSampler to oversample hard outlier analogues (2.5x weight on top quartile difficulty)
- Loss: λ_ssim = 0.38, λ_sobel = 0.15, w_charb = 0.47
- LR: 6e-5 -> 1e-6 (CosineAnnealingLR, 12 epochs)
- Early stopping on validation SSIM (patience = 6)
- Saves best weights to checkpoints/best_model_hard_weighted.pth
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
from torch.utils.data import DataLoader, WeightedRandomSampler

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


class HardWeightedCompoundLoss(nn.Module):
    """
    Compound loss with λ_ssim = 0.38, λ_sobel = 0.15, Charbonnier = 0.47.
    """
    def __init__(self, lambda_ssim=0.38, lambda_sobel=0.15):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_sobel = lambda_sobel
        self.lambda_charb = 1.0 - lambda_ssim - lambda_sobel

        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.ssim_loss = FastSSIMLoss(window_size=7, sigma=1.5, channels=1, val_range=1.0)
        self.sobel = SobelGradientLoss()

    def forward(self, pred, target):
        l_charb = self.charbonnier(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_sobel = self.sobel(pred, target)

        total = self.lambda_charb * l_charb + self.lambda_ssim * l_ssim + self.lambda_sobel * l_sobel
        return total, l_charb, l_ssim


@torch.no_grad()
def evaluate_val_set(model, loader, device):
    model.eval()
    psnr_list, ssim_list = [], []
    for lr, gt, _ in loader:
        lr_dev = lr.to(device)
        pred = model(lr_dev)
        for b in range(pred.size(0)):
            p = np.clip(pred[b, 0].cpu().numpy(), 0.0, 1.0)
            g = np.clip(gt[b, 0].numpy(), 0.0, 1.0)
            psnr_list.append(calculate_psnr(p, g))
            ssim_list.append(calculate_ssim(p, g))
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def compute_training_sample_weights(model, train_dataset, device):
    """Compute difficulty score for all 2,880 training pairs to create sampling weights."""
    print("[*] Profiling training set difficulty distribution for hard-example reweighting...")
    eval_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    losses = []
    criterion = CharbonnierLoss(eps=1e-3)

    with torch.no_grad():
        for lr, gt, _ in eval_loader:
            lr_dev, gt_dev = lr.to(device), gt.to(device)
            pred = model(lr_dev)
            for b in range(pred.size(0)):
                diff = torch.abs(pred[b] - gt_dev[b])
                # Overshoot penalty
                overshoot = float(torch.sum(lr[b] > 1.0) + torch.sum(lr[b] < 0.0)) / lr[b].numel()
                score = diff.mean().item() + 2.0 * overshoot
                losses.append(score)

    losses = np.array(losses)
    # Assign higher weights to top quartile difficulty (2.5x oversampling)
    q75 = np.percentile(losses, 75)
    weights = np.ones_like(losses)
    weights[losses >= q75] = 2.5
    weights[losses >= np.percentile(losses, 90)] = 3.5
    normalized_weights = weights / weights.sum()

    print(f"[*] Training set difficulty profiled: Min loss={losses.min():.4f}, Max={losses.max():.4f}, Mean={losses.mean():.4f}")
    print(f"[*] Hard-example sampling: {np.sum(weights > 1.0)} / {len(weights)} images assigned 2.5x–3.5x oversampling weight.")
    return normalized_weights


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Back up current best checkpoint to phase3_best_model.pth
    shutil.copyfile("checkpoints/best_model.pth", "checkpoints/phase3_best_model.pth")
    print("[*] Backed up current checkpoint to 'checkpoints/phase3_best_model.pth'")

    # 2. Dataset Loaders
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

    # 3. Model setup & resume
    model = SemiconductorRestorationNet(in_channels=1, base_channels=32, scale_factor=2).to(device)
    ckpt = torch.load("checkpoints/best_model.pth", map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # Profiling difficulty weights for hard-example oversampling
    sample_weights = compute_training_sample_weights(model, train_dataset, device)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)

    batch_size = 12
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initial baseline evaluation
    base_psnr, base_ssim = evaluate_val_set(model, val_loader, device)
    print(f"[*] Starting Baseline: SSIM = {base_ssim:.4f}, PSNR = {base_psnr:.2f} dB")

    # Hyperparameters
    epochs = 12
    lr = 6e-5
    min_lr = 1e-6
    patience = 6

    criterion = HardWeightedCompoundLoss(lambda_ssim=0.38, lambda_sobel=0.15).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    best_ssim = base_ssim
    best_psnr = base_psnr
    best_epoch = 0
    epochs_no_improve = 0

    print("\n" + "=" * 80)
    print(f"HARD-EXAMPLE-WEIGHTED FINE-TUNING (lambda_ssim=0.38, Sobel=0.15, LR={lr:.1e}, Epochs={epochs})")
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
        ep_loss = total_loss / len(train_dataset)
        val_psnr, val_ssim = evaluate_val_set(model, val_loader, device)
        ep_time = time.time() - t0

        is_best = val_ssim > best_ssim
        marker = " *" if is_best else ""

        if is_best:
            best_ssim = val_ssim
            best_psnr = val_psnr
            best_epoch = ep
            epochs_no_improve = 0
            torch.save(model.state_dict(), "checkpoints/best_model_hard_weighted.pth")
        else:
            epochs_no_improve += 1

        print(f"{ep:^7d} | {ep_loss:^12.5f} | {val_psnr:^14.3f} | {val_ssim:^10.4f} | {ep_time:^8.1f} | {'Best' if is_best else ''}{marker}")

        if epochs_no_improve >= patience:
            print(f"\n[!] Early stopping on val SSIM triggered (No improvement for {patience} epochs).")
            break

    print("\n" + "=" * 80)
    print(f"[*] Training finished. Best Single-Pass SSIM: {best_ssim:.4f} (at Epoch {best_epoch})")

    # If new checkpoint improved, promote it to best_model.pth
    if best_ssim > base_ssim:
        shutil.copyfile("checkpoints/best_model_hard_weighted.pth", "checkpoints/best_model.pth")
        print("[*] Successfully updated 'checkpoints/best_model.pth' with new superior weights!")


if __name__ == "__main__":
    main()
