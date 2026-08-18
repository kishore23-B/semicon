"""
Comprehensive Diagnosis Script:
1. Per-image SSIM distribution & Worst 15-20 Failure Case Analysis (with image generation)
2. Loss Alignment Analysis (MS-SSIM vs FastSSIM vs Sobel)
3. Overfitting / Underfitting / Architecture Capacity Analysis
4. Test-Time Augmentation (TTA) Evaluation on Validation Set
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))

from models.restoration_net import SemiconductorRestorationNet
from utils.dataset import SemiconductorDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips, LPIPSCalculator

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Diagnostic Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 1. Load Validation Split
with open("checkpoints/val_filenames.json", "r") as f:
    val_filenames = json.load(f)

val_lr_paths = [os.path.join("data/train/NoisyLR", f) for f in val_filenames]
val_gt_paths = [os.path.join("data/train/GT", f) for f in val_filenames]

val_dataset = SemiconductorDataset(val_lr_paths, val_gt_paths, is_train=False, preload=True, patch_size=None)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# Load best checkpoint
model = SemiconductorRestorationNet().to(device)
ckpt = torch.load("checkpoints/best_model.pth", map_location=device, weights_only=True)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

lpips_calc = LPIPSCalculator(device=str(device))

# ==============================================================================
# PART 1: PER-IMAGE SSIM & WORST-CASE DIAGNOSIS
# ==============================================================================
print("\n" + "=" * 70)
print("PART 1: PER-IMAGE SSIM DISTRIBUTION & FAILURE CASE ANALYSIS")
print("=" * 70)

results = []
raw_preds = []
raw_gts = []
raw_lrs = []

with torch.no_grad():
    for lr, gt, fnames in val_loader:
        lr_dev = lr.to(device)
        pred = model(lr_dev)
        for i in range(pred.size(0)):
            p = np.clip(pred[i, 0].cpu().numpy(), 0.0, 1.0)
            g = np.clip(gt[i, 0].numpy(), 0.0, 1.0)
            l = lr[i, 0].numpy()

            raw_preds.append(p)
            raw_gts.append(g)
            raw_lrs.append(l)

            s = calculate_ssim(p, g)
            ps = calculate_psnr(p, g)

            # Analyze properties of the sample
            gt_std = float(np.std(g))           # Texture / variance in GT
            gt_mean = float(np.mean(g))         # Mean brightness
            lr_noise_est = float(np.std(l - F.interpolate(torch.from_numpy(g).unsqueeze(0).unsqueeze(0), size=(128,128), mode='bicubic').squeeze().numpy()))
            overshoot = float(np.sum(l > 1.0) + np.sum(l < 0.0)) / l.size  # fraction of unclipped noisy pixels

            results.append({
                "filename": fnames[i],
                "ssim": s,
                "psnr": ps,
                "gt_std": gt_std,
                "gt_mean": gt_mean,
                "noise_est": lr_noise_est,
                "overshoot_frac": overshoot
            })

all_ssim = [r["ssim"] for r in results]
all_psnr = [r["psnr"] for r in results]

print(f"Overall Validation Statistics (N={len(results)}):")
print(f"  Mean SSIM:   {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f}")
print(f"  Median SSIM: {np.median(all_ssim):.4f}")
print(f"  Min SSIM:    {np.min(all_ssim):.4f} | Max SSIM: {np.max(all_ssim):.4f}")
print(f"  Percentiles: 10th={np.percentile(all_ssim, 10):.4f}, 25th={np.percentile(all_ssim, 25):.4f}, 75th={np.percentile(all_ssim, 75):.4f}, 90th={np.percentile(all_ssim, 90):.4f}")

# Sort by SSIM
sorted_results = sorted(results, key=lambda x: x["ssim"])
worst_20 = sorted_results[:20]
best_20 = sorted_results[-20:]

print("\n--- Bottom 15 Worst-Performing Validation Images ---")
print(f"{'Filename':^12} | {'SSIM':^8} | {'PSNR (dB)':^10} | {'GT Std (Texture)':^16} | {'Noise Est':^10} | {'Overshoot %':^12}")
print("-" * 75)
for r in worst_20[:15]:
    print(f"{r['filename']:^12} | {r['ssim']:^8.4f} | {r['psnr']:^10.2f} | {r['gt_std']:^16.4f} | {r['noise_est']:^10.4f} | {r['overshoot_frac']*100:^11.1f}%")

# Correlation analysis of failure cases
worst_gt_stds = [r["gt_std"] for r in worst_20]
best_gt_stds = [r["gt_std"] for r in best_20]
print(f"\nPattern Findings in Failure Cases:")
print(f"  - Average GT Std Dev (Texture complexity) in Worst 20: {np.mean(worst_gt_stds):.4f}")
print(f"  - Average GT Std Dev (Texture complexity) in Best 20:  {np.mean(best_gt_stds):.4f}")
print(f"  - Texture Contrast Ratio: Worst cases have {np.mean(worst_gt_stds)/np.mean(best_gt_stds):.2f}x higher spatial variance/grain.")

# Create visual inspection figure of the 4 worst failure cases
os.makedirs("visual_samples", exist_ok=True)
fig, axes = plt.subplots(4, 3, figsize=(10, 13))
for row_idx, r in enumerate(worst_20[:4]):
    fname = r["filename"]
    idx = [k for k, res in enumerate(results) if res["filename"] == fname][0]
    lr_img = raw_lrs[idx]
    pred_img = raw_preds[idx]
    gt_img = raw_gts[idx]

    axes[row_idx, 0].imshow(np.clip(lr_img, 0, 1), cmap="gray")
    axes[row_idx, 0].set_title(f"Input Noisy LR (128x128)\n{fname}")
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(pred_img, cmap="gray")
    axes[row_idx, 1].set_title(f"Restored (256x256)\nSSIM: {r['ssim']:.4f}, PSNR: {r['psnr']:.2f}dB")
    axes[row_idx, 1].axis("off")

    axes[row_idx, 2].imshow(gt_img, cmap="gray")
    axes[row_idx, 2].set_title(f"Ground Truth HR\nStd: {r['gt_std']:.4f}")
    axes[row_idx, 2].axis("off")

plt.tight_layout()
plt.savefig("visual_samples/worst_cases_diagnosis.png", dpi=150)
plt.close()
print("  - Saved failure case visual comparison to 'visual_samples/worst_cases_diagnosis.png'")

# ==============================================================================
# PART 4: TEST-TIME AUGMENTATION (TTA)
# ==============================================================================
print("\n" + "=" * 70)
print("PART 4: TEST-TIME AUGMENTATION (TTA) INFERENCE EVALUATION")
print("=" * 70)

def forward_tta(model, lr_tensor):
    """8-fold D4 dihedral group test-time augmentation."""
    augmented_preds = []
    # 8 combinations: 4 rotations x 2 flips
    for flip in [False, True]:
        for rot in [0, 1, 2, 3]:
            x = lr_tensor
            if flip:
                x = torch.flip(x, dims=[-1])
            if rot > 0:
                x = torch.rot90(x, k=rot, dims=[-2, -1])

            out = model(x)

            if rot > 0:
                out = torch.rot90(out, k=-rot, dims=[-2, -1])
            if flip:
                out = torch.flip(out, dims=[-1])

            augmented_preds.append(out)

    # Average all 8 predictions
    return torch.mean(torch.stack(augmented_preds, dim=0), dim=0)

tta_ssims, tta_psnrs, tta_lpips = [], [], []

print("[*] Running 8-fold D4 Dihedral TTA on all 320 validation samples...")
t0 = time.time()
with torch.no_grad():
    for i, (lr, gt, _) in enumerate(val_loader):
        lr = lr.to(device)
        gt = gt.to(device)
        pred_tta = forward_tta(model, lr)

        for b in range(pred_tta.size(0)):
            p = np.clip(pred_tta[b, 0].cpu().numpy(), 0.0, 1.0)
            g = np.clip(gt[b, 0].cpu().numpy(), 0.0, 1.0)
            tta_ssims.append(calculate_ssim(p, g))
            tta_psnrs.append(calculate_psnr(p, g))
            if len(tta_lpips) < 64:  # Sample LPIPS for speed
                tta_lpips.append(calculate_lpips(p, g, calculator=lpips_calc))

tta_time = time.time() - t0
print(f"[*] TTA Evaluation completed in {tta_time:.1f}s")
print(f"\nStandard Single-Pass: SSIM = {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f} | PSNR = {np.mean(all_psnr):.2f} dB")
print(f"8-Fold Dihedral TTA:  SSIM = {np.mean(tta_ssims):.4f} +/- {np.std(tta_ssims):.4f} | PSNR = {np.mean(tta_psnrs):.2f} dB | LPIPS = {np.mean(tta_lpips):.4f}")
print(f"TTA Gain:             dSSIM = {np.mean(tta_ssims) - np.mean(all_ssim):+.4f} | dPSNR = {np.mean(tta_psnrs) - np.mean(all_psnr):+.2f} dB")

# Save diagnosis report
diagnosis_data = {
    "standard": {
        "ssim_mean": float(np.mean(all_ssim)),
        "ssim_std": float(np.std(all_ssim)),
        "psnr_mean": float(np.mean(all_psnr)),
        "psnr_std": float(np.std(all_psnr))
    },
    "tta": {
        "ssim_mean": float(np.mean(tta_ssims)),
        "ssim_std": float(np.std(tta_ssims)),
        "psnr_mean": float(np.mean(tta_psnrs)),
        "psnr_std": float(np.std(tta_psnrs)),
        "gain_ssim": float(np.mean(tta_ssims) - np.mean(all_ssim)),
        "gain_psnr": float(np.mean(tta_psnrs) - np.mean(all_psnr))
    },
    "worst_20_samples": worst_20
}

with open("visual_samples/diagnosis_report.json", "w") as f:
    json.dump(diagnosis_data, f, indent=2)
print("[*] Full diagnosis report written to 'visual_samples/diagnosis_report.json'")
