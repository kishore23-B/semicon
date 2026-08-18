"""
Step Final: Rigorous Verification, Comparative Evaluation, and Visual Inspection
Compares Phase 4 baseline (checkpoints/best_model.pth) vs Phase 5 (checkpoints/best_model_phase5.pth)
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))

from models.restoration_net import SemiconductorRestorationNet
from utils.dataset import SemiconductorDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips, LPIPSCalculator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Evaluation Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

lpips_calc = LPIPSCalculator(device=str(device))

with open("checkpoints/val_filenames.json", "r") as f:
    val_filenames = json.load(f)

val_lr_paths = [os.path.join("data/train/NoisyLR", f) for f in val_filenames]
val_gt_paths = [os.path.join("data/train/GT", f) for f in val_filenames]

val_dataset = SemiconductorDataset(val_lr_paths, val_gt_paths, is_train=False, preload=True, patch_size=None)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# 1. Load Phase 4 Baseline Model
model_p4 = SemiconductorRestorationNet().to(device)
p4_ckpt = torch.load("checkpoints/best_model.pth", map_location=device, weights_only=True)
if "model_state_dict" in p4_ckpt:
    model_p4.load_state_dict(p4_ckpt["model_state_dict"])
else:
    model_p4.load_state_dict(p4_ckpt)
model_p4.eval()

# 2. Load Phase 5 Refined Model
model_p5 = SemiconductorRestorationNet().to(device)
p5_ckpt = torch.load("checkpoints/best_model_phase5.pth", map_location=device, weights_only=True)
if "model_state_dict" in p5_ckpt:
    model_p5.load_state_dict(p5_ckpt["model_state_dict"])
else:
    model_p5.load_state_dict(p5_ckpt)
model_p5.eval()

p4_psnr, p4_ssim, p4_lpips = [], [], []
p5_psnr, p5_ssim, p5_lpips = [], [], []

sample_deltas = []

with torch.no_grad():
    for lr, gt, fnames in val_loader:
        lr = lr.to(device)
        gt = gt.to(device)

        out_p4 = model_p4(lr)
        out_p5 = model_p5(lr)

        for i in range(lr.size(0)):
            fname = fnames[i]
            p4_np = np.clip(out_p4[i, 0].cpu().numpy(), 0.0, 1.0)
            p5_np = np.clip(out_p5[i, 0].cpu().numpy(), 0.0, 1.0)
            gt_np = np.clip(gt[i, 0].cpu().numpy(), 0.0, 1.0)

            # Metrics for P4
            p4_p = calculate_psnr(p4_np, gt_np)
            p4_s = calculate_ssim(p4_np, gt_np)
            p4_l = calculate_lpips(p4_np, gt_np, calculator=lpips_calc)

            # Metrics for P5
            p5_p = calculate_psnr(p5_np, gt_np)
            p5_s = calculate_ssim(p5_np, gt_np)
            p5_l = calculate_lpips(p5_np, gt_np, calculator=lpips_calc)

            p4_psnr.append(p4_p)
            p4_ssim.append(p4_s)
            p4_lpips.append(p4_l)

            p5_psnr.append(p5_p)
            p5_ssim.append(p5_s)
            p5_lpips.append(p5_l)

            sample_deltas.append({
                "filename": fname,
                "p4_ssim": p4_s,
                "p5_ssim": p5_s,
                "delta_ssim": p5_s - p4_s,
                "p4_psnr": p4_p,
                "p5_psnr": p5_p,
                "delta_psnr": p5_p - p4_p,
                "p4_lpips": p4_l,
                "p5_lpips": p5_l
            })

print("\n" + "=" * 70)
print("COMPREHENSIVE VALIDATION SET COMPARISON (320 Images)")
print("=" * 70)
print(f"Metric          | Phase 4 (Baseline) | Phase 5 (Refined)  | Delta")
print("-" * 70)
print(f"SSIM (mean)     | {np.mean(p4_ssim):.4f} +/- {np.std(p4_ssim):.4f}   | {np.mean(p5_ssim):.4f} +/- {np.std(p5_ssim):.4f}   | {np.mean(p5_ssim)-np.mean(p4_ssim):+.4f}")
print(f"PSNR (mean dB)  | {np.mean(p4_psnr):.2f} +/- {np.std(p4_psnr):.2f} dB  | {np.mean(p5_psnr):.2f} +/- {np.std(p5_psnr):.2f} dB  | {np.mean(p5_psnr)-np.mean(p4_psnr):+.2f} dB")
print(f"LPIPS (mean)    | {np.mean(p4_lpips):.4f} +/- {np.std(p4_lpips):.4f}   | {np.mean(p5_lpips):.4f} +/- {np.std(p5_lpips):.4f}   | {np.mean(p5_lpips)-np.mean(p4_lpips):+.4f}")
print("=" * 70)

# Analyze improvement rate across samples
improved_ssim_count = sum(1 for s in sample_deltas if s["delta_ssim"] > 0)
print(f"\n[*] Sample-Level Analysis:")
print(f"  - Images with improved SSIM: {improved_ssim_count} / {len(sample_deltas)} ({improved_ssim_count/len(sample_deltas)*100:.1f}%)")

# Inspect lowest SSIM samples from Phase 4 (previously weak samples)
sample_deltas_sorted_by_p4 = sorted(sample_deltas, key=lambda x: x["p4_ssim"])
print("\n[*] Inspection of Previously Weakest Samples (Lowest Phase 4 SSIM):")
for s in sample_deltas_sorted_by_p4[:8]:
    print(f"  - {s['filename']}: P4 SSIM = {s['p4_ssim']:.4f} -> P5 SSIM = {s['p5_ssim']:.4f} (Delta: {s['delta_ssim']:+.4f}) | P4 PSNR: {s['p4_psnr']:.2f} dB -> P5 PSNR: {s['p5_psnr']:.2f} dB")

# Save detailed comparative report
report_data = {
    "phase4": {
        "ssim_mean": float(np.mean(p4_ssim)),
        "ssim_std": float(np.std(p4_ssim)),
        "psnr_mean": float(np.mean(p4_psnr)),
        "psnr_std": float(np.std(p4_psnr)),
        "lpips_mean": float(np.mean(p4_lpips)),
        "lpips_std": float(np.std(p4_lpips))
    },
    "phase5": {
        "ssim_mean": float(np.mean(p5_ssim)),
        "ssim_std": float(np.std(p5_ssim)),
        "psnr_mean": float(np.mean(p5_psnr)),
        "psnr_std": float(np.std(p5_psnr)),
        "lpips_mean": float(np.mean(p5_lpips)),
        "lpips_std": float(np.std(p5_lpips))
    },
    "delta": {
        "ssim": float(np.mean(p5_ssim) - np.mean(p4_ssim)),
        "psnr": float(np.mean(p5_psnr) - np.mean(p4_psnr)),
        "lpips": float(np.mean(p5_lpips) - np.mean(p4_lpips))
    },
    "improved_fraction": improved_ssim_count / len(sample_deltas),
    "samples": sample_deltas
}

with open("visual_samples/phase5_comparison_report.json", "w") as f:
    json.dump(report_data, f, indent=2)
print("\n[*] Detailed comparative report saved to 'visual_samples/phase5_comparison_report.json'")
