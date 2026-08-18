"""
Step 4: Comprehensive Re-Evaluation Script
1. Evaluate new best model on all 320 validation samples (Single-Pass and 8-Fold D4 TTA).
2. Compute SSIM, PSNR, and LPIPS.
3. Compare bottom 15% outlier tail (48 images) before vs after.
4. Save report to visual_samples/phase4_hard_weighted_report.json.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from models.restoration_net import SemiconductorRestorationNet
from utils.dataset import SemiconductorDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips, LPIPSCalculator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

with open("checkpoints/val_filenames.json", "r") as f:
    val_filenames = json.load(f)

val_lr_paths = [os.path.join("data/train/NoisyLR", f) for f in val_filenames]
val_gt_paths = [os.path.join("data/train/GT", f) for f in val_filenames]

val_dataset = SemiconductorDataset(val_lr_paths, val_gt_paths, is_train=False, preload=True, patch_size=None)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load Phase 3 baseline model
model_p3 = SemiconductorRestorationNet().to(device)
ckpt_p3 = torch.load("checkpoints/phase3_best_model.pth", map_location=device, weights_only=True)
if "model_state_dict" in ckpt_p3:
    model_p3.load_state_dict(ckpt_p3["model_state_dict"])
else:
    model_p3.load_state_dict(ckpt_p3)
model_p3.eval()

# Load Phase 4 hard-weighted model
model_p4 = SemiconductorRestorationNet().to(device)
ckpt_p4 = torch.load("checkpoints/best_model.pth", map_location=device, weights_only=True)
if "model_state_dict" in ckpt_p4:
    model_p4.load_state_dict(ckpt_p4["model_state_dict"])
else:
    model_p4.load_state_dict(ckpt_p4)
model_p4.eval()

lpips_calc = LPIPSCalculator(device=str(device))


def forward_d4_tta(model, x_batch):
    preds = []
    for flip in [False, True]:
        for rot in [0, 1, 2, 3]:
            x = x_batch
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
    return torch.mean(torch.stack(preds, dim=0), dim=0)


# Run full evaluation
p3_single_ssim, p3_single_psnr, p3_single_lpips = [], [], []
p4_single_ssim, p4_single_psnr, p4_single_lpips = [], [], []
p4_tta_ssim, p4_tta_psnr, p4_tta_lpips = [], [], []

per_sample = []

print("[*] Evaluating Phase 3 vs Phase 4 (Single + TTA) on all 320 samples...")
with torch.no_grad():
    for lr, gt, fnames in val_loader:
        lr_dev = lr.to(device)
        p3_s = model_p3(lr_dev)
        p4_s = model_p4(lr_dev)
        p4_t = forward_d4_tta(model_p4, lr_dev)

        for b in range(lr.size(0)):
            fname = fnames[b]
            g = np.clip(gt[b, 0].numpy(), 0.0, 1.0)
            p3s_np = np.clip(p3_s[b, 0].cpu().numpy(), 0.0, 1.0)
            p4s_np = np.clip(p4_s[b, 0].cpu().numpy(), 0.0, 1.0)
            p4t_np = np.clip(p4_t[b, 0].cpu().numpy(), 0.0, 1.0)

            # Metrics
            s_p3s = calculate_ssim(p3s_np, g)
            p_p3s = calculate_psnr(p3s_np, g)
            l_p3s = calculate_lpips(p3s_np, g, calculator=lpips_calc)

            s_p4s = calculate_ssim(p4s_np, g)
            p_p4s = calculate_psnr(p4s_np, g)
            l_p4s = calculate_lpips(p4s_np, g, calculator=lpips_calc)

            s_p4t = calculate_ssim(p4t_np, g)
            p_p4t = calculate_psnr(p4t_np, g)
            l_p4t = calculate_lpips(p4t_np, g, calculator=lpips_calc)

            p3_single_ssim.append(s_p3s)
            p3_single_psnr.append(p_p3s)
            p3_single_lpips.append(l_p3s)

            p4_single_ssim.append(s_p4s)
            p4_single_psnr.append(p_p4s)
            p4_single_lpips.append(l_p4s)

            p4_tta_ssim.append(s_p4t)
            p4_tta_psnr.append(p_p4t)
            p4_tta_lpips.append(l_p4t)

            per_sample.append({
                "filename": fname,
                "p3_ssim": s_p3s,
                "p3_psnr": p_p3s,
                "p4_single_ssim": s_p4s,
                "p4_single_psnr": p_p4s,
                "p4_tta_ssim": s_p4t,
                "p4_tta_psnr": p_p4t,
            })

# Isolate bottom 15% (48 worst images under Phase 3)
sorted_by_p3 = sorted(per_sample, key=lambda x: x["p3_ssim"])
bottom_15 = sorted_by_p3[:48]
top_85 = sorted_by_p3[48:]

p3_b15_ssim = float(np.mean([x["p3_ssim"] for x in bottom_15]))
p4_s_b15_ssim = float(np.mean([x["p4_single_ssim"] for x in bottom_15]))
p4_t_b15_ssim = float(np.mean([x["p4_tta_ssim"] for x in bottom_15]))

print("\n" + "=" * 80)
print("COMPREHENSIVE BENCHMARK: PHASE 3 BASELINE vs PHASE 4 (HARD-WEIGHTED)")
print("=" * 80)
print(f"Dataset Split / Mode            | PSNR (dB) ^   | SSIM ^        | LPIPS v")
print("-" * 80)
print(f"Phase 3 Baseline (Single-Pass)  | {np.mean(p3_single_psnr):.2f} +/- {np.std(p3_single_psnr):.2f} dB | {np.mean(p3_single_ssim):.4f} +/- {np.std(p3_single_ssim):.4f} | {np.mean(p3_single_lpips):.4f}")
print(f"Phase 4 (Single-Pass)           | {np.mean(p4_single_psnr):.2f} +/- {np.std(p4_single_psnr):.2f} dB | {np.mean(p4_single_ssim):.4f} +/- {np.std(p4_single_ssim):.4f} | {np.mean(p4_single_lpips):.4f}")
print(f"Phase 4 + 8-Fold D4 TTA         | {np.mean(p4_tta_psnr):.2f} +/- {np.std(p4_tta_psnr):.2f} dB | {np.mean(p4_tta_ssim):.4f} +/- {np.std(p4_tta_ssim):.4f} | {np.mean(p4_tta_lpips):.4f}")
print("=" * 80)

print("\n" + "=" * 80)
print("OUTLIER TAIL (BOTTOM 15%, 48 IMAGES) SPECIFIC PERFORMANCE")
print("=" * 80)
print(f"Metric                          | Phase 3 Baseline | Phase 4 Single | Phase 4 + TTA | Net Gain")
print("-" * 80)
print(f"Bottom 15% Mean SSIM            |      {p3_b15_ssim:.4f}      |     {p4_s_b15_ssim:.4f}     |    {p4_t_b15_ssim:.4f}     | {p4_t_b15_ssim - p3_b15_ssim:+.4f}")
print(f"Top 85% Mean SSIM               |      {np.mean([x['p3_ssim'] for x in top_85]):.4f}      |     {np.mean([x['p4_single_ssim'] for x in top_85]):.4f}     |    {np.mean([x['p4_tta_ssim'] for x in top_85]):.4f}     | {np.mean([x['p4_tta_ssim'] for x in top_85]) - np.mean([x['p3_ssim'] for x in top_85]):+.4f}")
print("=" * 80)

# Save json report
report = {
    "overall": {
        "p3_single_ssim": float(np.mean(p3_single_ssim)),
        "p3_single_psnr": float(np.mean(p3_single_psnr)),
        "p3_single_lpips": float(np.mean(p3_single_lpips)),
        "p4_single_ssim": float(np.mean(p4_single_ssim)),
        "p4_single_psnr": float(np.mean(p4_single_psnr)),
        "p4_single_lpips": float(np.mean(p4_single_lpips)),
        "p4_tta_ssim": float(np.mean(p4_tta_ssim)),
        "p4_tta_psnr": float(np.mean(p4_tta_psnr)),
        "p4_tta_lpips": float(np.mean(p4_tta_lpips)),
    },
    "bottom_15_percent": {
        "p3_ssim": p3_b15_ssim,
        "p4_single_ssim": p4_s_b15_ssim,
        "p4_tta_ssim": p4_t_b15_ssim,
        "gain": p4_t_b15_ssim - p3_b15_ssim
    }
}
with open("visual_samples/phase4_hard_weighted_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("[*] Saved full report to 'visual_samples/phase4_hard_weighted_report.json'")
