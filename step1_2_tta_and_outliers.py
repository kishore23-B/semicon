"""
Step 1 & Step 2:
1. Run 8-fold D4 TTA and report full val metrics (SSIM, PSNR, LPIPS).
2. Rank 320 validation samples, isolate bottom 15% (48 images), and extract 5 detailed triplet examples.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

model = SemiconductorRestorationNet().to(device)
ckpt = torch.load("checkpoints/best_model.pth", map_location=device, weights_only=True)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

lpips_calc = LPIPSCalculator(device=str(device))

# ------------------------------------------------------------------------------
# 1. EVALUATE SINGLE-PASS AND 8-FOLD TTA
# ------------------------------------------------------------------------------
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

single_psnr, single_ssim, single_lpips = [], [], []
tta_psnr, tta_ssim, tta_lpips = [], [], []
sample_records = []

raw_lrs, raw_preds, raw_preds_tta, raw_gts = [], [], [], []

print("[*] Computing Single-Pass and 8-Fold D4 TTA metrics on all 320 validation samples...")
with torch.no_grad():
    for lr, gt, fnames in val_loader:
        lr_dev = lr.to(device)
        p_single = model(lr_dev)
        p_tta = forward_d4_tta(model, lr_dev)

        for b in range(lr.size(0)):
            fname = fnames[b]
            l_np = lr[b, 0].numpy()
            g_np = np.clip(gt[b, 0].numpy(), 0.0, 1.0)
            ps_np = np.clip(p_single[b, 0].cpu().numpy(), 0.0, 1.0)
            pt_np = np.clip(p_tta[b, 0].cpu().numpy(), 0.0, 1.0)

            raw_lrs.append(l_np)
            raw_preds.append(ps_np)
            raw_preds_tta.append(pt_np)
            raw_gts.append(g_np)

            # Single metrics
            s_s = calculate_ssim(ps_np, g_np)
            p_s = calculate_psnr(ps_np, g_np)
            l_s = calculate_lpips(ps_np, g_np, calculator=lpips_calc)

            # TTA metrics
            s_t = calculate_ssim(pt_np, g_np)
            p_t = calculate_psnr(pt_np, g_np)
            l_t = calculate_lpips(pt_np, g_np, calculator=lpips_calc)

            single_ssim.append(s_s)
            single_psnr.append(p_s)
            single_lpips.append(l_s)

            tta_ssim.append(s_t)
            tta_psnr.append(p_t)
            tta_lpips.append(l_t)

            # Record sample properties
            overshoot = float(np.sum(l_np > 1.0) + np.sum(l_np < 0.0)) / l_np.size
            sample_records.append({
                "filename": fname,
                "single_ssim": s_s,
                "single_psnr": p_s,
                "single_lpips": l_s,
                "tta_ssim": s_t,
                "tta_psnr": p_t,
                "tta_lpips": l_t,
                "gt_mean": float(np.mean(g_np)),
                "gt_std": float(np.std(g_np)),
                "overshoot_frac": overshoot,
                "min_lr": float(np.min(l_np)),
                "max_lr": float(np.max(l_np))
            })

print("\n" + "=" * 75)
print("STEP 1 RESULTS: TEST-TIME AUGMENTATION (TTA) ON VALIDATION SET")
print("=" * 75)
print(f"Metric          | Single-Pass        | 8-Fold D4 TTA      | TTA Gain")
print("-" * 75)
print(f"SSIM (mean)     | {np.mean(single_ssim):.4f} +/- {np.std(single_ssim):.4f}  | {np.mean(tta_ssim):.4f} +/- {np.std(tta_ssim):.4f}  | {np.mean(tta_ssim)-np.mean(single_ssim):+.4f}")
print(f"PSNR (mean dB)  | {np.mean(single_psnr):.2f} +/- {np.std(single_psnr):.2f} dB | {np.mean(tta_psnr):.2f} +/- {np.std(tta_psnr):.2f} dB | {np.mean(tta_psnr)-np.mean(single_psnr):+.2f} dB")
print(f"LPIPS (mean)    | {np.mean(single_lpips):.4f} +/- {np.std(single_lpips):.4f}  | {np.mean(tta_lpips):.4f} +/- {np.std(tta_lpips):.4f}  | {np.mean(tta_lpips)-np.mean(single_lpips):+.4f}")
print("=" * 75)

# ------------------------------------------------------------------------------
# 2. IDENTIFY & ISOLATE BOTTOM 15% WORST CASES
# ------------------------------------------------------------------------------
sorted_records = sorted(sample_records, key=lambda x: x["single_ssim"])
n_bottom_15 = int(len(sorted_records) * 0.15)  # 48 images
bottom_15 = sorted_records[:n_bottom_15]
top_85 = sorted_records[n_bottom_15:]

print("\n" + "=" * 75)
print(f"STEP 2 RESULTS: OUTLIER TAIL ISOLATION (Bottom 15% = {n_bottom_15} images)")
print("=" * 75)
print(f"  * Bottom 15% Mean SSIM: {np.mean([r['single_ssim'] for r in bottom_15]):.4f} (Min: {bottom_15[0]['single_ssim']:.4f}, Max: {bottom_15[-1]['single_ssim']:.4f})")
print(f"  * Top 85% Mean SSIM:    {np.mean([r['single_ssim'] for r in top_85]):.4f} (Median: {np.median([r['single_ssim'] for r in top_85]):.4f})")
print(f"  * Overall Mean SSIM:   {np.mean(single_ssim):.4f}")

print("\n--- 5 Characteristic Worst-Case Example Triplets ---")
print(f"{'Filename':^12} | {'SSIM (Single)':^14} | {'SSIM (TTA)':^12} | {'PSNR (dB)':^10} | {'LR Dynamic Range':^20} | {'Overshoot %':^12}")
print("-" * 88)
example_indices = [0, 1, 2, 5, 8]  # Representative worst cases
selected_examples = [bottom_15[k] for k in example_indices]
for r in selected_examples:
    print(f"{r['filename']:^12} | {r['single_ssim']:^14.4f} | {r['tta_ssim']:^12.4f} | {r['single_psnr']:^10.2f} | [{r['min_lr']:.2f}, {r['max_lr']:.2f}] | {r['overshoot_frac']*100:^11.1f}%")

# Generate visualization figure of the 5 triplets
fig, axes = plt.subplots(5, 4, figsize=(14, 16))
for row_idx, r in enumerate(selected_examples):
    fname = r["filename"]
    orig_idx = [k for k, x in enumerate(sample_records) if x["filename"] == fname][0]
    lr_arr = raw_lrs[orig_idx]
    pred_arr = raw_preds[orig_idx]
    pred_tta_arr = raw_preds_tta[orig_idx]
    gt_arr = raw_gts[orig_idx]

    # Noisy LR
    axes[row_idx, 0].imshow(np.clip(lr_arr, 0.0, 1.0), cmap="gray")
    axes[row_idx, 0].set_title(f"Input Noisy LR (128x128)\n{fname}\nRange: [{r['min_lr']:.2f}, {r['max_lr']:.2f}]")
    axes[row_idx, 0].axis("off")

    # Single-pass Restored
    axes[row_idx, 1].imshow(pred_arr, cmap="gray")
    axes[row_idx, 1].set_title(f"Restored (Single-Pass)\nSSIM: {r['single_ssim']:.4f}\nPSNR: {r['single_psnr']:.2f} dB")
    axes[row_idx, 1].axis("off")

    # TTA Restored
    axes[row_idx, 2].imshow(pred_tta_arr, cmap="gray")
    axes[row_idx, 2].set_title(f"Restored (8-Fold TTA)\nSSIM: {r['tta_ssim']:.4f}\nPSNR: {r['tta_psnr']:.2f} dB")
    axes[row_idx, 2].axis("off")

    # Ground Truth
    axes[row_idx, 3].imshow(gt_arr, cmap="gray")
    axes[row_idx, 3].set_title(f"Ground Truth HR (256x256)\nStd: {r['gt_std']:.4f}\nMean: {r['gt_mean']:.4f}")
    axes[row_idx, 3].axis("off")

plt.tight_layout()
plt.savefig("visual_samples/worst_case_triplets_detailed.png", dpi=150)
plt.close()
print(f"[*] Visualized 5 worst-case triplets to 'visual_samples/worst_case_triplets_detailed.png'")

# Save diagnosis metadata
with open("visual_samples/tail_outliers_diagnosis.json", "w") as f:
    json.dump({
        "single_mean_ssim": float(np.mean(single_ssim)),
        "tta_mean_ssim": float(np.mean(tta_ssim)),
        "bottom_15_mean_ssim": float(np.mean([r['single_ssim'] for r in bottom_15])),
        "top_85_mean_ssim": float(np.mean([r['single_ssim'] for r in top_85])),
        "bottom_15_samples": bottom_15
    }, f, indent=2)
