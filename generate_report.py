"""
Benchmark, Metrics Evaluation, Visual Comparison, and Architecture Diagram Generator.
Computes PSNR, SSIM, LPIPS on the held-out validation set and creates visual figures.
"""

import os
import glob
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from models.restoration_net import SemiconductorRestorationNet, count_parameters
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips, LPIPSCalculator


def evaluate_val_set(
    data_dir="data/train",
    checkpoint_path="checkpoints/best_model.pth",
    val_split_path="checkpoints/val_filenames.json",
    device_str="cpu"
):
    device = torch.device(device_str)
    print(f"[*] Evaluating on device: {device}")

    # Load validation filenames
    if os.path.exists(val_split_path):
        with open(val_split_path, "r") as f:
            val_fnames = json.load(f)
    else:
        # Fallback: sorted 10%
        all_gt = sorted(os.listdir(os.path.join(data_dir, "GT")))
        val_fnames = all_gt[: int(len(all_gt) * 0.1)]

    print(f"[*] Total validation samples: {len(val_fnames)}")

    # Load Model
    model = SemiconductorRestorationNet(in_channels=1, base_channels=32, scale_factor=2)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    lpips_calc = LPIPSCalculator(device=device_str)

    bicubic_psnrs, bicubic_ssims, bicubic_lpips = [], [], []
    model_psnrs, model_ssims, model_lpips = [], [], []
    latencies = []

    os.makedirs("visual_samples", exist_ok=True)

    print("[*] Computing validation metrics (Bicubic Baseline vs Model)...")
    for i, fname in enumerate(val_fnames):
        gt_path = os.path.join(data_dir, "GT", fname)
        lr_path = os.path.join(data_dir, "NoisyLR", fname)

        gt_arr = np.load(gt_path).astype(np.float32)
        lr_arr = np.load(lr_path).astype(np.float32)

        lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0).to(device)

        # 1. Bicubic Baseline
        lr_clamped = torch.clamp(lr_tensor, 0.0, 1.0)
        bicubic_t = F.interpolate(lr_clamped, scale_factor=2, mode="bicubic", align_corners=False)
        bicubic_arr = bicubic_t.squeeze().cpu().numpy()

        # 2. Model Inference with latency timing
        t0 = time.perf_counter()
        with torch.no_grad():
            restored_t = model(lr_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)
        restored_arr = restored_t.squeeze().cpu().numpy()

        # Compute Metrics
        bicubic_psnrs.append(calculate_psnr(bicubic_arr, gt_arr))
        bicubic_ssims.append(calculate_ssim(bicubic_arr, gt_arr))

        model_psnrs.append(calculate_psnr(restored_arr, gt_arr))
        model_ssims.append(calculate_ssim(restored_arr, gt_arr))

        # Sample LPIPS across subset (every 5th image for efficiency)
        if i % 5 == 0:
            bicubic_lpips.append(calculate_lpips(bicubic_arr, gt_arr, lpips_calc, device=device_str))
            model_lpips.append(calculate_lpips(restored_arr, gt_arr, lpips_calc, device=device_str))

        if (i + 1) % 50 == 0 or (i + 1) == len(val_fnames):
            print(f"    Processed {i+1}/{len(val_fnames)} samples...")

    # Summary results
    metrics_summary = {
        "bicubic_baseline": {
            "psnr_mean": float(np.mean(bicubic_psnrs)),
            "psnr_std": float(np.std(bicubic_psnrs)),
            "ssim_mean": float(np.mean(bicubic_ssims)),
            "ssim_std": float(np.std(bicubic_ssims)),
            "lpips_mean": float(np.mean(bicubic_lpips)),
            "lpips_std": float(np.std(bicubic_lpips)),
        },
        "restoration_model": {
            "psnr_mean": float(np.mean(model_psnrs)),
            "psnr_std": float(np.std(model_psnrs)),
            "ssim_mean": float(np.mean(model_ssims)),
            "ssim_std": float(np.std(model_ssims)),
            "lpips_mean": float(np.mean(model_lpips)),
            "lpips_std": float(np.std(model_lpips)),
        },
        "latency_cpu_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        },
        "num_val_samples": len(val_fnames),
        "total_parameters": count_parameters(model),
        "checkpoint_file_size_mb": float(os.path.getsize(checkpoint_path) / (1024 * 1024))
    }

    # Print Table
    print("\n" + "=" * 70)
    print("VALIDATION SET BENCHMARK RESULTS (320 Held-Out Semiconductor Images)")
    print("=" * 70)
    print(f"{'Method':<25} | {'PSNR (dB) ^':<14} | {'SSIM ^':<12} | {'LPIPS v':<12}")
    print("-" * 70)
    print(f"{'Bicubic Baseline':<25} | {metrics_summary['bicubic_baseline']['psnr_mean']:<6.2f} +/- {metrics_summary['bicubic_baseline']['psnr_std']:<5.2f} | {metrics_summary['bicubic_baseline']['ssim_mean']:<6.4f}     | {metrics_summary['bicubic_baseline']['lpips_mean']:<6.4f}")
    print(f"{'RestorationNet (Ours)':<25} | {metrics_summary['restoration_model']['psnr_mean']:<6.2f} +/- {metrics_summary['restoration_model']['psnr_std']:<5.2f} | {metrics_summary['restoration_model']['ssim_mean']:<6.4f}     | {metrics_summary['restoration_model']['lpips_mean']:<6.4f}")
    print("=" * 70)
    print(f"[*] Model Parameters: {metrics_summary['total_parameters']:,}")
    print(f"[*] Checkpoint Size:  {metrics_summary['checkpoint_file_size_mb']:.2f} MB")
    print(f"[*] CPU Latency:      {metrics_summary['latency_cpu_ms']['mean']:.2f} +/- {metrics_summary['latency_cpu_ms']['std']:.2f} ms per image")

    # Save validation metrics JSON
    with open("visual_samples/val_metrics_report.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


def plot_visual_comparisons(
    data_dir="data/train",
    checkpoint_path="checkpoints/best_model.pth",
    val_split_path="checkpoints/val_filenames.json",
    num_samples=4
):
    """Generates multi-sample before/after/GT visual comparison figure."""
    device = torch.device("cpu")
    model = SemiconductorRestorationNet(in_channels=1, base_channels=32, scale_factor=2)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    with open(val_split_path, "r") as f:
        val_fnames = json.load(f)

    # Pick representative samples
    selected_fnames = val_fnames[:num_samples]

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    plt.subplots_adjust(wspace=0.08, hspace=0.2)

    titles = [
        "(a) Degraded Input (128x128)\nSpeckle Noise + Low-Res",
        "(b) Bicubic Baseline (256x256)\nStandard Upsampling",
        "(c) Restored Model (256x256)\nOurs (Denoised + SR)",
        "(d) Ground Truth (256x256)\nClean High-Resolution"
    ]

    for row, fname in enumerate(selected_fnames):
        gt_arr = np.load(os.path.join(data_dir, "GT", fname)).astype(np.float32)
        lr_arr = np.load(os.path.join(data_dir, "NoisyLR", fname)).astype(np.float32)

        lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0)
        bicubic_t = F.interpolate(torch.clamp(lr_tensor, 0.0, 1.0), scale_factor=2, mode="bicubic", align_corners=False)
        bicubic_arr = bicubic_t.squeeze().numpy()

        with torch.no_grad():
            restored_arr = model(lr_tensor).squeeze().numpy()

        psnr_bic = calculate_psnr(bicubic_arr, gt_arr)
        ssim_bic = calculate_ssim(bicubic_arr, gt_arr)

        psnr_our = calculate_psnr(restored_arr, gt_arr)
        ssim_our = calculate_ssim(restored_arr, gt_arr)

        images = [np.clip(lr_arr, 0.0, 1.0), bicubic_arr, restored_arr, gt_arr]
        subtitles = [
            f"Sample: {fname}\nRange: [{lr_arr.min():.2f}, {lr_arr.max():.2f}]",
            f"PSNR: {psnr_bic:.2f} dB\nSSIM: {ssim_bic:.4f}",
            f"PSNR: {psnr_our:.2f} dB (+{psnr_our - psnr_bic:.2f})\nSSIM: {ssim_our:.4f}",
            "Reference Target\nPhysical [0.0, 1.0]"
        ]

        for col in range(4):
            ax = axes[row, col] if num_samples > 1 else axes[col]
            ax.imshow(images[col], cmap="gray", vmin=0.0, vmax=1.0)
            if row == 0:
                ax.set_title(titles[col], fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel(subtitles[col], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_path = "visual_samples/restoration_visual_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[*] Visual comparison figure saved to '{out_path}'.")


def draw_architecture_diagram():
    """Renders a clear, publication-quality architecture pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    ax.axis("off")

    # Define box styles
    box_blue = dict(boxstyle="round,pad=0.6", facecolor="#EBF5FB", edgecolor="#2E86C1", linewidth=2)
    box_purple = dict(boxstyle="round,pad=0.6", facecolor="#F4ECF7", edgecolor="#7D3C98", linewidth=2)
    box_green = dict(boxstyle="round,pad=0.6", facecolor="#E8F8F5", edgecolor="#17A589", linewidth=2)
    box_orange = dict(boxstyle="round,pad=0.6", facecolor="#FEF9E7", edgecolor="#D4AC0D", linewidth=2)
    box_gray = dict(boxstyle="round,pad=0.6", facecolor="#F2F4F4", edgecolor="#7F8C8D", linewidth=2)

    # Blocks
    ax.text(0.06, 0.5, "Input Image\n(128x128, float32)\nSpeckle Noisy LR", ha="center", va="center", bbox=box_gray, fontsize=10, fontweight="bold")
    ax.text(0.22, 0.5, "Shallow Conv\nConv2d(1→32) + GELU\n(128x128x32)", ha="center", va="center", bbox=box_blue, fontsize=9)
    ax.text(0.40, 0.5, "Residual U-Net Body\nMulti-Scale RCAB Groups\n(32→64→96→64→32)\nChannel Attention", ha="center", va="center", bbox=box_purple, fontsize=9)
    ax.text(0.60, 0.5, "PixelShuffle Head\nConv2d(32→128)\nPixelShuffle(2x)\n(256x256x32)", ha="center", va="center", bbox=box_green, fontsize=9)
    ax.text(0.78, 0.5, "HR Refinement\nConv2d(32→16→1)\n(256x256x1)", ha="center", va="center", bbox=box_orange, fontsize=9)
    ax.text(0.94, 0.5, "Restored Output\n(256x256, [0,1])\nClean Denoised HR", ha="center", va="center", bbox=box_green, fontsize=10, fontweight="bold")

    # Global Bicubic Skip
    ax.text(0.50, 0.85, "Global Bicubic Baseline Skip Connection: Clamp(x, 0, 1) -> Upsample(2x, bicubic)", ha="center", va="center", bbox=box_blue, fontsize=9)
    ax.text(0.86, 0.5, "+\nClamp[0,1]", ha="center", va="center", fontsize=14, fontweight="bold")

    # Arrows
    arrow_kw = dict(arrowstyle="->", lw=2, color="#2C3E50")
    ax.annotate("", xy=(0.14, 0.5), xytext=(0.11, 0.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.30, 0.5), xytext=(0.27, 0.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.50, 0.5), xytext=(0.47, 0.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.69, 0.5), xytext=(0.66, 0.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.84, 0.5), xytext=(0.83, 0.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.88, 0.5), xytext=(0.87, 0.5), arrowprops=arrow_kw)

    # Skip arrow
    ax.annotate("", xy=(0.14, 0.85), xytext=(0.06, 0.62), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2E86C1", connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    ax.annotate("", xy=(0.86, 0.58), xytext=(0.75, 0.85), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2E86C1", connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    ax.set_title("SemiconductorRestorationNet Architecture Pipeline", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    diagram_path = "visual_samples/architecture_pipeline.png"
    plt.savefig(diagram_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[*] Architecture pipeline diagram saved to '{diagram_path}'.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark reports and visual comparisons")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_v4.pth", help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, default="data/train", help="Path to train data directory")
    parser.add_argument("--val_split", type=str, default="checkpoints/val_filenames.json", help="Path to validation split json")
    parser.add_argument("--device", type=str, default="cpu", help="Device for evaluation ('cpu' or 'cuda')")
    args = parser.parse_args()

    print("=" * 60)
    print(f"GENERATING BENCHMARKS FOR: {args.checkpoint}")
    print("=" * 60)
    evaluate_val_set(data_dir=args.data_dir, checkpoint_path=args.checkpoint, val_split_path=args.val_split, device_str=args.device)
    plot_visual_comparisons(data_dir=args.data_dir, checkpoint_path=args.checkpoint, val_split_path=args.val_split)
    draw_architecture_diagram()
    print("=" * 60)
    print("[*] All evaluation artifacts successfully generated.")


if __name__ == "__main__":
    main()
