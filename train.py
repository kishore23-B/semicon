"""
Training Script for Semiconductor Inspection Image Restoration Network
=====================================================================
Joint Denoising and 2x Super-Resolution.
Trains from scratch or fine-tunes from a pre-trained checkpoint with
re-configured LR schedule, early stopping, and full training history logging.
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.restoration_net import SemiconductorRestorationNet, count_parameters
from utils.dataset import get_train_val_loaders
from utils.losses import CompoundRestorationLoss
from utils.metrics import calculate_psnr, calculate_ssim


def set_seed(seed: int = 42):
    """Sets random seeds across Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
):
    model.train()
    total_loss = 0.0
    total_charb = 0.0
    total_ssim_l = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for lr_imgs, gt_imgs, _ in pbar:
        lr_imgs = lr_imgs.to(device)
        gt_imgs = gt_imgs.to(device)

        optimizer.zero_grad()
        restored = model(lr_imgs)
        loss, l_charb, l_ssim = criterion(restored, gt_imgs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = lr_imgs.size(0)
        total_loss += loss.item() * batch_size
        total_charb += l_charb.item() * batch_size
        total_ssim_l += l_ssim.item() * batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "charb": f"{l_charb.item():.4f}"})

    n = len(loader.dataset)
    return total_loss / n, total_charb / n, total_ssim_l / n


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    model.eval()
    total_loss = 0.0
    psnr_list = []
    ssim_list = []

    for lr_imgs, gt_imgs, _ in tqdm(loader, desc="Validating", leave=False):
        lr_imgs = lr_imgs.to(device)
        gt_imgs = gt_imgs.to(device)

        restored = model(lr_imgs)
        loss, _, _ = criterion(restored, gt_imgs)

        total_loss += loss.item() * lr_imgs.size(0)

        for i in range(restored.size(0)):
            p = restored[i, 0].cpu().numpy()
            g = gt_imgs[i, 0].cpu().numpy()
            psnr_list.append(calculate_psnr(p, g))
            ssim_list.append(calculate_ssim(p, g))

    n = len(loader.dataset)
    val_loss = total_loss / n
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))

    return val_loss, mean_psnr, mean_ssim


def main():
    parser = argparse.ArgumentParser(description="Train Semiconductor Restoration Network")
    parser.add_argument("--data_dir", type=str, default="data/train", help="Path to training data folder containing GT/ and NoisyLR/")
    parser.add_argument("--epochs", type=int, default=60, help="Total epochs for new run or additional epochs if resuming")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--patch_size", type=int, default=96, help="LR patch crop size for training (None for full 128x128)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate for this training session")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for Cosine Annealing")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set split fraction (0.1 = 320 images)")
    parser.add_argument("--lambda_ssim", type=float, default=0.25, help="Weight for SSIM structural loss component")
    parser.add_argument("--lambda_sobel", type=float, default=0.2, help="Weight for Sobel gradient loss component")
    parser.add_argument("--save_best_as", type=str, default="best_model_phase5.pth", help="Filename to save the best model weights to")
    parser.add_argument("--monitor_metric", type=str, default="ssim", choices=["psnr", "ssim"], help="Metric to monitor for early stopping and best checkpoint selection")
    parser.add_argument("--phase_name", type=str, default="Phase 5", help="Phase tag name for training history log")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for exact reproducibility")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience in epochs")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")

    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Multi-threading optimization
    if hasattr(os, "cpu_count") and os.cpu_count():
        torch.set_num_threads(os.cpu_count())

    # Device selection
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training device: {device} (CPU threads: {torch.get_num_threads()})", flush=True)

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data Loaders
    print(f"[*] Loading dataset from '{args.data_dir}' (Validation ratio: {args.val_ratio}, Training Patch Size: {args.patch_size}x{args.patch_size})...", flush=True)
    train_loader, val_loader, train_files, val_files = get_train_val_loaders(
        train_dir=args.data_dir,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        preload=True,
        patch_size=args.patch_size
    )
    print(f"[*] Dataset split: {len(train_loader.dataset)} Train images, {len(val_loader.dataset)} Validation images.", flush=True)

    # Save validation split file list for reproducible evaluation reporting
    val_split_path = os.path.join(args.checkpoint_dir, "val_filenames.json")
    if not os.path.exists(val_split_path):
        with open(val_split_path, "w") as f:
            json.dump(val_files, f, indent=2)

    # Initialize Model
    model = SemiconductorRestorationNet(in_channels=1, base_channels=32, scale_factor=2).to(device)
    total_params = count_parameters(model)
    print(f"[*] Model initialized: {total_params:,} trainable parameters (~{total_params * 4 / (1024**2):.2f} MB float32 weights)", flush=True)

    # Check for resuming
    start_epoch = 1
    best_val_psnr = -float("inf")
    best_epoch = 0
    training_history = []
    log_path = os.path.join(args.checkpoint_dir, "training_log.json")

    if args.resume_checkpoint is not None and os.path.isfile(args.resume_checkpoint):
        print(f"[*] Resuming weights from: {args.resume_checkpoint}", flush=True)
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        # If existing training_log.json exists, load history
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                prev_log = json.load(f)
                training_history = prev_log.get("history", [])
                best_val_psnr = prev_log.get("best_val_psnr", -float("inf"))
                best_epoch = prev_log.get("best_epoch", 0)
                start_epoch = len(training_history) + 1
        print(f"[*] Resumed training from epoch {start_epoch}. Prior best PSNR: {best_val_psnr:.4f} dB at Epoch {best_epoch}.", flush=True)

    end_epoch = start_epoch + args.epochs - 1
    best_val_metric_session = -float("inf")

    # Loss, Optimizer, Scheduler configured for the new training session
    criterion = CompoundRestorationLoss(lambda_ssim=args.lambda_ssim, lambda_sobel=args.lambda_sobel).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    print("\n" + "=" * 80, flush=True)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val PSNR (dB)':^14} | {'Val SSIM':^10} | {'Time (s)':^8} | {'ETA':^8}", flush=True)
    print("=" * 80, flush=True)

    start_train_time = time.time()
    epochs_without_improvement = 0

    for current_step, epoch in enumerate(range(start_epoch, end_epoch + 1), start=1):
        t0 = time.time()
        train_loss, train_charb, train_ssim_l = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim = evaluate_validation(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]

        # Metric to monitor
        current_metric = val_ssim if args.monitor_metric == "ssim" else val_psnr

        # Check for new global best (across all training history)
        is_best_global = val_psnr > best_val_psnr
        if is_best_global:
            best_val_psnr = val_psnr
            best_epoch = epoch

        # Check for session-best (to save to custom filename)
        is_best_session = current_metric > best_val_metric_session
        best_marker = " *" if is_best_session else ""
        if is_best_session:
            best_val_metric_session = current_metric
            epochs_without_improvement = 0
            best_ckpt_path = os.path.join(args.checkpoint_dir, args.save_best_as)
            torch.save(model.state_dict(), best_ckpt_path)
            print(f" [*] Saved new best session checkpoint ({args.monitor_metric.upper()}: {current_metric:.4f}) to {best_ckpt_path}", flush=True)
        else:
            epochs_without_improvement += 1

        # Always save latest
        latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim
        }, latest_ckpt_path)

        # ETA calculation
        remaining_epochs = end_epoch - epoch
        avg_time = (time.time() - start_train_time) / current_step
        eta_sec = int(remaining_epochs * avg_time)
        eta_str = f"{eta_sec // 60}m {eta_sec % 60}s" if eta_sec > 0 else "Done"

        print(f"{epoch:^7d} | {train_loss:^12.5f} | {val_loss:^10.5f} | {val_psnr:^14.3f} | {val_ssim:^10.4f} | {epoch_time:^8.1f} | {eta_str:^8}{best_marker}", flush=True)

        phase_name = args.phase_name
        training_history.append({
            "epoch": epoch,
            "phase": phase_name,
            "train_loss": train_loss,
            "train_charb": train_charb,
            "train_ssim_l": train_ssim_l,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "epoch_time_sec": epoch_time,
            "lr": current_lr
        })

        # Save training log dynamically per epoch so no progress is lost
        with open(log_path, "w") as f:
            json.dump({
                "args": vars(args),
                "total_params": total_params,
                "best_val_psnr": best_val_psnr,
                "best_epoch": best_epoch,
                "total_duration_sec": time.time() - start_train_time,
                "history": training_history
            }, f, indent=2)

        # Early stopping check
        if epochs_without_improvement >= args.early_stopping_patience:
            print("=" * 80, flush=True)
            print(f"[!] Early stopping triggered! No {args.monitor_metric.upper()} improvement for {args.early_stopping_patience} consecutive epochs.", flush=True)
            print(f"[!] Optimal session {args.monitor_metric.upper()}: {best_val_metric_session:.4f}.", flush=True)
            break

    total_duration = time.time() - start_train_time
    print("=" * 80, flush=True)
    print(f"[*] Training finished in {total_duration:.1f}s ({total_duration/60:.2f} mins).", flush=True)
    print(f"[*] Best validation PSNR: {best_val_psnr:.3f} dB at Epoch {best_epoch}.", flush=True)
    print(f"[*] Best model saved to '{os.path.join(args.checkpoint_dir, 'best_model.pth')}'.", flush=True)
    print(f"[*] Training history saved to '{log_path}'.", flush=True)


if __name__ == "__main__":
    main()
