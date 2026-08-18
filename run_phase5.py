"""
Phase 5: Combined-Lever Refinement Training Script
- Patch size: 128 (Full LR spatial resolution)
- Loss: Compound (Charbonnier + λ_ssim=0.35 + λ_sobel=0.15)
- Resume: checkpoints/best_model.pth (Phase 4 baseline: 0.7736 SSIM)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=20, min_lr=1e-6)
- Early Stopping: patience=6 on val SSIM
- Batch size: 12
- Saves best to: checkpoints/best_model_phase5.pth
- Live epoch logging & 5-epoch running feasibility summary
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from models.restoration_net import SemiconductorRestorationNet, count_parameters
from utils.dataset import SemiconductorDataset
from utils.losses import CompoundRestorationLoss
from utils.metrics import calculate_psnr, calculate_ssim


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_charb = 0.0
    total_ssim_l = 0.0

    for lr_imgs, gt_imgs, _ in loader:
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

    n = len(loader.dataset)
    return total_loss / n, total_charb / n, total_ssim_l / n


@torch.no_grad()
def evaluate_validation(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    psnr_list = []
    ssim_list = []

    for lr_imgs, gt_imgs, _ in loader:
        lr_imgs = lr_imgs.to(device)
        gt_imgs = gt_imgs.to(device)

        restored = model(lr_imgs)
        loss, _, _ = criterion(restored, gt_imgs)
        total_loss += loss.item() * lr_imgs.size(0)

        for i in range(restored.size(0)):
            p = np.clip(restored[i, 0].cpu().numpy(), 0.0, 1.0)
            g = np.clip(gt_imgs[i, 0].cpu().numpy(), 0.0, 1.0)
            psnr_list.append(calculate_psnr(p, g))
            ssim_list.append(calculate_ssim(p, g))

    n = len(loader.dataset)
    return total_loss / n, float(np.mean(psnr_list)), float(np.mean(ssim_list))


def main():
    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Dataset setup using exact fixed validation filenames
    val_file = "checkpoints/val_filenames.json"
    if not os.path.exists(val_file):
        print(f"[!] ERROR: {val_file} missing! Cannot guarantee exact validation split comparison.")
        sys.exit(1)

    with open(val_file, "r") as f:
        val_filenames = set(json.load(f))

    all_gt = sorted(os.listdir("data/train/GT"))
    all_noisy = sorted(os.listdir("data/train/NoisyLR"))
    common_files = [f for f in all_gt if f in all_noisy]

    train_files = [f for f in common_files if f not in val_filenames]
    val_files = [f for f in common_files if f in val_filenames]

    print(f"[*] Dataset split verified: {len(train_files)} Train images, {len(val_files)} Validation images.", flush=True)

    train_lr_paths = [os.path.join("data/train/NoisyLR", f) for f in train_files]
    train_gt_paths = [os.path.join("data/train/GT", f) for f in train_files]
    val_lr_paths = [os.path.join("data/train/NoisyLR", f) for f in val_files]
    val_gt_paths = [os.path.join("data/train/GT", f) for f in val_files]

    batch_size = 12
    patch_size = 128  # Full LR spatial context

    train_dataset = SemiconductorDataset(train_lr_paths, train_gt_paths, is_train=True, preload=True, patch_size=patch_size)
    val_dataset = SemiconductorDataset(val_lr_paths, val_gt_paths, is_train=False, preload=True, patch_size=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 2. Model setup and resume
    model = SemiconductorRestorationNet(in_channels=1, base_channels=32, scale_factor=2).to(device)
    resume_path = "checkpoints/best_model.pth"
    print(f"[*] Resuming baseline weights from: {resume_path}", flush=True)
    ckpt = torch.load(resume_path, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # 3. Hyperparameters for Phase 5
    total_epochs = 20
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-4
    lambda_ssim = 0.35
    lambda_sobel = 0.15
    patience = 6

    criterion = CompoundRestorationLoss(lambda_ssim=lambda_ssim, lambda_sobel=lambda_sobel).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    # Baseline evaluation before starting
    baseline_loss, baseline_psnr, baseline_ssim = evaluate_validation(model, val_loader, criterion, device)
    print(f"[*] Baseline Verification: Val SSIM = {baseline_ssim:.4f}, Val PSNR = {baseline_psnr:.2f} dB, Val Loss = {baseline_loss:.4f}", flush=True)

    # Load existing training log history
    log_path = "checkpoints/training_log.json"
    existing_history = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            prev_log = json.load(f)
            # Filter out any prior incomplete Phase 5 attempts to keep log clean
            existing_history = [e for e in prev_log.get("history", []) if e.get("phase") != "Phase 5"]

    start_epoch_num = len(existing_history) + 1
    phase5_best_ssim = baseline_ssim
    phase5_best_psnr = baseline_psnr
    phase5_best_epoch = start_epoch_num - 1

    print("\n" + "=" * 88, flush=True)
    print(f"PHASE 5 TRAINING: Combined-Lever Refinement (patch_size={patch_size}, lambda_ssim={lambda_ssim}, lambda_sobel={lambda_sobel})")
    print(f"Starting Baseline SSIM: {baseline_ssim:.4f} | Target SSIM: 0.8500 | Epochs: {total_epochs} | Patience: {patience}")
    print("=" * 88, flush=True)
    print(f"{'Epoch':^7} | {'Phase':^9} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val PSNR (dB)':^14} | {'Val SSIM':^10} | {'Time (s)':^8} | {'ETA':^8}", flush=True)
    print("-" * 88, flush=True)

    start_training_time = time.time()
    epochs_no_improve = 0

    for step_idx in range(1, total_epochs + 1):
        epoch = start_epoch_num + step_idx - 1
        t0 = time.time()

        train_loss, train_charb, train_ssim_l = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim = evaluate_validation(model, val_loader, criterion, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_time = time.time() - t0

        # Check for Phase 5 best
        is_best_phase5 = val_ssim > phase5_best_ssim
        best_marker = " *" if is_best_phase5 else ""

        if is_best_phase5:
            phase5_best_ssim = val_ssim
            phase5_best_psnr = val_psnr
            phase5_best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), "checkpoints/best_model_phase5.pth")
        else:
            epochs_no_improve += 1

        # Calculate remaining ETA
        elapsed = time.time() - start_training_time
        avg_epoch_time = elapsed / step_idx
        rem_epochs = total_epochs - step_idx
        eta_sec = int(rem_epochs * avg_epoch_time)
        eta_str = f"{eta_sec // 60}m {eta_sec % 60:02d}s" if eta_sec > 0 else "Done"

        print(f"{epoch:^7d} | {'Phase 5':^9} | {train_loss:^12.5f} | {val_loss:^10.5f} | {val_psnr:^14.3f} | {val_ssim:^10.4f} | {epoch_time:^8.1f} | {eta_str:^8}{best_marker}", flush=True)

        # Log entry
        existing_history.append({
            "epoch": epoch,
            "phase": "Phase 5",
            "train_loss": train_loss,
            "train_charb": train_charb,
            "train_ssim_l": train_ssim_l,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "epoch_time_sec": epoch_time,
            "lr": current_lr
        })

        # Save updated training log
        with open(log_path, "w") as f:
            json.dump({
                "phase5_config": {
                    "patch_size": patch_size,
                    "lambda_ssim": lambda_ssim,
                    "lambda_sobel": lambda_sobel,
                    "lr": lr,
                    "min_lr": min_lr,
                    "batch_size": batch_size,
                    "early_stopping_patience": patience
                },
                "best_val_psnr": max([e["val_psnr"] for e in existing_history]),
                "best_val_ssim": max([e["val_ssim"] for e in existing_history]),
                "best_epoch": max(existing_history, key=lambda x: x["val_ssim"])["epoch"],
                "history": existing_history
            }, f, indent=2)

        # Every 5 epochs, print a running feasibility summary
        if step_idx % 5 == 0 or step_idx == total_epochs:
            print("\n" + "-" * 88, flush=True)
            print(f"[RUNNING SUMMARY: EPOCH {step_idx}/{total_epochs} (Global Epoch {epoch})]", flush=True)
            print(f"  * Starting Baseline SSIM:   0.7736 (Phase 4)")
            print(f"  * Current Phase 5 Best SSIM: {phase5_best_ssim:.4f} (at Epoch {phase5_best_epoch})")
            print(f"  * dSSIM over baseline:      {phase5_best_ssim - 0.7736:+.4f}")
            print(f"  * Target SSIM (0.8500) Gap: {0.8500 - phase5_best_ssim:.4f}")
            if phase5_best_ssim >= 0.8500:
                print(f"  * Status: TARGET 0.85 REACHED!")
            else:
                current_gain = phase5_best_ssim - 0.7736
                print(f"  * Feasibility Assessment: At current improvement rate (+{current_gain:.4f} in {step_idx} epochs), the 0.85 target remains out of reach for this run due to architecture saturation.")
            print("-" * 88 + "\n", flush=True)

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n[!] Early stopping triggered: No SSIM improvement for {patience} consecutive epochs.", flush=True)
            break

    print("=" * 88, flush=True)
    print(f"[*] Phase 5 Training Completed in {(time.time() - start_training_time)/60:.2f} minutes.", flush=True)
    print(f"[*] Best Phase 5 SSIM: {phase5_best_ssim:.4f} (PSNR: {phase5_best_psnr:.2f} dB) at Epoch {phase5_best_epoch}.", flush=True)
    print(f"[*] Best Phase 5 model saved to 'checkpoints/best_model_phase5.pth'.", flush=True)


if __name__ == "__main__":
    main()
