import os
import sys
import json
import time
import torch
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))

print("=" * 60)
print("STEP 0 — ENVIRONMENT VERIFICATION ON THIS MACHINE")
print("=" * 60)

# 1. best_model.pth
pth = "checkpoints/best_model.pth"
if os.path.exists(pth):
    sz = os.path.getsize(pth)
    mtime = datetime.fromtimestamp(os.path.getmtime(pth)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"1. checkpoints/best_model.pth:")
    print(f"   - Status: EXISTS")
    print(f"   - File size: {sz:,} bytes ({sz / (1024 * 1024):.2f} MB)")
    print(f"   - Modification date: {mtime}")
else:
    print(f"1. checkpoints/best_model.pth: MISSING! (ERROR)")
    sys.exit(1)

# 2. training_log.json
log_path = "checkpoints/training_log.json"
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        log_data = json.load(f)
    history = log_data.get("history", [])
    print(f"\n2. checkpoints/training_log.json:")
    print(f"   - Status: EXISTS")
    print(f"   - Total history entries: {len(history)}")
    print(f"   - Last 5 entries:")
    for entry in history[-5:]:
        ep = entry.get("epoch")
        phase = entry.get("phase", "N/A")
        v_ssim = entry.get("val_ssim", 0.0)
        v_psnr = entry.get("val_psnr", 0.0)
        v_loss = entry.get("val_loss", 0.0)
        lr = entry.get("lr", 0.0)
        print(f"     * Epoch {ep:2d} | Phase: {phase:<8} | Val SSIM: {v_ssim:.4f} | Val PSNR: {v_psnr:.2f} dB | Val Loss: {v_loss:.4f} | LR: {lr:.2e}")
else:
    print(f"\n2. checkpoints/training_log.json: MISSING! (ERROR)")
    sys.exit(1)

# 3. Data counts
gt_dir = "data/train/GT"
noisy_dir = "data/train/NoisyLR"
test_dir = "data/test/NoisyLR"
gt_count = len([f for f in os.listdir(gt_dir) if f.endswith(".npy")]) if os.path.exists(gt_dir) else 0
noisy_count = len([f for f in os.listdir(noisy_dir) if f.endswith(".npy")]) if os.path.exists(noisy_dir) else 0
test_count = len([f for f in os.listdir(test_dir) if f.endswith(".npy")]) if os.path.exists(test_dir) else 0

print(f"\n3. Dataset file counts:")
print(f"   - data/train/GT:       {gt_count} files (Expected: 3200) -> {'OK' if gt_count == 3200 else 'MISMATCH'}")
print(f"   - data/train/NoisyLR:  {noisy_count} files (Expected: 3200) -> {'OK' if noisy_count == 3200 else 'MISMATCH'}")
print(f"   - data/test/NoisyLR:   {test_count} files (Expected: 400)  -> {'OK' if test_count == 400 else 'MISMATCH'}")

# 4. val_filenames.json
val_path = "checkpoints/val_filenames.json"
if os.path.exists(val_path):
    with open(val_path, "r") as f:
        val_files = json.load(f)
    print(f"\n4. checkpoints/val_filenames.json:")
    print(f"   - Status: EXISTS (Exact validation split preserved)")
    print(f"   - Validation samples count: {len(val_files)} files (10% of 3200 = 320 files)")
    print(f"   - Sample validation filenames: {val_files[:3]} ... {val_files[-2:]}")
else:
    print(f"\n4. checkpoints/val_filenames.json: MISSING! STOPPING per instructions.")
    sys.exit(1)

# 5. Device info & benchmark speed
print(f"\n5. Device Information & Benchmark:")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"   - GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"   - CUDA Version: {torch.version.cuda}")
    device = torch.device("cuda")
else:
    print(f"   - GPU: None (Running on CPU)")
    device = torch.device("cpu")

# Benchmark a single forward/backward pass with batch_size=12, patch_size=128
from models.restoration_net import SemiconductorRestorationNet
from utils.losses import CompoundRestorationLoss

model = SemiconductorRestorationNet().to(device)
criterion = CompoundRestorationLoss(lambda_ssim=0.35, lambda_sobel=0.15).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

dummy_lr = torch.randn(12, 1, 128, 128, device=device)
dummy_gt = torch.randn(12, 1, 256, 256, device=device)

# Warmup
for _ in range(3):
    optimizer.zero_grad()
    out = model(dummy_lr)
    loss, _, _ = criterion(out, dummy_gt)
    loss.backward()
    optimizer.step()
if torch.cuda.is_available():
    torch.cuda.synchronize()

# Time 10 batches
start = time.time()
n_bench = 10
for _ in range(n_bench):
    optimizer.zero_grad()
    out = model(dummy_lr)
    loss, _, _ = criterion(out, dummy_gt)
    loss.backward()
    optimizer.step()
if torch.cuda.is_available():
    torch.cuda.synchronize()
batch_time = (time.time() - start) / n_bench

# Train set has 2880 samples -> 2880 / 12 = 240 batches
est_train_time = 240 * batch_time
# Val set has 320 samples -> evaluate on 320 images (full 128x128 -> 256x256)
start_val = time.time()
with torch.no_grad():
    for _ in range(10):
        out = model(dummy_lr)
if torch.cuda.is_available():
    torch.cuda.synchronize()
val_batch_time = (time.time() - start_val) / 10
est_val_time = (320 / 12) * val_batch_time

total_est_epoch = est_train_time + est_val_time
print(f"   - Benchmark batch time (batch=12, patch=128): {batch_time*1000:.1f} ms / batch")
print(f"   - Estimated time per epoch on {device} GPU: {total_est_epoch:.1f} seconds (~{total_est_epoch/60:.2f} min)")
print("=" * 60)
