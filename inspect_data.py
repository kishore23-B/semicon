"""
Dataset Inspection Script for Semiconductor Inspection Image Restoration
Verifies dataset directory layout, pairing, shapes, dtypes, and value statistics.
"""

import os
import glob
import numpy as np


def inspect_dataset(data_dir="data"):
    print("=" * 60)
    print("SEMICONDUCTOR RESTORATION DATASET INSPECTION")
    print("=" * 60)

    train_gt_dir = os.path.join(data_dir, "train", "GT")
    train_lr_dir = os.path.join(data_dir, "train", "NoisyLR")
    test_lr_dir = os.path.join(data_dir, "test", "NoisyLR")

    gt_files = sorted(glob.glob(os.path.join(train_gt_dir, "*.npy")))
    lr_files = sorted(glob.glob(os.path.join(train_lr_dir, "*.npy")))
    test_files = sorted(glob.glob(os.path.join(test_lr_dir, "*.npy")))

    print(f"\n[Directory & File Counts]")
    print(f"  - Clean Ground Truth (Train GT)    : {len(gt_files)} files at '{train_gt_dir}'")
    print(f"  - Degraded Input (Train NoisyLR)   : {len(lr_files)} files at '{train_lr_dir}'")
    print(f"  - Degraded Input (Test NoisyLR)    : {len(test_files)} files at '{test_lr_dir}'")

    # Verify 1-to-1 pairing
    gt_names = [os.path.basename(f) for f in gt_files]
    lr_names = [os.path.basename(f) for f in lr_files]
    pairing_exact = (gt_names == lr_names)
    print(f"  - Exact 1-to-1 Filename Pairing    : {pairing_exact}")

    if not pairing_exact:
        diff = set(gt_names) ^ set(lr_names)
        print(f"    WARNING: Unpaired files found: {diff}")

    # Inspect sample pairs
    print(f"\n[Sample Inspection - First 3 Training Pairs]")
    for i in range(min(3, len(gt_files))):
        gt_arr = np.load(gt_files[i])
        lr_arr = np.load(lr_files[i])
        print(f"  Sample #{i+1} ({gt_names[i]}):")
        print(f"    NoisyLR : shape={lr_arr.shape}, dtype={lr_arr.dtype}, min={lr_arr.min():.4f}, max={lr_arr.max():.4f}, mean={lr_arr.mean():.4f}, std={lr_arr.std():.4f}")
        print(f"    GT      : shape={gt_arr.shape}, dtype={gt_arr.dtype}, min={gt_arr.min():.4f}, max={gt_arr.max():.4f}, mean={gt_arr.mean():.4f}, std={gt_arr.std():.4f}")

    print(f"\n[Sample Inspection - First 3 Test Inputs]")
    for i in range(min(3, len(test_files))):
        test_arr = np.load(test_files[i])
        fname = os.path.basename(test_files[i])
        print(f"  Test Sample #{i+1} ({fname}):")
        print(f"    NoisyLR : shape={test_arr.shape}, dtype={test_arr.dtype}, min={test_arr.min():.4f}, max={test_arr.max():.4f}, mean={test_arr.mean():.4f}, std={test_arr.std():.4f}")

    # Global statistics over entire dataset (sample all)
    print(f"\n[Dataset Global Statistics (3200 train images)]")
    lr_min_all, lr_max_all, lr_mean_all = float("inf"), float("-inf"), []
    gt_min_all, gt_max_all, gt_mean_all = float("inf"), float("-inf"), []

    for gt_f, lr_f in zip(gt_files, lr_files):
        gt = np.load(gt_f)
        lr = np.load(lr_f)
        lr_min_all = min(lr_min_all, float(lr.min()))
        lr_max_all = max(lr_max_all, float(lr.max()))
        lr_mean_all.append(float(lr.mean()))

        gt_min_all = min(gt_min_all, float(gt.min()))
        gt_max_all = max(gt_max_all, float(gt.max()))
        gt_mean_all.append(float(gt.mean()))

    print(f"  - Train NoisyLR: Global Min = {lr_min_all:.4f}, Global Max = {lr_max_all:.4f}, Overall Mean = {np.mean(lr_mean_all):.4f}")
    print(f"  - Train GT     : Global Min = {gt_min_all:.4f}, Global Max = {gt_max_all:.4f}, Overall Mean = {np.mean(gt_mean_all):.4f}")
    print(f"  - Scale Factor : {gt_arr.shape[0] // lr_arr.shape[0]}x ({lr_arr.shape} -> {gt_arr.shape})")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
