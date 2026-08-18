"""
Official Submission Entrypoint for KLA Problem Statement:
AI-Based Restoration of Degraded Semiconductor Images
=============================================================================
Usage:
    python run.py <input-dir> <output-dir>
    python run.py --input_dir <input-dir> --output_dir <output-dir>

Requirements satisfied:
✅ Reads all .npy files from the input directory.
✅ Creates the output directory if it does not already exist.
✅ Generates one restored .npy file for every input file with identical filename.
✅ Grayscale 2D float32 array output with shape (H, W).
✅ Output values strictly bounded in [0.0, 1.0], no NaN, no Inf.
✅ Correct 2x super-resolved target resolution (e.g., 256x256 from 128x128).
✅ Embedded weights included with zero internet access, no API keys, no downloads.
✅ Automatic GPU acceleration with CPU fallback.
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import torch

# Ensure local script directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.restoration_net import SemiconductorRestorationNet


def find_checkpoint_path():
    """Locate bundled model checkpoint with multiple relative fallbacks."""
    candidate_paths = [
        os.path.join(SCRIPT_DIR, "models", "best_model.pth"),
        os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth"),
        os.path.join(SCRIPT_DIR, "best_model.pth"),
        "models/best_model.pth",
        "checkpoints/best_model.pth",
        "best_model.pth"
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"Model checkpoint 'best_model.pth' not found in candidate locations: {candidate_paths}"
    )


def forward_d4_tta(model, x_tensor):
    """
    8-Fold D4 Dihedral Group Test-Time Augmentation (TTA).
    Averages predictions across all 8 spatial rotations and flips for maximum restoration accuracy.
    """
    preds = []
    for flip in [False, True]:
        for rot in [0, 1, 2, 3]:
            x = x_tensor
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


def restore_directory(input_dir: str, output_dir: str, device_str: str = None, use_tta: bool = True):
    # Determine execution device
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[*] Execution device: {device} ({device_name})")
    print(f"[*] Test-Time Augmentation (TTA): {'8-Fold D4 Dihedral (Active)' if use_tta else 'Disabled'}")

    # Validate input directory
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: '{input_dir}'")

    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if len(npy_files) == 0:
        # Check subdirectories if any
        npy_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.npy"), recursive=True))

    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in input directory: '{input_dir}'")

    print(f"[*] Found {len(npy_files)} .npy input files in '{input_dir}'")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Output directory ready: '{output_dir}'")

    # Locate and load model checkpoint
    ckpt_path = find_checkpoint_path()
    print(f"[*] Loading model weights from: '{ckpt_path}'...")
    
    model = SemiconductorRestorationNet(in_channels=1, base_channels=32, scale_factor=2)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Process all input images
    latencies = []
    print(f"[*] Processing and restoring {len(npy_files)} images...")
    t_start = time.perf_counter()

    with torch.no_grad():
        for idx, file_path in enumerate(npy_files):
            fname = os.path.basename(file_path)

            # Load input .npy array
            lr_arr = np.load(file_path).astype(np.float32)

            # Handle shape variants (H, W) or (H, W, 1) or (1, H, W)
            if lr_arr.ndim == 2:
                lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0).to(device)
            elif lr_arr.ndim == 3 and lr_arr.shape[-1] == 1:
                lr_tensor = torch.from_numpy(lr_arr.squeeze(-1)).unsqueeze(0).unsqueeze(0).to(device)
            elif lr_arr.ndim == 3 and lr_arr.shape[0] == 1:
                lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0).to(device)
            else:
                raise ValueError(f"Unexpected input array shape {lr_arr.shape} in '{file_path}'")

            # Forward inference
            t0 = time.perf_counter()
            if use_tta:
                restored_tensor = forward_d4_tta(model, lr_tensor)
            else:
                restored_tensor = model(lr_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

            # Convert to 2D numpy array (H, W)
            restored_arr = restored_tensor.squeeze().cpu().numpy().astype(np.float32)

            # Enforce strict bounded range [0.0, 1.0] and clean valid values
            restored_arr = np.clip(restored_arr, 0.0, 1.0)
            restored_arr = np.nan_to_num(restored_arr, nan=0.0, posinf=1.0, neginf=0.0)

            # Verification checks on output
            assert restored_arr.ndim == 2, f"Expected 2D array, got {restored_arr.shape}"
            assert not np.isnan(restored_arr).any(), f"NaN detected in output '{fname}'"
            assert not np.isinf(restored_arr).any(), f"Inf detected in output '{fname}'"
            assert restored_arr.min() >= 0.0 and restored_arr.max() <= 1.0, f"Out of bounds in '{fname}'"

            # Save restored .npy file with matching filename
            out_path = os.path.join(output_dir, fname)
            np.save(out_path, restored_arr)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(npy_files):
                print(f"    Restored {idx + 1}/{len(npy_files)} files...")

    total_time = time.perf_counter() - t_start
    avg_latency = float(np.mean(latencies))
    print("\n" + "=" * 60)
    print("RESTORATION COMPLETE - TECHNICAL CHECKS PASSED")
    print("=" * 60)
    print(f"[*] Total files restored: {len(npy_files)}")
    print(f"[*] Total elapsed time:   {total_time:.2f} seconds")
    print(f"[*] Average latency:      {avg_latency:.2f} ms per image")
    print(f"[*] All outputs saved to: '{output_dir}'")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="KLA Problem Statement: AI-Based Restoration of Degraded Semiconductor Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python run.py data/test/NoisyLR restored_outputs\n  python run.py --input_dir data/test/NoisyLR --output_dir restored_outputs"
    )

    # Support positional arguments (as required by specification)
    parser.add_argument("input_pos", nargs="?", default=None, help="Path to input directory containing degraded .npy files")
    parser.add_argument("output_pos", nargs="?", default=None, help="Path to output directory to save restored .npy files")

    # Support named arguments for flexibility
    parser.add_argument("-i", "--input_dir", type=str, default=None, help="Path to input directory")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Path to output directory")
    parser.add_argument("--device", type=str, default=None, help="Execution device: 'cpu' or 'cuda' (default: auto)")
    parser.add_argument("--no_tta", dest="use_tta", action="store_false", default=True, help="Disable 8-fold Test-Time Augmentation")

    args = parser.parse_args()

    input_dir = args.input_dir or args.input_pos
    output_dir = args.output_dir or args.output_pos

    if input_dir is None or output_dir is None:
        parser.print_help()
        print("\n[ERROR] Both input directory and output directory must be provided.")
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    restore_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        device_str=args.device,
        use_tta=args.use_tta
    )


if __name__ == "__main__":
    main()
