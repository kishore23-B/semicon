"""
Standalone Evaluation & Inference Script for Semiconductor Image Restoration
=============================================================================
Restores degraded semiconductor inspection images (.npy) by joint speckle denoising
and 2x super-resolution.

Usage:
    python evaluate.py --input_dir path/to/noisy_lr --output_dir path/to/restored_output
    python evaluate.py --input_dir data/test/NoisyLR --output_dir restored_test --save_png
"""

import os
import glob
import time
import argparse
import numpy as np
import torch
from PIL import Image

from models.restoration_net import SemiconductorRestorationNet


def restore_dataset(
    input_dir: str,
    output_dir: str,
    checkpoint_path: str = "checkpoints/best_model.pth",
    save_png: bool = True,
    device_str: str = None
):
    # Set up device
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference device: {device}")

    # Verify input directory
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in input directory: {input_dir}")
    print(f"[*] Found {len(npy_files)} .npy files in '{input_dir}'")

    # Verify checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    # Load Model
    print(f"[*] Loading model checkpoint from '{checkpoint_path}'...")
    model = SemiconductorRestorationNet(in_channels=1, base_channels=32, scale_factor=2)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    png_dir = os.path.join(output_dir, "png_previews") if save_png else None
    if save_png:
        os.makedirs(png_dir, exist_ok=True)

    latencies = []

    print(f"[*] Restoring images and saving to '{output_dir}'...")
    with torch.no_grad():
        for file_path in npy_files:
            fname = os.path.basename(file_path)
            
            # Load degraded float32 numpy array
            lr_arr = np.load(file_path).astype(np.float32)
            lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)

            # Measure inference time
            t0 = time.perf_counter()
            restored_tensor = model(lr_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(t_elapsed)

            # Convert to numpy array (H, W) in [0.0, 1.0]
            restored_arr = restored_tensor.squeeze().cpu().numpy().astype(np.float32)

            # Save restored .npy array
            out_npy_path = os.path.join(output_dir, fname)
            np.save(out_npy_path, restored_arr)

            # Save visual sanity check .png
            if save_png:
                png_name = os.path.splitext(fname)[0] + ".png"
                out_png_path = os.path.join(png_dir, png_name)
                uint8_img = np.clip(restored_arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
                Image.fromarray(uint8_img, mode="L").save(out_png_path)

    avg_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    print(f"[*] Finished restoring {len(npy_files)} images.")
    print(f"[*] Average inference latency: {avg_lat:.2f} +/- {std_lat:.2f} ms per image on {device}.")
    print(f"[*] Restored .npy outputs saved to: '{output_dir}'")
    if save_png:
        print(f"[*] Visual preview PNGs saved to: '{png_dir}'")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Inference & Evaluation for Semiconductor Image Restoration"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing degraded input .npy images (e.g. data/test/NoisyLR)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory where restored .npy files will be saved"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained model weights (.pth)"
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        default=True,
        help="Save uint8 visual preview .png files alongside .npy arrays (default: True)"
    )
    parser.add_argument(
        "--no_png",
        dest="save_png",
        action="store_false",
        help="Disable saving PNG preview files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu' or 'cuda' (default: auto-select)"
    )

    args = parser.parse_args()

    restore_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        save_png=args.save_png,
        device_str=args.device
    )


if __name__ == "__main__":
    main()
