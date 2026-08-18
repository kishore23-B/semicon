import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))

with open("checkpoints/training_log.json", "r") as f:
    log_data = json.load(f)

history = log_data.get("history", [])
print(f"Total training log entries: {len(history)}")

# Group by phase
phases = {}
for entry in history:
    phase = entry.get("phase", "Unknown")
    if phase not in phases:
        phases[phase] = []
    phases[phase].append(entry)

print("\n--- Summary by Phase in History ---")
prev_best_ssim = 0.5564 # baseline bicubic
for phase_name, entries in phases.items():
    best_entry = max(entries, key=lambda x: x.get("val_ssim", 0.0))
    best_ssim = best_entry.get("val_ssim", 0.0)
    best_psnr = best_entry.get("val_psnr", 0.0)
    gain = best_ssim - prev_best_ssim
    print(f"Phase '{phase_name}': {len(entries)} epochs (Epochs {entries[0]['epoch']}–{entries[-1]['epoch']})")
    print(f"  Best Val SSIM: {best_ssim:.4f} (gain: +{gain:.4f}) | Best Val PSNR: {best_psnr:.2f} dB")
    prev_best_ssim = best_ssim
