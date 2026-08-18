import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from models.restoration_net import SemiconductorRestorationNet
from utils.dataset import SemiconductorDataset
from torch.utils.data import DataLoader
from utils.metrics import calculate_psnr, calculate_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

psnr_list, ssim_list = [], []
with torch.no_grad():
    for lr, gt, fnames in val_loader:
        lr, gt = lr.to(device), gt.to(device)
        pred = model(lr)
        for i in range(pred.size(0)):
            p = np.clip(pred[i, 0].cpu().numpy(), 0.0, 1.0)
            g = np.clip(gt[i, 0].cpu().numpy(), 0.0, 1.0)
            psnr_list.append(calculate_psnr(p, g))
            ssims = calculate_ssim(p, g)
            ssim_list.append(ssims)

print(f"=== Baseline Model (best_model.pth) on {len(val_filenames)} Val Images ===")
print(f"Validation SSIM: {np.mean(ssim_list):.4f} (std: {np.std(ssim_list):.4f})")
print(f"Validation PSNR: {np.mean(psnr_list):.4f} dB (std: {np.std(psnr_list):.4f})")
