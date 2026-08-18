"""
Dataset and DataLoader utilities with in-memory caching and patch cropping.
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


class SemiconductorDataset(Dataset):
    """
    In-memory cached paired dataset for semiconductor inspection image restoration.
    """
    def __init__(
        self,
        lr_paths: List[str],
        gt_paths: Optional[List[str]] = None,
        is_train: bool = False,
        preload: bool = True,
        patch_size: Optional[int] = None,
        scale: int = 2
    ):
        self.lr_paths = lr_paths
        self.gt_paths = gt_paths
        self.is_train = is_train
        self.preload = preload
        self.patch_size = patch_size
        self.scale = scale

        if self.gt_paths is not None:
            assert len(self.lr_paths) == len(self.gt_paths)

        self.lr_data = []
        self.gt_data = []
        self.filenames = [os.path.basename(p) for p in self.lr_paths]

        if self.preload:
            for i in range(len(self.lr_paths)):
                lr_arr = np.load(self.lr_paths[i]).astype(np.float32)
                self.lr_data.append(torch.from_numpy(lr_arr).unsqueeze(0))

                if self.gt_paths is not None:
                    gt_arr = np.load(self.gt_paths[i]).astype(np.float32)
                    self.gt_data.append(torch.from_numpy(gt_arr).unsqueeze(0))

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, idx: int):
        if self.preload:
            lr_tensor = self.lr_data[idx].clone()
            if self.gt_paths is None:
                return lr_tensor, self.filenames[idx]
            gt_tensor = self.gt_data[idx].clone()
        else:
            lr_arr = np.load(self.lr_paths[idx]).astype(np.float32)
            lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0)
            if self.gt_paths is None:
                return lr_tensor, self.filenames[idx]
            gt_arr = np.load(self.gt_paths[idx]).astype(np.float32)
            gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0)

        if self.is_train:
            # Random patch crop
            if self.patch_size is not None and lr_tensor.shape[-1] > self.patch_size:
                lr_h, lr_w = lr_tensor.shape[-2], lr_tensor.shape[-1]
                ps_lr = self.patch_size
                ps_gt = self.patch_size * self.scale
                
                top_lr = random.randint(0, lr_h - ps_lr)
                left_lr = random.randint(0, lr_w - ps_lr)
                
                top_gt = top_lr * self.scale
                left_gt = left_lr * self.scale
                
                lr_tensor = lr_tensor[:, top_lr:top_lr + ps_lr, left_lr:left_lr + ps_lr]
                gt_tensor = gt_tensor[:, top_gt:top_gt + ps_gt, left_gt:left_gt + ps_gt]

            # D4 Dihedral group augmentations
            lr_tensor, gt_tensor = self._apply_augmentations(lr_tensor, gt_tensor)

        return lr_tensor, gt_tensor, self.filenames[idx]

    @staticmethod
    def _apply_augmentations(lr: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            lr = torch.flip(lr, dims=[-1])
            gt = torch.flip(gt, dims=[-1])

        if random.random() > 0.5:
            lr = torch.flip(lr, dims=[-2])
            gt = torch.flip(gt, dims=[-2])

        k = random.randint(0, 3)
        if k > 0:
            lr = torch.rot90(lr, k=k, dims=[-2, -1])
            gt = torch.rot90(gt, k=k, dims=[-2, -1])

        return lr, gt


def get_train_val_loaders(
    train_dir: str = "data/train",
    val_ratio: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    preload: bool = True,
    patch_size: Optional[int] = 64
) -> Tuple[DataLoader, DataLoader, List[str], List[str]]:
    gt_dir = os.path.join(train_dir, "GT")
    lr_dir = os.path.join(train_dir, "NoisyLR")

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))

    lr_dict = {os.path.basename(f): f for f in lr_files}
    paired_gt = []
    paired_lr = []
    for gt_path in gt_files:
        base = os.path.basename(gt_path)
        if base in lr_dict:
            paired_gt.append(gt_path)
            paired_lr.append(lr_dict[base])

    rng = random.Random(seed)
    indices = list(range(len(paired_gt)))
    rng.shuffle(indices)

    num_val = int(len(indices) * val_ratio)
    val_indices = set(indices[:num_val])
    train_indices = [i for i in indices if i not in val_indices]

    train_gt = [paired_gt[i] for i in train_indices]
    train_lr = [paired_lr[i] for i in train_indices]

    val_gt = [paired_gt[i] for i in sorted(list(val_indices))]
    val_lr = [paired_lr[i] for i in sorted(list(val_indices))]

    train_dataset = SemiconductorDataset(train_lr, train_gt, is_train=True, preload=preload, patch_size=patch_size)
    val_dataset = SemiconductorDataset(val_lr, val_gt, is_train=False, preload=preload, patch_size=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, val_loader, [os.path.basename(f) for f in train_gt], [os.path.basename(f) for f in val_gt]
