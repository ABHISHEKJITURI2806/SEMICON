import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class SemiconDataset(Dataset):
    """
    PyTorch Dataset for Paired Semiconductor Image Restoration.
    NoisyLR: Low-resolution, noisy input image (.npy format)
    GT: Full-resolution, clean ground truth target image (.npy format)
    """
    def __init__(self, data_dir, is_train=True, val_split=0.1, crop_size=96, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.is_train = is_train
        self.crop_size = crop_size

        gt_dir = os.path.join(data_dir, 'train', 'GT')
        noisy_dir = os.path.join(data_dir, 'train', 'NoisyLR')

        gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))
        noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.npy')))

        assert len(gt_files) == len(noisy_files), f"Mismatch in files: {len(gt_files)} GT vs {len(noisy_files)} NoisyLR"

        pairs = list(zip(noisy_files, gt_files))
        random.seed(seed)
        random.shuffle(pairs)

        val_size = int(len(pairs) * val_split)
        if is_train:
            self.pairs = pairs[val_size:]
        else:
            self.pairs = pairs[:val_size]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]
        
        noisy_img = np.load(noisy_path).astype(np.float32)
        gt_img = np.load(gt_path).astype(np.float32)

        # Augmentation and Crop during training
        if self.is_train:
            h_noisy, w_noisy = noisy_img.shape
            cs = self.crop_size if self.crop_size < h_noisy else h_noisy

            # Random crop location on LR image
            top = random.randint(0, h_noisy - cs)
            left = random.randint(0, w_noisy - cs)

            # Corresponding GT crop (scale factor = 2)
            gt_top, gt_left = top * 2, left * 2
            gt_cs = cs * 2

            noisy_img = noisy_img[top:top+cs, left:left+cs]
            gt_img = gt_img[gt_top:gt_top+gt_cs, gt_left:gt_left+gt_cs]

            # Random Horizontal Flip
            if random.random() > 0.5:
                noisy_img = np.fliplr(noisy_img).copy()
                gt_img = np.fliplr(gt_img).copy()
            # Random Vertical Flip
            if random.random() > 0.5:
                noisy_img = np.flipud(noisy_img).copy()
                gt_img = np.flipud(gt_img).copy()
            # Random 90 degree rotation
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                noisy_img = np.rot90(noisy_img, rot_k).copy()
                gt_img = np.rot90(gt_img, rot_k).copy()

        # Add channel dimension: (H, W) -> (1, H, W)
        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0)

        return noisy_tensor, gt_tensor, os.path.basename(gt_path)
