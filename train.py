import os
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models.restoration_model import SemiconRestorationNet
from utils.dataset import SemiconDataset
from utils.losses import CompositeRestorationLoss
from utils.metrics import calculate_psnr, calculate_ssim

def train_model(data_dir='dataset', epochs=12, batch_size=32, lr=8e-4, crop_size=64, device=None):
    # Maximize CPU parallel performance
    cpus = os.cpu_count() or 4
    torch.set_num_threads(cpus)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Training SemiconRestorationNet on device: {device} ({cpus} threads) ---")

    # Create directories for saving weights
    os.makedirs('weights', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Datasets and Loaders
    train_dataset = SemiconDataset(data_dir=data_dir, is_train=True, val_split=0.1, crop_size=crop_size)
    val_dataset = SemiconDataset(data_dir=data_dir, is_train=False, val_split=0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Dataset Loaded: {len(train_dataset)} Training samples, {len(val_dataset)} Validation samples.")

    # Model, Optimizer, Loss, Scheduler
    model = SemiconRestorationNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CompositeRestorationLoss(alpha=0.2, beta=0.1)

    best_val_psnr = 0.0
    best_val_ssim = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_l1 = 0.0

        start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for noisy, gt, _ in pbar:
            noisy = noisy.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            output = model(noisy)

            loss, l1, ssim_l, edge = criterion(output, gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)
            running_l1 += l1.item() * noisy.size(0)

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'L1': f"{l1.item():.4f}"})

        scheduler.step()

        train_loss = running_loss / len(train_dataset)
        train_l1 = running_l1 / len(train_dataset)
        epoch_time = time.time() - start_time

        # Validation Phase
        model.eval()
        val_psnr_list = []
        val_ssim_list = []

        with torch.no_grad():
            for noisy, gt, _ in val_loader:
                noisy = noisy.to(device)
                outputs = model(noisy).cpu().numpy().squeeze(1)
                gts = gt.numpy().squeeze(1)

                for pred_img, gt_img in zip(outputs, gts):
                    p_score = calculate_psnr(pred_img, gt_img)
                    s_score = calculate_ssim(pred_img, gt_img)
                    val_psnr_list.append(p_score)
                    val_ssim_list.append(s_score)

        mean_val_psnr = np.mean(val_psnr_list)
        mean_val_ssim = np.mean(val_ssim_list)

        print(f"Epoch [{epoch}/{epochs}] ({epoch_time:.1f}s) - Train Loss: {train_loss:.4f} (L1: {train_l1:.4f}) | Val PSNR: {mean_val_psnr:.2f} dB | Val SSIM: {mean_val_ssim:.4f}")

        # Save Best Model Weights
        if mean_val_psnr > best_val_psnr:
            best_val_psnr = mean_val_psnr
            best_val_ssim = mean_val_ssim
            best_path = os.path.join('weights', 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f" --> Best Model Saved to {best_path} (PSNR: {best_val_psnr:.2f} dB, SSIM: {best_val_ssim:.4f})")

    # Save final model weights
    final_path = os.path.join('weights', 'semicon_restoration_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining Complete! Best Validation PSNR: {best_val_psnr:.2f} dB, SSIM: {best_val_ssim:.4f}")
    return best_val_psnr, best_val_ssim

if __name__ == '__main__':
    train_model(epochs=10, batch_size=32, crop_size=64)
