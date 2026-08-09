import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) between two 2D numpy arrays [0.0, 1.0].
    """
    img1 = np.clip(img1, 0.0, data_range)
    img2 = np.clip(img2, 0.0, data_range)
    return psnr_func(img1, img2, data_range=data_range)

def calculate_ssim(img1, img2, data_range=1.0):
    """
    Calculates Structural Similarity Index (SSIM) between two 2D numpy arrays [0.0, 1.0].
    """
    img1 = np.clip(img1, 0.0, data_range)
    img2 = np.clip(img2, 0.0, data_range)
    return ssim_func(img1, img2, data_range=data_range)

class SSIMLoss(nn.Module):
    """
    Differentiable 2D SSIM Loss for PyTorch Tensors.
    """
    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]))
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()
