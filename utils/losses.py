import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import SSIMLoss

class SobelEdgeLoss(nn.Module):
    """
    Computes Sobel gradient magnitude loss to enforce edge sharpness and fine structural detail restoration.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        pred_mag = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)

        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        target_mag = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)

        return F.l1_loss(pred_mag, target_mag)

class CompositeRestorationLoss(nn.Module):
    """
    Composite Loss for Semiconductor Image Restoration:
    L_total = L1_loss + alpha * SSIM_loss + beta * Sobel_Edge_loss
    """
    def __init__(self, alpha=0.2, beta=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.sobel_loss = SobelEdgeLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        edge = self.sobel_loss(pred, target)
        total = l1 + self.alpha * ssim + self.beta * edge
        return total, l1, ssim, edge
