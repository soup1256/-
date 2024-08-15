import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_features = vgg19(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg_features)[:36])
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        sr_features = self.layers(sr)
        hr_features = self.layers(hr)
        perceptual_loss = F.mse_loss(sr_features, hr_features)
        return perceptual_loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        sobel_kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_filter.weight = nn.Parameter(sobel_kernel, requires_grad=False)

    def forward(self, sr, hr):
        sr_gray = torch.mean(sr, dim=1, keepdim=True)
        hr_gray = torch.mean(hr, dim=1, keepdim=True)
        sr_edges = self.sobel_filter(sr_gray)
        hr_edges = self.sobel_filter(hr_gray)
        return F.l1_loss(sr_edges, hr_edges)

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.edge_loss = EdgeLoss()

    def forward(self, sr, hr):
        l1 = self.l1_loss(sr, hr)
        perceptual = self.perceptual_loss(sr, hr)
        edge = self.edge_loss(sr, hr)
        return l1 + 0.1 * perceptual + 0.1 * edge
