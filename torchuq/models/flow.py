import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os, sys, shutil, copy, time
from torch.utils.data import Dataset, DataLoader



class NafFlow(nn.Module):
    def __init__(self, feature_size=20):
        super(NafFlow, self).__init__()
        self.feature_size = feature_size
        self.log_a = nn.Parameter(
            torch.randn(feature_size, 1) * 5
        )
        self.b = nn.Parameter(
            torch.randn(feature_size, 1)
        )
        self.logit_w = nn.Parameter(
            torch.randn(1, feature_size)
        )
        self.activation = torch.sigmoid
        self.inverse_activation = self.logit

    def logit(self, x):
        return torch.log(x) - torch.log1p(-x)

    def forward(self, x):
        a = torch.exp(self.log_a) #(d, 1)
        w = self.logit_w - self.logit_w.permute(1, 0)
        w = -torch.logsumexp(w, 1, keepdim=True)
        w = torch.exp(w)
        # print(w.max(), w.min(), w.sum())
        # check size (d, 1)

        y = a.unsqueeze(dim=0) * x.reshape(x.shape[0], -1, 1) + self.b.unsqueeze(dim=0) # (batch, d, 1)
        y = self.activation(y)
        jacobian = a.unsqueeze(dim=0) * y * (1-y)
        w = w.permute(1, 0) #(1, d)
        y = torch.matmul(w.unsqueeze(dim=0), y)
        jacobian = torch.matmul(w.unsqueeze(dim=0), jacobian) * (1./y + 1./ (1-y))
        y = y.reshape(y.shape[0], 1)
        jacobian = jacobian.reshape(y.shape[0], 1)
        log_det = torch.log(jacobian)
        y = self.inverse_activation(y)
        return y, log_det # (batch, 1)

    def invert(self, y, left=-10., right=10., stop_gap=1e-5, tolerance=100):
        step = 0
        left = torch.ones_like(y) * left
        right = torch.ones_like(y) * right
        with torch.no_grad():
            gap = (right - left).max()
            while gap > stop_gap and step < tolerance:
                mid = (right + left) / 2.
                # right_val, _ = self.forward(right)
                # left_val, _ = self.forward(left)
                mid_val, _ = self.forward(mid)
                # if mid_val < y:
                left = mid * torch.ones_like(mid) * (mid_val < y).float() + left * (1. - torch.ones_like(mid) * (mid_val < y).float())
                right = mid * torch.ones_like(mid) * (mid_val >= y).float() + right * (1. - torch.ones_like(mid) * (mid_val >= y).float())
                # else:
                    # right = mid
                step += 1
                gap = (right - left).max()
            if gap <= stop_gap:
                return mid
            else:
                print("no inverse found")
