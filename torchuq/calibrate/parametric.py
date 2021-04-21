import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os, sys, shutil, copy, time, random
from torchuq.models.flow import NafFlow


class CalibratorDirichlet:
    """
    Device is the preferred device, all computation is executed on that device as much as possible, 
    even though the input tensors do not have to be on the device, 
    """
    def __init__(self, verbose=False, device=None):  
        self.calibrator = None
        self.device = device
        self.verbose = verbose
    
    def train(self, predictions, labels, num_epochs=100):            
        if self.device is not None:
            device = self.device
        else:
            device = predictions.device
            
        self.calibrator = nn.Linear(predictions.shape[1], predictions.shape[1]).to(device)
        calib_optim = optim.Adam(self.calibrator.parameters(), lr=1e-3)
        
        # Load the data sequentially because 1. SGD usually converges faster than GD 2. less memory use
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(predictions.cpu(), labels.cpu()), # Seems like pytorch dataset has to live in cpu memory 
                                                 batch_size=256, shuffle=True, num_workers=2)
        for epoch in range(num_epochs):
            for bp, bl in val_loader:
                calib_optim.zero_grad()
                bp, bl = bp.to(device), bl.to(device)
                bp_adjusted = self.calibrator(bp)
                
                nll_loss = F.cross_entropy(bp_adjusted, bl)
                nll_loss.backward()
                calib_optim.step()
            if self.verbose and epoch % 5 == 0:
                print("Finished training %d epochs, nll = %.3f" % (epoch, nll_loss.detach().cpu()))
    
    def __call__(self, predictions):
        if self.calibrator is None:
            print("Error: need to first train before calling this function")
        return torch.softmax(self.calibrator(predictions), dim=-1)

    
    