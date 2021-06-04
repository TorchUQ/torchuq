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
from .basic import Calibrator


class TemperatureScaling(Calibrator):
    def __init__(self, verbose=False):  # Algorithm can be hb (histogram binning) or kernel 
        super(TemperatureScaling, self).__init__(input_type='categorical')
        self.verbose = verbose
        self.temperature = None
        
    def train(self, predictions, labels, num_classes=None):
        # Use gradient descent to find the optimal temperature
        # Can add bisection option in the future, since it should be considerably faster
        self.temperature = torch.ones(1, 1, requires_grad=True, device=predictions.device)
        optim = torch.optim.Adam([self.temperature], lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=3, threshold=1e-6, factor=0.5)
            
        log_prediction = torch.log(predictions + 1e-10).detach()
        
        for iteration in range(10000): # Iterate at most 10k iterations, but expect to stop early
            optim.zero_grad()
            adjusted_predictions = log_prediction / self.temperature
            loss = F.cross_entropy(adjusted_predictions, labels)
            loss.backward()
            optim.step()
            lr_scheduler.step(loss)
            if optim.param_groups[0]['lr'] < 1e-6:   # Hitchhike the lr scheduler to terminate if no progress
                break
            if self.verbose and iteration % 100 == 0:
                print("Iteration %d, lr=%.5f, NLL=%.3f" % (iteration, optim.param_groups[0]['lr'], loss.cpu().item()))

    def __call__(self, predictions):
        log_prediction = torch.log(predictions + 1e-10)
        return torch.softmax(log_prediction / self.temperature, dim=1)
    
    
    
    
    
class CalibratorDirichlet:
    """
    Device is the preferred device, all computation is executed on that device as much as possible, 
    even though the input tensors do not have to be on the device, 
    """
    def __init__(self, verbose=False, device=None):  
        self.calibrator = None
        self.device = device
        self.verbose = verbose
    
    # The input to this has to be log probabilities
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

    
    