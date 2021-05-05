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


class Calibrator:
    def __init__(self, input_type='auto'):
        """
        input_type should be one of the supported datatypes
        If input_type is 'auto' then it is automatically induced when Calibrator.train() or update() is first time called, it cannot be changed after the first call to train() or update()
        Input_type must be explicitly specificied when there is ambiguity
        """
        self.input_type = input_type
    
    # Input an array of shape predictions=[dataset_size, num_classes], labels=[dataset_size]
    # Optionally input side features such as an array of shape [dataset_size, num_features]. Not all calibrators consider side feature when recalibrating 
    def train(self, predictions, labels, side_features=None):
        pass
    
    # Same as train, but updates the calibrator online 
    # If half_life is not None, then it is the number of calls to this function where the sample is discounted to 1/2 weight
    # Not all calibration functions support half_life
    def update(self, predictions, labels, side_features=None, half_life=None):
        pass
    
    # Input an array of shape [batch_size, num_classes], output the recalibrated array
    # predictions should be in the same pytorch device 
    # If side_feature is not None when calling train, it shouldn't be None here either. 
    def __call__(self, predictions, side_features=None):
        pass
    
    def check_type(self, predictions):
        if self.input_type == 'point':
            assert len(predictions.shape) == 1, "Point prediction should have shape [batch_size]"
        elif self.input_type == 'interval':
            assert len(predictions.shape) == 2 and predictions.shape[1] == 2, "interval predictions should have shape [batch_size, 2]"
        elif self.input_type == 'quantile':
            assert len(predictions.shape) == 2 or (len(predictions.shape) == 3 and predictions.shape[2] == 2), "quantile predictions should have shape [batch_size, num_quantile] or [batch_size, num_quantile, 2]" 
            
    def assert_type(self, input_type, valid_types):
        msg = "Input data type not supported, input data type is %s, supported types are %s" % (input_type, " ".join(valid_types)) 
        assert input_type in valid_types, msg 
            


class HistogramBinning:
    def __init__(self, requires_grad=False, adaptive=True, bin_count='auto', verbose=False):  # Algorithm can be hb (histogram binning) or kernel 
        self.adaptive = adaptive
        self.bin_count = bin_count
        self.verbose = verbose
        self.bin_boundary = None
        self.bin_adjustment = None
        assert not requires_grad, "Histogram binning does not support gradient with respect to HB parameters"

    # side_feature is not used
    def train(self, predictions, labels, side_feature=None):
        with torch.no_grad():
            # Get the maximum confidence prediction and its confidence
            max_confidence, prediction = predictions.max(dim=1)
            # Indicator variable for whether each top 1 prediction is correct
            correct = (prediction == labels).type(torch.float32)
            
            if self.verbose:
                print("Top-1 accuracy of predictor is %.3f" % correct.mean()) 
            
            # Decide the number of histogram binning bins
            if self.bin_count == 'auto':
                self.bin_count = int(np.sqrt(len(predictions)) / 5.)
            if self.bin_count < 1:
                self.bin_count = 1
            if self.verbose:
                print("Number of histogram binning bins is %d" % self.bin_count)
            
            confidence_ranking = torch.argsort(max_confidence)
            
            # Compute the boundary of the bins
            indices = torch.linspace(0, len(predictions)-1, self.bin_count+1).round().type(torch.long).cpu()
            self.bin_boundary = max_confidence[confidence_ranking].cpu()[indices] 
            self.bin_boundary[0] = -1.0
            self.bin_boundary[-1] = 2.0
            if self.verbose:
                print(self.bin_boundary)
            
            # Compute the adjustment for each bin
            diff = max_confidence[confidence_ranking] - correct[confidence_ranking] 
            self.bin_adjustment = diff.view(self.bin_count, -1).mean(dim=1)

    def __call__(self, predictions, side_feature=None):
        if self.bin_boundary is None:
            print("Error: need to first call ConfidenceHBCalibrator.train before calling this function")
        with torch.no_grad():
            max_confidence, max_index = predictions.max(dim=1)
            max_confidence = max_confidence.view(-1, 1).repeat(1, len(self.bin_boundary))
            tiled_boundary = self.bin_boundary.to(predictions.device).view(1, -1).repeat(len(predictions), 1)
            tiled_adjustment = self.bin_adjustment.to(predictions.device).view(1, -1).repeat(len(predictions), 1)
            
            index = (max_confidence < tiled_boundary).type(torch.float32)
            index = index[:, 1:] - index[:, :-1]
#             print(index.shape)
            
#             print(index.sum(dim=1))
            adjustment = (tiled_adjustment * index).sum(dim=1)
            # print(adjustment.shape)

        predictions = predictions + adjustment.view(-1, 1) / (predictions.shape[1] - 1)
        predictions[torch.arange(len(predictions)), max_index] -= adjustment 
        return predictions
    