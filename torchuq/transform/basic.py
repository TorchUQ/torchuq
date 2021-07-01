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
import copy
from .. import _implicit_quantiles, _get_prediction_device, _move_prediction_device



class ConcatDistribution():
    """
    Class that concat multiple classes that behave like torch.distributions.Distribution.
    This class supports the cdf, icdf, log_prob, sample, rsample and sample_n interface, but nothing else, the interface is compatible with torch Distribution
    This class also supports the .to(device) interface, .device attribute and batch_shape attribute
    """
    def __init__(self, distributions, dim=0):
        """
        Inputs: 
            distributions: a list of instances that inherit the torch Distribution interface. Each instance must have a 1 dimensional batch_shape 
            dim: dimension to concat the distributions, any dimension other than the concat dimension must have equal size
        """
        assert len(distributions) != 0, "Need to concat at least one distribution"
        assert dim >= 0 and dim < len(distributions[0].batch_shape), "Concat dimension invalid"
        
        self.distributions = distributions
        self.dim = dim
        
        # The index boundary between different batches
        self.sizes = torch.Tensor([distribution.batch_shape[dim] for distribution in distributions]).type(torch.int)
        self.indices = torch.cumsum(self.sizes, dim=0)  
        
        # Compute the batch_shape of self
        self.batch_shape = list(distributions[0].batch_shape[:dim]) + [self.indices[-1].item()] + list(distributions[0].batch_shape[dim+1:])
        self.device = _get_prediction_device(distributions[0])
        
    def cdf(self, value):
        split_value, split_dim = self._split_input(value)
        cdfs = [distribution.cdf(val) for val, distribution in zip(split_value, self.distributions)]  # Get the CDF value for each split
        return torch.cat(cdfs, dim=split_dim) 
    
    def icdf(self, value):
        split_value, split_dim = self._split_input(value)
        icdfs = [distribution.icdf(val) for val, distribution in zip(split_value, self.distributions)]  # Get the CDF value for each split
        return torch.cat(icdfs, dim=split_dim) 
    
    def log_prob(self, value):
        split_value, split_dim = self._split_input(value)
        log_probs = [distribution.log_prob(val) for val, distribution in zip(split_value, self.distributions)]  # Get the CDF value for each split
        return torch.cat(log_probs, dim=split_dim) 
    
    def rsample(self, sample_shape=torch.Size([])):
        split_dim = len(sample_shape) + self.dim 
        return torch.cat([distribution.rsample(sample_shape) for distribution in self.distributions], dim=split_dim)
    
    def sample(self, sample_shape=torch.Size([])):
        split_dim = len(sample_shape) + self.dim 
        return torch.cat([distribution.sample(sample_shape) for distribution in self.distributions], dim=split_dim)
    
    def sample_n(self, n):
        return torch.cat([distribution.sample_n(n) for distribution in self.distributions], dim=self.dim+1)

    def to(self, device):
        self.distributions = [_move_prediction_device(pred, device) for pred in self.distributions]
        
    def _split_input(self, value):
        """
        Split the input along the concatenated dimension
        """
        split_dim = len(value.shape) - len(self.batch_shape) + self.dim  # Which dim to split the data 
        assert value.shape[split_dim] == self.batch_shape[self.dim] or value.shape[split_dim] == 1, \
            "The batch_shape is %d, but the input value has batch_shape %d along the concatenated dimension" % (value.shape[split_dim], self.batch_shape[self.dim])
        
        if value.shape[split_dim] == 1:
            split_value = [value.clone() for i in range(len(self.sizes))]
        else:
            split_value = torch.split(value, split_size_or_sections=list(self.sizes), dim=split_dim)  # Split the input value 
        return split_value, split_dim
    
    
class Calibrator:
    def __init__(self, input_type='auto'):
        """
        input_type should be one of the supported datatypes
        If input_type is 'auto' then it is automatically induced when Calibrator.train() or update() is called, it cannot be changed after the first call to train() or update()
        Input_type must be explicitly specificied for many subclasses
        """
        self.input_type = input_type
        self.device = None
    
#     def _change_device(self, predictions):
#         """ Move everything into the same device as predictions, do nothing if they are already on the same device """
#         device = _get_prediction_device(predictions)
#         # device = self.get_device(predictions)
#         self.to(device)
#         self.device = device
#         return device
    
    def to(self, device):
        assert False, "Calibrator.to has not been implemented"
    
    # Input an array of shape predictions=[dataset_size, num_classes], labels=[dataset_size]
    # Optionally input side features such as an array of shape [dataset_size, num_features]. Not all calibrators consider side feature when recalibrating 
    def train(self, predictions, labels, *args, **kwargs):
        pass
    
    # Same as train, but updates the calibrator online 
    # If half_life is not None, then it is the number of calls to this function where the sample is discounted to 1/2 weight
    # Not all calibration functions support half_life
    def update(self, predictions, labels, half_life=None):
        pass
    
    # Input an array of shape [batch_size, num_classes], output the recalibrated array
    # predictions should be in the same pytorch device 
    # If side_feature is not None when calling train, it shouldn't be None here either. 
    def __call__(self, predictions, *args, **kwargs):
        pass
    
    def check_type(self, predictions):
        if self.input_type == 'point':
            assert len(predictions.shape) == 1, "Point prediction should have shape [batch_size]"
        elif self.input_type == 'interval':
            assert len(predictions.shape) == 2 and predictions.shape[1] == 2, "interval predictions should have shape [batch_size, 2]"
        elif self.input_type == 'quantile':
            assert len(predictions.shape) == 2 or (len(predictions.shape) == 3 and predictions.shape[2] == 2), "quantile predictions should have shape [batch_size, num_quantile] or [batch_size, num_quantile, 2]" 
        elif self.input_type == 'distribution':
            assert hasattr(predictions, 'cdf') and hasattr(predictions, 'icdf'), "Distribution predictions should have a cdf and icdf method"
            
    def assert_type(self, input_type, valid_types):
        msg = "Input data type not supported, input data type is %s, supported types are %s" % (input_type, " ".join(valid_types)) 
        assert input_type in valid_types, msg 
            



    