import numpy as np
import torch
import os, sys, shutil, copy, time, random, itertools
import copy, math
from .. import _implicit_quantiles, _get_prediction_device, _move_prediction_device


class DistributionBase:
    """ Abstract baseclass for a distribution that arises from conformal calibration. 
    
    This class behaves like torch.distribution.Distribution, and supports the cdf, icdf and rsample functions. 
    """ 
    def __init__(self):
        pass
        
    def to(self, device):
        assert False, "to not implemented" 
        
    def cdf(self, value):
        assert False, "cdf not implemented"
        
    def icdf(self, value):
        assert False, "icdf not implemented"
        
    def rsample(self, sample_shape=torch.Size([])):
        """
        Draw a set of samples from the distribution
        """
        rand_vals = torch.rand(list(sample_shape) + [self.batch_shape[0]])
        return self.icdf(rand_vals.view(-1, self.batch_shape[0])).view(rand_vals.shape)
    
    def sample(self, sample_shape=torch.Size([])):
        return self.rsample(sample_shape)
    
    def sample_n(self, n):
        return self.rsample(torch.Size([n]))
    
    def log_prob(self, value):
        """
        Compute the log probability. This default implementation is not great as it is numerically unstable and require tricks to not throw faults. 
        """
#         # Get the shape 
#         shape = e
#         eps = self.test_std * 1e-3   # Use the same unit as the std 
#         if len(values) == 0:
        eps = 1e-4
        return torch.log(self.cdf(value + eps) - self.cdf(value) + 1e-10) - math.log(eps)
    
    def shape_inference(self, value):
        """ 
        Handle all unusual shapes 
        """
        # Enumerate all the valid input shapes for value
        if type(value) == int or type(value) == float:  
            return value.view(1, 1).repeat(1, self.batch_shape[0]), self.batch_shape[0]
        elif len(value.shape) == 1 and value.shape[0] == 1:  # If the value is 1-D it must be either 1 or equal to batch_shape[0]
            return value.view(1, 1).repeat(1, self.batch_shape[0]), self.batch_shape[0]
        elif len(value.shape) == 1 and value.shape[0] == self.batch_shape[0]:   # If the value is 1-D it must be either 1 or equal to batch_shape[0]
            return value.view(1, -1), self.batch_shape[0]
        elif len(value.shape) == 2 and value.shape[1] == 1:
            return value.repeat(1, self.batch_shape[0]), [len(value), self.batch_shape[0]]
        elif len(value.shape) == 2 and value.shape[1] == self.batch_shape[0]:
            return value, [len(value), self.batch_shape[0]]
        else:
            assert False, "Shape [%s] invalid" % ', '.join([str(shape) for shape in value.shape])
    
    
class ConcatDistribution():
    """ Class that concat multiple distribution instances. 
    
    torch.distributions.Distribution does not yet have a concatenation function (as of 1.1), 
    making it difficult to concate two distribution instances similar to concating two torch tensors. 
    This class fills this gap by concating multiple distribution instances into a single class that behaves like torch.distributions.Distribution. 
    
    This class supports a subset of functions for torch.distributions.Distribution, including cdf, icdf, log_prob, sample, rsample, sample_n. 
    
    Args:
        distributions (list): a list of torch Distribution instances. 
        dim: dimension to concat the distributions, any dimension other than the concat dimension must have equal size
    """
    def __init__(self, distributions, dim=0):
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
        """ Returns the cumulative density (CDF) evaluated at value. 
        
        Args:
            value (tensor): the values to evaluate the CDF. 
            
        Returns: 
            tensor: the evaluated CDF.
        """
        split_value, split_dim = self._split_input(value)
        cdfs = [distribution.cdf(val) for val, distribution in zip(split_value, self.distributions)]  # Get the CDF value for each split
        return torch.cat(cdfs, dim=split_dim) 
    
    def icdf(self, value):
        """ Returns the inverse cumulative density (ICDF) evaluated at value. 
        
        Args:
            value (tensor): the values to evaluate the ICDF. 
            
        Returns:
            tensor: the evaluated ICDF. 
        """
        split_value, split_dim = self._split_input(value)
        icdfs = [distribution.icdf(val) for val, distribution in zip(split_value, self.distributions)]  # Get the CDF value for each split
        return torch.cat(icdfs, dim=split_dim) 
    
    def log_prob(self, value):
        """ Returns the log of the probability density evaluated at value.
        
        Args:
            value (tensor): the values to evaluate the ICDF. 
            
        Returns:
            tensor: the evaluated log_prob. 
        """ 
        split_value, split_dim = self._split_input(value)
        log_probs = [distribution.log_prob(val) for val, distribution in zip(split_value, self.distributions)]  # Get the CDF value for each split
        return torch.cat(log_probs, dim=split_dim) 
    
    def rsample(self, sample_shape=torch.Size([])):
        """ Generates a sample_shape shaped (batch of) sample. 
        
        Args:
            sample_shape (torch.Size): the shape of the samples.
            
        Returns:
            tensor: the drawn samples. 
        """
        split_dim = len(sample_shape) + self.dim 
        return torch.cat([distribution.rsample(sample_shape) for distribution in self.distributions], dim=split_dim)
    
    def sample(self, sample_shape=torch.Size([])):
        split_dim = len(sample_shape) + self.dim 
        return torch.cat([distribution.sample(sample_shape) for distribution in self.distributions], dim=split_dim)
    
    def sample_n(self, n):
        """ Generates n batches of samples. 
        
        Args:
            n (int): the number of batches of samples. 
            
        Returns:
            tensor: the drawn samples. 
        """
        return torch.cat([distribution.sample_n(n) for distribution in self.distributions], dim=self.dim+1)

    def to(self, device):
        """ Move this class and all the tensors it owns to a specified device. 
        
        Args:
            device (torch.device): the device to move this class to. 
        """
        self.distributions = [_move_prediction_device(pred, device) for pred in self.distributions]
        
    def _split_input(self, value):
        """ Split the input along the concatenated dimension
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
    """ The abstract base class for all calibrator classes. 
    
    Args:
        input_type (str): the input prediction type. 
            If input_type is 'auto' then it is automatically induced when Calibrator.train() or update() is called, it cannot be changed after the first call to train() or update(). 
            Not all sub-classes support 'auto' input_type, so it is strongly recommended to explicitly specify the prediction type. 
    """
    def __init__(self, input_type='auto'):
        self.input_type = input_type
        self.device = None
    
    def _change_device(self, predictions):
        """ Move everything into the same device as predictions, do nothing if they are already on the same device """
        # print("_change_device is deprecated ")
        device = _get_prediction_device(predictions)
        # device = self.get_device(predictions)
        self.to(device)
        self.device = device
        return device
    
    
    def to(self, device):
        """ Move this class and all the tensors it owns to a specified device. 
        
        Args:
            device (torch.device): the device to move this class to. 
        """
        assert False, "Calibrator.to has not been implemented"
    

    def train(self, predictions, labels, *args, **kwargs):
        """ The train abstract class. Learn the recalibration map based on labeled data. 
        
        This function uses the training data to learn any parameters that is necessary to transform a low quality (e.g. uncalibrated) prediction into a higher quality (e.g. calibrated) prediction. 
        It takes as input a set of predictions and the corresponding labels. 
        In addition, a few recalibration algorithms --- such as group calibration or multicalibration --- can take as input additional side features, and the transformation depends on the side feature. 
        
        Args:
            predictions (object): a batched prediction object, must match the input_type argument when calling __init__. 
            labels (tensor): the labels with shape [batch_size]
            side_feature (tensor): some calibrator instantiations can use additional side feature, when used it should be a tensor of shape [batch_size, n_features]
            
        Returns: 
            object: an optional log object that contains information about training history. 
        """
        assert False, "Calibrator.train has not been implemented"
    
    # 
    # If half_life is not None, then it is the number of calls to this function where the sample is discounted to 1/2 weight
    # Not all calibration functions support half_life
    def update(self, predictions, labels, *args, **kwargs):
        """ Same as Calibrator.train, but updates the calibrator online with the new data (while train erases any existing data in the calibrator and learns it from scratch)
        
        Args:
            predictions (object): a batched prediction object, must match the input_type argument when calling __init__. 
            labels (tensor): the labels with shape [batch_size]
            side_feature (tensor): some calibrator instantiations can use additional side feature, when used it should be a tensor of shape [batch_size, n_features]
            
        Returns:
            object: an optional log object that contains information about training history. 
        """
        assert False, "Calibrator.update has not been implemented"
    
    # Input an array of shape [batch_size, num_classes], output the recalibrated array
    # predictions should be in the same pytorch device 
    # If side_feature is not None when calling train, it shouldn't be None here either. 
    def __call__(self, predictions, *args, **kwargs):
        """ Use the learned calibrator to transform new data. 
        
        Args:
            predictions (prediction object): a batched prediction object, must match the input_type argument when calling __init__. 
            labels (tensor): the labels with shape [batch_size]
            side_feature (tensor): some calibrator instantiations can use additional side feature, when used it should be a tensor of shape [batch_size, n_features]
            
        Returns:
            prediction object: the transformed predictions
        """
        assert False, "Calibrator.__call__ has not been implemented"
    
    def check_type(self, predictions):
        """ Checks that the prediction has the correct shape specified by input_type. 
        
        Args:
            predictions (prediction object): a batched prediction object, must match the input_type argument when calling __init__. 
        """
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
            



    