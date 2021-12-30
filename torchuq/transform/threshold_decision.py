import numpy as np
import itertools
from matplotlib import pyplot as plt
import torch
from .basic import Calibrator   # Note can use from .basic import Calibrator
from .. import _get_prediction_device

class ThresholdCalibrator(Calibrator):
    """
    Recalibrate probabilistic predictions to achieve threshold calibration.
    Args: 
            verbose (bool): if set to True than print additional performance information during training
            save_path (float): the path to checkpoint any training weights. If set to None the weights are not saved. 
    """

    def __init__(self, verbose=True, save_path=False):
        self.verbose = verbose
        self.save_path = save_path


    def __call__(self, predictions):
        """ Use the learned recalibration map to transform predictions into decision-calibrated new predictions. 
        
        Args:
            predictions (tensor): a batch of categorical predictions.
        Returns:
            tensor: the transformed predictions. 
        """

        pass

    def to(self, device):
        """Move every torch tensor owned by this class to a new device
        
        Args:
            device: a torch.device instance, alternatively it could be a torch.Tensor or a prediction object
        """

    def train(self, predictions, labels, calib_steps, test_predictions, test_labels, seed):
        """ Train the decision calibrator for calib_steps. 
        If you call this function multiple times, this function does not erase previously trained calibration maps, and only appends additional recalibration steps
        
        Args:
            predictions (tensor): a categorical prediction with shape [batch_size, n_classes]
            labels (tensor): an array of int valued labels
            calib_steps (int): number of calibration iterations (this is the number of iteration steps in Algorithm 2 of the paper)
            num_critic_epoch (int): number of gradient descent steps when optimizing the worst case b in Algorithm 2 of the paper
            test_predictions (tensor): a categorical prediction for measuring test performance, can be set to None if measuring test performance is not needed
            test_labels (tensor): an array of int valued labels for measuring test performance, can be set to None if measuring test performance is not needed
            seed: float, the random seed for reproducibility. 
            
        Returns:
            recorder: a PerformanceRecord object, the measured performance 
        """

        pass
