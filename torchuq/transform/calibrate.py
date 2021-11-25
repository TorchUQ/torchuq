import numpy as np
import itertools
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os, sys, shutil, copy, time, random
from torchuq.models.flow import NafFlow
from .basic import Calibrator
from .utils import PerformanceRecord


class TemperatureScaling(Calibrator):
    """ The class to recalibrate a categorical prediction with temperature scaling 
    
    Args:
        verbose (bool): if verbose=True print non-error messsages 
    """
    def __init__(self, verbose=False):  # Algorithm can be hb (histogram binning) or kernel 
        super(TemperatureScaling, self).__init__(input_type='categorical')
        self.verbose = verbose
        self.temperature = None
        
    def train(self, predictions, labels, *args, **kwargs):
        """ Find the optimal temperature with gradient descent. 
        
        Args:
            predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]
            labels (tensor): a batch of labels with shape [batch_size]
        """
        # Use gradient descent to find the optimal temperature
        # Can add bisection option in the future, since it should be considerably faster
        self._change_device(predictions) 
        
        self.temperature = torch.ones(1, 1, requires_grad=True, device=self.device)
        optim = torch.optim.Adam([self.temperature], lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=3, threshold=1e-6, factor=0.5)
        
        log_prediction = torch.log(predictions + 1e-10).detach()
        
        # Iterate at most 100k iterations, but expect to stop early
        for iteration in range(100000):
            optim.zero_grad()
            adjusted_predictions = log_prediction / self.temperature
            loss = F.cross_entropy(adjusted_predictions, labels)
            loss.backward()
            optim.step()
            lr_scheduler.step(loss)
            
            # Hitchhike the lr scheduler to terminate if no progress
            if optim.param_groups[0]['lr'] < 1e-6:   
                break
            if self.verbose and iteration % 100 == 0:
                print("Iteration %d, lr=%.5f, NLL=%.3f" % (iteration, optim.param_groups[0]['lr'], loss.cpu().item()))
    
    def __call__(self, predictions, *args, **kwargs):
        """ Use the learned temperature to calibrate the predictions. 
        
        Only use this after calling TemperatureScaling.train. 
        
        Args:
            predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]
        
        Returns:
            tensor: the calibrated categorical prediction, it should have the same shape as the input predictions
        """
        if self.temperature is None:
            print("Error: need to first train before calling this function")
        self._change_device(predictions)
        log_prediction = torch.log(predictions + 1e-10)
        return torch.softmax(log_prediction / self.temperature, dim=1)
    
    def to(self, device):
        """ Move all assets of this class to a torch device. 
        
        Args:
            device (device): the torch device (such as torch.device('cpu'))
        """
        if self.temperature is not None:
            self.temperature.to(device)
        return self

    
    
class DirichletCalibrator(Calibrator):
    """ The class to recalibrate a categorical prediction with dirichlet calibration 
    
    Args:
        verbose (bool): if verbose=True print non-error messsages 
    """
    
    def __init__(self, verbose=False):  
        super(DirichletCalibrator, self).__init__(input_type='categorial')
        self.calibrator = None
        self.verbose = verbose
 
    def train(self, predictions, labels, param_lambda=1e-3, param_mu=0.1, max_epochs=1000, *args, **kwargs):   
        """ Train the Dirichlet recalibration map 
        
        Args:
            predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]
            labels (tensor): a batch of labels with shape [batch_size]
            param_lambda (float): the regularization hyper-parameter. According to E.3 of https://arxiv.org/pdf/1910.12656.pdf, the best hyper-parameter is usually 1e-3
            param_mu (float): the regularization hyper-parameter. 
            max_epochs (int): the maximum number of epochs to train the model. This function might terminate when training makes no additional progress before max_epochs is reached
        
        Returns:
            PerformanceRecord: a PerformanceRecord instance with detailed training log 
        """
        self._change_device(predictions)
        
        predictions = (predictions + 1e-10).log().detach()   # The input to the calibration map has to be log probabilities, we also don't want to propagate gradients w.r.t. original predictions 
        n_classes = predictions.shape[1]
        self.calibrator = nn.Linear(n_classes, n_classes).to(self.device)
        
        calib_optim = optim.Adam(self.calibrator.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(calib_optim, mode='min', patience=2, threshold=1e-6, factor=0.5)
        
        # Load the data sequentially because 1. SGD usually converges faster than GD 2. less memory use
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(predictions.cpu(), labels.cpu()), # Seems like pytorch dataset has to live in cpu memory 
                                                 batch_size=256, shuffle=True, num_workers=2)
        
        recorder = PerformanceRecord()
        for epoch in range(max_epochs):
            # Epoch average nll and reg loss
            nll_loss_epoch = 0.0
            reg_loss_epoch = 0.0

            for bp, bl in val_loader:
                calib_optim.zero_grad()
                bp, bl = bp.to(self.device), bl.to(self.device)
                bp_adjusted = self.calibrator(bp)
                
                # The negative loss likelihood loss
                nll_loss = F.cross_entropy(bp_adjusted, bl)
                
                # Regularization loss (the ODIR regularization discussed in the original paper)
                reg_loss = param_lambda * (self.calibrator.weight.pow(2).sum() - self.calibrator.weight.pow(2).diagonal().sum()) / n_classes / (n_classes - 1)
                reg_loss += param_mu * self.calibrator.bias.pow(2).mean() 
                
                total_loss = nll_loss + reg_loss 
                total_loss.backward()
                
                nll_loss_epoch += nll_loss.detach() * len(bp)
                reg_loss_epoch += reg_loss.detach() * len(bp)
                
                calib_optim.step()
            
            nll_loss_epoch /= len(predictions)
            reg_loss_epoch /= len(predictions)
            
            # Write the loss to the recorder 
            recorder.add_scalar('nll_loss', nll_loss_epoch.item(), epoch)
            recorder.add_scalar('reg_loss', reg_loss_epoch.item(), epoch)
                
            # Decay learning rate if the total loss did not improve
            lr_scheduler.step(nll_loss_epoch + reg_loss_epoch) 
            if calib_optim.param_groups[0]['lr'] < 1e-5:   # Terminate if the learning rate has dropped to 1e-5
                break  
                
            if self.verbose and epoch % 10 == 0:
                print("Finished training %d epochs, lr=%.5f, nll = %.3f, reg = %.3f" % 
                      (epoch, calib_optim.param_groups[0]['lr'], nll_loss_epoch.cpu().item(), 1000. * reg_loss_epoch.cpu().item()))

        return recorder 
                
    def __call__(self, predictions):
        """ Use the learned calibration map to calibrate the predictions. 
        
        Only use this after calling DirichletCalibrator.train. 
        
        Args:
            predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]
        
        Returns:
            tensor: the calibrated categorical prediction, it should have the same shape as the input predictions
        """
        if self.calibrator is None:
            print("Error: need to first train before calling this function")
        self._change_device(predictions)
        predictions = (predictions + 1e-10).log()   # The input to the calibration map has to be log probabilities 
        return torch.softmax(self.calibrator(predictions), dim=-1)

    def to(self, device):
        """ Move all assets of this class to a torch device. 
        
        Args:
            device (device): the torch device (such as torch.device('cpu'))
        """
        if self.calibrator is not None:
            self.calibrator.to(device)
            
            

class HistogramBinning:
    """ The class to recalibrate a categorical prediction with temperature scaling 
    
    Args:
        adaptive (bool): if adaptive is true, use the same number of samples per bin,
            if adaptive if false, use equal width bins
        bin_count (int or str): the number of bins, if 'auto' set the number of bins automatically as sqrt(number of samples) / 5 
        verbose (bool): if verbose=True print non-error messsages 
    """
    def __init__(self, adaptive=True, bin_count='auto', verbose=False):  # Algorithm can be hb (histogram binning) or kernel 
        self.adaptive = adaptive
        self.bin_count = bin_count
        self.verbose = verbose
        self.bin_boundary = None
        self.bin_adjustment = None

    def train(self, predictions, labels, *args, **kwargs):
        """ Train the recalibration map 
        
        Args:
            predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]
            labels (tensor): a batch of labels with shape [batch_size]
        """
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
        """ Use the learned calibration map to calibrate the predictions. 
        
        Only use this after calling HistogramBinning.train. 
        
        Args:
            predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]
        
        Returns:
            tensor: the calibrated categorical prediction, it should have the same shape as the input predictions
        """
        if self.bin_boundary is None:
            print("Error: need to first call ConfidenceHBCalibrator.train before calling this function")
        with torch.no_grad():
            max_confidence, max_index = predictions.max(dim=1)
            max_confidence = max_confidence.view(-1, 1).repeat(1, len(self.bin_boundary))
            tiled_boundary = self.bin_boundary.to(predictions.device).view(1, -1).repeat(len(predictions), 1)
            tiled_adjustment = self.bin_adjustment.to(predictions.device).view(1, -1).repeat(len(predictions), 1)
            
            index = (max_confidence < tiled_boundary).type(torch.float32)
            index = index[:, 1:] - index[:, :-1]

            adjustment = (tiled_adjustment * index).sum(dim=1)

        predictions = predictions + adjustment.view(-1, 1) / (predictions.shape[1] - 1)
        predictions[torch.arange(len(predictions)), max_index] -= adjustment 
        return predictions
    
    
    def to(self, device):
        """ Move all assets of this class to a torch device. 
        
        Args:
            device (device): the torch device (such as torch.device('cpu'))
        """
        if self.bin_boundary is not None:
            self.bin_boundary.to(device)
        if self.bin_adjustment is not None:
            self.bin_adjustment.to(device)