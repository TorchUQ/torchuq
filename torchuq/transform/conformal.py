import pandas as pd
import numpy as np
import itertools, math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import os, sys, shutil, copy, time, random
from .basic import Calibrator, ConcatDistribution, DistributionBase
from .utils import BisectionInverse
from ..models.flow import NafFlow
from .. import _implicit_quantiles, _get_prediction_device, _move_prediction_device, _parse_name, _get_prediction_batch_shape
from ..evaluate.distribution import compute_std, compute_mean_std

        
            
class DistributionConformal:
    """
    Abstract baseclass for a distribution that arises from conformal calibration. 
    This class behaves like torch.distribution.Distribution, and supports the cdf, icdf and rsample functions. 
    """
    def __init__(self, val_predictions, val_labels, test_predictions, score_func, iscore_func):
        """
        Inputs:
            val_predictions: a set of validation predictions, the type must be compatible with score_func 
            val_labels: array [validation_batch_shape], a batch of labels
            test_predictions: a set of test predictions, the type must be compatible with score func
            score_func: non-conformity score function. A function that take as input a batched predictions q, and an array v of values with shape [n_evaluations, batch_shape] 
            returns an array s of shape [n_evaluations, batch_shape] where s_{ij} is the score of v_{ij} under prediction q_j 
            iscore_func: inverse non-conformity score function: a function that take as input a batched prediction q, and an array s os scores with shape [n_evaluations, batch_shape] 
            returns an array v of shape [n_evaluations, batch_shape] which is the inverse of score_func (i.e. iscore_func(q, score_func(q, v)) = v)
        """
        self.score = score_func 
        self.iscore = iscore_func 
        self.test_predictions = test_predictions 
        self.device = _get_prediction_device(test_predictions)
        
        with torch.no_grad():
            self.batch_shape = self.score(test_predictions, torch.zeros(1, 1, device=self.device)).shape[1:2]  # A hack to find out the number of distributions
            
            # Compute the scores for all the predictions in the validation set and sort them from small to large
            val_scores = self.score(val_predictions, val_labels.view(1, -1)).flatten()
            # Need to fix this: To avoid numerical instability neighboring values should not be too similar, maybe add a tiny biy of noise 
            val_scores = val_scores.sort()[0] 
            # Prepend the 0 quantile and append the 1 quantile for convenient handling of boundary conditions
            self.val_scores = torch.cat([val_scores[:1] - (val_scores[1:] - val_scores[:-1]).mean(dim=0, keepdims=True), 
                                         val_scores, 
                                         val_scores[-1:] + (val_scores[1:] - val_scores[:-1]).mean(dim=0, keepdims=True)])   
        # self.test_std = compute_std(self) + 1e-10  # This is the last thing that can be called 
             
            if iscore_func == _conformal_iscore_ensemble:
                min_label = val_labels.min()
                max_label = val_labels.max()
                min_search = min_label - (max_label - min_label)
                max_search = max_label + (max_label - min_label)
                self.iscore = partial(iscore_func, min_search=min_search, max_search=max_search)
            
    def to(self, device):
        if self.device != device:
            self.device = device
            self.val_scores = self.val_scores.to(device)
            self.test_predictions = _move_prediction_device(self.test_predictions, device)
    
    
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
            
class DistributionConformalLinear(DistributionConformal):
    def __init__(self, val_predictions, val_labels, test_predictions, score_func, iscore_func, verbose=False):
        super(DistributionConformalLinear, self).__init__(val_predictions, val_labels, test_predictions, score_func, iscore_func)
        
    def cdf(self, value):
        """
        The CDF at value
        Input:
        - value: an array of shape [n_evaluations, batch_shape] or shape [batch_size] 
        """
        # First perform automatic shape induction and convert value into an array of shape [n_evaluations, batch_shape]
        value, out_shape = self.shape_inference(value)
        # self.to(value.device)   # Move all assets in this class to the same device as value to avoid device mismatch error
        
        # Move value to the device of test_predictions to avoid device mismatch error
        out_device = value.device
        value = value.to(self.device)
        
        # Non-conformity score
        scores = self.score(self.test_predictions, value)
        
        # Compare the non-conformity score to the validation set non-conformity scores
        quantiles = self.val_scores.view(1, 1, -1)
        comparison = (scores.unsqueeze(-1) - quantiles[:, :, :-1]) / (quantiles[:, :, 1:] - quantiles[:, :, :-1] + 1e-20) 
        cdf = comparison.clamp(min=0, max=1).sum(dim=-1) / (len(self.val_scores) - 1)
        return cdf.view(out_shape).to(out_device) 
    
    def icdf(self, value):
        """
        Get the inverse CDF
        Input:
        - value: an array of shape [n_evaluations, batch_shape] or shape [batch_shape], each entry should take values in [0, 1]
        Supports automatic shape induction, e.g. if cdf has shape [n_evaluations, 1] it will automatically be converted to shape [n_evaluations, batch_shape]
        
        Output:
        The value of inverse CDF function at the queried cdf values
        """
        cdf, out_shape = self.shape_inference(value)   # Convert cdf to have shape [n_evaluations, batch_shape]
        # self.to(cdf.device)   # Move all assets in this class to the same device as value to avoid device mismatch error
        
        # Move cdf to the device of test_predictions to avoid device mismatch error
        out_device = cdf.device
        cdf = cdf.to(self.device)
        
#         out_device = cdf.device
#         cdf = cdf.to(self.test_predictions)
        
        quantiles = cdf * (len(self.val_scores) - 1)
        ratio = torch.ceil(quantiles) - quantiles
        target_score = self.val_scores[torch.floor(quantiles).type(torch.long)] * ratio + \
            self.val_scores[torch.ceil(quantiles).type(torch.long)] * (1 - ratio) 
        value = self.iscore(self.test_predictions, target_score)
        return value.view(out_shape).to(out_device)  # Output the original device
        
        
class DistributionConformalRandom(DistributionConformal):
    """
    Use randomization to interpolate the non-conformity score. 
    This distribution does not have a differentiable CDF (i.e. it does not have a density), so the behavior of functions such as log_prob and plot_density are undefined. 
    """
    def __init__(self, val_predictions, val_labels, test_predictions, score_func, iscore_func, verbose=False):
        super(DistributionConformalRandom, self).__init__(val_predictions, val_labels, test_predictions, score_func, iscore_func)
        self.rand_cdf = torch.rand(1, _get_prediction_batch_shape(test_predictions), device=self.device)
        
        # Random interpolation is special, the ICDF could be infinite 
        self.val_scores[0] = -float('inf')
        self.val_scores[-1] = float('inf')
        
    def cdf(self, value):
        """
        The CDF at value. This function is NOT differentiable
        Input:
        - value: an array of shape [n_evaluations, batch_shape] or shape [batch_size] 
        """
        # First perform automatic shape induction and convert value into an array of shape [n_evaluations, batch_shape]
        value, out_shape = self.shape_inference(value)

        # Move value to the device of test_predictions to avoid device mismatch error
        out_device = value.device
        value = value.to(self.device)
        
        # Non-conformity score
        scores = self.score(self.test_predictions, value)
        
        # Compare the non-conformity score to the validation set non-conformity scores
        quantiles = self.val_scores.view(1, 1, -1)
        cdf = (scores.unsqueeze(-1) > quantiles).type(value.dtype).sum(dim=-1)   # Compute the ranking of the value among all validation values. This value should be between [0, len(val_score)]
        # If cdf is 0, then set it to 0
        # If cdf is 1, then set it to U[0, 1]
        # ...
        # If cdf is N, then set it to U[N-1, N]
        # If cdf is N+2, set it to N+1
        # Note len(val_scores) = N+2
        cdf = ((cdf - 1) + self.rand_cdf).clamp(min=0, max=len(self.val_scores)-1)
        cdf = cdf / (len(self.val_scores) - 1)
        return cdf.view(out_shape).to(out_device) 
    
    
    def icdf(self, value):
        """
        Get the inverse CDF. This function is NOT differentiable
        Input:
        - value: an array of shape [n_evaluations, batch_shape] or shape [batch_shape], each entry should take values in [0, 1]
        Supports automatic shape induction, e.g. if cdf has shape [n_evaluations, 1] it will automatically be converted to shape [n_evaluations, batch_shape]
        
        Output:
        The value of inverse CDF function at the queried cdf values
        """
        cdf, out_shape = self.shape_inference(value)   # Convert cdf to have shape [n_evaluations, batch_shape]
        # self.to(cdf.device)   # Move all assets in this class to the same device as value to avoid device mismatch error
        
        # Move cdf to the device of test_predictions to avoid device mismatch error
        out_device = cdf.device
        cdf = cdf.to(self.device)
        
        quantiles = cdf * (len(self.val_scores) - 1)
        # The following is carefully crafted to exactly invert the cdf function. This code must be exactly as it is
        quantiles = torch.floor(quantiles + 1 - self.rand_cdf).type(torch.long).clamp(min=0, max=len(self.val_scores)-1)
        target_score = self.val_scores[quantiles]
        value = self.iscore(self.test_predictions, target_score)
        return value.view(out_shape).to(out_device)  # Output the original device
    
    
class DistributionConformalNAF(DistributionConformal):
    def __init__(self, val_predictions, val_labels, test_predictions, score_func, iscore_func, verbose=True):
        super(DistributionConformalNAF, self).__init__(val_predictions, val_labels, test_predictions, score_func, iscore_func)
        
        # Train both a flow and an inverse flow to avoid the numerical instability of inverting a flow
        self.flow = NafFlow(feature_size=200).to(val_labels.device)
        self.iflow = NafFlow(feature_size=200).to(val_labels.device)
        target_cdf = torch.linspace(0, 1, len(self.val_scores), device=val_labels.device)  # The goal of the flow is to map non-conformity scores to CDF values uniformly in [0, 1]
         
        flow_optim = optim.Adam(list(self.flow.parameters()) + list(self.iflow.parameters()), lr=1e-3)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(flow_optim, mode='min', patience=2, threshold=1e-2, threshold_mode='rel', factor=0.5)

        for iteration in range(50000):
            flow_optim.zero_grad()

            cdfs, _ = self.flow(self.val_scores.view(-1, 1).type(torch.float32))
            scores, _ = self.iflow(target_cdf.view(-1, 1).type(torch.float32))
            
            loss = (cdfs.flatten() - target_cdf).pow(2).mean() + (scores.flatten() - self.val_scores).pow(2).mean()
            loss.backward()
            flow_optim.step()
            
            if iteration % 100 == 0:
                lr_scheduler.step(loss)  # Reduce the learning rate 
                if flow_optim.param_groups[0]['lr'] < 1e-5 or loss < 1e-5:   # Hitchhike the lr scheduler to terminate if no progress, or the loss is extremely small
                    break
                
            if verbose and iteration % 1000 == 0:
                print("Iteration %d, loss=%.5f, lr=%.5f" % (iteration, loss, flow_optim.param_groups[0]['lr']))
    
    def cdf(self, value):
        """
        The CDF at value
        Input:
        - value: an array of shape [n_evaluations, batch_shape] or shape [batch_size] 
        """
        # First perform automatic shape induction and convert value into an array of shape [n_evaluations, batch_shape]
        value, out_shape = self.shape_inference(value)
        # self.to(value.device)   # Move all assets in this class to the same device as value to avoid device mismatch error
        
        # Move value to the device of test_predictions to avoid device mismatch error
        out_device = value.device
        value = value.to(self.device)
        
        score = self.score(self.test_predictions, value)
        cdf, _ = self.flow(score.view(-1, 1))
        return cdf.clamp(min=1e-6, max=1-1e-6).view(out_shape).to(out_device)
    
    def icdf(self, value):
        """
        Get the inverse CDF
        Input:
        - value: an array of shape [n_evaluations, batch_shape] or shape [batch_shape], each entry should take values in [0, 1]
        Supports automatic shape induction, e.g. if cdf has shape [n_evaluations, 1] it will automatically be converted to shape [n_evaluations, batch_shape]
        
        Output:
        The value of inverse CDF function at the queried cdf values
        """
        cdf, out_shape = self.shape_inference(value)   # Convert cdf to have shape [n_evaluations, batch_shape]
        # self.to(cdf.device)   # Move all assets in this class to the same device as value to avoid device mismatch error
        
        # Move cdf to the device of test_predictions to avoid device mismatch error
        out_device = cdf.device
        cdf = cdf.to(self.device)
        
        adjusted = self.iflow(cdf.view(-1, 1))[0].view(cdf.shape)
        value = self.iscore(self.test_predictions, adjusted)
        return value.view(out_shape).to(out_device)
    

def _conformal_score_point(predictions, values):
    """
    Compute the non-conformity score of a set of values under some baseline predictor 
    Input:
        predictions: array [batch_shape], a batch of point predictions
        values: array [n_evaluations, batch_shape], note that for values batch_shape is the last dimension while for predictions batch_shape is the first dimension
    Output:
        score: array [n_evaluations, batch_shape], where score[i, j] is the non-conformity score of values[i, j] under the prediction[j]
    """
    score = values - predictions.view(1, -1)
    return score 

    
def _conformal_iscore_point(predictions, score):
    """
    Compute the inverse of the non-conformity score defined in conformal_score_quantile. 
    The goal is that conformal_iscore_quantile(predictions, conformal_score_quantile(predictions, labels))) = labels
    
    Input:
        predictions: array [batch_size, n_quantiles] or [batch_size, n_quantiles, 2], a batch of quantile predictions
        score: array [n_evaluations, batch_shape]
        
    Output:
        value: array [n_evaluations, batch_shape], where value[i, j] is the inverse non-conformity score of score[i, j] under prediction[j]
    """
    return predictions.view(1, -1) + score

def _conformal_score_interval(predictions, values):
    """
    Compute the non-conformity score of a set of values under some baseline predictor 
    Input:
        predictions: array [batch_shape, 2], a batch of interval predictions
        values: array [n_evaluations, batch_shape], note that for values batch_shape is the last dimension while for predictions batch_shape is the first dimension
        
    Output:
        score: array [n_evaluations, batch_shape], where score[i, j] is the non-conformity score of values[i, j] under the prediction[j]
    """
    score = (values - predictions.min(dim=1, keepdims=True)[0].permute(1, 0)) / (predictions[:, 1:2] - predictions[:, 0:1]).abs().permute(1, 0) - 0.5
    return score 
    
def _conformal_iscore_interval(predictions, score):
    """
    Compute the inverse of the non-conformity score defined in conformal_score_quantile. 
    The goal is that conformal_iscore_quantile(predictions, conformal_score_quantile(predictions, labels))) = labels
    
    Input:
        predictions: array [batch_size, 2], a batch of interval predictions
        score: array [n_evaluations, batch_shape]
        
    Output:
        value: array [n_evaluations, batch_shape], where value[i, j] is the inverse non-conformity score of score[i, j] under prediction[j]
    """
    return predictions.min(dim=1, keepdims=True)[0].permute(1, 0) + (score + 0.5) * (predictions[:, 1:2] - predictions[:, 0:1]).abs().permute(1, 0)
    

def _conformal_score_interval1(predictions, values, max_interval=1e+3):
    """
    Compute the alternative non-conformity score of a set of values under interval predictions
    """
    diff = values - predictions.mean(dim=1).view(1, -1) 
    return torch.sign(diff) * max_interval + diff


def _conformal_iscore_interval1(predictions, values, max_interval=1e+3):
    """
    Compute the alternative inverse non-conformity score of a set of values under interval predictions
    """
    return values - torch.sign(values) * max_interval + predictions.mean(dim=1).view(1, -1)



def _conformal_score_quantile(predictions, values):
    """
    Compute the non-conformity score of a set of values under some baseline predictor 
    Input:
        predictions: array [batch_shape, n_quantiles] or [batch_shape, n_quantiles, 2], a batch of quantile predictions
        values: array [n_evaluations, batch_shape], note that for values batch_shape is the last dimension while for predictions batch_shape is the first dimension
        
    Output:
        score: array [n_evaluations, batch_shape], where score[i, j] is the non-conformity score of values[i, j] under the prediction[j]
    """
    if len(predictions.shape) == 2:
        # sorted_quantile = torch.linspace(0, 1, predictions.shape[1]+2, device=predictions.device)[1:-1].view(-1, 1)
        sorted_quantile = _implicit_quantiles(predictions.shape[1]).to(predictions.device).view(1, -1)  
        sorted_pred, _ = torch.sort(predictions, dim=1)
    else:
        sorted_quantile, _ = torch.sort(predictions[:, :, 1], dim=1)   # [batch_shape, num_quantiles]
        sorted_pred, _ = torch.sort(predictions[:, :, 0], dim=1)
        
    sorted_quantile = sorted_quantile.permute(1, 0) # [num_quantiles, batch_shape] This is needed because torch Distribution has different convention from torchuq
    sorted_pred = sorted_pred.permute(1, 0).unsqueeze(1)  # [num_quantiles, 1, batch_shape]
    quantile_gap = (sorted_quantile[1:] - sorted_quantile[:-1]).unsqueeze(1) # [num_quantiles-1, 1, batch_shape]
    
    # The score is equal to how many quantiles the value exceeds
    score = (values.unsqueeze(0) - sorted_pred[:-1]) / (sorted_pred[1:] - sorted_pred[:-1])   # [num_quantiles-1, n_evaluations, batch_shape]
    score = sorted_quantile[:1] + (score.clamp(min=0.0, max=1.0) * quantile_gap).sum(dim=0)   # If value exceeds all samples, its score so far is 1, [n_evaluations, batch_shape]

    # Also consider values that are below the smallest sample or greater than the largest sample
    # A value has score <0 iff it is less than the smallest sample, and score >num_quantile iff it is greater than the largest sample
    score = score + (values - sorted_pred[0]).clamp(max=0.0) + (values - sorted_pred[-1]).clamp(min=0.0) 
    score = score - 0.5 # Center the score around 0, a label would take that score if it's exactly the 50\% quantile (based on linear interpolation)
    return score 


def _conformal_iscore_quantile(predictions, score):
    """
    Compute the inverse of the non-conformity score defined in conformal_score_quantile. 
    The goal is that conformal_iscore_quantile(predictions, conformal_score_quantile(predictions, labels))) = labels
    
    Input:
        predictions: array [batch_size, n_quantiles] or [batch_size, n_quantiles, 2], a batch of quantile predictions
        score: array [n_evaluations, batch_shape]
        
    Output:
        value: array [n_evaluations, batch_shape], where value[i, j] is the inverse non-conformity score of score[i, j] under prediction[j]
    """
    if len(predictions.shape) == 2:
        # sorted_quantile = torch.linspace(0, 1, predictions.shape[1]+2, device=predictions.device)[1:-1].view(-1, 1)
        sorted_quantile = _implicit_quantiles(predictions.shape[1]).to(predictions.device).view(1, -1)    # [1, n_quantiles]
        sorted_pred, _ = torch.sort(predictions, dim=1)  # [batch_shape, n_quantiles]
    else:
        sorted_quantile, _ = torch.sort(predictions[:, :, 1], dim=1)   
        sorted_pred, _ = torch.sort(predictions[:, :, 0], dim=1)
    
    score = score + 0.5    # Recover the centering in the _conformal_score_quantile function
    
    sorted_pred = sorted_pred.permute(1, 0)  # [n_quantiles, batch_shape]
    sorted_quantile = sorted_quantile.permute(1, 0).unsqueeze(1) # [n_quantiles, 1, batch_shape]
    pred_gap = (sorted_pred[1:] - sorted_pred[:-1]).unsqueeze(1)  # [num_quantiles-1, 1, batch_shape]
    
    # For each interval between two adjacent samples, compute whether the score is large enough such that this interval should be added
    value = (score.unsqueeze(0) - sorted_quantile[:-1]) / (sorted_quantile[1:] - sorted_quantile[:-1])  # [n_quantiles, n_evaluation, batch_shape]
    value = sorted_pred[:1] + (value.clamp(min=0.0, max=1.0) * pred_gap).sum(dim=0) # [n_evaluation, batch_shape] 
    
    # Sum up the interval that should be added, and consider the boundary case where score<0 or score>num_particle-1
    value = value + (score - sorted_quantile[0]).clamp(max=0.0) + (score - sorted_quantile[-1]).clamp(min=0.0) 
    return value


def _conformal_score_distribution(predictions, values):
    """
    Compute the non-conformity score of a set of values under some baseline predictor 
    Input:
        predictions: an instance that behaves like torch Distribution 
        values: array [n_evaluations, batch_shape], note that for values batch_shape is the last dimension while for predictions batch_shape is the first dimension
        
    Output:
        score: array [n_evaluations, batch_shape], where score[i, j] is the non-conformity score of values[i, j] under the prediction[j]
    """
    score = predictions.cdf(values) - 0.5
    return score 


def _conformal_iscore_distribution(predictions, score):
    """
    Compute the inverse of the non-conformity score defined in conformal_score_distribution.
    The goal is that conformal_iscore_distribution(predictions, conformal_score_distribution(predictions, labels))) = labels
    
    Input:
        predictions: an instance that behaves like torch Distribution
        score: array [n_evaluations, batch_shape]
        
    Output:
        value: array [n_evaluations, batch_shape], where value[i, j] is the inverse non-conformity score of score[i, j] under prediction[j]
    """
    return predictions.icdf((score + 0.5).clamp(min=1e-6, max=1-1e-6))  # Clamp it for numerical stability 



def _conformal_score_distribution1(predictions, values):
    """
    An alternative choice of the non-conformity score for distribution predictions
    """
    mean, std = compute_mean_std(predictions, reduction='none')
    score = (values - mean.view(1, -1)) / (1e-4 + std.view(1, -1))
    return score


def _conformal_iscore_distribution1(predictions, score):
    """
    An alternative choice of the non-conformity score for distribution predictions
    """
    mean, std = compute_mean_std(predictions, reduction='none')
    return score * (1e-4 + std.view(1, -1)) + mean.view(1, -1)


def _conformal_score_particle(predictions, values):
    """
    Compute the non-conformity score of a set of values under some baseline predictor 
    Input:
        predictions: array [batch_size, n_particles], a batch of particle predictions
        values: array [n_evaluations, batch_shape], note that for values batch_shape is the last dimension while for predictions batch_shape is the first dimension
        one_side: boolean, set one_side=False for conformal calibration. Set one_side=True for conformal interval prediction 
        
    Output:
        score: array [n_evaluations, batch_shape], where score[i, j] is the non-conformity score of values[i, j] under the prediction[j]
    """
    # print(predictions.shape, values.shape)
    sorted_pred = torch.sort(predictions, dim=1)[0].permute(1, 0)  
    sorted_pred = sorted_pred.unsqueeze(1)  # [num_particles, 1, batch_shape]

    # The score is equal to how many samples the value exceeds 
    score = (values.unsqueeze(0) - sorted_pred[:-1]) / (sorted_pred[1:] - sorted_pred[:-1] + 1e-10)
    score = score.clamp(min=0.0, max=1.0).sum(dim=0)   # If value exceeds all samples, its score so far is num_particle-1

    # Also consider values that are below the smallest sample or greater than the largest sample
    # A value has score <0 iff it is less than the smallest sample, and score >1 iff it is greater than the largest sample
    score = score + (values - sorted_pred[0]).clamp(max=0.0) + (values - sorted_pred[-1]).clamp(min=0.0) 
    score = score / (predictions.shape[1]-1) - 0.5  # Normalize the range of the score to be between [0, 1] if it's smaller within the smallest/largest particle
    return score
    
    
def _conformal_iscore_particle(predictions, score):
    """
    Compute the inverse of the non-conformity score defined in conformal_score_distribution.
    The goal is that conformal_iscore_distribution(predictions, conformal_score_distribution(predictions, labels))) = labels
    
    Input:
        predictions: array [batch_size, n_particles], a batch of particle predictions
        score: array [n_evaluations, batch_shape]
        
    Output:
        value: array [n_evaluations, batch_shape], where value[i, j] is the inverse non-conformity score of score[i, j] under prediction[j]
    """
    sorted_pred = torch.sort(predictions, dim=1)[0].permute(1, 0)

    # For each interval between two adjacent samples, compute whether the score is large enough such that this interval should be added
    num_particle = sorted_pred.shape[0] 
    score = (score + 0.5) * (num_particle-1) 
    
    interval_contribution = (score.unsqueeze(0) - torch.linspace(0, num_particle, num_particle+1, device=score.device)[:-2].view(-1, 1, 1)).clamp(min=0.0, max=1.0) 
    interval_value = sorted_pred.unsqueeze(1)[1:] - sorted_pred.unsqueeze(1)[:-1] # [num_particles, n_evaluations, batch_shape]

    # Sum up the interval that should be added, and consider the boundary case where score<0 or score>num_particle-1
    value = sorted_pred[:1] + score.clamp(max=0.0) + (score - num_particle+1).clamp(min=0.0) + \
        (interval_value * interval_contribution).sum(dim=0)
    return value
    
    
_conformal_score_functions = {'point_0': _conformal_score_point, 'interval_0': _conformal_score_interval,
                              'interval_1': _conformal_score_interval1, 
                            'particle_0': _conformal_score_particle, 'quantile_0': _conformal_score_quantile,
                            'distribution_1': _conformal_score_distribution,
                             'distribution_0': _conformal_score_distribution1}
_conformal_iscore_functions = {'point_0': _conformal_iscore_point, 'interval_0': _conformal_iscore_interval,
                               'interval_1': _conformal_iscore_interval1, 
                             'particle_0': _conformal_iscore_particle, 'quantile_0': _conformal_iscore_quantile,
                             'distribution_1': _conformal_iscore_distribution,
                              'distribution_0': _conformal_iscore_distribution1} 
_concat_predictions = {
    'point': lambda x: torch.cat(x, dim=0),
    'interval': lambda x: torch.cat(x, dim=0),
    'particle': lambda x: torch.cat(x, dim=0),
    'quantile': lambda x: torch.cat(x, dim=0),
    'distribution': lambda x: ConcatDistribution(x),
}

# Need to define these functions after _conformal_score_functions are defined because this function is built upon other score functions
def _conformal_score_ensemble(predictions, values):
    """
    Compute the non-conformity score of an ensemble prediction
    """
    # This function is special because it takes as input weights 
    scores = []
    for key in predictions:
        pred_type, pred_name = _parse_name(key)
        pred_component = predictions[key]
        scores.append(_conformal_score_functions['%s_0' % pred_type](pred_component, values))
    scores = torch.stack(scores, dim=0)
    return scores.sum(dim=0)


def _conformal_iscore_ensemble(predictions, score, min_search=-1e5, max_search=1e5):
    def forward_func(val):
        # The range of this function might not be [0, 1]
        # Force its range to be [0, 1] 
        return _conformal_score_ensemble(predictions, val)
    
    values = BisectionInverse(forward_func, min_search=min_search, max_search=max_search)(score)
    values[torch.isposinf(score)] = float('inf')
    values[torch.isneginf(score)] = -float('inf')
    return values


def _concat_ensemble_prediction(predictions):
    """
    Concatenate a list of ensemble predictions. Each ensemble prediction in the list must have the same keys
    """
    assert len(predictions) > 0, "Must have at least one predictions to concatenate"
    
    result = {}
    for key in predictions[0]:
        pred_type, pred_name = _parse_name(key)
        pred_list = [component[key] for component in predictions]
        result[key] = _concat_predictions[pred_type](pred_list)
    return result


# Update the score function list to contain the ensemble score function 
_conformal_score_functions['ensemble_0'] = _conformal_score_ensemble
_conformal_iscore_functions['ensemble_0'] = _conformal_iscore_ensemble 
_concat_predictions['ensemble'] = _concat_ensemble_prediction



class ConformalBase(Calibrator):
    def __init__(self, input_type='interval', score_func=0, verbose=False):
        """
        Inputs:
            input_type: str, one of the regression input types
            score_func: int, the score function to use. The index corresponds to the paper (cite). 
        """
        super(ConformalBase, self).__init__(input_type=input_type)
        self.verbose = verbose
        self.predictions = []
        self.labels = []
#         if input_type not in _conformal_score_functions:
#             assert False, "Input type %s not supported, supported types are %s" % (input_type, '/'.join(_conformal_score_functions.keys()))

        self.score_func = score_func
        assert '%s_%d' % (input_type, score_func) in _conformal_score_functions, "score function %s_%d not available" % (input_type, score_func)
        
    def train(self, predictions, labels):
        """
        Train the conformal calibration from scratch 
        Inputs:
            predictions: a batch of predictions generated by the base predictor. Must have the correct type that matches the input_type argument
            labels: a batch of labels, must be on the same device as predictions
        """
        self.check_type(predictions)
        self.to(predictions) 
        
        self.predictions = [predictions]
        self.labels = [labels.to(_get_prediction_device(predictions))]
        return self
    
    def update(self, predictions, labels):
        """
        Update the conformal calibrator online with new labels 
        Inputs:
            predictions: a batch of predictions generated by the base predictor. Must have the correct type that matches the input_type argument
            labels: a batch of labels, must be on the same device as predictions
        """
        self.check_type(predictions)
        self.to(predictions)   # Optionally change the device to the device predictions resides in
        
        self.predictions.append(predictions)
        self.labels.append(labels.to(_get_prediction_device(predictions)))
        return self 
    
    def to(self, device):
        """ 
        Move every torch tensor owned by this class to a new device 
        Inputs:
            device: a torch.device instance, alternatively it could be a torch.Tensor or a prediction object
        """
        if not type(device).__name__ == 'device':
            device = _get_prediction_device(device)   # This handles the case that the input is a tensor or a prediction
        if len(self.predictions) != 0 and device != self.labels[0].device:
            # if self.device is not None:   # If this is not the first time device is set, then issue a warning
            #     print("Warning: device of conformal calibrator has been changed from %s to %s, this could be because the inputs had difference devices (not recommended)." % (str(self.labels[0].device), str(device)))
            self.predictions = [_move_prediction_device(pred, device) for pred in self.predictions]
            self.labels = [label.to(device) for label in self.labels]
        self.device = device 
        return self 
    
    def __call__(self, predictions):
        assert False, "ConformalBase.__call__ is unimplemented"
    
    
class ConformalCalibrator(ConformalBase):
    def __init__(self, input_type='interval', interpolation='linear', score_func=0, verbose=False):
        """
        Inputs:
            interpolation: 'linear', 'random' or 'naf', the interpolation used when computing the CDF/ICDF functions. 
                NAF is slow but produces smoother CDFs. 
                Random has better calibration error (it should have perfect calibration) but has non-continuous CDF. 
                Linear achieves good trade-off between speed, smoothness of CDF and calibration error. 
        """
        super(ConformalCalibrator, self).__init__(input_type=input_type, score_func=score_func, verbose=verbose) 
        if interpolation == 'linear':
            self.distribution_class = DistributionConformalLinear
        elif interpolation == 'random':
            self.distribution_class = DistributionConformalRandom
        else:
            assert interpolation == 'naf', 'interpolation can only be linear/naf'
            self.distribution_class = DistributionConformalNAF 
            
    def __call__(self, predictions):
        """ 
        Output the calibrated probabilities given input base prediction. Because __call__ returns meaningful results, there must be at least two samples (observed by either train or update). 
        Inputs:
            predictions: a batch of predictions generated by the base predictor. Must have the correct type that matches the input_type argument
        Outputs:
            results: a class that behaves like torch Distribution. The calibrated probabilities. 
        """
        self.check_type(predictions) 
        self.to(predictions)
        score_func_name = '%s_%d' % (self.input_type, self.score_func)
        results = self.distribution_class(val_predictions=_concat_predictions[self.input_type](self.predictions), 
                                         val_labels=torch.cat(self.labels, dim=0),
                                         test_predictions=predictions,
                                         score_func=_conformal_score_functions[score_func_name], 
                                         iscore_func=_conformal_iscore_functions[score_func_name],
                                         verbose=self.verbose)
        return results

    
class ConformalIntervalPredictor(ConformalBase):
    def __init__(self, input_type='interval', coverage='exact', score_func=0, confidence=0.95, verbose=False):
        """
        Inputs:
            input_type: str, one of the regression input types
            score_func: int, the score function to use. The index corresponds to the paper (cite). 
            coverage: the coverage can be 'exact' or '1/n'. If the coverage is exact, then the algorithm can output [-inf, +inf] intervals
        """
        super(ConformalIntervalPredictor, self).__init__(input_type=input_type, score_func=score_func, verbose=verbose)
        self.val_scores = None
        
        assert coverage == 'exact' or coverage == '1/n' or coverage == '1/N', "Coverage can only be 'exact' or '1/N'"
        self.coverage = coverage
        self.confidence = confidence
        
    
    def __call__(self, predictions, confidence=None):
        """
        Input: 
            confidence: float, the confidence level of the prediction intervals. If None then uses the default confidence interval specified in the constructor 
        """
        if confidence is None:
            confidence = self.confidence 
        assert confidence > 0 and confidence < 1., 'Confidence must be a number of (0, 1)'
        
        self.check_type(predictions) 
        self.to(predictions)
        
        score_func_name = '%s_%d' % (self.input_type, self.score_func)
        score_func = _conformal_score_functions[score_func_name]
        iscore_func = _conformal_iscore_functions[score_func_name]
        
        # Get the sorted non-conformity score on the validation set 
        test_shape = _get_prediction_batch_shape(predictions)
        
        val_scores = torch.sort(score_func(_concat_predictions[self.input_type](self.predictions), torch.cat(self.labels, dim=0)).abs().flatten())[0]
        val_scores_ge = torch.linspace(0, 1, len(val_scores)+2, device=self.device)[:-1]  # Generate 0, 1/N+1, ..., N/N+1
        val_scores_geq = torch.linspace(0, 1, len(val_scores)+2, device=self.device)[:-1]
        
        # Compute the quantiles of the non-conformity scores, and handle situations where the quantiles are identical. 
        while True:   # This iteration is for handling values with identical non-conformity score 
            new_val_scores_ge = val_scores_ge.clone()
            new_val_scores_ge[1:-1][val_scores[:-1] == val_scores[1:]] = val_scores_ge[2:][val_scores[:-1] == val_scores[1:]]
            if (new_val_scores_ge - val_scores_ge).sum() == 0:
                break
            val_scores_ge = new_val_scores_ge

        val_scores_geq = val_scores_ge[:-1] + (val_scores_ge[1:] - val_scores_ge[:-1]).view(1, -1) * torch.rand(test_shape, device=self.device).view(-1, 1)
        val_scores_ge = val_scores_ge[1:].view(1, -1).repeat(test_shape, 1)

        while True:   # This iteration is for handling values with identical non-conformity score 
            new_val_scores_geq = val_scores_geq.clone()
            new_val_scores_geq[:, 1:][:, val_scores[:-1] == val_scores[1:]] = val_scores_geq[:, :-1][:, val_scores[:-1] == val_scores[1:]]
            if (new_val_scores_geq - val_scores_geq).sum() == 0:
                break
            val_scores_geq = new_val_scores_geq
        
        
        # Now we should have the following:
        # val_scores_ge[i] is 1/(N+1)#{n | A(x_n, y_n) < a}, if a > val_scores[i]
        # val_scores_geq[i] is 1/(N+1)#{n | A(x_n, y_n) < a} + U(0, 1)/(N+1)#{n | A(x_n, y_n) = a} if a = val_scores[i]
        
        if self.coverage == 'exact':   # If coverage is exact, add U(0, 1)/(N+1)
            noise = torch.rand(test_shape, 1, device=self.device) / (len(val_scores) + 1)
            val_scores_geq = val_scores_geq + noise
            val_scores_ge = val_scores_ge + noise
        else:   # If coverage is up to 1/N accuracy, then add 1/(N+1), we always under cover to achieve the smallest intervals 
            val_scores_geq = val_scores_geq + 1/(len(val_scores) + 1)
            val_scores_ge = val_scores_ge + 1/(len(val_scores) + 1)

        geq_index = (val_scores_geq <= confidence).type(torch.int32).sum(dim=1) 
        ge_index = (val_scores_ge <= confidence).type(torch.int32).sum(dim=1) 

        eps = 1e-5 
        target_scores = torch.maximum((val_scores[ge_index.clamp(max=len(val_scores)-1)] - eps), val_scores[geq_index-1])
        y_ub = iscore_func(predictions, target_scores.view(1, -1)).flatten()
        y_lb = iscore_func(predictions, -target_scores.view(1, -1)).flatten()
        
        y_ub[ge_index > len(val_scores)-1] = float('inf')
        y_lb[ge_index > len(val_scores)-1] = -float('inf')

        result = torch.stack([y_lb, y_ub], axis=-1)
        return result