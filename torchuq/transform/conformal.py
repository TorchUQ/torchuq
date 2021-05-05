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
import matplotlib.pyplot as plt
import os, sys, shutil, copy, time, random
from .basic import Calibrator 
from .. import models
from torchuq.models.flow import NafFlow


class DistributionConformal:
    """
    Abstract baseclass for a distribution that arises from conformal calibration. 
    This class behaves like torch.distribution.Distribution, and supports the cdf, icdf and rsample functions. 
    
    score_func is a function that take as input a batched predictions q, and an array v of values with shape [n_values, batch_shape] 
    returns an array s of shape [n_values, batch_shape] where s_{ij} is the score of v_{ij} under prediction q_j 
    
    iscore_func is a function that take as input a batched prediction q, and an array s os scores with shape [n_values, batch_shape] 
    returns an array v of shape [n_values, batch_shape]  which is the inverse of score_func (i.e. iscore_func(q, score_func(q, v)) = v)
    """
    def __init__(self, val_predictions, val_labels, test_predictions, score_func, iscore_func):
        self.score = score_func 
        self.iscore = iscore_func 
        self.test_predictions = test_predictions 
        self.device = test_predictions.device
        with torch.no_grad():
            self.batch_shape = self.score(test_predictions, torch.zeros(1, 1, device=test_predictions.device)).shape[1]  # A hack to find out the number of distributions
            
            # Compute the scores for all the predictions in the validation set and sort them from small to large
            val_scores = self.score(val_predictions, val_labels.view(1, -1)).flatten().sort()[0]
            # Prepend the 0 quantile and append the 1 quantile for convenient handling of boundary conditions
            self.val_scores = torch.cat([val_scores[:1] - (val_scores[1:] - val_scores[:-1]).mean(dim=0, keepdims=True), 
                                         val_scores, 
                                         val_scores[-1:] + (val_scores[1:] - val_scores[:-1]).mean(dim=0, keepdims=True)])   
            
    def cdf(self, value):
        assert False, "Not implemented"
        
    def icdf(self, cdf):
        assert False, "Nor implemented"
        
    def rsample(self, sample_shape):
        """
        Draw a set of samples from the distribution
        """
        rand_vals = torch.rand(list(sample_shape) + [self.batch_shape])
        return self.icdf(rand_vals.view(-1, self.batch_shape)).view(rand_vals.shape)
    
    def shape_inference(self, value):
        # Enumerate all the valid input shapes for value
        if type(value) == int or type(value) == float:
            return value.view(1, 1), [self.batch_shape]
        elif len(value.shape) == 1 and value.shape[0] == 1:
            return value.view(1, 1), [self.batch_shape]
        elif len(value.shape) == 1 and value.shape[0] == self.batch_shape:
            return value.view(1, -1), [self.batch_shape]
        elif len(value.shape) == 2:
            return value, [len(value), self.batch_shape]
        else:
            assert False, "Shape [%s] invalid" % ', '.join([str(shape) for shape in value.shape])
    
    
class DistributionConformalLinear(DistributionConformal):
    def __init__(self, val_predictions, val_labels, test_predictions, score_func, iscore_func):
        super(DistributionConformalLinear, self).__init__(val_predictions, val_labels, test_predictions, score_func, iscore_func)
        
    def cdf(self, value):
        """
        The CDF at value
        Input:
        - value: an array of shape [val_batch_size, batch_shape] or shape [batch_size] 
        """
        # First perform automatic shape induction and convert value into an array of shape [batch_size, batch_shape]
        value, out_shape = self.shape_inference(value)
        
        # Move value to the device of test_predictions to avoid device mismatch error
        out_device = value.device
        value = value.to(self.test_predictions)
        
        # Non-conformity score
        scores = self.score(self.test_predictions, value)
        
        # Compare the non-conformity score to the validation set non-conformity scores
        quantiles = self.val_scores.view(1, 1, -1)
        comparison = (scores.unsqueeze(-1) - quantiles[:, :, :-1]) / (quantiles[:, :, 1:] - quantiles[:, :, :-1] + 1e-20) 
        cdf = comparison.clamp(min=0, max=1).sum(dim=-1) / (len(self.val_scores) - 1)
        return cdf.view(out_shape).to(out_device) 
    
    def icdf(self, cdf):
        """
        Get the inverse CDF
        Input:
        - cdf: an array of shape [batch_size, n_prediction] or shape [n_prediction], each entry should take values in [0, 1]
        Supports automatic shape induction, e.g. if cdf has shape [batch_size, 1] it will automatically be converted to shape [batch_size, n_prediction]
        
        Output:
        The value of inverse CDF function at the queried cdf values
        """
        cdf, out_shape = self.shape_inference(cdf)   # Convert cdf to have shape [batch_size, batch_shape]
        
        # Move value to the device of test_predictions to avoid device mismatch error
        out_device = cdf.device
        cdf = cdf.to(self.test_predictions)
        
        quantiles = cdf * (len(self.val_scores) - 1)
        ratio = torch.ceil(quantiles) - quantiles
        target_score = self.val_scores[torch.floor(quantiles).type(torch.long)] * ratio + \
            self.val_scores[torch.ceil(quantiles).type(torch.long)] * (1 - ratio) 
        value = self.iscore(self.test_predictions, target_score)
        return value.view(out_shape).to(out_device)


class DistributionConformalNAF(DistributionConformal):
    def __init__(self, val_predictions, val_labels, test_predictions, score_func, iscore_func, verbose=True):
        super(DistributionConformalNAF, self).__init__(val_predictions, val_labels, test_predictions, score_func, iscore_func)

        # Train both a flow and an inverse flow to avoid the numerical instability of inverting a flow
        self.flow = NafFlow(feature_size=30).to(val_labels.device)
        self.iflow = NafFlow(feature_size=30).to(val_labels.device)
        target_cdf = torch.linspace(0, 1, len(self.val_scores), device=val_labels.device)
        flow_optim = optim.Adam(list(self.flow.parameters()) + list(self.iflow.parameters()), lr=1e-3)
        # TODO: need to tune these training parameters for better performance, mostly the learning rate and whether annealing is needed
        for iteration in range(10000):
            flow_optim.zero_grad()

            cdfs, _ = self.flow(self.val_scores.view(-1, 1).type(torch.float32))
            scores, _ = self.iflow(target_cdf.view(-1, 1).type(torch.float32))
            
            loss = (cdfs.flatten() - target_cdf).pow(2).mean() + (scores.flatten() - self.val_scores).pow(2).mean()
            loss.backward()
            flow_optim.step()
            if verbose and iteration % 1000 == 0:
                # print(cdfs.shape, target_cdf.shape, scores.shape, self.val_scores.shape)
                print("Iteration %d, loss=%.5f" % (iteration, loss))
    
    def cdf(self, value):
        value, out_shape = self.shape_inference(value)
        score = self.score(self.test_predictions, value)
        cdf, _ = self.flow(score.view(-1, 1))
        return cdf.view(out_shape).clamp(min=0.0, max=1.0)
    
    def icdf(self, cdf):
        cdf, out_shape = self.shape_inference(cdf)
        score, _ = self.iflow(cdf.view(-1, 1))
        score = score.view(cdf.shape)
        value = self.iscore(self.test_predictions, score)
        return value.view(out_shape)
    
    
class DistributionConcat:
    """
    Class that concat multiple classes that behave like torch.distributions.Distribution (but only the cdf, icdf and rsample interface are implemented)
    """
    def __init__(self, predictions):
        self.predictions = predictions
        self.batch_shapes = torch.tensor([0] + [prediction.batch_shape[0] for prediction in predictions])
        self.batch_shapes = torch.cumsum(self.batch_shapes, dim=0)
        self.batch_shape = torch.Size([self.batch_shapes[-1]])

    def cdf(self, value):
        return torch.cat([prediction.cdf(value[..., self.batch_shapes[i]:self.batch_shapes[i+1]]) for i, prediction in enumerate(self.predictions)], dim=-1)
    
    def icdf(self, value):
        return torch.cat([prediction.icdf(value[..., self.batch_shapes[i]:self.batch_shapes[i+1]]) for i, prediction in enumerate(self.predictions)], dim=-1)
    
    def rsample(self, sample_shape):
        return torch.cat([prediction.rsample(sample_shape) for prediction in enumerate(self.predictions)], dim=-1)
    
    
    
def conformal_score_point(predictions, values):
    return values - predictions.view(1, -1)

def conformal_iscore_point(predictions, score):
    return predictions.view(1, -1) + score

def conformal_score_interval(predictions, values):
    return (values - predictions.min(dim=0, keepdims=True)[0]) / (predictions[:, 1:2] - predictions[:, 0:1]).abs()
    
def conformal_iscore_interval(predictions, score):
    return predictions.min(dim=0, keepdims=True)[0] + score * (predictions[:, 1:2] - predictions[:, 0:1]).abs()
    
def conformal_score_quantile(predictions, values):
    if len(predictions.shape) == 2:
        sorted_quantile = torch.linspace(0, 1, predictions.shape[1]+2, device=predictions.device)[1:-1].view(-1, 1)
        sorted_pred, _ = torch.sort(predictions.transpose(1, 0), dim=0)
    else:
        sorted_quantile, _ = torch.sort(predictions[:, :, 1].transpose(1, 0), dim=0)
        sorted_pred, _ = torch.sort(predictions[:, :, 0].transpose(1, 0), dim=0)
        
    sorted_pred = sorted_pred.unsqueeze(1)  # [num_quantiles, 1, batch_shape]
    quantile_gap = (sorted_quantile[1:] - sorted_quantile[:-1]).unsqueeze(1) # [num_quantiles-1, 1, batch_shape]
    
    # print('quantile_gap', quantile_gap.shape)
    # The score is equal to how many quantiles the value exceeds
    score = (values.unsqueeze(0) - sorted_pred[:-1]) / (sorted_pred[1:] - sorted_pred[:-1])
    score = sorted_quantile[0] + (score.clamp(min=0.0, max=1.0)  * quantile_gap).sum(dim=0)   # If value exceeds all samples, its score so far is num_particle-1

    # Also consider values that are below the smallest sample or greater than the largest sample
    # A value has score <0 iff it is less than the smallest sample, and score >1 iff it is greater than the largest sample
    score = score + (values - sorted_pred[0]).clamp(max=0.0) + (values - sorted_pred[-1]).clamp(min=0.0) 
    # print('score', score.shape)
    return score

def conformal_iscore_quantile(predictions, score):
    if len(predictions.shape) == 2:
        sorted_quantile = torch.linspace(0, 1, predictions.shape[1]+2, device=predictions.device)[1:-1].view(-1, 1)
        sorted_pred, _ = torch.sort(predictions.transpose(1, 0), dim=0)
    else:
        sorted_quantile, _ = torch.sort(predictions[:, :, 1].transpose(1, 0), dim=0)
        sorted_pred, _ = torch.sort(predictions[:, :, 0].transpose(1, 0), dim=0)
    
    sorted_quantile = sorted_quantile.unsqueeze(1) # [num_quantiles, batch_size, batch_shape]
    pred_gap = (sorted_pred[1:] - sorted_pred[:-1]).unsqueeze(1)
    
    # For each interval between two adjacent samples, compute whether the score is large enough such that this interval should be added
    value = (score.unsqueeze(0) - sorted_quantile[:-1]) / (sorted_quantile[1:] - sorted_quantile[:-1])
    value = sorted_pred[:1] + (value.clamp(min=0.0, max=1.0) * pred_gap).sum(dim=0) 
    
    # Sum up the interval that should be added, and consider the boundary case where score<0 or score>num_particle-1
    value = value + (score - sorted_quantile[0]).clamp(max=0.0) + (score - sorted_quantile[-1]).clamp(min=0.0) 
    return value

def conformal_score_distribution(predictions, values):
    return predictions.cdf(values)

def conformal_iscore_distribution(predictions, score):
    return predictions.icdf(score)

def conformal_score_particle(predictions, values):
    """
    predictions is an array of shape [num_particles, batch_shape]
    values is an array of shape [batch_size, batch_shape]

    Return the inverse of the score as an array of shape [batch_size, batch_shape]
    """
    # print(predictions.shape, values.shape)
    sorted_pred, _ = torch.sort(predictions, dim=0)  
    sorted_pred = sorted_pred.unsqueeze(1)  # [num_particles, batch_size, batch_shape]

    # The score is equal to how many samples the value exceeds 
    score = (values.unsqueeze(0) - sorted_pred[:-1]) / (sorted_pred[1:] - sorted_pred[:-1])
    score = score.clamp(min=0.0, max=1.0).sum(dim=0)   # If value exceeds all samples, its score so far is num_particle-1

    # Also consider values that are below the smallest sample or greater than the largest sample
    # A value has score <0 iff it is less than the smallest sample, and score >1 iff it is greater than the largest sample
    score = score + (values - sorted_pred[0]).clamp(max=0.0) + (values - sorted_pred[-1]).clamp(min=0.0) 
    return score
    
    
def conformal_iscore_particle(predictions, score):
    """
    predictions is an array of shape [num_particles, batch_shape]
    score is an array of shape [batch_size, batch_shape]

    Return the inverse of the score as an array of shape [batch_size, batch_shape]
    """
    sorted_pred, _ = torch.sort(predictions, dim=0)
        
    # For each interval between two adjacent samples, compute whether the score is large enough such that this interval should be added
    num_particle = sorted_pred.shape[0] 
    interval_contribution = (score.unsqueeze(0) - torch.linspace(0, num_particle, num_particle+1)[:-2].view(-1, 1, 1)).clamp(min=0.0, max=1.0) 
    interval_value = sorted_pred.unsqueeze(1)[1:] - sorted_pred.unsqueeze(1)[:-1] # [num_particles, batch_size, batch_shape]

    # Sum up the interval that should be added, and consider the boundary case where score<0 or score>num_particle-1
    value = sorted_pred[:1] + score.clamp(max=0.0) + (score - num_particle+1).clamp(min=0.0) + \
        (interval_value * interval_contribution).sum(dim=0)
    return value
    
conformal_score_functions = {'point': conformal_score_point, 'interval': conformal_score_interval,
                            'particle': conformal_score_particle, 'quantile': conformal_score_quantile,
                            'distribution': conformal_score_distribution}
conformal_iscore_functions = {'point': conformal_iscore_point, 'interval': conformal_iscore_interval,
                             'particle': conformal_iscore_particle, 'quantile': conformal_iscore_quantile,
                             'distribution': conformal_iscore_distribution} 

concat_predictions = {
    'point': lambda x: torch.cat(x, dim=-1),
    'interval': lambda x: torch.cat(x, dim=-1),
    'particle': lambda x: torch.cat(x, dim=-1),
    'quantile': lambda x: torch.cat(x, dim=-1),
    'distribution': lambda x: DistributionConcat(x)
}
    

class ConformalCalibrator(Calibrator):
    def __init__(self, input_type='interval', interpolation='naf'):
        """
        interpolation: linear or naf 
        """
        super(ConformalCalibrator, self).__init__(input_type=input_type)
        self.predictions = []
        self.labels = []
        if input_type not in conformal_score_functions:
            assert False, "Input type %s not supported, supported types are %s" % (input_type, '/'.join(conformal_score_functions.keys()))
        if interpolation == 'linear':
            self.distribution_class = DistributionConformalLinear
        else:
            assert interpolation == 'naf', 'interpolation can only be linear/naf'
            self.distribution_class = DistributionConformalNAF 
        
    def train(self, predictions, labels):
        """
        prediction can be either: a point prediction, a confidence interval, 
        """
        self.check_type(predictions)
        self.predictions = [predictions]
        self.labels = [labels]
    
    def update(self, predictions, labels):
        self.check_type(predictions)
        self.predictions.append(predictions)
        self.labels.append(labels)
        
    def __call__(self, predictions):
        return self.distribution_class(val_predictions=concat_predictions[self.input_type](self.predictions), 
                                     val_labels=torch.cat(self.labels, dim=0),
                                     test_predictions=predictions,
                                     score_func=conformal_score_functions[self.input_type], 
                                     iscore_func=conformal_iscore_functions[self.input_type])


    
