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
from .basic import Calibrator, ConcatDistribution
from ..models.flow import NafFlow
from .. import _implicit_quantiles, _get_prediction_device, _move_prediction_device

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
            val_scores = self.score(val_predictions, val_labels.view(1, -1)).flatten().sort()[0] 
            # Prepend the 0 quantile and append the 1 quantile for convenient handling of boundary conditions
            self.val_scores = torch.cat([val_scores[:1] - (val_scores[1:] - val_scores[:-1]).mean(dim=0, keepdims=True), 
                                         val_scores, 
                                         val_scores[-1:] + (val_scores[1:] - val_scores[:-1]).mean(dim=0, keepdims=True)])   
    
    def to(self, device):
        if self.device != device:
            self.device = device
            self.val_scores = self.val_scores.to(device)
            self.test_predictions = _move_prediction_device(self.test_predictions, device)
            
    def cdf(self, value):
        assert False, "Not implemented"
        
    def icdf(self, cdf):
        assert False, "Not implemented"
        
    def rsample(self, sample_shape):
        """
        Draw a set of samples from the distribution
        """
        rand_vals = torch.rand(list(sample_shape) + [self.batch_shape[0]])
        return self.icdf(rand_vals.view(-1, self.batch_shape[0])).view(rand_vals.shape)
    
    def sample(self, batch_shape=torch.Size([])):
        pass
    
    def log_prob(self, value):
        pass
    
    def shape_inference(self, value):
        # Enumerate all the valid input shapes for value
        if type(value) == int or type(value) == float:  
            return value.view(1, 1), self.batch_shape[0]
        elif len(value.shape) == 1 and value.shape[0] == 1:  # If the value is 1-D it must be either 1 or equal to batch_shape[0]
            return value.view(1, 1), self.batch_shape[0]
        elif len(value.shape) == 1 and value.shape[0] == self.batch_shape[0]:   # If the value is 1-D it must be either 1 or equal to batch_shape[0]
            return value.view(1, -1), self.batch_shape[0]
        elif len(value.shape) == 2:
            return value, [len(value), self.batch_shape[0]]
        else:
            assert False, "Shape [%s] invalid" % ', '.join([str(shape) for shape in value.shape])
    
    
class DistributionConformalLinear(DistributionConformal):
    def __init__(self, val_predictions, val_labels, test_predictions, score_func, iscore_func):
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
        return value.view(out_shape)

    

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
    
    
def _conformal_score_point(predictions, values):
    """
    Compute the non-conformity score of a set of values under some baseline predictor 
    Input:
        predictions: array [batch_shape], a batch of point predictions
        values: array [n_evaluations, batch_shape], note that for values batch_shape is the last dimension while for predictions batch_shape is the first dimension
        
    Output:
        score: array [n_evaluations, batch_shape], where score[i, j] is the non-conformity score of values[i, j] under the prediction[j]
    """
    return values - predictions.view(1, -1)

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
    return (values - predictions.min(dim=1, keepdims=True)[0].permute(1, 0)) / (predictions[:, 1:2] - predictions[:, 0:1]).abs().permute(1, 0)
    
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
    return predictions.min(dim=1, keepdims=True)[0].permute(1, 0) + score * (predictions[:, 1:2] - predictions[:, 0:1]).abs().permute(1, 0)
    


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
    score = sorted_quantile[:1] + (score.clamp(min=0.0, max=1.0) * quantile_gap).sum(dim=0)   # If value exceeds all samples, its score so far is num_particle-1, [n_evaluations, batch_shape]

    # Also consider values that are below the smallest sample or greater than the largest sample
    # A value has score <0 iff it is less than the smallest sample, and score >1 iff it is greater than the largest sample
    score = score + (values - sorted_pred[0]).clamp(max=0.0) + (values - sorted_pred[-1]).clamp(min=0.0) 
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
    return predictions.cdf(values)

def _conformal_iscore_distribution(predictions, score):
    """
    Compute the inverse of the non-conformity score defined in conformal_score_distribution.
    The goal is that conformal_iscore_distribution(predictions, conformal_score_distribution(predictions, labels))) = labels
    
    Input:
        predictions: array [batch_size, n_quantiles] or [batch_size, n_quantiles, 2], a batch of quantile predictions
        score: array [n_evaluations, batch_shape]
        
    Output:
        value: array [n_evaluations, batch_shape], where value[i, j] is the inverse non-conformity score of score[i, j] under prediction[j]
    """
    return predictions.icdf(score)

def _conformal_score_particle(predictions, values):
    """
    Compute the non-conformity score of a set of values under some baseline predictor 
    Input:
        predictions: array [batch_size, n_particles], a batch of particle predictions
        values: array [n_evaluations, batch_shape], note that for values batch_shape is the last dimension while for predictions batch_shape is the first dimension
        
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
    interval_contribution = (score.unsqueeze(0) - torch.linspace(0, num_particle, num_particle+1)[:-2].view(-1, 1, 1)).clamp(min=0.0, max=1.0) 
    interval_value = sorted_pred.unsqueeze(1)[1:] - sorted_pred.unsqueeze(1)[:-1] # [num_particles, n_evaluations, batch_shape]

    # Sum up the interval that should be added, and consider the boundary case where score<0 or score>num_particle-1
    value = sorted_pred[:1] + score.clamp(max=0.0) + (score - num_particle+1).clamp(min=0.0) + \
        (interval_value * interval_contribution).sum(dim=0)
    return value
    
    
conformal_score_functions = {'point': _conformal_score_point, 'interval': _conformal_score_interval,
                            'particle': _conformal_score_particle, 'quantile': _conformal_score_quantile,
                            'distribution': _conformal_score_distribution}
conformal_iscore_functions = {'point': _conformal_iscore_point, 'interval': _conformal_iscore_interval,
                             'particle': _conformal_iscore_particle, 'quantile': _conformal_iscore_quantile,
                             'distribution': _conformal_iscore_distribution} 

concat_predictions = {
    'point': lambda x: torch.cat(x, dim=0),
    'interval': lambda x: torch.cat(x, dim=0),
    'particle': lambda x: torch.cat(x, dim=0),
    'quantile': lambda x: torch.cat(x, dim=0),
    'distribution': lambda x: ConcatDistribution(x)
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
        self._change_device(predictions)
        
        self.predictions.append(predictions)
        self.labels.append(labels)
        
    def to(self, device):
        """ Move every torch tensor owned by this class to a new device """
        if len(self.predictions) != 0 and device != self.labels[0].device:
            self.predictions = [_move_prediction_device(pred, device) for pred in self.predictions]
            self.labels = [label.to(device) for label in self.labels]
            
    def __call__(self, predictions):
        self._change_device(predictions)
        return self.distribution_class(val_predictions=concat_predictions[self.input_type](self.predictions), 
                                     val_labels=torch.cat(self.labels, dim=0),
                                     test_predictions=predictions,
                                     score_func=conformal_score_functions[self.input_type], 
                                     iscore_func=conformal_iscore_functions[self.input_type])
