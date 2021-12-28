import torch
from torch.nn import functional as F
from torch.distributions.normal import Normal 
from .utils import BisectionInverse
from .. import _implicit_quantiles, _check_valid, _parse_name
from ..evaluate.distribution import compute_mean, compute_std, plot_density_sequence


def distribution_to_particle(predictions, n_particles=50):
    """ Convert a distribution prediction to a particle prediction
    
    Args:
        predictions (Distribution): a batch of distribution predictions. 
        n_particles (int): the number of particles to sample. 
    
    """
    return predictions.sample((n_particles,)).permute(1, 0)


def distribution_to_quantile(predictions, quantiles=None, n_quantiles=None):
    """ Convert a distribution prediction to a quantile prediction 
    
    Args:
        predictions (Distribution): a batch of distribution predictions
        quantiles (tensor): a set of quantiles. One and only one of quantiles or n_quantiles can be specified (the other should be None). 
        n_quantiles (int): the number of quantiles. 
        
    Returns:
        tensor: a batch of quantile predictions. 
    """
    assert n_quantiles is None or quantiles is None, "Cannot specify the quantiles or n_quantiles simultaneously"
    assert n_quantiles is not None or quantiles is not None, "Must specify either quantiles or n_quantiles"

    if n_quantiles is not None:
        quantiles = _implicit_quantiles(n_quantiles).view(-1, 1)  
        results = predictions.icdf(quantiles).transpose(1, 0)
    elif len(quantiles.shape) == 1:
        results = predictions.icdf(quantiles.view(-1, 1)).transpose(1, 0)
        results = torch.stack([results, quantiles.view(1, -1).repeat(results.shape[0], 1)], dim=-1)
    else:
        results = predictions.icdf(quantiles.transpose(1, 0)).transpose(1, 0)
        results = torch.stack([results, quantiles], dim=-1)
    
    return results
    
    
def distribution_to_point(predictions, functional='mean'):
    """Convert a distribution prediction to a point prediction by taking the mean or the median
    
    Args:
        predictions (Distribution): a batch of distribution predictions.
        functional (str): can be 'mean' or 'median'.
        
    Returns:
        tensor: a batch of point predictions. 
    """
    assert functional == 'mean' or functional == 'median'
    if functional == 'mean':
        return predictions.icdf(_implicit_quantiles(100).view(-1, 1)).mean(dim=0)
    else:
        return predictions.icdf(0.5)


def distribution_to_interval(predictions, confidence=0.95):
    """ Convert a distribution prediction to an interval prediction by finding the smallest interval
    
    Args:
        predictions (Distribution): a batch of distribution predictions
        confidence (float): the probability of the credible interval. 
    
    Returns:
        tensor: a batch of interval predictions 
    """
    # Grid search the best interval with the minimum average length 
    l_start = 0.0
    r_start = 1-confidence
    with torch.no_grad():
        for iteration in range(2): # Repeat the search procedure to get more finegrained results
            avg_length = torch.zeros(100)
            queried_values = torch.linspace(l_start, r_start, 100)
            for i, c_start in enumerate(queried_values):
                interval = predictions.icdf(torch.tensor([c_start, c_start+confidence]).view(-1, 1))
                avg_length[i] = (interval[1] - interval[0]).mean()
            best_ind = avg_length.argmin()
            l_start = queried_values[max(best_ind - 1, 0)]
            r_start = queried_values[min(best_ind + 1, 99)]
    c_start = (l_start + r_start) / 2.
    return predictions.icdf(torch.tensor([c_start, c_start+confidence]).view(-1, 1)).transpose(1, 0)


class DistributionKDE:
    """ A mixture of Gaussian distribution 
    
    Even though pytorch has native support for mixture distribution, it does not have an icdf method which is crucial and difficult to implement
    This implementation includes an icdf function 
        
    Args:
        loc (tensor): array [batch_size, n_components], the set of centers of the Gaussian distributions
        scale (tensor): array [batch_size, n_components], the set of stddev of the Gaussian distributions
        weight (tensor): array [batch_size, n_components], the weight of each mixture component. If set to None then all mixture components have the same weight
    """    
    def __init__(self, loc, scale, weight=None):
        self.loc = loc
        self.scale = scale
        if weight is None:
            self.weight = torch.ones_like(loc) / loc.shape[1]
        else:
            self.weight = weight / weight.sum(dim=1, keepdim=True)
            
        self.normals = Normal(loc=loc, scale=scale)
        self.min_search = (loc - 10 * scale).min()
        self.max_search = (loc + 10 * scale).max()
        self.device = self.loc.device
        self.batch_shape = loc.shape[0]
        
    def cdf(self, value):
        """ Computes the cumulative density evaluated at value
        
        Args:
            value (tensor): an array of shape [batch_size], [1] or [num_cdfs, batch_size]
        
        Returns:
            tensor: the evaluated CDF. 
        """
        cdf = self.normals.cdf(value.unsqueeze(-1))  # This should have shape [..., batch_size, n_components]
        
        weight = self.weight
        for i in range(len(value.shape) - 1):
            weight = weight.unsqueeze(0)   # Prepand enough dimensions so that size would match
        return (cdf * weight).sum(dim=-1)  
    
    def log_prob(self, value):
        """ Computes the log likelihood evaluated at value 
        
        Args:
            value (tensor): an array of shape [batch_size], [1] or [num_cdfs, batch_size]
        
        Returns:
            tensor: the evaluated log probability. 
        """
        log_prob = self.normals.log_prob(value.unsqueeze(-1))
        
        weight = self.weight
        for i in range(len(value.shape) - 1):
            weight = weight.unsqueeze(0)   # Prepand enough dimensions so that size would match
            
        return torch.logsumexp(log_prob + (1e-10 + weight).log(), dim=-1)
    
    def icdf(self, value):
        """ Computes the inverse cumulative density evaluated at value
        
        Args:
            value (tensor): an array of shape [batch_size], [1] or [num_cdfs, batch_size]
        
        Returns:
            tensor: the evaluated inverse CDF. 
        """
        return BisectionInverse(self.cdf, min_search=self.min_search, max_search=self.max_search)(value)
                
    
def quantile_to_distribution(predictions, bandwidth_ratio=2.5):
    """ Convert a quantile prediction to a distribution prediction by kernel density estimation (KDE). 
    
    Args:
        predictions (tensor): a batch of quantile predictions, should be an array of shape [batch_size, n_quantiles] or [batch_size, n_quantiles, 2]. 
        bandwidth_ratio (float): the bandwidth of the kernel density estimator (relative to the average distance between quantiles). 
        
    Returns:
        Distribution: a batch of distribution predictions 
    """
    if len(predictions.shape) == 2:
        quantiles = _implicit_quantiles(predictions.shape[1]).view(1, -1).to(predictions.device)
    else:
        quantiles = predictions[:, :, 1]
        predictions = predictions[:, :, 0] 
#     print(quantiles)
    
    # Up weigh the quantiles that are far from other quantiles
    weight = (quantiles[:, 1:] - quantiles[:, :-1]) / 2.
    weight = torch.cat([quantiles[:, :1], weight, 1-quantiles[:, -1:]], dim=-1)
    weight = (weight[:, 1:] + weight[:, :-1]) 
    # If the quantiles are uniform them these weights should be identical 
    # print(weight)
    
    distance_neighbor = ((predictions[:, 1:] - predictions[:, :-1]).abs().mean(dim=1) + 1e-7) * bandwidth_ratio  
    results = DistributionKDE(predictions, distance_neighbor.view(-1, 1).repeat(1, predictions.shape[1]), weight=weight)
    return results


def particle_to_distribution(predictions, bandwidth_ratio=2.5):
    """ Convert a particle prediction to a distribution prediction by kernel density estimation (KDE)
    
    Args:
        predictions (tensor): a batch of quantile predictions, should be an array of shape [batch_size, n_quantiles] or [batch_size, n_quantiles, 2]. 
        bandwidth_ratio (float): the bandwidth of the kernel density estimator (relative to the average distance between quantiles). 
        
    Returns:
        Distribution: a batch of distribution predictions
    """
    return quantile_to_distribution(particle_to_quantile(predictions))


def particle_to_quantile(predictions):
    """ Converts a particle prediction to a quantile prediction. 
    
    Not fully differentiable because of the sorting operation involved. Should be thought of as a permutation function 
    
    Args: 
        predictions (tensor): array [batch_size, n_particles], a batch of particle predictions
        
    Returns: 
        tensor: a batch of quantile predictions
    """
    return torch.sort(predictions, dim=1)[0]


def interval_to_point(predictions):
    """ Converts an interval prediction to a point prediction by taking the middle of the interval.
    
    Args: 
        predictions (tensor): a batch of interval predictions.
        
    Returns:
        tensor: a batch of point predictions 
    """
    return (predictions[:, 0] + predictions[:, 1]) / 2.

    
def ensemble_to_distribution(predictions):
    """ Convert an ensemble prediction to a distribution prediction. 
    
    This is based on the conversion scheme described in https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf
    
    Args:
        predictions: an ensemble prediction.
    
    Returns:
        Distribution: a batch of distribution predictions.
    """ 
    _check_valid(predictions, 'ensemble')
    means = []
    sigmas = []
    for key in predictions:
        pred_type, pred_name = _parse_name(key)
        pred_component = predictions[key]   

        # First convert any type of prediction to a distribution prediction 
        if pred_type == 'quantile':
            pred_component = quantile_to_distribution(pred_component)
        elif pred_type == 'particle':
            pred_component = quantile_to_distribution(particle_to_quantile(pred_component))
        elif pred_type == 'distribution':
            pass
        else:
            print("Warning in ensemble_to_distribution, cannot convert a %s prediciton to a distribution, it has been ignored" % pred_type)
            continue
            
        # Compute the mean and std of the distribution prediction 
        means.append(compute_mean(pred_component))
        sigmas.append(compute_std(pred_component))
            
    assert len(means) > 0, "Error in ensemble_to_distribution, there is no valid prediction in the ensemble to use"
    means = torch.stack(means, dim=0)
    sigmas = torch.stack(sigmas, dim=0)
    
    mean_combined = means.mean(dim=0)
    sigma_combined = ((sigmas.pow(2) + means.pow(2)).mean(dim=0) - mean_combined.pow(2)).pow(0.5)
    return Normal(loc=mean_combined, scale=sigma_combined)



def categorical_to_uset(predictions, threshold=0.95):
    """ Convert a categorical prediction to a set prediction by taking the labels with highest probability until their total probability exceeds threshold
    
    This function is not differentiable. 
    
    Args:
        predictions (tensor): a batch of categorical predictions.
        threshold (float): the minimum probability of the confidence set.
    
    Returns:
        tensor: the set prediction. 
    """
    # A cutoff of shape [batch_size, 1], the extra dimension of 1 is for shape inference
    # We are going to use binary search to find the cut-off threshold for each sample
    cutoff_ub = torch.ones_like(predictions[:, 0:1]) 
    cutoff_lb = torch.zeros_like(predictions[:, 0:1])
    
    # Run 20 iterations, each iteration is guaranteed to reduce the cutoff range by half.
    # Therefore, after 20 iterations, the cut-off should be accurate up to 1e-6 which should be sufficient. 
    for iteration in range(20): 
        cutoff = (cutoff_lb + cutoff_ub) / 2
        
        # The current total probability if we take every label that has greater probability than cutoff
        total_prob = (predictions * (predictions >= cutoff)).sum(dim=1, keepdims=True)  
        
        # If the current total probability is too large, increase the lower bound (i.e higher cut-off)
        cutoff_lb[total_prob > threshold] = cutoff[total_prob > threshold]
        
        # If the current total probability is too small, decrease the upper bound (i.e. lower cut-off)
        cutoff_ub[total_prob <= threshold] = cutoff[total_prob <= threshold] 
    # Return the final result based on current cut-off
    # It is extremely important to use the cutoff_lb instead of ub
    # This ensures that the total probability is at least threshold (rather than at most threshold)
    return (predictions >= cutoff_lb).type(torch.int)


def categorical_to_topk(predictions, k=1):
    """ Convert a categorical prediction to a a top-k prediction by taking the k indices with largest probability
    
    This function is not differentiable. 
    
    Args:
        predictions (tensor): a batch of categorical predictions.
        k (int): the number of predicted labels.
    
    Returns:
        tensor: the topk prediction. 
    """
    topk = torch.topk(predictions, k, dim=1)[1]
    if k == 1:
        topk = topk.flatten()
    return topk


def topk_to_uset(predictions, num_classes=-1):
    """ Convert a topk prediction to a uncertainty set prediction 
    
    Args:
        predictions (tensor): a batch of topk predictions
        num_classes (int): total number of classes. If set to -1, the number of classes will be inferred as one greater than the largest class value in the input tensor.
    
    Returns:
        tensor: the uset prediction 
    """ 
    if num_classes <= 0:
        num_classes = predictions.max() + 1
    
    # Use the [batch_size, k] shape
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(-1)
       
    # Obtain the initial prediction by taking the one-hot of the top1 
    result = F.one_hot(predictions[:, 0], num_classes)
        
    # For all topk that is not top1, set the corresponding element to 1
    for c in range(1, predictions.shape[1]):
        result[torch.arange(len(predictions)), predictions[:, c]] = 1
    
    return result
    