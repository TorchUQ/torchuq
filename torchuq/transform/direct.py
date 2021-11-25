import torch
from torch.distributions.normal import Normal 
from .utils import BisectionInverse
from .. import _implicit_quantiles, _check_valid, _parse_name
from ..evaluate.distribution import compute_mean, compute_std, plot_density_sequence


def distribution_to_particle(predictions, n_particles=50):
    return predictions.sample((n_particles,)).permute(1, 0)


def distribution_to_quantile(predictions, quantiles=None, n_quantiles=None):
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
    """
    Convert a probability distribution to a point prediction 
    Input:
        functional: can be mean or median
    """
    assert functional == 'mean' or functional == 'median'
    if functional == 'mean':
        return predictions.icdf(_implicit_quantiles(100).view(-1, 1)).mean(dim=0)
    else:
        return predictions.icdf(0.5)


def distribution_to_interval(predictions, confidence=0.95):
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
    def __init__(self, loc, scale, weight=None):
        """ 
        Define a distribution given by a mixture of Gaussian 
        Even though pytorch has native support for mixture distribution, it does not have an icdf method which is crucial and difficult to implement
        This implementation includes an icdf function 
        
        Input:
            loc: array [batch_size, n_components], the set of centers of the Gaussian distributions
            scale: array [batch_size, n_components], the set of stddev of the Gaussian distributions
            weight: array [batch_size, n_components], the weight of each mixture component. If set to None then all mixture components have the same weight
        """
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
        """
        Inputs:
            val: an array of shape [batch_size], [1] or [num_cdfs, batch_size]
        """
        cdf = self.normals.cdf(value.unsqueeze(-1))  # This should have shape [..., batch_size, n_components]
        
        weight = self.weight
        for i in range(len(value.shape) - 1):
            weight = weight.unsqueeze(0)   # Prepand enough dimensions so that size would match
        return (cdf * weight).sum(dim=-1)  
    
    def log_prob(self, value):
        log_prob = self.normals.log_prob(value.unsqueeze(-1))
        
        weight = self.weight
        for i in range(len(value.shape) - 1):
            weight = weight.unsqueeze(0)   # Prepand enough dimensions so that size would match
            
        return torch.logsumexp(log_prob + (1e-10 + weight).log(), dim=-1)
    
    def icdf(self, value):
        return BisectionInverse(self.cdf, min_search=self.min_search, max_search=self.max_search)(value)
                
    
def quantile_to_distribution(predictions, bandwidth_ratio=2.5):
    """
    Inputs:
        predictions: array [batch_size, n_quantiles] or [batch_size, n_quantiles, 2], a batch of quantile predictions
        bandwidth_ratio: float, the bandwidth of the kernel density estimator (relative to the average distance between quantiles). 
        
    Outputs:
        results: torch Distribution class, the converted predictions
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
    """
    """
    return quantile_to_distribution(particle_to_quantile(predictions))


def particle_to_quantile(predictions):
    """
    Maintains the ordering of the quantiles (e.g. so the 10% quantile is always smaller than the 20% quantiles)
    Not fully differentiable because of the sorting operation involved. Should be thought of as a permutation function 
    
    Input: 
        predictions: array [batch_size, n_particles], a batch of particle predictions
    """
    return torch.sort(predictions, dim=1)[0]


def ensemble_to_distribution(predictions):
    """ Convert an ensemble prediction to a single distribution. This is based on the conversion scheme in https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf
    
    Input:
        predictions: an ensemble prediction 
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



def categorical_to_set(predictions, threshold=0.95):
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