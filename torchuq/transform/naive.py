import torch
from torch.distributions.normal import Normal 


def _implicit_quantiles(n_quantiles):
    # Induce the implicit quantiles, these quantiles should be equally spaced 
    quantiles = torch.linspace(0, 1, n_quantiles+1)
    quantiles = (quantiles[1:] - quantiles[1] * 0.5)
    return quantiles 


def distribution_to_particle(predictions, n_particles=50):
    


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
    
    def icdf(self, val):
        # Use vectorized bisection to find the inverse 
        with torch.no_grad():
            dummy = self.cdf(val)   # Only a trick to get the right shape of the output tensor 
            current_lb = torch.ones_like(dummy) * self.min_search   # Initialize the upper and lower search interval
            current_ub = torch.ones_like(dummy) * self.max_search 
            for _ in range(50):
                mid_point = (current_lb + current_ub) / 2
                cdfs = self.cdf(mid_point)

                current_ub = current_ub * (cdfs < val) + mid_point * (cdfs >= val)   # If the CDF value is greater than the target value, shrink the upper bound
                current_lb = current_lb * (cdfs >= val) + mid_point * (cdfs < val)   # If the CDF value is smaller than the target value, shrink the upper bound
            icdf = (current_ub + current_lb) / 2
            
        # The exciting part: to generate the right gradients this function has different forward and backward computations
        # This is quite clumsy, but it seems that otherwise mixture of Gaussians do not have a analytical ICDF function, so it's unclear how to do this otherwise
        
#         icdf_copy = Variable(icdf, requires_grad=True)
#         grad_output = self.cdf(icdf_copy)
            
#         manual_grad = grad(outputs=grad_output, inputs=icdf_copy,
#                            create_graph=True, retain_graph=True)[0]
        return icdf
                
    
def quantile_to_distribution(predictions, bandwidth_ratio=2.5):
    """
    Inputs:
        predictions: 
        bandwidth_ratio: float, the bandwidth of the kernel density estimator (relative to the average quantile distance). The default value of 2.5 generates reasonable results. 
        
    Outputs:
        out_predictions: torch Distribution class, the converted predictions
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
    
    distance_neighbor = (predictions[:, 1:] - predictions[:, :-1]).abs().mean(dim=1) * bandwidth_ratio  
    return DistributionKDE(predictions, distance_neighbor.view(-1, 1).repeat(1, predictions.shape[1]), weight=weight)


def particle_to_distribution(predictions, bandwidth_ratio=2.5):
    """
    """
    pass