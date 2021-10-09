from scipy.stats import binom
from .conformal import ConformalCalibrator
from .. import _get_prediction_device, _implicit_quantiles
from .basic import Calibrator
import torch 
from torch.nn import functional as F

    
def _compute_dynamics(old_pred, force_external, lower_bound=-1000, upper_bound=1000, elastic_coeff=0.1):
    """
    Inputs:
        old_pred: an array of shape [batch_size, n_quantiles], the original predictions
        force_external: an array of shape [batch_size, n_quantiles], the force from mis-calibration
        lower_bound: no label should be less than the lower bound
        upper_bound: no label should be greater than the upper bound
    """
    assert force_external.shape[0] == 1 or force_external.shape[0] == old_pred.shape[0], "Shape mismatch, the shape of force_external cannot broadcast to the shape of old_pred"
    assert force_external.shape[1] == old_pred.shape[1], "Shape mismatch" 
    
    if force_external.shape[0] == 1 and old_pred.shape[0] != 1:
        force_external = torch.tile(force_external, [old_pred.shape[1], 1])
        
    # Some hyper-parameters, these should be very reasonable hyper-parameters that should not require tuning unless in exceptional circumstances
    lr_arr = torch.ones_like(force_external) * 0.8   # We define a LR separately for 
    eps = 1e-3   # Trick 1: add eps to anything at risk of division by 0 to improve stability, this could bais the final results but is somewhat inevitable. 
    n_iter = 300
    
    # Pad the predictions with upper and lower bound
    old_pred = F.pad(old_pred, [1, 1])  
    old_pred[:, -1] = upper_bound
    old_pred[:, 0] = lower_bound
    
    # If the original prediction does not satisfy validity constraints, then set first make it satisfies validity with an eps margin
    while (old_pred[:, :-1] > old_pred[:, 1:] - eps).type(torch.int).sum() > 0:
        old_pred[:, :-1] = torch.minimum(old_pred[:, :-1], old_pred[:, 1:] - eps)     # Clip new_pred[i] to be no greater than cur_pred[i+1] - eps
    
    cur_pred = old_pred.clone()
    

    force_prev = torch.zeros_like(force_external)
    prev_pred = cur_pred.clone()
    
    for rep in range(n_iter):
        force_internal = torch.zeros_like(force_external)
        
        # Compute the forces generated from streching or compressing an interval
        cur_dist = (cur_pred[:, 1:] - cur_pred[:, :-1]).abs()
        old_dist = (old_pred[:, 1:] - old_pred[:, :-1]).abs()
        force_internal += elastic_coeff * (cur_dist / (old_dist + eps) - old_dist / (cur_dist + eps))[:, 1:]  # Compute the force from above
        force_internal += elastic_coeff * (old_dist / (cur_dist + eps) - cur_dist / (old_dist + eps))[:, :-1] # Compute the force from below
        
        # Compute the forces generated from deviating from the original prediction
        force_internal += old_pred[:, 1:-1] - cur_pred[:, 1:-1]
        
        force_total = force_external + force_internal
        
        # Trick: force cannot increase by more than 2x+0.1, this significantly improves stability
        # This does not lead to any bias upon convergence 
        force_total = torch.minimum(force_total.abs(), force_prev.abs() * 1.5 + 0.1) * force_total.sign() 
        
        # Trick: gradient descent with too large lr can violate constraints. We reduce the lr for violated quantiles
        for tries in range(100):
            new_pred = cur_pred.clone()
            new_pred[:, 1:-1] = new_pred[:, 1:-1] + force_total * lr_arr 
            
            violated = (new_pred[:, 1:] - new_pred[:, :-1]) < 0
            lr_arr[violated[:, 1:]] *= 0.8
            lr_arr[violated[:, :-1]] *= 0.8
            
            # Try again if the previous learning rate is too big
            if violated.type(torch.int).sum() == 0:
                cur_pred = new_pred 
                break
            if tries == 99:
                print("Warning: failed to enforce validity constraint")
                
        # Break if converged 
        if (cur_pred - prev_pred).abs().mean() < 0.1 * eps:
            break
        lr_arr *= 0.98
        
        # Issue a warning if convergence failed
        if rep == 299 and (cur_pred - prev_pred).abs().mean() > eps:
            print("Warning: failed to reach convergence, final error %.4f" % (new_pred - prev_pred).abs().mean())
            
        # Record the pred and force. Must be the last thing to do before moving to next iteration. 
        prev_pred = cur_pred
        force_prev = force_total 
    return cur_pred[:, 1:-1]


class OnlineQuantileCalibrator(Calibrator):
    def __init__(self, n_quantiles=10, input_type='interval', interpolation='linear', score_func=0, verbose=False):
        self.resolution = n_quantiles
        self.input_type = input_type
        self.interpolation = interpolation 
        
        # Default hyper-parameters
        self.kp = 1.0
        self.ki = torch.cat([torch.linspace(0.09, 0.16, self.resolution // 2), torch.linspace(0.09, 0.16, self.resolution - self.resolution // 2).flip(dims=[0])])
        self.kd =  0.9
        self.elastic_coeff = 0.1878

        self.alpha = 0.2
        self.beta = 0.553
        
        self._init()
        self.target = _implicit_quantiles(n_quantiles)                                   

        
    def _init(self):
        self.conformal = ConformalCalibrator(input_type=self.input_type, interpolation=self.interpolation)
        
        self.num_leq = torch.zeros(self.resolution)
         # These are used to compute the integral and derivative in PID control 
        self.integral_adj = torch.zeros(self.resolution)
        self.prev_adj = torch.zeros(self.resolution)
        self.cur_adj = torch.zeros(self.resolution)
        self.t = 0
        
    
    def to(self, device):
        if not type(device).__name__ == 'device':
            device = _get_prediction_device(device)   # This handles the case that the input is a tensor or a prediction
            
        # Move all assets of this class to device
        self.target.to(device)
        self.num_leq.to(device)
        self.integral_adj.to(device)
        self.prev_adj.to(device)
        self.cur_adj.to(device)
        self.conformal.to(device)
        
    def train(self, predictions, labels):
        """ Same as update but clears all the states 
        """
        self._init()
        self.update(predictions, labels)
        
    def update(self, predictions, labels):
        """
        Inputs:
            predictions: an array [batch_size], a quantile prediction 
            labels: an array [batch_size]
        """
        for t in range(len(predictions)):
            if self.t > 1:   # Conformal calibrator need at least 2 samples to start outputting non-nan probabilities
                adjusted_quantiles = self.__call__(predictions[t:t+1]).flatten()
                self.num_leq = self.num_leq + (labels[t].flatten() < adjusted_quantiles).type(torch.float32)

                # Compute the proportional, integral and derivative terms 
                self.prev_adj = self.cur_adj
                tol = torch.from_numpy(binom.ppf(1-self.alpha, self.t, self.target) - binom.ppf(self.alpha, self.t, self.target)).clamp(min=1.0)
                self.cur_adj = F.relu((self.beta * ((self.num_leq - self.target * self.t).abs() - tol)).exp() - 1) * (self.target * self.t - self.num_leq).sign() 
                self.integral_adj = self.integral_adj + self.cur_adj
                
                initial_pred = self.conformal(predictions).icdf(self.target.view(-1, 1)).flatten()
                self.conformal.update(predictions[t:t+1], self.inverse(initial_pred, adjusted_quantiles, labels[t:t+1]).flatten())
            else:  # Only update conformal calibrator until after t=2
                self.conformal.update(predictions[t:t+1], labels[t:t+1])
            self.t += 1
        
    def __call__(self, predictions):
        """ Take as input a probabilistic prediction, and outputs the quantiles based on the resolution """
        initial_pred = self.conformal(predictions).icdf(self.target.view(-1, 1)).permute(1, 0)  # An array of shape [batch_size, resolution]
        if self.t > 1:
            total_adj = self.cur_adj * self.kp + self.integral_adj * self.ki + (self.cur_adj - self.prev_adj) * self.kd
            return _compute_dynamics(initial_pred, total_adj.view(1, -1), elastic_coeff=self.elastic_coeff)
        else:
            return initial_pred
        
    def inverse(self, original_quantiles, adjusted_quantiles, labels):
        inverted = (original_quantiles[1:] - original_quantiles[:-1]) * ((labels.view(1) - adjusted_quantiles[:-1]) / (adjusted_quantiles[1:] - adjusted_quantiles[:-1] + 1e-10)).clamp(min=0.0, max=1.0)
        invert_overflow = (labels - adjusted_quantiles[-1]).clamp(min=0.0)
        invert_underflow = (labels - adjusted_quantiles[0]).clamp(max=0.0)
        return (inverted.sum() + original_quantiles[0] + invert_overflow + invert_underflow).view(1, 1)

    
    
# class OnlineRecalibrator:
#     def __init__(self, device, momentum=0.2, alpha=0.04, beta=1.0):
#         self.resolution = 99
#         self.device = device
#         self.num_leq = torch.zeros(self.resolution, device=device)
#         self.cur_adj = torch.zeros(self.resolution, device=device)
#         self.target = torch.linspace(0, 1, self.resolution+2, device=device)[1:-1]
#         self.momentum = momentum
#         self.alpha = alpha
#         self.beta = beta
#         self.t = 0
        
#     def update(self, predictions, labels):
#         adjusted_quantiles = self.__call__(predictions)
#         self.num_leq += (labels.flatten() < adjusted_quantiles).type(torch.float32)
#         self.t += 1
        
#         adjustment = self.compute_adjustment()
#         self.cur_adj = adjustment * self.momentum + self.cur_adj * (1 - self.momentum)
        
    
#     def __call__(self, predictions):
#         """ Take as input a probabilistic prediction, and outputs the quantiles based on the resolution """
#         adjustment = self.compute_adjustment()
#         adjusted_quantiles = predictions.icdf(self.target.view(-1, 1)).flatten() + adjustment
#         return adjusted_quantiles 
    
#     def compute_adjustment(self):
#         if self.t != 0:
#             tol = torch.from_numpy(binom.ppf(1-self.alpha, self.t, self.target.cpu()) - binom.ppf(self.alpha, self.t, self.target.cpu())).to(self.device).clamp(min=1.0)
#             adjustment = F.relu((self.beta * ((self.num_leq - self.target * self.t).abs() - tol)).exp() - 1) * (self.target * self.t - self.num_leq).sign() + self.cur_adj
#             return adjustment 
#         else:
#             return torch.zeros(self.resolution, device=self.device)

#     def inverse(self, predictions, labels):
#         adjustment = self.compute_adjustment()
#         original_quantiles = predictions.icdf(self.target.view(-1, 1)).flatten()
#         # print(original_quantiles)
#         adjusted_quantiles = original_quantiles + adjustment 
#         # print(adjusted_quantiles)
#         # print(labels.shape)
#         inverted = (original_quantiles[1:] - original_quantiles[:-1]) * ((labels.view(1) - adjusted_quantiles[:-1]) / (adjusted_quantiles[1:] - adjusted_quantiles[:-1])).clamp(min=0.0, max=1.0)
#         invert_overflow = (labels - adjusted_quantiles[-1]).clamp(min=0.0)
#         invert_underflow = (labels - adjusted_quantiles[0]).clamp(max=0.0)
#         # print(inverted)
#         return (inverted.sum() + original_quantiles[0] + invert_overflow + invert_underflow).view(1, 1)

    