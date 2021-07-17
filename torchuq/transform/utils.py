import copy 
import torch 


class PerformanceRecord:
    def __init__(self):
        self.records = {}

    def add_scalar(self, name, value, iteration=0):
        if name not in self.records:
            self.records[name] = {}
        if iteration not in self.records[name]:
            self.records[name][iteration] = [1, value]
        else:
            self.records[name][iteration][0] += 1
            self.records[name][iteration][1] += value
        
    def get_scalar(self, name):
        if not name in self.records:
            print("%s not found. Available records include " + " ".join([self.records.keys()]))
            return None
        
        iterations = list(self.records[name].keys())
        values = [self.records[name][iteration][1] / self.records[name][iteration][0] for iteration in iterations]
        return iterations, values
    
    
class BisectionInverse:
    """ Given a monotonic function forward_func, computes its inverse with bisection. 
    This function is differentiable if the forward_func is differentiable and has invertiable gradients
    """
    def __init__(self, forward_func, min_search=-1.0, max_search=1.0):
        """
        Inputs: 
        forward_func: a function that take as input a tensor [...] and outputs a tensor [...] with the same size as the input. 
            It must be a monotonic function. If requires_grad is true for the input, then forward_func should also be differentiable and have non-zero gradients (otherwise its inverse if not differentiable)
        min_search: (float) the minimum value that bisection algorithm will search
        max_search: (float) the maximum value that the bisection algorithm will search 
        """
        self.forward_func = forward_func
        self.min_search = min_search
        self.max_search = max_search
        self.warning = False
        
    def __call__(self, val):
        return self.forward(val)
    
    def forward(self, val):
        """
        Compute the inverse of the forward_func. This function is differentiable if the forward_func has non-zero gradients 
        """
        with torch.no_grad():
            dummy = self.forward_func(val)   # Only a trick to get the right shape of the output tensor 
            current_lb = torch.ones_like(dummy) * self.min_search   # Initialize the upper and lower search interval
            current_ub = torch.ones_like(dummy) * self.max_search 
            for _ in range(50):
                mid_point = (current_lb + current_ub) / 2
                cdfs = self.forward_func(mid_point)

                current_ub = current_ub * (cdfs < val) + mid_point * (cdfs >= val)   # If the CDF value is greater than the target value, shrink the upper bound
                current_lb = current_lb * (cdfs >= val) + mid_point * (cdfs < val)   # If the CDF value is smaller than the target value, shrink the upper bound
            result = (current_ub + current_lb) / 2
            
        # Generate the correct backward gradient
        if val.requires_grad and torch.is_grad_enabled():   # Only do this if gradients are enabled
            
            # Compute the forward gradient
            result_ = copy.deepcopy(result)
            result_.requires_grad = True
            out = self.forward_func(result_)
            grad = torch.autograd.grad(out, result_, only_inputs=True, grad_outputs=torch.ones_like(out))[0]
            
            # The backward gradient is the inverse of the forward gradient
            grad = 1. / grad.data   
            
            # If the original function is not invertible (up to numerical precisions), then set nan values to 0 and issue a warning
            isnan = torch.isnan(grad)
            if isnan.sum() != 0:
                if not self.warning:
                    print("Warning: non-differentiable function encountered in BisectionInverse. The gradient is inf. It has been replaced with 0. This might happen multiple times, but only one warning will be issued.")
                    self.warning = True
                grad[isnan] = 0
            
            return (result - grad * val).detach() + grad * val   # A trick to coax pytorch to generate the gradient we want
        else:
            return result