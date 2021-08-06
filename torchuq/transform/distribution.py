import torch
import torch.optim as optim
from copy import deepcopy
from ..models.flow import NafFlow
from .basic import DistributionBase
from .. import _get_prediction_device, _get_prediction_batch_shape


class MonotonicRegressionCalibrator:
    def __init__(self, verbose=False):
        self.flow = None
        self.iflow = None
        self.predictions = []
        self.labels = []
        self.verbose = verbose
        
    def train(self, predictions, labels):
        self.flow = NafFlow(feature_size=30).to(labels.device)
        self.iflow = NafFlow(feature_size=30).to(labels.device)
        flow_optim = optim.Adam(list(self.flow.parameters()) + list(self.iflow.parameters()), lr=1e-3)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(flow_optim, mode='min', patience=3, threshold=1e-5, factor=0.5)
        
        cdf = predictions.cdf(labels.view(1, -1)).flatten()
        # Append 0 and 1 at to avoid 
        cdf = torch.cat([torch.zeros(1, device=cdf.device), cdf, torch.ones(1, device=cdf.device)])
        cdf_target = torch.argsort(torch.argsort(cdf)).type(cdf.dtype) / (len(cdf) - 1)   # Get the ranking of each cdf value among all cdf values
        print(cdf.min(), cdf.max(), cdf_target.min(), cdf_target.max())
        
        for iteration in range(50000): # Iterate at most 100k iterations, but expect to stop early
            flow_optim.zero_grad()

            cdf_pred, _ = self.flow(cdf.view(-1, 1))
            icdf_pred, _ = self.iflow(cdf_target.view(-1, 1))

            loss = (cdf_pred.flatten() - cdf_target).pow(2).mean() + (icdf_pred.flatten() - cdf).pow(2).mean()
            loss.backward()
            flow_optim.step()
            
            lr_scheduler.step(loss)  # Reduce the learning rate 
            if flow_optim.param_groups[0]['lr'] < 1e-5:   # Hitchhike the lr scheduler to terminate if no progress
                break
                
            if self.verbose and iteration % 1000 == 0:
                print("Iteration %d, loss=%.5f, lr=%.5f" % (iteration, loss, flow_optim.param_groups[0]['lr']))
                
    def __call__(self, predictions):
        device = _get_prediction_device(predictions)
        return NafDistribution(deepcopy(self.flow).to(device), deepcopy(self.iflow).to(device), predictions)
    
    
class NafDistribution(DistributionBase):
    def __init__(self, flow, iflow, predictions):
        super(NafDistribution, self).__init__()
        self.flow = flow.eval()
        self.iflow = iflow.eval()
        self.predictions = predictions
        self.device = _get_prediction_device(self.predictions)
        self.batch_shape = self.predictions.batch_shape
        
    def to(self, device):
        if device != self.device:
            self.flow = self.flow.to(device)
            self.iflow = self.iflow.to(device)
            self.device = device 
            
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
        
        adjusted, _ = self.flow(self.predictions.cdf(value).view(-1, 1))
        return adjusted.clamp(min=1e-6, max=1-1e-6).view(out_shape).to(out_device)
    
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

        # Move cdf to the device of test_predictions to avoid device mismatch error
        out_device = cdf.device
        cdf = cdf.to(self.device)
        
        adjusted = self.iflow(cdf.view(-1, 1))[0].view(cdf.shape).clamp(min=1e-6, max=1-1e-6)
        adjusted = self.predictions.icdf(adjusted)
        return adjusted.to(out_device)
    