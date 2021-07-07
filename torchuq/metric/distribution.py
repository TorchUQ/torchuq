import seaborn as sns
import numpy as np
import torch
from matplotlib import pyplot as plt 
import numpy as np
from scipy.stats import binom
from torch.nn import functional as F
from .utils import metric_plot_colors as mcolors
from .. import _get_prediction_device, _implicit_quantiles

    
def compute_all(predictions, labels, resolution=500):
    results_all = {}
    results_all['crps'] = compute_crps(predictions, labels, resolution)
    results_all['std'] = compute_std(predictions, resoluion)
    results_all['nll'] = compute_nll(predictions, labels)
    return results_all 


def compute_crps(predictions, labels, resolution=500):  
    """
    Compute the CRPS score 
    
    Input:
        predictions: a batch of distribution predictions
        labels: array [batch_size] of labels
        resolution: the number of discretization bins, higher resolution increases estimation accuracy but also requires more memory/compute
        
    Output:
        crps: array [batch_size], the CRPS score 
    """
    intervals = torch.linspace(0, 1, resolution+2, device=labels.device)[1:-1].view(-1, 1)
    weights = (intervals ** 2)[1:] - (intervals ** 2)[:-1]   # This is a trick to compute the Lesbegue integral of CDF^2 (when we only have access to inverse CDF)
    weights = torch.cat([weights[:1], (weights[1:] + weights[:-1])/2, weights[-1:]], dim=0)   # Handle the boundary 

    icdf = predictions.icdf(intervals)  
    partition = (labels.view(1, -1) > icdf).type(torch.float32)  # Partition the intervals based on whether it's greater or less than the labels. This biases the gradient, so use with caution
    part_under = (labels - icdf) * weights * partition   # Compute the Lesbegue integral for the \int_{x, x < y} (F[x] - I(y < x))^2 dx
    part_over = (icdf - labels) * weights.flip(dims=[0]) * (1 - partition)  # Compute the Lesbegue integral for the \int_{x, x > y} (F[x] - I(y < x))^2 dx
    crps = (part_under + part_over).abs().sum(dim=0)   # The abs is not necessary, but placed here just in case numerical errors cause CRPS to be below zero. 
    return crps
  
    
def compute_nll(predictins, labels):
    return -predictions.log_prob(labels)


def compute_std(predictions, resolution=500):
    """ 
    Compute the standard deviation of the predictions
    
    Inputs:
        predictions: a batch of distribution predictions
        resolution: the number of discretization bins, higher resolution increases estimation accuracy but also requires more memory/compute
    
    Outputs:
        std: array [batch_size], the standard deviation
    """
    quantiles = _implicit_quantiles(resolution) 
    samples = predictions.icdf(quantiles.to(_get_prediction_device(predictions)).view(-1, 1))  
    return (samples - samples.mean(dim=0, keepdim=True)).pow(2).mean(dim=0).pow(0.5)


def compute_ece(predictions, labels, debiased=False):
    """
    Compute the ECE score as in https://arxiv.org/abs/1807.00263 
    
    Note that this function has biased gradient because of the non-differentiable nature of sorting. 
    
    Input:
        predictions: a batch of distribution predictions
        labels: array [batch_size] of labels
        debiased: if debiased=True then the finite sample bias is removed. If the labels are truely drawn from the predictions, the this function will in expectation return 0. 
        
    Output:
        crps: array [], the ECE score
    """
    cdfs = predictions.cdf(labels).flatten() 
    ranking = torch.argsort(cdfs)
    cdfs = cdfs[ranking]  # Manually sort so that at least this part is always differentiable 
    ece = (cdfs - _implicit_quantiles(len(labels)).to(labels.device)).abs().mean()

    if debiased:
        n_sample = 1000
        comparison_samples = torch.sort(torch.rand(n_sample, len(labels)), dim=1)[0] - _implicit_quantiles(len(labels)).view(1, -1) # The empirical ECE score if the predictions are perfect 
        baseline_ece = comparison_samples.abs().mean()
        ece = ece - baseline_ece
    return ece


def compute_mean(predictions, resolution=500):
    """ 
    Compute the mean of the predictions
    
    Inputs:
        predictions: a batch of distribution predictions
        resolution: the number of discretization bins, higher resolution increases estimation accuracy but also requires more memory/compute
    
    Outputs:
        std: array [batch_size], the mean of the predictions
    """
    quantiles = _implicit_quantiles(resolution) 
    samples = predictions.icdf(quantiles.to(_get_prediction_device(predictions)).view(-1, 1))  
    return samples.mean(dim=0)
    
    
def plot_reliability_diagram(predictions, labels, ax=None):
    """
    Plot the reliability diagram https://arxiv.org/abs/1807.00263 
    
    Input:
        predictions: required Distribution instance, a batch of distribution predictions
        labels: required array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        
    Output:
        ax: matplotlib.axes.Axes, the ax on which the plot is made
    """
    with torch.no_grad():
        cdfs = predictions.cdf(labels).flatten() 
        cdfs, _ = torch.sort(cdfs) 
        
        if ax is None:
            plt.figure(figsize=(5, 5))
            ax = plt.gca() 
            
        # Compute confidence bound
        min_vals = binom.ppf(0.005, len(cdfs), np.linspace(0, 1, 102)) / len(cdfs)
        max_vals = binom.ppf(0.995, len(cdfs), np.linspace(0, 1, 102)) / len(cdfs)
        ax.fill_between(np.linspace(0, 1, 102), min_vals, max_vals, alpha=0.1, color='C1')

        ax.plot(cdfs.cpu().numpy(), np.linspace(0, 1, len(cdfs)), c='C0')
        ax.plot([0,1], [0,1], c='C1', linestyle=':')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('quantiles', fontsize=14)
        ax.set_ylabel('proportion samples below quantile', fontsize=14)
        return ax
        
        
        
        
def plot_density(predictions, labels, max_count=100, ax=None, resolution=100, smooth_bw=0):
    """ 
    Plot the PDF of the predictions and the labels. For aesthetics the PDFs are reflected along y axis to make a symmetric violin shaped plot
    
    Input:
        predictions: required Distribution instance, a batch of distribution predictions
        labels: required array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        max_count: optional int, the maximum number of PDFs to plot
        smooth_bw: optional int, smooth the PDF with a uniform kernel whose bandwidth is smooth_bw / resolution
    Output:
        ax: matplotlib.axes.Axes, the ax on which the plot is made
    """
    values = predictions.icdf(torch.linspace(0, 1, resolution+2)[1:-1].view(-1, 1)).cpu()  # [n_quantiles, batch_shape]
    centers = (values[1:] + values[:-1]) * 0.5  
    interval = values[1:] - values[:-1] 
    
    # Smooth the interval values, this is to ensure that non-smooth CDFs can be plotted nicely
    if smooth_bw != 0:
        interval = F.conv1d(torch.cat([interval[:smooth_bw], interval, interval[-smooth_bw:]], dim=0).permute(1, 0).unsqueeze(1), 
                            weight=torch.ones(1, 1, smooth_bw*2+1, device=interval.device)).squeeze(1).permute(1, 0)
    density = 1. / resolution / interval   # Compute the empirical density
#     np.set_printoptions(suppress=True)
#     print(values[:, 0])
#     print(density[:, 0].numpy())
#     print(values.max(), values.min())
    vpstats = [{'coords': centers[:, i].numpy(), 'vals': density[:, i].numpy(),
                'mean': values[:, i].mean(), 'median': values[resolution // 2, i], 
                'min': values[0, i], 'max': values[-1, i]} for i in range(density.shape[1]) if i < max_count]
    if ax is None:
        optimal_width = len(vpstats) / 4
        if optimal_width < 4:
            optimal_width = 4
        plt.figure(figsize=(optimal_width, 4))
        ax = plt.gca() 
        
    ax.violin(vpstats=vpstats, positions=range(len(vpstats)), showextrema=False, widths=0.7)
    ax.scatter(range(len(vpstats)), labels[:len(vpstats)].cpu().numpy(), marker='x')
    ax.set_ylabel('label value', fontsize=14)
    ax.set_xlabel('sample index', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    return ax


def plot_cdf(predictions, labels, ax=None, max_count=30, resolution=200):
    """ 
    Plot the CDF functions
    
    Input:
        predictions: required Distribution instance, a batch of distribution predictions
        labels: required array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        
    Output:
        ax: matplotlib.axes.Axes, the ax on which the plot is made
    """
    n_plot = len(labels)
    if n_plot > max_count:
        n_plot = max_count
        
    margin = (labels[:n_plot].max() - labels[:n_plot].min()) * 0.2   # Get the range of the plot
    y_range = torch.linspace(labels[:n_plot].min() - margin, labels[:n_plot].max() + margin, resolution, device=labels.device)  # Get the values on which to evaluate the CDF
    # Evaluate the CDF
    with torch.no_grad():
        cdfs = predictions.cdf(y_range.view(-1, 1))[:, :n_plot].cpu().numpy()
        label_cdfs = predictions.cdf(labels)[:n_plot].cpu().numpy()
    
    if ax is None: 
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 
            

    palette = np.array(sns.color_palette("husl", n_plot))
    for i in range(n_plot):
        ax.plot(y_range.cpu().numpy(), cdfs[:, i], c=palette[i], alpha=0.5)

    ax.scatter(labels[:n_plot].cpu().numpy(), label_cdfs, color=palette[np.arange(n_plot)], marker='x', zorder=3)
    ax.set_ylabel('cdf', fontsize=14)
    ax.set_xlabel('value', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    return ax
    
    
def plot_icdf(predictions, labels, ax=None, max_count=30, resolution=200):
    """
    Plot the inverse CDF functions
    
    Input:
        predictions: required Distribution instance, a batch of distribution predictions
        labels: required array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 

    Output:
        ax: matplotlib.axes.Axes, the ax on which the plot is made
    """
    n_plot = len(labels)
    if n_plot > max_count:
        n_plot = max_count
        
    margin = (labels[:n_plot].max() - labels[:n_plot].min()) * 0.2
    c_range = torch.linspace(0.01, 0.99, resolution, device=labels.device)
    with torch.no_grad():
        values = predictions.icdf(c_range.view(-1, 1))[:, :n_plot].cpu().numpy()
        label_cdfs = predictions.cdf(labels)[:n_plot].cpu().numpy()
    
    if ax is None: 
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 
        
    palette = np.array(sns.color_palette("husl", n_plot))
    for i in range(n_plot):
        ax.plot(values[:, i], c_range.cpu().numpy(), c=palette[i], alpha=0.5)
    
    ax.scatter(labels[:n_plot].cpu().numpy(), label_cdfs, color=palette[np.arange(n_plot)], marker='x', alpha=0.5, zorder=2)
    ax.set_ylabel('cdf', fontsize=14)
    ax.set_xlabel('value', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    return ax