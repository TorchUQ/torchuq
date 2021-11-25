import seaborn as sns
import numpy as np
import torch
from matplotlib import pyplot as plt 
import numpy as np
from scipy.stats import binom
from torch.nn import functional as F
from .utils import metric_plot_colors as mcolors
from .. import _get_prediction_device, _implicit_quantiles, _get_prediction_batch_shape
from .utils import _compute_reduction 

    
def compute_all_metrics(predictions, labels, resolution=500):
    results_all = {}
    results_all['crps'] = compute_crps(predictions, labels, resolution)
    results_all['std'] = compute_std(predictions, resolution)
    results_all['nll'] = compute_nll(predictions, labels)
    results_all['ece'] = compute_ece(predictions, labels, debiased=True)
    return results_all 


def compute_crps(predictions, labels, reduction='mean', resolution=500):  
    """Compute the CRPS score.

    The CRPS score is a proper score that measures the quality of a prediction.
    
    Args:
        predictions (distribution): a batch of distribution predictions.
        labels (tensor): array [batch_size] of labels.
        reduction (str): the method to aggregate the results across the batch. Can be 'none', 'mean', 'sum', 'median', 'min', or 'max'. 
        resolution (int): the number of discretization bins, higher resolution increases estimation accuracy but also requires more memory/compute
        
    Returns:
        tensor: the CRPS score, an array with shape [batch_size] or shape [] depending on the reduction.
    """
    intervals = torch.linspace(0, 1, resolution+2, device=labels.device)[1:-1].view(-1, 1)
    weights = (intervals ** 2)[1:] - (intervals ** 2)[:-1]   # This is a trick to compute the Lesbegue integral of CDF^2 (when we only have access to inverse CDF)
    weights = torch.cat([weights[:1], (weights[1:] + weights[:-1])/2, weights[-1:]], dim=0)   # Handle the boundary 

    icdf = predictions.icdf(intervals)  
    partition = (labels.view(1, -1) > icdf).type(torch.float32)  # Partition the intervals based on whether it's greater or less than the labels. This biases the gradient, so use with caution
    part_under = (labels - icdf) * weights * partition   # Compute the Lesbegue integral for the \int_{x, x < y} (F[x] - I(y < x))^2 dx
    part_over = (icdf - labels) * weights.flip(dims=[0]) * (1 - partition)  # Compute the Lesbegue integral for the \int_{x, x > y} (F[x] - I(y < x))^2 dx
    crps = (part_under + part_over).abs().sum(dim=0)   # The abs is not necessary, but placed here just in case numerical errors cause CRPS to be below zero. 
    return _compute_reduction(crps, reduction)
  
    
def compute_nll(predictions, labels, reduction='mean'):
    return _compute_reduction(-predictions.log_prob(labels), reduction)


def compute_std(predictions, reduction='mean', resolution=500):
    """Compute the standard deviation of the predictions
    
    Args:
        predictions (distribution): a batch of distribution predictions
        reduction (str): the method to aggregate the results across the batch. Can be 'none', 'mean', 'sum', 'median', 'min', or 'max'. 
        resolution (int): the number of discretization bins, higher resolution increases estimation accuracy but also requires more memory/compute
    
    Returns:
        tensor: the standard deviation, array [batch_size] or array [], the standard deviation
    """
    quantiles = _implicit_quantiles(resolution) 
    samples = predictions.icdf(quantiles.to(_get_prediction_device(predictions)).view(-1, 1))  
    std = (samples - samples.mean(dim=0, keepdim=True)).pow(2).mean(dim=0).pow(0.5)
    return _compute_reduction(std, reduction)


def compute_mean(predictions, reduction='mean', resolution=500):
    """Compute the mean of the predictions
    
    Args:
        predictions (distribution): a batch of distribution predictions
        reduction (str): the method to aggregate the results across the batch. Can be 'none', 'mean', 'sum', 'median', 'min', or 'max'. 
        resolution (int): the number of discretization bins, higher resolution increases estimation accuracy but also requires more memory/compute
    
    Returns:
        tensor: the computed mean, array [batch_size] or array [] depending on the reduction. 
    """
    quantiles = _implicit_quantiles(resolution) 
    samples = predictions.icdf(quantiles.to(_get_prediction_device(predictions)).view(-1, 1))  
    return _compute_reduction(samples.mean(dim=0), reduction)
    
    
def compute_mean_std(predictions, reduction='mean', resolution=500):
    """Same as compute_mean and compute_std, but combines into one function for better efficiency
    
    Args:
        predictions (distribution): a batch of distribution predictions
        reduction (str): the method to aggregate the results across the batch. Can be 'none', 'mean', 'sum', 'median', 'min', or 'max'. 
        resolution: the number of discretization bins, higher resolution increases estimation accuracy but also requires more memory/compute
    
    Returns:
        tensor: the computed mean, array [batch_size] or array [] depending on the reduction.
        tensor: the computed standard deviation, array [batch_size] or array [] depending on the reduction. 
    """
    quantiles = _implicit_quantiles(resolution) 
    samples = predictions.icdf(quantiles.to(_get_prediction_device(predictions)).view(-1, 1))  
    mean = samples.mean(dim=0)
    std = (samples - samples.mean(dim=0, keepdim=True)).pow(2).mean(dim=0).pow(0.5)
    return _compute_reduction(mean, reduction), _compute_reduction(std, reduction) 

    
_baseline_ece_cache = {}   # Cache for the ECE baselines. Because bootstrap is expensive, use cache to store the results and not recompute


def compute_ece(predictions, labels, debiased=False):
    """Compute the (weighted) ECE score as in https://arxiv.org/abs/1807.00263
    
    Note that this function has biased gradient because of the non-differentiable nature of sorting. 
    
    Args:
        predictions (distribution): a batch of distribution predictions
        labels (tensor): array [batch_size] of labels
        debiased (bool): if debiased=True then the finite sample bias is removed. If the labels are truely drawn from the predictions, the this function will in expectation return 0. 
        
    Returns:
        tensor: the ECE score, which is an scalar array (array of shape []). 
    """
    cdfs = predictions.cdf(labels).flatten() 
    ranking = torch.argsort(cdfs)
    cdfs = cdfs[ranking]  # Manually sort so that at least this part is always differentiable 
    ece = (cdfs - _implicit_quantiles(len(labels)).to(labels.device)).abs().mean()

    if debiased:
        if int(len(labels)) not in _baseline_ece_cache:   # Check if the baseline value is already in the cache. In the future can also store this permanently
            n_sample = 1000
            comparison_samples = torch.sort(torch.rand(n_sample, len(labels)), dim=1)[0] - _implicit_quantiles(len(labels)).view(1, -1) # The empirical ECE score if the predictions are perfect 
            _baseline_ece_cache[int(len(labels))] = comparison_samples.abs().mean()
            
        baseline_ece = _baseline_ece_cache[int(len(labels))].to(ece.device)
        ece = ece - baseline_ece
    return ece



    
def plot_reliability_diagram(predictions, labels, ax=None):
    """Plot the reliability diagram https://arxiv.org/abs/1807.00263
    
    Args:
        predictions (distribution): a batch of distribution predictions
        labels (tensor): the labels, array [batch_size]
        ax (axes): optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        
    Returns:
        axes: the ax on which the plot is made, it is an instance of matplotlib.axes.Axes. 
    """
    with torch.no_grad():
        device = _get_prediction_device(predictions)
        cdfs = predictions.cdf(labels.to(device)).flatten() 
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
        
        
        
        
def plot_density_sequence(predictions, labels=None, max_count=100, ax=None, resolution=100, smooth_bw=0):
    """Plot the PDF of the predictions and the labels. For aesthetics the PDFs are reflected along y axis to make a symmetric violin shaped plot
    
    Args:
        predictions (distribution): a batch of distribution predictions
        labels (tensor): the labels, if None then the labels are not plotted
        ax (axes): the axes to plot the figure on, if None automatically creates a figure with recommended size 
        max_count (int): the maximum number of PDFs to plot
        resolution (int): the number of points to compute the density. Higher resolution leads to a more accurate plot, but also requires more computation. 
        smooth_bw (int): smooth the PDF with a uniform kernel whose bandwidth is smooth_bw / resolution

    Returns:
        axes: the ax on which the plot is made, it is an instance of matplotlib.axes.Axes. 
    """
    device = _get_prediction_device(predictions)
    values = predictions.icdf(torch.linspace(0, 1, resolution+2, device=device)[1:-1].view(-1, 1)).detach().cpu()  # [n_quantiles, batch_shape]
    centers = (values[1:] + values[:-1]) * 0.5  
    interval = values[1:] - values[:-1] 
    
    # Smooth the interval values, this is to ensure that non-smooth CDFs can be plotted nicely
    if smooth_bw != 0:
        interval = F.conv1d(torch.cat([interval[:smooth_bw], interval, interval[-smooth_bw:]], dim=0).permute(1, 0).unsqueeze(1), 
                            weight=torch.ones(1, 1, smooth_bw*2+1, device=interval.device)).squeeze(1).permute(1, 0)
    density = 1. / resolution / interval   # Compute the empirical density

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
    if labels is not None:
        ax.scatter(range(len(vpstats)), labels[:len(vpstats)].detach().cpu().numpy(), marker='x')
    ax.set_ylabel('label value', fontsize=14)
    ax.set_xlabel('sample index', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    return ax


def _get_predictions_range(predictions, labels, n_plot, resolution):
    """
    Get a reasonable range to plot the CDFs for better visual appeal. 
    If labels is not None, then use labels to decide the range, otherwise use predictions to decide the range
    """
    # 
    with torch.no_grad():
        if labels is not None: 
            # If labels are available, use labels to decide the range to plot the CDFs
            margin = (labels[:n_plot].max() - labels[:n_plot].min()) * 0.2   # Get the range of the plot
            min_eval = (labels[:n_plot].min() - margin).cpu().item()
            max_eval = (labels[:n_plot].max() + margin).cpu().item()
        else:   
            device = _get_prediction_device(predictions)
            c_range = torch.linspace(0.01, 0.99, resolution, device=device) 
            values = torch.sort(predictions.icdf(c_range.view(-1, 1))[:, :n_plot].flatten())[0]
            values = values[~torch.isnan(values)]
            # Choose the bottom 2% as the min value and top 2% as the max value
            values = values[len(values) // 50: len(values) - (len(values) // 50) - 1]
            margin = values.max() - values.min()
            min_eval = (values.min() - margin).cpu().item()
            max_eval = (values.max() + margin).cpu().item()
        return min_eval, max_eval


def plot_cdf_sequence(predictions, labels=None, ax=None, max_count=20, resolution=200):
    """Plot the CDF functions
    
    Args:
        predictions (distribution): a batch of distribution predictions
        labels (tensor): the labels, an array [batch_size], if not provided then no label will be plotted. 
        ax (axes): the axes to plot the figure on, if None automatically creates a figure with recommended size 
        max_count (int): the maximum number of CDFs to plot
        resolution (int): the number of points to compute the density. Higher resolution leads to a more accurate plot, but also requires more computation. 
        
    Returns:
        axes: the ax on which the plot is made
    """
    # Figure out the maximum number of CDFs to plot
    n_plot = _get_prediction_batch_shape(predictions)
    if n_plot > max_count:
        n_plot = max_count
    
    device = _get_prediction_device(predictions)
    
    with torch.no_grad():
        min_eval, max_eval = _get_predictions_range(predictions, labels, n_plot, resolution)
        y_range = torch.linspace(min_eval, max_eval, resolution, device=device)  # Get the values on which to evaluate the CDF
        
        # Evaluate the CDF
        cdfs = predictions.cdf(y_range.view(-1, 1))[:, :n_plot].cpu().numpy()
        if labels is not None:
            label_cdfs = predictions.cdf(labels.to(device))[:n_plot].cpu().numpy()
    
    if ax is None:
        optimal_width = n_plot 
        if optimal_width < 4:
            optimal_width = 4
        plt.figure(figsize=(optimal_width, 4))
        ax = plt.gca() 
            
    
    x_shift = np.array([(max_eval - min_eval)*i for i in range(n_plot)])   # Shift each plotted CDF by the right amount 
    x_centers = np.array([(max_eval - min_eval)*(i+0.5)+min_eval for i in range(n_plot)]) # The center of the shifted CDF. Used for labeling the axis
    for i in range(n_plot):
        ax.plot(y_range.cpu().numpy() + x_shift[i], cdfs[:, i], c='C0', alpha=0.5)
        ax.axvline((max_eval - min_eval)*(i+0.5)+min_eval, c='C1', linestyle=':')
    
    ax.set_xticks(x_centers)
    ax.set_xticklabels(range(n_plot))
    
    # Plot the labels
    if labels is not None:
        ax.scatter(labels[:n_plot].cpu().numpy() + x_shift, label_cdfs, color='C2', marker='x', zorder=3)
        
    ax.set_ylabel('CDF', fontsize=14)
    ax.set_xlabel('sample index', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    return ax


def plot_cdf(predictions, labels=None, ax=None, max_count=30, resolution=200):
    """Plot the CDF functions
    
    Args:
        predictions (distribution): a batch of distribution predictions
        labels (tensor): the labels
        ax (axes): the axes to plot the figure on, if None automatically creates a figure with recommended size 
        max_count (int): the maximum number of CDFs to plot
        resolution (int): the number of points to compute the density. Higher resolution leads to a more accurate plot, but also requires more computation. 
        
    Returns:
        axes: the ax on which the plot is made
    """
    n_plot = _get_prediction_batch_shape(predictions)
    if n_plot > max_count:
        n_plot = max_count
    
    device = _get_prediction_device(predictions)

    # Evaluate the CDF
    with torch.no_grad():
        min_eval, max_eval = _get_predictions_range(predictions, labels, n_plot, resolution)

        y_range = torch.linspace(min_eval, max_eval, resolution, device=device)  # Get the values on which to evaluate the CDF
        cdfs = predictions.cdf(y_range.view(-1, 1))[:, :n_plot].cpu().numpy()
        if labels is not None:
            label_cdfs = predictions.cdf(labels.to(device))[:n_plot].cpu().numpy()
    
    if ax is None: 
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 
            

    palette = np.array(sns.color_palette("husl", n_plot))
    for i in range(n_plot):
        ax.plot(y_range.cpu().numpy(), cdfs[:, i], c=palette[i], alpha=0.5)
    
    if labels is not None:
        ax.scatter(labels[:n_plot].cpu().numpy(), label_cdfs, color=palette[np.arange(n_plot)], marker='x', zorder=3)
    ax.set_ylabel('cdf', fontsize=14)
    ax.set_xlabel('value', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    return ax
    
    
def plot_icdf(predictions, labels=None, ax=None, max_count=30, resolution=200):
    """Plot the inverse CDF functions
    
    Args:
        predictions (distribution): a batch of distribution predictions
        labels (tensor): the labels, an array [batch_size]
        ax (axes): optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        max_count (int): the maximum number of CDFs to plot
        resolution (int): the number of points to compute the density. Higher resolution leads to a more accurate plot, but also requires more computation. 
        
    Returns:
        axes: the ax on which the plot is made
    """
    n_plot = _get_prediction_batch_shape(predictions)
    if n_plot > max_count:
        n_plot = max_count
    
    # In case labels is not on the same device as predictions, move them to the same device
    device = _get_prediction_device(predictions)

    c_range = torch.linspace(0.01, 0.99, resolution, device=device)
    with torch.no_grad():
        values = predictions.icdf(c_range.view(-1, 1))[:, :n_plot].cpu().numpy()
        if labels is not None:
            labels = labels.to(device)
            label_cdfs = predictions.cdf(labels)[:n_plot].cpu().numpy()
    
    if ax is None: 
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 
        
    palette = np.array(sns.color_palette("husl", n_plot))
    for i in range(n_plot):
        ax.plot(values[:, i], c_range.cpu().numpy(), c=palette[i], alpha=0.5)
    
    if labels is not None:
        ax.scatter(labels[:n_plot].cpu().numpy(), label_cdfs, color=palette[np.arange(n_plot)], marker='x', alpha=0.5, zorder=2)
    ax.set_ylabel('cdf', fontsize=14)
    ax.set_xlabel('value', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    return ax

