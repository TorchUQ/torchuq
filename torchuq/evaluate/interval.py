from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import torch
from .utils import metric_plot_colors as mcolors
from .utils import _compute_reduction


def compute_length(predictions, reduction='mean'):
    """Compute the average length of an interval prediction.

    Args:
        predictions (tensor): a batch of interval predictions, which is an array [batch_size, 2].
        reduction (str): the method to aggregate the results across the batch. Can be 'none', 'mean', 'sum', 'median', 'min', or 'max'.

    Returns:
        tensor: the interval length, an array with shape [batch_size] or shape [] depending on the reduction.
    """
    length = (predictions[:, 1] - predictions[:, 0]).abs()
    return _compute_reduction(length, reduction)


def compute_coverage(predictions, labels, reduction='mean'):
    """Compute the empirical coverage. This function is not differentiable.

    Args:
        predictions (tensor): a batch of interval predictions, which is an array [batch_size, 2].
        labels (tensor): the labels, an array of shape [batch_size].
        reduction (str): the method to aggregate the results across the batch. Can be 'none', 'mean', 'sum', 'median', 'min', or 'max'.

    Returns:
        tensor: the coverage, an array with shape [batch_size] or shape [] depending on the reduction.
    """
    coverage = (labels >= predictions.min(dim=1)[0]).type(torch.float32) * (labels <= predictions.max(dim=1)[0]).type(torch.float32)
    return _compute_reduction(coverage, reduction)


def plot_interval_sequence(predictions, labels=None, ax=None, max_count=100):
    """Plot the PDF of the predictions and the labels.

    For aesthetics the PDFs are reflected along y axis to make a symmetric violin shaped plot.

    Args:
        predictions (tensor): a batch of interval predictions, which is an array [batch_size, 2].
        labels (tensor): the labels, an array of shape [batch_size].
        ax (axes): the axes to plot the figure on. If None, automatically creates a figure with recommended size.
        max_count (int): the maximum number of intervals to plot.

    Returns:
        axes: the ax on which the plot is made.
    """
    # Plot at most max_count predictions
    if len(labels) <= max_count:
        max_count = len(predictions)

    if ax is None:
        optimal_width = max_count / 4
        if optimal_width < 4:
            optimal_width = 4
        plt.figure(figsize=(optimal_width, 4))
        ax = plt.gca() 
    
    predictions = predictions.cpu()
    if labels is not None:
        labels = labels.cpu()
        valid_interval = (labels < predictions[:, 1]) & (labels > predictions[:, 0])
        colors = np.array(['#e67e22', mcolors['label']])[valid_interval[:max_count].cpu().detach().numpy().astype(np.int)]
    
    max_y = predictions[:max_count][torch.isfinite(predictions[:max_count])].max()
    min_y = predictions[:max_count][torch.isfinite(predictions[:max_count])].min()
    if labels is not None:
        max_y = max(max_y, labels[:max_count].max())
        min_y = min(min_y, labels[:max_count].min())
    max_y, min_y = max_y + (max_y - min_y) * 0.1, min_y - (max_y - min_y) * 0.1
    
    im = ax.eventplot(predictions[:max_count].cpu().numpy(), orientation='vertical', linelengths=0.5, colors='#3498db')   # Plot the quantiles as an event plot
    filled = predictions[:max_count].clone()
    filled[torch.isposinf(filled)] = max_y 
    filled[torch.isneginf(filled)] = min_y
    for i in range(max_count):
        ax.plot([i, i], [filled[i, 0], filled[i, 1]], c='#3498db')
    
    if labels is not None:
        ax.scatter(range(max_count), labels[:max_count].cpu().numpy(), marker='x', zorder=3, color=colors)
    
    # Plot the observed samples
    ax.set_ylabel('label value', fontsize=14)
    ax.set_xlabel('sample index', fontsize=14)
    ax.set_ylim([min_y, max_y])
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    return ax 


def plot_length_cdf(predictions, ax=None, plot_median=True):
    """Plot the CDF of interval length.

    Args:
        predictions (tensor): a batch of interval predictions, which is an array [batch_size, 2].
        ax (axes): the axes to plot the figure on, if None automatically creates a figure with recommended size.
        plot_median (bool): if true plot the median interval length.

    Returns:
        axes: the ax on which the plot is made.
    """
    length = torch.sort((predictions[:, 1] - predictions[:, 0]).abs())[0]
    
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 
    
    quantiles = torch.linspace(0, 1, len(length))
    ax.plot(length.cpu(), quantiles, c='C0')
    ax.set_xlabel('Interval length', fontsize=14)
    ax.set_ylabel('Prop. of intervals with smaller length', fontsize=14)
    ax.set_ylim([-0.05, 1.05])
    if plot_median:
        ax.scatter([torch.quantile(length.cpu(), 0.5).item()], [torch.quantile(quantiles, 0.5).item()], c='C0')
    return ax 
