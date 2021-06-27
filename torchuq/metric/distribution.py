import seaborn as sns
import numpy as np
import torch
from matplotlib import pyplot as plt 


def plot_cdf(predictions, labels):
    margin = (labels.max() - labels.min()) * 0.3
    y_range = torch.linspace(labels.min() - margin, labels.max() + margin, 100)
    with torch.no_grad():
        cdfs = predictions.cdf(y_range.view(-1, 1))
        label_cdfs = predictions.cdf(labels)
    
    
    max_plot = cdfs.shape[1]
    if max_plot > 50:
        max_plot = 50
    palette = np.array(sns.color_palette("husl", max_plot))
    for i in range(max_plot):
        plt.plot(y_range, cdfs[:, i], c=palette[i], alpha=0.5)

    plt.scatter(labels[:max_plot], label_cdfs[:max_plot], color=palette[np.arange(max_plot)], marker='x', alpha=0.5, zorder=2)
    plt.ylabel('cdf')
    plt.xlabel('value')
    
    
def plot_icdf(predictions, labels):
    margin = (labels.max() - labels.min()) * 0.3
    c_range = torch.linspace(0.01, 0.99, 100)
    with torch.no_grad():
        values = predictions.icdf(c_range.view(-1, 1))
        label_cdfs = predictions.cdf(labels)
    
    max_plot = values.shape[1]
    if max_plot > 50:
        max_plot = 50
    palette = np.array(sns.color_palette("husl", max_plot))
    for i in range(max_plot):
        plt.plot(values[:, i], c_range, c=palette[i], alpha=0.5)
    
    plt.scatter(labels[:max_plot], label_cdfs[:max_plot], color=palette[np.arange(max_plot)], marker='x', alpha=0.5, zorder=2)
    plt.ylabel('cdf')
    plt.xlabel('value')
    

def compute_crps(predictions, labels, resolution=500):  
    """
    Compute the CRPS score 
    
    Input:
        predictions: a batch of distribution predictions
        labels: array [batch_size] of labels
        resolution: the discretization bins, higher resolution increases estimation accuracy but also requires more memory/compute
        
    Output:
        crps: array [batch_size], the CRPS score 
    """
    intervals = torch.linspace(0, 1, resolution+2)[1:-1].view(-1, 1)
    weights = (intervals ** 2)[1:] - (intervals ** 2)[:-1]   # This is a trick to compute the Lesbegue integral of CDF^2 (when we only have access to inverse CDF)
    weights = torch.cat([weights[:1], (weights[1:] + weights[:-1])/2, weights[-1:]], dim=0)   # Handle the boundary 

    icdf = predictions.icdf(intervals)  
    partition = (labels.view(1, -1) > icdf).type(torch.float32)  # Partition the intervals based on whether it's greater or less than the labels. This biases the gradient, so use with caution
    part_under = (labels - icdf) * weights * partition   # Compute the Lesbegue integral for the \int_{x, x < y} (F[x] - I(y < x))^2 dx
    part_over = (icdf - labels) * weights.flip(dims=[0]) * (1 - partition)  # Compute the Lesbegue integral for the \int_{x, x > y} (F[x] - I(y < x))^2 dx
    crps = (part_under + part_over).abs().sum(dim=0)   # The abs is not necessary, but placed here just in case numerical errors cause CRPS to be below zero. 
    return crps
  

    
def plot_reliability_diagram(predictions, labels, ax=None, plot_confidence=True):
    """
    Plot the reliability diagram https://arxiv.org/abs/1807.00263 
    
    Input:
        predictions: required Distribution instance, a batch of distribution predictions
        labels: required array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        plot_confidence: optional boolean, if set to true also plot the confidence interval. This is the range we expect the reliability diagram to fluctuate even when the predictions are perfect. 
    """
    with torch.no_grad():
        cdfs = predictions.cdf(labels).flatten() 
        cdfs, _ = torch.sort(cdfs) 
        
        if ax is None:
            plt.figure(figsize=(5, 5))
            ax = plt.gca() 

        # Use bootstrap to compute if the predictions are correct, what is the range of the reliability diagram 
        # I think it is possible to compute this in closed form, but bootstrap should work well enough
        if plot_confidence:
            samples = torch.rand(100, len(cdfs))
            samples, _ = torch.sort(samples, dim=1)
            min_curve, _ = samples.min(dim=0)
            max_curve, _ = samples.max(dim=0)
            ax.fill_between(np.linspace(0, 1, len(cdfs)), min_curve.cpu(), max_curve.cpu(), alpha=0.1, color='C1')
        
        ax.plot(cdfs.cpu().numpy(), np.linspace(0, 1, len(cdfs)), c='C0')
        ax.plot([0,1], [0,1], c='C1', linestyle=':')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('quantiles', fontsize=14)
        ax.set_ylabel('proportion samples below quantile', fontsize=14)
        
        
        
def plot_density(predictions, labels, max_count=100, ax=None):
    """ 
    Plot the PDF of the predictions and the labels. For aesthetics the PDFs are reflected along y axis to make a symmetric violin shaped plot
    
    Input:
        predictions: required Distribution instance, a batch of distribution predictions
        labels: required array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        max_count: optional int, the maximum number of PDFs to plot
    """
    resolution = 100
    values = predictions.icdf(torch.linspace(0, 1, resolution+2)[1:-1].view(-1, 1)).cpu()
    centers = (values[1:] + values[:-1]) * 0.5
    density = 1. / resolution / (values[1:] - values[:-1])  # Compute the empirical density
    vpstats = [{'coords': centers[:, i].numpy(), 'vals': density[:, i].numpy(),
                'mean': values[:, i].mean(), 'median': values[resolution // 2, i], 
                'min': values[0, i], 'max': values[-1, i]} for i in range(len(density)) if i < max_count]
    if ax is None:
        optimal_width = len(vpstats) / 5
        if optimal_width < 4:
            optimal_width = 4
        plt.figure(figsize=(optimal_width, 4))
        ax = plt.gca() 
        
    ax.violin(vpstats=vpstats, positions=range(len(vpstats)), showextrema=False, widths=0.7)
    ax.scatter(range(len(vpstats)), labels[:len(vpstats)].cpu().numpy(), marker='x')
    ax.set_ylabel('label value', fontsize=14)
    ax.set_xlabel('sample index', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)