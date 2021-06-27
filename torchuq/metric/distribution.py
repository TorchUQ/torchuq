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
  
