import torch
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import binom
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl


def _implicit_quantiles(n_quantiles):
    # Induce the implicit quantiles, these quantiles should be equally spaced 
    quantiles = torch.linspace(0, 1, n_quantiles+1)
    quantiles = (quantiles[1:] - quantiles[1] * 0.5) 
    return quantiles 


def compute_pinball_loss(predictions, labels):
    """
    Compute the pinball loss, which is a proper scoring rule for quantile predictions
    
    Input:
        predictions: required array [batch_size, n_quantiles] or [batch_size, 2, n_quantiles], a batch of quantile predictions
        labels: required array [batch_size], the labels
    """
    if len(predictions.shape) == 2:
        quantiles = _implicit_quantiles(predictions.shape[1]).to(predictions.device).view(1, -1) 
        residue = labels.view(-1, 1) - predictions  
    else:
        quantiles = predictions[:, :, 1]
        residue = labels.view(-1, 1) - predictions[:, :, 1] 
    loss = torch.maximum(residue * quantiles, residue * (quantiles-1)).mean()
    return loss




def plot_quantiles(predictions, labels, max_count=100, ax=None):
    """ 
    Plot the PDF of the predictions and the labels. For aesthetics the PDFs are reflected along y axis to make a symmetric violin shaped plot
    
    Input:
        predictions: required Distribution instance, a batch of distribution predictions
        labels: required array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        max_count: optional int, the maximum number of PDFs to plot
    """
    # Plot at most 100 predictions
    if len(labels) <= max_count:
        max_count = len(labels)
        
    if len(predictions.shape) == 2:
        quantiles = _implicit_quantiles(predictions.shape[1]).view(1, -1).repeat(max_count, 1)
        predictions = predictions[:max_count]
    else:
        quantiles = predictions[:max_count, :, 1]
        predictions = predictions[:max_count, :, 0]  # Watchout for different type renaming
        
    if ax is None:
        optimal_width = max_count / 4
        if optimal_width < 4:
            optimal_width = 4 
        optimal_width += 1 # Plot bar
        plt.figure(figsize=(optimal_width+1, 4))
        ax = plt.gca() 

    colors = cm.get_cmap('coolwarm')(quantiles.numpy())
    
    im = ax.eventplot(predictions.cpu().numpy(), orientation='vertical', colors=colors)   # Plot the quantiles as an event plot
    ax.scatter(range(max_count), labels[:max_count].cpu().numpy(), marker='x', zorder=3, color='#27ae60')  # Plot the observed samples
    ax.set_ylabel('label value', fontsize=14)
    ax.set_xlabel('sample index', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot colorbar
    scalarmappaple = cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap=cm.get_cmap('coolwarm'))
    scalarmappaple.set_array(quantiles.numpy())
    cbar = plt.colorbar(scalarmappaple, ax=ax)
    cbar.set_label(label='quantiles', size=14) 
    cbar.ax.tick_params(labelsize=14) 
    
    
def plot_quantile_calibration(predictions, labels, ax=None):
    """
    Plot the reliability diagram  for quantiles
    
    Input:
        predictions: required array [n_quantiles, batch_size] or [2, n_quantiles, batch_size], a batch of quantile predictions
        labels: required array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
    """
    
    with torch.no_grad():
        labels = labels.to(predictions.device)
        if len(predictions.shape) == 2:
            quantiles = _implicit_quantiles(predictions.shape[1]).cpu()
            below_fraction = (labels.view(-1, 1).cpu() < predictions).type(torch.float32).mean(dim=0)   # Move everything to cpu
        else:
            quantiles = predictions[:, : 1].cpu()
            below_fraction = (labels.view(-1, 1).cpu() < predictions[:, :, :]).type(torch.float32).mean(dim=0)
        
        # Compute confidence bound
        min_vals = binom.ppf(0.005, len(labels), np.linspace(0, 1, 102)[1:-1]) / len(labels)
        max_vals = binom.ppf(0.995, len(labels), np.linspace(0, 1, 102)[1:-1]) / len(labels)
        
        if ax is None:
            plt.figure(figsize=(5, 5))
            ax = plt.gca() 
        
        ax.scatter(quantiles.cpu(), below_fraction.cpu(), c='C0')   
        ax.plot([0, 1], [0, 1], c='C1', linestyle=':')
        ax.fill_between(np.linspace(0, 1, 102)[1:-1], min_vals, max_vals, alpha=0.1, color='C1')   # Plot the confidence interval
        ax.set_xlabel('target quantiles', fontsize=14)
        ax.set_ylabel('actual quantiles', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
