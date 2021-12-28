
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F 
from .utils import _compute_reduction
from ..transform import direct
from .distribution import plot_density_sequence


def plot_particle_sequence(predictions, labels=None, ax=None, max_count=100):
    """ Plot the PDF of the predictions and the labels. 
    
    For aesthetics the PDFs are reflected along y axis to make a symmetric violin shaped plot
    
    Args:
        predictions (tensor): a batch of particle with shape [batch_size, n_particles]
        labels (tensor): a batch of labels with shape [batch_size]
        ax (axes): the axes to plot the figure on, if None automatically creates a figure with recommended size.
        max_count (int): the maximum number of predictions to plot.
    """
    pred_dist = direct.particle_to_distribution(predictions)
    return plot_density_sequence(pred_dist, labels, ax=ax, max_count=max_count)



def plot_particle_trend(preds, labels=None, ax=None, smooth_bw=0):
    """ Plot the batch of predictions as a time series, plot both the mean and standard deviation of the prediction. 
    
    Args:
        predictions (tensor): a batch of particle with shape [batch_size, n_particles]
        labels (tensor): a batch of labels with shape [batch_size]
        ax (axes): the axes to plot the figure on, if None automatically creates a figure with recommended size.
        smooth_bw (int): if smooth_bw is not 0, smooth the sequence with by convolution with a window of size smooth_bw*2+1 
        
    Returns:
        axes: the ax on which the plot is made, it is an instance of matplotlib.axes.Axes. 
    """
    
    # Compute the mean and std of the particles
    pred_mean = preds.mean(dim=1).cpu().detach()
    pred_std = preds.std(dim=1).cpu().detach()
    
    # Function to smooth a sequence
    def smooth(seq):
        padded_seq = torch.cat([seq[:smooth_bw], seq, seq[-smooth_bw:]], dim=0).unsqueeze(0).unsqueeze(0)
        conv_kernel = torch.ones(1, 1, smooth_bw*2+1, device=seq.device) / (smooth_bw*2+1)
        smooth_seq = F.conv1d(padded_seq, weight=conv_kernel) 
        return smooth_seq.flatten()
    
    # Smooth the sequence
    if smooth_bw != 0:
        pred_mean = smooth(pred_mean)
        pred_std = smooth(pred_std)
        
    # Create a figure of the right size if not provided
    if ax is None:
        plt.figure(figsize=(6, 4))
        ax = plt.gca() 
        
    # Plot the prediction sequence and the standard deviation
    plt.plot(range(len(preds)), pred_mean, c='C1', label='mean prediction')
    plt.fill_between(range(len(preds)), pred_mean-pred_std, pred_mean+pred_std, alpha=0.1, color='C1')
    
    # Plot the labels
    if labels is not None:
        ax.scatter(range(len(labels)), labels.detach().cpu().numpy(), marker='x', label='label')
    
    # Label the plot
    ax.set_ylabel('label value', fontsize=14)
    ax.set_xlabel('sample index', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14)
    
    return ax 