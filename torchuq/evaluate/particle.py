
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import torch
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

