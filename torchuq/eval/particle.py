
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import torch
from .utils import _compute_reduction


def plot_particles(predictions, labels=None, ax=None, max_count=100):
    """ 
    Plot the PDF of the predictions and the labels. For aesthetics the PDFs are reflected along y axis to make a symmetric violin shaped plot
    
    Input:
        predictions: required array [batch_size, n_particles] instance, a batch of particle
        labels: optinal array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        max_count: optional int, the maximum number of PDFs to plot
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
    
    