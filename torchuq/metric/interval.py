from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np

def plot_intervals(predictions, labels, ax=None, max_count=100):
    """ 
    Plot the PDF of the predictions and the labels. For aesthetics the PDFs are reflected along y axis to make a symmetric violin shaped plot
    
    Input:
        predictions: required array [batch_size, 2] instance, a batch of interval predictions
        labels: required array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        max_count: optional int, the maximum number of PDFs to plot
    """
    # Plot at most max_count predictions
    if len(labels) <= max_count:
        max_count = len(labels)

    if ax is None:
        optimal_width = max_count / 4
        if optimal_width < 4:
            optimal_width = 4
        plt.figure(figsize=(optimal_width, 4))
        ax = plt.gca() 

    valid_interval = (labels < predictions[:, 1]) & (labels > predictions[:, 0])
    colors = np.array(['#e67e22', '#27ae60'])[valid_interval[:max_count].cpu().detach().numpy().astype(np.int)]

    im = ax.eventplot(predictions.cpu().numpy(), orientation='vertical', colors='#3498db')   # Plot the quantiles as an event plot
    ax.scatter(range(max_count), labels[:max_count].cpu().numpy(), marker='x', zorder=3, color=colors)  # Plot the observed samples
    ax.set_ylabel('label value', fontsize=14)
    ax.set_xlabel('sample index', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)