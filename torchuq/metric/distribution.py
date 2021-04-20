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
    plt.show()
    
    
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
    plt.show()