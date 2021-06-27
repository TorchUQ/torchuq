from matplotlib import pyplot as plt

def compute_l2_loss(predictions, labels):
    return (predictions - labels).pow(2).mean()


def plot_scatter(predictions, labels, ax=None):
    """
    Plot the scatter plot between the point predictions and the labels
    
    Input:
        predictions: required array [batch_size], a batch of point predictions
        labels: required array [batch_size], the labels
        ax: optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
    """
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 
    
    r_max = max(predictions.max(), labels.max())
    r_min = min(predictions.min(), labels.min())
    r_max, r_min = r_max + (r_max - r_min) * 0.1, r_min - (r_max - r_min) * 0.1 # Margin of the plot for aesthetics
    
    ax.scatter(predictions.cpu().flatten().numpy(), labels.cpu().numpy(), c='C0')
    ax.set_xlabel('predictions', fontsize=14)
    ax.set_ylabel('labels', fontsize=14)
    ax.plot([r_min, r_max], [r_min, r_max], c='C1', linestyle=':')
    ax.set_xlim([r_min, r_max])
    ax.set_ylim([r_min, r_max])
    ax.tick_params(axis='both', which='major', labelsize=14)