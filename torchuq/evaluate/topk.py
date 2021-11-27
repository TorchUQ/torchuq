from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



def compute_accuracy(predictions, labels):
    """ Compute the frequency that the label belongs to the predicted set 
    
    Args:
        predictions (tensor): a batch of topk predictions with shape [batch_size] or [batch_size, k]
        labels (tensor): a batch of labels with shape [batch_size]
    """
    return 
    
    
def plot_confusion_matrix(predictions, labels, ax=None, label_values=True):
    """ Plot the confusion matrix.
    
    Among the samples where the prediction = class i, how many labels belong to class j. 

    Args:
        predictions (tensor): a batch of topk predictions (but must be top1) with shape [batch_size] or [batch_size, 1]
        labels (tensor): a batch of labels with shape [batch_size]
        ax (axes): the axes to plot the figure on, if None automatically creates a figure with recommended size 
        label_values (bool): if set to true, label all the values with text. 
        
    Returns:
        axes: the ax on which the plot is made
    """
    hist_list = []

    # Check that the predictions is a top-1 prediction
    assert len(predictions.shape) == 1 or predictions.shape[1] == 1, 'plot_confusion_matrix only supported top-1 predictions'
    predictions = predictions.flatten()
    
    # Figure out the maximum label index
    n_classes = max(predictions.max(), labels.max()) + 1
    
    with torch.no_grad():
        for li in range(n_classes):
            # Distribution over possible labels when the prediction is li
            hist = torch.bincount(labels[predictions == li], minlength=n_classes).type(torch.float)
            if hist.sum() > 0:
                hist /= hist.sum()
            hist_list.append(hist)
        hist_list = torch.stack(hist_list).cpu()
        
    if ax is None:
        plt.figure(figsize=(n_classes / 1.5, n_classes / 1.5))
        ax = plt.gca()
            
    # Plot the color on log scale
    # This highlight values that are close to zero (but not exactly zero)
    gamma = 0.05
    ax.imshow((gamma + hist_list).log() - np.log(gamma), vmax=(np.log(1 + gamma) - np.log(gamma)) * 2, cmap='Blues')

    ax.set_xlabel('true label', fontsize=14)
    ax.set_ylabel('predicted label', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # Plot the exact numbers 
    if label_values:
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, '%.3f' % hist_list[i, j], ha='center', ma='center')
    return ax



