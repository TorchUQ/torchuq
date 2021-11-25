from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
from .utils import _compute_reduction 


def compute_scores(predictions, labels, reduction='mean'):
    scores = {
        'L2': compute_l2_loss(predictions, labels, reduction=reduction),
        'Huber': compute_huber_loss(predictions, labels, reduction=reduction),
    }
    scores.update({'pinball_%.1f' % (i / 10.): compute_pinball_loss(predictions, labels, alpha=i/10., reduction=reduction)  for i in range(1, 10)})
    scores['MAE'] = scores['pinball_0.5']
    return scores 


def compute_l2_loss(predictions, labels, reduction='mean'):
    """Compute the L2 loss.

    Args:
        predictions (tensor): a batch of point predictions.
        labels (tensor): the labels, an array of shape [batch_size].
        reduction (str): the method to aggregate the results across the batch. Can be 'none', 'mean', 'sum', 'median', 'min', or 'max'.

    Returns:
        tensor: the l2 loss, an array with shape [batch_size] or shape [] depending on the reduction.
    """
    return _compute_reduction((predictions - labels).pow(2).mean(), reduction)


def compute_pinball_loss(predictions, labels, alpha=0.5, reduction='mean'):
    """Compute the pinball loss for the alpha-th quantile.

    Args:
        predictions (tensor): a batch of point predictions.
        labels (tensor): the labels, an array of shape [batch_size].
        reduction (str): the method to aggregate the results across the batch. Can be 'none', 'mean', 'sum', 'median', 'min', or 'max'.
        alpha (float): the quantile to compute the pinball loss for.

    Returns:
        tensor: the pinball loss, an array with shape [batch_size] or shape [] depending on the reduction.
    """
    residue = labels - predictions
    pinball = torch.maximum(residue * alpha, residue * (alpha-1))
    return _compute_reduction(pinball, reduction)


def compute_huber_loss(predictions, labels, reduction='mean', delta=None):
    """Compute the Huber loss.

    Args:
        predictions (tensor): a batch of point predictions.
        labels (tensor): the labels, an array of shape [batch_size].
        reduction (str): the method to aggregate the results across the batch. Can be 'none', 'mean', 'sum', 'median', 'min', or 'max'.
        delta (float): the delta parameter for the huber loss, if None then automatically set it as the top 20% largest absolute error.

    Returns:
        tensor: the huber loss, an array with shape [batch_size] or shape [] depending on the reduction.
    """
    abs_err = (predictions - labels).abs()
    if delta is None:
        with torch.no_grad():
            sorted_err = torch.sort(abs_err)[0]      
            delta = sorted_err[-len(sorted_err) // 5]
    option = (abs_err < delta).type(torch.float32)
    huber_loss = 0.5 * (abs_err ** 2) * option + delta * (abs_err - 0.5 * delta) * (1 - option)
    return _compute_reduction(huber_loss, reduction)


def plot_scatter(predictions, labels, ax=None):
    """Plot the scatter plot between the point predictions and the labels.

    Args:
        predictions (tensor): a batch of point predictions.
        labels (tensor): the labels, an array of shape [batch_size].
        ax (axes): the axes to plot the figure on, if None automatically creates a figure with recommended size.

    Returns:
        axes: the ax on which the plot is made.
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
    return ax


def plot_conditional_bias(predictions, labels, ax=None, knn=None, conditioning='label'):
    """Make the conditional bias diagram as described in [TODO: add paper reference].

    Args:
        predictions (tensor): a batch of point predictions.
        labels (tensor): the labels, an array of shape [batch_size].
        ax (axes): the axes to plot the figure on, if None automatically creates a figure with recommended size.
        knn (int): the number of nearest neighbors to average over. If None knn is set automatically.
        conditioning (str): can be 'label' or 'prediction'.

    Returns:
        axes: the ax on which the plot is made.
    """
    # Set the number of nearest neighbors to average over. 
    if knn is None:
        knn = int(max(math.sqrt(len(predictions)), 10))   # By default choose knn as the square root of the number of data points
    if knn % 2 != 0:  # Require that knn is an even number
        knn += 1
        
    with torch.no_grad():
        labels = labels.to(predictions.device)
        if conditioning == 'label':
            ranking = torch.argsort(labels)
        else:
            assert conditioning == 'prediction'
            ranking = torch.argsort(predictions)

        sorted_labels = labels[ranking]
        sorted_predictions = predictions[ranking]
        
        # Compute the average over k nearest neighbors
        smooth_kernel = 1./ (knn+1) * torch.ones(1, 1, knn+1, device=predictions.device, requires_grad=False)
        smoothed_predictions = F.conv1d(sorted_predictions.view(1, 1, -1),  weight=smooth_kernel, padding=knn // 2).flatten()[knn//2+1:-knn//2-1]
        smoothed_labels = F.conv1d(sorted_labels.view(1, 1, -1), weight=smooth_kernel, padding=knn // 2).flatten()[knn//2+1:-knn//2-1]
        min_val = min(smoothed_predictions.min(), smoothed_labels.min())
        max_val = max(smoothed_predictions.max(), smoothed_labels.max())
        
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 
        
    ax.plot(smoothed_predictions, smoothed_labels, c='C0')
    plt.plot([min_val, max_val], [min_val, max_val], linestyle=':', c='C1')
    ax.set_xlabel('prediction', fontsize=14)
    ax.set_ylabel('label', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    return ax
