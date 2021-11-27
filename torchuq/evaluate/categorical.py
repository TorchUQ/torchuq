from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .utils import _get_uniform_filter
from .topk import plot_confusion_matrix as plot_confusion_matrix_topk


    
def compute_ece_smooth(predictions, labels):
    max_confidence, prediction = predictions.max(dim=1)
    correct = (prediction == labels).type(torch.float32)
    confidence_ranking = torch.argsort(max_confidence)    
    sorted_confidence = max_confidence[confidence_ranking]
    sorted_correct = correct[confidence_ranking] 
    smooth_filter = get_uniform_filter(1000, predictions.device)
    return (smooth_filter(sorted_confidence) - smooth_filter(sorted_correct)).abs().mean()


def compute_ece(predictions, labels):
    confidence = predictions.max(dim=1)[0]
    correct = (predictions.argmax(dim=1) == labels.to(predictions.device)).type(torch.float32)

    # Put the confidences and accuracies into bins
    confidences = []
    accuracy = []

    # Sort by confidence 
    ranking = torch.argsort(confidence)
    sorted_confidence = confidence[ranking]
    sorted_correct = correct[ranking]

    # Divide all values into bins
    bin_elem = int(np.ceil(len(predictions) / num_bins))
    for i in range(num_bins):
        confidence_bin = sorted_confidence[::bin_elem].mean()
        accuracy_bin = sorted_correct[::bin_elem].mean()
        confidence_bin - accuracy_bin
        
        accuracy.append(sorted_correct[i*bin_elem:(i+1)*bin_elem].mean().cpu().item())
    
       


def compute_accuracy(predictions, labels):
    return (torch.argmax(predictions, dim=1) == labels).type(torch.float32).mean()


def compute_classwise_ece(predictions, labels):
    class_ece = []
    for class_index in range(predictions.shape[1]):
        prob_ranking = torch.argsort(prediction[:, class_index])
        sorted_prob = prediction[prob_ranking, class_index]
        sorted_correct = (labels == class_idx).type(torch.float32)[prob_ranking]
        smooth_filter = get_uniform_filter(1000, predictions.device)
        class_ece.append((smooth_filter(sorted_prob).cpu() - smooth_filter(sorted_correct)).abs().mean())
    return torch.cat(class_ece).mean()


def _plot_calibration_diagram_naf(predictions, labels, verbose=False):
    max_confidence, prediction = predictions.max(dim=1)
    correct = (prediction == labels).type(torch.float32)
    confidence_ranking = torch.argsort(max_confidence)    
    sorted_confidence = max_confidence[confidence_ranking]
    sorted_correct = correct[confidence_ranking]

    flow = NafFlow(feature_size=40).to(predictions.device) # This flow model is too simple, might need more layers and latents?
    flow_optim = optim.Adam(flow.parameters(), lr=1e-3)
    for iteration in range(5000):
        flow_optim.zero_grad()
        loss_all = 0.0
        
        output, _ = flow(sorted_confidence.view(-1, 1))
        loss = (output.flatten() - sorted_correct).pow(2).mean()
        loss.backward()
        flow_optim.step()
        if verbose and iteration % 100 == 0:
            print("Iteration %d, loss_bias=%.5f" % (iteration, loss_bias))

    with torch.no_grad():
        output, _ = flow.cpu()(torch.linspace(0, 1, 1000).view(-1, 1))
        flow.to(predictions.device)
        make_figure_calibration(torch.linspace(0, 1, 1000), output.flatten())

        
def plot_confusion_matrix(predictions, labels, ax=None, label_values=True):
    """ Plot the confusion matrix.
    
    Among the samples where the prediction = class i, how many labels belong to class j. 

    Args:
        predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]
        labels (tensor): a batch of labels with shape [batch_size]
        ax (axes): the axes to plot the figure on, if None automatically creates a figure with recommended size 
        label_values (bool): if set to true, label all the values with text. 
        
    Returns:
        axes: the ax on which the plot is made
    """
    predictions_max = torch.argmax(predictions, dim=1)
    return plot_confusion_matrix_topk(predictions_max, labels, ax, label_values)


def plot_reliability_diagram_smooth(predictions, labels, ax=None):
    """ Plot the reliability diagram with smoothing

    Args:
        predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]
        labels (tensor): a batch of labels with shape [batch_size]
        ax (axes): optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
    
    Returns:
        axes: the ax on which the plot is made
    """
    
    # Get the confidence and the accuracy
    max_confidence, predictions = predictions.detach().cpu().max(dim=1)
    correct = (predictions == labels).type(torch.float32).detach().cpu()

    # Sort the confidences
    confidence_ranking = torch.argsort(max_confidence)    
    sorted_confidence = max_confidence[confidence_ranking]
    sorted_correct = correct[confidence_ranking] 
    
    if ax is None: 
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 
    
    # Define a filter that smoothes a sequence
    bandwidth = int(np.sqrt(len(predictions))) * 3 + 1
    smooth_filter = _get_uniform_filter(bandwidth, device=torch.device('cpu'))
    
    with torch.no_grad():
        # Plot the smoothed confidence against the smoothed accuracy
        ax.plot(smooth_filter(sorted_confidence), smooth_filter(sorted_correct), c='C0')
    ax.plot([0,1], [0,1], c='C2')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    ax.set_xlabel('confidence', fontsize=14)
    ax.set_ylabel('accuracy', fontsize=14)
    return ax


def plot_reliability_diagram(predictions, labels, ax=None, num_bins=15, binning='adaptive'):
    """ Plot the calibration diagram with binning 

    Args:
        predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]
        labels (tensor): a batch of labels with shape [batch_size]
        ax (axes): optional matplotlib.axes.Axes, the axes to plot the figure on, if None automatically creates a figure with recommended size 
        num_bins (int): number of bins to bin the confidences
        binning (str): the binning method, can be 'adaptive' or 'uniform'. 
            For adaptive binning each bin have the same number of data points, 
            for uniform binning each bin have the same width. 
        
    Returns:
        axes: the ax on which the plot is made
    """
    with torch.no_grad():
        # Get the confidence and accuracy
        confidence = predictions.max(dim=1)[0]
        correct = (predictions.argmax(dim=1) == labels.to(predictions.device)).type(torch.float32)

        # Put the confidences and accuracies into bins
        confidences = []
        accuracy = []
        
        if binning == 'adaptive':
            # In adaptive binning, put all the values into equal width bins
            # Sort by confidence 
            ranking = torch.argsort(confidence)
            sorted_confidence = confidence[ranking]
            sorted_correct = correct[ranking]
            
            # Divide all values into bins
            bin_elem = len(predictions) // num_bins
            for i in range(num_bins):
                confidences.append(sorted_confidence[i*bin_elem:(i+1)*bin_elem].mean().cpu().item())
                accuracy.append(sorted_correct[i*bin_elem:(i+1)*bin_elem].mean().cpu().item())
                confidence_boundary = sorted_confidence[::bin_elem].cpu()
    
        elif binning == 'uniform':
            # In uniform binning, put all the values into evenly spaced bins
            confidence_boundary = torch.linspace(0, 1, num_bins+1)
            for i in range(num_bins):
                index = (confidence >= confidence_boundary[i]) & (confidence < confidence_boundary[i+1])
                if index.sum() < 5:
                    # Do not plot the bin unless it contains at least 5 data points. 
                    confidences.append(np.nan)
                    accuracy.append(np.nan)
                else:
                    confidences.append(confidence[index].mean())
                    accuracy.append(correct[index].mean())
        else:
            assert False, 'binning can only be adaptive or uniform'
            
        confidences = np.array(confidences)
        accuracy = np.array(accuracy)
        # Set the accuracy = 0 for any bin with insufficient datapoints. This will make them unplotted. 
        accuracy[np.isnan(accuracy)] = 0.0

        if ax is None: 
            plt.figure(figsize=(5, 5))
            ax = plt.gca() 
        
        ax.bar(x=confidence_boundary[:-1], height=accuracy, 
                width=confidence_boundary[1:] - confidence_boundary[:-1], alpha=0.5, align='edge', edgecolor='C0', linewidth=1.5)
        for i in range(num_bins):
            ax.plot([confidences[i], confidences[i]], [0, accuracy[i]], c='C1')
        ax.plot([0,1], [0,1], c='C2')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('confidence', fontsize=14)
        ax.set_ylabel('accuracy', fontsize=14)

        return ax