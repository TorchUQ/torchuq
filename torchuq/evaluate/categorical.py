from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from .utils import *



    
def compute_ece(predictions, labels):
    max_confidence, prediction = predictions.max(dim=1)
    correct = (prediction == labels).type(torch.float32)
    confidence_ranking = torch.argsort(max_confidence)    
    sorted_confidence = max_confidence[confidence_ranking]
    sorted_correct = correct[confidence_ranking] 
    smooth_filter = get_uniform_filter(1000, predictions.device)
    return (smooth_filter(sorted_confidence) - smooth_filter(sorted_correct)).abs().mean()


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


def plot_calibration_diagram_naf(predictions, labels, verbose=False):
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

        
def plot_calibration_diagram_simple(predictions, labels, verbose=False, plot_ax=None):
    if len(predictions.shape) == 2:
        max_confidence, prediction = predictions.max(dim=1)
        correct = (prediction == labels).type(torch.float32)
    else:
        max_confidence = predictions
        correct = labels
    confidence_ranking = torch.argsort(max_confidence)    
    sorted_confidence = max_confidence[confidence_ranking]
    sorted_correct = correct[confidence_ranking] 
    
    smooth_filter = get_uniform_filter(1000, predictions.device)
    make_figure_calibration(smooth_filter(sorted_confidence).cpu(), smooth_filter(sorted_correct).cpu(), plot_ax=plot_ax)   
    
    
def plot_calibration_bin(predictions, labels, num_bins=15, ax=None):
    """ Plot the calibration diagram with binning 

    Args:
        predictions: a batch of categorical predictions with shape [batch_size, num_classes]
        labels: a batch of labels with shape [batch_size]
        num_bins: number of bins to plot the categorical 
    """
    with torch.no_grad():
        ranking = torch.argsort(predictions.max(dim=1)[0])
        correct = (predictions.argmax(dim=1) == labels.to(predictions.device)).type(torch.float32)
        sorted_confidence = predictions.max(dim=1)[0][ranking]
        sorted_correct = correct[ranking]
        bin_elem = len(predictions) // num_bins
        confidences = []
        accuracy = []
        for i in range(num_bins):
            confidences.append(sorted_confidence[i*bin_elem:(i+1)*bin_elem].mean().cpu().item())
            accuracy.append(sorted_correct[i*bin_elem:(i+1)*bin_elem].mean().cpu().item())
        confidence_boundary = sorted_confidence[::bin_elem].cpu()
        confidences = np.array(confidences)
        
        if ax is None: 
            plt.figure(figsize=(5, 5))
            ax = plt.gca() 
        
        plt.bar(x=confidence_boundary[:-1], height=accuracy, 
                width=confidence_boundary[1:] - confidence_boundary[:-1], alpha=0.5, align='edge')
        for i in range(num_bins):
            plt.plot([confidences[i], confidences[i]], [0, accuracy[i]], c='b')
        plt.plot([0,1], [0,1], c='g')