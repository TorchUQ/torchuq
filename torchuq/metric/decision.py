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


def compute_decision_utility(predictions, labels, num_loss=200, num_action=2):
    device = predictions.device
    num_classes = predictions.shape[1]
    losses = torch.randn(num_loss, num_action, num_classes, device=device)
    losses.to(device)
    with torch.no_grad():
        pred_loss_all = torch.zeros(len(losses), device=device)     # Container for predicted loss 
        true_loss_all = torch.zeros(len(losses), device=device)   # Container for true loss
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(predictions.cpu(), labels.cpu()), 
                                                  batch_size=256, shuffle=False, num_workers=2)
        count = 0
        for i, data in enumerate(data_loader):   
            bpred, blabels = data[0].to(device), data[1].to(device)
                # The predicted loss for each loss / sample / action
            pred_loss = (losses.view(len(losses), 1, num_action, num_classes) * bpred.reshape(1, -1, 1, num_classes)).sum(dim=-1) # number of losses, batch_size, number of actions
            bayes_loss, bayes_action = pred_loss.min(dim=-1)   # number of losses, batch_size
            pred_loss_all += bayes_loss.sum(dim=1)
            for k in range(bpred.shape[0]):   # For each sample in the batch, compute the true loss of the bayes action (for each of the losses)
                true_loss_all += losses[range(len(losses)), bayes_action[:, k], blabels[k]]
    return pred_loss_all / len(predictions), true_loss_all / len(predictions)


def compute_accuracy(predictions, labels):
    return (torch.argmax(predictions, dim=1) == labels).type(torch.float32).mean()


# Simulate a decision of one-vs-all
def compute_decision_loss_one_vs_all(predictions, labels):
    num_classes = predictions.shape[1]
    device = predictions.device
    num_action = 2
    losses = torch.zeros(predictions.shape[1], 2, num_classes, device=device)
    
    for i in range(num_classes):
        losses[i, 0, :] = 0.
        losses[i, 0, i] = 50.
        losses[i, 1, :] = 1.
        losses[i, 1, i] = 0.
        
    with torch.no_grad():
        pred_loss_all = torch.zeros(len(losses), device=device)     # Container for predicted loss 
        true_loss_all = torch.zeros(len(losses), device=device)   # Container for true loss
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(predictions.cpu(), labels.cpu()), 
                                                  batch_size=256, shuffle=False, num_workers=2)
        count = 0
        for i, data in enumerate(data_loader):   
            bpred, blabels = data[0].to(device), data[1].to(device)
                # The predicted loss for each loss / sample / action
            pred_loss = (losses.view(len(losses), 1, num_action, num_classes) * bpred.reshape(1, -1, 1, num_classes)).sum(dim=-1) # number of losses, batch_size, number of actions
            bayes_loss, bayes_action = pred_loss.min(dim=-1)   # number of losses, batch_size
            pred_loss_all += bayes_loss.sum(dim=1)
            for k in range(bpred.shape[0]):   # For each sample in the batch, compute the true loss of the bayes action (for each of the losses)
                true_loss_all += losses[range(len(losses)), bayes_action[:, k], blabels[k]]
    return pred_loss_all / len(predictions), true_loss_all / len(predictions)