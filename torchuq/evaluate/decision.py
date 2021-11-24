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

# Very annoying python 3.3+ grammar for import something from parent module
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from transform.decision import CriticDecision 


# losses should be an array of shape [num_losses, num_actions, num_classes] where losses_{ijk} should be the loss i with action=j and true label=k
def compute_decision_loss(predictions, labels, losses):
    """
    Computes the predicted prediction loss and the true prediction loss 
    """
    num_classes = predictions.shape[1]
    device = predictions.device
    num_action = losses.shape[1]
    

    pred_loss_all = torch.zeros(len(losses), device=device)     # Container for predicted loss 
    true_loss_all = torch.zeros(len(losses), device=device)   # Container for true loss
    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(predictions.cpu(), labels.cpu()), 
                                              batch_size=64, shuffle=False, num_workers=2)
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


def compute_decision_loss_random(predictions, labels, num_loss=500, num_action=2, seed=None):
    """ Simulate a set of random loss functions, and compute the Bayes decision under the prediction. Current this class only supports categorical probability predictions.
    
    Returns:
        pred_loss: predicted loss of the Bayes decision
        true_loss: true loss of the Bayes decision according to the provided true labels
    """
    device = predictions.device
    num_classes = predictions.shape[1]
    
    if seed is not None:   # Fix the random seed for reproducibility
        cur_seed = torch.get_rng_state()
        torch.manual_seed(seed)
        
    losses = torch.randn(num_loss, num_action, num_classes, device=device)
    if seed is not None:
        torch.set_rng_state(cur_seed)
        
    return compute_decision_loss(predictions, labels, losses)


def compute_decision_loss_worst(predictions, labels, num_action=2, num_critic_epoch=500, num_fold=2):
    # Do random cross validated splits 
    assert len(predictions) > num_fold * 2 
    fold_size = len(predictions) // num_fold

    loss_total = 0.0
    for fold in range(num_fold):
        left, right = fold_size * fold, fold_size * (fold + 1) if fold != num_fold - 1 else len(predictions)
    #     print(left, right) 
        train_predictions = torch.cat([predictions[:left], predictions[right:]])
        train_labels = torch.cat([labels[:left], labels[right:]])
        test_predictions = predictions[left:right]
        test_labels = labels[left:right]

        critic = CriticDecision(num_action=num_action, num_classes=predictions.shape[1]).to(predictions.device)
        critic.optimize(predictions=predictions, labels=labels, num_epoch=num_critic_epoch)
        with torch.no_grad():
            loss_total += critic.evaluate_soft_diff(test_predictions, test_labels).mean(dim=0, keepdim=True).norm(2).sum()
    loss_total /= num_fold 
    return loss_total



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