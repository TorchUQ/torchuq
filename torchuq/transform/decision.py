import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os, sys, shutil, copy, time, random

from ..metric.decision import *


# The critic model that take as input a probability predictor, and find the maximum discrepancy 
class CriticDecision(nn.Module):
    def __init__(self, num_classes=1000, num_action=2):
        super(CriticDecision, self).__init__()
        self.fc = nn.Linear(num_classes, num_action)
        self.adjustment = nn.Parameter(torch.zeros(num_action, num_classes), requires_grad=False)  # The adjustment for examples that belong to an action and for each class 
        self.num_action = num_action
        self.num_classes = num_classes
    
    # Input the predicted probability (array of size [batch_size, number_of_classes]), and the labels (int array of size [batch_size])
    # Learn the optimal critic function and the new recalibration adjustment 
    # norm should be 1 or 2
    def optimize(self, predictions, labels, num_epoch=20, norm=1, writer=None):
        device = predictions.device
        num_classes = predictions.shape[1]
        
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(predictions.cpu(), labels.cpu()), 
                                                 batch_size=256, shuffle=True, num_workers=2)
        critic_optim = optim.Adam(self.fc.parameters(), lr=1e-2)
        for epoch in range(num_epoch):
            diff_all = torch.zeros(1, self.num_action, num_classes, device=device)        # Compute the value of diff_all_{aj} = E[(Y_j - \hat{p}_j(x)) softmax(<p(x), l_a>)] 
            diff_all_true = torch.zeros(1, self.num_action, num_classes, device=device)   # Compute the binarized loss
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    bpred, blabel = data[0].to(device), data[1].to(device)
                    diff_all += self.evaluate_soft_diff(blabel, bpred).sum(dim=0, keepdim=True)
                    diff_all_true += self.evaluate_true_diff(blabel, bpred).sum(dim=0, keepdim=True)
                    # self.compute_adjustment(labels, pred_prob)
                
                if norm == 1:
                    diff_all = diff_all.sign()
                    diff_all_true = -(diff_all_true / len(predictions)).abs().mean()
                else:
                    assert norm == 2
                    diff_all /= len(predictions)
                    diff_all_true = -(diff_all_true / len(predictions)).pow(2).mean()
                    
                if writer is not None:
                    writer.add_scalar('loss_true', -diff_all_true.abs().mean(), writer.global_iteration)
                
            for i, data in enumerate(val_loader):
                critic_optim.zero_grad()
                bpred, blabel = data[0].to(device), data[1].to(device)
                loss = self.evaluate_soft_diff(blabel, bpred)
                loss = -(loss * diff_all).sum() # This is to trick the autodiff into generating the gradient we want 
                loss.backward()
                critic_optim.step()
                if writer is not None:
                    writer.add_scalar('loss_surrogate', loss, writer.global_iteration)
            
            if writer is not None:
                writer.add_scalar('epoch', epoch, writer.global_iteration)
                writer.global_iteration += 1
            
        # Compute the adjustment
        counter = torch.ones(self.num_action, device=device) * 20
        for i, data in enumerate(val_loader):
            bpred, blabel = data[0].to(device), data[1].to(device)
            # For each partition the critic finds, compute the probability adjustment to remove discrepancy 
            max_action = self.forward(bpred).argmax(dim=1)
            for action in range(self.num_action):
                selected = (F.one_hot(blabel, num_classes=num_classes) - bpred)[max_action == action]
                self.adjustment[action] += selected.sum(dim=0) 
                counter[action] += len(selected)
        self.adjustment /= counter.view(-1, 1)
                
    # For each input probability output the (relaxed) probability of taking each action
    def forward(self, x):
        fc = F.softmax(self.fc(x), dim=1)
        return fc
    
    def evaluate_soft_diff(self, labels, pred_prob):
        labels = F.one_hot(labels, num_classes=pred_prob.shape[1])  #[batch_size, num_classes]

        # print(z.shape, labels.shape)
        weights = self.forward(pred_prob).view(-1, self.num_action, 1)  # shape should be batch_size, number of actions, 1
        diff = weights * (labels - pred_prob).view(-1, 1, pred_prob.shape[1])   # diff_{iaj} = y_ij - \hat{p}(x_i)_j) softmax(<p(x_i), l_a>
        return diff 
    
    # Input the label and the prediction probability, output the true loss of the critic
    def evaluate_true_diff(self, labels, pred_prob):
        labels = F.one_hot(labels, num_classes=pred_prob.shape[1])  #[batch_size, 10]

        weight_binary = F.one_hot(self.forward(pred_prob).argmax(dim=1), num_classes=self.num_action).view(-1, self.num_action, 1)
        diff = (weight_binary * (labels - pred_prob).view(-1, 1, pred_prob.shape[1])) # disc_{iaj} = y_ij - \hat{p}_j(x_i) I(a^*(X) = a)] 
        return diff
    

    
    
            
            
class CalibratorDecision():
    def __init__(self, verbose=False, device=None, save_path=None):
        self.critics = []
        self.verbose = verbose
        self.device = device
        self.save_path = save_path
        
    def __call__(self, x, max_critic=-1):
        for index, critic in enumerate(self.critics):
            with torch.no_grad():
                bias = critic.adjustment[critic(x).argmax(dim=1)]
            x = x + bias   # Don't use inplace here 
            if index + 1 == max_critic:
                break
        return x
    
    def train(self, predictions, labels, calib_steps=200, num_action=2, num_critic_epoch=50, writer=None, norm=1, test_predictions=None, test_labels=None):
        start_time = time.time()
        for step in range(calib_steps):
            with torch.no_grad():
                updated_predictions = self(predictions)
            critic = CriticDecision(num_action=num_action, num_classes=predictions.shape[1]).to(self.device)
            critic.optimize(predictions=updated_predictions, labels=labels, num_epoch=num_critic_epoch,
                            writer=writer, norm=norm)
            self.critics.append(critic) 
            
            with torch.no_grad():
                modified_prediction = self(predictions) 
                pred_loss, true_loss = compute_decision_utility(modified_prediction.to(self.device),
                                                                labels.to(self.device), 
                                                                num_action=num_action)
                accuracy = compute_accuracy(modified_prediction.to(self.device), labels.to(self.device))
                gap = pred_loss - true_loss
                if writer is not None: 
                    writer.add_scalar('true_loss', true_loss.mean(), step)
                    writer.add_scalar('gap', gap.abs().mean(), step)
                    writer.add_scalar('accuracy', accuracy, step)
                    writer.add_histogram('true_loss_hist', true_loss, step)
                    writer.add_histogram('gap_hist', gap, step)
                if test_predictions is not None:
                    modified_prediction = self(test_predictions) 
                    pred_loss, true_loss = compute_decision_utility(modified_prediction.to(self.device),
                                                                    test_labels.to(self.device), 
                                                                    num_action=num_action)
                    accuracy = compute_accuracy(modified_prediction.to(self.device), test_labels.to(self.device))
                    gap = pred_loss - true_loss
                    if writer is not None: 
                        writer.add_scalar('true_loss_test', true_loss.mean(), step)
                        writer.add_scalar('gap_test', gap.abs().mean(), step)
                        writer.add_scalar('accuracy_test', accuracy, step)
                        writer.add_histogram('true_loss_hist_test', true_loss, step)
                        writer.add_histogram('gap_hist_test', gap, step)
                
                if self.verbose:
                    print("Step %d, time=%.1f" % (step, time.time() - start_time))
            if step % 10 == 0 and self.save_path is not None:
                self.save(os.path.join(self.save_path, 
                                       '%d-%d-%d.tar' % (predictions.shape[1], num_action, norm)))
                
    def save(self, save_path):
        if len(self.critics) == 0:
            return
        save_dict = {}
        for index, critic in enumerate(self.critics):
            save_dict[str(index)] = critic.state_dict()
        save_dict['num_action'] = self.critics[0].num_action
        save_dict['num_classes'] = self.critics[0].num_classes
        torch.save(save_dict, save_path)
    
    def load(self, save_path):
        self.critics = []
        loader = torch.load(save_path)
        print(len(loader))
        
        for index in range(len(loader) - 2):
            critic = CriticDecision(num_action=loader['num_action'], num_classes=loader['num_classes']).to(self.device)
            critic.load_state_dict(loader[str(index)])
            self.critics.append(critic)