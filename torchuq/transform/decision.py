import numpy as np
import itertools
from matplotlib import pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
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
from .basic import Calibrator   # Note can use from .basic import Calibrator
from .. import _get_prediction_device
from ..evaluate.decision import compute_decision_loss_random
from ..evaluate.categorical import compute_accuracy
from .utils import PerformanceRecord
    
    
    
class CriticDecision(nn.Module):
    def __init__(self, num_classes=1000, num_action=2):
        super(CriticDecision, self).__init__()
        self.fc = nn.Linear(num_classes, num_action, bias=False)
        self.adjustment = nn.Parameter(torch.zeros(num_action, num_classes), requires_grad=False)  # The adjustment for examples that belong to an action and for each class 
        self.num_action = num_action
        self.num_classes = num_classes
    
    # Input the predicted probability (array of size [batch_size, number_of_classes]), and the labels (int array of size [batch_size])
    # Learn the optimal critic function and the new recalibration adjustment 
    # norm should be 1 or 2
    def optimize(self, predictions, labels, num_epoch=20, *args, **kwargs):
        device = predictions.device
        num_classes = predictions.shape[1]
        
        critic_optim = optim.Adam(self.fc.parameters(), lr=1e-2)
        lr_schedule = optim.lr_scheduler.StepLR(critic_optim, step_size=50, gamma=0.8)
        
        for epoch in range(num_epoch):
            for rep in range(10):
                critic_optim.zero_grad()
                loss = -self.evaluate_soft_diff(predictions.detach(), labels.detach()).mean(dim=0, keepdim=True).pow(2).sum()
                loss.backward()
                critic_optim.step()
            
            with torch.no_grad():
                adjustment = self.evaluate_adjustment(predictions, labels)
                self.adjustment = nn.Parameter(adjustment, requires_grad=False)
            lr_schedule.step()
        return loss
    
    # For each input probability output the (relaxed) probability of taking each action
    def forward(self, predictions):
        fc = self.fc(predictions)
        fc = F.softmax(fc, dim=1)
        return fc
    
    def evaluate_soft_diff(self, predictions, labels):
        labels = F.one_hot(labels, num_classes=predictions.shape[1])  #[batch_size, num_classes]

        # print(z.shape, labels.shape)
        weights = self.forward(predictions).view(-1, self.num_action, 1)   # shape should be batch_size, number of actions, 1
        diff = weights * (labels - predictions).view(-1, 1, predictions.shape[1])   # diff_{iaj} = (y_ij - \hat{p}(x_i)_j) softmax(<\hat{p}(x_i), l_a>
        return diff 
    
    def evaluate_adjustment(self, predictions, labels):
        labels = F.one_hot(labels, num_classes=predictions.shape[1])  #[batch_size, num_classes]

        # print(z.shape, labels.shape)
        weights = self.forward(predictions).unsqueeze(-1)   # shape should be batch_size, number of actions, 1
        diff = weights * (labels - predictions).view(-1, 1, predictions.shape[1])   # diff_{iaj} = (y_ij - \hat{p}(x_i)_j) softmax(<\hat{p}(x_i), l_a>
        coeff = torch.linalg.inv(torch.matmul(weights[:, :, 0].transpose(1, 0), weights[:, :, 0]))
        return torch.matmul(coeff, diff.sum(dim=0)).unsqueeze(0) 
    
    
class DecisionCalibrator(Calibrator):
    """ Recalibrate a categorical prediction to achieve decision calibration. 
    
    Args: 
        verbose (bool): if set to True than print additional performance information during training
    """
    def __init__(self, verbose=True, save_path=None):
        super(DecisionCalibrator, self).__init__()
        self.critics = []
        self.verbose = verbose
        self.save_path = save_path
        
    def __call__(self, predictions, max_critic=-1, *args, **kwargs):
        """ Use the learned recalibration map to transform predictions into decision-calibrated new predictions. 
        
        Args:
            predictions (tensor): a batch of categorical predictions.
            
        Returns:
            tensor: the transformed predictions. 
        """
        self.to(predictions)
        for index, critic in enumerate(self.critics):
            if index == max_critic:
                break
            with torch.no_grad():
                bias = (critic.adjustment * critic(predictions).unsqueeze(-1)).sum(dim=1) # bias should be [batch_size, num_class]
            predictions = predictions + bias   # Don't use inplace here!! 
        return predictions.clamp(min=1e-7, max=1-1e-7)
            
    def to(self, device):
        """ Move every torch tensor owned by this class to a new device 
        
        Args:
            device: a torch.device instance, alternatively it could be a torch.Tensor or a prediction object
        """
        if not type(device).__name__ == 'device':
            device = _get_prediction_device(device)   # This handles the case that the input is a tensor or a prediction
        if self.device is None or self.device != device:
            for critic in self.critics:
                critic.to(device)   # Critic is a subclass of nn.Module so has this method
            self.device = device
        return self 
    
    def train(self, predictions, labels, calib_steps=100, num_action=2, num_critic_epoch=500, test_predictions=None, test_labels=None, seed=0, *args, **kwargs):
        """ Train the decision calibrator for calib_steps. 
        If you call this function multiple times, this function does not erase previously trained calibration maps, and only appends additional recalibration steps
        
        Args:
            predictions (tensor(batch_size, num_classes)): a categorical probability prediction 
            labels (tensor(batch_size)): an array of int valued labels
            calib_steps (int): number of calibration iterations (this is the number of iteration steps in Algorithm 2 of the paper)
            num_critic_epoch (int): number of gradient descent steps when optimizing the worst case b in Algorithm 2 of the paper
            test_predictions (tensor(batch_size, num_classes)): a categorical probability prediction, can be set to None if measuring test performance is not needed
            test_labels (tensor(batch_size, num_classes)): an array of int valued labels, can be set to None if measuring test performance is not needed
            seed: float, the random seed when measuring performance
            
        Returns:
            recorder: a PerformanceRecord object, the measured performance 
        """
        
        self.to(predictions)
        labels = labels.to(self.device)
        
        start_time = time.time()
        recorder = PerformanceRecord()
        for step in range(calib_steps):
            # Apply the current recalibration map to the train predictions
            with torch.no_grad():
                modified_predictions = self(predictions) 
                
            # Train the new worst case critic
            critic = CriticDecision(num_action=num_action, num_classes=predictions.shape[1]).to(predictions.device)
            critic.optimize(predictions=modified_predictions, labels=labels, num_epoch=num_critic_epoch)
            
            with torch.no_grad():
                # Evaluate train performances
                pred_loss, true_loss = compute_decision_loss_random(modified_predictions, labels, num_action=num_action, seed=seed)
                accuracy = compute_accuracy(modified_predictions, labels)
                gap = pred_loss - true_loss
                gap_norm = critic.evaluate_soft_diff(modified_predictions, labels).mean(dim=0, keepdim=True).norm(2).sum()
                
                recorder.add_scalar('decision_loss_train', true_loss.mean().item(), step)
                recorder.add_scalar('gap_train', gap.abs().mean().item(), step)
                recorder.add_scalar('gap_max_train', gap.abs().max().item(), step)
                recorder.add_scalar('gap_norm_train', gap_norm.item(), step)
                recorder.add_scalar('accuracy_train', accuracy.item(), step)
            
                # Evaluate test performance if applicable
                if test_predictions is not None:
                    if issubclass(type(test_predictions), dict):   # This is for evaluating with multiple testing data
                        for pred_type in test_predictions.keys():
                            modified_test_predictions = self(test_predictions[pred_type]) 
                            cur_test_labels = test_labels[pred_type].to(self.device)
                            pred_loss, true_loss = compute_decision_loss_random(modified_test_predictions, cur_test_labels, 
                                                                                num_action=num_action, seed=seed)
                            test_accuracy = compute_accuracy(modified_test_predictions, cur_test_labels)
                            test_gap = pred_loss - true_loss
                            test_gap_norm = critic.evaluate_soft_diff(modified_test_predictions, cur_test_labels).mean(dim=0, keepdim=True).norm(2).sum() 

                            recorder.add_scalar('decision_loss_test_%s' % pred_type, true_loss.mean().item(), step)
                            recorder.add_scalar('gap_test_%s' % pred_type, test_gap.abs().mean().item(), step)
                            recorder.add_scalar('gap_max_test_%s' % pred_type, test_gap.abs().max().item(), step)
                            recorder.add_scalar('gap_norm_test_%s' % pred_type, test_gap_norm.item(), step)
                            recorder.add_scalar('accuracy_test_%s' % pred_type, test_accuracy.item(), step)                            
                    else:
                        modified_test_predictions = self(test_predictions) 
                        test_labels = test_labels.to(self.device)
                        pred_loss, true_loss = compute_decision_loss_random(modified_test_predictions, test_labels, 
                                                                        num_action=num_action, seed=seed)
                        test_accuracy = compute_accuracy(modified_test_predictions, test_labels)
                        test_gap = pred_loss - true_loss
                        test_gap_norm = critic.evaluate_soft_diff(modified_test_predictions, test_labels).mean(dim=0, keepdim=True).norm(2).sum() 

                        recorder.add_scalar('decision_loss_test', true_loss.mean().item(), step)
                        recorder.add_scalar('gap_test', test_gap.abs().mean().item(), step)
                        recorder.add_scalar('gap_max_test', test_gap.abs().max().item(), step)
                        recorder.add_scalar('gap_norm_test', test_gap_norm.item(), step)
                        recorder.add_scalar('accuracy_test', test_accuracy.item(), step)
                if self.verbose:
                    print("Step %d, time=%.1f, on the val/test set acc=%.3f/%.3f, avg loss gap=%.4f/%.4f, max loss gap=%.4f/%.4f, gap norm=%.4f/%.4f" % 
                          (step, time.time() - start_time, accuracy, test_accuracy, 
                           gap.abs().mean().item(), test_gap.abs().mean().item(),
                           gap.abs().max().item(), test_gap.abs().max().item(),
                           gap_norm.item(), test_gap_norm.item()))

            self.critics.append(critic) 
            
            if step % 10 == 0 and self.save_path is not None:
                self.save(os.path.join(self.save_path, 
                                       '%d-%d-%d.tar' % (predictions.shape[1], num_action, norm)))
        return recorder 
    
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

