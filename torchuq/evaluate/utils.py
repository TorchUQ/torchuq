from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def make_figure_calibration(confidence, accuracy, plot_ax=None):
#     if plot_ax is None:
#         plot_ax = plt.gca()
    plt.figure(figsize=(5, 5))
    plt.plot(confidence, accuracy)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle=':')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
    
    
def _compute_reduction(original, reduction):
    assert reduction in ['none', 'mean', 'sum', 'median', 'min', 'max'], "Reduction can only be none, mean, sum, min, max or median"
    if reduction == 'none':
        return original
    elif reduction == 'mean':
        return original.mean()
    elif reduction == 'sum':
        return original.sum()
    elif reduction == 'median':
        return torch.quantile(original, 0.5)
    elif reduction == 'min':
        return original.min()
    elif reduction == 'max':
        return original.max()
    
    
def get_gaussian_filter(kernel_size=3, device=None):
    x_grid = torch.linspace(-5, 5, kernel_size)
    gaussian_kernel = (-x_grid ** 2).exp()
    gaussian_kernel /= gaussian_kernel.sum()  # Make sure sum of values in gaussian kernel equals 1.
    gaussian_filter = nn.Conv1d(1, 1, kernel_size, bias=False)

    gaussian_filter.weight.data = gaussian_kernel.view(1, 1, -1)
    gaussian_filter.weight.requires_grad = False
    
    if device is not None:
        gaussian_filter.to(device)
    return lambda x: gaussian_filter(x.view(1, 1, -1)).flatten()


def _get_uniform_filter(kernel_size=3, device=None):
    op = nn.Conv1d(1, 1, kernel_size, padding=0, bias=False) 
    # op = nn.Conv1d(1, 1, kernel_size, padding_mode='replicate', padding=kernel_size // 2, bias=False)
    op = op.to(device)
    op.weight = nn.Parameter(1./ (kernel_size+1) * torch.ones(1, 1, kernel_size+1, device=device, requires_grad=False))
    op.requires_grad = False
    return lambda x: op(x.view(1, 1, -1)).flatten()


metric_plot_colors = {
    'label': '#27ae60',
    'prediction': '#3498db',
    'quantile': '#3498db',
}