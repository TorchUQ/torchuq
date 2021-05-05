import torch
from matplotlib import pyplot as plt

def pinball_loss(predictions, labels):
    if len(predictions.shape) == 2:
        quantiles = torch.linspace(0, 1, predictions.shape[1]+2, device=predictions.device)[1:-1]
        residue = labels.view(-1, 1) - predictions 
    else:
        quantiles = predictions[:, :, 1]
        residue = labels.view(-1, 1) - predictions[:, :, 0] 
    loss = torch.maximum(residue * quantiles, residue * (quantiles-1)).mean()
    return loss


def plot_quantile_calibration(predictions, labels):
    with torch.no_grad():
        labels = labels.to(predictions.device)
        if len(predictions.shape) == 2:
            quantiles = torch.linspace(0, 1, predictions.shape[1]+2, device=predictions.device)[1:-1]
            below_fraction = (labels.view(-1, 1) < predictions).type(torch.float32).mean(dim=0)
        else:
            quantiles = predictions[:, :, 1] 
            below_fraction = (labels.view(-1, 1) < predictions[:, :, 0]).type(torch.float32).mean(dim=0)
        plt.scatter(quantiles.cpu(), below_fraction.cpu(), c='b')
        # Bootstrap confidence bound
        
        plt.plot([0, 1], [0, 1], c='g')
        plt.xlabel('target quantiles')
        plt.ylabel('actual quantiles')