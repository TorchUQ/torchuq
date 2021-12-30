
# Define several data types

import torch 

from .regression import get_regression_datasets
from .classification import get_classification_datasets


def create_example_regression():
    """ Get the example regression prediction used in the tutorials
    
    Returns:
        distribution: an example batch of distribution predictions 
        tensor: an example batch of labels
    """
    means = torch.randn(50) * 2
    predictions = torch.distributions.normal.Normal(loc=means, scale=torch.ones(50)) 
    labels = (predictions.sample((1,)).flatten() - means) * 0.5 + means + 0.5
    return predictions, labels



