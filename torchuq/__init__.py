import torch
import inspect 
# Define several data types

data_types = ['point', 'interval', 'particle', 'distribution', 'quantile']


# Some utility functions shared across the entire library

def _implicit_quantiles(n_quantiles):
    # Induce the implicit quantiles, these quantiles should be equally spaced 
    quantiles = torch.linspace(0, 1, n_quantiles+1)
    quantiles = (quantiles[1:] - quantiles[1] * 0.5) 
    return quantiles 


def _move_prediction_device(predictions, device):  
    """
    Move the prediction to specified device. Warning: This function may modify predictions in place
    
    Inputs:
        predictions: original prediction
        device: any torch device
        
    Outputs:
        new_predictions: the prediction in the new device
    """
    if issubclass(type(predictions), torch.distributions.distribution.Distribution):  # Trick to set the device of a torch Distribution class because there is no interface for this
        for (name, value) in inspect.getmembers(predictions, lambda v: isinstance(v, torch.Tensor)):
            try:
                setattr(predictions, name, value.to(device))
            except:   # This is a hack, some properties cannot be set because they are calculated from other properties
                pass
        return predictions
    elif issubclass(type(predictions), dict):  # If dictionary then move each element to device separately 
        for key, item in predictions.items():
            predictions[key] = _move_prediction_device(item, device) 
        return predictions
    else:
        return predictions.to(device)
    
    
def _get_prediction_device(predictions):
    """
    Get the device of a prediction
    Inputs: 
        predictions: any prediction 
    Outputs:
        device: the torch device that prediction is on
    """
    if issubclass(type(predictions), torch.distributions.distribution.Distribution):
        with torch.no_grad():  
            device = predictions.sample().device    # Trick to get the device of a torch Distribution class because there is no interface for this
    elif issubclass(type(predictions), dict):
        assert len(predictions.keys()) != 0, "Must have at least one element in the ensemble"
        device = _get_prediction_device(predictions[next(iter(predictions))])   # Return the device of the first element in the dictionary 
    else:
        device = predictions.device
    return device


def _get_prediction_batch_shape(predictions):
    """
    Get the batch_shape of the prediction 
    Inputs:
        predictions: any prediction
    Outputs:
        shape: int
    """
    if hasattr(predictions, 'batch_shape'):
        return predictions.batch_shape[0]
    if hasattr(predictions, 'shape'):   # If predictions has attribute shape 
        return predictions.shape[0]
    if issubclass(type(predictions), dict):
        return _get_prediction_batch_shape(predictions[next(iter(predictions))])
    assert False, "Cannot get the batch shape of prediction, unrecognized type"


def _parse_name(name):
    """
    Parse the name of an ensemble prediction 
    """
    components = name.split('_')
    assert len(components) == 2, 'Name does not follow the convention of type_userdefined, such as point_alice or distribution_bob123'
    assert components[0] in ['point', 'distribution', 'interval', 'quantile', 'particle'], 'Prediction type %s not valid' % components[0]
    return components[0], components[1]


def _check_valid(predictions, ptype):
    if ptype == 'ensemble':
        assert issubclass(type(predictions), dict), "Type error: an ensemble prediciton must be a python dictionary"
        for key in predictions:
            pred_type, pred_name = _parse_name(key)
            _check_valid(predictions[key], pred_type)
            
    elif ptype == 'distribution':
        required_attr = ['cdf', 'icdf']
        for attr in required_attr:
            assert hasattr(predictions, attr), "Type error: a distribution prediction must have attribute %s" % attr 

    elif ptype == 'point':
        assert issubclass(type(predictions), torch.Tensor), "Type error: a point prediction must be a torch tensor"
        assert len(predictions.shape) == 1, "Shape error, a point prediction must have shape [batch_size]"
    
    elif ptype == 'quantile':
        assert issubclass(type(predictions), torch.Tensor), "Type error: a quantile prediction must be a torch tensor"
        assert len(predictions.shape) == 2 or len(predictions.shape) == 3, "Shape error, a quantile prediction must have shape [batch_size, num_quantile, 2] or shape [batch_size, num_quantile]"
        
    elif ptype == 'particle':
        assert issubclass(type(predictions), torch.Tensor), "Type error: a particle prediction must be a torch tensor"
        assert len(predictions.shape) == 2, "Shape error, a particle prediction must have shape [batch_size, num_particle]"
        
    elif ptype == 'interval':
        assert issubclass(type(predictions), torch.Tensor), "Type error: an interval prediction must be a torch tensor"
        assert len(predictions.shape) == 2 and predictions.shape[1] == 2, "Shape error, an interval prediction must have shape [batch_size, 2]"
    
    else:
        assert False, "Unknown prediction type %s" % ptype
