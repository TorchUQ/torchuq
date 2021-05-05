import torch




def distribution_to_point(predictions):
    pass


def distribution_to_quantile(predictions, quantiles=None, n_quantiles=None):
    assert n_quantiles is None or quantiles is None, "Cannot specify the quantiles or n_quantiles simultaneously"
    assert n_quantiles is not None or quantiles is not None, "Must specify either quantiles or n_quantiles"
    
    if n_quantiles is not None:
        quantiles = torch.linspace(0, 1, n_quantiles+2)[1:-1]
    if len(quantiles.shape) == 1:
        quantiles = quantiles.unsqueeze(1)
    return predictions.icdf(quantiles).transpose(1, 0)


def distribution_to_interval(predictions, confidence=0.95):
    # Grid search the best interval with the minimum average length 
    l_start = 0.0
    r_start = 1-confidence
    with torch.no_grad():
        for iteration in range(2): # Repeat the search procedure to get more finegrained results
            avg_length = torch.zeros(100)
            queried_values = torch.linspace(l_start, r_start, 100)
            for i, c_start in enumerate(queried_values):
                interval = predictions.icdf(torch.tensor([c_start, c_start+confidence]).view(-1, 1))
                avg_length[i] = (interval[1] - interval[0]).mean()
            best_ind = avg_length.argmin()
            l_start = queried_values[max(best_ind - 1, 0)]
            r_start = queried_values[min(best_ind + 1, 99)]
    c_start = (l_start + r_start) / 2.
    return predictions.icdf(torch.tensor([c_start, c_start+confidence]).view(-1, 1)).transpose(1, 0)


