
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torchuq.models.network import NetworkFC, NetworkFCDrop


def get_adv_bx(bx, by, loss_func):
    # Use adversarial perturbation by computing the gradient of the loss 
    bx_adv = bx.clone()
    bx_adv.requires_grad = True
    loss = loss_func(bx_adv, by)
    grads = torch.autograd.grad(loss, bx_adv, only_inputs=True)[0]

    # Add the sign of the gradient to the input 
    bx_adv.data += 0.05 * torch.sign(grads.data)
    bx_adv.requires_grad = False
    return bx_adv


def train_epoch(net, optimizer, train_loader, loss_func, use_adv, device=torch.device('cpu')):
    net.train()
    for i, (bx, by) in enumerate(train_loader):  # Standard pytorch training loop
        bx, by = bx.to(device), by.to(device)
        if use_adv:
            bx = get_adv_bx(bx, by, loss_func)
        optimizer.zero_grad()
        loss = loss_func(bx, by)
        loss.backward()
        optimizer.step()
        
        

def train_point(train_dataset, val_dataset, test_dataset, network_class=NetworkFC, use_adv=False, verbose=False, device=torch.device('cpu')):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    x_dim = len(train_dataset[0][0])
    
    from torchuq.metric.point import compute_l2_loss

    net = NetworkFC(x_dim, out_dim=1).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, threshold=1e-4, factor=0.5)
    
    def point_loss(bx, by):
        pred = net(bx).flatten()
        loss = compute_l2_loss(pred, by)
        return loss 
    
    best_loss = None
    val_x, val_y = val_dataset[:]
    test_x, test_y = test_dataset[:]
    
    for epoch in range(2000):
        if epoch % 5 == 0:    # Evaluate the validation performance
            with torch.no_grad():  
                net.eval()
                val_x, val_y = val_dataset[:]
                loss = point_loss(val_x.to(device), val_y.to(device))
                
                # Record the predictions with the best validation performance
                if best_loss is None or best_loss > loss:
                    predictions_val = net(val_x.to(device)).flatten()
                    predictions_test = net(test_x.to(device)).flatten()

                lr_scheduler.step(loss)
                if optimizer.param_groups[0]['lr'] < 1e-5:   # Hitchhike the lr scheduler to terminate if no progress
                    break
                
                if verbose and epoch % 20 == 0:
                    print("Epoch %d, loss=%.4f, lr=%.5f" % (epoch, loss, optimizer.param_groups[0]['lr']))
        train_epoch(net, optimizer, train_loader, point_loss, use_adv, device=device)

    return predictions_val, predictions_test




def train_quantile(train_dataset, val_dataset, test_dataset, network_class=NetworkFC, n_quantiles=10, use_adv=False, verbose=False, device=torch.device('cpu')):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    x_dim = len(train_dataset[0][0])
    
    from torchuq.metric.quantile import compute_pinball_loss

    # Train quantile loss
    net = NetworkFC(x_dim, out_dim=n_quantiles).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, threshold=1e-4, factor=0.5)
    
    def quantile_loss(bx, by):
        pred = net(bx)
        return compute_pinball_loss(pred, by).mean()
                
    best_loss = None
    val_x, val_y = val_dataset[:]
    test_x, test_y = test_dataset[:]
    
    for epoch in range(2000):
        if epoch % 5 == 0:    # Evaluate the validation performance
            with torch.no_grad():  
                net.eval()
                val_x, val_y = val_dataset[:]
                # pred_val, _ = torch.cummax(net(val_x.to(device)), dim=1)   # The network outputs an array of shape [batch_size, n_quantiles], do cummax to enforce ordering between quantiles
                loss = quantile_loss(val_x.to(device), val_y.to(device))
                
                # Record the predictions with the best validation performance
                if best_loss is None or best_loss > loss:
                    predictions_val = net(val_x.to(device))
                    predictions_test = net(test_x.to(device))
                    
                lr_scheduler.step(loss)
                if optimizer.param_groups[0]['lr'] < 1e-5:   # Hitchhike the lr scheduler to terminate if no progress
                    break
                    
                if verbose and epoch % 20 == 0:
                    print("Epoch %d, loss=%.4f, lr=%.5f" % (epoch, loss, optimizer.param_groups[0]['lr']))
                
        train_epoch(net, optimizer, train_loader, quantile_loss, use_adv, device=device)

            
    return predictions_val, predictions_test


def train_normal(train_dataset, val_dataset, test_dataset, network_class=NetworkFC, use_adv=False, verbose=False, device=torch.device('cpu')):
    from torchuq.metric.distribution import compute_nll
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    x_dim = len(train_dataset[0][0])
    
    net = network_class(x_dim, out_dim=2).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, threshold=1e-4, factor=0.5)
    
    # Initialize better for better numerical stability, initialize the model to have the same mean as the training data labels, and half the standard deviation
    # Empirically better initialization is crucial for consistent success in training the model. Without this initialization training fails often
    with torch.no_grad():
        train_x, train_y = train_dataset[:]
        train_x, train_y = train_x.to(device), train_y.to(device)
        data_mean = train_y.mean()
        data_std = train_y.std()
        pred_raw = net(train_x)
        mean_shift = data_mean - pred_raw[:, 0].mean()                               # Set the initial average of the mean predictions
        mean_scale = data_std / (pred_raw[:, 0].mean() - pred_raw[:, 0]).std() / 4.0 # Set the initial variance in the mean predictions
        std_shift = data_std / pred_raw[:, 1].abs().mean() / 2.0                     # Set the initial standard dev to be half the data stddev
        # print(mean_shift, std_shift, mean_scale)
              
        mean_raw = pred_raw[:, 0] + mean_shift 
        scale_raw = pred_raw[:, 1].abs() * std_shift + 1e-4
        # print(mean_raw.mean(), scale_raw.min())
        
    def get_prediction(bx):
        pred_raw = net(bx) 
        mean_raw = (pred_raw[:, 0] + mean_shift) * mean_scale 
        scale_raw = pred_raw[:, 1].abs() * std_shift + 1e-4
        pred = Normal(loc=mean_raw, scale=scale_raw)
        return pred
        
        
    def nll_loss(bx, by):
        return compute_nll(get_prediction(bx), by).mean()  
    
    best_loss = None
    val_x, val_y = val_dataset[:]
    test_x, test_y = test_dataset[:]
    
    for epoch in range(2000):
        if epoch % 5 == 0:    # Evaluate the validation performance
            with torch.no_grad():  
                net.eval()
                val_x, val_y = val_dataset[:]
                loss = nll_loss(val_x.to(device), val_y.to(device))

                if best_loss is None or best_loss > loss:
                    predictions_val = get_prediction(val_x.to(device))
                    predictions_test = get_prediction(test_x.to(device))
                    
                lr_scheduler.step(loss)
                if optimizer.param_groups[0]['lr'] < 1e-5:   # Hitchhike the lr scheduler to terminate if no progress
                    break
                    
                if verbose and epoch % 20 == 0:
                    print("Epoch %d, loss=%.4f, lr=%.5f" % (epoch, loss, optimizer.param_groups[0]['lr']))
                    
        net.train()
        train_epoch(net, optimizer, train_loader, nll_loss, use_adv, device=device)
        
    return predictions_val, predictions_test