import torch
import torch.nn as nn

class NetworkFC(nn.Module):
    def __init__(self, x_dim, out_dim=1, num_feat=30):
        super(NetworkFC, self).__init__()
        self.fc1 = nn.Linear(x_dim, num_feat)
        self.fc2 = nn.Linear(num_feat, num_feat)
        self.fc3 = nn.Linear(num_feat, out_dim)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x))))
        out = self.fc3(x)
        return out