import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """ Actor (Policy) Model """
    
    def __init__(self, state_size, action_size, fc_layers, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_layers[0]: Number of nodes in first hidden layer
            fc_layers[1]: Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_layers[0])
        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.actions = nn.Linear(fc_layers[1], action_size)
    
    def forward(self, state):
        """Forward pass to map state to action values"""
        # Use Rectified Linear Units (ReLU) activation function 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.actions(x)