import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DQN(nn.Module):
    # Attribute type declarations
    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear
    
    def __init__(self, state_dim: int, action_dim: int) -> None:
        '''
        Parameters:
        -state_dim (int): The dimensionality of the state input vector (4 for cartpole)
        -action_dim (int): The number of actions available (2 for cartpole)
        '''
        super(DQN, self).__init__()
        
        # Layer definitions. Forms a fully connected (fc) feedforward network.
        self.fc1 = nn.Linear(state_dim, 128) # state (4) -> hidden 1 (128)
        self.fc2 = nn.Linear(128, 128) # hidden 1 (128) -> hidden 2 (128)
        self.fc3 = nn.Linear(128, action_dim) # hidden 2 (128) -> outputs (2)
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        This does the forward pass of the network.
        Input: x (tensor, shape is [batch_size, state_dim]
        Output: Tensor w/ shape [batch_size, action_dim] where each element represents a Q-value predicted by the model.
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x