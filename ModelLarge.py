import torch
import torch.nn as nn


class ModelLarge(nn.Module):
    """
    
    class to define the model large 
    
    """
    def __init__(self, n_init, hidden_dim):
        """
        Inputs: 
        
            n_init : int
                    number of feature of the X matrix 
            
            hidden_dim : int 
        
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_init, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        output = self.model(x)
        return output
