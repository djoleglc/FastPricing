import torch
import torch.nn as nn


class ModelDeep(nn.Module):
    """
    
    class to define the model deep 
    
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            # output layer
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        Inputs : 
        
            x : torch tensor 
                input for which we need to estimate the price through the model 
        
        """
        output = self.model(x)
        return output
