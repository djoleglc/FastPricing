import torch
import torch.nn as nn


class ModelDeep(nn.Module):
    def __init__(self, n_init, hidden_dim):
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
        output = self.model(x)
        return output
