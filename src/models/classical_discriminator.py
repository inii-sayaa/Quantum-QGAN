import torch
import torch.nn as nn


class ClassicalDiscriminator(nn.Module):
    def __init__(self, input_dim: int = 2):
       
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(128, 128), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
