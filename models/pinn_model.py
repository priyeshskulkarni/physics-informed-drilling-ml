import torch
import torch.nn as nn

class PINN(nn.Module):

    def __init__(self):

        super(PINN, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(5,128),
            nn.Tanh(),

            nn.Linear(128,128),
            nn.Tanh(),

            nn.Linear(128,64),
            nn.Tanh(),

            nn.Linear(64,1)

        )

    def forward(self,x):

        return self.network(x)