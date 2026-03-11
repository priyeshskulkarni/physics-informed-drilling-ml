import torch
from physics.rheology_models import herschel_bulkley

def physics_loss(pred,shear_rate):

    tau_y = torch.tensor(5.0)
    K = torch.tensor(0.8)
    n = torch.tensor(0.6)

    expected = herschel_bulkley(
        shear_rate,tau_y,K,n
    )

    loss = torch.mean((pred-expected)**2)

    return loss