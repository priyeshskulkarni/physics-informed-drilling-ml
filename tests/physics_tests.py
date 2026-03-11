import torch
from physics.rheology_models import herschel_bulkley

def test_rheology():

    gamma=torch.tensor([100.0])

    tau=herschel_bulkley(gamma,5,0.8,0.6)

    assert tau>0