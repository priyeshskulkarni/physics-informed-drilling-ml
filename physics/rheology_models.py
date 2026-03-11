import torch

def herschel_bulkley(shear_rate,tau_y,K,n):

    return tau_y + K * torch.pow(shear_rate,n)