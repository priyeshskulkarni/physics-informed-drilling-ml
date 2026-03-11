import torch

def navier_stokes_residual(u,p,mu,rho):

    grad_u = torch.autograd.grad(
        u.sum(),u,create_graph=True
    )[0]

    residual = rho*grad_u - p + mu*grad_u

    return torch.mean(residual**2)