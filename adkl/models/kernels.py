import torch 
import numpy as np


def linear_kernel(x, y, diag_only=False, **kwargs):
    if diag_only:
        batch_K = 1 + (x * y).sum(dim=-1)
    else:
        batch_K = 1 + torch.bmm(x, y.transpose(1, 2))
    return batch_K


def rbf_kernel(x, y, gamma, diag_only=False, **kwargs):
    bsize = y.shape[0]
    if diag_only:
        x_y_distance = (x - y).pow(2).sum(dim=-1)
    else:
        x_y_distance = (x.unsqueeze(2) - y.unsqueeze(1)).pow(2).sum(-1)
    gamma = gamma.reshape((bsize,) + (1,) * (x_y_distance.dim() - 1))
    batch_K = torch.exp(-1 * x_y_distance / gamma)
    return batch_K


def expsine_kernel(x, y, gamma, length_scale, periodicity, diag_only=False, **kwargs):
    bsize = y.shape[0]
    eps = 1e-8
    xx =  torch.cat((torch.sin(x), torch.cos(x)), dim=-1)
    yy = torch.cat((torch.sin(y), torch.cos(y)), dim=-1)
    return rbf_kernel(xx, yy, gamma, diag_only=diag_only)
    # if diag_only:
    #     x_y_distance = ((x - y).pow(2).sum(-1) + eps).sqrt()
    #     # x_y_distance = (x - y).abs().sum(dim=-1) / x.shape[-1]  + eps
    # else:
    #     x_y_distance = ((x.unsqueeze(2) - y.unsqueeze(1)).pow(2).sum(-1) + eps).sqrt()
    #     # x_y_distance = (x.unsqueeze(2) - y.unsqueeze(1)).abs().sum(dim=-1) / x.shape[-1] + eps
    # gamma = gamma.reshape((bsize,) + (1,) * (x_y_distance.dim() - 1))
    # periodicity = periodicity.reshape((bsize,) + (1,) * (x_y_distance.dim() - 1))
    # length_scale = length_scale.reshape((bsize,) + (1,) * (x_y_distance.dim() - 1))
    # batch_K = gamma * torch.exp(-2 * torch.sin(np.pi * x_y_distance / periodicity).pow(2) / length_scale)
    # return batch_K


def polynomial_kernel(x, y, degree, diag_only=False, **kwargs):
    if diag_only:
        batch_K = (1 + (x * y).sum(dim=-1)).pow(degree)
    else:
        batch_K = (1 + torch.bmm(x, y.transpose(1, 2))).pow(degree)
    return batch_K


def gs_kernel(x, y, sigma_c, sigma_p, diag_only=False, **kwargs):
    bsize = y.shape[0]
    if diag_only:
        x_y_sim = (x.unsqueeze(3) - y.unsqueeze(2)).pow(2).sum(-1)
        x_pos = torch.arange(x.shape[2], dtype=x.dtype, device=x.device).reshape(1, 1, -1)
        y_pos = torch.arange(y.shape[2], dtype=y.dtype, device=y.device).reshape(1, 1, -1)
        x_y_pos = (x_pos.unsqueeze(3) - y_pos.unsqueeze(2)).pow(2)
    else:
        x_y_sim = (x.unsqueeze(2).unsqueeze(4) - y.unsqueeze(1).unsqueeze(3)).pow(2).sum(-1)
        x_pos = torch.arange(x.shape[2], dtype=x.dtype, device=x.device).reshape(1, 1, 1, -1)
        y_pos = torch.arange(y.shape[2], dtype=y.dtype, device=y.device).reshape(1, 1, 1, -1)
        x_y_pos = (x_pos.unsqueeze(4) - y_pos.unsqueeze(3)).pow(2)
    sigma_c = sigma_c.reshape((bsize,) + (1,) * (x_y_sim.dim() - 1))
    x_y_sim = torch.exp(-1 * x_y_sim / sigma_c)
    sigma_p = sigma_p.reshape((bsize,) + (1,) * (x_y_pos.dim() - 1))
    x_y_pos = torch.exp(-1 * x_y_pos / sigma_p).expand(*x_y_sim.shape)
    batch_K = (x_y_sim * x_y_pos).sum(-1).sum(-1)
    return batch_K


def spectral_kernel(x, y, sigma_c, diag_only=False, **kwargs):
    bsize = y.shape[0]
    if diag_only:
        x_y_sim = (x.unsqueeze(3) - y.unsqueeze(2)).pow(2).sum(-1)
    else:
        x_y_sim = (x.unsqueeze(2).unsqueeze(4) - y.unsqueeze(1).unsqueeze(3)).pow(2).sum(-1)
    sigma_c = sigma_c.abs().reshape((bsize,) + (1,) * (x_y_sim.dim() - 1))
    batch_K = torch.exp(-1 * x_y_sim / sigma_c).sum(-1).sum(-1)
    return batch_K


KERNEL_MAP = dict(linear=linear_kernel,
                  poly_2=polynomial_kernel,
                  poly_3=polynomial_kernel,
                  poly_4=polynomial_kernel,
                  rbf=rbf_kernel,
                  expsine=expsine_kernel,
                  gs=gs_kernel,
                  spectral=spectral_kernel,
                  )
def compute_batch_gram_matrix(x, y, kernel='linear', diag_only=False, normalize=False,
                              gamma=torch.Tensor([1.]),
                              sigma_c=torch.Tensor([1.]),
                              sigma_p=torch.Tensor([1.]),
                              periodicity=torch.Tensor([1.]),
                              length_scale=torch.Tensor([1.])
                              ):
    assert x.shape[0] == y.shape[0]
    assert gamma.dim() == 1 and len(gamma) in [1, x.shape[0]]
    assert sigma_c.dim() == 1 and len(sigma_c) in [1, x.shape[0]]
    assert sigma_p.dim() == 1 and len(sigma_p) in [1, x.shape[0]]
    bsize = y.shape[0]
    kernel = kernel.lower()

    if len(gamma) == 1:
        gamma = gamma.expand((bsize, ))

    if len(sigma_c) == 1:
        sigma_c = sigma_c.expand((bsize, ))

    if len(sigma_p) == 1:
        sigma_p = sigma_p.expand((bsize, ))

    if len(periodicity) == 1:
        periodicity = periodicity.expand((bsize, ))

    if len(length_scale) == 1:
        length_scale = length_scale.expand((bsize, ))

    if kernel.startswith('poly'):
        if '_' in kernel:
            degree = int(kernel.split('_')[-1])
        else:
            degree = 2
        kernel = 'poly'
    else:
        degree = 2

    if kernel in KERNEL_MAP:
        batch_K = KERNEL_MAP[kernel](x, y, gamma=gamma, sigma_c=sigma_c, sigma_p=sigma_p, periodicity=periodicity,
                                     length_scale=length_scale, degree=degree, diag_only=diag_only)
    else:
        raise Exception('Unhandled kernel name')

    if normalize and not diag_only:
        xx = compute_batch_gram_matrix(x, x, kernel=kernel, diag_only=True, normalize=False)
        yy = compute_batch_gram_matrix(y, y, kernel=kernel, diag_only=True, normalize=False)
        deno = (torch.bmm(xx.unsqueeze(2), yy.unsqueeze(1)) + 1e-10).sqrt()
        batch_K = batch_K / (deno + 1e-10)

    if normalize and diag_only:
        batch_K = batch_K / (batch_K + 1e-10)

    return batch_K


if __name__ == '__main__':
    a = torch.randn(3, 4)
    a = torch.stack([a, a], dim=0)
    for k in ['l', 'rbf', 'poly_5', 'poly_2', 'poly_3']:
        print(k)
        x = compute_batch_gram_matrix(a, a, kernel=k, gamma=torch.Tensor([1]))
        print(x)
        # print(torch.diagonal(x, dim1=1, dim2=2))
        # y = compute_batch_gram_matrix(a, a, kernel=k, diag_only=True, gamma=torch.Tensor([1, 2]))
        # print(y)
    # a = torch.nn.Sequential(
    #     torch.nn.Linear(10, 100),
    #     torch.nn.Linear(100, 100),
    #     torch.nn.Linear(100, 200),
    #     torch.nn.Linear(200, 55),)
