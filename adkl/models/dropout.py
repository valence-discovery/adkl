from torch.distributions import Bernoulli
from torch.nn import Module


def dropout(input, p, training, dropper=None, return_dropper=False):
    idim = input.dim()
    assert idim in [3, 4], f'the number of dimensions should be 3 or 4 but got {idim}'

    batch_size = input.shape[0]
    last_dim = input.shape[-1]
    if training:
        if dropper is None:
            dropper = Bernoulli(1-p).sample((batch_size, last_dim))
        dropper_ret = dropper[:]
        for _ in range(idim-2):
            dropper = dropper.unsqueeze(1)
        res = input * dropper
    else:
        res = input * (1-p)
        dropper_ret = None

    if return_dropper:
        res = (res, dropper_ret)

    return res


class MetaDropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input, dropper=None, return_dropper=False):
        return dropout(input, self.p, self.training, dropper=dropper, return_dropper=return_dropper)


if __name__ == '__main__':
    import torch
    a = torch.randn((3, 2, 5))
    drop = MetaDropout(0.5)
    print(drop(a))
