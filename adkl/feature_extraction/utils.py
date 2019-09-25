import functools
import types

import six
import torch
import torch.nn as nn
from torch.nn import Module

SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh',
                            'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}

OPTIMIZERS = {
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adam': torch.optim.Adam,
    'sparseadam': torch.optim.SparseAdam,
    'asgd': torch.optim.ASGD,
    'sgd': torch.optim.SGD,
    'rprop': torch.optim.Rprop,
    'rmsprop': torch.optim.RMSprop,
    'optimizer': torch.optim.Optimizer,
    'lbfgs': torch.optim.LBFGS
}


def is_callable(func):
    FUNCTYPES = (types.FunctionType, types.MethodType, functools.partial)
    return func and (isinstance(func, FUNCTYPES) or callable(func))


class GlobalMaxPool1d(nn.Module):
    # see stackoverflow
    # https://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
    def __init__(self, dim=1):
        super(GlobalMaxPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim)[0]


class GlobalAvgPool1d(nn.Module):
    def __init__(self, dim=1):
        super(GlobalAvgPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class GlobalSumPool1d(nn.Module):
    def __init__(self, dim=1):
        super(GlobalSumPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sum(x, dim=self.dim)


class GaussianDropout(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = torch.autograd.Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x


class ClonableModule(nn.Module):
    def __init__(self, ):
        super(ClonableModule, self).__init__()
        self._params = dict()

    def clone(self):
        for key in ['__class__', 'self']:
            if key in self._params:
                del self._params[key]
        model = self.__class__(**self._params)
        if next(self.parameters()).is_cuda:
            model = model.cuda()
        return model

    @property
    def output_dim(self):
        raise NotImplementedError


class Transpose(Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
        # return x.view(x.size(0), x.size(2), x.size(1))


class ResidualBlockMaker(Module):
    def __init__(self, base_module, downsample=None):
        super(ResidualBlockMaker, self).__init__()
        self.base_module = base_module
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.base_module(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def get_activation(activation):
    if is_callable(activation):
        return activation
    activation = [
        x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) > 0 and isinstance(activation[0], six.string_types), \
        'Unhandled activation function'
    if activation[0].lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation[0]]()


def get_pooling(pooling, **kwargs):
    if is_callable(pooling):
        return pooling
    # there is a reason for this to not be outside
    POOLING_MAP = {"max": GlobalMaxPool1d, "avg": GlobalAvgPool1d,
                   "sum": GlobalSumPool1d, "mean": GlobalAvgPool1d}
    return POOLING_MAP[pooling.lower()](**kwargs)


def get_optimizer(optimizer):
    if not isinstance(optimizer, six.string_types) and issubclass(optimizer, torch.optim.Optimizer):
        return optimizer
    return OPTIMIZERS[optimizer.lower()]
