import torch
import hashlib
import numpy as np
import operator as op
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, DotProduct
from sklearn.model_selection import GridSearchCV, ParameterGrid
from torch import nn
from torch.nn import MSELoss, ConstantPad1d
from torch.nn.functional import mse_loss, cosine_similarity
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict
from functools import reduce


class MaskedSoftmax(object):
    """
    Numerically stable implementation of the softmax.

    We use the implementation suggested in the practical assignment 2 of IFT6135 at UdeM.
    """

    def __init__(self, dim=1):

        self.softmax = nn.Softmax(dim=dim)

    def __call__(self, x, mask=None):
        r"""
        Performs the softmax along the given dimension, taking the mask into account.

        This version is numerically stable,
        and equivalent (up to numerical precision) as long as :math:`x \gg 10^{-9}`.

        Parameters
        ----------
        x: torch.Tensor
            Tensor on which to apply the softmax.
        mask: torch.Tensor
            The mask. Defaults to `None`.

        Returns
        -------
        s: torch.Tensor
            Tensor with the masked softmax applied on the given dimension.
        """

        if mask is not None:
            # Transforms x to account for the mask
            x = x * mask - 1e9 * (1 - mask)

        s = self.softmax(x)

        return s


class PlainWriter(object):

    def __init__(self, folder):

        if folder[-1] == '/':
            folder = folder[:-1]

        self.file = folder + '_plain_tb.csv'

        with open(self.file, 'w') as f:
            f.write('mode,tag,value,t\n')

    # def open(self):
    #     self.f = open(self.file, 'w')
    #     self.f.write('mode,tag,value,t\n')

    def write(self, mode, tag, value, t):
        with open(self.file, 'a') as f:
            f.write(f'{mode},{tag},{value},{t}\n')

    def close(self):
        pass


def compute_cos_similarity(query, support):
    # return cosine_similarity(support.unsqueeze(0), query.unsqueeze(1), dim=2)

    norms_query = torch.norm(query, p=None, dim=1)
    norms_support = torch.norm(support, p=None, dim=1)
    num = torch.mm(query, support.t())
    deno = norms_query.unsqueeze(1) * norms_support.unsqueeze(0) + 1e-10

    return num / deno


def to_numpy_vec(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy().flatten()


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def to_unit(t):
    if isinstance(t, torch.Tensor):
        x = t.data.cpu().numpy()
    else:
        x = t
    return x


def prod(iterable):
    return reduce(op.mul, iterable, 1)


def set_params(module, new_params, prefix='', exclude=()):
    """
    This allows to set the params of a module without messing with the variables and thus incapacitate the backprop
    :param module: the module of interest
    :param new_params: the params new params of the module.
            see torch.nn.Module.named_parameters() for the format of this arg
    :param prefix: default '', otherwise the name of the module
    :return:
    """

    module._parameters = OrderedDict((name, new_params[prefix + ('.' if prefix != '' else '') + name])
                                     for name, _ in module.named_parameters(recurse=False))

    for mname, submodule in module.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        set_params(submodule, new_params, submodule_prefix)


def named_parameter_sizes(module, prefix='', out=OrderedDict()):
    """
    This allows to get the size of all  params of a module
    :param module: the module of interest
    :param out: the output dict
    :param prefix: default '', otherwise the name of the module
    :return:
    """
    def prefixed_name(p, n):
        return p + ('.' if p != '' else '') + n

    for name, p in module.named_parameters(recurse=False):
        out[prefixed_name(prefix, name)] = (p.shape, p.numel())

    for mname, submodule in module.named_children():
        submodule_prefix = prefixed_name(prefix, mname)
        named_parameter_sizes(submodule, submodule_prefix, out)
    return out


def vector_to_named_parameters(vec, named_sizes):
    r"""Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))

    res = OrderedDict()
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for name, (shape_, numel_) in named_sizes.items():
        # Slice the vector, reshape it, and replace the old data of the parameter
        res[name] = vec[pointer:(pointer + numel_)].view(shape_)

        # Increment the pointer
        pointer += numel_
    return res


def __process_partition(episodes, p, return_ys=True):
    res = zip(*[(ep[p][0], ep[p][1], ep[p][0].shape[1], len(ep[p][1]),)
                for ep in episodes])
    xs, ys, dims_in, lens = res
    device = xs[0].device
    if min(dims_in) != max(dims_in):
        m = max(dims_in)
        xs = [ConstantPad1d((0, m - x.shape[1]), 0)(x) for x in xs]
    n_max = max(lens)
    mask = torch.FloatTensor([([1] * len(x)) + ([0] * (n_max - len(x))) for x in xs]).to(device)
    xs = pad_sequence(xs, batch_first=True)
    ys = pad_sequence(ys, batch_first=True)
    lens = torch.LongTensor(list(lens)).to(device)
    if return_ys:
        res = (xs, ys, lens, mask)
    else:
        res = (xs, lens, mask)
    return res


def pack_episodes(episodes, return_ys_test=False, return_query=False, return_tasks_descr=False):
    outs = ()
    train = __process_partition(episodes, 'Dtrain')
    test = __process_partition(episodes, 'Dtest', return_ys=return_ys_test)
    outs = (train, test)

    if return_query:
        query = __process_partition(episodes, 'Dquery')
        outs += (query, )

    if return_tasks_descr:
        tasks_descr = [ep['task_descr'] for ep in episodes]
        # if any([el is None for el in task_descr]):
        #     tasks_descr = None

        tasks_descr = pad_sequence(tasks_descr, batch_first=True)
        outs += (tasks_descr)

    return outs


class MaskedMSE(MSELoss):
    def __init__(self, size_average=True, reduce=True):
        super(MaskedMSE, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target_and_mask):
        target, mask = target_and_mask
        non_zeros = torch.nonzero(mask)
        y_true = target[non_zeros[:, 0], non_zeros[:, 1]]
        y_pred = input[non_zeros[:, 0], non_zeros[:, 1]]
        return mse_loss(y_pred, y_true, size_average=self.size_average, reduce=self.reduce)


sklearn_algos = dict(krr=KernelRidge, gp=GaussianProcessRegressor, gb=GradientBoostingRegressor, rf=RandomForestRegressor)

# Hyperparamters to explore for each algorithm
kernel_grids = {'rbf': {"alpha": np.logspace(-3, 2, 10), "length_scale": np.logspace(-3, 2, 10)},
                'linear': {"alpha": np.logspace(-3, 2, 10)},
                'expsine': {"alpha": np.logspace(-3, 2, 10),
                            'periodicity': np.logspace(-3, 2, 10), 'length_scale': np.logspace(-3, 2, 10)}
                }
kernel_class = {'rbf': RBF, 'expsine': ExpSineSquared}
sklearn_algo_grids = {'krr': kernel_grids,
                      'gp': kernel_grids,
                      'gb': {"n_estimators": 400},
                      'rf': {"n_estimators": 400, 'n_jobs': -1}}


def fit_and_score(x_train, y_train, x_test, y_test, algo, metrics, kernel='linear'):
    train_size = len(x_train)
    model_cls = sklearn_algos[algo]
    param_grid = sklearn_algo_grids[algo]
    if algo in ["gb", "rf"]:
        model = model_cls(**param_grid)
    elif algo in ['gp', 'krr']:
        all_params = param_grid.get(kernel).copy()
        alpha = all_params.pop('alpha')
        kernel = ([kernel_class.get(kernel)(**p) for p in ParameterGrid(all_params)]
                  if kernel != 'linear' else [DotProduct(sigma_0=0.5)])
        param_grid = dict(alpha=alpha, kernel=kernel)
        model = GridSearchCV(model_cls(), param_grid, cv=min(10, train_size), refit=True, n_jobs=1,
                             scoring='neg_mean_squared_error')
    else:
        raise Exception('Known algorithm')
    model.fit(to_numpy(x_train), to_numpy(y_train).flatten())
    if hasattr(model, 'best_params_'):
        res = dict(best_params=model.best_params_)
    else:
        res = dict()
    y_test, y_pred = to_numpy(y_test).flatten(), model.predict(to_numpy(x_test))

    res.update({metric.__name__: metric(y_pred, y_test) for metric in metrics})
    return res


def cat_uneven_length_tensors(xs):
    if isinstance(xs, torch.Tensor):
        return xs
    dims = [x.shape[1] for x in xs]
    if min(dims) != max(dims):
        m = max(dims)
        res = [nn.ConstantPad1d((0, m - x.shape[1]), 0)(x) for x in xs]
    else:
        res = xs
    return torch.cat(res)




def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return OrderedDict(sorted(items, key=(lambda x: x[0])))


def hash_params(params):
    params_flatten = flatten_dict(params)
    base = str(params_flatten)
    uid = hashlib.md5(str(base).encode()).hexdigest()
    return uid


if __name__ == '__main__':
    pass
