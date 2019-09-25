import torch
import math
import numpy as np
from torch.distributions import Normal

TEST = True
if TEST:
    from torch.nn.functional import mse_loss
    from sklearn.gaussian_process.gpr import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct
    from sklearn.metrics import mean_squared_error


def compute_kernel(x, y):
    return torch.mm(x, y.t())
    # if kernel.lower() == 'linear':
    #     K = torch.mm(x, y.t())
    # elif kernel.lower() == 'rbf':
    #     x_i = x.unsqueeze(1)
    #     y_j = y.unsqueeze(0)
    #     xmy = ((x_i - y_j) ** 2).sum(2)
    #     K = torch.exp(-gamma_init * xmy)
    # else:
    #     raise Exception('Unhandled kernel name')
    # return K


def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


class GPLearner(torch.nn.Module):
    def __init__(self, l2):
        super(GPLearner, self).__init__()
        self.l2 = l2
        self.alpha = None
        self.phis_train = None

    def fit(self, phis, y):
        self.phis_train = phis
        self.y_train = y

        self.K_ = compute_kernel(phis, phis)
        I = eye_like(self.K_)
        I.requires_grad = False
        # self.L_ = torch.potrf(self.K_ + self.l2 * I, upper=False)
        # self.alpha = torch.potrs(y_, self.L_, upper=False)
        self.K_inv_ = torch.inverse(self.K_ + self.l2 * I)
        self.alpha = torch.mm(self.K_inv_, self.y_train)

        return self

    def log_marginal_likelihood(self, phis, y, is_train=False):
        # lml = -0.5 * torch.sum(self.alpha * self.y_train, dim=0)
        # lml -= torch.log(torch.diag(self.L_)).sum()
        # lml -= 0.5 * self.y_train.size(0) * np.log(2*np.pi)

        # I = eye_like(self.K_)
        # I.requires_grad = False
        # lml = -0.5 * torch.mm(self.K_, self.alpha)
        # print(lml)
        # lml -= 0.5 * torch.log((self.K_ + self.l2 * I).diag()).sum()
        # print(lml)
        # lml -= 0.5 * self.y_train.size(0) * np.log(2 * np.pi)
        # return lml.sum()

        # y = torch.mm(self.K_, self.alpha)
        # Normal(,)

        eps = torch.Tensor([1e-10])
        y_mean, y_std = self.forward(phis, is_train)
        std = torch.sqrt(y_std.pow(2) + self.l2) + eps
        return Normal(y_mean.view(-1), std).log_prob(y.view(-1)).sum()

    def forward(self, phis, is_train=False):
        K_trans = compute_kernel(phis, self.phis_train)
        y_mean = torch.mm(K_trans, self.alpha)

        if is_train:
            y_var = self.K_inv_.diag()
        else:
            K_test = compute_kernel(phis, phis)
            y_var = (K_test - torch.mm(K_trans, torch.mm(K_trans, self.K_inv_).t())).diag()
        return y_mean, torch.sqrt(y_var)


if __name__ == '__main__':
    from torch.nn.functional import mse_loss
    from sklearn.gaussian_process.gpr import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct
    from sklearn.metrics import mean_squared_error

    # Batch training test: Let's learn hyperparameters on a sine dataset, but test on a sine dataset and a cosine dataset
    # in parallel.
    train_x1 = torch.linspace(0, 1, 11).unsqueeze(-1)
    train_y1 = torch.sin(train_x1.data * (2 * math.pi))
    test_x1 = torch.linspace(0, 1, 51).unsqueeze(-1)
    test_y1 = torch.sin(test_x1.data * (2 * math.pi))
    print(train_x1.size(), train_y1.size(), test_x1.size(), test_y1.size())

    train_x2 = torch.linspace(0, 1, 11).unsqueeze(-1)
    train_y2 = torch.cos(train_x2.data * (2 * math.pi)).squeeze()
    test_x2 = torch.linspace(0, 1, 51).unsqueeze(-1)
    test_y2 = torch.cos(test_x2.data * (2 * math.pi)).squeeze()

    model = GPLearner(l2=torch.FloatTensor([1]))
    model.fit(train_x1, train_y1)
    y_pred, y_var = model(test_x1)

    model_true = GaussianProcessRegressor(alpha=1, kernel=DotProduct(sigma_0=0), optimizer=None)
    model_true.fit(train_x1.data.numpy(), train_y1.data.numpy())
    y_pred_true, y_var_true = model_true.predict(test_x1.data.numpy(), return_std=True)
    lml = model_true.log_marginal_likelihood()

    print(mse_loss(y_pred, test_y1).data.numpy())
    print(mean_squared_error(y_pred_true, test_y1.data.numpy()))
    assert np.allclose(mse_loss(y_pred, test_y1).data.numpy(), mean_squared_error(y_pred_true, test_y1.data.numpy()))
    print(lml)
    print(model.log_marginal_likelihood(train_x1, train_y1, is_train=True).data.numpy())

    # print(y_var_true, y_var.data.numpy())
    assert np.allclose(y_var_true, y_var.data.numpy())
