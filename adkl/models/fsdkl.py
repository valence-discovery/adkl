import copy
import torch
import numpy as np
from sklearn.metrics import r2_score
from torch.nn import Parameter
from torch.nn.functional import mse_loss, hardtanh
from adkl.feature_extraction import FeaturesExtractorFactory
from adkl.models.base import MetaLearnerRegression, MetaNetwork
from adkl.models.dropout import MetaDropout
from .utils import pack_episodes
from .kernels import compute_batch_gram_matrix


class MetaKrrSKNetwork(MetaNetwork):

    def __init__(self, feature_extractor_params, l2=0.1, kernel='linear',
                 gamma=1.0, sigma_c=1.0, sigma_p=1.0, periodicity=1.0, length_scale=1.0, normalize_kernel=False,
                 meta_dropout=0.5, hp_mode='fixe', device='cuda'):
        """
        In the constructor we instantiate an lstm module
        """
        super(MetaKrrSKNetwork, self).__init__()
        if kernel in ['gs', 'spectral']:
            feature_extractor_params['pooling_fn'] = None
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.kernel = kernel
        self.hp_mode = hp_mode
        self.device = device
        self.l2 = l2
        self.gamma = gamma
        self.sigma_c = sigma_c
        self.sigma_p = sigma_p
        self.periodicity = periodicity
        self.length_scale = length_scale
        self.meta_dropout = MetaDropout(meta_dropout)
        self.normalize_kernel = normalize_kernel

        if hp_mode.lower() in ['fixe', 'fixed', 'f']:
            self.hp_mode = 'f'
        elif hp_mode.lower() in ['learn', 'learned', 'l']:
            self.hp_mode = 'l'
        elif hp_mode.lower() in ['cv', 'valid', 'crossvalid']:
            self.hp_mode = 'cv'
        else:
            raise Exception('hp_mode should be one of those: fixe, learn, cv')
        self._init_kernel_params(device)

    def _init_kernel_params(self, device):
        self.gamma = torch.FloatTensor([self.gamma]).to(device)
        self.sigma_c = torch.FloatTensor([self.sigma_c]).to(device)
        self.sigma_p = torch.FloatTensor([self.sigma_p]).to(device)
        self.l2 = torch.FloatTensor([self.l2]).to(device)
        self.periodicity = torch.FloatTensor([self.periodicity]).to(device)
        self.length_scale = torch.FloatTensor([self.length_scale]).to(device)

        if self.hp_mode == 'l':
            self.gamma = Parameter(self.gamma)
            self.sigma_c = Parameter(self.sigma_c)
            self.sigma_p = Parameter(self.sigma_p)
            self.periodicity = Parameter(self.periodicity)
            self.l2 = Parameter(self.l2)
            self.length_scale = Parameter(self.length_scale)

        self.kernel_params = dict(l2=self.l2, sigma_c=self.sigma_c, sigma_p=self.sigma_p,
                                  gamma=self.gamma, periodicity=self.periodicity, length_scale=self.length_scale)

    def set_kernel_params(self):
        res = dict(
            l2=hardtanh(self.l2, 1e-3, 1e1),
            gamma=hardtanh(self.gamma, 1e-4, 1e4),
            sigma_c=hardtanh(self.sigma_c, 1e-4, 1e4),
            sigma_p=hardtanh(self.sigma_p, 1e-4, 1e4),
            periodicity=hardtanh(self.periodicity, 1e-4, 1e4),
            length_scale=hardtanh(self.length_scale, 1e-4, 1e4))
        self.kernel_params = res.copy()
        return res

        # self.l2_grid = torch.logspace(-4, 1, 10).to(self.device) if not self.fixe_hps else self.l2s
        # self.kernel_params_grid = dict()
        # if self.kernel == 'rbf':
        #     self.kernel_params_grid.update(dict(gamma=torch.logspace(-4, 1, 10).to(self.device)))

    def forward(self, episodes):
        kernel_params = self.set_kernel_params()
        l2 = kernel_params.pop('l2')

        train, test = pack_episodes(episodes)
        xs_train, ys_train, lens_train, mask_train = train
        xs_test, lens_test, mask_test = test
        n_train = xs_train.shape[1]
        # print(xs_train.shape, xs_test.shape)
        phis_train = self.feature_extractor(xs_train.reshape(-1, *xs_train.shape[2:]))
        phis_train = phis_train.reshape(*(xs_train.shape[:2] + phis_train.shape[1:]))
        # phis_train, dropper = self.meta_dropout(phis_train, return_dropper=True)
        batch_K = compute_batch_gram_matrix(phis_train, phis_train, kernel=self.kernel,
                                            normalize=self.normalize_kernel, **kernel_params)
        batch_K = batch_K * (mask_train[:, :, None] * mask_train[:, None, :])
        Identity = torch.eye(n_train, device=batch_K.device)
        self.alphas = torch.bmm(torch.inverse(batch_K + l2.unsqueeze(1).unsqueeze(1) * Identity), ys_train)

        phis_test = self.feature_extractor(xs_test.reshape(-1, *xs_test.shape[2:]))
        phis_test = phis_test.reshape(*xs_test.shape[:2], *phis_test.shape[1:])
        # phis_test = self.meta_dropout(phis_test, dropper=dropper)
        batch_K = compute_batch_gram_matrix(phis_test, phis_train, kernel=self.kernel,
                                            normalize=self.normalize_kernel, **kernel_params)
        batch_K = batch_K * (mask_test[:, :, None] * mask_train[:, None, :])
        preds = torch.bmm(batch_K, self.alphas)

        return [preds[:n] for n, preds in zip(lens_test, preds)]


class MetaKrrSKLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = MetaKrrSKNetwork(*args, **kwargs, device=device)
        super(MetaKrrSKLearner, self).__init__(network, optimizer, lr,
                                               weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()
        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test)
                                       for y_pred, y_test in zip(y_preds, y_tests)]))
        res.update(dict(mse=loss))
        res.update(self.model.kernel_params)

        r2 = np.mean([r2_score(pred.detach().cpu(), target.cpu()) for pred, target in zip(y_preds, y_tests)])

        res.update(dict(r2=r2))
        return loss, res

    def _fit_batch(self, *args, **kwargs):
        result = super()._fit_batch(*args, **kwargs)

        return result


if __name__ == '__main__':
    pass
