import math

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn import Parameter, Linear, Sequential, ReLU, init, Module
from torch.nn.functional import mse_loss, hardtanh

from adkl.feature_extraction import FeaturesExtractorFactory
from adkl.feature_extraction.utils import GlobalAvgPool1d
from adkl.models.base import MetaLearnerRegression, MetaNetwork
from .set_encoding import DatasetEncoderFactory
from .utils import pack_episodes, compute_cos_similarity


def get_optimizer_cls(optimizer):
    OPTIMIZERS = {k.lower(): v for k, v in vars(torch.optim).items()
                  if not k.startswith('__')}
    return OPTIMIZERS[optimizer]


class NonStationaryKernel(Module):
    def __init__(self, x_dim, t_dim, hidden_dim, spectral_kernel=False):
        super().__init__()
        self.spectral_kernel = spectral_kernel
        # self.pairs_net = Sequential(Linear(x_dim, hidden_dim), ReLU())
        # self.pairs_net = Sequential(Linear(2 * x_dim, hidden_dim), ReLU())
        self.bi_modal_net = Sequential(Linear(hidden_dim + t_dim, hidden_dim), ReLU(),
                                       Linear(hidden_dim, hidden_dim), ReLU(),
                                       Linear(hidden_dim, 1))

    def forward(self, *input):
        x, y, t = input
        xy = (x.unsqueeze(2) - y.unsqueeze(1)).pow(2)
        # x_ = x.unsqueeze(2) + torch.zeros_like(y).unsqueeze(1) # b, n, 1, d -- b, 1, m, d --> b, n, m, d
        # y_ = torch.zeros_like(x).unsqueeze(2) + y.unsqueeze(1)
        # xy = self.pairs_net(torch.cat((x_, y_), dim=-1)) + self.pairs_net(torch.cat((y_, x_), dim=-1))
        t_phis = t.unsqueeze(1).unsqueeze(1).expand((*xy.shape[:-1], t.shape[1]))
        return self.bi_modal_net(torch.cat((xy, t_phis), dim=-1)).squeeze(-1)


class StationaryKernel(Module):
    def __init__(self, x_dim, t_dim, hidden_dim, spectral_kernel=False):
        super().__init__()
        self.spectral_kernel = spectral_kernel
        self.bi_modal_net = Sequential(Linear(x_dim + t_dim, hidden_dim), ReLU(),
                                       Linear(hidden_dim, hidden_dim), ReLU(),
                                       Linear(hidden_dim, 1))

    def forward(self, *input):
        x, y, t = input
        if (not self.spectral_kernel) and x.dim() == 3:
            diff = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).pow(2)
            t_phis = t.unsqueeze(1).unsqueeze(1).expand((*diff.shape[:-1], t.shape[1]))
            K = self.bi_modal_net(torch.cat((diff, t_phis), dim=-1)).squeeze(-1)
        elif self.spectral_kernel and x.dim() == 4:
            diff = (x.unsqueeze(2).unsqueeze(4) - y.unsqueeze(1).unsqueeze(3)).pow(2)
            t_phis = t.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand((*diff.shape[:-1], t.shape[1]))

            K = self.bi_modal_net(torch.cat((diff, t_phis), dim=-1)).squeeze(-1)
            K = K.sum(-1).sum(-1) / K.shape[-1] * K.shape[-2]
        else:
            raise Exception('wrong input dimensions for kernel computation')
        return K


class TaskEncoderNet(MetaNetwork):

    def __init__(self, input_features_extractor_params,
                 target_features_extractor_params,
                 dataset_encoder_params, complement_module_input_fextractor=None):
        super(TaskEncoderNet, self).__init__()
        self.input_fextractor = FeaturesExtractorFactory()(**input_features_extractor_params)
        in_dim = self.input_fextractor.output_dim
        if complement_module_input_fextractor is not None:
            self.complement_i = complement_module_input_fextractor
        else:
            self.complement_i = Sequential()
        self.target_fextractor = FeaturesExtractorFactory()(**target_features_extractor_params)

        out_dim = self.target_fextractor.output_dim
        de_fac = DatasetEncoderFactory()
        self.f_net = de_fac(input_dim=in_dim, output_dim=out_dim, **dataset_encoder_params)
        self._output_dim = self.f_net.output_dim
        self.__params = dict(in_params=input_features_extractor_params,
                             out_params=target_features_extractor_params,
                             set_params=dataset_encoder_params)

    @property
    def output_dim(self):
        return self._output_dim

    def get_phis(self, xs, ys, mask):
        xx = self.complement_i(self.input_fextractor(xs.view(-1, xs.shape[2])))
        xx = (xx.t() * mask.view(-1)).t().reshape(*xs.shape[:2], -1)
        yy = self.target_fextractor(ys.view(-1, ys.shape[2]))
        yy = (yy.t() * mask.view(-1)).t().reshape(*ys.shape[:2], -1)
        xxyy = torch.cat((xx, yy), dim=2)
        phis = self.f_net(xxyy)
        return phis

    def forward(self, episodes, xs_train=None, ys_train=None, mask_train=None):

        if episodes is None and xs_train is not None and ys_train is not None:
            return self.get_phis(xs_train, ys_train, mask_train)

        train, test = pack_episodes(episodes, return_ys_test=True)
        xs_train, ys_train, _, mask_train = train
        xs_test, ys_test, _, mask_test = test

        phis_train = self.get_phis(xs_train, ys_train, mask_train)
        phis_test = self.get_phis(xs_test, ys_test, mask_test)

        return compute_cos_similarity(phis_test, phis_train)


class ADKL_KRR_net(MetaNetwork):
    TRAIN = 0
    DESCR = 1
    BOTH = 2
    NB_KERNEL_PARAMS = 1

    def __init__(self, input_features_extractor_params,
                 target_features_extractor_params,
                 condition_on='train', task_descr_extractor_params=None, dataset_encoder_params=None,
                 hp_mode='fixe', l2=0.1, device='cuda', task_encoder_reg=0.,
                 n_pseudo_inputs=0, pseudo_inputs_reg=0, stationary_kernel=False):
        """
        In the constructor we instantiate an lstm module
        """
        super(ADKL_KRR_net, self).__init__()

        if condition_on.lower() in ['train', 'train_samples']:
            assert dataset_encoder_params is not None, 'dataset_encoder_params must be specified'
            self.condition_on = self.TRAIN
        elif condition_on.lower() in ['descr', 'task_descr']:
            assert task_descr_extractor_params is not None, 'task_descr_extractor_params must be specified'
            self.condition_on = self.DESCR
        elif condition_on.lower() in ['both']:
            assert dataset_encoder_params is not None, 'dataset_encoder_params must be specified'
            assert task_descr_extractor_params is not None, 'task_descr_extractor_params must be specified'
            self.condition_on = self.BOTH
        else:
            raise ValueError('Invalid option for parameter condition_on')

        self.task_encoder_reg = task_encoder_reg

        if input_features_extractor_params.get('pooling_fn', 0) is None:
            pooling = GlobalAvgPool1d(dim=1)
            spectral_kernel = True
        else:
            pooling = None
            spectral_kernel = False

        task_encoder = TaskEncoderNet(
            input_features_extractor_params,
            target_features_extractor_params,
            dataset_encoder_params,
            complement_module_input_fextractor=pooling
        )
        self.features_extractor = task_encoder.input_fextractor
        fe_dim = self.features_extractor.output_dim

        self.task_descr_extractor = None
        tde_dim, de_dim = 0, 0
        if self.condition_on in [self.DESCR, self.BOTH]:
            self.task_descr_extractor = FeaturesExtractorFactory()(**task_descr_extractor_params)
            tde_dim = self.task_descr_extractor.output_dim
        if self.condition_on in [self.TRAIN, self.BOTH]:
            self.dataset_encoder = task_encoder
            de_dim = self.dataset_encoder.output_dim

        self.l2 = l2
        self.pseudo_inputs_reg = pseudo_inputs_reg
        self.hp_mode = hp_mode
        self.device = device
        if not stationary_kernel:
            self.kernel_network = NonStationaryKernel(fe_dim, de_dim + tde_dim, fe_dim, spectral_kernel)
        else:
            self.kernel_network = StationaryKernel(fe_dim, de_dim + tde_dim, fe_dim, spectral_kernel)

        if n_pseudo_inputs > 0:
            if spectral_kernel:
                self.pseudo_inputs = Parameter(torch.Tensor(n_pseudo_inputs, fe_dim)).to(device)
            else:
                self.pseudo_inputs = Parameter(torch.Tensor(n_pseudo_inputs, fe_dim)).to(device)
        else:
            self.pseudo_inputs = None
        self.phis_train_mean, self.phis_train_std = 0, 0

        if hp_mode.lower() in ['learn', 'learned', 'l']:
            self.hp_mode = 'l'
        elif hp_mode.lower() in ['predicted', 'predict', 'p', 't', 'task-specific', 'per-task']:
            self.hp_mode = 't'
            d = (de_dim if de_dim else 0) + (tde_dim if tde_dim else 0)
            self.hp_net = Linear(d, self.NB_KERNEL_PARAMS)
        else:
            raise Exception('hp_mode should be one of those: fixe, learn, cv')
        self._init_kernel_params(device)

    def _init_kernel_params(self, device):
        if self.pseudo_inputs is not None:
            init.kaiming_uniform_(self.pseudo_inputs, a=math.sqrt(5))
        self.l2 = torch.FloatTensor([self.l2]).to(device)

        if self.hp_mode == 'l':
            self.l2 = Parameter(self.l2)

    def compute_batch_gram_matrix(self, x, y, task_phis):
        k_ = self.kernel_network(x, y, task_phis)
        if self.pseudo_inputs is not None:
            ps = self.pseudo_inputs.unsqueeze(0).expand(x.shape[0], *self.pseudo_inputs.shape)
            k_g = self.kernel_network(x, ps, task_phis)
            k_ = torch.cat((k_, k_g), dim=-1)
        return k_

    def set_kernel_params(self, task_phis):
        if self.hp_mode == 't':
            self.l2 = self.hp_net(task_phis).squeeze(-1)

        l2 = hardtanh(self.l2.exp(), 1e-4, 1e1)
        return l2

    def add_pseudo_inputs_loss(self, loss):
        n = self.pseudo_inputs.shape[0]
        d = self.pseudo_inputs.shape[-1]
        p = self.pseudo_inputs.reshape(-1, d)

        # reg = torch.exp(-0.5 * (p.unsqueeze(2) - p.unsqueeze(1)).pow(2).sum(-1))
        # reg = torch.tril(reg).sum() / (n * (n - 1))

        if self.pseudo_inputs.dim() == 2:
            pi_mean = torch.mean(self.pseudo_inputs, dim=0)
            pi_std = torch.std(self.pseudo_inputs, dim=0)
        elif self.pseudo_inputs.dim() == 3:
            pi_mean = torch.mean(self.pseudo_inputs, dim=(0, 1)),
            pi_std = torch.std(self.pseudo_inputs, dim=(0, 1))
        else:
            raise Exception('Pseudo inputs: the number of dimensions is incorrect')

        kl = kl_divergence(MultivariateNormal(pi_mean, torch.diag(pi_std + 0.1)),
                           MultivariateNormal(self.phis_train_mean, torch.diag(self.phis_train_std + 0.1)))

        res = self.pseudo_inputs_reg * kl
        return loss + res, res

    def add_task_encoder_loss(self, loss):
        reg = self.task_encoder_reg * self.task_encoder_loss
        return loss + reg, reg

    def get_alphas(self, phis, ys, masks, task_phis=None):
        l2 = self.set_kernel_params(task_phis)
        bsize, n_train = phis.shape[:2]

        k_ = self.compute_batch_gram_matrix(
            phis,
            phis,
            task_phis=task_phis
        )
        k = torch.bmm(k_, k_.transpose(1, 2))
        k_mask = masks[:, None, :] * masks[:, :, None]
        k = k * k_mask

        identity = torch.eye(n_train, device=k.device).unsqueeze(0).expand((bsize, n_train, n_train))
        batch_K_inv = torch.inverse(k + l2.unsqueeze(1).unsqueeze(1) * identity)
        alphas = torch.bmm(batch_K_inv, ys)

        return alphas, k_

    def get_preds(self, alphas, K_train, phis_train, masks_train, phis_test, masks_test, task_phis=None):
        k = self.compute_batch_gram_matrix(
            phis_test,
            phis_train,
            task_phis=task_phis
        )
        k = torch.bmm(k, K_train.transpose(1, 2))
        k_mask = masks_test[:, :, None] * masks_train[:, None, :]
        k = k * k_mask

        preds = torch.bmm(k, alphas)
        return preds

    def get_task_phis(self, tasks_descr, xs_train, ys_train, mask_train):
        if self.condition_on == self.DESCR:
            task_phis = self.task_descr_extractor(tasks_descr)
        elif self.condition_on == self.TRAIN:
            task_phis = self.dataset_encoder(None, xs_train, ys_train, mask_train)
        else:
            task_phis = torch.cat(
                [
                    self.task_descr_extractor(tasks_descr),
                    self.dataset_encoder(None, xs_train, ys_train, mask_train)
                ], dim=1)
        return task_phis

    def get_phis(self, xs, train=False):
        phis = self.features_extractor(xs.reshape(-1, xs.shape[2]))
        if train:
            alpha = 0.8
            if phis.dim() == 2:
                self.phis_train_mean = (1 - alpha) * self.phis_train_mean + alpha * torch.mean(phis, dim=0).detach()
                self.phis_train_std = (1 - alpha) * self.phis_train_std + alpha * torch.std(phis, dim=0).detach()
            elif phis.dim() == 3:
                self.phis_train_mean = (1 - alpha) * self.phis_train_mean + alpha * torch.mean(phis,
                                                                                               dim=(0, 1)).detach()
                self.phis_train_std = (1 - alpha) * self.phis_train_std + alpha * torch.std(phis, dim=(0, 1)).detach()

        phis = phis.reshape(*xs.shape[:2], *phis.shape[1:])
        return phis

    def forward(self, episodes):
        if self.condition_on == self.TRAIN:
            train, test = pack_episodes(episodes, return_tasks_descr=False)
            xs_train, ys_train, lens_train, mask_train = train
            xs_test, lens_test, mask_test = test
            task_phis = self.get_task_phis(None, xs_train, ys_train, mask_train)
        else:
            train, test, tasks_descr = pack_episodes(episodes, return_tasks_descr=True)
            xs_train, ys_train, lens_train, mask_train = train
            xs_test, lens_test, mask_test = test
            task_phis = self.get_task_phis(tasks_descr, xs_train, ys_train, mask_train)

        phis_train, phis_test = self.get_phis(xs_train, train=True), self.get_phis(xs_test, train=False)

        # training
        alphas, K_train = self.get_alphas(phis_train, ys_train, mask_train, task_phis)

        # testing
        preds = self.get_preds(alphas, K_train, phis_train, mask_train, phis_test, mask_test, task_phis)

        self.compute_task_encoder_loss_last_batch(episodes)

        if isinstance(preds, tuple):
            return [tuple(x[:n] for x in pred) for n, pred in zip(lens_test, preds)]
        else:
            return [x[:n] for n, x in zip(lens_test, preds)]

    def compute_task_encoder_loss_last_batch(self, episodes):
        # train x test
        set_code = self.dataset_encoder(episodes)

        y_preds_class = torch.arange(len(set_code))
        if set_code.is_cuda:
            y_preds_class = y_preds_class.to('cuda')

        accuracy = (set_code.argmax(dim=1) == y_preds_class).sum().item() / len(set_code)

        b = set_code.size(0)
        mi = set_code.diagonal().mean() \
             - torch.log((set_code * (1 - torch.eye(b))).exp().sum() / (b * (b - 1)))
        loss = - mi

        self.accuracy = accuracy
        self.task_encoder_loss = loss


class ADKL_KRR(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0,
                 **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = ADKL_KRR_net(*args, **kwargs, device=device)

        super(ADKL_KRR, self).__init__(network, optimizer, lr,
                                       weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()

        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test)
                                       for y_pred, y_test in zip(y_preds, y_tests)]))

        r2 = np.mean([r2_score(pred.detach().cpu(), target.cpu()) for pred, target in zip(y_preds, y_tests)])

        res.update(dict(mse=loss, task_encoder_accuracy=self.model.accuracy, r2=r2, logl2=self.model.l2[0]))

        if self.model.training:
            if self.model.pseudo_inputs is not None:
                loss, pseudo_inputs_reg = self.model.add_pseudo_inputs_loss(loss)
                res.update(dict(pseudo_inputs_reg=pseudo_inputs_reg))
            loss, task_enc_loss = self.model.add_task_encoder_loss(loss)
            res.update(dict(task_enc_loss=task_enc_loss))

        return loss, res


class ADKL_GP_net(ADKL_KRR_net):
    def __init__(self, *args, std_annealing_steps=0, **kwargs):
        super(ADKL_GP_net, self).__init__(*args, **kwargs)
        # the number of steps for which the std is put small to put emphasize on the mean
        self.std_annealing_steps = std_annealing_steps
        self.step = 0

    @property
    def return_var(self):
        return True

    def get_alphas(self, phis, ys, masks, task_phis=None, return_K_inv=False):
        l2 = self.set_kernel_params(task_phis)
        bsize, n_train = phis.shape[:2]

        k_ = self.compute_batch_gram_matrix(
            phis,
            phis,
            task_phis=task_phis
        )
        k = torch.bmm(k_, k_.transpose(1, 2))
        k_mask = masks[:, None, :] * masks[:, :, None]
        k = k * k_mask

        Identity = torch.eye(n_train, device=k.device).unsqueeze(0).expand((bsize, n_train, n_train))
        batch_K_inv = torch.inverse(k + l2.view(-1, 1, 1) * Identity)
        alphas = torch.bmm(batch_K_inv, ys)
        if return_K_inv:
            return alphas, k_, batch_K_inv
        else:
            return alphas, k_

    def get_preds(self, alphas, K_train, batch_K_inv, phis_train, masks_train, phis_test, masks_test, task_phis=None):
        eps = 1e-6
        batch_K_cross = self.compute_batch_gram_matrix(phis_test, phis_train, task_phis)
        batch_K_cross = torch.bmm(batch_K_cross, K_train.transpose(1, 2)) * (
                masks_test[:, :, None] * masks_train[:, None, :])
        batch_K_test = self.compute_batch_gram_matrix(phis_test, phis_test, task_phis)
        batch_K_test = torch.diagonal(batch_K_test, dim1=1, dim2=2)
        batch_K_test = batch_K_test * masks_test
        ys_mean = torch.bmm(batch_K_cross, alphas)

        temp = torch.bmm(batch_K_cross, batch_K_inv).transpose(1, 2)

        ys_var = batch_K_test - torch.diagonal(torch.bmm(batch_K_cross, temp), dim1=1, dim2=2)
        ys_var = ys_var.clamp(min=eps)
        # ys_var = torch.ones_like(ys_mean)
        if self.step > self.std_annealing_steps:
            ys_std = (torch.sqrt(ys_var)).reshape(*ys_mean.shape)
        else:
            ys_std = torch.ones_like(ys_mean) * 1e-2

        return ys_mean, ys_std

    def forward(self, episodes, train=None, test=None, query=None, trim_ends=True):
        if self.training:
            self.step += 1
        if episodes is not None:
            train, test = pack_episodes(episodes, return_query=False)

            self.compute_task_encoder_loss_last_batch(episodes)
            query = None
        else:
            assert (train is not None) and (test is not None)

        xs_train, ys_train, lens_train, mask_train = train
        xs_test, lens_test, mask_test = test
        task_phis = self.dataset_encoder(None, xs_train, ys_train, mask_train)

        phis_train = self.get_phis(xs_train, train=True)
        phis_test = self.get_phis(xs_test)

        # training
        alphas, K_train, batch_K_inv = self.get_alphas(phis_train, ys_train, mask_train, task_phis, return_K_inv=True)

        # testing
        mus, stds = self.get_preds(alphas, K_train, batch_K_inv, phis_train, mask_train, phis_test, mask_test,
                                   task_phis)
        test_res = [(mu[:n], std[:n]) for n, mu, std in zip(lens_test, mus, stds)] if trim_ends else (mus, stds)
        if query is not None:
            xs_query, _, lens_query, mask_query = query
            phis_query = self.get_phis(xs_query)
            mus, stds = self.get_preds(alphas, K_train, batch_K_inv, phis_train, mask_train, phis_query, mask_query,
                                       task_phis)
            query_res = [(mu[:n], std[:n]) for n, mu, std in zip(lens_query, mus, stds)] if trim_ends else (mus, stds)
            return test_res, query_res
        else:
            return test_res


class ADKL_GP(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0,
                 **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = ADKL_GP_net(*args, **kwargs, device=device)
        super(ADKL_GP, self).__init__(network, optimizer, lr,
                                      weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        def nll(mu, std, y):
            return - Normal(mu.view(-1), std.view(-1)).log_prob(y.view(-1)).sum()

        res = dict()
        loss = torch.mean(torch.stack([nll(y_mean, y_std, y_test)
                                       for (y_mean, y_std), y_test in zip(y_preds, y_tests)]))
        mse = torch.mean(torch.stack([mse_loss(y_mean, y_test)
                                      for (y_mean, _), y_test in zip(y_preds, y_tests)]))

        r2 = np.mean([r2_score(pred.detach().cpu(), target.cpu()) for (pred, _), target in zip(y_preds, y_tests)])

        stds = torch.tensor([y_std.mean() for _, y_std in y_preds]).mean()

        res.update(dict(loss=loss, mse=mse, stds=stds, task_encoder_accuracy=self.model.accuracy, logl2=self.model.l2[0]), r2=r2)
        # loss = loss + stds

        if self.model.training:
            if self.model.pseudo_inputs is not None:
                loss, pseudo_inputs_reg = self.model.add_pseudo_inputs_loss(loss)
                res.update(dict(pseudo_inputs_reg=pseudo_inputs_reg))
            loss, task_enc_loss = self.model.add_task_encoder_loss(loss)
            res.update(dict(task_enc_loss=task_enc_loss))
        else:
            loss = mse

        if torch.isnan(loss):
            print('\nloss', loss)
            print('\ny_preds', y_preds)
            print('\ny_tests', y_tests)
            print('\nNll', nll)

            raise Exception("Loss goes NaN in MetaGP_MK2")

        return loss, res

    # def _score_episode(self, episode, y_test, metrics, **kwargs):
    #     x_train, y_train = episode['Dtrain']
    #     x_test, _ = episode['Dtest']
    #     self.model.eval()
    #     condition, phis_train = self.model.get_condition(episode, return_phi_train=True)
    #     phis_test = self.model.input_features_extractor(x_test)
    #     phi_train = to_numpy(self.model.conditioner(phis_train, condition.expand(x_train.shape[0], -1)))
    #     phi_test = to_numpy(self.model.conditioner(phis_test, condition.expand(x_test.shape[0], -1)))
    #     y_train, y_test = to_numpy(y_train), to_numpy(y_test)
    #     finals = dict()
    #     for algo in ['rf', 'krr']:
    #         res = fit_and_score(phi_train, y_train, phi_test, y_test, algo, metrics)
    #         finals.update({(algo + '_' + k): v for k, v in res.items()})
    #     res = super(MetaGPMKLearner, self)._score_episode(episode, y_test, metrics, **kwargs)
    #     finals.update({('native' + '_' + k): v for k, v in res.items()})
    #     return finals


if __name__ == '__main__':
    pass
