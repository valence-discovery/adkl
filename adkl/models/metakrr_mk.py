import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d
from torch.nn.functional import mse_loss, hardtanh

from adkl.feature_extraction import FeaturesExtractorFactory
from adkl.feature_extraction.utils import (GlobalAvgPool1d, GlobalMaxPool1d, GlobalSumPool1d)
from adkl.models.base import MetaLearnerRegression, MetaNetwork
from adkl.models.dropout import MetaDropout
from .conditioning import ConditionerFactory
from .task_encoder import TaskEncoderNet
from .utils import pack_episodes
from .kernels import compute_batch_gram_matrix


def get_optimizer_cls(optimizer):
    OPTIMIZERS = {k.lower(): v for k, v in vars(torch.optim).items()
                  if not k.startswith('__')}
    return OPTIMIZERS[optimizer]


class MetaKrrMKNetwork(MetaNetwork):
    TRAIN = 0
    DESCR = 1
    BOTH = 2
    NB_KERNEL_PARAMS = 6

    def __init__(self, input_features_extractor_params,
                 target_features_extractor_params,
                 tie_input_features_extractor_weights=True,
                 condition_on='train', kernel='linear',
                 gamma=1.0, sigma_c=1.0, sigma_p=1.0, periodicity=1.0, length_scale=1.0,
                 task_descr_extractor_params=None, dataset_encoder_params=None,
                 hp_mode='fixe', l2=0.1, meta_dropout=0.,
                 conditioner_mode='cat', conditioner_params=None,
                 pretrain_task_encoder=False, fixe_task_encoder=False, device='cuda', joint_training=1.,
                 joint_training_mine=False,
                 normalize_kernel=True):
        """
        In the constructor we instantiate an lstm module
        """
        super(MetaKrrMKNetwork, self).__init__()

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

        self.joint_training = joint_training

        if kernel in ['gs', 'spectral']:
            pooling_fn = input_features_extractor_params.get('pooling_fn', 'avg')
            input_features_extractor_params['pooling_fn'] = None
            if pooling_fn == 'avg':
                pooling = GlobalAvgPool1d(dim=1)
            elif pooling_fn == 'sum':
                pooling = GlobalSumPool1d(dim=1)
            elif pooling_fn == 'max':
                pooling = GlobalMaxPool1d(dim=1)
            else:
                raise Exception("Unknown pooling function")
        else:
            pooling = None

        task_encoder = TaskEncoderNet.get_model(
            input_features_extractor_params,
            target_features_extractor_params,
            dataset_encoder_params,
            pretrained=pretrain_task_encoder,
            fixe_params=fixe_task_encoder,
            complement_module_input_fextractor=pooling
        )

        if tie_input_features_extractor_weights and (not fixe_task_encoder):
            self.features_extractor = task_encoder.input_fextractor
        else:
            self.features_extractor = FeaturesExtractorFactory()(**input_features_extractor_params)

        self.task_descr_extractor = None
        tde_dim, de_dim = None, None
        if self.condition_on in [self.DESCR, self.BOTH]:
            self.task_descr_extractor = FeaturesExtractorFactory()(**task_descr_extractor_params)
            tde_dim = self.task_descr_extractor.output_dim
        if self.condition_on in [self.TRAIN, self.BOTH]:
            self.dataset_encoder = task_encoder
            de_dim = self.dataset_encoder.output_dim
        cpa = dict() if conditioner_params is None else conditioner_params

        self.conditioner = ConditionerFactory()(conditioner_mode, self.features_extractor.output_dim,
                                                tde_dim,
                                                de_dim, **cpa)

        self.kernel = kernel
        self.l2 = l2
        self.gamma = gamma
        self.sigma_c = sigma_c
        self.sigma_p = sigma_p
        self.periodicity = periodicity
        self.length_scale = length_scale
        self.normalize_kernel = normalize_kernel
        self.meta_dropout = MetaDropout(meta_dropout)
        self.joint_training_mine = joint_training_mine

        self.hp_mode = hp_mode
        self.device = device

        if hp_mode.lower() in ['fixe', 'fixed', 'f']:
            self.hp_mode = 'f'
        elif hp_mode.lower() in ['learn', 'learned', 'l']:
            self.hp_mode = 'l'
        elif hp_mode.lower() in ['predicted', 'predict', 'p', 't', 'task-specific', 'per-task']:
            self.hp_mode = 't'
            d = (de_dim if de_dim else 0) + (tde_dim if tde_dim else 0)
            self.hp_net = Linear(d, self.NB_KERNEL_PARAMS)
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
            self.length_scale = Parameter(self.length_scale)
            self.l2 = Parameter(self.l2)

        self.kernel_params = dict(l2=self.l2, sigma_c=self.sigma_c, sigma_p=self.sigma_p,
                                  gamma=self.gamma, periodicity=self.periodicity, length_scale=self.length_scale)

    def get_adapted_phis(self, task_phis, xs):
        t_phis = task_phis.unsqueeze(1).expand((*xs.shape[:2], task_phis.shape[1])).reshape(-1, task_phis.shape[1])
        phis = self.features_extractor(xs.reshape(-1, xs.shape[2]))
        phis = self.conditioner(phis, t_phis)
        phis = phis.reshape(*xs.shape[:2], *phis.shape[1:])
        return phis

    def set_kernel_params(self, task_phis):
        if self.hp_mode == 't':
            hp = self.hp_net(task_phis).exp()
            self.l2, self.gamma, self.sigma_c, self.sigma_p, self.periodicity, self.length_scale = hp.t()

        res = dict(
            l2=hardtanh(self.l2, 1e-3, 1e1),
            gamma=hardtanh(self.gamma, 1e-4, 1e4),
            sigma_c=hardtanh(self.sigma_c, 1e-4, 1e4),
            sigma_p=hardtanh(self.sigma_p, 1e-4, 1e4),
            periodicity=hardtanh(self.periodicity, 1e-4, 1e4),
            length_scale=hardtanh(self.length_scale, 1e-4, 1e4))
        self.kernel_params = {k: v[0] for k, v in res.items()}
        return res


    def get_alphas(self, phis, ys, masks, task_phis=None):
        kernel_params = self.set_kernel_params(task_phis)
        l2 = kernel_params.pop('l2')
        bsize, n_train = phis.shape[:2]

        k = compute_batch_gram_matrix(
            phis,
            phis,
            kernel=self.kernel,
            normalize=self.normalize_kernel,
            **kernel_params
        )
        k_mask = masks[:, None, :] * masks[:, :, None]
        k = k * k_mask

        identity = torch.eye(n_train, device=k.device).unsqueeze(0).expand((bsize, n_train, n_train))
        batch_K_inv = torch.inverse(k + l2.unsqueeze(1).unsqueeze(1) * identity)
        alphas = torch.bmm(batch_K_inv, ys)

        return alphas

    def get_preds(self, alphas, phis_train, masks_train, phis_test, masks_test, task_phis=None):
        kernel_params = self.set_kernel_params(task_phis)
        l2 = kernel_params.pop('l2')

        k = compute_batch_gram_matrix(
            phis_test,
            phis_train,
            kernel=self.kernel,
            normalize=self.normalize_kernel,
            **kernel_params
        )
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

        phis_train = self.get_adapted_phis(task_phis, xs_train)
        phis_train, dropper = self.meta_dropout(phis_train, return_dropper=True)
        phis_test = self.get_adapted_phis(task_phis, xs_test)
        phis_test = self.meta_dropout(phis_test, dropper=dropper)

        # training
        alphas = self.get_alphas(phis_train, ys_train, mask_train, task_phis)

        # testing
        preds = self.get_preds(alphas, phis_train, mask_train, phis_test, mask_test, task_phis)

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

        if self.joint_training_mine:
            b = set_code.size(0)
            mi = set_code.diagonal().mean() \
                 - torch.log((set_code * (1 - torch.eye(b))).exp().sum() / (b * (b - 1)))
            loss = - mi
        else:
            loss = torch.nn.functional.cross_entropy(set_code, y_preds_class)

        self.accuracy = accuracy
        self.loss = loss


class MetaKrrMKLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0,
                 **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = MetaKrrMKNetwork(*args, **kwargs, device=device)

        super(MetaKrrMKLearner, self).__init__(network, optimizer, lr,
                                               weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()

        task_loss = self.model.loss

        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test)
                                       for y_pred, y_test in zip(y_preds, y_tests)]))

        r2 = np.mean([r2_score(pred.detach().cpu(), target.cpu()) for pred, target in zip(y_preds, y_tests)])

        res.update(dict(task_loss=task_loss))
        res.update(self.model.kernel_params)

        res.update(dict(mse=loss, task_encoder_accuracy=self.model.accuracy, r2=r2, ))

        if self.model.training:
            loss = loss + self.model.joint_training * task_loss

        return loss, res


if __name__ == '__main__':
    pass
