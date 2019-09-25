import torch
from torch.nn.functional import mse_loss
from torch.distributions.normal import Normal
from adkl.models.base import MetaLearnerRegression
from .utils import pack_episodes
from .kernels import compute_batch_gram_matrix
from .metakrr_mk import MetaKrrMKNetwork


class MetaGPMKNetwork(MetaKrrMKNetwork):
    def __init__(self, *args, std_annealing_steps=0, **kwargs):
        super(MetaGPMKNetwork, self).__init__(*args, **kwargs)
        # the number of steps for which the std is put small to put emphasize on the mean
        self.std_annealing_steps = std_annealing_steps
        self.step = 0

    @property
    def return_var(self):
        return True

    def get_alphas(self, phis, ys, masks, task_phis=None, return_K_inv=False):
        kernel_params = self.set_kernel_params(task_phis)
        l2 = kernel_params.pop('l2')
        bsize, n_train = phis.shape[:2]

        k = compute_batch_gram_matrix(
            phis,
            phis,
            kernel = self.kernel,
            normalize = self.normalize_kernel,
            **kernel_params

        )
        k_mask = masks[:, None, :] * masks[:, :, None]
        k = k * k_mask

        Identity = torch.eye(n_train, device=k.device).unsqueeze(0).expand((bsize, n_train, n_train))
        batch_K_inv = torch.inverse(k + l2.view(-1, 1, 1) * Identity)
        alphas = torch.bmm(batch_K_inv, ys)
        if return_K_inv:
            return alphas, batch_K_inv
        else:
            return alphas

    def get_preds(self, alphas, batch_K_inv, phis_train, masks_train, phis_test, masks_test, task_phis=None):
        kernel_params = self.set_kernel_params(task_phis)
        l2 = kernel_params.pop('l2')
        eps = 1e-6
        batch_K_cross = compute_batch_gram_matrix(phis_test, phis_train,
            kernel=self.kernel, normalize=self.normalize_kernel,
            **kernel_params) * (masks_test[:, :, None] * masks_train[:, None, :])
        batch_K_test = compute_batch_gram_matrix(phis_test, phis_test, diag_only=True,
            kernel=self.kernel, normalize=self.normalize_kernel,
            **kernel_params) * masks_test
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

        phis_train = self.get_adapted_phis(task_phis, xs_train)
        phis_train, dropper = self.meta_dropout(phis_train, return_dropper=True)
        phis_test = self.get_adapted_phis(task_phis, xs_test)
        phis_test = self.meta_dropout(phis_test, dropper=dropper)

        # training
        alphas, batch_K_inv = self.get_alphas(phis_train, ys_train, mask_train, task_phis, return_K_inv=True)

        # testing
        mus, stds = self.get_preds(alphas, batch_K_inv, phis_train, mask_train, phis_test, mask_test, task_phis)
        test_res = [(mu[:n], std[:n]) for n, mu, std in zip(lens_test, mus, stds)] if trim_ends else (mus, stds)
        if query is not None:
            xs_query, _, lens_query, mask_query = query
            phis_query = self.get_adapted_phis(task_phis, xs_query)
            phis_query = self.meta_dropout(phis_query, dropper=dropper)
            mus, stds = self.get_preds(alphas, batch_K_inv, phis_train, mask_train, phis_query, mask_query, task_phis)
            query_res = [(mu[:n], std[:n]) for n, mu, std in zip(lens_query, mus, stds)] if trim_ends else (mus, stds)
            return test_res, query_res
        else:
            return test_res


class MetaGPMKLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0,
                 **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = MetaGPMKNetwork(*args, **kwargs, device=device)
        super(MetaGPMKLearner, self).__init__(network, optimizer, lr,
                                              weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        def nll(mu, std, y):
            return - Normal(mu.view(-1), std.view(-1)).log_prob(y.view(-1)).sum()

        res = dict()
        loss_nll = torch.mean(torch.stack([nll(y_mean, y_std, y_test)
                                           for (y_mean, y_std), y_test in zip(y_preds, y_tests)]))
        mse = torch.mean(torch.stack([mse_loss(y_mean, y_test)
                                      for (y_mean, _), y_test in zip(y_preds, y_tests)]))

        # stds = torch.tensor([y_std.mean() for _, y_std in y_preds]).mean()

        res.update(dict(loss=loss_nll, mse=mse, task_encoder_accuracy=self.model.accuracy))
        res.update({k: self.model.kernel_params[k] for k in self.model.kernel_params})

        task_loss = self.model.loss

        res.update(dict(task_loss=task_loss))

        loss = loss_nll + self.model.joint_training * task_loss

        if torch.isnan(loss):
            nll = torch.stack([nll(y_mean, y_std, y_test)
                               for (y_mean, y_std), y_test in zip(y_preds, y_tests)])
            print('\nTask loss', task_loss)
            print('\nloss', loss)
            print('\ny_preds', y_preds)
            print('\ny_tests', y_tests)
            print('\nNll', nll)

            if not torch.isnan(loss_nll):
                loss = loss_nll
            else:
                loss = task_loss

        if not self.model.training:
            loss = mse

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
