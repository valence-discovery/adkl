import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss, softplus, cross_entropy
from torch.nn import MSELoss, Parameter, ParameterDict, Sequential, ReLU
from torch.distributions.normal import Normal
from poutyne.framework import Model
from adkl.feature_extraction import FeaturesExtractorFactory, FcFeaturesExtractor
from .base import MetaLearnerRegression, MetaNetwork
from .utils import to_unit, pack_episodes, compute_cos_similarity
from torch import nn


class CNPNetwork(MetaNetwork):

    def __init__(self, feature_extractor_params, target_dim=1,
                 encoder_hidden_sizes=[128], decoder_hidden_sizes=[128], device=None):
        """
        In the constructor we instantiate an lstm module
        """
        super(CNPNetwork, self).__init__()
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        phi_dim = self.feature_extractor.output_dim
        self.encoder = FcFeaturesExtractor(input_size=(phi_dim + target_dim),
                                           hidden_sizes=encoder_hidden_sizes)

        self.decoder = nn.Sequential(
            FcFeaturesExtractor(input_size=self.encoder.output_dim + phi_dim,
                                hidden_sizes=decoder_hidden_sizes),
            nn.Linear(decoder_hidden_sizes[-1], target_dim * 2)
        )

    @property
    def return_var(self):
        return True

    def get_data_repr(self, xs, ys, masks):
        nbatches, nsamples, _ = xs.shape

        phis = self.feature_extractor(xs.reshape(-1, xs.shape[-1]))
        phis = (phis.t() * masks.view(-1)).t()

        phis_ys = torch.cat((phis, ys.reshape(-1, ys.shape[-1])), dim=1)

        encoded_phis_ys = self.encoder(phis_ys).reshape(nbatches, nsamples, -1)

        data_repr = torch.sum(encoded_phis_ys, dim=1, keepdim=True)
        data_repr = data_repr / masks.sum(dim=1, keepdim=True).unsqueeze(dim=2)
        return data_repr.squeeze(1)

    def get_preds(self, phis_task, xs, masks, lens, trim_ends=True):
        phis_task = phis_task.unsqueeze(1).expand(
            (*xs.shape[:2], phis_task.shape[-1])
        ).reshape(-1, phis_task.shape[-1])
        phis = self.feature_extractor(xs.reshape(-1, xs.shape[-1]))
        phis_c = torch.cat((phis, phis_task), dim=1)
        # Masking the phi
        phis_c = (phis_c.t() * masks.view(-1)).t()
        ys_pred = self.decoder(phis_c).reshape(*xs.shape[:2], -1)
        # Get the mean an the variance and bound the variance
        mu, log_sigma = ys_pred.chunk(2, dim=-1)
        sigma = 0.1 + (0.9 * softplus(log_sigma))
        res = [(m[:n], s[:n]) for n, m, s in zip(lens, mu, sigma)] if trim_ends else (mu, sigma)
        return res

    def compute_task_encoder_loss_last_batch(self, episodes):
        train, test = pack_episodes(episodes, return_ys_test=True,
                                    return_query=False)
        xs_train, ys_train, lens_train, mask_train = train
        xs_test, ys_test, lens_test, mask_test = test

        nbatches, nsamples, _ = xs_train.shape
        data_repr_train = self.get_data_repr(xs_train, ys_train, mask_train)
        data_repr_test = self.get_data_repr(xs_test, ys_test, mask_test)
        cosine_similarity = compute_cos_similarity(data_repr_train, data_repr_test)
        task_targets = torch.arange(nbatches)
        if xs_train.is_cuda:
            task_targets = task_targets.to('cuda')
        self.task_loss = cross_entropy(cosine_similarity, task_targets)
        self.task_accuracy = (cosine_similarity.argmax(dim=1) == task_targets).sum().item() / nbatches

    def forward(self, episodes, train=None, test=None, query=None, trim_ends=True):
        if episodes is not None:
            self.compute_task_encoder_loss_last_batch(episodes)
            train, test = pack_episodes(episodes, return_query=False)
            query = None
        else:
            assert (train is not None) and (test is not None)

        xs_train, ys_train, lens_train, mask_train = train
        xs_test, lens_test, mask_test = test

        nbatches, nsamples, _ = xs_train.shape
        data_repr_train = self.get_data_repr(xs_train, ys_train, mask_train)
        test_res = self.get_preds(data_repr_train, xs_test, mask_test, lens_test, trim_ends=trim_ends)
        if query is not None:
            xs_query, _, lens_query, mask_query = query
            query_res = self.get_preds(data_repr_train, xs_query, mask_query, lens_query, trim_ends=trim_ends)
            return test_res, query_res
        else:
            return test_res


def nll(y, mu, std):
    log_p = - Normal(mu.view(-1), std.view(-1)).log_prob(y.view(-1)).mean()
    return log_p


class CNPLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, dataenc_beta=1.0, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = CNPNetwork(*args, **kwargs)
        super(CNPLearner, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()
        mse = torch.mean(torch.stack([mse_loss(mu, y_test)
                                      for (mu, sigma), y_test in zip(y_preds, y_tests)]))
        loss = torch.mean(torch.stack([nll(y_test, mu, sigma)
                                       for (mu, sigma), y_test in zip(y_preds, y_tests)]))

        res.update(dict(
            mse=mse,
            marginal_likelihood=loss,
            task_loss=self.model.task_loss,
            task_encoder_accuracy=self.model.task_accuracy
        ))
        return loss, res

    # def _score_episode(self, episode, y_test, metrics, return_data=False, **kwargs):
    #     with torch.no_grad():
    #         x_train, y_train = episode['Dtrain']
    #         y_pred = self.model([episode])[0]
    #         if self.model.return_var:
    #             y_pred, y_var = y_pred
    #         else:
    #             y_pred = y_pred
    #     res = dict(test_size=y_pred.shape[0], train_size=y_train.shape[0])
    #     res.update({metric.__name__: to_unit(metric(y_pred, y_test)) for metric in metrics})

    #     if return_data:
    #         x_test, y_test = episode['Dtest']

    #         data = {
    #             'x_train': x_train,
    #             'y_train': y_train,
    #             'x_test': x_test,
    #             'y_test': y_test,
    #             'y_pred': y_pred,
    #             'y_var': y_var
    #         }

    #         return res, data

    #     return res


if __name__ == '__main__':
    pass
