import torch
import torch.nn as nn

from adkl.feature_extraction import FeaturesExtractorFactory
from adkl.models.base import MetaLearnerRegression
from .utils import to_numpy, fit_and_score, cat_uneven_length_tensors


class MultiTaskNet(nn.Module):
    def __init__(self, feature_extractor_params, n_tasks, **kwargs):
        super(MultiTaskNet, self).__init__()
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        in_size = self.feature_extractor.output_dim
        self.top_layers = nn.ModuleList([nn.Linear(in_size, 1) for n in range(n_tasks)])

    def forward(self, input_x):
        xs, idx = zip(*input_x)
        lens = [x.shape[0] for x in xs]
        xs = cat_uneven_length_tensors(xs)
        phis_x = self.feature_extractor(xs)
        out = [self.top_layers[i](z) for z, i in zip(torch.split(phis_x, lens, dim=0), idx)]
        return out


class MultiTaskLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MultiTaskNet(*args, **kwargs)
        super(MultiTaskLearner, self).__init__(network, optimizer, lr, weight_decay)

    def _score_episode(self, episode, y_test, metrics, **kwargs):
        x_train, y_train = episode['Dtrain']
        x_test, _ = episode['Dtest']
        phi_train = to_numpy(self.model.feature_extractor(x_train))
        phi_test = to_numpy(self.model.feature_extractor(x_test))
        y_train, y_test = to_numpy(y_train), to_numpy(y_test)
        finals = dict()
        for algo in ['rf', 'krr']:
            res = fit_and_score(phi_train, y_train, phi_test, y_test, algo, metrics)
            finals.update({(algo + '_' + k): v for k, v in res.items()})
        return finals
