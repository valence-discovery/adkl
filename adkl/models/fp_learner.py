# -*- coding: utf-8 -*-
import sys
import warnings

import numpy as np
from torch.nn.functional import mse_loss

from adkl.feature_extraction import FingerprintsTransformer
from .base import MetaLearnerRegression
from .utils import fit_and_score

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def transform_and_filter(x, y, fp):
    transformer = FingerprintsTransformer(kind=fp)
    x, idx = transformer(x, dtype=np.float64)
    return x, y[idx]


class FPLearner(MetaLearnerRegression):
    def __init__(self, fp, *args, **kwargs):
        self.fp = fp

    def fit(self, *args, **kwargs):
        pass

    def _score_episode(self, episode, y_test, metrics, **kwargs):
        x_train, y_train = transform_and_filter(*episode['Dtrain'], self.fp)
        x_test, y_test = transform_and_filter(episode['Dtest'][0], y_test, self.fp)
        finals = dict()
        for algo in ['rf', 'krr']:
            res = fit_and_score(x_train, y_train, x_test, y_test,
                                algo, metrics)
            finals.update({(algo + '_' + k): v for k, v in res.items()})
        return finals

    def evaluate(self, metatest, metrics=[mse_loss], **kwargs):
        metatest.dataset.raw_inputs = True
        return super(FPLearner, self).evaluate(metatest, metrics)
