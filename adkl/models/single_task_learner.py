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


class STLearner(MetaLearnerRegression):
    def __init__(self, algo='gp', kernel='linear', *args, **kwargs):
        self.algo = algo
        self.kernel = kernel

    def fit(self, *args, **kwargs):
        pass

    def _score_episode(self, episode, y_test, metrics, **kwargs):
        x_train, y_train = episode['Dtrain']
        x_test = episode['Dtest'][0]
        res = fit_and_score(x_train, y_train, x_test, y_test,
                            self.algo, metrics, kernel=self.kernel)
        print(res)
        return res

    def evaluate(self, metatest, metrics=[mse_loss], **kwargs):
        metatest.dataset.raw_inputs = True
        return super(STLearner, self).evaluate(metatest, metrics)
