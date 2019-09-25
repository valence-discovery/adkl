import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.nn.functional import mse_loss

from adkl.models.base import MetaLearnerRegression
from .light_bmaml import MAMLParticles


class ProtoMAML(MetaLearnerRegression):
    """
    ProtoMAML is a model described by [Triantafillou et al](https://arxiv.org/pdf/1903.03096.pdf).
    It supposed to get the advantages of proto-networks while allowing for inner update.

    It turns out that ProtoMAML is the same as the light version of BMAML with one particle.
    """
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, n_steps=1, **kwargs):

        # Deletes the unnecessary arguments
        kwargs.pop('m_particles', None)
        kwargs.pop('s_epochs_leader', None)
        kwargs.pop('n_epochs_chaser', None)

        network = MAMLParticles(*args, m_particles=1, s_epochs_leader=0, n_epochs_chaser=n_steps, **kwargs)
        super(ProtoMAML, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        """
        Computes different metrics such as the MSE loss and R2 score.

        Notes
        -----
        In the case of ProtoMAML and contrary to BMAML, we do not use the chaser loss.

        Parameters
        ----------
        y_preds: list(tuple)
            The predictions returned by the forward method of the model.
        y_tests: list(torch.Tensor)
            The targets associated with the prediction.

        Returns
        -------
        loss: torch.Tensor
            The MSE loss.
        metrics: dict
            A dictionary containing relevant metrics to be plotted
            (MSE and R2, as well as chaser loss during training).
        """

        mean, std = list(zip(*y_preds))

        mse = torch.stack([mse_loss(pred, target) for pred, target in zip(mean, y_tests)]).mean()
        r2 = np.mean([r2_score(pred.detach().cpu(), target.cpu()) for pred, target in zip(mean, y_tests)])

        loss = mse
        metrics = dict(mse=mse, r2=r2)

        return loss, metrics
