from abc import ABC
from collections import OrderedDict

import numpy as np
import torch
from poutyne.framework import Callback
from torch import nn
from torch import autograd
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.nn import Linear, Sequential
from torch.nn.functional import mse_loss

from adkl.feature_extraction import FeaturesExtractorFactory
from adkl.feature_extraction.utils import ClonableModule
from adkl.models.base import MetaLearnerRegression, MetaNetwork
from adkl.models.utils import to_unit
from .utils import set_params, vector_to_named_parameters,named_parameter_sizes


def check_nan(grads):
    for grad in grads:
        if torch.isnan(grad).any():
            return True

    return False


def log_pdf(y, mu, kappa):
    """
    Computes the log PDF of a spherical Gaussian.

    Parameters
    ----------
    y: torch.Tensor
        The sample.
    mu: torch.Tensor
        The mean.
    kappa: torch.Tensor
        The scale parameter (same parameterisation as BMAML paper).

    Returns
    -------
        The log PDF of a spherical Gaussian opf mean mu and scale parameter kappa.
    """

    log_p = Normal(mu.view(-1), 1 / kappa.pow(.5)).log_prob(y.view(-1)).sum()
    return log_p


class Regressor(ClonableModule, ABC):
    def __init__(self, feature_extractor_params, output_dim=1,
                 a_likelihood=2., b_likelihood=.2, a_prior=2., b_prior=.2):
        super(Regressor, self).__init__()
        self._params = locals()
        feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.out_dim = output_dim
        self.net = Sequential(feature_extractor, Linear(feature_extractor.output_dim, output_dim))

        self.gamma_likelihood = Gamma(a_likelihood, b_likelihood)
        self.gamma_prior = Gamma(a_prior, b_prior)

        # We sample initial values for kappa_prior and kappa_likelihood
        self.kappa_likelihood = torch.nn.Parameter(self.gamma_likelihood.rsample((1,)))
        self.kappa_prior = torch.nn.Parameter(self.gamma_prior.rsample((1,)))

        # # We need to register them in the autograd graph
        # self.kappa_likelihood.requires_grad_(True)
        # self.kappa_prior.requires_grad_(True)

    def forward(self, x):
        return self.net(x)


class MAMLNetwork(MetaNetwork, ABC):

    def __init__(self, feature_extractor_params, loss=mse_loss, lr_chaser=0.02, lr_leader=None,
                 a_likelihood=2., b_likelihood=.2, a_prior=2., b_prior=.2):
        super(MAMLNetwork, self).__init__()
        # feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)

        if lr_leader is None:
            lr_leader = lr_chaser / 10

        self.lr_chaser = lr_chaser
        self.lr_leader = lr_leader

        self.base_learner = Regressor(feature_extractor_params, 1, a_likelihood=a_likelihood,
                                      b_likelihood=b_likelihood, a_prior=a_prior, b_prior=b_prior)

        self.loss = loss

    def step(self, dataset):
        """
        Performs a single forward step.
        (we cannot compute the update yet, since it depends on other particles).

        This method returns either the log-posterior of the model, or the output.

        Parameters
        ----------
        dataset: tuple(torch.Tensor)
            The data for an episode (task) sampled from the task distribution.

        Returns
        -------
        output: torch.Tensor
            The output for the given episode.
        """
        x, y = dataset

        output = self.base_learner(x)

        return output

    def posterior(self, dataset):
        r"""
        Given an output, returns the log-posterior of the model :

        .. math::

            p(\theta_\tau | D^{train}_\tau) & \propto p(D^{train}_\tau | \theta_\tau) p(\theta_\tau)\\
            & = \prod_{(x, y) \in D^{train}_\tau} \mathcal{N}(y | f_\theta(x), \kappa_{l})
            \prod_{w \in \theta_\tau} \mathcal{N}(w | 0, \kappa_{p})
            \mathrm{Gamma}(\kappa_{l} | a, b) \mathrm{Gamma}(\kappa_{p} | a', b')

        Where :math:`\kappa_l` and :math:`\kappa_p` are scaling parameters.

        Parameters
        ----------
        dataset: torch.Tensor
            The dataset on which to compute the posterior.

        Returns
        -------
        objective: torch.Tensor
            The log-posterior of the model, computed for a given output.
        """

        x, y = dataset
        network = self.base_learner

        # We first compute the output
        output = self.step(dataset)

        # We need access to the parameters to enforce a Gaussian prior
        parameters = [
            param for name, param in network.named_parameters()
            if name not in {'kappa_likelihood', 'kappa_prior'}
        ]
        parameter_vector = torch.nn.utils.parameters_to_vector(parameters)

        # kappa is a scaling parameter
        log_likelihood = log_pdf(output, y, network.kappa_likelihood) + \
            network.gamma_likelihood.log_prob(network.kappa_likelihood)

        # We enforce a Gaussian prior
        log_prior = log_pdf(parameter_vector, torch.zeros_like(parameter_vector), network.kappa_prior) + \
            network.gamma_prior.log_prob(network.kappa_prior)

        objective = log_likelihood + log_prior

        return objective


class MAMLParticles(MetaNetwork):
    """
    Object that contains all the particles.
    """

    def __init__(self, feature_extractor_params, loss=mse_loss, lr_chaser=0.001, lr_leader=None,
                 n_epochs_chaser=1, n_epochs_predict=5, s_epochs_leader=1, m_particles=2, kernel_function='rbf'):
        """
        Initialises the object.

        Parameters
        ----------
        feature_extractor_params
        loss
        lr_chaser
        lr_leader
        n_epochs_chaser
        s_epochs_leader
        m_particles
        kernel_function
        """

        super(MAMLParticles, self).__init__()

        self.kernel_function = kernel_function

        self.n_epochs_chaser = n_epochs_chaser
        self.s_epochs_leader = s_epochs_leader

        self.n_epochs_predict = n_epochs_predict

        if lr_leader is None:
            lr_leader = lr_chaser / 10

        self.lr = {
            'chaser': lr_chaser,
            'leader': lr_leader,
        }

        self.m_particles = m_particles

        particles = [
            MAMLNetwork(feature_extractor_params, loss=loss, lr_chaser=lr_chaser, lr_leader=lr_leader)
            for _ in range(m_particles)
        ]

        self.particles = nn.Parameter(torch.stack(tuple([
            torch.nn.utils.parameters_to_vector(particle.base_learner.parameters())
            for particle in particles
        ])))

        self.networks = {
            'prototype': MAMLNetwork(feature_extractor_params, loss=loss, lr_chaser=lr_chaser, lr_leader=lr_leader)
        }

        # Since 'n' > 'k', the kappas are always first in the
        self.named_sizes = named_parameter_sizes(
            self.networks['prototype']
        )

    def get_model(self, parameters):
        """
        Puts the parameters in the prototype empty shell.

        Parameters
        ----------
        parameters: torch.Tensor
            The parameters of a given particle.

        Returns
        -------
        network: torch.nn.Module
            The prototype attribute, filled with the new parameters.
        """

        # Check that the kappas are more than 0, for numerical stability.
        if (parameters[:2] < 0).any():
            parameters = torch.cat(
                torch.clamp(parameters[:2], min=1e-8),
                parameters[2:]
            )

        named_parameters = vector_to_named_parameters(
            vec=parameters,
            named_sizes=self.named_sizes
        )

        network = self.networks['prototype']
        set_params(network, named_parameters)

        return network

    def kernel(self, parameter_vectors):
        """
        Computes the cross-particle kernel. Given the stacked parameter vectors of the particles,
        outputs the kernel (be it RBF or quadratic).

        Parameters
        ----------
        parameter_vectors: torch.Tensor
            m x n tensor representing the full parametrisation of the particles.

        Returns
        -------
        kernel: torch.Tensor
            m x m tensor representing the cross-particle kernel.
        """

        def rbf_kernel(pv):
            """
            Computes the RBF kernel for a set of parameter vectors.

            Parameters
            ----------
            pv: torch.Tensor
                Stack of flatten parameters for each particle.

            Returns
            -------
            kernel: m x m torch.Tensor
                A m x m torch tensor representing the kernel.
            """

            x = pv - pv.transpose(0, 1)
            x = - x.norm(2, dim=2).pow(2) / 2
            x = x.exp()

            return x

        def quadratic_kernel(pv):
            """
            Computes the RBF kernel for a set of parameter vectors.

            Parameters
            ----------
            pv: torch.Tensor
                Stack of flatten parameters for each particle.

            Returns
            -------
            kernel: m x m torch.Tensor
                A m x m torch tensor representing the kernel.
            """

            x = pv - pv.transpose(0, 1)
            x = 1 + x.norm(2, dim=2).pow(2)
            x = 1 / x

            return x

        kernel_functions = {
            'rbf': rbf_kernel,
            'quadratic': quadratic_kernel
        }

        kernel = kernel_functions[self.kernel_function]

        return kernel(parameter_vectors)

    def svgd(self, dataset, parameters, update_type='chaser'):
        r"""
        Performs the Stein Variational Gradient Update on the particles.

        For each particle, the update is given by
        :math:`\theta_{t+1} \gets \theta_t + \varepsilon_t \phi(\theta_t)` where:
        .. math::

            \phi(\theta_t) = \frac{1}{M} \sum_{m=1}^M \left[ k(\theta_t^{(m)}, \theta_t)
            \nabla_{\theta_t^{(m)}} \log p(\theta_t^{(m)}) +
            \nabla_{\theta_t^{(m)}} k(\theta_t^{(m)}, \theta_t) \right]

        Parameters
        ----------
        dataset: tuple
            A tuple containing the dataset (data, labels).
        parameters: torch.Tensor
            m_particles * nb_parameters tensor containing the full parameters.
        update_type: str, 'chaser' or 'leader'
            Defines which learning rate to use.
        """

        expanded_parameters = parameters.unsqueeze(0).expand(self.m_particles, *parameters.size())

        kernel = self.kernel(expanded_parameters)

        objectives = torch.stack([
            self.get_model(parameter).posterior(dataset)
            for parameter in parameters
        ])

        objective_grads = autograd.grad(objectives.sum(), parameters, create_graph=True)[0]

        kernel_grads = autograd.grad(kernel.sum(), expanded_parameters, create_graph=True)[0]

        gradient = kernel @ objective_grads + kernel_grads.mean(dim=1)

        new_parameters = parameters + self.lr[update_type] * gradient

        # We need to make sure that the kappas remain in the right range for numerical stability
        new_parameters = torch.cat([
            torch.clamp(new_parameters[:, :2], min=1e-8),
            new_parameters[:, 2:]
        ], dim=1)

        return new_parameters

    def __forward(self, episode):

        d_train = episode['Dtrain']
        d_test = episode['Dtest']
        d_full = (
            torch.cat((d_train[0], d_test[0])),
            torch.cat((d_train[1], d_test[1])),
        )

        with autograd.enable_grad():
            # Initialise the chaser as a new tensor
            chaser = self.particles + 0.
            for i in range(self.n_epochs_chaser):
                chaser = self.svgd(d_train, parameters=chaser, update_type='chaser')

            leader = chaser + 0.
            for i in range(self.s_epochs_leader):
                leader = self.svgd(d_full, parameters=leader, update_type='leader')

        # Added stability (according to Taesup)
        loss = (leader.detach() - chaser)[:, 2:].pow(2).sum()

        # We compute the backward pass here to minimise the load on memory.
        if self.training:
            loss.backward()

        with autograd.enable_grad():
            for i in range(self.n_epochs_predict):
                chaser = self.svgd(d_train, parameters=chaser, update_type='chaser')

        test_prediction = torch.stack([
            self.get_model(parameter).step(d_test)
            for parameter in chaser
        ]).mean(dim=0)

        test_variance = torch.stack([
            self.get_model(parameter).step(d_test)
            for parameter in chaser
        ]).var(dim=0)

        return loss, test_prediction, test_variance

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]

    def predict(self, episode):

        d_train = episode['Dtrain']
        d_test = episode['Dtest']

        with autograd.enable_grad():
            # Initialise the chaser as a new tensor
            chaser = self.particles + 0.
            for i in range(self.n_epochs_predict):
                chaser = self.svgd(d_train, parameters=chaser, update_type='chaser')

        predictions = torch.stack([
            self.get_model(parameter).step(d_test)
            for parameter in chaser
        ])

        y_preds = predictions.mean(dim=0)
        y_var = predictions.var(dim=0)

        return y_preds, y_var


class BMAML(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MAMLParticles(*args, **kwargs)
        super(BMAML, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        losses = [y[0] for y in y_preds]
        predictions = [y[1] for y in y_preds]

        loss = torch.stack(tuple(losses)).sum()
        # mse = mse_loss(torch.cat(tuple(predictions)), torch.cat(tuple(y_tests)))

        mse = torch.stack([mse_loss(y_p, y_t) for y_p, y_t in zip(predictions, y_tests)]).mean()

        return loss, dict(chaser_loss=loss, mse=mse)

    def make_prediction(self, episodes):
        return [self.model.predict(episode) for episode in episodes]

    def _score_episode(self, episode, y_test, metrics, return_data=False, **kwargs):
        with torch.no_grad():
            x_train, y_train = episode['Dtrain']
            y_pred, _ = self.model.predict(episode)
        res = dict(test_size=y_pred.shape[0], train_size=y_train.shape[0])
        res.update({metric.__name__: to_unit(metric(y_pred, y_test)) for metric in metrics})

        if return_data:
            x_test, y_test = episode['Dtest']

            data = {
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test,
                'y_pred': y_pred
            }

            return res, data

        return res

    def _fit_batch(self, x, y, *, callback=Callback(), step=None, return_pred=False):
        """
        Since BMAML is so memory-intensive and the computations cannot be parallelised anyway,
        the backward steps are made in the forward to make sure that the buffers are finally freed.

        To keep the advantages of batch computations (ie stabilise the gradient), the optimiser
        is only called after the whole "batch".

        Parameters
        ----------
        x
        y
        callback
        step
        return_pred

        Returns
        -------

        """

        loss_tensor, metrics, pred_y = self._compute_loss_and_metrics(
            x, y, return_loss_tensor=True, return_pred=return_pred
        )
        # if hasattr(self, 'grad_flow'):
        #     self.grad_flow.on_backward_start(step, loss_tensor)f
        # loss_tensor.backward()

        callback.on_backward_end(step)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # # We need to make sure that the kappas remain in the right range for numerical stability
        # parameters = self.model.particles
        # new_parameters = torch.cat([
        #     torch.clamp(parameters[:, :2], min=1e-8),
        #     parameters[:, 2:]
        # ], dim=1)
        # self.model.particles = nn.Parameter(new_parameters)

        loss = float(loss_tensor)
        return loss, metrics, pred_y
