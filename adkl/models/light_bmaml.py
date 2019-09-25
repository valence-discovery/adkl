import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import autograd
from torch import nn
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss

from adkl.feature_extraction import FeaturesExtractorFactory
from adkl.models.base import MetaLearnerRegression, MetaNetwork
from .utils import pack_episodes


def check_nan(grads):
    for grad in grads:
        if torch.isnan(grad).any():
            return True

    return False


def log_pdf(y, kappa):
    """
    Computes the log PDF of a spherical Gaussian.

    Parameters
    ----------
    y: torch.Tensor
        B * M * x tensor, representing the centered sample.
        Depending on the use case, x is either N, the number of samples in the batch
        or (D + 1), the number of parameters.
    kappa: torch.Tensor
        B * M tensor, representing the scale parameter (same parametrisation as BMAML paper).

    Returns
    -------
    log_p: torch.Tensor
        B * M * x tensor (x = N or D + 1), the log PDF.
    """

    # Adapting the dimension of kappa
    kappa = kappa.unsqueeze(2)
    kappa = kappa.expand(y.shape)

    log_p = Normal(0, 1 / kappa.pow(.5)).log_prob(y)

    return log_p


class MAMLParticles(MetaNetwork):
    """
    Object that contains all the particles.
    """

    def __init__(self, feature_extractor_params, lr_chaser=0.001, lr_leader=None,
                 n_epochs_chaser=1, n_epochs_predict=0, s_epochs_leader=1,
                 m_particles=2, kernel_function='rbf', n_samples=10, a_likelihood=2., b_likelihood=.2,
                 a_prior=2., b_prior=.2, use_mse=False):
        """
        Initialises the object.

        Parameters
        ----------
        feature_extractor_params: dict
            Parameters for the feature extractor.
        lr_chaser: float
            Learning rate for the chaser
        lr_leader: float
            Learning rate for the leader
        n_epochs_chaser: int
            Number of steps to be performed by the chaser.
        s_epochs_leader: int
            Number of steps to be performed by the leader.
        m_particles:
            Number of particles.
        kernel_function: str, {'rbf', 'quadratic'}
            The kernel function to use.
        use_mse: bool
            Whether to use MSE loss or Chaser loss.
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

        self.n_samples = n_samples

        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.fe_output_dim = self.feature_extractor.output_dim

        self.gamma_likelihood = Gamma(a_likelihood, b_likelihood)
        self.gamma_prior = Gamma(a_prior, b_prior)

        # The particles only implement the last (linear) layer.
        # The first two columns are the kappas (likelihood then prior)
        self.particles = nn.Parameter(
            torch.cat((
                self.gamma_likelihood.sample((m_particles, 1)),
                self.gamma_prior.sample((m_particles, 1)),
                nn.init.kaiming_uniform(torch.empty((m_particles, self.fe_output_dim + 1))),
            ), dim=1)
        )

        self.loss = 0

        self.use_mse = use_mse

    @property
    def return_var(self):
        return True

    def kernel(self, weights):
        """
        Computes the cross-particle kernel. Given the stacked parameter vectors of the particles,
        outputs the kernel (be it RBF or quadratic).

        Parameters
        ----------
        weights: torch.Tensor
            B * M * M * (D + 1) tensor. Expanded versions of the weights.

        Returns
        -------
        kernel: torch.Tensor
            B * M * M tensor representing the cross-particle kernel.
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

            x = pv - pv.transpose(1, 2)
            x = - x.norm(2, dim=3).pow(2) / 2
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

            x = pv - pv.transpose(1, 2)
            x = - x.norm(2, dim=3).pow(2)
            x = 1 / x

            return x

        kernel_functions = {
            'rbf': rbf_kernel,
            'quadratic': quadratic_kernel
        }

        kernel = kernel_functions[self.kernel_function]

        return kernel(weights)

    @staticmethod
    def compute_predictions(features, parameters):
        """

        Parameters
        ----------
        features: torch.Tensor
            B * N * D tensor representing the features.
        parameters: torch.Tensor
            B * M * (D + 3) tensor representing the M particles
            (including the bias-feature trick and two kappa vectors).

        Returns
        -------
        predictions: torch.Tensor
            B * M * N tensor, representing the predictions.
        """
        # Obtains the weights
        weights = parameters[..., 2:]

        # Implements the bias-feature trick
        features = torch.cat((features, torch.ones_like(features[..., :1])), dim=2)

        predictions = torch.bmm(weights, features.transpose(1, 2))

        return predictions

    def compute_mean_std(self, features, parameters):
        """

        Parameters
        ----------
        features: torch.Tensor
            B * N * D tensor representing the features.
        parameters: torch.Tensor
            B * M * (D + 3) tensor representing the M particles
            (including the bias-feature trick and two kappa vectors).

        Returns
        -------
        predictions: torch.Tensor
            B * M * N tensor, representing the predictions.
        """
        # Obtains the kappas (B * M)
        kappa_likelihood = parameters[..., 0]

        # Computes the predictions (B * M * N)
        predictions = self.compute_predictions(features, parameters)

        # Transposes the predictions to B * N * M
        predictions = predictions.transpose(1, 2)

        # Computes the mean
        mean = predictions.mean(dim=2)

        # Adds the variability
        variability = torch.randn((*predictions.size(), self.n_samples)).to(mean.device)
        variability = variability / kappa_likelihood.unsqueeze(1).unsqueeze(3).pow(.5)
        predictions = predictions.unsqueeze(3) + variability

        # Reshapes the predictions to B * N * (M x S), where S is the number of samples
        predictions = predictions.view(*predictions.shape[:2], -1)

        # mean = predictions.mean(dim=2)
        std = predictions.std(dim=2)

        return mean, std

    def posterior(self, predictions, targets, mask, weights, kappa_likelihood, kappa_prior):
        r"""
        Computes the posterior of the configuration.

        Parameters
        ----------
        predictions: torch.Tensor
            B * M * N tensor representing the prediction made by the network.
        targets: torch.Tensor
            B * N * 1 tensor representing the targets.
        mask: torch.Tensor
            B * N mask of the examples (some tasks have less than N examples).
        weights: torch.Tensor
            B * M * (D + 1) tensor representing the weights, including the bias-feature trick
        kappa_likelihood: torch.Tensor:
            B * M tensor representing $\kappa_{likelihood}$.
        kappa_prior: torch.Tensor:
            B * M tensor representing $\kappa_{prior}$.

        Returns
        -------
        objective: torch.Tensor
            B * M tensor, representing the posterior of each particle, for each batch.
        """
        # Computing the log-likelihood
        log_likelihood = log_pdf(predictions - targets.transpose(1, 2), kappa_likelihood)  # B * M * N
        log_likelihood = log_likelihood * mask.unsqueeze(1)  # Keep only the actual examples
        log_likelihood = log_likelihood.sum(dim=2)

        # We enforce a Gaussian prior on the weights
        log_prior = log_pdf(weights[..., :-1], kappa_prior).sum(dim=2)

        # Gamma prior on the kappas
        log_prior_kappa = self.gamma_likelihood.log_prob(kappa_likelihood)
        log_prior_kappa = log_prior_kappa + self.gamma_prior.log_prob(kappa_prior)

        objective = log_likelihood + log_prior + log_prior_kappa

        return objective

    def svgd(self, features, targets, mask, parameters, update_type='chaser'):
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
        features: torch.Tensor
            B * N * D tensor. The precomputed features associated with the dataset.
        targets: torch.Tensor
            B * N * 1 tensor. The targets associated to the features. Useful to compute the posterior.
        mask: torch.Tensor
            B * N mask of the examples (some tasks have less than N examples).
        parameters: torch.Tensor
            B * M * (D + 3) tensor containing the full parameters, already expanded along a batch dimension.
        update_type: str, 'chaser' or 'leader'
            Defines which learning rate to use.
        """

        # Expands the parameters : B * M * (D + 3) -> B * M * M * (D + 3)
        expanded_parameters = parameters.unsqueeze(1)
        expanded_parameters = expanded_parameters.expand(
            (parameters.size(0), self.m_particles, *parameters.shape[1:])
        )

        # Splits the different parameters
        kappa_likelihood = parameters[..., 0]
        kappa_prior = parameters[..., 1]
        weights = parameters[..., 2:]

        expanded_weights = expanded_parameters[..., 2:]

        # weights is B * M * (D + 1), features is B * N * D
        # predictions is B * M * N
        predictions = self.compute_predictions(features, parameters)

        # B * M * M
        kernel = self.kernel(expanded_weights)

        # B * M
        objectives = self.posterior(
            predictions=predictions,
            targets=targets,
            mask=mask,
            weights=weights,
            kappa_likelihood=kappa_likelihood,
            kappa_prior=kappa_prior,
        )

        # Computes the gradients for the objective (B * M * (D + 3))
        objective_grads = autograd.grad(objectives.sum(), parameters, create_graph=True)[0]

        # Computes the gradients for the kernel, using the expanded parameters (B * M * M * (D + 3))
        kernel_grads = autograd.grad(kernel.sum(), expanded_parameters, create_graph=True)[0]

        # Computes the update
        # The matmul term multiplies batches of matrices that are B * M * M and B * M * (D + 3)
        update = torch.matmul(kernel, objective_grads) / self.m_particles + kernel_grads.mean(dim=2)

        # Performs the update
        new_parameters = parameters + self.lr[update_type] * update

        # We need to make sure that the kappas remain in the right range for numerical stability
        new_parameters = torch.cat([
            torch.clamp(new_parameters[..., :2], min=1e-8),
            new_parameters[..., 2:]
        ], dim=2)

        return new_parameters

    def forward(self, episodes, train=None, test=None, query=None, trim_ends=True):
        """
        Performs a forward and backward pass on a single episode.
        To keep memory load low, the backward pass is done simultaneously.

        Parameters
        ----------
        episodes: list
            A batch of meta-learning episodes.
        train: dataset
            The train dataset.
        test: dataset
            The test dataset.
        query: dataset
            The query dataset.
        trim_ends: bool
            Whether to trim the results.

        Returns
        -------
        results: list(tuple)
            A list of tuples containing the mean and standard deviation computed by the network
            for each episodes.
        query_results: list(tuple)
            A list of tuples containing the mean and standard deviation computed by the network
            for each episodes of the query set.
        """

        if episodes is not None:
            train, test = pack_episodes(episodes, return_ys_test=True,
                                        return_query=False)
            x_test, y_test, len_test, mask_test = test
            query = None
        else:
            assert (train is not None) and (test is not None)
            x_test, len_test, mask_test = test

        # x is B * N * D dimensional, y is B * N * 1 dimensional
        x_train, y_train, len_train, mask_train = train

        b, n, d = x_train.size()

        train_features = self.feature_extractor(x_train.reshape(-1, d)).reshape(b, -1, self.fe_output_dim)
        test_features = self.feature_extractor(x_test.reshape(-1, d)).reshape(b, -1, self.fe_output_dim)

        # Expands the parameters along the batch dimension : M * (D + 3) -> B * M * (D + 3)
        parameters = self.particles.unsqueeze(0).expand((b, *self.particles.size()))

        with autograd.enable_grad():
            # Initialise the chaser as a new tensor
            chaser = parameters + 0.
            for i in range(self.n_epochs_chaser):
                chaser = self.svgd(train_features, y_train, mask_train, parameters=chaser, update_type='chaser')

            if self.training and not self.use_mse:
                full_features = torch.cat((train_features, test_features), dim=1)
                y_full = torch.cat((y_train, y_test), dim=1)
                mask_full = torch.cat((mask_train, mask_test), dim=1)

                leader = chaser + 0.
                for i in range(self.s_epochs_leader):
                    leader = self.svgd(full_features, y_full, mask_full, parameters=leader, update_type='leader')

                # Added stability
                self.loss = (leader.detach() - chaser)[..., 2:].pow(2).sum() / b

        with autograd.enable_grad():
            for i in range(self.n_epochs_predict):
                chaser = self.svgd(train_features, y_train, mask_train, parameters=chaser, update_type='chaser')

        # Computes the mean and standard deviation
        mean, std = self.compute_mean_std(test_features, chaser)

        # Unsqueezes the results to keep the same shape as the targets
        mean = mean.unsqueeze(2)
        std = std.unsqueeze(2)

        # Re-organises the results in the episodic form
        mean = [m[:n] for m, n in zip(mean, len_test)]
        std = [s[:n] for s, n in zip(std, len_test)]

        results = [(m[:n], s[:n]) for m, s, n in zip(mean, std, len_test)] if trim_ends else (mean, std)

        if query is None:
            return results

        x_query, _, len_query, mask_query = query

        query_features = self.feature_extractor(x_query.reshape(-1, d)).reshape(b, -1, self.fe_output_dim)

        mean, std = self.compute_mean_std(query_features, chaser)

        # Unsqueezes the results to keep the same shape as the targets
        mean = mean.unsqueeze(2)
        std = std.unsqueeze(2)

        query_results = [(m[:n], s[:n]) for m, s, n in zip(mean, std, len_test)] if trim_ends else (mean, std)

        return results, query_results


class LightBMAML(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MAMLParticles(*args, **kwargs)
        super(LightBMAML, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        """
        Computes different metrics such as the loss and the MSE.

        Notes
        -----
        In the case of BMAML, we need to do early stopping based on the MSE loss
        and not the chaser loss (which is not computed at evaluation time anyway).

        Parameters
        ----------
        y_preds: list(tuple)
            The predictions returned by the forward method of the model.
        y_tests: list(torch.Tensor)
            The targets associated with the prediction.

        Returns
        -------
        loss: torch.Tensor
            Depending on the mode (training or evaluation), either the chaser loss or the MSE loss.
        metrics: dict
            A dictionary containing relevant metrics to be plotted
            (MSE and R2, as well as chaser loss during training).
        """

        mean, std = list(zip(*y_preds))

        mse = torch.stack([mse_loss(pred, target) for pred, target in zip(mean, y_tests)]).mean()
        r2 = np.mean([r2_score(pred.detach().cpu(), target.cpu()) for pred, target in zip(mean, y_tests)])

        if self.model.training and not self.model.use_mse:
            loss = self.model.loss
            metrics = dict(chaser_loss=loss, mse=mse, r2=r2)
        else:
            # During evaluation, we want to return the MSE loss.
            loss = mse
            metrics = dict(mse=mse, r2=r2)

        return loss, metrics
