import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss

from adkl.feature_extraction import FeaturesExtractorFactory
from .attention import SelfAttentionBlock, Attention, MultiHeadAttentionLayer
from .base import MetaLearnerRegression, MetaNetwork
from .utils import pack_episodes


def kl_divergence(p, q):
    r"""
    Pytorch does not implement the diagonal Gaussian out-of-the-box,
    but we can use Normal. However, that means we need to implement the KL divergence
    by hand.

    .. math::

        D_\text{KL} (\mathcal{N}_0 \parallel \mathcal{N}_1) =
        \frac{1}{2}\left( \operatorname{tr} \left(\Sigma_1^{-1}\Sigma_0\right)
        + (\mu_1 - \mu_0)^\mathsf{T} \Sigma_1^{-1}(\mu_1 - \mu_0) - k
        + \ln \left(\frac{\det\Sigma_1}{\det\Sigma_0}\right) \right).

    Parameters
    ----------
    p: torch.distributions.normal.Normal
        Normal distribution from Pytorch, used as a diagonal Gaussian distribution.
    q: torch.distributions.normal.Normal
        Normal distribution from Pytorch, used as a diagonal Gaussian distribution.

    Returns
    -------
    kl: torch.Tensor
        The KL divergence between p and q.
    """

    mean_p, std_p = p.mean, p.scale
    mean_q, std_q = q.mean, q.scale

    var_p = std_p.pow(2)
    var_q = std_q.pow(2)

    d = mean_q - mean_p

    b, k = mean_p.size()

    kl = (var_p / var_q).sum(dim=1) + (d / std_p).norm(2, dim=1)
    kl = kl - k + var_q.log().sum(dim=1) - var_p.log().sum(dim=1)

    kl = kl / 2

    return kl


class DeterministicEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims):
        super(DeterministicEncoder, self).__init__()

        self.attention = SelfAttentionBlock(input_dim, hidden_dims)
        self.cross_attention = Attention()

        # self.mlp = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims[-1], output_dim)
        # )

        self.output_dim = hidden_dims[-1]

    def forward(self, train_features, y_train, test_features, mask_train):
        """
        Computes the deterministic encoding for the targets.

        Parameters
        ----------
        train_features: torch.Tensor
            B * N * Df tensor representing the features for the context.
        y_train: torch.Tensor
            B * N * 1/Dy tensor representing the targets for the context.
        test_features: torch.Tensor
            B * N * Df tensor representing the features for the test.
        mask_train: torch.Tensor
            B * N tensor masking the incomplete sets.

        Returns
        -------
        r: torch.Tensor
            The encoding of queries.
        """

        mask_train = mask_train.unsqueeze(2)

        phi_train = torch.cat((train_features, y_train), dim=2)

        context_r = self.attention(phi_train, mask_train)

        # Applies the mask
        context_r = context_r * mask_train

        r = self.cross_attention(train_features, test_features, context_r, mask_train)

        return r


class LatentEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(LatentEncoder, self).__init__()

        self.attention = SelfAttentionBlock(input_dim, hidden_dims)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], latent_dim * 2)
        )

        self.latent_dim = latent_dim

    def forward(self, train_features, y_train, mask_train):
        """
        Computes the deterministic encoding for the targets.

        Parameters
        ----------
        train_features: torch.Tensor
            B * N * Df tensor representing the features for the context.
        y_train: torch.Tensor
            B * N * 1/Dy tensor representing the targets for the context.
        mask_train: torch.Tensor
            B * N tensor masking the incomplete sets.

        Returns
        -------
        s: torch.Tensor
            The latent encoding of the tasks.
        """

        mask_train = mask_train.unsqueeze(2)

        phi_train = torch.cat((train_features, y_train), dim=2)

        s = self.attention(phi_train, mask_train)
        s = self.mlp(s)

        # Applies the mask
        s = s * mask_train

        s = s.sum(dim=1) / mask_train.sum(dim=1)

        mean, logvar = s.chunk(chunks=2, dim=1)
        std = (.5 * logvar).exp()

        distribution = Normal(mean, std)

        return distribution


class Decoder(nn.Module):
    """
    Decoder module in the ANP implementation.
    """

    def __init__(self, input_dim, hidden_dims, target_dim):
        """
        Initialises the object.

        Parameters
        ----------
        input_dim: int
            The dimension of the input.
        hidden_dims: iterable(int)
            An iterable containing the hidden dimensions.
        target_dim: int
            The last dimension, corresponding to the
        """
        super(Decoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
            ),
            *[
                nn.Sequential(
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                )
                for h1, h2 in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
            nn.Linear(hidden_dims[-1], 2 * target_dim)
        )

        self.target_dim = target_dim

    def forward(self, features, deterministic_encoding, latent_encoding):
        """
        Computes the deterministic encoding for the targets.

        Parameters
        ----------
        features: torch.Tensor
            B * N * Df tensor representing the features.
        deterministic_encoding: torch.Tensor
            B * N * Dr tensor representing the deterministic encoding.
        latent_encoding: torch.Tensor
            B * N * Dz tensor representing the latent encoding.
        """

        phi = torch.cat((features, deterministic_encoding, latent_encoding), dim=2)

        x = self.mlp(phi)

        mean, logvar = x.chunk(chunks=2, dim=2)

        std = (.5 * logvar).exp()

        return mean, std


class ANPNetwork(MetaNetwork):

    def __init__(self, feature_extractor_params, target_dim, deterministic_encoder_params,
                 latent_encoder_params, decoder_params):
        super(ANPNetwork, self).__init__()

        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.df = self.feature_extractor.output_dim

        # Encoders
        self.deterministic_encoder = DeterministicEncoder(
            input_dim=self.df + target_dim,
            **deterministic_encoder_params
        )
        self.latent_encoder = LatentEncoder(
            input_dim=self.df + target_dim,
            **latent_encoder_params
        )

        input_dim = self.df + self.deterministic_encoder.output_dim + self.latent_encoder.latent_dim

        # Decoder
        self.decoder = Decoder(input_dim=input_dim, target_dim=target_dim, **decoder_params)

        self.kl = None

    @property
    def return_var(self):
        return True

    def forward(self, episodes):
        """
        Performs a forward and backward pass on a single episode.
        To keep memory load low, the backward pass is done simultaneously.

        Parameters
        ----------
        episodes: list
            A batch of meta-learning episodes.

        Returns
        -------
        predictions: list(tuple(torch.Tensor))
            A list of tuples containing the mean and standard deviation computed by the network
            for each episodes.
        """

        train, test = pack_episodes(episodes, return_ys_test=True, return_query=False)

        # x is B * N * Dx dimensional, y is B * N * Dy dimensional
        x_train, y_train, len_train, mask_train = train
        x_test, y_test, len_test, mask_test = test

        b, n, dx = x_test.size()

        # B * N * Df
        train_features = self.feature_extractor(x_train.reshape(-1, dx)).reshape(b, -1, self.df)
        test_features = self.feature_extractor(x_test.reshape(-1, dx)).reshape(b, -1, self.df)

        # Applies the mask
        train_features = train_features * mask_train.unsqueeze(2)
        test_features = test_features * mask_test.unsqueeze(2)

        # Computes the deterministic encoding
        r = self.deterministic_encoder(train_features, y_train, test_features, mask_train)

        # Computes the latent encoding
        prior = self.latent_encoder(train_features, y_train, mask_train)

        if self.training:
            posterior = self.latent_encoder(test_features, y_test, mask_test)
            self.kl = kl_divergence(posterior, prior)

        z = prior.sample((n,)).transpose(0, 1)

        mean, std = self.decoder(test_features, r, z)

        predictions = [(m[:n], s[:n]) for m, s, n in zip(mean, std, len_test)]

        return predictions


class AttentiveNeuralProcess(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = ANPNetwork(*args, **kwargs)
        super(AttentiveNeuralProcess, self).__init__(network, optimizer, lr, weight_decay)

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

        nll = torch.stack([- Normal(m, s).log_prob(t).sum() for m, s, t in zip(mean, std, y_tests)]).mean()

        if self.model.training:
            kl = self.model.kl.mean()
            loss = nll + kl
            metrics = dict(loss=loss, nll=nll, mse=mse, r2=r2, kl=kl)
        else:
            loss = mse
            metrics = dict(nll=nll, mse=mse, r2=r2)

        return loss, metrics
