import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.nn.functional import mse_loss

from adkl.feature_extraction import FeaturesExtractorFactory
from adkl.models.base import MetaLearnerRegression, MetaNetwork
from .attention import SelfAttentionBlock
from .utils import pack_episodes


class WeightsGenerator(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=(128, 64), n_attentions=3):
        super(WeightsGenerator, self).__init__()

        attention_dim = hidden_dims[-1]

        self.network = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            *[
                SelfAttentionBlock(
                    input_dim=attention_dim,
                    hidden_dims=hidden_dims,
                )
                for _ in range(n_attentions)
            ]
        )

        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(attention_dim, output_dim),
        )

    def forward(self, x):
        x = self.network(x)
        x = self.output_layer(x)

        return x


class TaskLabelGenerator(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, n_attentions):
        super(TaskLabelGenerator, self).__init__()

        attention_dim = hidden_dims[-1]

        self.network = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            *[
                SelfAttentionBlock(
                    input_dim=attention_dim,
                    hidden_dims=hidden_dims,
                )
                for _ in range(n_attentions)
            ]
        )

        self.output_layer = nn.Linear(attention_dim, output_dim)

    def forward(self, x):
        x = self.network(x)
        x = nn.functional.relu(x)

        x = self.output_layer(x)

        return x


class BasisFunctionNetwork(MetaNetwork):
    """
    Implementation of the paper
    [Few-shot Regression via Learned Basis Functions](https://openreview.net/pdf?id=r1ldYi9rOV)
    by Loo et al (2019).
    """

    def __init__(self, target_dim, task_encoding_dim, feature_extractor_params,
                 weights_generator_params, task_label_generator_params):
        super(BasisFunctionNetwork, self).__init__()

        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.df = self.feature_extractor.output_dim

        self.target_dim = target_dim

        self.task_label_generator = TaskLabelGenerator(
            input_dim=self.df + target_dim,
            output_dim=task_encoding_dim,
            **task_label_generator_params,
        )

        self.weights_generator = WeightsGenerator(
            input_dim=self.df + target_dim + task_encoding_dim,
            output_dim=self.df,
            **weights_generator_params,
        )

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
        results: list(tuple)
            A list of tuples containing the mean and standard deviation computed by the network
            for each episodes.
        """

        train, test = pack_episodes(episodes, return_ys_test=True, return_query=False)

        # x is B * N * Dx dimensional, y is B * N * Dy dimensional
        x_train, y_train, len_train, mask_train = train
        x_test, y_test, len_test, mask_test = test

        b, n, dx = x_train.size()

        # B * N * Df
        train_features = self.feature_extractor(x_train.reshape(-1, dx)).reshape(b, -1, self.df)
        test_features = self.feature_extractor(x_test.reshape(-1, dx)).reshape(b, -1, self.df)

        # B * N * (Df + Dy) -> B * 1 * Dt
        x = torch.cat((train_features, y_train), dim=2)
        task_labels = self.task_label_generator(x)
        mask = mask_train.unsqueeze(2)
        task_labels = (task_labels * mask).sum(dim=1) / mask.sum(dim=1)  # Removing dummy examples
        task_labels = task_labels.unsqueeze(1).expand(b, n, task_labels.size(1))

        phis = torch.cat((train_features, task_labels), dim=2)

        # B * N * (D + Dy + Dt) -> B * Dy * Df
        x = torch.stack(
            [
                torch.cat((phis, y_train[..., i:i + 1]), dim=2)
                for i in range(y_train.size(-1))
            ],
            dim=2
        )
        x = x.reshape(b, n * self.target_dim, -1)
        weights = self.weights_generator(x).reshape(b, n, self.df, self.target_dim)
        mask = mask_train.unsqueeze(2).unsqueeze(2)
        weights = (weights * mask).sum(dim=1) / mask.sum(dim=1)  # Removing dummy examples

        # B * N * Dy
        predictions = torch.matmul(test_features, weights)

        pred = [p[:n] for p, n in zip(predictions, len_test)]

        return pred


class LearnedBasisFunctions(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = BasisFunctionNetwork(*args, **kwargs)
        super(LearnedBasisFunctions, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        """
        Computes different metrics such as the loss (MSE) and the R2 score.

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
            (MSE and R2).
        """

        mse = torch.stack([mse_loss(pred, target) for pred, target in zip(y_preds, y_tests)]).mean()
        r2 = np.mean([r2_score(pred.detach().cpu(), target.cpu()) for pred, target in zip(y_preds, y_tests)])

        loss = mse
        metrics = dict(mse=mse, r2=r2)

        return loss, metrics
