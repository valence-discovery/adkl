from collections import OrderedDict

import torch
from torch.nn import Linear, Sequential
from torch.nn.functional import mse_loss

from adkl.feature_extraction import FeaturesExtractorFactory
from adkl.feature_extraction.utils import ClonableModule
from adkl.models.base import MetaLearnerRegression, MetaNetwork
from .utils import set_params


class Regressor(ClonableModule):
    def __init__(self, feature_extractor_params, output_dim=1):
        super(Regressor, self).__init__()
        self._params = locals()
        feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.out_dim = output_dim
        self.net = Sequential(feature_extractor, Linear(feature_extractor.output_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class MAMLNetwork(MetaNetwork):

    def __init__(self, feature_extractor_params, loss=mse_loss, lr_learner=0.02, n_epochs_learner=1):
        """
        In the constructor we instantiate an lstm module
        """
        super(MAMLNetwork, self).__init__()
        # feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.lr_learner = lr_learner

        self.base_learner = Regressor(feature_extractor_params, 1)

        self.n_epochs_learner = n_epochs_learner
        self.loss = loss

    def __forward(self, episode):
        with torch.enable_grad():
            x_train, y_train = episode['Dtrain']
            x_test, _ = episode['Dtest']
            # x_train.volatile = False

            learner_network = self.base_learner.clone()

            initial_params = OrderedDict((name, param - 0.0) for (name, param) in self.base_learner.named_parameters())
            set_params(learner_network, initial_params)

            for i in range(self.n_epochs_learner):
                # forward pass with x_train
                output = learner_network(x_train)
                # computation of the loss and the gradients wrt the parameters
                # with torch.enable_grad()
                loss = self.loss(output, y_train)
                grads = torch.autograd.grad(loss, learner_network.parameters(), create_graph=True)
                new_weights = OrderedDict((name, param - self.lr_learner * grad)
                                          for ((name, param), grad) in
                                          zip(learner_network.named_parameters(), grads))
                set_params(learner_network, new_weights)

            n = len(x_test)
            bsize = 10
            res = torch.cat([learner_network(x_test[i:i + bsize]) for i in range(0, n, bsize)])
            return res

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class MAML(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MAMLNetwork(*args, **kwargs)
        super(MAML, self).__init__(network, optimizer, lr, weight_decay)
