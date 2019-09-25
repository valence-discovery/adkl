import torch
from torch.nn import LSTMCell
from torch.nn.functional import mse_loss

from adkl.feature_extraction import FeaturesExtractorFactory
from adkl.models.base import MetaLearnerRegression, MetaNetwork
from .utils import pack_episodes


def compute_attention(query, support, apply_on=None):
    norms_query = torch.norm(query, p=None, dim=2)
    norms_support = torch.norm(support, p=None, dim=2)
    num = torch.bmm(query, support.transpose(1, 2))
    deno = (norms_query.unsqueeze(2) + norms_support.unsqueeze(1) + torch.FloatTensor([1e-8]))
    cos_sim = num / deno
    attention = cos_sim / cos_sim.sum(dim=2, keepdim=True)
    if apply_on is not None:
        return attention, torch.bmm(attention, apply_on)
    else:
        return attention, torch.bmm(attention, support)


class IrlMachtingNet(MetaNetwork):

    def __init__(self, feature_extractor_params, nb_ref_iterations, device='cuda'):
        """
        In the constructor we instantiate an lstm module
        """
        super(IrlMachtingNet, self).__init__()
        self.nb_ref_iterations = nb_ref_iterations
        self.f_net = FeaturesExtractorFactory()(**feature_extractor_params)
        self.g_net = FeaturesExtractorFactory()(**feature_extractor_params)
        dim = self.f_net.output_dim
        self.lstm_query = LSTMCell(dim, dim)
        self.lstm_support = LSTMCell(dim, dim)

    def iterative_refinement(self, phis_query, phis_support):
        q_shape, s_shape = phis_query.shape, phis_support.shape
        d = q_shape[-1]
        hiddens_query = torch.zeros_like(phis_query)
        cells_query = torch.zeros_like(phis_query)
        hiddens_support = torch.zeros_like(phis_support)
        cells_support = torch.zeros_like(phis_support)

        r_support = phis_support + hiddens_support

        for _ in range(self.nb_ref_iterations):
            _, r_query = compute_attention(phis_query + hiddens_query.view(q_shape), r_support)
            hiddens_query, cells_query = self.lstm_query(r_query.view(-1, d),
                                                         (hiddens_query.view(-1, d), cells_query.view(-1, d)))

            _, r_support = compute_attention(r_support + hiddens_support.view(s_shape), phis_support)
            hiddens_support, cells_support = self.lstm_support(r_support.view(-1, d),
                                                               (hiddens_support.view(-1, d), cells_support.view(-1, d)))

        return (phis_query + hiddens_query.reshape(q_shape),
                phis_support + hiddens_support.reshape(s_shape))

    def forward(self, episodes):
        train, test = pack_episodes(episodes)
        xs_train, ys_train, lens_train, mask_train = train
        xs_test, lens_test, mask_test = test
        phis_train = self.f_net(xs_train.view(-1, *xs_train.shape[2:])).view(*xs_train.shape[:2], -1)
        phis_test = self.g_net(xs_test.view(-1, *xs_test.shape[2:])).view(*xs_test.shape[:2], -1)
        phis_test, phis_train = self.iterative_refinement(phis_test, phis_train)
        _, ys_pred = compute_attention(phis_test, phis_train, apply_on=ys_train)

        return [preds[:n] for n, preds in zip(lens_test, ys_pred)]


class IrlMachtingNetLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = IrlMachtingNet(*args, **kwargs, device=device)
        super(IrlMachtingNetLearner, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()
        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test)
                                       for y_pred, y_test in zip(y_preds, y_tests)]))
        res.update(dict(mse=loss))
        return loss, res


if __name__ == '__main__':
    pass
