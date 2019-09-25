import torch
import torch.nn as nn

from .attention import StandardAttentionEncoder, MultiHeadAttentionEncoder


class DeepSetEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=0, num_layers=0, functions='meanstd'):
        super(DeepSetEncoder, self).__init__()
        assert functions in ['meanstd', 'stdmean', 'maxsum', 'summax']
        layers = []
        in_dim, out_dim = input_dim, hidden_dim
        for i in range(1, num_layers + 1):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim, out_dim = out_dim, out_dim
        self.phi_net = nn.Sequential(*layers)

        layers = []
        in_dim, out_dim = hidden_dim * 2, hidden_dim
        for i in range(1, num_layers + 1):
            layers.append(nn.Linear(in_dim, out_dim))
            # if i != num_layers:
            layers.append(nn.ELU())
            in_dim, out_dim = out_dim, out_dim
        self.rho_net = nn.Sequential(*layers)
        self.functions = functions
        self._output_dim = 2 * input_dim if num_layers <= 0 else hidden_dim

    def forward(self, x):
        phis_x = self.phi_net(x)
        if self.functions in ['meanstd', 'stdmean']:
            x1 = phis_x.mean(dim=1, keepdim=True)

            # WATCH OUT !! Using .std() leads to CATASTROPHIC numerical instabilities
            x2 = (phis_x.var(dim=1, keepdim=True) + 1e-8).sqrt()

        else:
            x1 = phis_x.sum(dim=1, keepdim=True)
            x2, _ = phis_x.max(dim=1, keepdim=True)
        z = torch.cat((x1, x2), dim=2)
        res = self.rho_net(z).squeeze(1)
        return res

    @property
    def output_dim(self):
        return self._output_dim


class CnpEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(CnpEncoder, self).__init__()
        self._output_dim = input_dim

    def forward(self, x):
        return x.mean(dim=1)

    @property
    def output_dim(self):
        return self._output_dim


class Set2SetEncoder(torch.nn.Module):
    r"""
    Set2Set global pooling operator from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper. This pooling layer performs the following operation

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Arguments
    ---------
        input_dim: int
            Size of each input sample.
        hidden_dim: int, optional
            the dim of set representation which corresponds to the input dim of the LSTM in Set2Set.
            This is typically the sum of the input dim and the lstm output dim.
            If not provided, it will be set to :obj:`input_dim*2`
        steps: int, optional
            Number of iterations :math:`T`. If not provided, the number of nodes will be used.
        num_layers : int, optional
            Number of recurrent layers (e.g., :obj:`num_layers=2` would mean stacking two LSTMs together)
            (Default, value = 1)
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Set2SetEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_lstm_dim = input_dim * 2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.hidden_lstm_dim, self.input_dim, num_layers=num_layers, batch_first=True)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(self.hidden_lstm_dim, self.hidden_dim)
        self._output_dim = hidden_dim

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        Arguments
        ----------
            x: torch.FloatTensor
                Input tensor of size (B, N, D)

        Returns
        -------
            x: `torch.FloatTensor`
                Tensor resulting from the  set2set pooling operation.
        """
        batch_size, n, _ = x.shape

        h = (x.new_zeros((self.num_layers, batch_size, self.input_dim)),
             x.new_zeros((self.num_layers, batch_size, self.input_dim)))

        q_star = x.new_zeros(batch_size, 1, self.hidden_lstm_dim)

        for i in range(n):
            # q: batch_size x 1 x input_dim
            q, h = self.lstm(q_star, h)
            # e: batch_size x n x 1
            e = torch.matmul(x, q.transpose(1, 2))
            a = self.softmax(e)
            r = torch.sum(a * x, dim=1, keepdim=True)
            q_star = torch.cat([q, r], dim=-1)

        return self.linear(torch.squeeze(q_star, dim=1))

    @property
    def output_dim(self):
        return self._output_dim


class RelationNetEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=20, num_layers=1):
        super(RelationNetEncoder, self).__init__()
        in_dim, out_dim = input_dim * 2, hidden_dim
        self.net = DeepSetEncoder(in_dim, out_dim, num_layers)
        self._output_dim = hidden_dim

    def forward(self, x):
        n = x.size(1)
        i, j = torch.tril(torch.ones(n, n, dtype=x.dtype), diagonal=-1).nonzero().t()
        z = torch.stack([torch.cat((x_[i], x_[j]), dim=1) for x_ in x])
        return self.net(z)

    @property
    def output_dim(self):
        return self._output_dim


class KRREncoder(torch.nn.Module):
    def __init__(self, x_dim, y_dim):
        super(KRREncoder, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self._output_dim = x_dim

    def forward(self, xy):
        assert xy.shape[2] == (self.x_dim + self.y_dim)
        xs, ys = torch.split(xy, (self.x_dim, self.y_dim), dim=2)
        n_train = xs.shape[1]
        batch_K = torch.bmm(xs, xs.transpose(1, 2))
        Identity = torch.eye(n_train, device=batch_K.device)
        alphas, _ = torch.gesv(ys, (batch_K + Identity))
        ws = torch.bmm(xs.transpose(1, 2), alphas).mean(dim=2, keepdim=False)
        return ws

    @property
    def output_dim(self):
        return self._output_dim


class DatasetEncoderFactory:
    name_map = dict(
        set2set=Set2SetEncoder,
        simple_attention=StandardAttentionEncoder,
        multihead_attention=MultiHeadAttentionEncoder,
        relation_net=RelationNetEncoder,
        deepset=DeepSetEncoder,
        krr=KRREncoder,
        cnp=CnpEncoder
    )

    def __init__(self):
        super(DatasetEncoderFactory, self).__init__()

    def __call__(self, arch, **kwargs):
        if arch not in self.name_map:
            raise Exception(f"Unhandled model. The name of \
             the model should be one of those: {list(self.name_map.keys())}")
        in_dim = kwargs.pop('input_dim')
        out_dim = kwargs.pop('output_dim')
        if arch.lower() == 'krr':
            modelclass = self.name_map[arch.lower()]
            model = modelclass(x_dim=in_dim, y_dim=out_dim, **kwargs)
        else:
            modelclass = self.name_map[arch.lower()]
            model = modelclass(input_dim=in_dim + out_dim, **kwargs)
        return model
