import torch
import torch.nn as nn
from .attention import StandardAttentionEncoder


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


class DatasetEncoderFactory:
    name_map = dict(
        simple_attention=StandardAttentionEncoder,
        relation_net=RelationNetEncoder,
        deepset=DeepSetEncoder,
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
