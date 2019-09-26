import math
import numpy as np
import torch
from torch import nn
from .utils import MaskedSoftmax


class AttentionLayer(nn.Module):
    """
    Attention layer that performs dot product att
    """
    def __init__(self, input_dim, value_dim, key_dim, pooling_function=None, nb_learn_key_val=0):

        super(AttentionLayer, self).__init__()

        self.query_network = nn.Linear(input_dim, key_dim)
        self.learn_key_val = nb_learn_key_val > 0
        if self.learn_key_val:
            self.l_keys = nn.Parameter(torch.Tensor(nb_learn_key_val, key_dim))
            self.l_vals= nn.Parameter(torch.Tensor(nb_learn_key_val, value_dim))
            nn.init.kaiming_uniform_(self.l_keys, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.l_vals, a=math.sqrt(5))
        else:
            self.key_network = nn.Linear(input_dim, key_dim)
            self.value_network = nn.Linear(input_dim, value_dim)

        self.norm_layer = nn.LayerNorm(value_dim)
        self.pooling_function = pooling_function
        self._output_dim = value_dim

        self.softmax = MaskedSoftmax(dim=2)

    def forward(self, query, key=None, value=None, mask=None):
        if key is None and value is None:
            key = query
            value = query
        assert query.dim() == 3

        query = self.query_network(query)

        if self.learn_key_val:
            key = self.l_keys.unsqueeze(0).expand(query.size(0), *self.l_keys.shape)
            value = self.l_vals.unsqueeze(0).expand(query.size(0), *self.l_vals.shape)
        else:
            key = self.key_network(key)
            value = self.value_network(value)

        attention_matrix = torch.bmm(query, key.transpose(1, 2))
        attention_matrix = attention_matrix / np.sqrt(query.size(2))

        if mask is not None:
            mask = mask.transpose(1, 2)

        attention_matrix = self.softmax(attention_matrix, mask)

        res = self.norm_layer(torch.bmm(attention_matrix, value))

        if self.pooling_function == 'max':
            res = torch.max(res, dim=1)[0]
        elif self.pooling_function == 'mean':
            res = torch.mean(res, dim=1)

        return res

    @property
    def output_dim(self):
        return self._output_dim

class StandardAttentionEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=20, num_layers=1, nb_learn_key_val=0):
        super(StandardAttentionEncoder, self).__init__()
        in_dim, out_dim = input_dim, hidden_dim
        layers = []
        for i in range(1, num_layers + 1):
            pf = None if i != num_layers else 'mean'
            layers.append(AttentionLayer(in_dim, out_dim, in_dim, pooling_function=pf, nb_learn_key_val=nb_learn_key_val))
            if i != num_layers:
                layers.append(nn.ReLU())
            in_dim, out_dim = out_dim, out_dim
        self._output_dim = out_dim
        self.net = nn.Sequential(*layers)
        self._output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)

    @property
    def output_dim(self):
        return self._output_dim


class Attention(nn.Module):
    """
    This module preforms the attentional layer.
    """

    def __init__(self):
        super(Attention, self).__init__()

        self.softmax = MaskedSoftmax(dim=1)

    def forward(self, key, query, value, mask=None):
        """
        Performs the forward computation.

        Parameters
        ----------
        key: torch.Tensor
            The key tensor (B * N * D).
        query: torch.Tensor
            The query tensor (B * N * D).
        value: torch.Tensor
            The value tensor (B * N * D).
        mask: torch.Tensor
            The mask tensor (B * N) for the key/value pair.
            Masked queries are dealt with outside this loop.
            Defaults to None (no masking)

        Returns
        -------
        output: torch.Tensor
            The output of the attention layer.
        """

        d = key.size(-1)

        # Computes the pre-softmax
        x = torch.matmul(query, key.transpose(1, 2)) / np.sqrt(d)

        # Applies the softmax
        if mask is not None:
            mask = mask.transpose(1, 2)

        x = self.softmax(x, mask)

        # Applies the softmax and multiplies by the value tensor
        output = torch.bmm(x, value)

        return output


class SelfAttention(nn.Module):
    """
    A self-attention block. The only difference is that the key,
    query and value tensors are one and the same.
    """

    def __init__(self):
        super(SelfAttention, self).__init__()

        self.attention_block = Attention()

    def forward(self, x, mask=None):
        """
        Computes the self-attention.

        Parameters
        ----------
        x: torch.Tensor
            This is self attention, which means that x is used as the key, query and value tensors.
        mask: torch.Tensor
            Tensor masking incomplete examples. Defaults to None (no masking).

        Returns
        -------
        attention: torch.Tensor
            The attention block.
        """
        attention = self.attention_block(x, x, x, mask)

        return attention