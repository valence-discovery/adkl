import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (AvgPool1d, BatchNorm1d, Conv1d, Dropout,
                      Embedding, MaxPool1d, Sequential, ModuleList)
from torch.nn.utils.rnn import pad_sequence
from .utils import (ClonableModule, GlobalAvgPool1d, GlobalMaxPool1d, GlobalSumPool1d,
                    Transpose, get_activation, get_pooling)


class StandardSelfAttention(nn.Module):
    r"""
    Standard Self Attention module to emulate interactions between elements of a sequence

    Arguments
    ----------
        input_size: int
            Size of the input vector at each time step
        output_size: int
            Size of the output at each time step
        outnet: Union[`torch.nn.module`, callable], optional:
            Neural network that will predict the output. If not provided,
            A MLP without activation will be used.
            (Default value = None)
        pooling: str, optional
            Pooling operation to perform. It can be either
            None, meaning no pooling is performed, or one of the supported pooling
            function name (see :func:`ivbase.nn.commons.get_pooling`)
            (Default value = None)

    Attributes
    ----------
        attention_net:
            linear function to use for computing the attention on input
        output_net:
            linear function for computing the output values on
            which attention should be applied
    """

    def __init__(self, input_size, output_size, pooling=None):
        super(StandardSelfAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.attention_net = nn.Linear(self.input_size, self.input_size, bias=False)
        self.output_net = nn.Linear(self.input_size, self.output_size)
        self.pooling = None
        if pooling:
            self.pooling = get_pooling(pooling)
            # any error here should be propagated immediately

    def forward(self, x, value=None, return_attention=False):
        r"""
        Applies attention on input

        Arguments
        ----------
            x: torch.FLoatTensor of size B*N*M
                Batch of B sequences of size N each.with M features.Note that M must match the input size vector
            value: torch.FLoatTensor of size B*N*D, optional
                Use provided values, instead of computing them again. This is to address case where the output_net has complex input.
                (Default value = None)
            return_attention: bool, optional
                Whether to return the attention matrix.

        Returns
        -------
            res: torch.FLoatTensor of size B*M' or B*N*M'
                The shape of the resulting output, will depends on the presence of a pooling operator for this layer
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        assert x.size(-1) == self.input_size
        query = x
        key = self.attention_net(x)
        if value is None:
            value = self.output_net(x)
        key = key.transpose(1, 2)
        attention_matrix = torch.bmm(query, key)
        attention_matrix = attention_matrix / math.sqrt(self.input_size)
        attention_matrix = F.softmax(attention_matrix, dim=2)
        applied_attention = torch.bmm(attention_matrix, value)
        if self.pooling is None:
            res = applied_attention
        else:
            res = self.pooling(applied_attention)
        if return_attention:
            return res, attention_matrix
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'insize=' + str(self.input_size) \
            + ', outsize=' + str(self.output_size) + ')'


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilatation=1, activation='ReLU',
                 b_norm=False, dropout=0.5, pooling_fn='max', pooling_len=1, use_self_attention=False,
                 padding=None):
        super(ConvBlock, self).__init__()
        if pooling_fn == 'avg':
            pool1d = AvgPool1d
        else:
            pool1d = MaxPool1d

        activation_cls = get_activation(activation)
        if padding is None:
            padding = (dilatation * (kernel_size - 1) + 1) // 2
        layers = [Conv1d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, dilation=dilatation)]
        if use_self_attention:
            layers.append(activation_cls)
            layers.append(StandardSelfAttention(out_channels, out_channels, None))
        if b_norm:
            layers.append(BatchNorm1d(out_channels))
        layers.append(activation_cls)
        layers.append(Dropout(dropout))
        if pooling_len > 1:
            layers.append(pool1d(pooling_len))

        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiKernelSizeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, kernel_aggregator='cat', **kwargs):
        super(MultiKernelSizeConvBlock, self).__init__()
        # print(in_channels, out_channels, kernel_sizes, kernel_aggregator)
        self.kernel_aggregator = kernel_aggregator
        self.net = ModuleList([ConvBlock(in_channels, out_channels, k, **kwargs) for k in kernel_sizes])
        if self.kernel_aggregator in ['cat', 'concat', 'c']:
            self._output_dim = out_channels * len(kernel_sizes)
        else:
            self._output_dim = out_channels

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x):
        res = [m(x).transpose(0, 2) for m in self.net]
        res = pad_sequence(res, batch_first=True).transpose(1, 3)
        if self.kernel_aggregator in ['cat', 'concat', 'c']:
            res = torch.cat(res.unbind(0), dim=1)
        elif self.kernel_aggregator in ['sum', 'add', 's', 'a']:
            res = res.sum(dim=0)
        return res


class Cnn1dFeaturesExtractor(ClonableModule):
    r"""
    Extract features from a sequence-like data using Convolutional Neural Network.
    Each time step or position of the sequence must be a discrete value.

    Arguments
    ----------
        vocab_size: int
            Size of the vocabulary, i.e the maximum number of discrete elements possible at each time step.
            Since padding will be used for small sequences, we expect the vocab size to be 1 + size of the alphabet.
            We also expect that 0 won't be use to represent any element of the vocabulary expect the padding.
        embedding_size: int
            The size of each embedding vector
        cnn_sizes: int list
            A list that specifies the size of each convolution layer.
            The size of the list implicitly defines the number of layers of the network
        kernel_size: int list or int list list
            the size of the kernels, i.e the number of time steps include in one convolution operation.
            An integer list means the same value will be used for each conv layer. A list of list allows to specify different sizes for different layers.
            The length of the list should match the length of cnn_sizes.
        pooling_len: int or int list, optional
            The number of time steps aggregated together by the pooling operation.
            An integer means the same pooling length is used for all layers.
            A list allows to specify different length for different layers. The length of the list should match the length of cnn_sizes
            (Default value = 1)
        pooling: str, optional
            One of {'avg', 'max'} (for AveragePooling and MaxPooling).
            It indicates the type of pooling operator to use after convolution.
            (Default value = 'avg')
        dilatation_rate: int or int list, optional
            The dilation factor tells how large are the gaps between elements in
            a feature map on which we apply a convolution filter.  If a integer is provided, the same value is used for all
            convolution layer. If dilation = 1 (no gaps),  every 1st element next to one position is included in the conv op.
            If dilation = 2, we take every 2nd (gaps of size 1), and so on. See https://arxiv.org/pdf/1511.07122.pdf for more info.
            (Default value = 1)
        activation: str or callable, optional
            The activation function. activation layer {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus'}
            The name of the activation function
        b_norm: bool, optional
            Whether to use Batch Normalization after each convolution layer.
            (Default value = False)
        use_self_attention: bool, optional
            Whether to use a self attention mechanism on the last conv layer before the pooling
            (Default value = False)
        dropout: float, optional
            Dropout probability to regularize the network. No dropout by default.
            (Default value = .0)

    Attributes
    ----------
        extractor: torch.nn.Module
            The underlying feature extractor of the model.
    """

    def __init__(self, vocab_size, embedding_size, cnn_sizes, kernel_sizes, pooling_len=1, pooling_fn='avg',
                 dilatation_rate=1, activation='ReLU', b_norm=False, use_self_attention=False,
                 dropout=0.0, kernel_agg_fn='sum'):
        super(Cnn1dFeaturesExtractor, self).__init__()

        self._params = locals()

        activation_cls = get_activation(activation)
        if pooling_fn not in ['avg', 'max', 'sum', None]:
            raise ValueError("the pooling type must be either 'max' or 'avg'")
        if len(cnn_sizes) <= 0:
            raise ValueError(
                "There should be at least on convolution layer (cnn_size should be positive.)")

        # network construction
        layers = [Embedding(vocab_size, embedding_size), Transpose(1, 2)]
        in_channel = embedding_size
        for i, out_channel in enumerate(cnn_sizes):
            is_not_last = i < (len(cnn_sizes) - 1)
            layer = MultiKernelSizeConvBlock(in_channel, out_channel, kernel_sizes, kernel_aggregator=kernel_agg_fn,
                                             dropout=dropout, pooling_len=pooling_len, pooling_fn=pooling_fn,
                                             activation=activation, b_norm=(b_norm and is_not_last),
                                             use_self_attention=use_self_attention, dilatation=(dilatation_rate ** i))
            layers.append(layer)
            in_channel = layer.output_dim
        layers.append(Transpose(1, 2))

        if pooling_fn == 'avg':
            layers.append(GlobalAvgPool1d(dim=1))
        elif pooling_fn == 'sum':
            layers.append(GlobalSumPool1d(dim=1))
        elif pooling_fn == 'max':
            layers.append(GlobalMaxPool1d(dim=1))

        self.__output_dim = in_channel
        self.extractor = Sequential(*layers)

    @property
    def output_dim(self):
        r"""
        Get the dimension of the feature space in which the sequences are projected

        Returns
        -------
        output_dim (int): Dimension of the output feature space

        """
        return self.__output_dim

    def forward(self, x):
        r"""
        Forward-pass method

        Arguments
        ----------
            x (torch.LongTensor of size N*L): Batch of N sequences of size L each.
                L is actually the length of the longest of the sequence in the bacth and we expected the
                rest of the sequences to be padded with zeros up to that length.
                Each entry of the tensor is supposed to be an integer representing an element of the vocabulary.
                0 is reserved as the padding marker.

        Returns
        -------
            phi_x: torch.FloatTensor of size N*D
                Batch of feature vectors. D is the dimension of the feature space.
                D is given by output_dim.
        """
        return self.extractor(x)
