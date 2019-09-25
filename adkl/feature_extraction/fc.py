from torch.nn import Dropout, Linear, Sequential, BatchNorm1d

from .utils import ClonableModule, get_activation


class FCLayer(ClonableModule):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer. Should be one supported by :func:`ivbase.nn.commons.get_activation`.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer

    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None):
        super(FCLayer, self).__init__()
        # Although I disagree with this it is simple enough and robust
        # if we trust the user base
        self._params = locals()
        activation = get_activation(activation)
        linear = Linear(in_size, out_size, bias=bias)
        if init_fn:
            init_fn(linear)
        layers = [linear, activation]
        if dropout:
            layers.append(Dropout(p=dropout))
        if b_norm:
            layers.append(BatchNorm1d(out_size))
        self.net = Sequential(*layers)


    @property
    def output_dim(self):
        r"""
        Dimension of the output feature space in which the input are projected

        Returns
        -------
            output_dim (int): Output dimension of this layer

        """
        return self.out_size

    def forward(self, x):
        r"""
        Compute the layer transformation on the input.

        Arguments
        ----------
            x: torch.Tensor
                input variable

        Returns
        -------
            out: torch.Tensor
                output of the layer
        """
        return self.net(x)


class FcFeaturesExtractor(ClonableModule):
    r"""
    Feature extractor using a Fully Connected Neural Network

    Arguments
    ----------
        input_size: int
            size of the input
        hidden_sizes: int list or int
            size of the hidden layers
        activation: str or callable
            activation function. Should be supported by :func:`ivbase.nn.commons.get_activation`
            (Default value = 'relu')
        b_norm: bool, optional):
            Whether batch norm is used or not.
            (Default value = False)
        dropout: float, optional
            Dropout probability to regularize the network. No dropout by default.
            (Default value = .0)

    Attributes
    ----------
        extractor: torch.nn.Module
            The underlying feature extractor of the model.
    """

    def __init__(self, input_size, hidden_sizes, activation='ReLU',
                 b_norm=False, dropout=0.0, ):
        super(FcFeaturesExtractor, self).__init__()
        self._params = locals()
        layers = []
        in_ = input_size
        for i, out_ in enumerate(hidden_sizes):
            layer = FCLayer(in_, out_, activation=activation,
                            b_norm=b_norm and (i == (len(hidden_sizes) - 1)),
                            dropout=dropout)
            layers.append(layer)
            in_ = out_

        self.__output_dim = in_
        self.extractor = Sequential(*layers)

    @property
    def output_dim(self):
        r"""
        Get the dimension of the feature space in which the elements are projected

        Returns
        -------
        output_dim: int
            Dimension of the output feature space
        """
        return self.__output_dim

    def forward(self, x):
        r"""
        Forward-pass method

        Arguments
        ----------
            x: torch.FloatTensor of size N*L
                Batch of N input vectors of size L (input_size).

        Returns
        -------
            phi_x: torch.FloatTensor of size N*D
                Batch of feature vectors.
                D is the dimension of the feature space (the model's output_dim).
        """
        return self.extractor(x)
