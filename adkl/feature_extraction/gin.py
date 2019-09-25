import torch
from torch import nn

from .fc import FCLayer
from .utils import get_pooling, to_sparse, ClonableModule


def pack_graph(batch_G, batch_x, return_sparse=False, fill_missing=0):
    """Pack a batch of graph and atom features into a single graph

    Parameters
    ----------
    batch_G : iterable (torch.LongTensor 2D), of size (n_i, n_i). Sparse tensor allowed
    batch_x: iterable (torch.Tensor 2D) of size (n_i,d_i)
    return_sparse: whether to return a sparse graph
    fill_missing: (int) fill out-of-graph bond
    Note that by filling out-of-graph positions, with anything other than 0, you cannot have a sparse tensor.

    Returns
    -------
    new_batch_G, new_batch_x: (torch.LongTensor 2D, torch.Tensor 2D)
        This tuple represent a new arbitrary graph and the corresponding atom feature matrix.
        new_batch_G has size (N, N), with $N = \sum_i n_i$, while new_batch_x has size (N,d)
    """
    out_x = torch.cat(tuple(batch_x), dim=0)
    n_neigb = out_x.shape[0]
    out_G = batch_G[0].new_zeros((n_neigb, n_neigb))
    cur_ind = 0
    n_per_mol = []  # should return this eventually
    for g in batch_G:
        g_size = g.shape[0] + cur_ind
        n_per_mol.append(g.shape[0])
        if g.is_sparse:
            g = g.to_dense()
        out_G[cur_ind:g_size, cur_ind:g_size] = g
        cur_ind = g_size

    if return_sparse and fill_missing == 0:
        out_G = to_sparse(out_G)
    return out_G, out_x


class TorchGINConv(nn.Module):
    r"""
    This layer implements the graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{t+1}_i = NN_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x^{t}}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x^{t+1}}_j \right),

    where :math:`NN_{\mathbf{\Theta}}` denotes a neural network.

    Arguments
    ---------
        in_size: int
            Input dimension of the node feature
        kernel_size : int
            Output dim of the node feature embedding
        eps : float, optional
            (Initial) :math:`\epsilon` value. If set to None, eps will be trained
            (Default value = None)
        net : `nn.Module`, optional
            Neural network :math:`NN_{\mathbf{\Theta}}`.
            If not provided, `ivbase.nn.base.FCLayer` will be used
            (Default value = None)
        init_fn: `torch.nn.init`, optional
            Initialization function to use for the dense layer.
            The default initialization for the `nn.Linear` will be used if this is not provided.
        pooling: str or callable, optional
            Pooling function to use.
            Should be one supported by :func:`ivbase.nn.commons.get_pooling`.
            (Default value = 'sum')
        pack_batch: bool, optional
            Whether to pack the batch of graph into a larger one.
            Use this if the batch of graphs have various size.
            (Default value = False)
        use_sparse: bool, optional
            Whether the input adjacency matrices are sparse or should be converted to sparse. This option is almost useless when the graph is not packed
            (Default value = False)
        kwargs:
            Optional named parameters to send to the neural network
    """

    def __init__(self, in_size, kernel_size, G_size=None, eps=None, net=None, init_fn=None, pooling="sum",
                 pack_batch=False, use_sparse=False, **kwargs):

        super(TorchGINConv, self).__init__()
        self.in_size = in_size
        self.out_size = kernel_size
        self._pooling = get_pooling(pooling)
        self.pack_batch = pack_batch
        self.use_sparse = use_sparse
        self.G_size = G_size
        self.init_fn = init_fn
        if 'normalize' in kwargs:
            kwargs.pop("normalize")
        self.net = (net or FCLayer)(in_size, kernel_size, **kwargs)
        self.chosen_eps = eps
        if eps is None:
            self.eps = torch.nn.Parameter(torch.Tensor([0]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Initialize weights of the models, as defined by the input initialization scheme.

        Arguments
        ----------
            init_fn (callable, optional): Function to initialize the linear weights. If it is not provided
                an attempt to use the object `self.init_fn` would be first made, before doing nothing.
                (Default value = None)

        See Also
        --------
            `nn.init` for more information
        """
        chosen_eps = self.chosen_eps or 0
        if not isinstance(self.eps, nn.Parameter):
            self.eps.data.fill_(chosen_eps)
        try:
            self.net.reset_parameters(self.init_fn)
        except:
            pass

    def gather(self, h, nodes_per_graph=None):
        r"""
        Graph level representation of the features.
        This function apply the pooling layer to gather and aggregate information on all nodes

        Arguments
        ----------
            h: torch.FloatTensor of size B x N x M of P x M
                Learned features for all atoms for each graph.
                If the graph is not packed, the first dimension should correspond to the batch size (B).
            node_per_graph: list, optional
                If the graph is packed, this argument is required to indicate
                the number of elements in each of the graph forming the packing. Original order is expected to be conserved.

        Returns
        -------
            out (torch.FloatTensor of size B x M): Aggregated features for the B graphs inside the input.

        """
        if self.pack_batch:
            if not nodes_per_graph:
                raise ValueError("Expect node_per_mol for packed graph")
            return torch.squeeze(torch.stack([self._pooling(mol_feat)
                                              for mol_feat in torch.split(h, nodes_per_graph, dim=1)], dim=0), dim=1)
        return torch.squeeze(self._pooling(h), dim=1)

    def forward(self, G, x):
        r"""
        Compute the output of the layer

        Arguments
        ----------
            batch_G (list of dgl.DGLGraph or dgl.BatchedDGLGraph): List of DGL graph object or a batched DGL graph
                Graphs in G should contains a feature vector stored under the key `hv` for nodes and `he` for edges.

        Returns
        -------
            G: dgl.BatchedDGLGraph
                Batched DGL graphs
            out: torch.FloatTensor of size N x M
                Pooled features for the current layer
        """
        G_size = self.G_size
        if not self.pack_batch and isinstance(G, (list, tuple)):
            G = torch.stack(G)
            x = torch.stack(x)  # .requires_grad_()

        if not isinstance(G, torch.Tensor) and self.pack_batch:
            G, h = pack_graph(G, x, self.use_sparse)
            G_size = h.shape[0]

        else:  # expect a batch here
            # ensure that batch dim is there
            xshape = x.shape[2] if x.dim() > 2 else x.shape[1]
            if G.is_sparse and self.pack_batch:
                h = x.view(-1, xshape)
            else:
                G = G.view(-1, G.shape[-2], G.shape[-1])
                h = x.view(-1, G.shape[1], xshape)
            G_size = h.shape[-2]

        out = torch.matmul(G, h)
        out = (1 + self.eps) * h + out
        out = self.net(out.view(-1, self.in_size))
        return G, out.view(-1, G_size, self.out_size)


class TorchLGINConv(TorchGINConv):
    r"""
    This layer implements the graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, with a
    modification to implement both the summation and the Laplacian. Hence, the input
    features for the MLP are a concatenation of the neighbour-summed features (same as GIN),
    and the neighbour-Laplacian features.

    Arguments
    ---------
        in_size: int
            Input dimension of the node feature
        kernel_size : int
            Output dim of the node feature embedding
        eps : float, optional
            (Initial) :math:`\epsilon` value. If set to None, eps will be trained
            (Default value = None)
        init_fn: `torch.nn.init`, optional
            Initialization function to use for the dense layer.
            The default initialization for the `nn.Linear` will be used if this is not provided.
        pooling: str or callable, optional
            Pooling function to use.
            Should be one supported by :func:`ivbase.nn.commons.get_pooling`.
            (Default value = 'sum')
        pack_batch: bool, optional
            Whether to pack the batch of graph into a larger one.
            Use this if the batch of graphs have various size.
            (Default value = False)
        use_sparse: bool, optional
            Whether the input adjacency matrices are sparse or should be converted to sparse. This option is almost useless when the graph is not packed
            (Default value = False)
        use_laplacian: bool, optional
            Whether to use the Laplacian alongside the summation. If True, then the feature vector becomes the concatenation
            of the neighbour features sum (same as standard GIN), and the neighbour feature Laplacian. If False, then a
            standard GIN is used.
            (Default value = True)
        kwargs:
            Optional named parameters to send to the neural network

    Attributes
    ----------
        in_size: int
            size of the input feature space
        out_size: int
            size of the output feature space
        G_size: int
            number of elements (atom) per graphs
        eps: float
            The eps self-loop parameter
        linears: torch.ModuleList(torch.nn.Linear)
            dense layers to project input feature
        use_sparse: bool
            Whether to use sparse tensors
        pack_batch: bool
            Whether to pack the batch of graphs
        normalize: bool
            Whether to normalize input
        use_laplacian: bool
            Whether to use the Laplacian before the MLP

    See Also
    --------
        :class:`ivbase.nn.graphs.conv.gin.GINLayer`,
        `ivbase.nn.graphs.conv.gin.TorchGINLayer`
    """

    def __init__(self, in_size, kernel_size, G_size=None, eps=None, net=None, init_fn=None, pooling="sum",
                 pack_batch=False, use_sparse=False, use_laplacian=True, **kwargs):

        self.use_laplacian = use_laplacian

        if self.use_laplacian:
            in_size *= 2
        super(TorchLGINConv, self).__init__(in_size=in_size, kernel_size=kernel_size,
                                            G_size=G_size, eps=eps, net=net, init_fn=init_fn,
                                            pooling=pooling, pack_batch=pack_batch, use_sparse=use_sparse, **kwargs)

    def forward(self, G, x):
        r"""
        Compute the output of the layer

        Arguments
        ----------
            batch_G (list of dgl.DGLGraph or dgl.BatchedDGLGraph): List of DGL graph object or a batched DGL graph
                Graphs in G should contains a feature vector stored under the key `hv` for nodes and `he` for edges.

        Returns
        -------
            G: dgl.BatchedDGLGraph
                Batched DGL graphs
            out: torch.FloatTensor of size N x M
                Pooled features for the current layer
        """
        G_size = self.G_size
        if not self.pack_batch and isinstance(G, (list, tuple)):
            G = torch.stack(G)
            x = torch.stack(x)  # .requires_grad_()

        if not isinstance(G, torch.Tensor) and self.pack_batch:
            G, h = pack_graph(G, x, self.use_sparse)
            G_size = h.shape[0]

        else:  # expect a batch here
            # ensure that batch dim is there
            xshape = x.shape[2] if x.dim() > 2 else x.shape[1]
            if G.is_sparse and self.pack_batch:
                h = x.view(-1, xshape)
            else:
                G = G.view(-1, G.shape[-2], G.shape[-1])
                h = x.view(-1, G.shape[1], xshape)
            G_size = h.shape[-2]

        if self.use_laplacian:
            laplacian_diag = - G.matmul(G) * torch.eye(G_size).unsqueeze(0)
            laplacian_mapper = laplacian_diag + G

            out_sum = torch.matmul(G, h)
            out_lap = laplacian_mapper.matmul(h)
            out = torch.cat([out_sum, out_lap], dim=-1)
        else:
            out = torch.matmul(G, h)
            out = (1 + self.eps) * h + out

        out = self.net(out.view(-1, self.in_size))
        return G, out.view(-1, G_size, self.out_size)


class GINFeaturesExtractor(ClonableModule):
    def __init__(self, in_size, layer_sizes=[64], activation='ReLU',
                 dropout=0.0, b_norm=False, **kwargs):
        super(GINFeaturesExtractor, self).__init__()
        self.in_size = in_size
        self.conv_layers = nn.ModuleList()
        self.pack_batch = True
        self.layer_sizes = layer_sizes

        for ksize in layer_sizes:
            gc_params = {}
            if isinstance(ksize, (tuple, list)) and len(ksize) == 2:  # so i can customize later
                ksize, gc_params = ksize
            gc = TorchGINConv(G_size=None, in_size=in_size,
                              kernel_size=ksize, pack_batch=self.pack_batch,
                              dropout=dropout, b_norm=b_norm,
                              activation=activation, **gc_params)
            self.conv_layers.append(gc)
            in_size = ksize

    def find_node_per_mol(self, G):
        return [g.shape[0] for g in G]

    def forward(self, input_x):
        G, x = zip(*input_x)
        h = x
        n_per_mol = self.find_node_per_mol(G)
        for i, cv_layer in enumerate(self.conv_layers):
            G, h = cv_layer(G, h)
        # h is batch_size, G_size, kernel_size
        # we sum on the graph dimension before going to the fully connected layers
        h = cv_layer.gather(h, nodes_per_mol=n_per_mol)  # h is now batch_size, kernel_size
        return h

    @property
    def output_dim(self):
        res = self.layer_sizes[-1]
        if isinstance(res, int):
            return res
        if isinstance(res, (tuple, list)) and len(res) == 2:
            return res[0]
        raise Exception('Impossible to find the size of the output dim')
