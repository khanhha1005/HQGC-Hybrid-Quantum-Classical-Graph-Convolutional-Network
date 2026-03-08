import torch
from torch.nn import Module, ModuleList, Linear, LeakyReLU
from torch_geometric.nn import global_mean_pool

try:
    from .GCNConv_Layers import QGCNConv
except ImportError:
    from GCNConv_Layers import QGCNConv


class QGCN(Module):
    """QGCN with Linear classifier only."""

    def __init__(self, input_dims, q_depths, output_dims, activ_fn=LeakyReLU(0.2), classifier=None, readout=False):

        super().__init__()
        layers = []
        max_qubits = 16
        n_qubits = min(input_dims, max_qubits)
        if n_qubits > 8:
            n_qubits = 16
        else:
            n_qubits = 8
        self.n_qubits = n_qubits

        for i, q_depth in enumerate(q_depths):
            layer_input_dims = input_dims if i == 0 else n_qubits
            qgcn_conv = QGCNConv(layer_input_dims, q_depth, n_qubits=n_qubits)
            layers.append(qgcn_conv)

        self.layers = ModuleList(layers)
        self.activ_fn = activ_fn

        if readout:
            self.readout = Linear(1, 1)
        else:
            self.readout = None

        self.classifier = Linear(self.n_qubits, output_dims)

    def forward(self, x, edge_index, batch):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        h = x
        for i in range(len(self.layers)):
            h = self.layers[i](h, edge_index)
            h = self.activ_fn(h)

        # readout layer to get the embedding for each graph in batch
        h = global_mean_pool(h, batch)
        h = self.classifier(h)

        if self.readout is not None:
            h = self.readout(h)

        # return the prediction from the postprocessing layer
        return h
