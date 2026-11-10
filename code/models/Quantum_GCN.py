import torch
from torch.nn import Module, ModuleList, Linear, LeakyReLU
from torch_geometric.nn import global_mean_pool

try:
    from .GCNConv_Layers import QGCNConv
except ImportError:
    from GCNConv_Layers import QGCNConv


class QGCN(Module):
    """
    HQGC: two QGCNConv layers of width n_qubits with LeakyReLU, global mean pooling, linear classifier.

    head / message_passing / angle_norm select the ablation variants reported in Table I.
    """

    def __init__(self, input_dims, q_depths, output_dims, n_qubits=8, activ_fn=LeakyReLU(0.2),
                 head='vqc', message_passing=True, angle_norm=True):

        super().__init__()
        self.n_qubits = n_qubits

        layers = []
        for i, q_depth in enumerate(q_depths):
            layer_input_dims = input_dims if i == 0 else n_qubits
            layers.append(QGCNConv(layer_input_dims, q_depth, n_qubits=n_qubits, head=head,
                                   message_passing=message_passing, angle_norm=angle_norm))

        self.layers = ModuleList(layers)
        self.activ_fn = activ_fn
        self.classifier = Linear(self.n_qubits, output_dims)

    def forward(self, x, edge_index, batch):
        h = x
        for layer in self.layers:
            h = self.activ_fn(layer(h, edge_index))

        # readout layer to get the embedding for each graph in batch
        h = global_mean_pool(h, batch)
        return self.classifier(h)
