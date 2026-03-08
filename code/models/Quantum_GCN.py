import torch
from torch.nn import Module, ModuleList, Linear, LeakyReLU
from torch_geometric.nn import global_mean_pool

try:
    from .GCNConv_Layers import QGCNConv
    from .Quantum_Classifiers import MPS, TTN
except ImportError:
    from GCNConv_Layers import QGCNConv
    from Quantum_Classifiers import MPS, TTN


class QGCN(Module):

    def __init__(self, input_dims, q_depths, output_dims, activ_fn=LeakyReLU(0.2), classifier=None, readout=False):

        super().__init__()
        layers = []
        # Use a fixed, manageable number of qubits (power of 2 for TTN compatibility)
        # This allows the model to work with any input dimension
        import numpy as np
        max_qubits = 16  # Maximum practical qubits
        n_qubits = min(input_dims, max_qubits)
        # Ensure power of 2 for TTN compatibility (8 or 16)
        if n_qubits > 8:
            n_qubits = 16
        else:
            n_qubits = 8
        self.n_qubits = n_qubits

        for i, q_depth in enumerate(q_depths):
            # First layer uses input_dims, subsequent layers use n_qubits
            layer_input_dims = input_dims if i == 0 else n_qubits
            qgcn_conv = QGCNConv(layer_input_dims, q_depth, n_qubits=n_qubits)
            layers.append(qgcn_conv)

        self.layers = ModuleList(layers)
        self.activ_fn = activ_fn

        if readout:
            self.readout = Linear(1, 1)
        else:
            self.readout = None

        if classifier == "MPS":
            # For binary classification, we need 1 measurement qubit
            # Use the last qubit for measurement
            meas_qubits = [self.n_qubits - 1] if output_dims == 1 else [i for i in range(
                self.n_qubits-1, max(self.n_qubits-1-output_dims, -1), -1)]
            self.classifier, _ = MPS(self.n_qubits, meas_qubits)

        elif classifier == "TTN":
            # n_qubits is already a power of 2, so we can use it directly
            self.classifier, _ = TTN(self.n_qubits)

        else:
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
