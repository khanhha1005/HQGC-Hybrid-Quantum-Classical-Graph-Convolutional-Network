import math

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

try:
    from ..QNN_Node_Embedding import TimedQuantumNet, MatchedClassicalNet, DenseClassicalNet
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from QNN_Node_Embedding import TimedQuantumNet, MatchedClassicalNet, DenseClassicalNet


class QGCNConv(MessagePassing):
    """
    Graph convolution whose node transformation is a variational quantum circuit:
    projection to n_qubits (if needed) -> pi * tanh (angle normalization) -> VQC per node
    -> symmetric-normalized aggregation over neighbors with self-loops -> bias.

    Args:
        in_channels: input node feature size
        n_layers: depth D of the variational circuit
        n_qubits: number of qubits n (= output width)
        head: 'vqc' (HQGC), 'classical' (parameter-matched classical map), 'dense' (dense layer tanh(Wx+b)),
              or 'none' (circuit removed)
        message_passing: if False, each node only sees itself (ablation)
        angle_norm: if True, map the circuit input to (-pi, pi) with pi * tanh
    """

    def __init__(self, in_channels, n_layers, n_qubits=8, head='vqc', message_passing=True, angle_norm=True):
        super().__init__(aggr='add')

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.message_passing = message_passing
        self.angle_norm = angle_norm

        # Linear layer to reduce dimensions if needed
        if in_channels != n_qubits:
            self.feature_reduction = Linear(in_channels, n_qubits, bias=False)
        else:
            self.feature_reduction = None

        if head == 'vqc':
            self.qc = TimedQuantumNet(n_qubits, n_layers)
        elif head == 'classical':
            self.qc = MatchedClassicalNet(n_qubits, n_layers)
        elif head == 'dense':
            self.qc = DenseClassicalNet(n_qubits, n_layers)
        elif head == 'none':
            self.qc = torch.nn.Identity()
        else:
            raise ValueError(f"unknown head '{head}'")

        self.bias = Parameter(torch.zeros(n_qubits))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Reduce feature dimensions if needed
        h = self.feature_reduction(x) if self.feature_reduction is not None else x

        # Step 2: Bound the rotation angles to (-pi, pi)
        if self.angle_norm:
            h = math.pi * torch.tanh(h)

        # Step 3: Apply quantum circuit
        h = self.qc(h)

        if not self.message_passing:
            return h + self.bias

        # Step 4: Add self-loops and compute normalization.
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        row, col = edge_index
        deg = degree(col, h.size(0), dtype=h.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 5: Start propagating messages.
        out = self.propagate(edge_index, x=h, norm=norm)

        # Step 6: Apply a final bias vector.
        return out + self.bias

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Normalize node features.
        return norm.view(-1, 1) * x_j
