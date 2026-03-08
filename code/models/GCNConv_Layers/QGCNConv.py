import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

try:
    from ..QNN_Node_Embedding import quantum_net
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from QNN_Node_Embedding import quantum_net


class QGCNConv(MessagePassing):
    def __init__(self, in_channels, n_layers, n_qubits=None):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        
        # Limit qubits to at most 16 for practical quantum simulation
        # Ensure it's a power of 2
        import numpy as np
        if n_qubits is None:
            n_qubits = min(in_channels, 16)
        else:
            n_qubits = min(n_qubits, 16)
        
        # Ensure power of 2
        if n_qubits > 8:
            n_qubits = 16
        elif n_qubits > 4:
            n_qubits = 8
        elif n_qubits > 2:
            n_qubits = 4
        else:
            n_qubits = 2
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.in_channels = in_channels
        
        # Add linear layer to reduce dimensions if needed
        if in_channels != n_qubits:
            self.feature_reduction = Linear(in_channels, n_qubits, bias=False)
        else:
            self.feature_reduction = None
        
        self.qc = quantum_net(self.n_qubits, self.n_layers)
        self.bias = Parameter(torch.empty(n_qubits))
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        if self.feature_reduction is not None:
            self.feature_reduction.reset_parameters()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Reduce feature dimensions if needed
        if self.feature_reduction is not None:
            x_reduced = self.feature_reduction(x)
        else:
            x_reduced = x

        # Step 3: Apply quantum circuit
        q_out = self.qc(x_reduced).float()

        # Step 4: Compute normalization.
        row, col = edge_index
        deg = degree(col, q_out.size(0), dtype=q_out.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 5: Start propagating messages.
        out = self.propagate(edge_index, x=q_out, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Normalize node features.
        return norm.view(-1, 1) * x_j
