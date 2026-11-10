"""
Quantum Neural Network circuit for node embedding.
Uses PennyLane with AngleEmbedding and variational layers.
"""

import time

import torch
import pennylane as qml

# Accumulated wall-clock time spent in circuit simulation (used for the latency breakdown).
CIRCUIT_TIME = [0.0]


def quantum_net(n_qubits, n_layers):
    """
    Create a quantum neural network module for node embedding.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers

    Returns:
        torch.nn.Module that maps [N, n_qubits] -> [N, n_qubits]
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    return qlayer


class TimedQuantumNet(torch.nn.Module):
    """quantum_net that records the time spent in circuit simulation."""

    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.q = quantum_net(n_qubits, n_layers)

    def forward(self, x):
        # CUDA kernels run asynchronously, so synchronize around the circuit to time it
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        t = time.perf_counter()
        # default.qubit builds its initial |0...0> state on torch's default device; make that the input's device
        with torch.device(x.device):
            out = self.q(x).float()
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        CIRCUIT_TIME[0] += time.perf_counter() - t
        return out


def cnot_ring_parity(z):
    """
    Pauli-Z expectations after the CNOT ring CNOT(0,1), ..., CNOT(n-1,0) applied to a product state
    with per-qubit expectations z: the ring maps bits to b_j' = b_0 ^ ... ^ b_j (j >= 1) and
    b_0' = b_1 ^ ... ^ b_{n-1}, so each output is a product of input expectations.
    """
    cum = torch.cumprod(z, -1)
    return torch.cat([torch.prod(z[..., 1:], -1, keepdim=True), cum[..., 1:]], -1)


class MatchedClassicalNet(torch.nn.Module):
    """
    Classical counterpart of quantum_net for the ablation: same [N, n] -> [N, n] shape and the same
    number of trainable parameters (n_layers * n). Each layer is an element-wise shift followed by
    tanh (bounded like Pauli-Z expectations) and the parameter-free mixing of the CNOT ring.
    """

    def __init__(self, n_qubits, n_layers):
        super().__init__()
        # tanh is not periodic, so the circuit's U(0, 2*pi) init would saturate it
        self.weights = torch.nn.Parameter(torch.empty(n_layers, n_qubits).uniform_(-1, 1))

    def forward(self, x):
        z = x
        for w in self.weights:
            z = cnot_ring_parity(torch.tanh(z + w))
        return z


class DenseClassicalNet(torch.nn.Module):
    """
    Standard classical head for the ablation: one fully connected layer tanh(W x + b) per circuit layer, with
    W in R^{n x n} (n_layers * (n * n + n) trainable parameters).
    """

    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_qubits, n_qubits) for _ in range(n_layers)])

    def forward(self, x):
        for lin in self.layers:
            x = torch.tanh(lin(x))
        return x
