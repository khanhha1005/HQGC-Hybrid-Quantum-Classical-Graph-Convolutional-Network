"""
Quantum Neural Network circuit for node embedding.
Uses PennyLane with AngleEmbedding and variational layers.
"""

import torch
import pennylane as qml


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
