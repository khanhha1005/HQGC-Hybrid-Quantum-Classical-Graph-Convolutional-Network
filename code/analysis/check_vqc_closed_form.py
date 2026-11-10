"""
For depth D = 1 (AngleEmbedding R_X + one R_X layer + CNOT ring), the Pauli-Z expectations of the
circuit equal products of cos(x_i + theta_i). This script compares the PennyLane simulation with
that closed form on random inputs.

    python code/analysis/check_vqc_closed_form.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.QNN_Node_Embedding import quantum_net, cnot_ring_parity

torch.manual_seed(0)
for n in (8, 16):
    q = quantum_net(n, 1)
    x = torch.randn(50, n) * 3
    z_circuit = q(x).float()
    z_closed = cnot_ring_parity(torch.cos(x + q.weights.detach()[0]))
    print(f"{n} qubits: max |VQC - closed form| = {(z_circuit - z_closed).abs().max().item():.2e}")
