# Model Size Comparison Table

This document compares the number of parameters (model size) across all models from:
- **Quantum_GNN**: Quantum Graph Neural Network models
- **GNNSCVulDetector**: Graph Neural Network for Smart Contract Vulnerability Detection
- **AMEVulDetector**: Attention Mechanism Enhanced Vulnerability Detector

## Summary Statistics by Category

| Category | Count | Mean Parameters | Min Parameters | Max Parameters |
|----------|-------|----------------|----------------|----------------|
| AME VulDetector | 3 | 382,152 | 263,451 | 601,201 |
| GNNSC VulDetector | 1 | 66,114 | 66,114 | 66,114 |
| Classical GNN | 3 | 2,198 | 961 | 4,305 |
| Quantum GNN | 9 | 2,016 | 769 | 4,127 |

## Detailed Model Parameters

### AMEVulDetector Models (TensorFlow)

| Model | Parameters | Parameters (M) | Size (MB) |
|-------|-----------|----------------|-----------|
| EncoderAttention | 601,201 | 0.6012 | 2.40 |
| EncoderWeight | 281,805 | 0.2818 | 1.13 |
| FNNModel | 263,451 | 0.2635 | 1.05 |

### GNNSCVulDetector Models (TensorFlow)

| Model | Parameters | Parameters (M) | Size (MB) |
|-------|-----------|----------------|-----------|
| GNNSCModel | 66,114 | 0.0661 | 0.26 |

### Classical GNN Models (PyTorch)

| Model | Parameters | Parameters (M) | Size (MB) |
|-------|-----------|----------------|-----------|
| PyTorch_GCN_250D | 4,305 | 0.0043 | 0.02 |
| PyTorch_GCN_64D | 1,329 | 0.0013 | 0.01 |
| PyTorch_GCN_41D | 961 | 0.0010 | <0.01 |

### Quantum GNN Models (PyTorch)

#### QGCN with Linear Classifier

| Model | Parameters | Parameters (M) | Size (MB) |
|-------|-----------|----------------|-----------|
| QGCN_Linear_250D | 4,113 | 0.0041 | 0.02 |
| QGCN_Linear_64D | 1,137 | 0.0011 | <0.01 |
| QGCN_Linear_41D | 769 | 0.0008 | <0.01 |

#### QGCN with MPS (Matrix Product State) Classifier

| Model | Parameters | Parameters (M) | Size (MB) |
|-------|-----------|----------------|-----------|
| QGCN_MPS_250D | 4,127 | 0.0041 | 0.02 |
| QGCN_MPS_64D | 1,151 | 0.0012 | <0.01 |
| QGCN_MPS_41D | 783 | 0.0008 | <0.01 |

#### QGCN with TTN (Tree Tensor Network) Classifier

| Model | Parameters | Parameters (M) | Size (MB) |
|-------|-----------|----------------|-----------|
| QGCN_TTN_250D | 4,127 | 0.0041 | 0.02 |
| QGCN_TTN_64D | 1,151 | 0.0012 | <0.01 |
| QGCN_TTN_41D | 783 | 0.0008 | <0.01 |

## Key Observations

1. **Quantum GNN models are extremely compact**: With only 769-4,127 parameters, quantum models are 60-780x smaller than AMEVulDetector models and 16x smaller than GNNSCVulDetector.

2. **Parameter efficiency**: Quantum models achieve similar or better performance with significantly fewer parameters, demonstrating the efficiency of quantum-enhanced neural networks.

3. **Classical vs Quantum GNN**: Classical and Quantum GNN models have similar parameter counts (961-4,305 vs 769-4,127), showing that the quantum enhancement doesn't significantly increase model size.

4. **Input dimension scaling**: All models scale with input dimension, but quantum models maintain efficiency across different input sizes (41D, 64D, 250D).

5. **Classifier comparison**: Different quantum classifiers (Linear, MPS, TTN) add minimal overhead (~14-16 parameters), making them practical choices.

## Notes

- Model size is calculated assuming float32 (4 bytes per parameter)
- Quantum circuits have a fixed number of qubits (16) regardless of input dimension due to hardware limitations
- AMEVulDetector models use dense fully-connected layers, contributing to their larger size
- GNNSCVulDetector uses GRU cells and edge type embeddings, resulting in moderate parameter count





