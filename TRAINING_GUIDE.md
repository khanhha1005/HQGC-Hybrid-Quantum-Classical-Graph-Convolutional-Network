# Quantum GNN Training Guide for Vulnerability Detection

## Overview

This guide explains how to train quantum GNN models on vulnerability detection datasets and generate results tables.

## Files Created

1. **`code/data/load_vulnerability_data.py`**: Data loader for vulnerability JSON data
2. **`code/train_vulnerability_models.py`**: Main training script
3. **`code/models/__init__.py`**: Model package initialization
4. **`code/models/GCNConv_Layers/__init__.py`**: GCN layer package initialization
5. **`code/models/Quantum_Classifiers/__init__.py`**: Quantum classifier package initialization

## Bug Fixes

Several bugs were fixed in the model files:
- Fixed `Quantum_GCN.py` to use `QGCNConv` instead of `GCNConv` with quantum net
- Fixed `QGCNConv.py` to properly handle quantum_net return value
- Fixed `QGATConv.py` to use `Parameter` instead of `nn.Parameter` and handle quantum classifiers
- Fixed `Custom_GCNConv.py` to handle edge cases with bias initialization
- Fixed import issues in all model files

## Models Trained

The script trains three quantum model variants:
1. **QGCN_Linear**: Quantum GCN with classical linear classifier
2. **QGCN_MPS**: Quantum GCN with Matrix Product State classifier
3. **QGCN_TTN**: Quantum GCN with Tree Tensor Network classifier

## Vulnerability Types

Models are trained on three vulnerability types:
- `integeroverflow`
- `reentrancy`
- `timestamp`

## How to Run

```bash
cd /home/HardDisk/CatKhanh/Quantum_GNN
python code/train_vulnerability_models.py
tensorboard --logdir=/home/HardDisk/CatKhanh/Quantum_GNN/runs
```

## Configuration

Edit the `TrainingConfig` class in `train_vulnerability_models.py` to modify:
- `epochs`: Number of training epochs (default: 20)
- `lr`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 32)
- `q_depths`: Quantum circuit depths for each layer (default: [1, 1])

## Output

The script generates a CSV file at:
```
/home/HardDisk/CatKhanh/Quantum_GNN/results_table.csv
```

The CSV contains columns:
- `Vulnerability_Type`: Type of vulnerability (integeroverflow, reentrancy, timestamp)
- `Model`: Model name (QGCN_Linear, QGCN_MPS, QGCN_TTN)
- `Accuracy`: Classification accuracy
- `Precision`: Precision score
- `Recall`: Recall score
- `F1`: F1 score

## Requirements

Make sure you have the following packages installed:
- torch
- torch-geometric
- pennylane
- sklearn
- pandas
- numpy
- tqdm

## Data Format

The script expects JSON files in the format:
```json
{
  "targets": "0" or "1",
  "graph": [[source, edge_type, target], ...],
  "contract_name": "filename.sol",
  "node_features": [[feature_vector], ...]
}
```

Data should be located at:
- `/home/HardDisk/CatKhanh/GNNSCVulDetector/train_data/{vulnerability_type}/train.json`
- `/home/HardDisk/CatKhanh/GNNSCVulDetector/train_data/{vulnerability_type}/valid.json`

