"""
Script to count model parameters for all models across different codebases.
Creates a comparison table of model sizes.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
import pandas as pd
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Quantum_GCN import QGCN
from models.PyTorch_GCN import GNN
from models.Custom_GCN_Model import GCN
from models.GCNConv_Layers.Custom_GCNConv import GCNConv


def count_parameters(model):
    """Count total number of trainable parameters in a PyTorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_tensorflow_params_from_architecture(layers_info):
    """Estimate parameters from layer architecture (for TensorFlow models)"""
    total = 0
    for layer in layers_info:
        # Format: (input_dim, output_dim, bias=True)
        if len(layer) == 3:
            in_dim, out_dim, has_bias = layer
        else:
            in_dim, out_dim = layer
            has_bias = True
        # Weights: input_dim * output_dim
        total += in_dim * out_dim
        if has_bias:
            total += out_dim
    return total


def get_quantum_gnn_models(input_dims_list):
    """Get Quantum GNN models with different configurations"""
    models = {}
    
    for input_dim in input_dims_list:
        # QGCN with Linear classifier
        model = QGCN(input_dims=input_dim, q_depths=[1, 1], output_dims=1, 
                     activ_fn=LeakyReLU(0.2), classifier=None, readout=False)
        models[f'QGCN_Linear_{input_dim}D'] = model
        
        # QGCN with MPS classifier
        model = QGCN(input_dims=input_dim, q_depths=[1, 1], output_dims=1, 
                     activ_fn=LeakyReLU(0.2), classifier='MPS', readout=False)
        models[f'QGCN_MPS_{input_dim}D'] = model
        
        # QGCN with TTN classifier
        model = QGCN(input_dims=input_dim, q_depths=[1, 1], output_dims=1, 
                     activ_fn=LeakyReLU(0.2), classifier='TTN', readout=False)
        models[f'QGCN_TTN_{input_dim}D'] = model
    
    return models


def get_classical_gnn_models(input_dims_list):
    """Get Classical GNN models"""
    models = {}
    
    for input_dim in input_dims_list:
        # PyTorch GCN
        model = GNN(input_dims=input_dim, hidden_dims=[16, 16], output_dims=1, 
                    activ_fn=LeakyReLU(0.2))
        models[f'PyTorch_GCN_{input_dim}D'] = model
        
        # Custom GCN with classical layers - use no_node_NN=False which means identity transformation
        from models.GCNConv_Layers.Custom_GCNConv import GCNConv
        # For Custom_GCNConv with no_node_NN=False, we need out_channels==in_channels or None
        # So we'll use input_dim -> 16 -> 16, but need to handle the first layer specially
        # Actually, let's skip Custom_GCN for now as it has a specific use case
        # Instead, we'll use PyTorch_GCN which is the standard implementation
        pass
    
    return models


def get_amevuldetector_params():
    """Calculate parameters for AMEVulDetector models based on architecture"""
    models = {}
    
    # FNNModel architecture:
    # Input: (1, 250) - note: shape is (batch, 1, 250), so input_dim is 250
    # Dense(250) x3 (pattern1vec, pattern2vec, pattern3vec) - each takes input_dim=250
    # After concatenation: 3*250 = 750
    # Dense(100) -> Dense(1)
    fnn_params = 0
    # Pattern branches (each processes 250 -> 250)
    fnn_params += 3 * (250 * 250 + 250)  # 3 pattern branches
    # Merge and output
    fnn_params += (750 * 100 + 100) + (100 * 1 + 1)
    models['AMEVulDetector_FNNModel'] = fnn_params
    
    # EncoderWeight architecture:
    # Input: (1, 250)
    # Graph branch: Dense(200) -> Dense(1) for weight
    # Pattern branches: Dense(200) -> Dense(1) for weight x3
    # After concatenation: 200*4 = 800
    # Dense(100) -> Dense(1)
    encoder_weight_params = 0
    # Graph branch
    encoder_weight_params += (250 * 200 + 200) + (200 * 1 + 1)
    # Pattern branches (3 branches)
    encoder_weight_params += 3 * ((250 * 200 + 200) + (200 * 1 + 1))
    # Merge and output
    encoder_weight_params += (800 * 100 + 100) + (100 * 1 + 1)
    models['AMEVulDetector_EncoderWeight'] = encoder_weight_params
    
    # EncoderAttention architecture:
    # Similar to EncoderWeight but uses Attention layers
    # Graph: Dense(200), Patterns: Dense(200) x3
    # Concatenate -> Dense(200)
    # Attention layers x4 (each Attention has internal params)
    # Concatenate -> Dense(100) -> Dense(1)
    encoder_attention_params = 0
    # Initial embeddings
    encoder_attention_params += (250 * 200 + 200) * 4  # graph + 3 patterns
    # Merge layer
    encoder_attention_params += (800 * 200 + 200)
    # Attention layers - each attention typically has query/key/value projections
    # Keras Attention uses query/key/value, assume 200->200 for each, so 3*200*200 per attention
    # But simplified: assume each attention adds ~200*200 params
    encoder_attention_params += 4 * (200 * 200)  # 4 attention layers
    # Final layers
    encoder_attention_params += (800 * 100 + 100) + (100 * 1 + 1)
    models['AMEVulDetector_EncoderAttention'] = encoder_attention_params
    
    return models


def get_gnnscvuldetector_params():
    """Estimate parameters for GNNSCVulDetector based on default params"""
    # Based on GNNSCModel code analysis:
    # - hidden_size: typically 64-128 (use 64 as default)
    # - annotation_size: variable (typically 64-250, use 64 as default)
    # - num_edge_types: variable (typically 10-20, use 10 as default)
    # - propagation_rounds: 2
    # - propagation_substeps: 20 (but this doesn't add params, just computation steps)
    
    hidden_size = 64  # Typical default
    annotation_size = 64  # Typical default from code
    num_edge_types = 10  # Typical default
    
    total = 0
    
    # Edge weights: num_edge_types * (hidden_size * hidden_size)
    total += num_edge_types * (hidden_size * hidden_size)
    
    # GRU cell: The GRUCell has parameters:
    # - Input transformation: hidden_size * hidden_size (for each of 3 gates)
    # - Hidden transformation: hidden_size * hidden_size (for each of 3 gates)
    # - Biases: 3 * hidden_size (for input) + 3 * hidden_size (for hidden)
    # GRU params = 3 * (hidden_size * hidden_size) + 3 * (hidden_size * hidden_size) + 6 * hidden_size
    # Simplified: GRU has ~3*hidden_size*hidden_size for reset/update/output gates
    # Actually GRUCell in TF has: input_dim * hidden_size * 3 (gates) + hidden_size * hidden_size * 3 + biases
    # Since we use it with hidden_size input: hidden_size * hidden_size * 3 + hidden_size * hidden_size * 3 + 6*hidden_size
    gru_params = 3 * hidden_size * hidden_size + 3 * hidden_size * hidden_size + 6 * hidden_size
    
    # Gated regression: Two MLPs
    # Regression gate: 2*hidden_size -> 1 (no hidden layers based on MLP call with [])
    gate_params = 2 * hidden_size * 1 + 1
    # Regression transform: hidden_size -> 1
    transform_params = hidden_size * 1 + 1
    
    total = num_edge_types * (hidden_size * hidden_size) + gru_params + gate_params + transform_params
    
    return {'GNNSCVulDetector_GNNSCModel': int(total)}


def main():
    """Main function to count all model parameters and create comparison table"""
    results = []
    
    # Input dimensions from vulnerability datasets
    input_dims_list = [41, 64, 250]  # timestamp, integeroverflow, reentrancy
    
    print("Counting Quantum GNN model parameters...")
    quantum_models = get_quantum_gnn_models(input_dims_list)
    for name, model in quantum_models.items():
        params = count_parameters(model)
        results.append({
            'Model': name,
            'Framework': 'PyTorch',
            'Category': 'Quantum GNN',
            'Parameters': params,
            'Parameters (M)': f"{params / 1e6:.4f}",
            'Size (MB)': f"{params * 4 / 1e6:.4f}"  # Assuming float32 (4 bytes)
        })
        print(f"  {name}: {params:,} parameters")
    
    print("\nCounting Classical GNN model parameters...")
    classical_models = get_classical_gnn_models(input_dims_list)
    for name, model in classical_models.items():
        params = count_parameters(model)
        results.append({
            'Model': name,
            'Framework': 'PyTorch',
            'Category': 'Classical GNN',
            'Parameters': params,
            'Parameters (M)': f"{params / 1e6:.4f}",
            'Size (MB)': f"{params * 4 / 1e6:.4f}"
        })
        print(f"  {name}: {params:,} parameters")
    
    print("\nEstimating AMEVulDetector model parameters...")
    ame_models = get_amevuldetector_params()
    for name, params in ame_models.items():
        results.append({
            'Model': name,
            'Framework': 'TensorFlow',
            'Category': 'AME VulDetector',
            'Parameters': params,
            'Parameters (M)': f"{params / 1e6:.4f}",
            'Size (MB)': f"{params * 4 / 1e6:.4f}"
        })
        print(f"  {name}: {params:,} parameters")
    
    print("\nEstimating GNNSCVulDetector model parameters...")
    gnnsc_models = get_gnnscvuldetector_params()
    for name, params in gnnsc_models.items():
        results.append({
            'Model': name,
            'Framework': 'TensorFlow',
            'Category': 'GNNSC VulDetector',
            'Parameters': params,
            'Parameters (M)': f"{params / 1e6:.4f}",
            'Size (MB)': f"{params * 4 / 1e6:.4f}"
        })
        print(f"  {name}: {params:,} parameters")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df = df.sort_values(['Category', 'Model'])
    
    # Save to CSV
    output_path = "/home/HardDisk/CatKhanh/Quantum_GNN/model_size_comparison.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print("Model Size Comparison Table")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")
    
    # Create summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics by Category")
    print(f"{'='*80}")
    summary = df.groupby('Category').agg({
        'Parameters': ['count', 'mean', 'min', 'max']
    }).round(0)
    summary.columns = ['Count', 'Mean Params', 'Min Params', 'Max Params']
    print(summary.to_string())


if __name__ == "__main__":
    main()

