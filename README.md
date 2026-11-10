# HQGC-Hybrid-Quantum-Classical-Graph-Convolutional-Network
HQGC (Hybrid Quantum–Classical Graph Convolutional Network) is a hybrid quantum–classical GNN for smart-contract
vulnerability detection. Each of its two graph-convolution layers projects the node features to `n = 8` dimensions,
maps them to (−π, π) with a scaled tanh, and passes each node through a variational quantum circuit (VQC). The
circuit uses angle embedding, `D = 1` layer of trainable R_X rotations, a CNOT ring, and Pauli-Z readout. The layer
then aggregates the result over the node's neighbors. Mean pooling and a linear layer produce the vulnerability logit.

### Requirements

Tested with Python 3.11, PyTorch 2.14 (CPU), PyTorch Geometric 2.8, PennyLane 0.45, scikit-learn 1.9, and pandas.

```
pip install torch torch_geometric pennylane scikit-learn pandas
```

### Data

We use the normalized contract graphs and the original training/test partition released with TMP in the
[GNNSCVulDetector](https://github.com/Messi-Q/GNNSCVulDetector) repository (`train_data/`). Clone it, then point
`--data_dir` at its `train_data/` folder:

```
train_data/
├── reentrancy/   train.json  valid.json
└── timestamp/    train.json  valid.json
```

In that release, `valid.json` is the **test** split. A stratified 10% of `train.json` is held out for validation and
early stopping. The test split is used only once, for the final evaluation.

Each JSON file contains a list of graphs with:
  - `node_features`: Feature vectors for each node (250-dim for reentrancy, 41-dim for timestamp)
  - `graph`: Edge list in format `[source, edge_type, target]` (edge types are not used by HQGC)
  - `targets`: Binary label (0 or 1)

### Reproducing the paper

```
# HQGC, both vulnerability classes, seeds 0-4 (Table I, "HQGC (ours)")
python code/train_fast.py --data_dir /path/to/GNNSCVulDetector/train_data

# Ablation and component analysis (Table I, lower block)
python code/train_fast.py --data_dir ... --variant classical novqc dense nomp nonorm

# Wider classical models (n = 16, 32, 64) and paired t-tests over seeds (Table R3 of the response letter)
python code/train_fast.py --data_dir ... --variant classical novqc --n_qubits 16   # repeat with 32 and 64
python code/analysis/paired_tests.py results

# Per-contract inference latency on one CPU thread (Sec. III-D)
python code/train_fast.py --data_dir ... --seeds 0 --threads 1 --latency

# Same on a CUDA GPU (circuits simulated by PennyLane default.qubit on the GPU)
python code/train_fast.py --data_dir ... --device cuda

# Benchmark redundancy, label ambiguity, and representation ceilings (Table R2 / last row of Table I)
python code/analysis/benchmark_stats.py /path/to/GNNSCVulDetector/train_data
```

| `--variant` | Table I row |
|---|---|
| `hqgc` | HQGC (ours) |
| `classical` | VQC → parameter-matched classical head: same n → n shape and parameter count. An element-wise shift followed by tanh and the parameter-free CNOT-ring mixing |
| `novqc` | VQC removed (no head) |
| `dense` | VQC → dense layer: a standard fully connected layer tanh(Wx + b), n² + n parameters per layer |
| `nomp` | Message passing removed |
| `nonorm` | Input normalization (π·tanh) removed |

Results are written to `--out_dir` (default `results/`) as one JSON file per run, `results_per_seed.csv`, and
`results_summary.csv` (mean ± sample standard deviation over seeds, in %).

Training settings are in `code/data/config.json`:
- binary cross-entropy, Adam (learning rate 1e-3), batch size 64, at most 100 epochs;
- validation F1 checked every 5 epochs; training stops after 10 checks without improvement, and the best checkpoint is restored;
- `n = 8` qubits, circuit depth 1 in each of the two layers, seeds 0–4.

Expected results (%, mean ± std over seeds 0–4, CPU):

| | Params | Acc | Pre | Rec | F1 |
|---|---|---|---|---|---|
| Reentrancy | 2,041 | 94.4 ± 0.9 | 88.6 ± 2.1 | 96.3 ± 0.0 | 92.2 ± 1.1 |
| Timestamp dependence | 369 | 88.9 ± 0.0 | 86.9 ± 0.0 | 86.2 ± 0.0 | 86.5 ± 0.0 |

On a GPU (tested with an RTX 5060, PyTorch 2.11 + CUDA 12.8), 46 of the 50 Table I runs give identical test
predictions and the means of all HQGC rows are unchanged. The remaining runs differ slightly because CUDA scatter-add is
not deterministic. With batch size 1, GPU inference is slower than one CPU thread (about 21 ms vs. 8 ms per contract),
because kernel-launch overhead dominates this small circuit.

`code/analysis/check_vqc_closed_form.py` checks that, at depth 1, the circuit's expectation values equal a closed-form
product of cosines.

### Changes from the first release
- The circuit input is normalized to (−π, π) with π·tanh before angle encoding.
- The number of qubits is fixed to 8. The first release picked 16 for both datasets.
- Model selection uses a held-out 10% of the training split. The first release used `valid.json` (the test split)
  both for early stopping and for the reported metrics.
- Early stopping monitors validation F1 instead of validation accuracy, and results are averaged over five seeds.
- Ablation variants, latency measurement, and benchmark statistics were added.
- `--n_qubits` sets the embedding width (wider classical comparisons), and `code/analysis/paired_tests.py` compares
  HQGC with every other model by a paired t-test over seeds.
- The `dense` variant replaces the circuit by a standard dense classical head of comparable size.
