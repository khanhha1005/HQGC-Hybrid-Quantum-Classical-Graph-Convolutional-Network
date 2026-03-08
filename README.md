# HQGC-Hybrid-Quantum-Classical-Graph-Convolutional-Network
HQGC (Quantum Graph Convolutional Network) is a hybrid quantum–classical GNN designed for smart-contract vulnerability detection .
It fuses a classical GNN encoder with a variational quantum circuit (VQC) head to learn richer contract-graph embeddings.


### Data Preparation

The training data in `train_data/` is obtained by running the **graph construction** and **graph normalization** steps from the [GNNSCVulDetector](https://github.com/Messi-Q/GNNSCVulDetector) repository. 

To prepare the training data:
1. Clone the [GNNSCVulDetector](https://github.com/Messi-Q/GNNSCVulDetector) repository
2. Follow their instructions to run graph construction and graph normalization on your smart contract dataset
3. The output JSON files should be placed in the `train_data/` directory with the following structure:
   ```
   train_data/
   ├── reentrancy/
   │   ├── train.json
   │   └── valid.json
   ├── integeroverflow/
   │   ├── train.json
   │   └── valid.json
   └── timestamp/
       ├── train.json
       └── valid.json
   ```

### Data Format

Each dataset contains:
- **Graph-structured code representations**: Nodes represent code elements (statements, expressions, etc.), edges represent relationships between code elements
- **Binary labels**: Each graph is labeled as vulnerable (1) or non-vulnerable (0)
- **JSON format**: Each JSON file contains a list of graphs with:
  - `node_features`: Feature vectors for each node
  - `graph`: Edge list in format `[source, edge_type, target]`
  - `targets`: Binary label (0 or 1)

