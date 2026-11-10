"""
Train and evaluate HQGC (and its ablation variants) on the TMP smart-contract benchmark.

Protocol (Sec. III-D of the paper):
  - original released partition: train.json for training, valid.json as the test split;
  - a stratified 10% of train.json is held out for early stopping; the test split is used once at the end;
  - BCE, Adam (lr 1e-3), batch size 64, at most 100 epochs;
  - validation F1 checked every 5 epochs, early stopping after 10 checks without improvement,
    best checkpoint restored;
  - five seeds (0-4), mean +- standard deviation reported.

Example:
  python code/train_fast.py --data_dir /path/to/GNNSCVulDetector/train_data
  python code/train_fast.py --data_dir ... --variant classical          # VQC -> matched classical map
  python code/train_fast.py --data_dir ... --vuln reentrancy --seeds 0 --latency
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import LeakyReLU
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add code directory to path
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from data.load_vulnerability_data import get_dataloaders
from models.Quantum_GCN import QGCN
from models.QNN_Node_Embedding import TimedQuantumNet, CIRCUIT_TIME

# Variants of Table I: one component changed at a time
VARIANTS = {
    'hqgc':      dict(head='vqc', message_passing=True, angle_norm=True),
    'classical': dict(head='classical', message_passing=True, angle_norm=True),  # VQC -> matched classical map
    'novqc':     dict(head='none', message_passing=True, angle_norm=True),       # VQC removed
    'dense':     dict(head='dense', message_passing=True, angle_norm=True),      # VQC -> dense layer tanh(Wx+b)
    'nomp':      dict(head='vqc', message_passing=False, angle_norm=True),       # message passing removed
    'nonorm':    dict(head='vqc', message_passing=True, angle_norm=False),       # input normalization removed
}


class TrainingConfig:
    """Configuration for training (defaults from data/config.json)"""
    def __init__(self, path=os.path.join(code_dir, 'data', 'config.json')):
        with open(path) as f:
            c = json.load(f)
        self.val_ratio = c['VAL RATIO']
        self.batch_size = c['BATCH SIZE']
        self.epochs = c['EPOCHS']
        self.lr = c['LEARNING RATE']
        self.n_qubits = c['N QUBITS']
        self.q_depths = c['Q DEPTHS']
        self.val_frequency = c['VAL FREQUENCY']              # validate every N epochs
        self.early_stop_patience = c['EARLY STOP PATIENCE']  # number of validation checks without improvement
        self.early_stop_min_delta = 1e-4
        self.seeds = c['SEEDS']


def model_device(model):
    return next(model.parameters()).device


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    device = model_device(model)
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds.append((torch.sigmoid(out) > 0.5).long().flatten().cpu())
        labels.append(batch.y.flatten().cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def evaluate_model(model, loader):
    """Evaluate model and return metrics"""
    y, p = predict(model, loader)
    return {
        'accuracy': accuracy_score(y, p),
        'precision': precision_score(y, p, zero_division=0),
        'recall': recall_score(y, p, zero_division=0),
        'f1': f1_score(y, p, zero_division=0),
    }


def train_model(model, train_loader, val_loader, config):
    """Train with early stopping on validation F1; returns the model with the best weights restored."""
    optimizer = Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()
    device = model_device(model)

    best_f1, best_state, best_epoch, bad_checks, epoch_times = -1.0, None, 0, 0, []
    for epoch in range(config.epochs):
        model.train()
        t0 = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
        synchronize(device)
        epoch_times.append(time.perf_counter() - t0)

        # Validation phase (only every N epochs)
        if (epoch + 1) % config.val_frequency and epoch != config.epochs - 1:
            continue
        val_f1 = evaluate_model(model, val_loader)['f1']
        if val_f1 > best_f1 + config.early_stop_min_delta:
            best_f1, best_epoch, bad_checks = val_f1, epoch + 1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_checks += 1
            if bad_checks >= config.early_stop_patience:
                print(f"  early stopping at epoch {epoch + 1}; best val F1 {best_f1:.4f} at epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    return model, dict(best_epoch=best_epoch, epochs_run=epoch + 1, val_f1=best_f1,
                       sec_per_epoch=float(np.mean(epoch_times)))


@torch.no_grad()
def measure_latency(model, dataset, warmup=20):
    """Per-contract latency (ms) from a cached graph to the decision, batch size 1."""
    model.eval()
    device = model_device(model)
    dataset = [d.to(device) for d in dataset]  # cached graphs already on the device
    for d in dataset[:warmup]:
        model(d.x, d.edge_index, torch.zeros(d.num_nodes, dtype=torch.long, device=device))
    synchronize(device)
    lat, circ = [], []
    for d in dataset:
        batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
        CIRCUIT_TIME[0] = 0.0
        t0 = time.perf_counter()
        decision = (torch.sigmoid(model(d.x, d.edge_index, batch)) > 0.5).item()
        lat.append(time.perf_counter() - t0)
        circ.append(CIRCUIT_TIME[0])
    lat, circ = np.array(lat) * 1e3, np.array(circ) * 1e3
    return dict(mean_ms=float(lat.mean()), std_ms=float(lat.std()), circuit_share=float(circ.sum() / lat.sum()))


def run(vuln_type, seed, variant, args, config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        os.path.join(args.data_dir, vuln_type, 'train.json'),
        os.path.join(args.data_dir, vuln_type, 'valid.json'),
        batch_size=config.batch_size, val_ratio=config.val_ratio, seed=seed)

    model = QGCN(input_dims=train_loader.dataset[0].x.size(1), q_depths=config.q_depths, output_dims=1,
                 n_qubits=config.n_qubits, activ_fn=LeakyReLU(0.2), **VARIANTS[variant])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_circuit = sum(p.numel() for m in model.modules() if isinstance(m, TimedQuantumNet) for p in m.parameters())
    model = model.to(args.device)

    model, info = train_model(model, train_loader, val_loader, config)
    result = dict(vulnerability=vuln_type, variant=variant, seed=seed, n_qubits=config.n_qubits,
                  device=str(args.device), n_params=n_params,
                  n_circuit_params=n_circuit, **info, **evaluate_model(model, test_loader))
    if args.latency:
        result.update(measure_latency(model, list(test_loader.dataset)))
    if args.save_models:
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"{variant}_{vuln_type}_s{seed}.pth"))
    return result


def main():
    config = TrainingConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', required=True, help='GNNSCVulDetector/train_data')
    ap.add_argument('--vuln', nargs='+', default=['reentrancy', 'timestamp'])
    ap.add_argument('--variant', nargs='+', default=['hqgc'], choices=list(VARIANTS))
    ap.add_argument('--seeds', nargs='+', type=int, default=config.seeds)
    ap.add_argument('--out_dir', default='results')
    ap.add_argument('--n_qubits', type=int, default=config.n_qubits,
                    help='embedding width n (number of qubits for the VQC variants)')
    ap.add_argument('--threads', type=int, default=0, help='torch CPU threads (0 = library default)')
    ap.add_argument('--device', default='cpu', help="'cpu' or 'cuda' (circuits are simulated on this device)")
    ap.add_argument('--latency', action='store_true', help='also measure per-contract inference latency')
    ap.add_argument('--save_models', action='store_true')
    args = ap.parse_args()

    config.n_qubits = args.n_qubits
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for vuln_type in args.vuln:
        for variant in args.variant:
            for seed in args.seeds:
                print(f"{vuln_type} / {variant} / seed {seed}")
                r = run(vuln_type, seed, variant, args, config)
                print(f"  test acc {r['accuracy']:.4f}  pre {r['precision']:.4f}  "
                      f"rec {r['recall']:.4f}  f1 {r['f1']:.4f}  ({r['n_params']} params)")
                tag = variant if args.n_qubits == 8 else f"{variant}_n{args.n_qubits}"
                with open(os.path.join(args.out_dir, f"{tag}_{vuln_type}_s{seed}.json"), 'w') as f:
                    json.dump(r, f, indent=1)
                results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, 'results_per_seed.csv'), index=False)

    # mean +- sample standard deviation over seeds, in %
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    summary = df.groupby(['vulnerability', 'variant', 'n_qubits']).agg(
        n_params=('n_params', 'first'), seeds=('seed', 'count'),
        **{f"{m}_mean": (m, 'mean') for m in metrics}, **{f"{m}_std": (m, 'std') for m in metrics}).reset_index()
    for m in metrics:
        summary[f"{m}_mean"] *= 100
        summary[f"{m}_std"] = summary[f"{m}_std"].fillna(0.0) * 100
    summary.to_csv(os.path.join(args.out_dir, 'results_summary.csv'), index=False)

    print("\nSummary (%, mean ± std over seeds):")
    for _, s in summary.iterrows():
        cells = "  ".join(f"{m[:3]} {s[m + '_mean']:.1f}±{s[m + '_std']:.1f}" for m in metrics)
        print(f"  {s['vulnerability']:<11s} {s['variant']:<10s} {int(s['n_params']):>5d} params  {cells}")


if __name__ == "__main__":
    main()
