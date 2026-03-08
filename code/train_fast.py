"""
Fast training script for quantum GNN models on vulnerability detection datasets.
Optimized for speed with 100 epochs.
"""

import os
import sys
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Add code directory to path
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from data.load_vulnerability_data import get_dataloaders
from models.Quantum_GCN import QGCN
from torch.nn import LeakyReLU


class TrainingConfig:
    """Configuration for training"""
    def __init__(self):
        self.epochs = 100  # Increased to 100 epochs
        self.lr = 0.001
        self.batch_size = 64  # Increased batch size for faster training
        self.q_depths = [1, 1]  # Quantum circuit depth for each layer
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        self.val_frequency = 5  # Validate every N epochs to save time
        self.log_frequency = 5  # Log to TensorBoard every N epochs
        self.print_frequency = 10  # Print progress every N epochs
        # Early stopping configuration
        self.early_stop_patience = 10  # Number of validation checks to wait before stopping
        self.early_stop_min_delta = 0.0001  # Minimum change to qualify as an improvement
        self.early_stop_monitor = 'val_acc'  # Metric to monitor: 'val_acc' or 'val_loss'
        self.early_stop_mode = 'max'  # 'max' for accuracy, 'min' for loss


def train_model(model, train_loader, val_loader, config, device, writer=None, best_model_path=None):
    """Train a model with optimizations and early stopping"""
    optimizer = Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Enable cuDNN benchmarking for faster operations (if CUDA available)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Use mixed precision training if CUDA is available
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Early stopping variables
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 1  # Initialize to 1 in case first epoch is best
    validation_count = 0  # Track number of validation checks
    
    # Determine which metric to monitor
    monitor_metric = config.early_stop_monitor
    is_better = lambda current, best: (current > best + config.early_stop_min_delta) if config.early_stop_mode == 'max' else (current < best - config.early_stop_min_delta)
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        
        # Remove tqdm for faster iteration
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Use mixed precision if available
            if use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(batch.x, batch.edge_index, batch.batch)
                    target = batch.y.float().unsqueeze(1)
                    loss = criterion(out, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
                target = batch.y.float().unsqueeze(1)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            pred = (torch.sigmoid(out) > 0.5).float()
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)
            
            # Collect predictions and labels for metrics calculation
            train_preds.extend(pred.cpu().numpy().flatten())
            train_labels.extend(target.cpu().numpy().flatten())
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # Calculate precision, recall, and F1 for training
        train_preds_array = np.array(train_preds)
        train_labels_array = np.array(train_labels)
        train_precision = precision_score(train_labels_array, train_preds_array, zero_division=0)
        train_recall = recall_score(train_labels_array, train_preds_array, zero_division=0)
        train_f1 = f1_score(train_labels_array, train_preds_array, zero_division=0)
        
        # Validation phase (only every N epochs to save time)
        val_acc = 0.0
        avg_val_loss = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        train_precision = 0.0
        train_recall = 0.0
        train_f1 = 0.0
        should_validate = (epoch + 1) % config.val_frequency == 0 or epoch == config.epochs - 1
        improved = False  # Track if model improved during validation
        
        if should_validate:
            validation_count += 1
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device, non_blocking=True)
                    
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            out = model(batch.x, batch.edge_index, batch.batch)
                            target = batch.y.float().unsqueeze(1)
                            loss = criterion(out, target)
                    else:
                        out = model(batch.x, batch.edge_index, batch.batch)
                        target = batch.y.float().unsqueeze(1)
                        loss = criterion(out, target)
                    
                    val_loss += loss.item()
                    pred = (torch.sigmoid(out) > 0.5).float()
                    val_correct += (pred == target).sum().item()
                    val_total += target.size(0)
                    
                    # Collect predictions and labels for metrics calculation
                    val_preds.extend(pred.cpu().numpy().flatten())
                    val_labels.extend(target.cpu().numpy().flatten())
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            
            # Calculate precision, recall, and F1
            val_preds_array = np.array(val_preds)
            val_labels_array = np.array(val_labels)
            val_precision = precision_score(val_labels_array, val_preds_array, zero_division=0)
            val_recall = recall_score(val_labels_array, val_preds_array, zero_division=0)
            val_f1 = f1_score(val_labels_array, val_preds_array, zero_division=0)
            
            # Early stopping logic (only check when validation is performed)
            current_metric = val_acc if monitor_metric == 'val_acc' else avg_val_loss
            best_metric = best_val_acc if monitor_metric == 'val_acc' else best_val_loss
            
            improved = False
            if monitor_metric == 'val_acc':
                # For first validation, always save
                if validation_count == 1:
                    best_val_acc = val_acc
                    improved = True
                elif is_better(val_acc, best_val_acc):
                    best_val_acc = val_acc
                    improved = True
            else:  # val_loss
                # For first validation, always save
                if validation_count == 1:
                    best_val_loss = avg_val_loss
                    improved = True
                elif is_better(avg_val_loss, best_val_loss):
                    best_val_loss = avg_val_loss
                    improved = True
            
            if improved:
                patience_counter = 0
                best_epoch = epoch + 1
                # Save best model weights
                if best_model_path is not None:
                    torch.save(model.state_dict(), best_model_path)
                    if writer is not None:
                        writer.add_scalar('EarlyStop/Best_Epoch', best_epoch, epoch + 1)
            else:
                patience_counter += 1
            
            # Check for early stopping
            if patience_counter >= config.early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1} (validation check {validation_count}). "
                      f"Best {monitor_metric} was {best_metric:.4f} at epoch {best_epoch}.")
                if writer is not None:
                    writer.add_scalar('EarlyStop/Stopped_At', epoch + 1, epoch + 1)
                break
        
        # Log to TensorBoard (only every N epochs to reduce I/O overhead)
        if writer is not None and ((epoch + 1) % config.log_frequency == 0 or epoch == config.epochs - 1):
            writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
            writer.add_scalar('Accuracy/Train', train_acc, epoch + 1)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch + 1)
            writer.add_scalar('Precision/Train', train_precision, epoch + 1)
            writer.add_scalar('Recall/Train', train_recall, epoch + 1)
            writer.add_scalar('F1/Train', train_f1, epoch + 1)
            if should_validate:
                writer.add_scalar('Precision/Validation', val_precision, epoch + 1)
                writer.add_scalar('Recall/Validation', val_recall, epoch + 1)
                writer.add_scalar('F1/Validation', val_f1, epoch + 1)
        
        # Print progress (only every N epochs to reduce I/O)
        if (epoch + 1) % config.print_frequency == 0 or epoch == config.epochs - 1:
            early_stop_info = f" [Early Stop: {patience_counter}/{config.early_stop_patience}]" if should_validate and patience_counter > 0 else ""
            best_info = f" [Best: Epoch {best_epoch}]" if should_validate and improved else ""
            if should_validate:
                print(f"Epoch {epoch+1}/{config.epochs}: Train Loss: {avg_train_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f} (P:{train_precision:.4f}, R:{train_recall:.4f}, F1:{train_f1:.4f}), "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} "
                      f"(P:{val_precision:.4f}, R:{val_recall:.4f}, F1:{val_f1:.4f}){early_stop_info}{best_info}")
            else:
                print(f"Epoch {epoch+1}/{config.epochs}: Train Loss: {avg_train_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f} (P:{train_precision:.4f}, R:{train_recall:.4f}, F1:{train_f1:.4f})")
    
    # Load best model weights before returning
    if best_model_path is not None and os.path.exists(best_model_path):
        print(f"Loading best model weights from epoch {best_epoch}...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        if writer is not None:
            writer.add_scalar('EarlyStop/Final_Best_Epoch', best_epoch, config.epochs)
    
    return model, best_epoch


def evaluate_model(model, test_loader, device):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    use_amp = torch.cuda.is_available()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(batch.x, batch.edge_index, batch.batch)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            
            pred = (torch.sigmoid(out) > 0.5).float().cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_preds.extend(pred.flatten())
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def get_input_dim(train_loader):
    """Get input dimension from first batch"""
    for batch in train_loader:
        return batch.x.size(1)
    return 64  # default


def main():
    # Configuration
    config = TrainingConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print("Mixed precision training: ENABLED")
        print("cuDNN benchmarking: ENABLED")
    else:
        print("Warning: CUDA is not available, training will be slower on CPU")
    
    # Paths to vulnerability data
    base_data_path = "/home/HardDisk/CatKhanh/GNNSCVulDetector/train_data"
    # Training order: reentrancy -> timestamp -> integeroverflow
    vulnerability_types = ['reentrancy', 'timestamp', 'integeroverflow']
    
    # Model configurations to test
    model_configs = [
        {'name': 'QGCN_Linear', 'classifier': None},
        {'name': 'QGCN_MPS', 'classifier': 'MPS'},
        {'name': 'QGCN_TTN', 'classifier': 'TTN'},
    ]
    
    results = []
    
    # Train each model on each vulnerability type
    for vuln_type in vulnerability_types:
        print(f"\n{'='*60}")
        print(f"Processing vulnerability type: {vuln_type}")
        print(f"{'='*60}\n")
        
        train_path = os.path.join(base_data_path, vuln_type, 'train.json')
        valid_path = os.path.join(base_data_path, vuln_type, 'valid.json')
        
        if not os.path.exists(train_path) or not os.path.exists(valid_path):
            print(f"Warning: Data files not found for {vuln_type}, skipping...")
            continue
        
        # Load data
        print("Loading data...")
        # Use entire train.json for training, valid.json for validation/testing
        train_loader, val_loader, test_loader = get_dataloaders(
            train_path, valid_path, test_path=None,
            batch_size=config.batch_size,
            train_ratio=config.train_ratio,  # Not used anymore
            val_ratio=config.val_ratio,      # Not used anymore
            test_ratio=config.test_ratio     # Not used anymore
        )
        
        # Get input dimension
        input_dims = get_input_dim(train_loader)
        output_dims = 1  # Binary classification
        
        print(f"Input dimension: {input_dims}")
        print(f"Training samples: {len(train_loader.dataset)} (entire train.json)")
        print(f"Validation samples: {len(val_loader.dataset)} (valid.json)")
        print(f"Test samples: {len(test_loader.dataset)} (valid.json)")
        print(f"Batch size: {config.batch_size}")
        print(f"Epochs: {config.epochs}")
        print(f"Validation frequency: every {config.val_frequency} epochs")
        print(f"Logging frequency: every {config.log_frequency} epochs")
        
        # Train each model configuration
        for model_config in model_configs:
            model_name = model_config['name']
            classifier_type = model_config['classifier']
            
            print(f"\n{'-'*60}")
            print(f"Training {model_name} on {vuln_type}")
            print(f"{'-'*60}\n")
            
            # Setup TensorBoard writer
            log_dir_base = "/home/HardDisk/CatKhanh/Quantum_GNN/runs_v2"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(log_dir_base, vuln_type, f"{model_name}_{timestamp}")
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs saved to: {log_dir}")
            print(f"To view, run: tensorboard --logdir={log_dir_base}\n")
            
            try:
                # Create model
                model = QGCN(
                    input_dims=input_dims,
                    q_depths=config.q_depths,
                    output_dims=output_dims,
                    activ_fn=LeakyReLU(0.2),
                    classifier=classifier_type,
                    readout=False
                ).to(device)
                
                # Note: Model compilation disabled for quantum models due to complex number operations
                # torch.compile() has limited support for complex operators used in quantum circuits
                # The warnings about "complex operators" and "cudagraphs" are expected and can be ignored
                # Training will continue normally without compilation
                
                # Create temporary path for best model weights during training
                models_dir = "/home/HardDisk/CatKhanh/Quantum_GNN/models"
                os.makedirs(models_dir, exist_ok=True)
                best_model_temp_path = os.path.join(models_dir, f"best_{model_name}_{vuln_type}_{timestamp}.pth")
                
                # Train model with early stopping
                print(f"Early stopping: patience={config.early_stop_patience} validation checks, "
                      f"monitor={config.early_stop_monitor}, mode={config.early_stop_mode}\n")
                model, best_epoch = train_model(model, train_loader, val_loader, config, device, 
                                  writer=writer, best_model_path=best_model_temp_path)
                
                # Evaluate model
                print("\nEvaluating on test set...")
                metrics = evaluate_model(model, test_loader, device)
                
                # Log final test metrics to TensorBoard
                writer.add_scalar('Metrics/Test_Accuracy', metrics['accuracy'], config.epochs)
                writer.add_scalar('Metrics/Test_Precision', metrics['precision'], config.epochs)
                writer.add_scalar('Metrics/Test_Recall', metrics['recall'], config.epochs)
                writer.add_scalar('Metrics/Test_F1', metrics['f1'], config.epochs)
                
                # Store results
                results.append({
                    'Vulnerability_Type': vuln_type,
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1']
                })
                
                print(f"\nResults for {model_name} on {vuln_type}:")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1:        {metrics['f1']:.4f}\n")
                
                # Save final model weights (best model from early stopping)
                model_filename = f"{model_name}_{vuln_type}_{timestamp}.pth"
                model_path = os.path.join(models_dir, model_filename)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_dims': input_dims,
                    'output_dims': output_dims,
                    'q_depths': config.q_depths,
                    'classifier': classifier_type,
                    'vulnerability_type': vuln_type,
                    'model_name': model_name,
                    'metrics': metrics,
                    'epochs': config.epochs,
                    'early_stopped': True,
                    'best_epoch': best_epoch
                }, model_path)
                print(f"Model weights saved to: {model_path}\n")
                
                # Clean up temporary best model file
                if os.path.exists(best_model_temp_path):
                    os.remove(best_model_temp_path)
                
                # Close writer
                writer.close()
                
            except Exception as e:
                print(f"Error training {model_name} on {vuln_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append({
                    'Vulnerability_Type': vuln_type,
                    'Model': model_name,
                    'Accuracy': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'F1': np.nan
                })
                # Close writer even on error
                if 'writer' in locals():
                    writer.close()
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = "/home/HardDisk/CatKhanh/Quantum_GNN/results_table_fast.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")
    print("Results Summary:")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")
    print(f"\nModel weights saved to: /home/HardDisk/CatKhanh/Quantum_GNN/models")
    print(f"TensorBoard logs saved to: /home/HardDisk/CatKhanh/Quantum_GNN/runs_v2")
    print(f"To view TensorBoard, run: tensorboard --logdir=/home/HardDisk/CatKhanh/Quantum_GNN/runs_v2")


if __name__ == "__main__":
    main()

