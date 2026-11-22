import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from config import (
    USERNAME, TRAINING_DATA, MODEL_DIR,
    LEARNING_RATE, MAX_STEPS, EARLY_STOP_PATIENCE,
    MAIA_BASE_MODEL, FREEZE_RESIDUALS
)
from src.models.maia_net import MaiaNet
from src.models.load_weights import load_maia_weights
from src.data.prepare_data import ChessPositionDataset
from src.utils.move_mapping import POLICY_SIZE

# Expected input board tensor planes (112 planes)
BOARD_PLANES = 112

# Configure CPU multi-threading if CUDA GPU is not available
if not torch.cuda.is_available():
    torch.set_num_threads(os.cpu_count() or 4)

# Hardware execution device (NVIDIA CUDA GPU or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_move_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes top-1 prediction accuracy by comparing argmax of model logits
    against the ground-truth target move vector.
    """
    pred = logits.argmax(dim=1)
    gold = targets.argmax(dim=1)
    return (pred == gold).float().mean().item()


def move_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Backward-compatible alias for compute_move_accuracy."""
    return compute_move_accuracy(logits, targets)


def run_training_epoch(model: nn.Module, loader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       step: int, max_steps: int) -> tuple[float, float, int]:
    """
    Executes one training epoch over the dataset loader.

    Returns:
      (average_loss, average_accuracy, updated_step_counter)
    """
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for planes, policy, winner in loader:
        if step >= max_steps:
            break

        planes = planes.to(DEVICE)
        policy = policy.to(DEVICE)
        winner = winner.to(DEVICE)

        # Forward pass: model outputs policy logits and position win evaluation
        policy_logits, value_pred = model(planes)

        # Cross-Entropy policy loss (with label smoothing) + MSE value evaluation loss
        policy_loss = F.cross_entropy(policy_logits, policy, label_smoothing=0.1)
        value_loss = F.mse_loss(value_pred.squeeze(), winner)
        loss = policy_loss + 0.01 * value_loss

        # Backpropagation gradient step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc = compute_move_accuracy(policy_logits, policy)
        total_loss += loss.item()
        total_acc  += acc
        n          += 1
        step       += 1

    return total_loss / max(n, 1), total_acc / max(n, 1), step


def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                step: int, max_steps: int) -> tuple[float, float, int]:
    """Backward-compatible alias for run_training_epoch."""
    return run_training_epoch(model, loader, optimizer, step, max_steps)


@torch.no_grad()
def evaluate_model_performance(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """
    Evaluates the model on the validation dataset without gradient calculations.

    Returns:
      (average_validation_loss, average_validation_accuracy)
    """
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for planes, policy, winner in loader:
        planes = planes.to(DEVICE)
        policy = policy.to(DEVICE)
        winner = winner.to(DEVICE)

        policy_logits, value_pred = model(planes)
        policy_loss = F.cross_entropy(policy_logits, policy, label_smoothing=0.1)
        value_loss  = F.mse_loss(value_pred.squeeze(), winner)
        loss        = policy_loss + 0.01 * value_loss

        total_loss += loss.item()
        total_acc  += compute_move_accuracy(policy_logits, policy)
        n          += 1

    return total_loss / max(n, 1), total_acc / max(n, 1)


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """Backward-compatible alias for evaluate_model_performance."""
    return evaluate_model_performance(model, loader)


def main() -> None:
    """Main model training and fine-tuning pipeline."""
    print(f"\n=== Fine-Tuning Personalized Bot for Player: {USERNAME} ===")
    print(f"    Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"    GPU    : {torch.cuda.get_device_name(0)}")

    # 1. Load Preprocessed Dataset Chunks
    train_dir = Path(TRAINING_DATA) / "train"
    val_dir   = Path(TRAINING_DATA) / "val"
    train_ds = ChessPositionDataset(train_dir, max_samples=None)
    val_ds   = ChessPositionDataset(val_dir, max_samples=None)

    print(f"\nDataset Overview:\n  Train Positions: {len(train_ds):,}\n  Val Positions  : {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=0)

    # 2. Instantiate MaiaNet Neural Network Architecture
    model = MaiaNet(num_blocks=6, channels=64).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture:\n  Total Parameters : {total_params:,}")

    # 3. Load Pre-Trained Maia Base Weights
    maia_path = Path(MAIA_BASE_MODEL)
    if maia_path.exists():
        partial_state = load_maia_weights(str(maia_path), policy_size=POLICY_SIZE)
        missing, unexpected = model.load_state_dict(partial_state, strict=False)
        print(f"  Loaded pre-trained weights successfully from {maia_path}")
    else:
        print(f"  ⚠ Maia weights not found at {maia_path} — Please verify base_models directory!")
        return

    # 4. Configure Layer Freezing (Playing Style Capture vs Full Fine-Tuning)
    model.set_backbone_freezing(FREEZE_RESIDUALS)
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable Params : {sum(p.numel() for p in trainable):,} (FREEZE_RESIDUALS={FREEZE_RESIDUALS})")

    # 5. Initialize Adam Optimizer and Cosine Annealing LR Scheduler
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=LEARNING_RATE / 10)

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc   = 0.0
    patience_count = 0
    step           = 0
    epoch          = 0

    print(f"\nTraining Progress (Max {MAX_STEPS} steps, Early Stop Patience = {EARLY_STOP_PATIENCE}):\n")
    print(f"  {'Epoch':>5}  {'Step':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Loss':>9}  {'Val Acc':>9}  {'Status':>10}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*10}")

    start_time = time.time()

    # 6. Main Training Loop
    while step < MAX_STEPS:
        epoch += 1
        train_loss, train_acc, step = run_training_epoch(model, train_loader, optimizer, step, MAX_STEPS)
        scheduler.step()
        val_loss, val_acc = evaluate_model_performance(model, val_loader)

        # Check for model accuracy improvement and save best checkpoint
        if val_acc > best_val_acc + 0.001:
            best_val_acc   = val_acc
            patience_count = 0
            status = "[OK] saved"
            torch.save({
                "epoch":       epoch,
                "step":        step,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "val_loss":    val_loss,
                "username":    USERNAME,
            }, model_dir / "best_model.pt")
        else:
            patience_count += 1
            status = f"patience {patience_count}/{EARLY_STOP_PATIENCE}"

        print(f"  {epoch:>5}  {step:>6}  {train_loss:>10.4f}  {train_acc*100:>9.2f}%  {val_loss:>9.4f}  {val_acc*100:>8.2f}%  {status}", flush=True)

        # Early stopping check
        if patience_count >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping triggered — Validation accuracy reached plateau.", flush=True)
            break

    total_time = time.time() - start_time
    print(f"\n-- Fine-Tuning Results Summary -----------------------", flush=True)
    print(f"  Best Validation Accuracy : {best_val_acc*100:.2f}%", flush=True)
    print(f"  Total Training Time      : {total_time:.0f} seconds", flush=True)
    print(f"  Checkpoint Saved Path    : {MODEL_DIR}/best_model.pt", flush=True)


if __name__ == "__main__":
    main()
