from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.test.evaluate import evaluate
from src.utils.utils import (
    _batch_inputs,
    _forward_model_with_inputs,
    _get_arg,
    _normalize_input_keys,
    set_seed,
)


def _read_loss_csv(path: Path) -> List[float]:
    if not path.is_file():
        return []
    losses: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            losses.append(float(row["train_loss"] if "train_loss" in row else row["val_loss"]))
    return losses


def _read_epoch_loss_csv(path: Path) -> List[tuple[int, float]]:
    if not path.is_file():
        return []
    rows: List[tuple[int, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            loss_key = "train_loss" if "train_loss" in row else "val_loss"
            rows.append((epoch, float(row[loss_key])))
    return rows


def _append_loss_rows(path: Path, header: List[str], rows: List[List[object]]) -> None:
    file_exists = path.is_file()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(rows)


def _save_loss_csv(path: Path, header: List[str], rows: List[List[object]], append: bool) -> None:
    mode = "a" if append else "w"
    with open(path, mode, newline="") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(header)
        writer.writerows(rows)

def train(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    metrics_dir: Path,
    checkpoint_dir: Path,
    input_key: Optional[str] = None,
    input_keys: Optional[Sequence[str]] = None,
    model_type: str = "model",
    resume_training: bool = False,
    resume_checkpoint_path: Optional[Path] = None,
    start_epoch: Optional[int] = None,
    metadata: Optional[Dict[str, object]] = None,
    seed: int = 42,
) -> nn.Module:
    set_seed(int(seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    resolved_input_keys = _normalize_input_keys(input_keys=input_keys, input_key=input_key)

    optim_name = str(_get_arg(args, "optim", default="adam")).lower()
    lr = float(_get_arg(args, "lr", default=1e-4))
    weight_decay = float(_get_arg(args, "weight_decay", "wd", default=0.0))
    beta1 = float(_get_arg(args, "beta1", default=0.9))

    if optim_name == "adam":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(beta1, 0.999),
            weight_decay=weight_decay,
        )
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.0,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("optim must be in [adam, sgd]")

    scheduler = None
    lr_scheduler = str(_get_arg(args, "lr_scheduler", default="none")).lower()
    if lr_scheduler == "plateau":
        min_lr = float(_get_arg(args, "min_lr", "fc_min_lr", default=1e-6))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(_get_arg(args, "lr_factor", default=0.1)),
            patience=int(_get_arg(args, "lr_patience", default=5)),
            min_lr=min_lr,
        )

    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = bool(_get_arg(args, "amp", default=False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    total_epochs = int(_get_arg(args, "epochs", "niter", default=1))
    save_every = int(_get_arg(args, "save_epoch", "save_epoch_freq", default=100))
    early_stop = bool(_get_arg(args, "early_stop", default=False))
    es_patience = int(_get_arg(args, "es_patience", default=10))
    es_min_delta = float(_get_arg(args, "es_min_delta", default=0.0))

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    effective_start_epoch = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_epoch_history: List[int] = []
    val_epoch_history: List[int] = []
    resume_epoch = None
    persisted_train_rows = 0
    persisted_val_rows = 0

    if resume_training:
        if resume_checkpoint_path is None:
            raise ValueError("resume_checkpoint_path must be provided when resume_training is True")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        checkpoint_epoch = checkpoint.get("epoch")
        if checkpoint_epoch is not None:
            checkpoint_start_epoch = int(checkpoint_epoch) + 1
            if start_epoch is not None and int(start_epoch) != checkpoint_start_epoch:
                raise ValueError(
                    f"start_epoch={start_epoch} is inconsistent with checkpoint epoch {checkpoint_epoch} "
                    f"(expected {checkpoint_start_epoch})"
                )
            effective_start_epoch = checkpoint_start_epoch
        elif start_epoch is None:  
            raise ValueError(
                "Checkpoint does not contain 'epoch'; pass start_epoch explicitly for resume"
            )
        else:
            effective_start_epoch = int(start_epoch)

        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        epochs_no_improve = int(checkpoint.get("epochs_no_improve", 0))
        resume_epoch = effective_start_epoch

        train_history = _read_epoch_loss_csv(metrics_dir / "train_loss.csv")
        val_history = _read_epoch_loss_csv(metrics_dir / "val_loss.csv")
        train_epoch_history = [epoch for epoch, _ in train_history]
        val_epoch_history = [epoch for epoch, _ in val_history]
        train_losses = [loss for _, loss in train_history]
        val_losses = [loss for _, loss in val_history]
        persisted_train_rows = len(train_epoch_history)
        persisted_val_rows = len(val_epoch_history)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    (train_line,) = ax.plot(train_epoch_history, train_losses, label="train_loss")
    (val_line,) = ax.plot(val_epoch_history, val_losses, label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    if resume_training and resume_epoch is not None:
        with open(metrics_dir / "resume_info.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["resume_epoch", "resume_checkpoint_path"])
            writer.writerow([resume_epoch, str(resume_checkpoint_path)])

    for epoch in range(effective_start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs = tuple(t.to(device, non_blocking=True) for t in _batch_inputs(batch, resolved_input_keys))
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = _forward_model_with_inputs(model, inputs)
                if logits.ndim != 2 or logits.shape[1] != 1:
                    raise ValueError(f"Expected model output shape [B, 1], got {tuple(logits.shape)}")
                logits = logits.squeeze(1)
                targets = y.float().view(-1)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).long()
            running_loss += loss.item() * targets.size(0)
            correct += (pred == targets.long()).sum().item()
            total += targets.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, val_r_acc, val_f_acc, val_ap = evaluate(
            model,
            val_loader,
            device,
            input_keys=resolved_input_keys,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_epoch_history.append(epoch)
        val_epoch_history.append(epoch)
        _append_loss_rows(metrics_dir / "train_loss.csv", ["epoch", "train_loss"], [[epoch, train_loss]])
        _append_loss_rows(metrics_dir / "val_loss.csv", ["epoch", "val_loss"], [[epoch, val_loss]])
        persisted_train_rows += 1
        persisted_val_rows += 1

        improved = epoch == 0 or (best_val_loss - val_loss) > es_min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_payload = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "input_keys": list(resolved_input_keys),
                "model_type": model_type,
            }
            if metadata:
                checkpoint_payload.update(metadata)
            torch.save(checkpoint_payload, checkpoint_dir / "best_model.pt")
            with open(checkpoint_dir / "best_params_model.json", "w") as f:
                json.dump(
                    {
                        "best_val_loss": float(best_val_loss),
                        "best_epoch": int(best_epoch),
                        "input_keys": list(resolved_input_keys),
                        "model_type": model_type,
                        **(metadata or {}),
                    },
                    f,
                    indent=2,
                )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d}/{total_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} r_acc {val_r_acc:.4f} f_acc {val_f_acc:.4f} ap {val_ap:.4f}"
        )

        train_line.set_data(train_epoch_history, train_losses)
        val_line.set_data(val_epoch_history, val_losses)
        ax.relim()
        ax.autoscale_view()
        fig.tight_layout()
        fig.savefig(metrics_dir / "loss_live.png", dpi=150)

        if ((epoch % save_every == 0) and (epoch != 0)) or (epoch == total_epochs - 1):
            checkpoint_payload = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "best_val_loss": float(best_val_loss),
                "best_epoch": int(best_epoch),
                "epochs_no_improve": int(epochs_no_improve),
                "input_keys": list(resolved_input_keys),
                "model_type": model_type,
            }
            if metadata:
                checkpoint_payload.update(metadata)
            torch.save(
                checkpoint_payload,
                checkpoint_dir / f"model.pt",
            )

        if early_stop and epochs_no_improve >= es_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch {best_epoch} (val_loss={best_val_loss:.6f})."
            )
            break

    return model
