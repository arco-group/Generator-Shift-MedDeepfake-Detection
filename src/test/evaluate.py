import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import _batch_inputs, _batch_values, _forward_model_with_inputs, _normalize_input_keys

def calculate_acc(y_true: np.ndarray, y_pred: np.ndarray, thres: float = 0.5) -> Tuple[float, float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0

    def _class_acc(class_value: int) -> float:
        mask = y_true == class_value
        if not np.any(mask):
            return 0.0
        return float(accuracy_score(y_true[mask], y_pred[mask] > thres))

    r_acc = _class_acc(0)
    f_acc = _class_acc(1)
    acc = float(accuracy_score(y_true, y_pred > thres))
    return r_acc, f_acc, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    input_key: str = "cube",
    input_keys: Optional[Sequence[str]] = None,
    return_rows: bool = False,
):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    resolved_input_keys = _normalize_input_keys(input_keys=input_keys, input_key=input_key)

    total_loss = 0.0
    total = 0
    y_true: List[float] = []
    y_pred: List[float] = []
    rows: List[dict] = []

    for batch in loader:
        inputs = tuple(t.to(device, non_blocking=True) for t in _batch_inputs(batch, resolved_input_keys))
        y = batch["label"].to(device, non_blocking=True)

        logits = _forward_model_with_inputs(model, inputs)
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError(f"Expected model output shape [B, 1], got {tuple(logits.shape)}")
        logits = logits.squeeze(1)
        targets = y.float().view(-1)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        y_true.extend(targets.detach().cpu().numpy().tolist())
        y_pred.extend(probs.detach().cpu().numpy().tolist())

        if return_rows:
            ids = _batch_values(batch, "img_id")
            split = _batch_values(batch, "derived_split")
            mods = _batch_values(batch, "mod")
            tys = _batch_values(batch, "ty")
            preds = (probs > threshold).long()
            if ids is not None and split is not None and mods is not None and tys is not None:
                for sample_id, sample_split, sample_mod, sample_ty, true_label, pred_label, prob in zip(
                    ids, split, mods, tys, targets, preds, probs
                ):
                    true_int = int(true_label.item())
                    pred_int = int(pred_label.item())
                    if true_int == 0 and pred_int == 0:
                        tag = "C0"
                    elif true_int == 1 and pred_int == 1:
                        tag = "C1"
                    elif true_int == 1 and pred_int == 0:
                        tag = "S0"
                    else:
                        tag = "S1"
                    rows.append(
                        {
                            "id": str(sample_id),
                            "derived_split": str(sample_split),
                            "mod": str(sample_mod),
                            "ty": str(sample_ty),
                            "prob": float(prob.item()),
                            "pred": pred_int,
                            "image_tag": tag,
                        }
                    )

    avg_loss = total_loss / max(total, 1)
    y_true_np = np.asarray(y_true, dtype=np.float32)
    y_pred_np = np.asarray(y_pred, dtype=np.float32)
    r_acc, f_acc, acc = calculate_acc(y_true_np, y_pred_np, threshold)
    ap = average_precision_score(y_true_np, y_pred_np) if len(y_true_np) else 0.0
    if return_rows:
        return avg_loss, acc, r_acc, f_acc, ap, rows
    return avg_loss, acc, r_acc, f_acc, ap


@torch.no_grad()
def evaluate_2dmodels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    input_key: str = "cube",
    input_keys: Optional[Sequence[str]] = None,
    slice_agg: str = "mean",
):
    model.eval()
    resolved_input_keys = _normalize_input_keys(input_keys=input_keys, input_key=input_key)
    if len(resolved_input_keys) != 1:
        raise ValueError("evaluate_2dmodels expects exactly one input key")
    if slice_agg not in {"mean", "max"}:
        raise ValueError("slice_agg must be one of {'mean', 'max'}")

    y_true: List[float] = []
    y_pred: List[float] = []
    rows: List[dict] = []
    max_slices = 0

    for batch in loader:
        inputs = _batch_inputs(batch, resolved_input_keys)
        if len(inputs) != 1:
            raise ValueError("evaluate_2dmodels expects a single tensor input")
        cube = inputs[0].to(device, non_blocking=True) # C,D,H,W
        y = batch["label"].to(device, non_blocking=True)

        if cube.ndim != 5:
            raise ValueError(f"Expected batched cube shape [B, C, D, H, W], got {tuple(cube.shape)}")

        targets = y.float().view(-1)
        batch_size = int(cube.shape[0])
        ids = _batch_values(batch, "img_id")
        split = _batch_values(batch, "derived_split")
        mods = _batch_values(batch, "mod")
        tys = _batch_values(batch, "ty")

        for i in range(batch_size):
            sample_cube = cube[i]
            if sample_cube.ndim != 4:
                raise ValueError(f"Expected sample cube shape [C, D, H, W], got {tuple(sample_cube.shape)}")
            depth = int(sample_cube.shape[1])
            slice_tensors = [sample_cube[:, d, :, :].contiguous() for d in range(depth)]
            slice_batch = torch.stack(slice_tensors, dim=0)
            slice_logits = _forward_model_with_inputs(model, (slice_batch,))
            if slice_logits.ndim != 2 or slice_logits.shape[1] != 1:
                raise ValueError(f"Expected model output shape [B, 1], got {tuple(slice_logits.shape)}")
            slice_logits = slice_logits.squeeze(1)
            slice_probs = torch.sigmoid(slice_logits)
            if slice_agg == "mean":
                agg_prob_tensor = slice_probs.mean()
            else:
                agg_prob_tensor = slice_probs.max()
            agg_prob = float(agg_prob_tensor.item())
            agg_pred = int(agg_prob > threshold)
            true_int = int(targets[i].item())
            y_true.append(float(targets[i].item()))
            y_pred.append(agg_prob)
            max_slices = max(max_slices, depth)

            if true_int == 0 and agg_pred == 0:
                tag = "C0"
            elif true_int == 1 and agg_pred == 1:
                tag = "C1"
            elif true_int == 1 and agg_pred == 0:
                tag = "S0"
            else:
                tag = "S1"

            row = {
                "id": str(ids[i]) if ids is not None else str(i),
                "derived_split": str(split[i]) if split is not None else "",
                "mod": str(mods[i]) if mods is not None else "",
                "ty": str(tys[i]) if tys is not None else "",
                "label": true_int,
                "agg_method": slice_agg,
                "agg_prob": agg_prob,
                "agg_pred": agg_pred,
                "image_tag": tag,
                "num_slices": depth,
                "_slice_probs": [float(p.item()) for p in slice_probs],
            }
            rows.append(row)

    for row in rows:
        slice_probs = row.pop("_slice_probs")
        for idx in range(max_slices):
            row[f"slice_{idx}_prob"] = float(slice_probs[idx]) if idx < len(slice_probs) else None

    y_true_np = np.asarray(y_true, dtype=np.float32)
    y_pred_np = np.asarray(y_pred, dtype=np.float32)
    r_acc, f_acc, acc = calculate_acc(y_true_np, y_pred_np, threshold)
    ap = average_precision_score(y_true_np, y_pred_np) if len(y_true_np) else 0.0
    return acc, r_acc, f_acc, ap, rows


def _get_arg(args: argparse.Namespace, *names: str, default: object = None) -> object:
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)
    return default
