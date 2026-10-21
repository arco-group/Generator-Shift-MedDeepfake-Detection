from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from src.utils.config_exp_utils import ExperimentConfig

SUMMARY_METRICS = [
    ("loss", "LOSS"),
    ("ap", "AP"),
    ("r_acc", "R_ACC"),
    ("f_acc", "F_ACC"),
    ("acc", "ACC"),
]

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def _get_arg(args: argparse.Namespace, *names: str, default: object = None) -> object:
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)
    return default

def parse_csv_list(raw: str) -> List[str]:
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items

def parse_gpu_ids(gpu_ids: str) -> List[int]:
    ids: List[int] = []
    for token in gpu_ids.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value >= 0:
            ids.append(value)
    return ids

def save_config_json(output_dir: Path, args: argparse.Namespace, experiments: Sequence[ExperimentConfig]) -> None:
    config = {
        "script": Path(__file__).name,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "experiments": [
            {
                "name": exp.name,
                "family": exp.family,
                "id_mods": exp.id_mods,
                "ood_mods": exp.ood_mods,
            }
            for exp in experiments
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def load_checkpoint(
    model: nn.Module, ckpt_path: Path, device: torch.device
) -> tuple[Optional[int], Optional[float]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    best_epoch = ckpt.get("best_epoch")
    best_val_loss = ckpt.get("best_val_loss")

    return (
        int(best_epoch) if best_epoch is not None else None,
        float(best_val_loss) if best_val_loss is not None else None,
    )


def load_best_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> int:
    best_epoch, _ = load_checkpoint(model, ckpt_path, device)
    return int(best_epoch) if best_epoch is not None else -1


def _batch_values(batch: Dict[str, object], key: str) -> Optional[List[object]]:
    value = batch.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif isinstance(value, tuple):
        value = list(value)
    elif not isinstance(value, list):
        value = [value]
    return value


def _batch_input(batch: Dict[str, object], input_key: str) -> torch.Tensor:
    if input_key not in batch:
        raise KeyError(f"Batch does not contain input key '{input_key}'")
    value = batch[input_key]
    if not torch.is_tensor(value):
        raise TypeError(f"Batch key '{input_key}' must be a tensor")
    return value

def _normalize_input_keys(
    input_keys: Optional[Sequence[str]] = None,  
    input_key: Optional[str] = None, 
) -> Tuple[str, ...]:
    if input_keys is not None:
        keys = (input_keys,) if isinstance(input_keys, str) else tuple(input_keys)
    elif input_key is not None:
        keys = (input_key,)
    else:
        raise ValueError("At least one input key must be provided")

    keys = tuple(str(key) for key in keys if str(key).strip())

    if not keys:
        raise ValueError("At least one input key must be provided")
    return keys


def _batch_inputs(batch: Dict[str, object], input_keys: Sequence[str] | str) -> Tuple[torch.Tensor, ...]:
    keys = (input_keys,) if isinstance(input_keys, str) else tuple(input_keys)
    return tuple(_batch_input(batch, key) for key in keys)


def _forward_model_with_inputs(model: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(inputs) == 1:
        output = model(inputs[0])
    elif len(inputs) == 2:
        output = model(inputs[0], inputs[1])
    else:
        raise ValueError(f"Unsupported number of inputs: {len(inputs)}. Expected 1 or 2.")

    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model returned an empty tuple/list")
        output = output[0]

    return output

def save_predictions_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "mod", "ty", "prob", "pred", "image_tag"])
        writer.writeheader()
        writer.writerows(rows)

def collect_loaded_samples(loader: DataLoader, source_split: str) -> pd.DataFrame:
    rows: List[dict] = []
    dataset = loader.dataset
    for idx in range(len(dataset)):
        sample = dataset[idx]
        derived_split = sample.get("derived_split")
        dataset_split = sample.get("dataset_split")
        rows.append(
            {
                "img_id": str(sample.get("img_id")),
                "source_split": str(source_split),
                "derived_split": str(derived_split) if derived_split is not None else "",
                "dataset_split": str(dataset_split) if dataset_split is not None else "",
                "mod": str(sample.get("mod")),
                "ty": str(sample.get("ty")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["img_id", "source_split", "derived_split", "dataset_split", "mod", "ty"],
    )


def save_loaded_samples_csv(output_path: Path, loader_map: Sequence[Tuple[str, DataLoader]]) -> None:
    frames = [collect_loaded_samples(loader, source_split) for source_split, loader in loader_map]
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["img_id", "source_split", "derived_split", "dataset_split", "mod", "ty"])
    if not df.empty:
        df = df.sort_values(["source_split", "derived_split", "dataset_split", "mod", "img_id"], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_experiment_flat_summary_df(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "experiment",
                "family",
                "arch",
                "id_mods",
                "ood_mods",
                "train_csv",
                "valid_csv",
                "test_id_csv",
                "test_ood_csv",
                "train_size",
                "valid_size",
                "test_id_size",
                "test_ood_size",
                "best_epoch",
                "best_val_loss",
                "val_loss",
                "val_acc",
                "val_r_acc",
                "val_f_acc",
                "val_ap",
                "test_id_loss",
                "test_id_acc",
                "test_id_r_acc",
                "test_id_f_acc",
                "test_id_ap",
                "test_ood_loss",
                "test_ood_acc",
                "test_ood_r_acc",
                "test_ood_f_acc",
                "test_ood_ap",
            ]
        )

    ordered_cols = [
        "experiment",
        "family",
        "arch",
        "id_mods",
        "ood_mods",
        "train_csv",
        "valid_csv",
        "test_id_csv",
        "test_ood_csv",
        "train_size",
        "valid_size",
        "test_id_size",
        "test_ood_size",
        "best_epoch",
        "best_val_loss",
        "val_loss",
        "val_acc",
        "val_r_acc",
        "val_f_acc",
        "val_ap",
        "test_id_loss",
        "test_id_acc",
        "test_id_r_acc",
        "test_id_f_acc",
        "test_id_ap",
        "test_ood_loss",
        "test_ood_acc",
        "test_ood_r_acc",
        "test_ood_f_acc",
        "test_ood_ap",
    ]
    present_cols = [c for c in ordered_cols if c in df.columns]
    return df.reindex(columns=present_cols)


def _build_detailed_sheet(rows: List[dict]) -> pd.DataFrame:
    ordered_columns: List[Tuple[str, str]] = []
    data: Dict[Tuple[str, str], List[object]] = {}
    index_labels: List[str] = []

    split_blocks = [
        ("VALID ID", "val"),
        ("TEST ID", "test_id"),
        ("TEST OOD", "test_ood"),
    ]

    for row in rows:
        index_labels.append(str(row.get("arch", "")))

    for split_title, prefix in split_blocks:
        for metric_key, metric_label in SUMMARY_METRICS:
            col_key = (split_title, metric_label)
            ordered_columns.append(col_key)
            values: List[object] = []
            for row in rows:
                values.append(row.get(f"{prefix}_{metric_key}"))
            data[col_key] = values

    table = pd.DataFrame(data, index=index_labels)
    if ordered_columns:
        table = table.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns))
    table.index.name = "arch"
    return table


def save_experiment_summary_excel(experiment_root: Path, rows: List[dict]) -> None:
    detailed_df = _build_detailed_sheet(rows)
    output_path = experiment_root / "summary_metrics.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        detailed_df.to_excel(writer, sheet_name="detailed")


def save_experiment_summary_csv(experiment_root: Path, rows: List[dict]) -> None:
    flat_df = build_experiment_flat_summary_df(rows)
    flat_df.to_csv(experiment_root / "summary_metrics.csv", index=False)


def load_dicom_scan(dicom_folder):
    slices = list()
    indexes = list()
    
    for dcm in os.listdir(dicom_folder):
        if dcm.endswith('.dcm'):
            slice_data = pydicom.dcmread(os.path.join(dicom_folder, dcm))
            slices.append(slice_data.pixel_array)
            indexes.append(-1*float(slice_data.ImagePositionPatient[2]))
 
    scan = np.stack([x for _, x in sorted(zip(indexes, slices))], 0)
    
    return scan


def scan_to_uint16(scan):
    if scan.dtype==np.uint16:
        return scan
    assert scan.dtype==np.int16
    list_min = [0, 512, 1024, 2000, 2048]
    vmin = -np.min(scan)
    if vmin not in list_min:
        f = min(5, scan.shape[0])
        o = np.stack((scan[ :f, :5, :5],
                      scan[ :f, :5,-5:],
                      scan[ :f,-5:, :5],
                      scan[ :f,-5:,-5:],
                      scan[-f:, :5, :5],
                      scan[-f:, :5,-5:],
                      scan[-f:,-5:, :5],
                      scan[-f:,-5:,-5:],),0)
        vmin = -np.median(o)
        if vmin not in list_min:
            vmin = list_min[np.argmax([np.count_nonzero(o == -1*_) for _ in list_min])]
    
    assert vmin in list_min
    return np.uint16(scan.clip(min=-vmin) + vmin)

def save_tiff_scan(dirname, scan, mode=None, compression='lzma'):
    os.makedirs(dirname, exist_ok=True)
    for index in range(scan.shape[0]):        
        Image.fromarray(scan[index], mode=mode).save(dirname+'/slide%04d.tiff' % index, compression=compression, lossless=True)
