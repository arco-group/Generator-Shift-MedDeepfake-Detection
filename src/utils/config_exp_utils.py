from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from src.data.dataloader_experiment import get_patch_dataloader
from src.data.dataloader_experiment import get_ifft_patch_dataloader
from src.data.dataloader_experiment import get_paired_embedding_dataloader
from src.data.dataloader_experiment import get_embedding_dataloader
from src.data.dataloader_experiment import get_paired_cube_dataloader

VALID_MODS = {"real", "cycle", "diffusion", "pix2pix"}
DEFAULT_REAL_MOD = "real"


@dataclass(frozen=True) 
class ExperimentConfig:
    name: str 
    family: str 
    id_mods: List[str] 
    ood_mods: List[str] 


def normalize_mod(mod: str) -> str:
    key = str(mod).strip().lower()
    if key not in VALID_MODS:
        raise ValueError(f"Unsupported mod={mod!r}. Expected one of {sorted(VALID_MODS)}")
    return key


def load_experiments(path: str | Path) -> List[ExperimentConfig]:
    path = Path(path)
    raw = json.loads(path.read_text())
    experiments = raw.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments json must contain a non-empty 'experiments' list")

    parsed: List[ExperimentConfig] = []
    for item in experiments:
        if not isinstance(item, dict):
            raise ValueError("each experiment entry must be an object")
        name = str(item["name"]).strip()
        family = str(item.get("family", "")).strip()
        id_mods = [normalize_mod(m) for m in item["id_mods"]]
        ood_mods = [normalize_mod(m) for m in item["ood_mods"]]
        if DEFAULT_REAL_MOD not in id_mods:
            raise ValueError(f"experiment {name!r} must include real in id_mods")
        if DEFAULT_REAL_MOD not in ood_mods:
            raise ValueError(f"experiment {name!r} must include real in ood_mods")
        if len(set(id_mods)) != len(id_mods):
            raise ValueError(f"experiment {name!r} has duplicated entries in id_mods")
        if len(set(ood_mods)) != len(ood_mods):
            raise ValueError(f"experiment {name!r} has duplicated entries in ood_mods")

        parsed.append(
            ExperimentConfig(
                name=name,
                family=family,
                id_mods=id_mods,
                ood_mods=ood_mods,
            )
        )

    return parsed


def load_experiment_map(path: str | Path) -> Dict[str, ExperimentConfig]:
    experiments = load_experiments(path)
    exp_map: Dict[str, ExperimentConfig] = {}
    for exp in experiments:
        if exp.name in exp_map:
            raise ValueError(f"Duplicate experiment name in json: {exp.name!r}")
        exp_map[exp.name] = exp
    return exp_map


def get_experiment(path: str | Path, name: str) -> ExperimentConfig:
    exp_map = load_experiment_map(path)
    if name not in exp_map:
        raise KeyError(
            f"Experiment {name!r} not found in {path}. Available: {sorted(exp_map)}"
        )
    return exp_map[name]

def resolve_experiment_split_dir(splits_root: str | Path, experiment: ExperimentConfig) -> Path:
    return Path(splits_root) / experiment.name

def resolve_split_csv_paths(
    splits_root: str | Path,
    experiment: ExperimentConfig,
) -> Dict[str, Path]:
    split_dir = resolve_experiment_split_dir(splits_root, experiment)
    paths = {
        "train": split_dir / "train.csv",
        "valid": split_dir / "valid.csv",
        "test_id": split_dir / "test_id.csv",
        "test_ood": split_dir / "test_ood.csv",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split CSVs for experiment {experiment.name!r} under {split_dir}: {missing}"
        )
    return paths


def build_experiment_loaders(
    experiment: ExperimentConfig,
    splits_root: str | Path,
    sample_root: str | Path,
    batch_size: int,
    num_workers: int,
    data_kind: str = "cube",
    sample_format: str = "npy",
    label_mode: str = "binary",
    real_mod: str = "real",
    pin_memory: bool = True,
    transform=None,
    norm_mode: bool = True,
    norm_stats_csv: str | Path | None = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, Path]]:
    csv_paths = resolve_split_csv_paths(splits_root, experiment)
    loaders: Dict[str, DataLoader] = {}
    data_kind = str(data_kind).strip().lower()
    if data_kind not in {"cube", "embedding", "embeddings", "ifft"}:
        raise ValueError("data_kind must be one of ['cube', 'embedding', 'embeddings', 'ifft']")

    for split_name, csv_path in csv_paths.items():
        if data_kind == "cube":
            loaders[split_name] = get_patch_dataloader(
                csv_path=str(csv_path),
                sample_root=str(sample_root),
                sample_format=sample_format,
                label_mode=label_mode,
                real_mod=real_mod,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=(split_name == "train"),
                pin_memory=pin_memory,
                transform=transform,
            )
        elif data_kind in {"embedding", "embeddings"}:
            loaders[split_name] = get_embedding_dataloader(
                csv_path=str(csv_path),
                sample_root=str(sample_root),
                label_mode=label_mode,
                real_mod=real_mod,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=(split_name == "train"),
                pin_memory=pin_memory,
            )
        else:
            loaders[split_name] = get_ifft_patch_dataloader(
                csv_path=str(csv_path),
                sample_root=str(sample_root),
                sample_format=sample_format,
                label_mode=label_mode,
                real_mod=real_mod,
                norm_mode=norm_mode,
                norm_stats_csv=str(norm_stats_csv) if norm_stats_csv is not None else None,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=(split_name == "train"),
                pin_memory=pin_memory,
            )
    return loaders, csv_paths


def build_experiment_paired_embedding_loaders(
    experiment: ExperimentConfig,
    splits_root: str | Path,
    image_root: str | Path,
    ifft_root: str | Path,
    batch_size: int,
    num_workers: int,
    label_mode: str = "binary",
    real_mod: str = "real",
    pin_memory: bool = True,
) -> Tuple[Dict[str, DataLoader], Dict[str, Path]]:
    csv_paths = resolve_split_csv_paths(splits_root, experiment)
    loaders: Dict[str, DataLoader] = {}

    for split_name, csv_path in csv_paths.items():
        loaders[split_name] = get_paired_embedding_dataloader(
            csv_path=str(csv_path),
            image_root=str(image_root),
            ifft_root=str(ifft_root),
            label_mode=label_mode,
            real_mod=real_mod,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split_name == "train"),
            pin_memory=pin_memory,
        )
    return loaders, csv_paths


def build_experiment_paired_cube_loaders(
    experiment: ExperimentConfig,
    splits_root: str | Path,
    image_root: str | Path,
    ifft_root: str | Path,
    batch_size: int,
    num_workers: int,
    label_mode: str = "binary",
    real_mod: str = "real",
    sample_format: str = "npy",
    image_sample_format: str | None = None,
    ifft_sample_format: str | None = None,
    ifft_norm_mode: bool = False,
    ifft_norm_stats_csv: str | Path | None = None,
    pin_memory: bool = True,
) -> Tuple[Dict[str, DataLoader], Dict[str, Path]]:
    csv_paths = resolve_split_csv_paths(splits_root, experiment)
    loaders: Dict[str, DataLoader] = {}

    for split_name, csv_path in csv_paths.items():
        loaders[split_name] = get_paired_cube_dataloader(
            csv_path=str(csv_path),
            image_root=str(image_root),
            ifft_root=str(ifft_root),
            sample_format=sample_format,
            label_mode=label_mode,
            real_mod=real_mod,
            image_sample_format=image_sample_format,
            ifft_sample_format=ifft_sample_format,
            ifft_norm_mode=ifft_norm_mode,
            ifft_norm_stats_csv=str(ifft_norm_stats_csv) if ifft_norm_stats_csv is not None else None,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split_name == "train"),
            pin_memory=pin_memory,
        )
    return loaders, csv_paths
