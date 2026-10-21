from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional dependency for .nii.gz support
    nib = None


SUPPORTED_SAMPLE_FORMATS = {"npy", "nii.gz"}


def _normalize_sample_format(sample_format: str) -> str:
    sample_format = sample_format.lower().strip()
    if sample_format not in SUPPORTED_SAMPLE_FORMATS:
        raise ValueError(
            f"Unsupported sample_format={sample_format!r}. "
            f"Expected one of {sorted(SUPPORTED_SAMPLE_FORMATS)}"
        )
    return sample_format


def _sample_dirname(sample_format: str) -> str:
    return "samples" if sample_format == "npy" else "samples_nii"


def _sample_suffix(sample_format: str) -> str:
    return ".npy" if sample_format == "npy" else ".nii.gz"


def _read_metadata_files(metadata_files: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in metadata_files:
        frame = pd.read_csv(csv_path)
        frame["metadata_csv"] = str(csv_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_metadata_files(
    root: Path,
    split: Optional[str] = None,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
) -> List[Path]:
    root = Path(root)
    search_root = root if split is None else root / split
    if not search_root.exists():
        raise FileNotFoundError(f"Root not found: {search_root}")

    mods_set = set(mods) if mods is not None else None
    metadata_files: List[Path] = []
    for csv_path in search_root.rglob("metadata/*.csv"):
        parts = csv_path.relative_to(root).parts
        if split is not None and (len(parts) < 4 or parts[0] != split):
            continue
        if ty is not None and (len(parts) < 4 or parts[1] != ty):
            continue
        if mods_set is not None and (len(parts) < 4 or parts[2] not in mods_set):
            continue
        metadata_files.append(csv_path)
    return sorted(metadata_files)


def _load_sample(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.name.endswith(".nii.gz"):
        if nib is None:
            raise ImportError("nibabel is required to load .nii.gz samples")
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    raise ValueError(f"Unsupported sample file: {path}")


class PreprocessingPatchDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        split: str,
        ty: Optional[str] = None,
        mods: Optional[Iterable[str]] = None,
        sample_format: str = "npy",
    ):
        self.root = Path(root)
        self.split = split
        self.ty = ty
        self.mods = list(mods) if mods is not None else None
        self.sample_format = _normalize_sample_format(sample_format)
        self.samples_dirname = _sample_dirname(self.sample_format)
        self.sample_suffix = _sample_suffix(self.sample_format)

        metadata_files = collect_metadata_files(
            self.root,
            split=self.split,
            ty=self.ty,
            mods=self.mods,
        )
        if not metadata_files:
            raise FileNotFoundError(
                f"No metadata CSV files found for split={self.split!r}, ty={self.ty!r}, mods={self.mods!r}"
            )

        self.metadata = _read_metadata_files(metadata_files).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def _sample_path(self, metadata_csv: Path, sample_name: str) -> Path:
        save_dir = metadata_csv.parent.parent
        return save_dir / self.samples_dirname / f"{sample_name}{self.sample_suffix}"

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        metadata_csv = Path(row["metadata_csv"])
        sample_name = metadata_csv.stem
        sample_path = self._sample_path(metadata_csv, sample_name)

        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")

        cube_np = _load_sample(sample_path)
        cube = torch.from_numpy(cube_np).float()
        if cube.ndim == 3:
            cube = cube.unsqueeze(0)

        patch_min_col = "patch_min" if "patch_min" in row else "min"
        patch_max_col = "patch_max" if "patch_max" in row else "max"

        return {
            "cube": cube,
            "img_id": row["img_id"],
            "orig_id": row["orig_id"],
            "split": self.split,
            "mod": row["mod"],
            "ty": row["ty"],
            "shape": row["shape"],
            "patch_min": float(row[patch_min_col]),
            "patch_max": float(row[patch_max_col]),
            "path": str(sample_path),
        }


def get_dataloader(
    root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    sample_format: str = "npy",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
):
    dataset = PreprocessingPatchDataset(
        root=root,
        split=split,
        ty=ty,
        mods=mods,
        sample_format=sample_format,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
