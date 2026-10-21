from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional dependency for .nii.gz support
    nib = None


SUPPORTED_SAMPLE_FORMATS = {"npy", "nii.gz"}
SUPPORTED_LABEL_MODES = {"binary", "multiclass"}

MULTICLASS_LABELS = {
    "real": 0,
    "cycle": 1,
    "diffusion": 2,
    "pix2pix": 3,
}

CLASS_NAMES = ["real", "cycle", "diffusion", "pix2pix"]

DEFAULT_SPLIT_REMAP = {"val": "valid"}

def _normalize_sample_format(sample_format: str) -> str:
    sample_format = str(sample_format).lower().strip()
    if sample_format not in SUPPORTED_SAMPLE_FORMATS:
        raise ValueError(
            "Unsupported sample_format={!r}. Expected one of {}".format(
                sample_format, sorted(SUPPORTED_SAMPLE_FORMATS)
            )
        )
    return sample_format


def _sample_dirname(sample_format: str) -> str:
    return "samples" if sample_format == "npy" else "samples_nii"


def _sample_suffix(sample_format: str) -> str:
    return ".npy" if sample_format == "npy" else ".nii.gz"


def _normalize_label_mode(label_mode: str) -> str:
    label_mode = str(label_mode).lower().strip()
    if label_mode not in SUPPORTED_LABEL_MODES:
        raise ValueError(
            "Unsupported label_mode={!r}. Expected one of {}".format(
                label_mode, sorted(SUPPORTED_LABEL_MODES)
            )
        )
    return label_mode


def _normalize_dataset_split(value: object) -> str:
    split = str(value).lower().strip()
    return DEFAULT_SPLIT_REMAP.get(split, split)


def _label_from_mod(mod: str, label_mode: str, real_mod: str) -> int:
    mod = str(mod).strip().lower()
    real_mod = str(real_mod).strip().lower()
    if label_mode == "binary":
        return 0 if mod == real_mod else 1
    if mod not in MULTICLASS_LABELS:
        raise ValueError(
            "Unsupported mod={!r} for multiclass labels. Expected one of {}".format(
                mod, sorted(MULTICLASS_LABELS)
            )
        )
    return MULTICLASS_LABELS[mod]


def _load_sample(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    if path.name.endswith(".nii.gz"):
        if nib is None:
            raise ImportError("nibabel is required to load .nii.gz samples")
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    raise ValueError("Unsupported sample file: {!r}".format(path))


def _load_embedding(path: Path) -> torch.Tensor:
    if path.suffix != ".pt":
        raise ValueError("Unsupported embedding file: {!r}".format(path))
    return torch.load(str(path), map_location="cpu")


def _read_selected_csv(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(str(csv_path))
    if frame.empty:
        raise ValueError("Selected CSV is empty: {!r}".format(csv_path))

    if "label" in frame.columns and "label_text" not in frame.columns:
        frame = frame.rename(columns={"label": "label_text"})

    return frame.reset_index(drop=True)


def _metadata_global_min_max(metadata: pd.DataFrame) -> Tuple[float, float]:
    """Extract global min/max from metadata columns."""
    if {"min", "max"}.issubset(metadata.columns):
        min_col, max_col = "min", "max"
    elif {"patch_min", "patch_max"}.issubset(metadata.columns):
        min_col, max_col = "patch_min", "patch_max"
    else:
        raise ValueError(
            "Metadata must contain either min/max or patch_min/patch_max columns"
        )

    mins = pd.to_numeric(metadata[min_col], errors="coerce")
    maxs = pd.to_numeric(metadata[max_col], errors="coerce")
    global_min = float(mins.min())
    global_max = float(maxs.max())
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError("Unable to compute finite global min/max from metadata")
    return global_min, global_max


def _require_columns(frame: pd.DataFrame, required: Iterable[str], csv_path: Path) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(
            "Missing required columns {} in {}".format(
                missing, csv_path
            )
        )


def _resolve_cube_sample_path(
    sample_root: Path,
    row: pd.Series,
    sample_format: str,
    dataset_split_col: str,
    ty_col: str,
    mod_col: str,
    img_id_col: str,
) -> Path:
    return (
        sample_root
        / _normalize_dataset_split(row[dataset_split_col])
        / str(row[ty_col]).strip()
        / str(row[mod_col]).strip().lower()
        / _sample_dirname(sample_format)
        / "{}{}".format(str(row[img_id_col]).strip(), _sample_suffix(sample_format))
    )


def _resolve_embedding_path(
    sample_root: Path,
    row: pd.Series,
    derived_split_col: str,
    ty_col: str,
    mod_col: str,
    img_id_col: str,
) -> Path:
    return (
        sample_root
        / _normalize_dataset_split(row[derived_split_col])
        / str(row[ty_col]).strip()
        / str(row[mod_col]).strip().lower()
        / "samples"
        / "{}.pt".format(str(row[img_id_col]).strip())
    )

###################################### PATCH IMG ############################################
class PatchDataset(Dataset):

    def __init__(
        self,
        csv_path: str, 
        sample_root: str,
        sample_format: str = "npy",
        label_mode: str = "binary",
        real_mod: str = "real",
        dataset_split_col: str = "dataset_split",
        ty_col: str = "ty",
        mod_col: str = "mod",
        img_id_col: str = "img_id",
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.sample_root = Path(sample_root)
        self.sample_format = _normalize_sample_format(sample_format)
        self.label_mode = _normalize_label_mode(label_mode)
        self.real_mod = str(real_mod).strip().lower()
        self.dataset_split_col = dataset_split_col
        self.ty_col = ty_col
        self.mod_col = mod_col
        self.img_id_col = img_id_col
        self.derived_split_name = self.csv_path.stem
        self.transform = transform

        self.metadata = _read_selected_csv(self.csv_path)
        _require_columns(
            self.metadata,
            [self.dataset_split_col, self.ty_col, self.mod_col, self.img_id_col],
            self.csv_path,
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def _sample_path(self, row: pd.Series) -> Path:
        return _resolve_cube_sample_path(
            self.sample_root,
            row,
            self.sample_format,
            self.dataset_split_col,
            self.ty_col,
            self.mod_col,
            self.img_id_col,
        )

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]

        sample_path = self._sample_path(row)
        if not sample_path.exists():
            raise FileNotFoundError("Sample file not found: {}".format(sample_path))
        cube_np = _load_sample(sample_path)

        cube_min = float(np.min(cube_np))
        cube_max = float(np.max(cube_np))
        eps = 1e-5
        if cube_min < -eps or cube_max > 1.0 + eps:
            raise ValueError(
                f"Sample outside [0, 1] range: path={sample_path}, min={cube_min:.6f}, max={cube_max:.6f}"
            )

        cube = torch.from_numpy(cube_np).float()
        if cube.ndim == 3:
            cube = cube.unsqueeze(0)
        if self.transform is not None:
            cube = self.transform(cube)
        

        derived_split = str(row.get("derived_split"))
        dataset_split = _normalize_dataset_split(row[self.derived_split_col])
        label_text = row.get("label_text")
        label = _label_from_mod(row[self.mod_col], self.label_mode, self.real_mod)

        return {
            "cube": cube,
            "label": label,
            "label_text": label_text,
            "img_id": row[self.img_id_col],
            "orig_id": row["orig_id"] if "orig_id" in row else None,
            "derived_split": derived_split,
            "dataset_split": dataset_split,
            "experiment": row["experiment"] if "experiment" in row else None,
            "family": row["family"] if "family" in row else None,
            "mod": row[self.mod_col],
            "ty": row[self.ty_col],
            "shape": tuple(cube.shape),
            "path": str(sample_path),
        }


def get_patch_dataloader(
    csv_path: str,
    sample_root: str,
    sample_format: str = "npy",
    label_mode: str = "binary",
    real_mod: str = "real",
    dataset_split_col: str = "dataset_split",
    ty_col: str = "ty",
    mod_col: str = "mod",
    img_id_col: str = "img_id",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
    transform=None,
):
    dataset = PatchDataset(
        csv_path=csv_path,
        sample_root=sample_root,
        sample_format=sample_format,
        label_mode=label_mode,
        real_mod=real_mod,
        dataset_split_col=dataset_split_col,
        ty_col=ty_col,
        mod_col=mod_col,
        img_id_col=img_id_col,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
##############################################################################################

###################################### PATCH IFFT ############################################
class ifftPatchDataset(PatchDataset):
    def __init__(
        self,
        csv_path: str,
        sample_root: str,
        sample_format: str = "npy",
        label_mode: str = "binary",
        real_mod: str = "real",
        norm_mode: bool = True,
        norm_stats_csv: Optional[str] = None,  
        dataset_split_col: str = "dataset_split",
        ty_col: str = "ty",
        mod_col: str = "mod",
        img_id_col: str = "img_id",
    ):
        super().__init__(
            csv_path=csv_path,
            sample_root=sample_root,
            sample_format=sample_format,
            label_mode=label_mode,
            real_mod=real_mod,
            dataset_split_col=dataset_split_col,
            ty_col=ty_col,
            mod_col=mod_col,
            img_id_col=img_id_col,
        ) 

        self.norm_mode = bool(norm_mode)
        self.norm_stats_csv = Path(norm_stats_csv) if norm_stats_csv is not None else None
        self.global_min = None
        self.global_max = None

        if self.norm_mode:
            if self.norm_stats_csv is None:
                raise ValueError(
                    "norm_stats_csv must be provided when norm_mode is True"
                )
            stats_frame = pd.read_csv(str(self.norm_stats_csv))
            self.global_min, self.global_max = _metadata_global_min_max(stats_frame)

    def _normalize_fft_volume(self, volume: np.ndarray) -> np.ndarray:
        if not self.norm_mode or self.global_min is None or self.global_max is None:
            return volume.astype(np.float32)
        if self.global_max <= self.global_min:
            return np.zeros_like(volume, dtype=np.float32)
        normalized = (volume.astype(np.float32) - self.global_min) / (
            self.global_max - self.global_min
        )
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        
        sample_path = self._sample_path(row)
        if not sample_path.exists():
            raise FileNotFoundError("Sample file not found: {}".format(sample_path))
        cube_np = _load_sample(sample_path)
        
        cube_np = self._normalize_fft_volume(cube_np)

        cube = torch.from_numpy(cube_np).float()
        if cube.ndim == 3:
            cube = cube.unsqueeze(0)

        derived_split = str(row["derived_split"])
        dataset_split = _normalize_dataset_split(row[self.dataset_split_col])
        label_text = row.get("label_text")
        label = _label_from_mod(row[self.mod_col], self.label_mode, self.real_mod)

        return {
            "cube": cube,
            "label": label,
            "label_text": label_text,
            "img_id": row[self.img_id_col],
            "orig_id": row["orig_id"] if "orig_id" in row else None,
            "derived_split": derived_split,
            "dataset_split": dataset_split,
            "experiment": row["experiment"] if "experiment" in row else None,
            "family": row["family"] if "family" in row else None,
            "mod": row[self.mod_col],
            "ty": row[self.ty_col],
            "shape": tuple(cube.shape),
            "path": str(sample_path),
        }


def get_ifft_patch_dataloader(
    csv_path: str,
    sample_root: str,
    sample_format: str = "npy",
    label_mode: str = "binary",
    real_mod: str = "real",
    norm_mode: bool = True,
    norm_stats_csv: Optional[str] = None,
    dataset_split_col: str = "dataset_split",
    ty_col: str = "ty",
    mod_col: str = "mod",
    img_id_col: str = "img_id",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
):
    dataset = ifftPatchDataset(
        csv_path=csv_path,
        sample_root=sample_root,
        sample_format=sample_format,
        label_mode=label_mode,
        real_mod=real_mod,
        norm_mode=norm_mode,
        norm_stats_csv=norm_stats_csv,
        dataset_split_col=dataset_split_col,
        ty_col=ty_col,
        mod_col=mod_col,
        img_id_col=img_id_col,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
##############################################################################################


###################################### PAIRED PATCH EMBEDDINGS ###############################
class PairedEmbeddingDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        ifft_root: str,
        label_mode: str = "binary",
        real_mod: str = "real",
        derived_split_col: str = "derived_split",
        ty_col: str = "ty",
        mod_col: str = "mod",
        img_id_col: str = "img_id",
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.ifft_root = Path(ifft_root)
        self.label_mode = _normalize_label_mode(label_mode)
        self.real_mod = str(real_mod).strip().lower()
        self.derived_split_col = derived_split_col
        self.ty_col = ty_col
        self.mod_col = mod_col
        self.img_id_col = img_id_col
        self.derived_split_name = self.csv_path.stem

        self.metadata = _read_selected_csv(self.csv_path)
        _require_columns(
            self.metadata,
            [self.derived_split_col, self.ty_col, self.mod_col, self.img_id_col],
            self.csv_path,
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def _image_path(self, row: pd.Series) -> Path:
        return _resolve_embedding_path(
            self.image_root,
            row,
            self.derived_split_col,
            self.ty_col,
            self.mod_col,
            self.img_id_col,
        )

    def _ifft_path(self, row: pd.Series) -> Path:
        return _resolve_embedding_path(
            self.ifft_root,
            row,
            self.derived_split_col,
            self.ty_col,
            self.mod_col,
            self.img_id_col,
        )

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]

        image_path = self._image_path(row)
        ifft_path = self._ifft_path(row)

        if not image_path.exists():
            raise FileNotFoundError("Embedding file not found: {}".format(image_path))
        if not ifft_path.exists():
            raise FileNotFoundError("Embedding file not found: {}".format(ifft_path))

        image_embedding = _load_embedding(image_path)
        ifft_embedding = _load_embedding(ifft_path)

        derived_split = str(row.get("derived_split"))
        label = _label_from_mod(row[self.mod_col], self.label_mode, self.real_mod)

        return {
            "image_embedding": image_embedding.float().view(-1),
            "ifft_embedding": ifft_embedding.float().view(-1),
            "label": torch.tensor(label, dtype=torch.float32),
            "label_text": row.get("label_text"),
            "img_id": row[self.img_id_col],
            "orig_id": row["orig_id"] if "orig_id" in row else None,
            "split": derived_split,
            "derived_split": derived_split,
            "experiment": row["experiment"] if "experiment" in row else None,
            "family": row["family"] if "family" in row else None,
            "mod": row[self.mod_col],
            "ty": row[self.ty_col],
            "image_path": str(image_path),
            "ifft_path": str(ifft_path),
        }


def get_paired_embedding_dataloader(
    csv_path: str,
    image_root: str,
    ifft_root: str,
    label_mode: str = "binary",
    real_mod: str = "real",
    derived_split_col: str = "derived_split",
    ty_col: str = "ty",
    mod_col: str = "mod",
    img_id_col: str = "img_id",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
):
    dataset = PairedEmbeddingDataset(
        csv_path=csv_path,
        image_root=image_root,
        ifft_root=ifft_root,
        label_mode=label_mode,
        real_mod=real_mod,
        derived_split_col=derived_split_col,
        ty_col=ty_col,
        mod_col=mod_col,
        img_id_col=img_id_col,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )

##############################################################################################


###################################### SINGLE PATCH EMBEDDINGS ######################################
class EmbeddingDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        sample_root: str,
        label_mode: str = "binary",
        real_mod: str = "real",
        derived_split_col: str = "derived_split",
        ty_col: str = "ty",
        mod_col: str = "mod",
        img_id_col: str = "img_id",
    ):
        self.csv_path = Path(csv_path)
        self.sample_root = Path(sample_root)
        self.label_mode = _normalize_label_mode(label_mode)
        self.real_mod = str(real_mod).strip().lower()
        self.derived_split_col = derived_split_col
        self.ty_col = ty_col
        self.mod_col = mod_col
        self.img_id_col = img_id_col
        self.derived_split_name = self.csv_path.stem

        self.metadata = _read_selected_csv(self.csv_path)
        _require_columns(
            self.metadata,
            [self.derived_split_col, self.ty_col, self.mod_col, self.img_id_col],
            self.csv_path,
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def _sample_path(self, row: pd.Series) -> Path:
        return _resolve_embedding_path(
            self.sample_root,
            row,
            self.derived_split_col,
            self.ty_col,
            self.mod_col,
            self.img_id_col,
        )

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]

        sample_path = self._sample_path(row)
        if not sample_path.exists():
            raise FileNotFoundError("Embedding file not found: {}".format(sample_path))

        embedding = _load_embedding(sample_path)
        
        derived_split = str(row.get("derived_split"))
        label_text = row.get("label_text")
        label = _label_from_mod(row[self.mod_col], self.label_mode, self.real_mod)

        return {
            "embedding": embedding.float().view(-1),
            "label": label,
            "label_text": label_text,
            "img_id": row[self.img_id_col],
            "orig_id": row["orig_id"] if "orig_id" in row else None,
            "split": derived_split,
            "derived_split": derived_split,
            "experiment": row["experiment"] if "experiment" in row else None,
            "family": row["family"] if "family" in row else None,
            "mod": row[self.mod_col],
            "ty": row[self.ty_col],
            "path": str(sample_path),
        }


def get_embedding_dataloader(
    csv_path: str,
    sample_root: str,
    label_mode: str = "binary",
    real_mod: str = "real",
    derived_split_col: str = "derived_split",
    ty_col: str = "ty",
    mod_col: str = "mod",
    img_id_col: str = "img_id",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
):
    dataset = EmbeddingDataset(
        csv_path=csv_path,
        sample_root=sample_root,
        label_mode=label_mode,
        real_mod=real_mod,
        derived_split_col=derived_split_col,
        ty_col=ty_col,
        mod_col=mod_col,
        img_id_col=img_id_col,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
##############################################################################################


###################################### PAIRED PATCHES ###################################
class PairedCubeDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        ifft_root: str,
        sample_format: str = "npy",
        label_mode: str = "binary",
        real_mod: str = "real",
        image_sample_format: Optional[str] = None,
        ifft_sample_format: Optional[str] = None,
        dataset_split_col: str = "dataset_split",
        ty_col: str = "ty",
        mod_col: str = "mod",
        img_id_col: str = "img_id",
        ifft_norm_mode: bool = False,
        ifft_norm_stats_csv: Optional[str] = None,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.ifft_root = Path(ifft_root)
        self.sample_format = _normalize_sample_format(sample_format)
        self.image_sample_format = _normalize_sample_format(image_sample_format or sample_format)
        self.ifft_sample_format = _normalize_sample_format(ifft_sample_format or sample_format)
        self.label_mode = _normalize_label_mode(label_mode)
        self.real_mod = str(real_mod).strip().lower()
        self.dataset_split_col = dataset_split_col
        self.ty_col = ty_col
        self.mod_col = mod_col
        self.img_id_col = img_id_col
        self.derived_split_name = self.csv_path.stem
        self.ifft_norm_mode = bool(ifft_norm_mode)
        self.ifft_norm_stats_csv = Path(ifft_norm_stats_csv) if ifft_norm_stats_csv is not None else None
        self.global_min = None
        self.global_max = None

        self.metadata = _read_selected_csv(self.csv_path)
        _require_columns(
            self.metadata,
            [self.dataset_split_col, self.ty_col, self.mod_col, self.img_id_col],
            self.csv_path,
        )
        if self.ifft_norm_mode:
            if self.ifft_norm_stats_csv is None:
                raise ValueError(
                    "ifft_norm_mode=True requires ifft_norm_stats_csv to be provided"
                )
            stats_frame = pd.read_csv(str(self.ifft_norm_stats_csv))
            self.global_min, self.global_max = _metadata_global_min_max(stats_frame)

    def __len__(self) -> int:
        return len(self.metadata)

    def _image_path(self, row: pd.Series) -> Path:
        return _resolve_cube_sample_path(
            self.image_root,
            row,
            self.image_sample_format,
            self.dataset_split_col,
            self.ty_col,
            self.mod_col,
            self.img_id_col,
        )

    def _ifft_path(self, row: pd.Series) -> Path:
        return _resolve_cube_sample_path(
            self.ifft_root,
            row,
            self.ifft_sample_format,
            self.dataset_split_col,
            self.ty_col,
            self.mod_col,
            self.img_id_col,
        )

    def _normalize_ifft_volume(self, volume: np.ndarray) -> np.ndarray:
        if not self.ifft_norm_mode or self.global_min is None or self.global_max is None:
            return volume.astype(np.float32)
        if self.global_max <= self.global_min:
            return np.zeros_like(volume, dtype=np.float32)
        normalized = (volume.astype(np.float32) - self.global_min) / (
            self.global_max - self.global_min
        )
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _to_cube_tensor(self, array: np.ndarray) -> torch.Tensor:
        cube = torch.from_numpy(array).float()
        if cube.ndim == 3:
            cube = cube.unsqueeze(0)
        return cube

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]

        image_path = self._image_path(row)
        ifft_path = self._ifft_path(row)

        if not image_path.exists():
            raise FileNotFoundError("Sample file not found: {}".format(image_path))
        if not ifft_path.exists():
            raise FileNotFoundError("Sample file not found: {}".format(ifft_path))

        image_np = _load_sample(image_path)
        ifft_np = _load_sample(ifft_path)
        ifft_np = self._normalize_ifft_volume(ifft_np)

        image_cube = self._to_cube_tensor(image_np)
        ifft_cube = self._to_cube_tensor(ifft_np)

        derived_split = str(row.get("derived_split"))
        label_text = row.get("label_text")
        label = _label_from_mod(row[self.mod_col], self.label_mode, self.real_mod)

        return {
            "image_cube": image_cube,
            "ifft_cube": ifft_cube,
            "label": torch.tensor(label, dtype=torch.float32),
            "label_text": label_text,
            "img_id": row[self.img_id_col],
            "orig_id": row["orig_id"] if "orig_id" in row else None,
            "split": derived_split,
            "derived_split": derived_split,
            "experiment": row["experiment"] if "experiment" in row else None,
            "family": row["family"] if "family" in row else None,
            "mod": row[self.mod_col],
            "ty": row[self.ty_col],
            "image_path": str(image_path),
            "ifft_path": str(ifft_path),
        }


def get_paired_cube_dataloader(
    csv_path: str,
    image_root: str,
    ifft_root: str,
    sample_format: str = "npy",
    label_mode: str = "binary",
    real_mod: str = "real",
    image_sample_format: Optional[str] = None,
    ifft_sample_format: Optional[str] = None,
    dataset_split_col: str = "dataset_split",
    ty_col: str = "ty",
    mod_col: str = "mod",
    img_id_col: str = "img_id",
    ifft_norm_mode: bool = False,
    ifft_norm_stats_csv: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
):
    dataset = PairedCubeDataset(
        csv_path=csv_path,
        image_root=image_root,
        ifft_root=ifft_root,
        sample_format=sample_format,
        label_mode=label_mode,
        real_mod=real_mod,
        image_sample_format=image_sample_format,
        ifft_sample_format=ifft_sample_format,
        dataset_split_col=dataset_split_col,
        ty_col=ty_col,
        mod_col=mod_col,
        img_id_col=img_id_col,
        ifft_norm_mode=ifft_norm_mode,
        ifft_norm_stats_csv=ifft_norm_stats_csv,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )

##############################################################################################
