#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch
import pandas as pd
from tqdm import tqdm
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_spectre.spectre import MODEL_CONFIGS, SpectreImageFeatureExtractor
from src.data.dataloader_experiment import get_ifft_patch_dataloader, get_patch_dataloader

def parse_grid_size(raw: str) -> Tuple[int, int, int]:
    cleaned = raw.replace("x", ",")
    parts = [p for p in cleaned.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Invalid grid size: {raw}")
    return tuple(int(p) for p in parts) 


def parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")


def split_volume_into_windows(volumes: torch.Tensor, grid_size: Sequence[int]) -> torch.Tensor:
    if volumes.ndim != 5:
        raise ValueError(f"Expected a 5D tensor (B,1,D,H,W), got shape {volumes.shape}")
    g_h, g_w, g_d = grid_size

    volumes_hwd = volumes.permute(0, 1, 3, 4, 2).contiguous()
    _, _, H, W, D = volumes_hwd.shape

    if (H % g_h) or (W % g_w) or (D % g_d):
        raise ValueError(
            f"Shape {H}x{W}x{D} is not divisible by the grid {grid_size} "
            "(make sure the volumes have already been resampled/resized correctly)."
        )

    h_size, w_size, d_size = H // g_h, W // g_w, D // g_d
    windows = volumes_hwd.view(-1, 1, g_h, h_size, g_w, w_size, g_d, d_size)
    windows = windows.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    windows = windows.view(-1, g_h * g_w * g_d, 1, h_size, w_size, d_size)
    return windows


@torch.no_grad()
def extract_embeddings(
    model: SpectreImageFeatureExtractor,
    loader: Iterable,
    output_dir: Path,
    device: torch.device,
    grid_size: Tuple[int, int, int],
    set_name: str,
    ty: str,
):
    def _python_value(value):
        if torch.is_tensor(value):
            return value.item()
        return value

    for batch in tqdm(loader, desc="Extracting", unit="vol"):
        cubes = batch["cube"].to(device, non_blocking=True)
        names = [f"{img}" for img in batch["img_id"]]
        mods = batch["mod"]
        orig_ids = batch["orig_id"]
        labels = batch["label"]
        tys = batch["ty"]
        source_paths = batch["path"]

        windows = split_volume_into_windows(cubes, grid_size)
        windows = windows.to(dtype=torch.float32)
        _, features = model(windows, grid_size=grid_size)
        features = features.cpu()

        for idx, feat in enumerate(features):
            name = names[idx]
            mod = mods[idx]
            save_dir = output_dir / set_name / ty / mod
            samples_dir = save_dir / "samples"
            metadata_dir = save_dir / "metadata"
            samples_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)

            save_path = samples_dir / f"{name}.pt"
            metadata_path = metadata_dir / f"{name}.csv"

            if feat.ndim >= 2:
                feat = feat[0].contiguous().clone()
            if not save_path.exists():
                torch.save(feat, save_path)
            else:
                tqdm.write(f"[SKIP] {save_path} already exists")

            if not metadata_path.exists():
                metadata_row = {
                    "img_id": name,
                    "orig_id": orig_ids[idx],
                    "mod": mod,
                    "ty": tys[idx],
                    "split": set_name,
                    "label": _python_value(labels[idx]),
                    "source_path": source_paths[idx],
                    "path": str(save_path),
                }
                pd.DataFrame([metadata_row]).to_csv(metadata_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from the M3Dsynth dataset.")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        help="List of experiment names to process. Each experiment must have a split CSV under data/splits/<experiment>/.",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory containing the samples referenced by the CSV.")
    parser.add_argument("--ty", type=str, default="inj", help="Manipulation type.")
    parser.add_argument("--set-name", type=str, default="train", help="Dataset split to process (train, valid, or test).")
    parser.add_argument("--grid-size", type=parse_grid_size, default="3,3,4", help="Window grid passed to the model.")
    parser.add_argument("--output-dir", type=Path, default=Path("./embeddings"), help="Base directory where the embeddings (.pt) will be saved.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used for inference.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes.")
    parser.add_argument(
        "--source-representation",
        type=str,
        default="img",
        choices=["img", "ifft"],
        help="Embedding source: 'img' uses get_dataloader, 'ifft' uses get_ifft_patch_dataloader.",
    )
    parser.add_argument("--sample_format", type=str, default="npy", choices=["npy", "nii.gz"],
                        help="Format of the saved cubes on disk.")
    parser.add_argument(
        "--norm_mode_ifft",
        type=parse_bool,
        default=True,
        help="Whether to normalize the IFFT volume before passing it to the encoder.",
    )
    parser.add_argument("--label_mode", type=str, default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--real_mod", type=str, default="real")
    parser.add_argument(
        "--norm_stats_csv",
        type=Path,
        default=None,
        help="CSV containing global min/max statistics used to normalize IFFT volumes.",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("./data/splits"),
        help="Root directory containing experiment split CSVs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = tuple(args.grid_size)

    config = MODEL_CONFIGS["spectre-large-pretrained"]
    model = SpectreImageFeatureExtractor.from_config(config)
    model.eval().to(device)

    for experiment in args.experiments:
        csv_path = args.splits_root / experiment / f"{args.set_name}.csv"
        output_dir = args.output_dir / experiment

        if args.source_representation == "img":
            loader = get_patch_dataloader(
                csv_path=str(csv_path),
                sample_root=str(args.data_root),
                sample_format=args.sample_format,
                label_mode=args.label_mode,
                real_mod=args.real_mod,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=(device.type == "cuda"),
            )
        elif args.source_representation == "ifft":
            loader = get_ifft_patch_dataloader(
                csv_path=str(csv_path),
                sample_root=str(args.data_root),
                sample_format=args.sample_format,
                label_mode=args.label_mode,
                real_mod=args.real_mod,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=(device.type == "cuda"),
                norm_mode=args.norm_mode_ifft,
                norm_stats_csv=str(args.norm_stats_csv) if args.norm_stats_csv is not None else None,
            )
        else:
            raise ValueError(
                f"Unsupported source-representation={args.source_representation!r}. "
                "Expected 'img' or 'ifft'."
            )

        extract_embeddings(
            model,
            loader,
            output_dir,
            device=device,
            grid_size=grid_size,
            set_name=args.set_name,
            ty=args.ty,
        )


if __name__ == "__main__":
    main()
