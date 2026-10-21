#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing.M3DSynth_dataset_utils import CTPatchExtractor, get_table_date


@torch.no_grad()
def save_cubes(
    loader: Iterable,
    output_dir: Path,
    device: torch.device,
    set_name: str,
    ty: str,
    sample_format: str,
) -> None:
    samples_dirname = "samples" if sample_format == "npy" else "samples_nii"
    sample_suffix = ".npy" if sample_format == "npy" else ".nii.gz"

    for batch in tqdm(loader, desc="Saving cubes", unit="vol"):
        cubes = batch["cube"].to(device, non_blocking=True)
        names = [f"{img}" for img in batch["img_id"]]
        mods = batch["mod"]
        orig_ids = batch["orig_id"]
        tys = batch["ty"]
        patch_mins = batch["patch_min"]
        patch_maxs = batch["patch_max"]
        for i, (name, mod, orig_id, ty, patch_min, patch_max) in enumerate(
            zip(names, mods, orig_ids, tys, patch_mins, patch_maxs)
        ):
            save_dir = output_dir / set_name / ty / mod
            samples_dir = save_dir / samples_dirname
            metadata_dir = save_dir / "metadata"
            samples_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)
            sample_path = samples_dir / f"{name}{sample_suffix}"
            metadata_path = metadata_dir / f"{name}.csv"

            cube_np = cubes[i].cpu().numpy()
            if cube_np.ndim == 4 and cube_np.shape[0] == 1:
                cube_np = cube_np[0]  # (1, D, H, W) -> (D, H, W)

            shape_value = tuple(cube_np.shape)

            if sample_path.exists() and metadata_path.exists():
                tqdm.write(f"[SKIP] {name} already saved")
                continue

            if not sample_path.exists():
                if sample_format == "npy":
                    np.save(sample_path, cube_np.astype(np.float32))
                else:
                    nifti_img = nib.Nifti1Image(cube_np.astype(np.float32), affine=np.eye(4))
                    nib.save(nifti_img, sample_path)

            if not metadata_path.exists():
                metadata_df = pd.DataFrame([
                    {
                        "img_id": name,
                        "orig_id": orig_id,
                        "mod": mod,
                        "ty": ty,
                        "shape": str(shape_value),
                        "patch_min": float(patch_min),
                        "patch_max": float(patch_max),
                        "path": str(sample_path),
                    }
                ])
                metadata_df.to_csv(metadata_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract patches and save them as .npy or .nii.gz files."
    )
    parser.add_argument("--data-root", type=Path, required=False, help="Fallback root directory for both real and fake scans.")
    parser.add_argument("--data-root-real", type=Path, required=False, help="Root directory containing real scans, organized as real/scan/<orig_id>.")
    parser.add_argument("--data-root-fake", type=Path, required=False, help="Root directory containing manipulated scans, organized as <mod>/scan/<img_id> and <mod>/label/<img_id>.")
    parser.add_argument("--data-csv", type=Path, required=True, help="Main dataset CSV (data.csv).")
    parser.add_argument("--sets-csv", type=Path, required=True, help="CSV that maps orig_id values to dataset splits (sets.csv).")
    parser.add_argument("--mods", type=str, required=True, help="Comma-separated list of modalities to include, for example 'real,cycle,diffusion'.")
    parser.add_argument("--ty", type=str, required=True, help="Value of the ty column to keep.")
    parser.add_argument("--set-name", type=str, default="train", help="Dataset split name to process (train, valid, or test).")
    parser.add_argument(
        "--lidc-stats-csv",
        type=Path,
        help="CSV containing LIDC scan statistics used to convert TIFF volumes back to HU using orig_id-specific metadata.",
    )
    parser.add_argument(
        "--spacing-csv",
        type=Path,
        required=True,
        help="CSV containing per-orig_id spacing values used to resample cropped patches to 1 mm isotropic spacing.",
    )
    parser.add_argument("--sample-format", type=str, choices=["npy", "nii.gz"], default="npy", help="Output sample format: npy or nii.gz.")
    parser.add_argument("--output-dir", type=Path, default=Path("./preprocessed_cubes"), help="Base directory where the extracted cubes will be saved.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used when saving cubes.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes.")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_root = args.data_root_real or args.data_root
    fake_root = args.data_root_fake or args.data_root
    if real_root is None or fake_root is None:
        raise ValueError("Specify either --data-root or both --data-root-real and --data-root-fake.")

    set_names = [s.strip() for s in str(args.set_name).split(",") if s.strip()]
    if not set_names:
        raise ValueError("Provide at least one split name via --set-name.")

    for set_name in set_names:
        full_tab = get_table_date(
            ty=args.ty,
            mods=args.mods,
            set_name=set_name,
            data_path=args.data_csv,
            set_path=args.sets_csv,
        )

        dataset = CTPatchExtractor(
            real_root,
            fake_root,
            full_tab,
            spacing_csv=args.spacing_csv,
            lidc_stats_csv=args.lidc_stats_csv,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        save_cubes(
            loader,
            args.output_dir,
            device=device,
            set_name=set_name,
            ty=args.ty,
            sample_format=args.sample_format,
        )


if __name__ == "__main__":
    main()
