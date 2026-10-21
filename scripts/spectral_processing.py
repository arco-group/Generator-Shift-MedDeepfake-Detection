import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd 
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader_preprocessing import get_dataloader


def make_radial_mask_3d(shape_zyx: Tuple[int, int, int], cutoff: float, kind: str) -> np.ndarray:
    Z, Y, X = shape_zyx
    zz = np.arange(Z) - (Z // 2)
    yy = np.arange(Y) - (Y // 2)
    xx = np.arange(X) - (X // 2)
    Zg, Yg, Xg = np.meshgrid(zz, yy, xx, indexing="ij")

    r = np.sqrt(Zg**2 + Yg**2 + Xg**2) 

    r_max = np.sqrt((Z//2)**2 + (Y//2)**2 + (X//2)**2)
    r0 = cutoff * r_max

    if kind == "lowpass":
        mask = (r <= r0).astype(np.float32)
    elif kind == "highpass":
        mask = (r >= r0).astype(np.float32)
    else:
        raise ValueError("kind must be 'lowpass' or 'highpass'")
    return mask


def minmax_normalize_volume(volume: np.ndarray) -> np.ndarray:
    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32)


def save_ifft_patches(
    loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    set_name: str,
    ty: str,
) -> None:
    samples_dirname = "samples" if args.sample_format == "npy" else "samples_nii"
    sample_suffix = ".npy" if args.sample_format == "npy" else ".nii.gz"

    for batch in tqdm(loader, desc="Extracting", unit="vol"):
        cubes = batch["cube"].to(device, non_blocking=True)
        names = [f"{img}" for img in batch["img_id"]]
        orig_ids = batch["orig_id"]
        mods = batch["mod"]

        cubes_np = cubes.detach().cpu().numpy()

        for cube_np, name, orig_id, mod in zip(cubes_np, names, orig_ids, mods):
            save_dir = output_dir / set_name / ty / mod
            samples_dir = save_dir / samples_dirname
            metadata_dir = save_dir / "metadata"
            samples_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)

            vol = cube_np[0].astype(np.float32)
            spectrum = np.fft.fftn(vol)
            spectrum_shifted = np.fft.fftshift(spectrum)
            mask = make_radial_mask_3d(vol.shape, cutoff=args.cutoff, kind="highpass")
            spectrum_filt_shifted = spectrum_shifted * mask
            spectrum_filt = np.fft.ifftshift(spectrum_filt_shifted)
            hf_data = np.real(np.fft.ifftn(spectrum_filt)).astype(np.float32)
            sample_path = samples_dir / f"{name}{sample_suffix}"
            metadata_path = metadata_dir / f"{name}.csv"
            if sample_path.exists() and metadata_path.exists():
                tqdm.write(f"[SKIP] {name} already exists")
                continue

            if not sample_path.exists():
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                if args.sample_format == "npy":
                    np.save(sample_path, hf_data.astype(np.float32))
                else:
                    nib.save(nib.Nifti1Image(hf_data.astype(np.float32), np.eye(4)), sample_path)

            if not metadata_path.exists():
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_df = pd.DataFrame([
                    {
                        "img_id": name,
                        "orig_id": orig_id,
                        "mod": mod,
                        "ty": ty,
                        "shape": str(hf_data.shape),
                        "representation": "ifft",
                        "freq": "hf",
                        "cutoff": float(args.cutoff),
                        "min": float(np.min(hf_data)),
                        "max": float(np.max(hf_data)),
                        "path": str(sample_path),
                    }
                ])
                metadata_df.to_csv(metadata_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute high-frequency IFFT on 3D patches."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory containing the precomputed cubes.")
    parser.add_argument("--mods", default="real", help="Comma-separated list of mods to process.")
    parser.add_argument("--ty", default="train", help="Dataset split (ty column) to filter.")
    parser.add_argument(
        "--set-name",
        default="train",
        help="Comma-separated list of splits to use, for example 'train' or 'train,valid,test'.",
    )
    parser.add_argument("--sample-format", choices=["npy", "nii.gz"], default="npy", help="Format of the samples to load.")
    parser.add_argument("--batch-size", type=int, default=4, help="DataLoader batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--output-dir",type=Path, default="./fft_outputs", help="Directory to save .npy results.")
    parser.add_argument(
    "--cutoff",
    type=float,
    default=0.15,
    help="Radial cutoff (fraction of the maximum radius in the shifted spectrum), typically 0.10-0.25.",
)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mods = [m.strip() for m in args.mods.split(",") if m.strip()]
    splits = [s.strip() for s in args.set_name.split(",") if s.strip()]

    if not splits:
        raise ValueError("Specify at least one split with --set-name.")

    for split in splits:
        loader = get_dataloader(
            root=args.data_root,
            split=split,
            ty=args.ty,
            mods=mods,
            sample_format=args.sample_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )

        save_ifft_patches(
            loader,
            args,
            output_dir,
            device=device,
            set_name=split,
            ty=args.ty,
        )


if __name__ == "__main__":
    main()
