import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import from_numpy as numpy2torch
from torch.utils.data.dataset import Dataset

from .tiff_utils import (
    load_tiff_scan,
)


def get_table_date(ty, mods, set_name="train", data_path="./data/data.csv", set_path="./data/sets.csv"):
    if isinstance(mods, str):
        mods = [m.strip() for m in mods.split(",") if m.strip()]
    if isinstance(ty, str):
        ty = [t.strip() for t in ty.split(",") if t.strip()]
    if isinstance(set_name, str):
        set_name = [s.strip() for s in set_name.split(",") if s.strip()]

    tab = pd.read_csv(data_path)
    tab = tab[tab["mod"].isin(mods)]
    tab = tab[tab["ty"].isin(ty)]

    sets = pd.read_csv(set_path)
    selected_sets = sets[sets["set"].isin(set_name)][["orig_id", "set"]]
    return tab.merge(selected_sets, on="orig_id", how="inner").reset_index(drop=True)


def resample_to_spacing(img_data, original_spacing, target_spacing=(1.0, 1.0, 1.0)):
    resize_factors = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(np.array(img_data.shape) * resize_factors).astype(int)
    img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resampled_tensor = F.interpolate(img_tensor, size=tuple(new_shape), mode="trilinear", align_corners=False)
    resampled = resampled_tensor.squeeze(0).squeeze(0).numpy()
    return resampled.astype(np.float32), resize_factors


def pad_to_shape(arr, target_shape):
    pads = []
    for dim, targ in zip(arr.shape, target_shape):
        if dim < targ:
            total = targ - dim
            before = total // 2
            after = total - before
        else:
            before = after = 0
        pads.append((before, after))
    return np.pad(arr, pads, mode="constant", constant_values=0)


def center_crop(arr, target_shape):
    slices = []
    for dim, targ in zip(arr.shape, target_shape):
        if dim > targ:
            start = (dim - targ) // 2
            end = start + targ
        else:
            start = 0
            end = dim
        slices.append(slice(start, end))
    return arr[tuple(slices)]


def resolve_scan_path(row, real_path_scan, fake_path_scan):
    if row["mod"] == "real":
        real_scan_path = real_path_scan % (row["orig_id"])
        if os.path.exists(real_scan_path):
            return real_scan_path
        fallback_real_path = real_path_scan.replace("/real/scan/%s", "/%s") % (row["orig_id"])
        if os.path.exists(fallback_real_path):
            return fallback_real_path
        return real_scan_path
    return fake_path_scan % (row["mod"], row["img_id"])


def load_lidc_scan_stats_table(stats_csv):
    tab = pd.read_csv(stats_csv)
    required_cols = {"orig_id", "min_export", "rescale_slope", "rescale_intercept"}
    missing = required_cols.difference(tab.columns)
    if missing:
        raise ValueError(f"Missing columns in the LIDC scan stats CSV: {sorted(missing)}")
    return tab[["orig_id", "min_export", "rescale_slope", "rescale_intercept"]].copy()


def restore_sv_and_convert_to_hu(scan, orig_id, lidc_stats_tab):
    row = lidc_stats_tab.loc[lidc_stats_tab["orig_id"] == orig_id]
    if row.empty:
        raise KeyError(f"LIDC statistics not found for orig_id={orig_id}")

    min_export = float(row["min_export"].iloc[0])
    slope = float(row["rescale_slope"].iloc[0])
    intercept = float(row["rescale_intercept"].iloc[0])

    scan_sv = np.float32(scan) - min_export
    scan_hu = scan_sv * slope + intercept
    return scan_hu.astype(np.float32)


def clip_hu_and_normalize_01(scan_hu, min_hu=-1000.0, max_hu=1000.0):
    scan_hu = np.clip(np.float32(scan_hu), min_hu, max_hu)
    scan_norm = (scan_hu - min_hu) / (max_hu - min_hu)
    return scan_norm.astype(np.float32)


class CTPatchExtractor(Dataset):
    def __init__(
        self,
        real_root,
        fake_root,
        tab,
        spacing_csv,
        fallback_label_mods=None,
        lidc_stats_csv=None,
    ):
        self.tab = tab.copy()
        self.real_path_scan = f"{real_root}/real/scan/%s"
        self.fake_path_scan = f"{fake_root}/%s/scan/%s"
        self.fake_path_label = f"{fake_root}/%s/label/%s"

        self.lidc_stats_csv = lidc_stats_csv
        if self.lidc_stats_csv is None:
            raise ValueError("lidc_stats_csv is required")
        self.lidc_stats_tab = None
        self.spacing_csv = spacing_csv
        if self.spacing_csv is None:
            raise ValueError("spacing_csv is required")
        self.spacing_tab = pd.read_csv(self.spacing_csv)

        self.target_shape = np.array([32, 32, 32], dtype=int)
        self.fallback_label_mods = fallback_label_mods or ["diffusion", "pix2pix", "cycle"]
        self.available_fake_mods = (
            self.tab[self.tab["mod"] != "real"]
            .groupby("img_id")["mod"]
            .apply(list)
            .to_dict()
        )

    def __len__(self):
        return len(self.tab)

    def _scan_dir(self, row):
        scan_dir = resolve_scan_path(row, self.real_path_scan, self.fake_path_scan)
        if row["mod"] == "real":
            mods = self.available_fake_mods.get(row["img_id"], [])
            if not mods:
                for candidate_mod in self.fallback_label_mods:
                    candidate_label = self.fake_path_label % (candidate_mod, row["img_id"])
                    if os.path.exists(candidate_label):
                        return scan_dir, candidate_label
                raise FileNotFoundError(f"No fake label found for img_id={row['img_id']}")
            return scan_dir, self.fake_path_label % (mods[0], row["img_id"])
        return scan_dir, self.fake_path_label % (row["mod"], row["img_id"])

    def __getitem__(self, idx):
        _data = self.tab.iloc[idx]
        scan_dirname, label_dirname = self._scan_dir(_data)
        pid = _data["orig_id"]

        ct_scan = load_tiff_scan(scan_dirname, np.uint16)
        if self.lidc_stats_tab is None:
            self.lidc_stats_tab = load_lidc_scan_stats_table(self.lidc_stats_csv)
        scan_hu = restore_sv_and_convert_to_hu(ct_scan, pid, self.lidc_stats_tab)
        norm_ct_scan = clip_hu_and_normalize_01(scan_hu, min_hu=-1000.0, max_hu=1000.0)

        ct_label = load_tiff_scan(label_dirname, np.bool_)
        if ct_label.shape[0] > norm_ct_scan.shape[0]:
            ct_label = ct_label[:-1, ...]
        if ct_label.shape != norm_ct_scan.shape:
            raise ValueError(
                f"Shape mismatch scan {norm_ct_scan.shape} vs label {ct_label.shape} "
                f"for img_id={_data['img_id']} orig_id={_data['orig_id']} mod={_data['mod']}"
            )

        masked_scan = norm_ct_scan * ct_label.astype(np.float32)
        nz = np.nonzero(masked_scan)
        if len(nz[0]) == 0:
            raise ValueError(f"No marked voxel found in the label for img_id={_data['img_id']}")

        z_min, z_max = nz[0].min(), nz[0].max() + 1
        y_min, y_max = nz[1].min(), nz[1].max() + 1
        x_min, x_max = nz[2].min(), nz[2].max() + 1
        cropped = masked_scan[z_min:z_max, y_min:y_max, x_min:x_max]

        patch_min = float(cropped.min())
        patch_max = float(cropped.max())

        spacing_row = self.spacing_tab.loc[self.spacing_tab["orig_id"] == pid]
        original_spacing = (
            float(spacing_row["spacing_z"].values[0]),
            float(spacing_row["spacing_y"].values[0]),
            float(spacing_row["spacing_x"].values[0]),
        )
        scaled_cropped, _ = resample_to_spacing(cropped, original_spacing=original_spacing, target_spacing=(1.0, 1.0, 1.0))
        resized = pad_to_shape(scaled_cropped, self.target_shape)
        resized = center_crop(resized, self.target_shape)

        cube = numpy2torch(resized[None, ...]).float()

        return dict(
            cube=cube,
            orig_id=_data["orig_id"],
            img_id=_data["img_id"],
            mod=_data["mod"],
            ty=_data["ty"],
            set=_data["set"],
            shape=cube.shape,
            patch_min=patch_min,
            patch_max=patch_max,
        )
