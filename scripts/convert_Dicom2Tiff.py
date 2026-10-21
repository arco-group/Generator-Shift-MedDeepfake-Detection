import os
import argparse
import sys
from pathlib import Path
import tqdm
from multiprocessing import get_context

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.utils import load_dicom_scan, save_tiff_scan, scan_to_uint16

def funpar(orig_id):
    dicom_folder = os.path.join(path_lidc_idri, orig_id)
    tiff_folder = os.path.join(output_dir_path, "real/scan/", orig_id)
    save_tiff_scan(tiff_folder, scan_to_uint16(load_dicom_scan(dicom_folder)))
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidc-root", required=True, help="Root directory containing the LIDC-IDRI DICOM folders.")
    parser.add_argument("--output-dir", required=True, help="Directory where TIFF scans will be written.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel worker processes.")
    args = parser.parse_args()

    path_lidc_idri = args.lidc_root
    output_dir_path = args.output_dir
    numThreads = args.num_workers

    orig_ids = sorted(
        entry
        for entry in os.listdir(path_lidc_idri)
        if os.path.isdir(os.path.join(path_lidc_idri, entry))
    )

    ctx = get_context("fork")
    with ctx.Pool(numThreads) as pool:
        list(tqdm.tqdm(pool.imap_unordered(funpar, orig_ids), total=len(orig_ids)))
    
