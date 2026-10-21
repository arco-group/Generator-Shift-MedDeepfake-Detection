from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.test.evaluate import evaluate
from src.training.train_encoderfusionmodel import full_finetune
from src.utils.config_loader import apply_config_to_parser, load_json_config, merge_config_dicts, parse_bool
from src.utils.utils import (
    load_checkpoint,
    parse_csv_list,
    parse_gpu_ids,
    save_config_json,
    save_predictions_csv,
)
from src.utils.config_exp_utils import build_experiment_paired_cube_loaders, get_experiment
from src.models.encoderfusionmodel import EncoderFusionModel


def _parse_grid_size(raw: str) -> Tuple[int, int, int]:
    cleaned = raw.replace("x", ",")
    parts = [p for p in cleaned.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Invalid grid size: {raw}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def main() -> None:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config_base", type=str, default="")
    bootstrap_parser.add_argument("--config", type=str, default="")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    base_config = (
        load_json_config(bootstrap_args.config_base)
        if str(bootstrap_args.config_base).strip()
        else {}
    )
    script_config = (
        load_json_config(bootstrap_args.config)
        if str(bootstrap_args.config).strip()
        else {}
    )
    merged_config = merge_config_dicts(base_config, script_config)

    parser = argparse.ArgumentParser(
        description="Full fine-tune a dual-SPECTRE fusion model on paired cube patches.",
        parents=[bootstrap_parser],
    )
    parser.add_argument(
        "--experiments_json",
        type=str,
        default=str(PROJECT_ROOT / "split_creation" / "experiments.json"),
        help="Path to the JSON file that defines the available experiments and their split metadata.",
    )
    parser.add_argument(
        "--splits_root",
        type=str,
        default=str(PROJECT_ROOT / "split_creation" / "output"),
        help="Root directory containing one subfolder per experiment, each with train/valid/test_id/test_ood CSV files.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "M3DSynth_patch_RadiomicFeatures" / "RadiomicFeatures"),
        help="Base directory used to resolve the image cube paths.",
    )
    parser.add_argument(
        "--ifft_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "M3DSynth_patch_RadiomicFeatures" / "RadiomicFeatures"),
        help="Base directory used to resolve the IFFT cube paths.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the single experiment to train and test, as defined in the experiments JSON.",
    )
    parser.add_argument(
        "--sample_format",
        type=str,
        default="npy",
        choices=["npy", "nii.gz"],
        help="File format used for cube samples when resolving paired image/IFFT paths.",
    )
    parser.add_argument(
        "--image_sample_format",
        type=str,
        default="",
        help="Optional override for the image branch sample format; defaults to --sample_format.",
    )
    parser.add_argument(
        "--ifft_sample_format",
        type=str,
        default="",
        help="Optional override for the IFFT branch sample format; defaults to --sample_format.",
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        default="binary",
        choices=["binary", "multiclass"],
        help="Label scheme used to build targets: binary maps real vs non-real, multiclass keeps all modality classes.",
    )
    parser.add_argument(
        "--real_mod",
        type=str,
        default="real",
        help="Modality name treated as the real class in binary mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of samples per batch used by the data loaders.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes used by the data loaders.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="Comma-separated GPU ids. The first available id is used as the active CUDA device.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated list of random seeds to repeat the run with.",
    )
    parser.add_argument(
        "--grid_size",
        type=_parse_grid_size,
        default="3,3,4",
        help="Grid used to split the input cube into non-overlapping windows, written as three integers.",
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=512,
        help="Projection dimension used inside the fusion block.",
    )
    parser.add_argument(
        "--class_head",
        type=str,
        default="mlp3",
        choices=["linear", "mlp3", "mlp5"],
        help="Classification head used after fusion.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha value used to disambiguate the run directory name",
    )
    parser.add_argument(
        "--fc_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension used by MLP classifier heads.",
    )
    parser.add_argument(
        "--fc_dropout",
        type=float,
        default=0.1,
        help="Dropout probability used by classifier heads that support dropout.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Shared learning rate for the classification head and the fusion model",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=1e-5,
        help="Learning rate used for the encoder parameters during full fine-tuning.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay coefficient passed to the optimizer.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["none", "plateau"],
        help="Learning-rate scheduler type: none disables scheduling, plateau reduces lr when validation stalls.",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="Multiplicative factor applied by the plateau scheduler.",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=5,
        help="Number of validation epochs to wait before reducing the learning rate.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Lower bound for the learning rate when a scheduler is active.",
    )
    parser.add_argument(
        "--early_stop",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help="Enable early stopping based on validation performance.",
    )
    parser.add_argument(
        "--es_patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement before early stopping triggers.",
    )
    parser.add_argument(
        "--es_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--save_epoch",
        type=int,
        default=100,
        help="Epoch interval used to save periodic checkpoints during training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--resume_training",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help="Resume training from an existing checkpoint instead of starting a new run.",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=-1,
        help="Epoch index to restart from when resuming checkpoints that do not store epoch metadata.",
    )
    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        default="",
        help="Root directory of previous runs used to locate the checkpoint to resume from.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision during training.",
    )
    parser.add_argument(
        "--ifft_norm_mode",
        action="store_true",
        help="Enable normalization for IFFT cubes before they are fed to the model.",
    )
    parser.add_argument(
        "--ifft_norm_stats_csv",
        type=str,
        default="",
        help="CSV used to compute global min/max for IFFT normalization, if enabled.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints"),
        help="Root directory where training outputs, metrics, and checkpoints are written.",
    )

    apply_config_to_parser(parser, merged_config)
    args = parser.parse_args()

    experiment = get_experiment(args.experiments_json, args.experiment_name)

    seed_list = [int(seed) for seed in parse_csv_list(args.seeds)]

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_ids and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")

    image_sample_format = args.image_sample_format.strip() or None
    ifft_sample_format = args.ifft_sample_format.strip() or None
    ifft_norm_stats_csv = args.ifft_norm_stats_csv.strip() or None

    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    experiment_root = root_output_dir / experiment.name
    experiment_root.mkdir(parents=True, exist_ok=True)

    grid_tag = "x".join(str(v) for v in args.grid_size)
    run_name = f"ablation_encoderoptimization"

    for seed in seed_list:
        run_dir = experiment_root / run_name / f"seed_{seed}" / "training"
        resume_training = bool(args.resume_training)
        resume_checkpoint_root = Path(args.resume_checkpoint_path).expanduser() if str(args.resume_checkpoint_path).strip() else None
        resume_checkpoint_path = (
            (resume_checkpoint_root if resume_checkpoint_root is not None else root_output_dir)
            / experiment.name
            / run_name
            / f"seed_{seed}"
            / "checkpoint"
            / "model.pt"
            if resume_training
            else None
        )

        if resume_training:
            if resume_checkpoint_root is None and not resume_checkpoint_path.is_file():
                raise ValueError("--resume_checkpoint_path must be set when --resume_training is true")
            if resume_checkpoint_root is not None and not resume_checkpoint_root.exists():
                raise FileNotFoundError(f"Resume checkpoint root does not exist: {resume_checkpoint_root}")
            if not resume_checkpoint_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint_path}")
            if not run_dir.exists():
                raise FileNotFoundError(
                    f"Run directory does not exist for resume: {run_dir}. "
                    "Create the run first or point to the correct training directory."
                )
        else:
            if run_dir.exists():
                raise FileExistsError(
                    f"Run directory already exists: {run_dir}. "
                    "Refusing to overwrite an existing training run."
                )
            run_dir.mkdir(parents=True, exist_ok=False)


        checkpoint_dir = run_dir / "checkpoint"
        metrics_dir = run_dir / "metrics"
        if not resume_training:
            save_config_json(run_dir, args, [experiment])

        loaders, csv_paths = build_experiment_paired_cube_loaders(
            experiment=experiment,
            splits_root=args.splits_root,
            image_root=args.image_root,
            ifft_root=args.ifft_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_mode=args.label_mode,
            real_mod=args.real_mod,
            sample_format=args.sample_format,
            image_sample_format=image_sample_format,
            ifft_sample_format=ifft_sample_format,
            ifft_norm_mode=bool(args.ifft_norm_mode),
            ifft_norm_stats_csv=ifft_norm_stats_csv,
            pin_memory=(device.type == "cuda"),
        )

        first_batch = next(iter(loaders["train"]))
        image_shape = tuple(int(dim) for dim in first_batch["image_cube"].shape[1:])
        ifft_shape = tuple(int(dim) for dim in first_batch["ifft_cube"].shape[1:])

        model = EncoderFusionModel(
            grid_size=args.grid_size,
            proj_dim=args.proj_dim,
            class_head=args.class_head,
            fc_hidden_dim=args.fc_hidden_dim,
            fc_dropout=args.fc_dropout,
            freeze_encoders=False,
        )

        full_finetune(
            args=args,
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["valid"],
            device=device,
            output_dir=run_dir,
            metrics_dir=metrics_dir,
            checkpoint_dir=checkpoint_dir,
            input_keys=("image_cube", "ifft_cube"),
            model_type="spectre_fusion_wrapper",
            seed=int(seed),
            resume_training=resume_training,
            resume_checkpoint_path=resume_checkpoint_path,
            start_epoch=None if int(args.start_epoch) < 0 else int(args.start_epoch),
            metadata={
                "model_name": "spectre_fusion_wrapper",
                "seed": int(seed),
                "grid_size": tuple(args.grid_size),
                "proj_dim": int(args.proj_dim),
                "class_head": args.class_head,
                "fc_hidden_dim": int(args.fc_hidden_dim),
                "fc_dropout": float(args.fc_dropout),
                "lr": float(args.lr),
                "encoder_lr": float(args.encoder_lr),
                "wd": float(args.wd),
                "image_root": args.image_root,
                "ifft_root": args.ifft_root,
                "image_sample_format": image_sample_format or args.sample_format,
                "ifft_sample_format": ifft_sample_format or args.sample_format,
                "ifft_norm_mode": bool(args.ifft_norm_mode),
                "ifft_norm_stats_csv": ifft_norm_stats_csv,
                "image_shape": image_shape,
                "ifft_shape": ifft_shape,
            },
        )

        selected_checkpoint_name = "best_model.pt" if bool(args.early_stop) else "model.pt"
        selected_checkpoint_path = checkpoint_dir / selected_checkpoint_name
        if not selected_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {selected_checkpoint_path}. "
                f"Expected '{selected_checkpoint_name}' based on early_stop={bool(args.early_stop)}."
            )

        model = EncoderFusionModel(
            grid_size=args.grid_size,
            proj_dim=args.proj_dim,
            class_head=args.class_head,
            fc_hidden_dim=args.fc_hidden_dim,
            fc_dropout=args.fc_dropout,
            freeze_encoders=False,
        )
        best_epoch, best_val_loss = load_checkpoint(model, selected_checkpoint_path, device)

        val_loss, val_acc, val_r_acc, val_f_acc, val_ap, val_rows = evaluate(
            model,
            loaders["valid"],
            device,
            input_keys=("image_cube", "ifft_cube"),
            return_rows=True,
        )
        test_id_loss, test_id_acc, test_id_r_acc, test_id_f_acc, test_id_ap, test_id_rows = evaluate(
            model,
            loaders["test_id"],
            device,
            input_keys=("image_cube", "ifft_cube"),
            return_rows=True,
        )
        test_ood_loss, test_ood_acc, test_ood_r_acc, test_ood_f_acc, test_ood_ap, test_ood_rows = evaluate(
            model,
            loaders["test_ood"],
            device,
            input_keys=("image_cube", "ifft_cube"),
            return_rows=True,
        )

        test_dir = experiment_root / run_name / f"seed_{seed}" / "testing"
        test_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = test_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        with open(metrics_dir / "final_metrics.json", "w") as f:
            json.dump(
                {
                    "checkpoint_path": str(selected_checkpoint_path),
                    "best_epoch": int(best_epoch) if best_epoch is not None else None,
                    "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
                    "valid": {
                        "loss": float(val_loss),
                        "acc": float(val_acc),
                        "r_acc": float(val_r_acc),
                        "f_acc": float(val_f_acc),
                        "ap": float(val_ap),
                    },
                    "test_id": {
                        "loss": float(test_id_loss),
                        "acc": float(test_id_acc),
                        "r_acc": float(test_id_r_acc),
                        "f_acc": float(test_id_f_acc),
                        "ap": float(test_id_ap),
                    },
                    "test_ood": {
                        "loss": float(test_ood_loss),
                        "acc": float(test_ood_acc),
                        "r_acc": float(test_ood_r_acc),
                        "f_acc": float(test_ood_f_acc),
                        "ap": float(test_ood_ap),
                    },
                },
                f,
                indent=2,
            )

        def _pred_rows(rows: List[dict]) -> List[dict]:
            return [
                {
                    "id": row["id"],
                    "split": row["derived_split"],
                    "mod": row["mod"],
                    "ty": row["ty"],
                    "prob": row["prob"],
                    "pred": row["pred"],
                    "image_tag": row["image_tag"],
                }
                for row in rows
            ]

        save_predictions_csv(_pred_rows(val_rows), metrics_dir / "valid_predictions.csv")
        save_predictions_csv(_pred_rows(test_id_rows), metrics_dir / "test_id_predictions.csv")
        save_predictions_csv(_pred_rows(test_ood_rows), metrics_dir / "test_ood_predictions.csv")


if __name__ == "__main__":
    main()
