from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusionmodel import build_fusion
from src.test.evaluate import evaluate
from src.training.train import train
from src.utils.config_loader import apply_config_to_parser, load_json_config, merge_config_dicts, parse_bool
from src.utils.utils import (
    parse_csv_list,
    parse_gpu_ids,
    save_config_json,
    save_loaded_samples_csv,
)
from src.utils.config_exp_utils import (
    build_experiment_paired_embedding_loaders,
    get_experiment,
    load_experiments,
)
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
        description="Train IntSoftAttentionFusionModel on paired embeddings.",
        parents=[bootstrap_parser],
    )
    parser.add_argument(
        "--experiments_json",
        type=str,
        default=str(PROJECT_ROOT / "split_creation" / "experiments.json"),
        help="Path to the experiment definition JSON",
    )
    parser.add_argument(
        "--splits_root",
        type=str,
        default=str(PROJECT_ROOT / "split_creation" / "output"),
        help="Root directory containing one subfolder per experiment with train/valid/test_id/test_ood.csv",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="", 
        help="Root directory for the image embeddings used by the paired loader",
    )
    parser.add_argument(
        "--ifft_root",
        type=str,
        default="", 
        help="Root directory for the IFFT embeddings used by the paired loader",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the single experiment to train, as defined in the experiments JSON.",
    )
    parser.add_argument("--label_mode", type=str, default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--real_mod", type=str, default="real")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument(
        "--fusion_arch",
        type=str,
        default="int_soft_attention",
        choices=["early", "int", "int_soft_attention", "late"],
        help="Fusion architecture to train.",
    )
    parser.add_argument(
        "--class_head",
        type=str,
        default="mlp3",
        choices=["linear", "mlp3", "mlp5"],
        help="Classification head used inside the fusion model",
    )
    parser.add_argument("--fc_hidden_dim", type=int, default=512)
    parser.add_argument("--fc_dropout", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated list of seeds to run.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "plateau"])
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--early_stop", type=parse_bool, default=False, metavar="{true,false}")
    parser.add_argument("--save_epoch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--resume_training", type=parse_bool, default=False, metavar="{true,false}")
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=-1,
        help="Optional epoch to start from when resuming legacy checkpoints without epoch metadata",
    )
    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        default="",
        help="Root directory of previous runs used to construct the resume checkpoint path",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints"),
        help="Root directory for experiment outputs",
    )

    apply_config_to_parser(parser, merged_config)
    args = parser.parse_args()

    experiment = get_experiment(args.experiments_json, args.experiment_name)
    image_root = Path(args.image_root).expanduser() / experiment.name
    ifft_root = Path(args.ifft_root).expanduser() / experiment.name
    
    fusion_arch = str(args.fusion_arch).strip()
    seed_list = [int(seed) for seed in parse_csv_list(args.seeds)]

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_ids and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")

    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    global_summary_rows: List[dict] = []

    experiment_root = root_output_dir / experiment.name
    experiment_root.mkdir(parents=True, exist_ok=True)

    for seed in seed_list:
        print(
            f"\n=== Experiment {experiment.name} | fusion_arch={fusion_arch} | class_head={args.class_head} | seed={seed} ==="
        )
        run_dir = experiment_root / f"{fusion_arch}_{args.class_head}" / f"seed_{seed}" / "training"
        resume_training = bool(args.resume_training)
        resume_checkpoint_root = Path(args.resume_checkpoint_path).expanduser() if str(args.resume_checkpoint_path).strip() else None
        resume_checkpoint_path = (
            (resume_checkpoint_root if resume_checkpoint_root is not None else root_output_dir)
            / experiment.name
            / f"{fusion_arch}_{args.class_head}"
            / f"seed_{seed}"
            / "training"
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
                raise FileNotFoundError(f"Run directory does not exist for resume: {run_dir}")
        else:
            if run_dir.exists():
                raise FileExistsError(
                    f"Run directory already exists: {run_dir}. "
                    "Refusing to overwrite an existing training run."
                )
            run_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = run_dir / "metrics"
        checkpoint_dir = run_dir / "checkpoint"
        if not resume_training:
            save_config_json(run_dir, args, [experiment])

        loaders, csv_paths = build_experiment_paired_embedding_loaders(
            experiment=experiment,
            splits_root=args.splits_root,
            image_root=image_root,
            ifft_root=ifft_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_mode=args.label_mode,
            real_mod=args.real_mod,
            pin_memory=(device.type == "cuda"),
        )

        save_loaded_samples_csv(
            run_dir / "loaded_samples.csv",
            [
                ("train", loaders["train"]),
                ("valid", loaders["valid"]),
                ("test_id", loaders["test_id"]),
                ("test_ood", loaders["test_ood"]),
            ],
        )

        first_batch = next(iter(loaders["train"]))
        image_dim = int(first_batch["image_embedding"].shape[1])
        ifft_dim = int(first_batch["ifft_embedding"].shape[1])

        model = build_fusion(
            image_dim=image_dim,
            ifft_dim=ifft_dim,
            proj_dim=args.proj_dim,
            fusion_arch=fusion_arch,
            class_head=args.class_head,
            fc_hidden_dim=args.fc_hidden_dim,
            fc_dropout=args.fc_dropout,
        )

        train(
            args=args,
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["valid"],
            device=device,
            output_dir=run_dir,
            metrics_dir=metrics_dir,
            checkpoint_dir=checkpoint_dir,
            input_keys=("image_embedding", "ifft_embedding"),
            model_type="fusion",
            resume_training=resume_training,
            resume_checkpoint_path=resume_checkpoint_path,
            start_epoch=None if int(args.start_epoch) < 0 else int(args.start_epoch),
            seed=int(seed),
            metadata={
                "fusion_arch": fusion_arch,
                "class_head": args.class_head,
                "seed": int(seed),
                "image_dim": image_dim,
                "ifft_dim": ifft_dim,
                "proj_dim": args.proj_dim,
                "fc_hidden_dim": args.fc_hidden_dim,
                "fc_dropout": args.fc_dropout,
                "image_root": str(image_root),
                "ifft_root": str(ifft_root),
            },
        )


if __name__ == "__main__":
    main()
