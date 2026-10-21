from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classifiers import build_classifier
from src.training.train import train
from src.utils.config_loader import apply_config_to_parser, load_json_config, merge_config_dicts, parse_bool
from src.utils.utils import parse_csv_list, parse_gpu_ids, save_config_json, save_loaded_samples_csv
from src.utils.config_exp_utils import build_experiment_loaders, get_experiment


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
        description="Train embedding-based classifiers on experiment-specific splits.",
        parents=[bootstrap_parser],
    )
    parser.add_argument(
        "--experiments_json",
        type=str,
        default=str("./data/splits/experiments.json"),
        help="Path to the JSON file that defines the available experiments.",
    )
    parser.add_argument(
        "--splits_root",
        type=str,
        default=str("./data/splits"),
        help="Root directory containing one subfolder per experiment, each with train/valid/test_id/test_ood CSV files.",
    )
    parser.add_argument(
        "--sample_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "M3DSynth_embedding" / "Embeddings"),
        help="Base directory used by the embedding dataloader to resolve .pt files for the selected representation.",
    )
    parser.add_argument(
        "--emb_kind",
        type=str,
        default="",
        help="Label appended to the output directory to distinguish the embedding source used in this run, such as image or ifft.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the single experiment to train, as defined in the experiments JSON.",
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        default="binary",
        choices=["binary", "multiclass"],
        help="Target label scheme: binary collapses all non-real classes into one class, multiclass keeps all modality classes.",
    )
    parser.add_argument(
        "--real_mod",
        type=str,
        default="real",
        help="Modality name treated as the positive class in binary mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples per batch for the data loaders.",
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
        help="Comma-separated GPU ids. The first available id is used as the active device.",
    )
    parser.add_argument(
        "--fc_arch",
        type=str,
        default="linear",
        help="Classifier head architecture to train, for example linear or mlp variants supported by build_classifier.",
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
        default=0.0,
        help="Dropout probability used by classifier heads that support dropout.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated list of random seeds to repeat the run with.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate used by the optimizer.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Lower bound for the learning rate when a scheduler is active.",
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
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay coefficient passed to the optimizer.",
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
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints"),
        help="Root directory where training outputs, metrics, and checkpoints are written.",
    )

    apply_config_to_parser(parser, merged_config)
    args = parser.parse_args()

    experiment = get_experiment(args.experiments_json, args.experiment_name)
    sample_root = Path(args.sample_root).expanduser() / experiment.name
    fc_arch = str(args.fc_arch).strip()
    seed_list = [int(seed) for seed in parse_csv_list(args.seeds)]

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_ids and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")

    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    experiment_root = root_output_dir / experiment.name
    experiment_root.mkdir(parents=True, exist_ok=True)

    for seed in seed_list:
        print(
            f"\n=== Experiment {experiment.name} | fc_arch={fc_arch} | emb_kind={args.emb_kind} | seed={seed} ==="
        )
        run_dir = experiment_root / f"ablation_multirepresentation_{args.emb_kind}" / f"seed_{seed}" / "training"
        resume_training = bool(args.resume_training)
        resume_checkpoint_root = Path(args.resume_checkpoint_path).expanduser() if str(args.resume_checkpoint_path).strip() else None
        resume_checkpoint_path = (
            (resume_checkpoint_root if resume_checkpoint_root is not None else root_output_dir)
            / experiment.name
            / f"ablation_multirepresentation_{args.emb_kind}"
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

        loaders, _ = build_experiment_loaders(
            experiment=experiment,
            splits_root=args.splits_root,
            sample_root=sample_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_kind="embedding",
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
        feature_dim = int(first_batch["embedding"].shape[1])
        model = build_classifier(
            feature_dim=feature_dim,
            arch=fc_arch,
            hidden_dim=args.fc_hidden_dim,
            dropout=args.fc_dropout,
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
            input_key="embedding",
            model_type="classifier",
            resume_training=resume_training,
            resume_checkpoint_path=resume_checkpoint_path,
            start_epoch=None if int(args.start_epoch) < 0 else int(args.start_epoch),
            seed=int(seed),
            metadata={
                "fc_arch": fc_arch,
                "emb_kind": args.emb_kind,
                "seed": int(seed),
                "feature_dim": feature_dim,
            },
        )


if __name__ == "__main__":
    main()
