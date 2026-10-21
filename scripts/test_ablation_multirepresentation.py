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

from src.models.classifiers import build_classifier
from src.test.common import save_test_config_json
from src.test.evaluate import evaluate
from src.utils.config_loader import apply_config_to_parser, load_json_config, merge_config_dicts
from src.utils.utils import (
    load_checkpoint,
    parse_csv_list,
    parse_gpu_ids,
    save_predictions_csv,
)
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
        description="Test embedding-based classifiers on experiment-specific splits.",
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
        "--sample_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "M3DSynth_embedding" / "Embeddings"),
        help="Base directory used by the embedding dataloader to resolve .pt files for the selected representation.",
    )
    parser.add_argument(
        "--emb_kind",
        type=str,
        default=None,
        help="Name of the embedding source used in this run, such as image or ifft.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the single experiment to test, as defined in the experiments JSON.",
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
        default=16,
        help="Number of samples per batch used by the evaluation loaders.",
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
        "--fc_arch",
        type=str,
        default="linear",
        help="Classifier head architecture to evaluate, for example linear or an MLP variant supported by build_classifier.",
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
        help="Comma-separated list of random seeds to repeat the evaluation with.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints"),
        help="Root directory containing training outputs and checkpoints to evaluate.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="model.pt",
        choices=["best_model.pt", "model.pt", "model_99.pt"],
        help="Checkpoint file to load from the corresponding training folder.",
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
        checkpoint_dir = run_dir / "checkpoint"
        checkpoint_path = checkpoint_dir / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        test_dir = experiment_root / f"ablation_multirepresentation_{args.emb_kind}" / f"seed_{seed}" / "testing"
        test_dir.mkdir(parents=True, exist_ok=True)
        save_test_config_json(test_dir, args, [experiment], Path(__file__).name)

        loaders, csv_paths = build_experiment_loaders(
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

        first_batch = next(iter(loaders["train"]))
        feature_dim = int(first_batch["embedding"].shape[1])
        model = build_classifier(
            feature_dim=feature_dim,
            arch=fc_arch,
            hidden_dim=args.fc_hidden_dim,
            dropout=args.fc_dropout,
        )

        best_epoch, best_val_loss = load_checkpoint(model, checkpoint_path, device)

        val_loss, val_acc, val_r_acc, val_f_acc, val_ap, val_rows = evaluate(
            model,
            loaders["valid"],
            device,
            input_key="embedding",
            return_rows=True,
        )
        test_id_loss, test_id_acc, test_id_r_acc, test_id_f_acc, test_id_ap, test_id_rows = evaluate(
            model,
            loaders["test_id"],
            device,
            input_key="embedding",
            return_rows=True,
        )
        test_ood_loss, test_ood_acc, test_ood_r_acc, test_ood_f_acc, test_ood_ap, test_ood_rows = evaluate(
            model,
            loaders["test_ood"],
            device,
            input_key="embedding",
            return_rows=True,
        )

        metrics_dir = test_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        with open(metrics_dir / "final_metrics.json", "w") as f:
            json.dump(
                {
                    "checkpoint_path": str(checkpoint_path),
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

        save_predictions_csv(_pred_rows(test_id_rows), metrics_dir / "test_id_predictions.csv")
        save_predictions_csv(_pred_rows(val_rows), metrics_dir / "valid_predictions.csv")
        save_predictions_csv(_pred_rows(test_ood_rows), metrics_dir / "test_ood_predictions.csv")


if __name__ == "__main__":
    main()
