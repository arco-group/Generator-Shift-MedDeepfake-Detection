from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusionmodel import build_fusion
from src.test.evaluate import evaluate
from src.utils.config_loader import apply_config_to_parser, load_json_config, merge_config_dicts
from src.utils.utils import (
    load_checkpoint,
    parse_csv_list,
    parse_gpu_ids,
    save_predictions_csv,
)
from src.utils.config_exp_utils import (
    build_experiment_paired_embedding_loaders,
    get_experiment,
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
        description="Test IntSoftAttentionFusionModel on paired embeddings.",
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
        help="Name of the single experiment to test, as defined in the experiments JSON.",
    )
    parser.add_argument("--label_mode", type=str, default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--real_mod", type=str, default="real")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument(
        "--fusion_arch",
        type=str,
        choices="int_soft_attention",
        help="Fusion architecture to test.",
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
        default="",
        help="Comma-separated list of seeds. If empty, uses --seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints"),
        help="Root directory for experiment outputs",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="model.pt",
        help="Checkpoint file to load from the training folder",
    )
    parser.add_argument(
        "--test_folder",
        type=str,
        default="testing",
        help="Name of the test_folder to save the results of the inference ",
    )

    apply_config_to_parser(parser, merged_config)
    args = parser.parse_args()

    experiment = get_experiment(args.experiments_json, args.experiment_name)
    image_root = Path(args.image_root).expanduser() / experiment.name
    ifft_root = Path(args.ifft_root).expanduser() / experiment.name
    fusion_arch = str(args.fusion_arch).strip()
    if str(args.seeds).strip():
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
        args.seed = int(seed)
        print(
            f"\n=== Experiment {experiment.name} | fusion_arch={fusion_arch} | class_head={args.class_head} | seed={seed} ==="
        )

        run_dir = experiment_root / f"{fusion_arch}_{args.class_head}" / f"seed_{seed}" / "training"
        checkpoint_dir = run_dir / "checkpoint"
        checkpoint_path = checkpoint_dir / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        test_dir = experiment_root / f"{fusion_arch}_{args.class_head}" / f"seed_{seed}" / args.test_folder
        test_dir.mkdir(parents=True, exist_ok=True)

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

        best_epoch, best_val_loss = load_checkpoint(model, checkpoint_path, device)

        val_loss, val_acc, val_r_acc, val_f_acc, val_ap, val_rows = evaluate(
            model,
            loaders["valid"],
            device,
            input_keys=("image_embedding", "ifft_embedding"),
            return_rows=True,
        )
        test_id_loss, test_id_acc, test_id_r_acc, test_id_f_acc, test_id_ap, test_id_rows = evaluate(
            model,
            loaders["test_id"],
            device,
            input_keys=("image_embedding", "ifft_embedding"),
            return_rows=True,
        )
        test_ood_loss, test_ood_acc, test_ood_r_acc, test_ood_f_acc, test_ood_ap, test_ood_rows = evaluate(
            model,
            loaders["test_ood"],
            device,
            input_keys=("image_embedding", "ifft_embedding"),
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
