from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.utils.config_exp_utils import ExperimentConfig


def save_test_config_json(output_dir: Path, args: argparse.Namespace, experiments: Sequence[ExperimentConfig], script_name: str) -> None:
    config = {
        "script": script_name,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "experiments": [
            {
                "name": exp.name,
                "family": exp.family,
                "id_mods": exp.id_mods,
                "ood_mods": exp.ood_mods,
            }
            for exp in experiments
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
