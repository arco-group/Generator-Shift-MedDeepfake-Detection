from __future__ import annotations

from typing import List

import torch.nn as nn

def _mlp_block(in_dim: int, out_dim: int, use_bn: bool, dropout: float) -> List[nn.Module]:
    layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if use_bn:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(nn.ReLU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


def build_classifier(feature_dim: int, arch: str, hidden_dim: int, dropout: float) -> nn.Module:
    arch = arch.lower()
    if arch == "linear":
        return nn.Linear(feature_dim, 1)
    if arch == "mlp3":
        layers: List[nn.Module] = []
        layers += _mlp_block(feature_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers += _mlp_block(hidden_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)
    if arch == "mlp5":
        layers = []
        layers += _mlp_block(feature_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers += _mlp_block(hidden_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers += _mlp_block(hidden_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers += _mlp_block(hidden_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)
    raise ValueError("fc_arch must be in [linear, mlp3, mlp5]")
