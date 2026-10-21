from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classifiers import build_classifier


FUSION_ARCHS = {"early", "int", "int_soft_attention", "late"}


class SoftAttentionFusionBlock(nn.Module):
    def __init__(
        self,
        image_dim: int,
        ifft_dim: int,
        proj_dim: int,
    ):
        super().__init__()
        self.proj_dim = int(proj_dim)
        self.image_proj = nn.Linear(image_dim, self.proj_dim)
        self.ifft_proj = nn.Linear(ifft_dim, self.proj_dim)
        self.score = nn.Linear(self.proj_dim, 1)

    def forward(self, img_emb: torch.Tensor, ifft_emb: torch.Tensor) -> torch.Tensor:
        img_emb_proj = self.image_proj(img_emb)
        ifft_emb_proj = self.ifft_proj(ifft_emb)
        inputs = torch.stack([img_emb_proj, ifft_emb_proj], dim=1)
        scores = self.score(inputs).squeeze(-1)
        attention_weights = F.softmax(scores, dim=1)
        fused = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)
        return fused


class IntSoftAttentionFusionModel(nn.Module):
    def __init__(
        self,
        image_dim: int,
        ifft_dim: int,
        proj_dim: Optional[int],
        class_head: str,
        fc_hidden_dim: int,
        fc_dropout: float,
    ):
        super().__init__()
        if proj_dim is None:
            raise ValueError("proj_dim must be provided for IntSoftAttentionFusionModel")
        self.fusion = SoftAttentionFusionBlock(image_dim=image_dim, ifft_dim=ifft_dim, proj_dim=int(proj_dim))
        self.class_head = class_head
        self.head = build_classifier(
            self.fusion.proj_dim,
            arch=class_head,
            hidden_dim=fc_hidden_dim,
            dropout=fc_dropout,
        )

    def forward(self, img_emb: torch.Tensor, ifft_emb: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(img_emb, ifft_emb)
        return self.head(fused)

    def fuse(self, img_emb: torch.Tensor, ifft_emb: torch.Tensor) -> torch.Tensor:
        return self.fusion(img_emb, ifft_emb)


def build_fusion(
    image_dim: int,
    ifft_dim: int,
    proj_dim: Optional[int],
    fusion_arch: str,
    class_head: str,
    fc_hidden_dim: int,
    fc_dropout: float,
) -> nn.Module:
    fusion_arch = str(fusion_arch).strip().lower()
    if fusion_arch not in FUSION_ARCHS:
        raise ValueError(f"fusion_arch must be one of {sorted(FUSION_ARCHS)}")
    if fusion_arch != "int_soft_attention":
        raise NotImplementedError(
            f"Fusion arch {fusion_arch!r} is not implemented yet. "
            "Only 'int_soft_attention' is currently available."
        )
    return IntSoftAttentionFusionModel(
        image_dim=image_dim,
        ifft_dim=ifft_dim,
        proj_dim=proj_dim,
        class_head=class_head,
        fc_hidden_dim=fc_hidden_dim,
        fc_dropout=fc_dropout,
    )
