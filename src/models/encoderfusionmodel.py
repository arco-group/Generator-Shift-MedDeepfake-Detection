from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn as nn

from src_spectre.spectre import MODEL_CONFIGS, SpectreImageFeatureExtractor
from src.models.classifiers import build_classifier
from src.models.fusionmodel import SoftAttentionFusionBlock


def split_volume_into_windows(volumes: torch.Tensor, grid_size: Sequence[int]) -> torch.Tensor:
    """Split a batch of 3D volumes into a grid of non-overlapping windows."""
    if volumes.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got {tuple(volumes.shape)}")

    g_h, g_w, g_d = grid_size
    volumes_hwd = volumes.permute(0, 1, 3, 4, 2).contiguous()
    _, _, H, W, D = volumes_hwd.shape

    if (H % g_h) or (W % g_w) or (D % g_d):
        raise ValueError(
            f"Shape {H}x{W}x{D} is not divisible by grid_size={tuple(grid_size)}. "
            "Make sure the input volumes are already resized/resampled accordingly."
        )

    h_size, w_size, d_size = H // g_h, W // g_w, D // g_d
    windows = volumes_hwd.view(-1, 1, g_h, h_size, g_w, w_size, g_d, d_size)
    windows = windows.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    windows = windows.view(-1, g_h * g_w * g_d, 1, h_size, w_size, d_size)
    return windows


class EncoderFusionModel(nn.Module):
    """End-to-end wrapper for two SPECTRE encoders, soft-attention fusion and classification.

    The wrapper accepts two raw 3D cube inputs, one for the image branch and one for the IFFT
    branch. Each branch is encoded by its own SPECTRE encoder initialized from the same
    pretrained configuration. The resulting embeddings are fused with soft attention and fed
    into a classification head.
    """

    def __init__(
        self,
        *,
        grid_size: Sequence[int] = (3, 3, 4),
        encoder_config_name: str = "spectre-large-pretrained",
        encoder_config: dict | None = None,
        proj_dim: int | None = None,
        class_head: str = "linear",
        fc_hidden_dim: int = 512,
        fc_dropout: float = 0.0,
        freeze_encoders: bool = False,
    ) -> None:
        super().__init__()
        self.grid_size = tuple(int(v) for v in grid_size)

        # IMAGE-ENCODER IFFT-ENCODER with shared weights 
        config = MODEL_CONFIGS["spectre-large-pretrained"]
        self.image_encoder = SpectreImageFeatureExtractor.from_config(config)
        self.ifft_encoder = SpectreImageFeatureExtractor.from_config(config)

        encoder_dim = int(self.image_encoder.backbone.embed_dim)
        self.proj_dim = int(proj_dim) if proj_dim is not None else encoder_dim
        
        # FUSION BLOCK 
        self.fusion = SoftAttentionFusionBlock(
            image_dim=encoder_dim,
            ifft_dim=encoder_dim,
            proj_dim=self.proj_dim,
        )

        # CLASSIFICATION HEAD 
        self.head = build_classifier(
            feature_dim=self.proj_dim,
            arch=class_head,
            hidden_dim=fc_hidden_dim,
            dropout=fc_dropout,
        )

        if freeze_encoders:
            self.freeze_encoders()

    def freeze_encoders(self) -> None:
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.ifft_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self) -> None:
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        for param in self.ifft_encoder.parameters():
            param.requires_grad = True

    def set_encoders_trainable(self, trainable: bool) -> None:
        if trainable:
            self.unfreeze_encoders()
        else:
            self.freeze_encoders()

    def image_encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.image_encoder.parameters()

    def ifft_encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.ifft_encoder.parameters()

    def fusion_parameters(self) -> Iterator[nn.Parameter]:
        return self.fusion.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        return self.head.parameters()

    def encode_image(self, cubes: torch.Tensor) -> torch.Tensor:
        windows = split_volume_into_windows(cubes, self.grid_size)
        _, features = self.image_encoder(windows, grid_size=self.grid_size)
        if features.ndim == 3:
            features = features[:, 0, :]
        return features

    def encode_ifft(self, cubes: torch.Tensor) -> torch.Tensor:
        windows = split_volume_into_windows(cubes, self.grid_size)
        _, features = self.ifft_encoder(windows, grid_size=self.grid_size)
        if features.ndim == 3:
            features = features[:, 0, :]
        return features

    def fuse_embeddings(self, img_emb: torch.Tensor, ifft_emb: torch.Tensor) -> torch.Tensor:
        return self.fusion(img_emb, ifft_emb)

    def forward(
        self,
        image_cubes: torch.Tensor,
        ifft_cubes: torch.Tensor,
        return_embedding: bool = False,
    ):
        image_embedding = self.encode_image(image_cubes)
        ifft_embedding = self.encode_ifft(ifft_cubes)
        fused_embedding = self.fuse_embeddings(image_embedding, ifft_embedding)
        logits = self.head(fused_embedding)
        if return_embedding:
            return logits, fused_embedding, image_embedding, ifft_embedding
        return logits
