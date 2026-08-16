"""Frozen DINOv2 backbone + the released linear probe head.

The head checkpoints under `checkpoints/teacher_heads/` are the ones used for
every CIFAR result in the paper. Only the head was trained; the backbone is
`facebook/dinov2-small` exactly as published.

CIFAR images arrive at 32x32 in CIFAR normalization and are bilinearly
upsampled to 224x224 before the ViT -- this matches Sec. 3.1 of the paper.

Requires `transformers` (only for training; eval_ood.py does not need it).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HF_NAME = {"small": "facebook/dinov2-small", "base": "facebook/dinov2-base"}
FEAT_DIM = {"small": 384, "base": 768}


class DINOv2Teacher(nn.Module):
    def __init__(self, num_classes, variant="small", head_ckpt="",
                 input_size=224):
        super().__init__()
        try:
            from transformers import Dinov2Model
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The teacher needs HuggingFace transformers:\n"
                "    pip install transformers"
            ) from e

        self.variant = variant
        self.input_size = int(input_size)
        self.backbone = Dinov2Model.from_pretrained(HF_NAME[variant])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.head = nn.Linear(FEAT_DIM[variant], num_classes)
        if head_ckpt:
            st = torch.load(head_ckpt, map_location="cpu", weights_only=True)
            self.head.weight.data.copy_(st["head_weight"])
            self.head.bias.data.copy_(st["head_bias"])
        for p in self.head.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()          # frozen regardless of mode
        return self

    @torch.no_grad()
    def get_feat(self, x):
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear",
                              align_corners=False)
        return self.backbone(pixel_values=x).last_hidden_state[:, 0]

    def forward(self, x):
        return self.head(self.get_feat(x))


def dinov2_small(num_classes, **kw):
    return DINOv2Teacher(num_classes, variant="small", **kw)


def dinov2_base(num_classes, **kw):
    return DINOv2Teacher(num_classes, variant="base", **kw)
