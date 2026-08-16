"""TANGO: LogitNormKD training + Entropy scoring.

Minimal, self-contained release package for
"TANGO: Logit-Normalized Distillation Preserves OOD Ability in Foundation
Models and Reshapes Which Scores Work" (BMVC 2026).

    from tango import resnet8x4, LogitNormKD, SCORES
"""
from .logitnorm_kd import (LogitNormKD, VanillaKD, kd_loss_normed,  # noqa: F401
                           logit_norm)
from .resnet import resnet8x4, resnet32x4  # noqa: F401
from .scores import (SCORES, auroc, collect_logits, energy,  # noqa: F401
                     entropy, fpr_at_95tpr, logit_norm_stats, max_logit, msp)

__all__ = [
    "LogitNormKD", "VanillaKD", "logit_norm", "kd_loss_normed",
    "resnet8x4", "resnet32x4",
    "SCORES", "entropy", "energy", "msp", "max_logit",
    "auroc", "fpr_at_95tpr", "collect_logits", "logit_norm_stats",
]
__version__ = "1.0.0"
