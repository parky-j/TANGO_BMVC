"""TANGO's training objective: LogitNormKD.

L2 logit normalization (Wei et al., ICML 2022) applied to *both* the CE and
the KD term, on *both* the student's and the teacher's logits:

    z~ = z / (tau * ||z||_2)

    L = ce_w * CE(z~_s, y)  +  kd_w * T^2 * KL( softmax(z~_t / T) || softmax(z~_s / T) )

Because the loss depends on z_s only through its direction z_s/||z_s||, the
gradient w.r.t. z_s is orthogonal to the radial direction, so nothing pushes
||z_s|| up. Together with weight decay this yields the ~30x logit-norm
compression that the paper's mechanism section analyses.

Normalization is TRAINING-ONLY. At inference the student is an ordinary
classifier scored on its raw logits with predictive entropy (see scores.py).

Paper defaults (configs/cifar100_logitnormkd_seed*.yaml):
    TEMPERATURE = 4.0, TAU = 0.04, CE_WEIGHT = 1.0, KD_WEIGHT = 1.0
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def logit_norm(z: torch.Tensor, tau: float) -> torch.Tensor:
    """z / (||z||_2 * tau), row-wise."""
    norm = z.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    return z / (norm * tau)


def kd_loss_normed(zs: torch.Tensor, zt: torch.Tensor, T: float) -> torch.Tensor:
    """T^2-scaled KL(teacher || student) on already-normalized logits."""
    log_ps = F.log_softmax(zs / T, dim=1)
    pt = F.softmax(zt / T, dim=1)
    return F.kl_div(log_ps, pt, reduction="batchmean") * (T ** 2)


class LogitNormKD(nn.Module):
    """Wraps a student and a frozen teacher; returns (logits, loss dict)."""

    def __init__(self, student, teacher, temperature=4.0, tau=0.04,
                 ce_weight=1.0, kd_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.T = float(temperature)
        self.tau = float(tau)
        self.ce_w = float(ce_weight)
        self.kd_w = float(kd_weight)
        self.label_smoothing = float(label_smoothing)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()          # teacher is frozen, always eval
        return self

    def parameters_to_optimize(self):
        return [p for p in self.student.parameters() if p.requires_grad]

    def forward_train(self, image, target):
        logits_s, _ = self.student(image)
        with torch.no_grad():
            logits_t = self.teacher(image)
            if isinstance(logits_t, tuple):
                logits_t = logits_t[0]

        zs = logit_norm(logits_s, self.tau)
        zt = logit_norm(logits_t, self.tau)

        loss_ce = self.ce_w * F.cross_entropy(
            zs, target, label_smoothing=self.label_smoothing)
        loss_kd = self.kd_w * kd_loss_normed(zs, zt, self.T)
        return logits_s, {"loss_ce": loss_ce, "loss_kd": loss_kd}

    def forward(self, image, target=None):
        if self.training:
            return self.forward_train(image, target)
        return self.student(image)[0]


class VanillaKD(nn.Module):
    """Hinton KD on raw logits -- the paper's main baseline.

    Paper defaults: T = 4, ce_w (alpha) = 0.1, kd_w (beta) = 9.
    """

    def __init__(self, student, teacher, temperature=4.0,
                 ce_weight=0.1, kd_weight=9.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.T = float(temperature)
        self.ce_w = float(ce_weight)
        self.kd_w = float(kd_weight)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def parameters_to_optimize(self):
        return [p for p in self.student.parameters() if p.requires_grad]

    def forward_train(self, image, target):
        logits_s, _ = self.student(image)
        with torch.no_grad():
            logits_t = self.teacher(image)
            if isinstance(logits_t, tuple):
                logits_t = logits_t[0]
        loss_ce = self.ce_w * F.cross_entropy(logits_s, target)
        loss_kd = self.kd_w * kd_loss_normed(logits_s, logits_t, self.T)
        return logits_s, {"loss_ce": loss_ce, "loss_kd": loss_kd}

    def forward(self, image, target=None):
        if self.training:
            return self.forward_train(image, target)
        return self.student(image)[0]
