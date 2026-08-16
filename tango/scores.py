"""Post-hoc OOD scores and metrics.

TANGO's deployment score is **predictive entropy on raw logits**
(`entropy`). The others are here so the training-scoring coupling in the
paper (Fig. 3, Tab. 1) can be reproduced: on LogitNormKD students the
magnitude-reading scores (Energy at T=1) lose to Entropy, while on vanilla
KD students they win.

Convention: every score returns "higher = more in-distribution", so AUROC is
computed with ID as the positive class.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """-H(softmax(z)). TANGO's recipe score."""
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    return (p * logp).sum(1)            # = -H, higher = ID


def energy(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Liu et al. energy, sign-flipped so higher = ID."""
    return T * torch.logsumexp(logits / T, dim=1)


def msp(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1).max(1).values


def max_logit(logits: torch.Tensor) -> torch.Tensor:
    return logits.max(1).values


SCORES = {
    "entropy": entropy,
    "energy_t1": lambda z: energy(z, 1.0),
    "energy_t01": lambda z: energy(z, 0.1),
    "msp": msp,
    "max_logit": max_logit,
}


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC with ID as the positive class. No sklearn dependency."""
    s = np.concatenate([id_scores, ood_scores]).astype(np.float64)
    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    order = np.argsort(s, kind="mergesort")
    s, y = s[order], y[order]
    # average ranks for ties
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fpr_at_95tpr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """FPR when 95% of ID is retained (higher score = ID)."""
    thr = np.percentile(id_scores, 5.0)      # keeps 95% of ID above thr
    return float((ood_scores >= thr).mean())


@torch.no_grad()
def collect_logits(model, loader, device) -> torch.Tensor:
    model.eval()
    out = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True).float()
        z = model(x)
        if isinstance(z, tuple):
            z = z[0]
        out.append(z.detach().cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def logit_norm_stats(model, loader, device) -> dict:
    """Mean ||z||_2 -- reproduces the paper's 31 -> 1.3 compression number."""
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        x = batch[0].to(device, non_blocking=True).float()
        z = model(x)
        if isinstance(z, tuple):
            z = z[0]
        tot += z.norm(p=2, dim=1).sum().item()
        n += z.shape[0]
    return {"mean_logit_norm": tot / max(n, 1), "n": n}
