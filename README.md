# TANGO: Logit-Normalized Distillation Preserves OOD Ability in Foundation Models and Reshapes Which Scores Work

**BMVC 2026** · Yeongje Park, Eui Chul Lee (Sangmyung University)

Foundation models such as DINOv2 and CLIP are strong out-of-distribution (OOD)
detectors out of the box, but distilling them into a small CNN breaks that
ability: vanilla KD, DKD and feature-distillation baselines all lose 7–10 %p of
far-OOD AUROC even while ID accuracy is preserved.

**TANGO** closes that gap with a one-line change to the loss and no inference
overhead: L2-normalize both teacher and student logits inside the KD objective,
then score with predictive entropy.

```python
z_tilde = z / (tau * z.norm(dim=1, keepdim=True))     # tau = 0.04, training only
loss = ce(z_tilde_s, y) + T**2 * kl(softmax(z_tilde_t / T), softmax(z_tilde_s / T))
```

![Method overview](assets/TANGO.png)

Normalization is applied **during training only** — at inference the student is
scored on raw logits, so deployment cost is identical to a standard classifier.

## Why it works, and why the usual scores stop working

Removing the incentive to encode confidence in logit magnitude compresses
‖z‖ by about 30× (31 → 1.3). The OOD signal does not disappear; it moves off the
magnitude axis and onto the *shape* of the softmax. That single mechanism
explains both the gain and an otherwise puzzling side effect: scores that read
magnitude get worse on exactly the students that detect OOD better.

![Method x score](assets/fig4_score_heatmap.png)

Going from vanilla KD to TANGO on the CIFAR-100 far-OOD suite:

| score reads | detector | change |
|---|---|---|
| softmax shape | MSP / Entropy / MaxLogit | **+0.124 / +0.107 / +0.045** |
| layer statistics | GRAM / RMDS | +0.002 / +0.018 |
| magnitude, features | KNN / MDSEns / Energy / MDS / ViM | −0.044 / −0.160 / −0.171 / −0.266 / −0.452 |

Entropy is therefore not an arbitrary choice — it is the simplest score that
reads the channel the training objective moved the signal into.

## Results

**CIFAR-100 → far-OOD** (resnet8x4 student, 1.4 M params, DINOv2-S teacher, 3 seeds).
far₄ = mean of MNIST/SVHN/DTD/Places365, Entropy-scored for every row.

| method | ID acc | SVHN | far₄ |
|---|---|---|---|
| Teacher (frozen DINOv2-S + linear) | 0.830 | 0.978 | 0.806 |
| Vanilla KD | 0.733 | 0.906 | 0.727 |
| DKD | 0.728 | 0.883 | 0.725 |
| Feat-MSE | 0.748 | 0.902 | 0.752 |
| Cosine-Feat | 0.730 | 0.867 | 0.748 |
| LogitNorm-CE (no teacher) | 0.736 | 0.907 | 0.767 |
| ELogitNorm-CE (no teacher) | 0.734 | 0.942 | 0.792 |
| **TANGO** | **0.771** | **0.970** | **0.834** |

![Gap to the teacher](assets/fig1_teaser.png)

**ImageNet-1K** (ResNet18 student, OpenOOD large-scale protocol, single runs).
Far-OOD = iNaturalist/Textures/OpenImage-O.

| method | ID acc | far Entropy | far Energy |
|---|---|---|---|
| Teacher (frozen) | 0.799 | 0.905 | 0.940 |
| Vanilla KD | 0.676 | 0.794 | **0.873** |
| **TANGO** | 0.675 | **0.880** | 0.220 |

At ImageNet scale magnitude-based scoring is genuinely strong — vanilla KD's best
score is Energy — yet the fixed Entropy readout still leads, with no score
selection. TANGO's own Energy inverts (0.220), which is the same
magnitude-versus-shape signature, now deterministic rather than seed-dependent.

TANGO also holds on CIFAR-10 ID (far₄ 0.947 vs 0.918 for the best baseline),
ImageNet-200, ImageNet-100, a wrn\_40\_2 student, and CLIP and DINOv2-B teachers.
Near-OOD is an explicit trade-off, not an oversight: the ‖z‖ compression that
recovers far-OOD discards class-confidence cues, costing 6–7 %p on
CIFAR-100 → CIFAR-10.

## Usage

### Check the paper's numbers (no GPU, no datasets, ~2 s)

```bash
python code/verify_paper_numbers.py
```

Re-derives every cell of Tables 1–4, the teacher rows, the detector-family table
and the coupling deltas from the per-seed records in `results/`, and compares them
against the printed values. Exits non-zero on any disagreement.

```text
195/195 checks passed
All paper numbers reproduce from the released records.
```

### Load a released student

```python
import torch
from mdistiller.models import cifar_model_dict, imagenet_model_dict

net, _ = cifar_model_dict["resnet8x4"]
model = net(num_classes=100)
model.load_state_dict(torch.load("checkpoints/cifar100_resnet8x4/logitnormkd_seed0.pt")["model"])

# ImageNet-1K
model = imagenet_model_dict["ResNet18"](num_classes=1000, pretrained=False)
sd = torch.load("checkpoints/imagenet1k_resnet18/logitnormkd_tau0.04.pt")["model"]
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
```

Score with entropy on the raw logits — no teacher and no normalization at
inference:

```python
p = model(x).softmax(1)
score = (p * p.clamp_min(1e-12).log()).sum(1)      # higher = in-distribution
```

### Train from scratch

`configs/` holds the exact configs behind the released checkpoints
(τ = 0.04, T = 4, 100 epochs SGD, DINOv2-S teacher with a cached linear probe from
`checkpoints/teacher_heads/`).

## Layout

```text
results/          per-seed AUROC/FPR95 for 9 detectors x 8 training objectives
                  x 3 seeds, on CIFAR-100, CIFAR-10, ImageNet-200 and ImageNet-100
checkpoints/      resnet8x4 students (TANGO and vanilla KD, 3 seeds each),
                  two ImageNet-1K ResNet18 students, teacher linear probes
configs/          training configs for the released checkpoints
code/             verification entry point
assets/           figures used in this README
```

## Notes for reproduction

- **Orientation.** Every record stores `auroc` (the detector's published sign,
  never flipped — a value below 0.5 means the ranking has inverted) and
  `auroc_oriented` (the better of the two orientations, the convention of the
  paper's Tables 1–2 and Fig. 3). TANGO + Energy is where this matters: 0.617 and
  0.796 on the CIFAR-100 far aggregate are the same measurement.
- **NINCO.** For ImageNet, load `NINCO/NINCO_OOD_classes` (5,878 images). A
  recursive glob over the NINCO release also picks up
  `popular_datasets_subsamples` and inflates near-OOD by several points.
- **MDSEns** uses equal ensemble weights. OpenOOD's `alpha_selector` fits them by
  logistic regression on ID-vs-OOD validation data, which leaks OOD information
  into the detector.


