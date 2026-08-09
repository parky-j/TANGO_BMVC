# TANGO — code, configs and checkpoints

Release accompanying *TANGO: Logit-Normalized Distillation Preserves OOD Ability in
Foundation Models and Reshapes Which Scores Work* (BMVC 2026).

**TANGO = LogitNormKD training + Entropy scoring.** `LogitNormKD` L2-normalizes both
teacher and student logits inside the KD loss, `z̃ = z / (τ‖z‖₂)` with τ = 0.04.
Normalization is applied **during training only** — inference scores raw logits.

---

## Verify the paper's numbers (no GPU, no datasets, ~2 s)

```bash
python code/verify_paper_numbers.py
```

Re-derives every cell of Tables 1–4, the teacher rows, the supplementary
detector-family table and the training–detector coupling deltas from the
per-seed evaluation records in `results/`, and compares them against the values
printed in the paper. Exit code is non-zero if anything disagrees.

Current status: **195 / 195 checks pass**, every one to within 0.002 of the
printed value.

## Conventions (read before comparing anything)

Every record stores **two AUROCs** per (split, detector):

| field | meaning |
|---|---|
| `auroc` | **fixed orientation** — the detector's published sign, never flipped. A value below 0.5 means the score has inverted its ID/OOD ordering. |
| `auroc_oriented` | the empirically better of the two orientations, with `flipped` marking where they differ. This is the convention of the main paper's Tables 1–2 and Fig. 3. |

The distinction matters for `LogitNormKD`: its Energy score inverts, so
`auroc` = 0.617 and `auroc_oriented` = 0.796 on the CIFAR-100 far aggregate
describe the same measurement. Mixing the two conventions across a row or a
figure is the single easiest way to draw a wrong conclusion from this data.

`far4` throughout is the mean over MNIST / SVHN / DTD / Places365, Entropy-scored
for every method so the aggregate compares training objectives rather than scores.

## Layout

```
results/
  detector_suite/    per-seed AUROC/FPR95 for MSP, Entropy, Energy, MaxLogit,
                     KNN, MDS, ViM  x  {CIFAR-100, CIFAR-10, ImageNet-200,
                     ImageNet-100} x 8 training objectives x 3 seeds
  gram_mdsens/       the same for GRAM and MDSEns (OpenOOD implementations)
  posthoc_rmds_*.csv relative Mahalanobis (RMDS / MDS++)
  teacher_ood_all.json  frozen DINOv2-S + linear probe, both ID setups, 7 splits
checkpoints/
  cifar100_resnet8x4/  logitnormkd_seed{0,1,2}.pt, vanillakd_seed{0,1,2}.pt
  imagenet1k_resnet18/ logitnormkd_tau0.04.pt, vanillakd_seed4.pt
  teacher_heads/       linear probes on cached DINOv2-S features
configs/               training configs for the released checkpoints
code/                  verification and evaluation entry points
```

Checkpoints are the headline student (resnet8x4, 1.4 M params) at all three seeds
plus its matched Vanilla-KD control, and the two ImageNet-1K ResNet18 students.
Other architectures, teachers and ablations from the paper are reproducible from
the configs but are not shipped here to keep the release small.

## Reproducing an evaluation from a checkpoint

Teacher heads are linear layers over frozen `facebook/dinov2-small` CLS features;
no teacher is needed at inference for a distilled student. For the ImageNet-1K
students the OOD protocol is OpenOOD large-scale: near-OOD = SSB-Hard (49,000) and
NINCO restricted to `NINCO/NINCO_OOD_classes` (5,878 — **not** the full release
tree, which contains ~15,400 images and inflates near-OOD by several points);
far-OOD = iNaturalist, Textures, OpenImage-O.

## Known pitfalls

1. **NINCO.** Load `NINCO/NINCO_OOD_classes` only. A recursive glob over the
   NINCO release picks up `popular_datasets_subsamples` as well.
2. **Orientation.** See the table above. `LogitNormKD` + Energy is the case that
   bites; on CIFAR-100 it even flips on one seed of three and not the other two.
3. **Places365 / CIFAR-10 for DKD.** Two evaluations in an earlier snapshot were
   superseded; `results/` holds the current values (Places365 0.771, CIFAR-10
   0.754). Aggregates and all conclusions are unchanged.
4. **MDSEns weighting.** The ensemble weights are equal by construction. OpenOOD's
   `alpha_selector` fits them by logistic regression on ID-vs-OOD validation data,
   which leaks OOD information into the detector; we do not use it.
