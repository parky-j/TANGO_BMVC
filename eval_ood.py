#!/usr/bin/env python
"""Evaluate a released (or your own) student checkpoint for OOD detection.

Reproduces the paper's Table 1 numbers for CIFAR-100 ID. Datasets download
automatically on first run.

    # TANGO student, all auto-downloading splits, every score
    python eval_ood.py --checkpoint checkpoints/cifar100_resnet8x4/logitnormkd_seed0.pt

    # the vanilla-KD baseline, SVHN only
    python eval_ood.py --checkpoint checkpoints/cifar100_resnet8x4/vanillakd_seed0.pt \
                       --ood svhn

Expected (seed 0, CIFAR-100 -> SVHN):
    logitnormkd  entropy 0.978    energy_t1 0.113  <- inverted
    vanillakd    entropy 0.839    energy_t1 0.907

That is the paper's point in one line. Under LogitNormKD the magnitude axis
is compressed ~30x, so Entropy (softmax shape) wins and Energy -- which reads
magnitude -- degrades; on this seed it inverts outright (AUROC < 0.5 means the
ID/OOD ordering flipped). Under vanilla KD the ranking is the other way round.
AUROC is reported RAW, without orientation correction, so inversions stay
visible; Tab. 1 and Fig. 3 of the paper orient each score empirically, and
report 3-seed means rather than the single seed you get here.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from tango import (SCORES, auroc, collect_logits, fpr_at_95tpr,
                   logit_norm_stats, resnet8x4)
from tango.data import AUTO_OOD, get_id_loaders, get_ood_loader


def load_student(path, num_classes, device):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
          for k, v in sd.items()}
    net = resnet8x4(num_classes=num_classes)
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--id-dataset", default="cifar100",
                    choices=("cifar100", "cifar10"))
    ap.add_argument("--ood", default="svhn,cifar10,mnist,dtd",
                    help=f"comma-separated. auto-download: {','.join(AUTO_OOD)}")
    ap.add_argument("--scores", default="entropy,energy_t1,msp,max_logit")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images per OOD split (smoke tests)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ood_names = [s.strip() for s in args.ood.split(",") if s.strip()]
    score_names = [s.strip() for s in args.scores.split(",") if s.strip()]
    for s in score_names:
        if s not in SCORES:
            raise SystemExit(f"unknown score {s!r}; available: {list(SCORES)}")

    _, id_test, nc = get_id_loaders(args.id_dataset, args.data_root,
                                    args.batch_size, args.num_workers)
    student = load_student(args.checkpoint, nc, dev)

    id_logits = collect_logits(student, id_test, dev)
    id_acc_note = ""
    stats = logit_norm_stats(student, id_test, dev)
    print(f"\ncheckpoint : {args.checkpoint}")
    print(f"device     : {dev}   ID: {args.id_dataset} "
          f"({id_logits.shape[0]} test images){id_acc_note}")
    print(f"mean ||z||_2 on ID = {stats['mean_logit_norm']:.2f}   "
          f"(paper: ~1.3 for LogitNormKD, ~31 for vanilla KD)\n")

    res = {"checkpoint": args.checkpoint, "id_dataset": args.id_dataset,
           "mean_logit_norm_id": stats["mean_logit_norm"], "ood": {}}

    w = max(len(s) for s in score_names) + 2
    header = f"{'OOD':<12}" + "".join(f"{s:>{w + 8}}" for s in score_names)
    print(header)
    print("-" * len(header))

    for name in ood_names:
        loader = get_ood_loader(name, args.id_dataset, args.data_root,
                                args.batch_size, args.num_workers, args.limit)
        ood_logits = collect_logits(student, loader, dev)
        row, cells = {}, []
        for s in score_names:
            fn = SCORES[s]
            a = auroc(fn(id_logits).numpy(), fn(ood_logits).numpy())
            f = fpr_at_95tpr(fn(id_logits).numpy(), fn(ood_logits).numpy())
            row[s] = {"auroc": a, "fpr95": f}
            cells.append(f"{a:.4f}/{f:.3f}".rjust(w + 8))
        res["ood"][name] = {"n": int(ood_logits.shape[0]), "scores": row}
        print(f"{name:<12}" + "".join(cells))

    print("\n(each cell is AUROC/FPR95; higher AUROC and lower FPR95 are better)")

    inverted = [(o, s) for o, d in res["ood"].items()
                for s, v in d["scores"].items() if v["auroc"] < 0.5]
    if inverted:
        print("\nAUROC < 0.5 (ID/OOD ordering inverted, reported raw):")
        for o, s in inverted:
            print(f"    {s} on {o}: {res['ood'][o]['scores'][s]['auroc']:.4f} "
                  f"-> {1 - res['ood'][o]['scores'][s]['auroc']:.4f} re-oriented")
        print("    On LogitNormKD students this is expected for magnitude-reading")
        print("    scores (Energy/ReAct/ASH): the training objective compressed the")
        print("    axis they depend on. See Sec. 4.4 and supp. Sec. 13.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
