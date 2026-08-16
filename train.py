#!/usr/bin/env python
"""Train a TANGO (LogitNormKD) student, or the vanilla-KD baseline.

Reproduces the paper's CIFAR-100 setup: frozen DINOv2-S + linear probe
teacher, resnet8x4 student, 100 epochs SGD.

    # TANGO, seed 0
    python train.py --method logitnormkd --seed 0

    # vanilla KD baseline
    python train.py --method vanillakd --seed 0

    # 2-epoch smoke test (no GPU needed, but slow on CPU)
    python train.py --method logitnormkd --epochs 2 --limit-batches 5

Hyperparameters default to the released configs
(configs/cifar100_*_seed*.yaml): 100 epochs, SGD lr 0.05, multistep
[60,80,90] gamma 0.1, weight decay 5e-4, momentum 0.9, batch 64.
LogitNormKD: T=4, tau=0.04, ce_w=kd_w=1. VanillaKD: T=4, alpha=0.1, beta=9.

The trained student is written to --out; evaluate it with eval_ood.py.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from tango import LogitNormKD, VanillaKD, resnet8x4
from tango.data import get_id_loaders
from tango.teacher import dinov2_small


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = student(x)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="logitnormkd",
                    choices=("logitnormkd", "vanillakd"))
    ap.add_argument("--id-dataset", default="cifar100",
                    choices=("cifar100", "cifar10"))
    ap.add_argument("--teacher-head", default=None,
                    help="default: checkpoints/teacher_heads/dinov2s_<id>_head.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--milestones", default="60,80,90")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="stop each epoch early (smoke tests)")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or f"runs/{args.method}_{args.id_dataset}_seed{args.seed}.pt"
    head = args.teacher_head or (
        f"checkpoints/teacher_heads/dinov2s_{args.id_dataset}_head.pt")

    train_loader, test_loader, nc = get_id_loaders(
        args.id_dataset, args.data_root, args.batch_size, args.num_workers)

    print(f"method={args.method} seed={args.seed} device={dev} classes={nc}")
    print(f"teacher head: {head}")
    teacher = dinov2_small(nc, head_ckpt=head).to(dev).eval()
    student = resnet8x4(num_classes=nc).to(dev)

    if args.method == "logitnormkd":
        model = LogitNormKD(student, teacher, temperature=args.temperature,
                            tau=args.tau, ce_weight=1.0, kd_weight=1.0)
    else:
        model = VanillaKD(student, teacher, temperature=args.temperature,
                          ce_weight=0.1, kd_weight=9.0)
    model = model.to(dev)

    opt = torch.optim.SGD(model.parameters_to_optimize(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=False)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(m) for m in args.milestones.split(",")],
        gamma=args.gamma)

    best = 0.0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), {}
        for i, (x, y) in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            _, losses = model(x, y)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k, v in losses.items():
                tot[k] = tot.get(k, 0.0) + v.item()
        sched.step()

        acc = evaluate(student, test_loader, dev)
        best = max(best, acc)
        nb = (args.limit_batches or len(train_loader))
        msg = "  ".join(f"{k}={v / nb:.4f}" for k, v in tot.items())
        print(f"ep {ep:3d}/{args.epochs}  {msg}  test_acc={acc:.4f}  "
              f"best={best:.4f}  lr={sched.get_last_lr()[0]:.4f}  "
              f"({time.time() - t0:.0f}s)")

        if acc >= best:
            torch.save({"model": student.state_dict(), "epoch": ep,
                        "test_acc": acc, "method": args.method,
                        "seed": args.seed}, out)

    print(f"\nbest test_acc = {best:.4f}   saved -> {out}")
    print(f"now run:  python eval_ood.py --checkpoint {out}")


if __name__ == "__main__":
    main()
