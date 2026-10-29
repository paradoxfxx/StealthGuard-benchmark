"""
Train a two-branch detector (or baseline) on the simulation benchmark.

    python train_detector.py --model mamba --aug none
    python train_detector.py --model mamba --aug diffusion --atk_frac 0.1
    python train_detector.py --model transformer --aug wgan --atk_frac 0.25

Saves results/det_<tag>.json with test metrics and val curves.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.nets import DetectorLoss, TwoBranchDetector  # noqa: E402
import torch as _t; _t.set_float32_matmul_precision("high")
from scripts.common import (ATTACK_CLASSES, CKPT_DIR, RESULTS,  # noqa: E402
                            class_weights, detection_metrics, device,
                            load_meta, load_split, macro_f1, physics_projector,
                            set_seed)
from scripts.train_gen import (sample_diffusion, sample_wgan,  # noqa: E402
                               subsample_attacks)
from models.generative import CondDDPM, CondGenerator  # noqa: E402

AUG_TARGET = 3000      # synthetic+real attack windows per class after augmentation


def build_augmented(args, L):
    """Returns X, y for training, with optional synthetic attack windows."""
    X, y = load_split("train", L)
    if args.atk_frac < 1.0:
        X, y = subsample_attacks(X, y, args.atk_frac, seed=args.seed)
    if args.aug == "none":
        return X, y
    m = X.shape[2]
    n_real = {c: int((y == c).sum()) for c in ATTACK_CLASSES}
    per_class = {c: max(0, args.aug_target - n_real[c])
                 for c in ATTACK_CLASSES}
    if sum(per_class.values()) == 0:
        return X, y
    if args.aug == "diffusion":
        gen = CondDDPM(m=m, L=L).to(device())
        gen.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f"ddpm_f{args.atk_frac}_s0_L{L}.pt"),
            weights_only=True))
        gen.eval()
        Xs, ys = sample_diffusion(gen, per_class, L, m)
    else:
        gen = CondGenerator(m, L=L).to(device())
        gen.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f"wgan_f{args.atk_frac}_s0_L{L}.pt"),
            weights_only=True))
        gen.eval()
        Xs, ys = sample_wgan(gen, per_class, L, m)
    print(f"  augmented with {len(ys)} synthetic windows "
          f"({per_class})")
    return np.concatenate([X, Xs]), np.concatenate([y, ys])


@torch.no_grad()
def evaluate(model, X, y, bs=512, dev=None, subtype=None):
    dev = dev or device()
    model.eval()
    logits = []
    for s in range(0, len(X), bs):
        xb = torch.tensor(X[s:s + bs], device=dev)
        logits.append(model(xb)[0].cpu().numpy())
    logits = np.concatenate(logits)
    pred = logits.argmax(1)
    rec = {f"rec_{c}": float((pred[y == c] == c).mean())
           for c in range(5)}
    out = {"macro_f1": float(macro_f1(y, pred)),
           "acc": float((pred == y).mean()), **rec}
    out.update(detection_metrics(y, logits))
    if subtype is not None:
        # per-subtype recall for natural-disturbance windows (y==1)
        out["subtype"] = {}
        for k, code in [("load_step", 0), ("gen_trip", 1),
                        ("line_outage", 2)]:
            m = (y == 1) & (subtype == code)
            if m.sum() == 0:
                continue
            out["subtype"][k] = {
                "n": int(m.sum()),
                "recall": float((pred[m] == 1).mean()),
                "as_fdia": float((pred[m] == 2).mean()),
            }
    return out, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba",
                    choices=["mamba", "gru", "lstm", "transformer", "cnn", "grumamba"])
    ap.add_argument("--aug", default="none",
                    choices=["none", "diffusion", "wgan"])
    ap.add_argument("--atk_frac", type=float, default=1.0)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--no_phys", action="store_true",
                    help="disable physics-informed loss (ablation)")
    ap.add_argument("--no_diff", action="store_true",
                    help="disable first-difference input stem (ablation)")
    ap.add_argument("--cw", default="inv", choices=["inv", "sqrt"],
                    help="class weighting scheme")
    ap.add_argument("--data_suffix", default="",
                    help="dataset variant suffix, e.g. _static")
    ap.add_argument("--aug_target", type=int, default=3000,
                    help="attack windows per class after augmentation")
    args = ap.parse_args()
    L = args.L or json.load(open(os.path.join(RESULTS, f"sim_meta{args.data_suffix}.json")))["L"]
    set_seed(args.seed)
    dev = device()

    Xtr, ytr = build_augmented(args, L)
    Xva, yva = load_split("val", L, suffix=args.data_suffix)
    Xte, yte = load_split("test", L, suffix=args.data_suffix)
    print(f"[{args.model}|aug={args.aug}|frac={args.atk_frac}|L={L}] "
          f"train={len(ytr)} val={len(yva)} test={len(yte)}")

    cw = class_weights(ytr, mode=args.cw).to(dev)
    M, sigma, _ = physics_projector()
    meta = json.load(open(os.path.join(RESULTS, f"sim_meta{args.data_suffix}.json")))
    mu = torch.tensor(meta["chan_mean"], dtype=torch.float32)
    sd = torch.tensor(meta["chan_std"], dtype=torch.float32)
    crit = DetectorLoss(torch.tensor(M), mu.to(dev), sd.to(dev), sigma,
                        lam_recon=0.3,
                        lam_phys=0.0 if args.no_phys else 0.1).to(dev)

    model = TwoBranchDetector(m=Xtr.shape[2], d=args.d, enc_layers=args.enc_layers,
                              encoder_name=args.model,
                              diff_input=not args.no_diff).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=len(ds) > 2000)

    tag = f"{args.model}_{args.aug}{'' if args.atk_frac==1.0 else args.atk_frac}" \
          f"_s{args.seed}_L{L}" + ("_nophys" if args.no_phys else "") \
          + ("_nodiff" if args.no_diff else "") \
          + (f"_t{args.aug_target}" if args.aug_target != 3000 else "") \
          + (f"_d{args.d}" if args.d != 128 else "") \
          + (f"_l{args.enc_layers}" if args.enc_layers != 2 else "") \
          + (args.data_suffix or "")
    best, best_state, curve = -1, None, []
    for ep in range(args.epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            logits, xr = model(xb)
            loss, parts = crit(logits, xr, xb, yb, cw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vm, _ = evaluate(model, Xva, yva, dev=dev)
        curve.append({"ep": ep, "val_f1": vm["macro_f1"]})
        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} val_f1 {vm['macro_f1']:.4f} "
                  f"({', '.join(f'{k}={v:.3f}' for k, v in parts.items())})")

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(CKPT_DIR, f"det_{tag}.pt"))
    te_sub = np.load(os.path.join(
        RESULTS, f"sim_test{args.data_suffix or ''}.npz"))["subtype"]
    tm, logits = evaluate(model, Xte, yte, dev=dev, subtype=te_sub)
    tm.update(params=n_par, val_f1=best, args=vars(args))
    print(f"[RESULT {tag}] " +
          " ".join(f"{k}={v:.4f}" for k, v in tm.items() if isinstance(v, float)))
    with open(os.path.join(RESULTS, f"det_{tag}.json"), "w") as fh:
        json.dump({"metrics": tm, "curve": curve}, fh, indent=1)
    np.savez_compressed(os.path.join(RESULTS, f"logits_{tag}.npz"),
                        logits=logits, y=yte, subtype=te_sub)


if __name__ == "__main__":
    main()
