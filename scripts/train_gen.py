"""
Train a conditional generative model (CDDM diffusion or WGAN-GP) on the
(real, possibly scarce) attack windows of the training set, then sample
synthetic attack windows for augmentation.

Usage:
    python train_gen.py --gen diffusion --atk_frac 1.0
    python train_gen.py --gen wgan     --atk_frac 0.1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.generative import CondDDPM, CondCritic, CondGenerator, wgan_gp_loss  # noqa: E402
from scripts.common import (ATTACK_CLASSES, CKPT_DIR, RESULTS,  # noqa: E402
                            device, load_split, set_seed)


def subsample_attacks(X, y, frac, seed=0):
    rng = np.random.default_rng(seed)
    keep = np.ones(len(y), bool)
    for c in ATTACK_CLASSES:
        idx = np.where(y == c)[0]
        n = max(6, int(round(frac * len(idx))))
        drop = rng.choice(idx, size=len(idx) - n, replace=False)
        keep[drop] = False
    return X[keep], y[keep]


def train_diffusion(Xa, ya, epochs=150, bs=128, tag=""):
    dev = device()
    model = CondDDPM(m=Xa.shape[2], L=Xa.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    ds = TensorDataset(torch.tensor(Xa), torch.tensor(ya))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = model(xb, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  [ddpm {tag}] ep{ep} loss {tot/nb:.4f}")
    return model


def train_wgan(Xa, ya, epochs=150, bs=128, n_critic=3, tag=""):
    dev = device()
    L, m = Xa.shape[1], Xa.shape[2]
    G = CondGenerator(m, L=L).to(dev)
    C = CondCritic(m, L=L).to(dev)
    optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optC = torch.optim.Adam(C.parameters(), lr=5e-5, betas=(0.0, 0.9))
    ds = TensorDataset(torch.tensor(Xa), torch.tensor(ya))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    y_oh = F.one_hot(torch.tensor(ya), 5).float()
    ds = TensorDataset(torch.tensor(Xa), y_oh)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    nz = G.z_fc.in_features
    for ep in range(epochs):
        for i, (xb, yb) in enumerate(dl):
            xb, yb = xb.to(dev), yb.to(dev)
            z = torch.randn(xb.shape[0], nz, device=dev)
            with torch.no_grad():
                fake = G(z, yb)
            dloss, wdist = wgan_gp_loss(C, xb, fake, yb)
            optC.zero_grad(); dloss.backward(); optC.step()
            if i % n_critic == 0:
                z = torch.randn(xb.shape[0], nz, device=dev)
                gfake = G(z, yb)
                gloss = -C(gfake, yb).mean()
                optG.zero_grad(); gloss.backward(); optG.step()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  [wgan {tag}] ep{ep} W-dist {wdist:.3f}")
    return G


@torch.no_grad()
def sample_diffusion(model, per_class, L, m, bs=256):
    dev = device()
    out_X, out_y = [], []
    for c, n in per_class.items():
        for s in range(0, n, bs):
            k = min(bs, n - s)
            y = torch.full((k,), c, dtype=torch.long, device=dev)
            x = model.ddim_sample(k, y, L, dev, steps=50).cpu().numpy()
            out_X.append(x); out_y.append(np.full(k, c))
    return np.concatenate(out_X), np.concatenate(out_y)


@torch.no_grad()
def sample_wgan(G, per_class, L, m, bs=256):
    dev = device()
    nz = G.z_fc.in_features
    out_X, out_y = [], []
    for c, n in per_class.items():
        for s in range(0, n, bs):
            k = min(bs, n - s)
            y = F.one_hot(torch.full((k,), c), 5).float().to(dev)
            z = torch.randn(k, nz, device=dev)
            x = G(z, y).cpu().numpy()
            out_X.append(x); out_y.append(np.full(k, c))
    return np.concatenate(out_X), np.concatenate(out_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", choices=["diffusion", "wgan"], required=True)
    ap.add_argument("--atk_frac", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=128)
    args = ap.parse_args()
    set_seed(args.seed)

    X, y = load_split("train", args.L)
    Xs, ys = subsample_attacks(X, y, args.atk_frac, seed=args.seed)
    Xa = Xs[np.isin(ys, ATTACK_CLASSES)]
    ya = ys[np.isin(ys, ATTACK_CLASSES)]
    print(f"[gen={args.gen}] training on {len(ya)} attack windows "
          f"(frac={args.atk_frac})")

    if args.gen == "diffusion":
        ep = args.epochs or 150
        model = train_diffusion(Xa, ya, epochs=ep, tag=f"f{args.atk_frac}")
        torch.save(model.state_dict(),
                   os.path.join(CKPT_DIR, f"ddpm_f{args.atk_frac}_s{args.seed}_L{args.L}.pt"))
    else:
        ep = args.epochs or 150
        G = train_wgan(Xa, ya, epochs=ep, tag=f"f{args.atk_frac}")
        torch.save(G.state_dict(),
                   os.path.join(CKPT_DIR, f"wgan_f{args.atk_frac}_s{args.seed}_L{args.L}.pt"))
    print("saved generator checkpoint")


if __name__ == "__main__":
    main()
