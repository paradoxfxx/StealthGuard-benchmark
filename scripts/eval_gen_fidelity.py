"""Quantitative fidelity of generative augmentation samples.

For each generator (CDDM diffusion, WGAN-GP) trained on the SAME scarce
FDIA subset (frac=0.1, seed 0), sample N synthetic windows and compare
channel-wise distributions against the real training subset:
  - mean |mean shift| per channel (standardized units)
  - mean |log std ratio| per channel
  - mean per-channel 1-D Wasserstein distance
Writes results/gen_fidelity.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from models.generative import CondDDPM, CondGenerator  # noqa: E402
from scripts.common import CKPT_DIR, RESULTS, device, load_split  # noqa: E402
from scripts.train_gen import subsample_attacks  # noqa: E402

L, M, N_SYN = 128, 131, 256


def stats_per_channel(W):
    return W.mean(axis=(0, 1)), W.std(axis=(0, 1))


def main():
    dev = device()
    X, y = load_split("train", L)
    Xs, ys = subsample_attacks(X, y, 0.1, seed=0)   # same subset as train_gen
    real = Xs[ys == 2]                                # FDIA subset (n=60)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(real), size=N_SYN, replace=True)
    real_boot = real[idx]        # bootstrap reference of same size

    out = {"n_real": int(len(real)), "n_synthetic": N_SYN}
    for name, gen in [("diffusion", None), ("wgan", None)]:
        if name == "diffusion":
            g = CondDDPM(m=M, L=L).to(dev)
            g.load_state_dict(torch.load(
                os.path.join(CKPT_DIR, "ddpm_f0.1_s0_L128.pt"),
                weights_only=True))
            g.eval()
            with torch.no_grad():
                yy = torch.full((N_SYN,), 2, dtype=torch.long, device=dev)
                syn = g.ddim_sample(N_SYN, yy, L, dev, steps=50).cpu().numpy()
        else:
            g = CondGenerator(M, L=L).to(dev)
            g.load_state_dict(torch.load(
                os.path.join(CKPT_DIR, "wgan_f0.1_s0_L128.pt"),
                weights_only=True))
            g.eval()
            with torch.no_grad():
                yy = torch.full((N_SYN,), 2, device=dev)
                yoh = F.one_hot(yy, 5).float()
                z = torch.randn(N_SYN, g.z_fc.in_features, device=dev)
                syn = g(z, yoh).cpu().numpy()
        mu_r, sd_r = stats_per_channel(real)
        mu_s, sd_s = stats_per_channel(syn)
        mean_shift = np.abs(mu_s - mu_r).mean()
        log_std = np.abs(np.log(sd_s / sd_r)).mean()
        wd = np.mean([wasserstein_distance(real[:, :, c].ravel(),
                                           syn[:, :, c].ravel())
                      for c in range(M)])
        # bootstrap floor: same-size resample of the real subset
        mu_b, sd_b = stats_per_channel(real_boot)
        wd_boot = np.mean([wasserstein_distance(real[:, :, c].ravel(),
                                                real_boot[:, :, c].ravel())
                           for c in range(M)])
        out[name] = {"mean_abs_shift": round(float(mean_shift), 4),
                     "mean_abs_log_std_ratio": round(float(log_std), 4),
                     "mean_channel_w1": round(float(wd), 4),
                     "real_bootstrap_w1_floor": round(float(wd_boot), 4)}
        print(name, out[name])
    with open(os.path.join(RESULTS, "gen_fidelity.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
