"""
Attack-axis evaluations on freshly simulated episodes (never seen in training):

  E4  FDIA detection rate vs attack strength (|a|_inf / sigma in 1.5..8)
  E5  detection delay: steps from attack onset to first alarm

Usage: python eval_attack_axis.py --ckpt mamba_none_s0_L64
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sim"))
from grid39 import Grid39                       # noqa: E402
from scenarios import ScenarioConfig, ScenarioRunner, DAY  # noqa: E402
from models.nets import TwoBranchDetector       # noqa: E402
from scripts.common import (ATTACK_CLASSES, CKPT_DIR, RESULTS, device,  # noqa: E402
                            load_meta, load_split)

L = 128


def fresh_fdia_episodes(grid, n_ep=30, strengths=None, seed=77_001):
    """FDIA episodes at controlled strengths; ~40 windows per level."""
    strengths = strengths or [1.5, 2, 3, 4, 5, 6, 8]
    rng = np.random.default_rng(seed)
    cfg = ScenarioConfig()
    X, y, s_levels = [], [], []
    meta = load_meta()
    mu = np.array(meta["chan_mean"], dtype=np.float32)
    sd = np.array(meta["chan_std"], dtype=np.float32)
    per_level = 100
    for lev in strengths:
        got = 0
        run_seed = seed + int(lev * 100)
        while got < per_level:
            run_seed += 1
            runner = ScenarioRunner(grid, cfg, seed=run_seed)
            T = DAY
            t0 = int(rng.integers(3 * 3600 // 5, T - 600))
            dur = int(rng.uniform(20, 30) * 12)
            k = int(rng.integers(2, 6))
            buses = sorted(rng.choice(
                [b for b in grid.bus_ids if b != 31], size=k,
                replace=False).tolist())
            a = grid.fdia_vector(buses, 0.02)
            a = a * (lev * grid.sigma / np.abs(a).max())
            ev = [dict(kind="fdia", t0=t0, dur=dur, t1=t0 + dur, a_vec=a,
                       onset=int(rng.integers(0, 60)), buses=buses)]
            z, lab, _ = runner.run(T, ev)
            zs = (z - mu) / sd
            # anchor the onset inside the middle half of the window,
            # matching the training distribution
            for _ in range(4):
                frac = rng.uniform(0.25, 0.75)
                st = int(t0 - frac * L)
                if st < 0 or st + L > T:
                    continue
                X.append(zs[st:st + L]); y.append(2)
                s_levels.append(lev)
                got += 1
    return (np.stack(X).astype(np.float32), np.array(y),
            np.array(s_levels, dtype=np.float32))


@torch.no_grad()
def batch_score(model, X, bs=512):
    dev = device()
    model.eval()
    logits = []
    for s in range(0, len(X), bs):
        xb = torch.tensor(X[s:s + bs], device=dev)
        logits.append(model(xb)[0].cpu().numpy())
    return np.concatenate(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="mamba_none_s0_L64")
    args = ap.parse_args()
    grid = Grid39()
    meta = load_meta()
    model = TwoBranchDetector(m=meta["m"], d=128, enc_layers=2,
                              encoder_name=args.ckpt.split("_")[0]).to(device())
    model.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, f"det_{args.ckpt}.pt"), weights_only=True))

    # ---- E4: strength sweep -------------------------------------------
    X, y, lev = fresh_fdia_episodes(grid)
    logits = batch_score(model, X)
    pred_cls = logits.argmax(1)
    pred_atk = np.isin(pred_cls, [2, 3, 4]).astype(int)
    out = {"strength_sweep": []}
    for lv in sorted(set(lev.tolist())):
        m = lev == lv
        out["strength_sweep"].append({
            "level_sigma": float(lv),
            "det_rate": float(pred_atk[m].mean()),
            "bdd_rate": 0.0 if lv > 0 else None,
        })
        print(f"level {lv:4.1f}s  det_rate {pred_atk[m].mean():.3f}")

    # ---- E5: detection delay ------------------------------------------
    cfg = ScenarioConfig()
    rng = np.random.default_rng(88_002)
    delays = {"fdia": [], "replay": [], "rfdi": []}
    mu = np.array(meta["chan_mean"], dtype=np.float32)
    sd = np.array(meta["chan_std"], dtype=np.float32)
    for kind in delays:
        for ep in range(8):
            runner = ScenarioRunner(grid, cfg, seed=90_000 + 100 * ep + 7)
            T = DAY
            t0 = int(rng.integers(3 * 3600 // 5, T - 600))
            dur = int(rng.uniform(15, 30) * 12)
            ev = None
            if kind == "fdia":
                k = int(rng.integers(2, 6))
                buses = sorted(rng.choice(
                    [b for b in grid.bus_ids if b != 31], size=k,
                    replace=False).tolist())
                a = grid.fdia_vector(buses, 0.02)
                a = a * (rng.uniform(3, 8) * grid.sigma / np.abs(a).max())
                ev = dict(kind="fdia", t0=t0, dur=dur, t1=t0 + dur, a_vec=a,
                          onset=int(rng.integers(0, 60)))
            elif kind == "rfdi":
                n_ch = int(rng.uniform(0.1, 0.3) * grid.m)
                ch = rng.choice(grid.m, size=n_ch, replace=False)
                bias = rng.choice([-1, 1], (n_ch,)) \
                    * rng.uniform(3, 8) * grid.sigma
                ev = dict(kind="rfdi", t0=t0, dur=dur, t1=t0 + dur,
                          channels=ch, bias=bias, onset=int(rng.integers(0, 30)))
            else:
                ev = dict(kind="replay", t0=t0, dur=dur, t1=t0 + dur,
                          lag=int(rng.uniform(12, 60) * 12))
            z, lab, _ = runner.run(T, [ev])
            zs = (z - mu) / sd
            # score every window whose END is within [t0-L, t0+dur]
            for tend in range(t0, min(t0 + dur, T)):
                if tend - L < 0:
                    continue
                w = torch.tensor(zs[tend - L:tend][None], device=device())
                with torch.no_grad():
                    lg = model(w)[0].cpu().numpy()
                pc = int(lg.argmax())
                if pc in (2, 3, 4):
                    delays[kind].append(tend - t0)   # steps (5 s each)
                    break
    out["delay_steps"] = {
        k: (float(np.mean(v)) if v else None,
            float(np.median(v)) if v else None,
            len(v),
            [float(np.percentile(v, 25)) if v else None,
             float(np.percentile(v, 75)) if v else None])
        for k, v in delays.items()}
    for k, v in out["delay_steps"].items():
        print(f"delay {k}: mean {v[0]} steps, median {v[1]}, detected {v[2]}/8")

    with open(os.path.join(RESULTS, f"attack_axis_{args.ckpt}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
