"""FDIA recall vs attack onset ramp length (fresh episodes, GRU).

ramp in {0, 30, 60, 128, 256} steps; magnitude fixed at 6 sigma.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.stdout.reconfigure(encoding="utf-8")
from grid39 import Grid39  # noqa: E402
from scenarios import ScenarioConfig, ScenarioRunner, DAY  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "models"))
from nets import TwoBranchDetector  # noqa: E402
from scripts.common import device, load_meta  # noqa: E402

L = 128

RAMPS = [0, 30, 60, 128, 256]
PER = 60          # windows per ramp level
MAG = 6.0         # |a|_inf in sigma


def main():
    grid = Grid39()
    meta = load_meta()
    mu = np.array(meta["chan_mean"], dtype=np.float32)
    sd = np.array(meta["chan_std"], dtype=np.float32)
    dev = device()
    model = TwoBranchDetector(m=meta["m"], d=128, enc_layers=2,
                              encoder_name="gru").to(dev)
    ck = os.path.join(ROOT, "ckpt", "det_gru_none_s0_L128.pt")
    model.load_state_dict(torch.load(ck, weights_only=True))
    model.eval()

    rng = np.random.default_rng(99_001)
    cfg = ScenarioConfig()
    out = []
    for ramp in RAMPS:
        got, dets = 0, []
        run_seed = 90_000
        while got < PER:
            run_seed += 1
            runner = ScenarioRunner(grid, cfg, seed=run_seed)
            T = DAY
            t0 = int(rng.integers(3 * 3600 // 5, T - 500))
            dur = int(rng.uniform(20, 30) * 12)
            k = int(rng.integers(2, 6))
            buses = sorted(rng.choice(
                [b for b in grid.bus_ids if b != 31], size=k,
                replace=False).tolist())
            a = grid.fdia_vector(buses, 0.02)
            a = a * (MAG * grid.sigma / np.abs(a).max())
            ev = dict(kind="fdia", t0=t0, dur=dur, t1=t0 + dur, a_vec=a,
                      onset=ramp, buses=buses)
            z, lab, _ = runner.run(T, [ev])
            zs = (z - mu) / sd
            for _ in range(4):
                frac = rng.uniform(0.25, 0.75)
                st = int(t0 - frac * L)
                if st < 0 or st + L > T:
                    continue
                w = torch.tensor(zs[st:st + L][None], device=dev)
                with torch.no_grad():
                    pc = int(model(w)[0].argmax())
                dets.append(pc in (2, 3, 4))
                got += 1
        rec = float(np.mean(dets))
        out.append({"ramp_steps": ramp, "fdia_recall": rec})
        print(f"ramp={ramp:4d} steps ({ramp*5/60:5.1f} min): "
              f"recall={rec:.3f} (n={got})")
    with open(os.path.join(ROOT, "results", "ramp_sweep.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
