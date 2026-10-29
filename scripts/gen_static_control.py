"""Control experiment: identical scenario machinery with the
correlated load dynamics REMOVED (constant loads, iid measurement noise)
=> the classic 'static snapshot + iid noise' evaluation setting.

Writes sim_{split}_static.npz + sim_meta_static.json (L=128).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.stdout.reconfigure(encoding="utf-8")
import run_sim  # noqa: E402
from scenarios import ScenarioConfig  # noqa: E402
from grid39 import Grid39  # noqa: E402

ScenarioConfig.daily_amp1 = 0.14     # keep the daily curve: operating-point
ScenarioConfig.daily_amp2 = 0.05     # diversity across the dataset
ScenarioConfig.ar_load_sig = 1e-9    # remove fast correlated load noise
ScenarioConfig.ar_gen_sig = 1e-9     # => near-static operating points,
                                     #    iid measurement noise only

run_sim.L = 128
RESULTS = os.path.join(ROOT, "results")
grid = Grid39()
meta = {"L": 128, "m": grid.m, "n_states": grid.n, "sigma": grid.sigma,
        "chi2_th": float(grid.chi2_th), "Ts_s": 5, "static_control": True,
        "splits": {}}
for split in ["train", "val", "test"]:
    X, y, sub = run_sim.build_split(split, grid)
    u, cnt = np.unique(y, return_counts=True)
    print(f"[{split}/static] X={X.shape} "
          f"classes={dict(zip(u.tolist(), cnt.tolist()))}")
    np.savez_compressed(os.path.join(RESULTS, f"sim_{split}_static.npz"),
                        X=X, y=y, subtype=sub)
    meta["splits"][split] = {"n": int(len(y)),
                             "counts": {int(k): int(v)
                                        for k, v in zip(u, cnt)}}
norm = X[y == 0].astype(np.float32) if split == "test" else None
Xtr = np.load(os.path.join(RESULTS, "sim_train_static.npz"))["X"]
ytr = np.load(os.path.join(RESULTS, "sim_train_static.npz"))["y"]
norm = Xtr[ytr == 0].astype(np.float32)
meta["chan_mean"] = norm.reshape(-1, norm.shape[-1]).mean(0).round(5).tolist()
meta["chan_std"] = norm.reshape(-1, norm.shape[-1]).std(0).round(5).tolist()
Xte = np.load(os.path.join(RESULTS, "sim_test_static.npz"))
meta["bdd_alarm_rates_test"] = run_sim.bdd_alarm_rates(
    grid, Xte["X"].astype(np.float32), Xte["y"])
with open(os.path.join(RESULTS, "sim_meta_static.json"), "w") as fh:
    json.dump(meta, fh, indent=1)
print("BDD(static):", meta["bdd_alarm_rates_test"])
