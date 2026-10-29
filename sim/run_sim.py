"""
Build the FDIA/anomaly benchmark dataset from the IEEE 39-bus simulation.

Splits (mutually exclusive seeds, no run shared across splits):

    split   1-day normal runs   1-day natural runs   2-day attack runs (per kind)
    train         12                   12                    14
    val            4                    4                     6
    test           5                    5                    10

Windows: L = 128 steps (10.7 min at Ts = 5 s), fully inside same-label
segments so every window has exactly one class.

Outputs (results/):
    sim_<split>.npz : X (N,L,m) float16, y (N,), subtype (N,)
    sim_meta.json   : counts, channel stats, BDD alarm rates
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
from grid39 import Grid39                       # noqa: E402
from scenarios import (ScenarioConfig, ScenarioRunner,  # noqa: E402
                       make_run, DAY, TS)

L = 128
RESULTS = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS, exist_ok=True)

NAT_SUB = {"load_step": 0, "gen_trip": 1, "line_outage": 2}
ATTACKS = [(2, "fdia"), (3, "replay"), (4, "rfdi")]
KIND_CLASS = {"load_step": 1, "gen_trip": 1, "line_outage": 1,
              "fdia": 2, "replay": 3, "rfdi": 4}

SPLIT_PLAN = {
    "train": dict(normal_runs=12, natural_runs=16, attack_runs=28,
                  targets=(10000, 2500, 600, 600, 600)),
    "val":   dict(normal_runs=4, natural_runs=4, attack_runs=6,
                  targets=(1500, 400, 200, 200, 200)),
    "test":  dict(normal_runs=5, natural_runs=5, attack_runs=10,
                  targets=(1500, 500, 500, 500, 500)),
}
SEED_BASE = {"train": 10_000, "val": 20_000, "test": 30_000}


def segments(lab: np.ndarray, c: int):
    """Contiguous (start, end) runs of label c."""
    mask = lab == c
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    splits = np.where(np.diff(idx) > 1)[0]
    starts = idx[np.concatenate([[0], splits + 1])]
    ends = idx[np.concatenate([splits, [len(idx) - 1]])] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def harvest(z, lab, events, rng, stride_n=8, mpe_nat=40, mpe_atk=25):
    """Collect windows of every class present in one run.

    Event windows (natural + attacks) are anchored on the event onset with
    random jitter, so each window contains the pre-event context and the
    onset transient (event-detection setting). Normal windows are placed
    freely inside normal segments.
    """
    sub_of = {}
    kind_of = {}
    for e in events:
        st = NAT_SUB.get(e["kind"], -1)
        for t in range(e["t0"], e["t1"]):
            sub_of[t] = st
            kind_of[t] = e["kind"]
    pools = {c: [] for c in range(5)}
    subs = {c: [] for c in range(5)}

    # normal windows
    for s, e in segments(lab, 0):
        if e - s < L:
            continue
        for st in range(s, e - L + 1, stride_n):
            pools[0].append(z[st:st + L])
            subs[0].append(-1)

    # event windows: onset at a random position in [L/4, 3L/4]
    for e in events:
        kind = e["kind"]
        if kind not in KIND_CLASS:
            continue
        c = KIND_CLASS[kind]
        t0, t1 = e["t0"], e["t1"]
        n_pos = max(2, min(mpe_nat if c == 1 else mpe_atk, t1 - t0))
        used = set()
        for _ in range(n_pos):
            frac = rng.uniform(0.25, 0.75)
            st = int(t0 - frac * L)
            if st < 0 or st + L > len(lab) or st in used:
                continue
            # window must not straddle a *different* labelled event
            labs = lab[st:st + L]
            if (labs != c).sum() > 3 * L // 4:
                continue
            used.add(st)
            pools[c].append(z[st:st + L])
            subs[c].append(sub_of.get(t0, -1) if c == 1 else -1)
    return pools, subs


def attack_schedule(grid, cfg, rng, T, kind, n):
    """n non-overlapping episodes of one attack kind."""
    events, slots = [], []
    guard = 0
    while len(events) < n and guard < 500:
        guard += 1
        dur = int(rng.uniform(*cfg.attack_minutes) * 60 / TS)
        t0 = int(rng.integers(3 * 3600 // TS, T - dur - 200))
        if not all(t0 + dur + 60 < s or s + d + 60 < t0 for s, d in slots):
            continue
        ev = {"kind": kind, "t0": t0, "dur": dur, "t1": t0 + dur}
        if kind == "fdia":
            k = int(rng.integers(*cfg.fdia_k_bus))
            buses = sorted(rng.choice(
                [b for b in grid.bus_ids if b != 31],
                size=k, replace=False).tolist())
            a = grid.fdia_vector(buses, 0.02)
            a = a * (rng.uniform(*cfg.fdia_bias_sig) * grid.sigma
                     / np.abs(a).max())
            ev.update(buses=buses, a_vec=a,
                      onset=int(rng.integers(*cfg.fdia_onset_steps)))
        elif kind == "rfdi":
            n_ch = int(rng.uniform(*cfg.rfdi_frac_channels) * grid.m)
            ch = rng.choice(grid.m, size=n_ch, replace=False)
            mag = rng.uniform(*cfg.rfdi_bias_sig) * grid.sigma
            bias = rng.choice([-1.0, 1.0], (n_ch,)) \
                * mag * rng.uniform(0.4, 1.0, (n_ch,))
            ev.update(channels=ch, bias=bias,
                      onset=int(rng.integers(0, 30)))
        else:
            ev["lag"] = int(rng.uniform(*cfg.replay_lag_minutes) * 60 / TS)
        slots.append((t0, dur))
        events.append(ev)
    return events


def forced_attack_run(grid, cfg, seed, T, kind, rng, n_attacks=8, n_natural=1):
    runner = ScenarioRunner(grid, cfg, seed=seed)
    att = attack_schedule(grid, cfg, rng, T, kind, n_attacks)
    nat = runner.sample_events(T, np.random.default_rng(seed + 9_999),
                               n_natural=n_natural, n_attack=0)
    events = sorted(att + nat, key=lambda e: e["t0"])
    keep, last_end = [], -1
    for e in events:
        if e["t0"] > last_end + 60:
            keep.append(e)
            last_end = e["t1"]
    z, lab, _ = runner.run(T, keep)
    return z, lab, keep


def build_split(split, grid):
    plan = SPLIT_PLAN[split]
    sid = SEED_BASE[split]
    cfg = ScenarioConfig()
    pools = {c: [] for c in range(5)}
    subs = {c: [] for c in range(5)}
    rng = np.random.default_rng(sid + 777)

    def nxt():
        nonlocal sid
        sid += 1
        return sid

    for _ in range(plan["normal_runs"]):
        z, lab, ev = make_run(grid, nxt(), DAY, 0, 0)
        p, s = harvest(z, lab, ev, rng)
        for c in range(5):
            pools[c] += p[c]; subs[c] += s[c]

    for _ in range(plan["natural_runs"]):
        z, lab, ev = make_run(grid, nxt(), DAY, 3, 0)
        p, s = harvest(z, lab, ev, rng)
        for c in range(5):
            pools[c] += p[c]; subs[c] += s[c]

    for _, kind in ATTACKS:
        for _ in range(plan["attack_runs"]):
            arng = np.random.default_rng(nxt() + 500)
            z, lab, ev = forced_attack_run(grid, cfg, nxt(), 2 * DAY, kind,
                                           arng, n_attacks=8, n_natural=1)
            p, s = harvest(z, lab, ev, rng)
            for c in range(5):
                pools[c] += p[c]; subs[c] += s[c]

    X, y, sub_all = [], [], []
    names = ["normal", "natural", "fdia", "replay", "rfdi"]
    for c in range(5):
        tgt = plan["targets"][c]
        pool = pools[c]
        if len(pool) < tgt:
            print(f"  [warn] {names[c]}: pool {len(pool)} < target {tgt}")
            idx = np.arange(len(pool))
        else:
            idx = rng.choice(len(pool), size=tgt, replace=False)
        for i in idx:
            X.append(pool[i]); y.append(c); sub_all.append(subs[c][i])
    return (np.stack(X).astype(np.float16), np.array(y, dtype=np.int8),
            np.array(sub_all, dtype=np.int8))


def bdd_alarm_rates(grid, X, y, n_per_class=150):
    out = {}
    for c in range(5):
        idx = np.where(y == c)[0][:n_per_class]
        if len(idx) == 0:
            out[c] = None
            continue
        alarms = sum(grid.bdd(X[i][-1].astype(np.float64)) for i in idx)
        out[c] = round(alarms / len(idx), 3)
    return out


def main(L_arg=None):
    t0 = time.time()
    global L
    if L_arg:
        L = int(L_arg)
    grid = Grid39()
    suffix = "" if L == 64 else f"_L{L}"
    meta = {"L": L, "m": grid.m, "n_states": grid.n, "sigma": grid.sigma,
            "chi2_th": float(grid.chi2_th), "Ts_s": 5, "splits": {}}
    for split in ["train", "val", "test"]:
        t1 = time.time()
        X, y, sub = build_split(split, grid)
        u, cnt = np.unique(y, return_counts=True)
        print(f"[{split}{suffix}] X={X.shape} classes={dict(zip(u.tolist(), cnt.tolist()))} "
              f"({time.time()-t1:.0f}s)")
        np.savez_compressed(os.path.join(RESULTS, f"sim_{split}{suffix}.npz"),
                            X=X, y=y, subtype=sub)
        meta["splits"][split] = {"n": int(len(y)),
                                 "counts": {int(k): int(v)
                                            for k, v in zip(u, cnt)}}
    if L == 64:
        Xtr = np.load(os.path.join(RESULTS, "sim_train.npz"))["X"]
        ytr = np.load(os.path.join(RESULTS, "sim_train.npz"))["y"]
        norm = Xtr[ytr == 0].astype(np.float32)
        meta["chan_mean"] = norm.reshape(-1, norm.shape[-1]).mean(0).round(5).tolist()
        meta["chan_std"] = norm.reshape(-1, norm.shape[-1]).std(0).round(5).tolist()
        Xte = np.load(os.path.join(RESULTS, "sim_test.npz"))
        meta["bdd_alarm_rates_test"] = bdd_alarm_rates(
            grid, Xte["X"].astype(np.float32), Xte["y"])
        with open(os.path.join(RESULTS, "sim_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print("BDD alarm rates (test):", meta["bdd_alarm_rates_test"])
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1] if len(_sys.argv) > 1 else None)
