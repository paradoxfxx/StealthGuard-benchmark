# -*- coding: utf-8 -*-
"""Backfill per-subtype recall into det_*.json from saved logits (no GPU needed).

Computes, for every det_<tag>.json that has a matching logits_<tag>.npz:
  metrics["subtype"] = {kind: {n, recall, as_fdia}}
for natural-disturbance windows (y==1), subtype from results/sim_test.npz.

Also writes results/subtype_metrics.json: per-encoder seed values, mean/std
(population, ddof=0, consistent with the paper's convention), and hit-set
intersection statistics used by the cross-model claims in Section 6.2.
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
KINDS = [("load_step", 0), ("gen_trip", 1), ("line_outage", 2)]

# NOTE: sim_test.npz / sim_test_L128.npz / sim_test_L256.npz are separate
# generation draws (same class counts, different windows & subtype columns).
# A run at window length L was trained/evaluated on the file load_split()
# would pick, so the subtype column must come from the SAME file.
def subtype_for(tag):
    if tag.endswith("_static"):
        return np.load(os.path.join(RESULTS, "sim_test_static.npz"))["subtype"]
    m = re.search(r"_L(\d+)$", tag)
    L = int(m.group(1)) if m else 64
    if L > 64:
        return np.load(os.path.join(RESULTS, f"sim_test_L{L}.npz"))["subtype"]
    return np.load(os.path.join(RESULTS, "sim_test.npz"))["subtype"]


def subtype_stats(pred, y, sub):
    out = {}
    hits = {}
    for kind, code in KINDS:
        m = (y == 1) & (sub == code)
        if m.sum() == 0:
            continue
        hit = m & (pred == 1)
        out[kind] = {
            "n": int(m.sum()),
            "recall": float(hit.sum() / m.sum()),
            "as_fdia": float((pred[m] == 2).mean()),
        }
        hits[kind] = np.where(hit)[0]
    return out, hits


def main():
    # 1) backfill every det_*.json that has logits
    for jf in sorted(glob.glob(os.path.join(RESULTS, "det_*.json"))):
        tag = os.path.basename(jf)[4:-5]
        lf = os.path.join(RESULTS, f"logits_{tag}.npz")
        if not os.path.exists(lf):
            continue
        d = np.load(lf, allow_pickle=True)
        if "logits" not in d or "y" not in d:
            continue
        pred = d["logits"].argmax(1)
        y = d["y"]
        sub = subtype_for(tag)
        if len(sub) != len(y):
            print(f"skip {tag}: len mismatch")
            continue
        st, _ = subtype_stats(pred, y, sub)
        doc = json.load(open(jf, encoding="utf-8"))
        doc.setdefault("metrics", {})["subtype"] = st
        with open(jf, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        print(f"backfilled {tag}")

    # 2) aggregate the six main encoders (aug=none, L=128)
    enc_seeds = {
        "cnn": [0, 1, 2], "gru": [0, 1, 2], "lstm": [0, 1, 2],
        "transformer": [0, 1, 2], "mamba": [0, 1, 2], "grumamba": [0, 1, 2],
    }
    agg, hitsets = {}, {}
    for enc, seeds in enc_seeds.items():
        agg[enc] = {}
        hitsets[enc] = {}
        sub128 = np.load(os.path.join(RESULTS, "sim_test_L128.npz"))["subtype"]
        for s in seeds:
            lf = os.path.join(RESULTS, f"logits_{enc}_none_s{s}_L128.npz")
            if not os.path.exists(lf):
                print(f"MISSING {lf}")
                continue
            d = np.load(lf, allow_pickle=True)
            pred = d["logits"].argmax(1)
            y = d["y"]
            st, hits = subtype_stats(pred, y, sub128)
            agg[enc][s] = st
            hitsets[enc][s] = hits
    # mean/std per encoder per kind (population std, ddof=0)
    summary = {}
    for enc, per_seed in agg.items():
        summary[enc] = {}
        for kind, _ in KINDS:
            vals = [v[kind]["recall"] for v in per_seed.values() if kind in v]
            fds = [v[kind]["as_fdia"] for v in per_seed.values() if kind in v]
            if vals:
                first_seed = next(iter(per_seed))
                summary[enc][kind] = {
                    "recall_mean": float(np.mean(vals)),
                    "recall_std": float(np.std(vals)),
                    "as_fdia_mean": float(np.mean(fds)),
                    "n": per_seed[first_seed][kind]["n"],
                }
    # 3) cross-model hit-set stats (seed 0 or first available seed)
    first = {enc: min(seeds) for enc, seeds in enc_seeds.items()
             if enc in hitsets and hitsets[enc]}
    inter = {}
    kinds = [k for k, _ in KINDS]
    for kind in kinds:
        hs = {enc: set(hitsets[enc][first[enc]][kind]) for enc in first
              if kind in hitsets[enc][first[enc]]}
        if len(hs) < 2:
            continue
        encs = list(hs)
        i = set.intersection(*hs.values())
        u = set.union(*hs.values())
        inter[kind] = {
            "encoders": encs, "seed_used": {e: first[e] for e in encs},
            "n_union": len(u), "n_intersection": len(i),
            "per_encoder_nhits": {e: len(hs[e]) for e in encs},
            "all_identical": all(len(hs[e]) == len(i) and hs[e] == i
                                 for e in encs),
        }
    # GRU vs LSTM load_step误判重叠(正文断言)
    if "gru" in first and "lstm" in first:
        g = set(hitsets["gru"][first["gru"]].get("load_step", []))
        l = set(hitsets["lstm"][first["lstm"]].get("load_step", []))
        inter["gru_lstm_load_step"] = {"n_gru": len(g), "n_lstm": len(l),
                                       "n_overlap": len(g & l)}
    out = {"per_seed": agg, "summary": summary, "intersection": inter}
    with open(os.path.join(RESULTS, "subtype_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n=== summary (mean over seeds) ===")
    for enc, kinds_d in summary.items():
        row = [f"{kinds_d[k]['recall_mean']:.3f}" if k in kinds_d else "--"
               for k, _ in KINDS]
        print(f"{enc:12s} " + "  ".join(row))
    print("\n=== intersections ===")
    print(json.dumps(inter, indent=1)[:1500])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
