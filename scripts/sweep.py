"""
Final experiment sweep, SINGLE GPU worker (desktop shares the GPU), idempotent.

Reference window length L* = 128. All main/augmentation/seed/ablation
experiments run at L*; the length study sweeps L in {16,32,64,128,256}.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PY = [sys.executable, "-u"]
LREF = "128"


def done(name: str) -> bool:
    return os.path.exists(os.path.join(RESULTS, name))


def jobs():
    det = os.path.join("scripts", "train_detector.py")
    gen = os.path.join("scripts", "train_gen.py")
    J = []

    # generators (all done; kept for idempotency)
    for g in ["diffusion", "wgan"]:
        for frac in ["0.1", "0.25", "1.0"]:
            J.append(PY + [gen, "--gen", g, "--atk_frac", frac])

    # mamba s0 with fidelity-fixed block (backbone table consistency)
    J.append(PY + [det, "--model", "mamba", "--aug", "none", "--L", LREF])

    # ablations on the recommended backbone (GRU)
    for flag in ["--no_phys", "--no_diff"]:
        J.append(PY + [det, "--model", "gru", "--aug", "none", "--L", LREF,
                       flag])

    # "none" controls at reduced attack fractions (aug table)
    for model in ["gru", "gru", "mamba"]:
        pass
    for frac in ["0.25", "0.1"]:
        J.append(PY + [det, "--model", "gru", "--aug", "none",
                       "--atk_frac", frac, "--L", LREF])
    J.append(PY + [det, "--model", "mamba", "--aug", "none",
                   "--atk_frac", "0.1", "--L", LREF])

    # augmentation on the recommended backbone (GRU) + one mamba cell
    for aug in ["diffusion", "wgan"]:
        for frac in ["0.1", "0.25", "1.0"]:
            J.append(PY + [det, "--model", "gru", "--aug", aug,
                           "--atk_frac", frac, "--L", LREF])
    J.append(PY + [det, "--model", "mamba", "--aug", "diffusion",
                   "--atk_frac", "0.1", "--L", LREF])

    # window-length study
    for L in ["16", "32", "64"]:
        for model in ["gru", "mamba"]:
            J.append(PY + [det, "--model", model, "--aug", "none", "--L", L])
    for model in ["transformer"]:
        for L in ["64", "256"]:
            J.append(PY + [det, "--model", model, "--aug", "none", "--L", L])

    # attack-axis evaluations (strength sweep + detection delay)
    ax = os.path.join("scripts", "eval_attack_axis.py")
    J.append(PY + [ax, "--ckpt", "gru_none_s0_L128"])
    J.append(PY + [ax, "--ckpt", "mamba_none_s0_L128"])
    return J


def tag_of(cmd):
    for i, c in enumerate(cmd):
        if "train_detector.py" in c:
            a = cmd[i + 1:]
            break
    else:
        return None
    model = a[a.index("--model") + 1]
    aug = a[a.index("--aug") + 1]
    frac = a[a.index("--atk_frac") + 1] if "--atk_frac" in a else "1.0"
    seed = a[a.index("--seed") + 1] if "--seed" in a else "0"
    L = a[a.index("--L") + 1] if "--L" in a else "128"
    tag = f"{model}_{aug}{'' if frac == '1.0' else frac}_s{seed}_L{L}"
    if "--no_phys" in a:
        tag += "_nophys"
    if "--no_diff" in a:
        tag += "_nodiff"
    return tag


def worker(queue, lock, results):
    while True:
        with lock:
            if not queue:
                return
            cmd = queue.pop(0)
        if any("train_gen.py" in c for c in cmd):
            frac = cmd[cmd.index("--atk_frac") + 1]
            g = cmd[cmd.index("--gen") + 1]
            flag = f"_gen_{g}_{frac}_L{LREF}.flag"
            if done(flag):
                print("skip", flag, flush=True)
                continue
            print("==>", " ".join(cmd), flush=True)
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            if rc == 0:
                open(os.path.join(RESULTS, flag), "w").write("1")
            results.append((flag, rc))
            continue
        if any("eval_attack_axis.py" in c for c in cmd):
            ck = cmd[cmd.index("--ckpt") + 1]
            out = f"attack_axis_{ck}.json"
            if done(out):
                print("skip", out, flush=True)
                continue
            print("==>", out, flush=True)
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            results.append((out, rc))
            continue
        tag = tag_of(cmd)
        if done(f"det_{tag}.json"):
            print("skip", tag, flush=True)
            continue
        print("==>", tag, flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        results.append((tag, rc))


def main():
    J = jobs()
    queue = list(J)
    lock = threading.Lock()
    results = []
    n_par = 1
    threads = [threading.Thread(target=worker, args=(queue, lock, results))
               for _ in range(n_par)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fails = [r for r in results if r[1] != 0]
    print(f"\nsweep done: {len(results)} runs, {len(fails)} failures")
    for f in fails:
        print("  FAIL:", f[0])


if __name__ == "__main__":
    main()
