"""Generate all paper figures from results/ + fresh simulation runs.

Outputs PDF (for LaTeX) and PNG (for quick viewing) into ../figures/.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sim"))
sys.stdout.reconfigure(encoding="utf-8")
from scripts.figstyle import (C_FDIA, C_NATURAL, C_NORMAL, C_REPLAY,  # noqa: E402
                              C_RFDI, CLASS_COLORS, CLASS_NAMES, INK,
                              INK2, PAL, SEQ, setup, style_ax)
from scripts.common import RESULTS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)
LREF = 128


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def load(tag):
    p = os.path.join(RESULTS, f"det_{tag}.json")
    return json.load(open(p))["metrics"] if os.path.exists(p) else None


# ------------------------------------------------------------------ #
def fig_sim():
    """One simulated day: 6 measurement channels with events shaded."""
    from grid39 import Grid39
    from scenarios import ScenarioRunner, ScenarioConfig, DAY
    setup()
    g = Grid39()
    cfg = ScenarioConfig()
    runner = ScenarioRunner(g, cfg, seed=42_001)
    a = g.fdia_vector([16, 19, 21], 0.02)
    a = a * (6 * g.sigma / np.abs(a).max())
    # channels that visibly carry each event
    fdia_ch = np.argsort(np.abs(a))[-3:].tolist()          # largest |a|
    rfdi_ch = [3, 90]
    events = [
        dict(kind="load_step", t0=2600, dur=1400, t1=4000, bus=21,
             mag=0.22, onset=12),
        dict(kind="fdia", t0=7000, dur=2400, t1=9400, a_vec=a, onset=20,
             buses=[16, 19, 21]),
        dict(kind="replay", t0=12500, dur=2000, t1=14500, lag=1440),
        dict(kind="rfdi", t0=16000, dur=900, t1=16900,
             channels=np.array(rfdi_ch), onset=0,
             bias=np.array([0.10, -0.11])),
    ]
    z, lab, _ = runner.run(DAY, events)
    t = np.arange(DAY) * 5 / 3600.0          # hours
    fig = plt.figure(figsize=(7.0, 4.8))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.28])
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    leg_ax = fig.add_subplot(gs[3, :])
    leg_ax.axis("off")
    chans = [(104, "injection, bus 21 (load-step bus)"),
             (fdia_ch[0], f"branch flow ch.{fdia_ch[0]} (FDIA carrier)"),
             (fdia_ch[1], f"branch flow ch.{fdia_ch[1]} (FDIA carrier)"),
             (fdia_ch[2], f"branch flow ch.{fdia_ch[2]} (FDIA carrier)"),
             (96, "injection, bus 16 (replay kink)"),
             (rfdi_ch[0], f"channel {rfdi_ch[0]} (RFDI-affected)")]
    spans = {1: ("load step", C_NATURAL), 2: ("FDIA", C_FDIA),
             3: ("replay", C_REPLAY), 4: ("RFDI", C_RFDI)}
    for k, (ax, (ch, ttl)) in enumerate(zip(axes, chans)):
        ax.plot(t, z[:, ch], lw=0.4, color=C_NORMAL, rasterized=True)
        for c, (name, col) in spans.items():
            m = lab == c
            if m.any():
                ax.axvspan(t[m].min(), t[m].max(), color=col, alpha=0.16,
                           lw=0)
        ax.set_title(ttl, fontsize=7, loc="left", color=INK2, pad=2.5)
        style_ax(ax)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.set_ylabel("p.u.")
        ax.set_xlabel("time (h)")
    handles = [plt.Line2D([], [], color=col, alpha=0.35, lw=6)
               for _, (_, col) in spans.items()]
    labels = [n for _, (n, _) in spans.items()]
    handles.append(plt.Line2D([], [], color=C_NORMAL, lw=1))
    labels.append("measurement")
    leg_ax.legend(handles, labels, ncol=5, loc="center", frameon=False)
    # B-6: detrended FDIA inset in the first FDIA-carrier panel
    ax_ins = axes[1].inset_axes([0.52, 0.62, 0.45, 0.32])
    t0i, t1i = events[1]["t0"], events[1]["t1"]
    pre = z[t0i - 60:t0i - 10, fdia_ch[0]].mean()
    seg = slice(t0i - 120, t0i + 360)
    ax_ins.plot((np.arange(t0i - 120, t0i + 360) - t0i) * 5 / 60,
                z[seg, fdia_ch[0]] - pre, lw=0.8, color=C_FDIA)
    ax_ins.axvline(0, color="#999999", lw=0.6, ls=":")
    ax_ins.set_title("detrended (onset at 0)", fontsize=6, color=INK2)
    ax_ins.tick_params(labelsize=5.5)
    style_ax(ax_ins)
    save(fig, "fig_sim")


def fig_bdd():
    """BDD statistic during attacks: stealth FDIA passes, RFDI trips."""
    from grid39 import Grid39
    from scenarios import ScenarioRunner, ScenarioConfig, DAY
    setup()
    g = Grid39()
    runner = ScenarioRunner(g, ScenarioConfig(), seed=43_001)
    a = g.fdia_vector([4, 5, 14], 0.02)
    a = a * (6 * g.sigma / np.abs(a).max())
    bias = np.zeros(g.m)
    bias[[7, 21, 60, 95, 118]] = [0.09, -0.08, 0.11, -0.09, 0.10]
    events = [
        dict(kind="fdia", t0=6000, dur=2600, t1=8600, a_vec=a, onset=20,
             buses=[4, 5, 14]),
        dict(kind="rfdi", t0=14000, dur=2000, t1=16000, channels=np.array(
            [7, 21, 60, 95, 118]), onset=0, bias=bias[[7, 21, 60, 95, 118]]),
    ]
    z, lab, _ = runner.run(DAY, events)
    idx = np.arange(2000, 17200, 4)
    J = np.array([g.wls(z[i])[2] for i in idx])
    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    ax.plot(idx * 5 / 3600, J, lw=0.7, color=C_NORMAL)
    ax.axhline(g.chi2_th, color=C_RFDI, ls="--", lw=1.0)
    ax.text(0.4, g.chi2_th + 6, r"BDD threshold $\chi^2_{0.95}$",
            fontsize=7, color=C_RFDI)
    for (c, col, name) in [(2, C_FDIA, "FDIA"), (4, C_RFDI, "RFDI")]:
        m = (lab[idx] == c)
        if m.any():
            ax.axvspan(idx[m].min() * 5 / 3600, idx[m].max() * 5 / 3600,
                       color=col, alpha=0.18, lw=0)
    ax.set_xlabel("time (h)")
    ax.set_ylabel(r"$J = r^\top R^{-1} r$")
    style_ax(ax)
    save(fig, "fig_bdd")


def fig_arch():
    """Architecture block diagram (orthogonal flow, separated ports)."""
    setup()
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.010",
                                    fc=fc, ec=INK2, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK)

    def harrow(x1, x2, y, col=INK2):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                     mutation_scale=8, color=col, lw=0.8))

    def elbow(x1, y1, x2, y2, col=INK2):
        """orthogonal elbow arrow: horizontal then vertical then horizontal"""
        xm = (x1 + x2) / 2
        ax.plot([x1, xm], [y1, y1], color=col, lw=0.8)
        ax.plot([xm, xm], [y1, y2], color=col, lw=0.8)
        ax.add_patch(FancyArrowPatch((xm, y2), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, color=col, lw=0.8))

    HL = "#eef3fb"      # highlight fill
    box(0.005, 0.36, 0.135, 0.28, "SCADA window\n$X\\in\\mathbb{R}^{L\\times m}$\n"
        f"($L$={LREF}, $m$=131)", fc=HL)
    box(0.165, 0.36, 0.175, 0.28, "temporal-difference stem\n"
        "$[X,\\ \\Delta X,\\ X-\\bar X_{[0:L/4]}]$")
    box(0.365, 0.36, 0.145, 0.28, "linear proj.\n$d=128$\n+ depthwise conv")
    box(0.535, 0.30, 0.205, 0.40, "bidirectional selective SSM\n"
        "(2 pre-norm BiMamba layers)\nforward + backward", fc=HL)
    box(0.775, 0.62, 0.115, 0.20, "mean/max pool\n+ MLP")
    box(0.775, 0.14, 0.115, 0.20, "reconstruction\ndecoder")
    box(0.905, 0.62, 0.090, 0.20, "5-class\nlogits", fc=HL)
    box(0.905, 0.14, 0.090, 0.20, "physics loss\n"
        "$\\|R^{-1/2}M\\hat X\\|$", fc=HL)
    harrow(0.140, 0.165, 0.50)
    harrow(0.340, 0.365, 0.50)
    harrow(0.510, 0.535, 0.50)
    elbow(0.740, 0.66, 0.775, 0.72)
    elbow(0.740, 0.34, 0.775, 0.24)
    harrow(0.890, 0.905, 0.72)
    harrow(0.890, 0.905, 0.24)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "fig_arch")


def fig_aug():
    """Macro-F1 vs number of real attack windows for 3 augmentation modes."""
    setup()
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    xs = [60, 150, 600]
    for aug, col, disp in [
            ("none", CLASS_COLORS[0], "no augmentation"),
            ("wgan", C_RFDI, "WGAN-GP"),
            ("diffusion", C_FDIA, "CDDM (diffusion)")]:
        ys = []
        for n, f in zip(xs, ["0.1", "0.25", "1.0"]):
            tag = (f"gru_none_s0_L{LREF}" if f == "1.0"
                   else f"gru_none{f}_s0_L{LREF}") if aug == "none" \
                else f"gru_{aug}{f}_s0_L{LREF}"
            m = load(tag)
            ys.append(m["macro_f1"] if m else np.nan)
        ax.plot(xs, ys, "o-", color=col, lw=1.4, ms=4, label=disp)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("real attack windows per attack class")
    ax.set_ylabel("macro-F1")
    ax.legend(loc="lower right")
    style_ax(ax)
    save(fig, "fig_aug")


def fig_strength():
    """Detection rate vs FDIA strength."""
    setup()
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for ck, col, disp, mk in [
            ("gru_none_s0_L128", C_FDIA, "GRU (recommended)", "o-"),
            ("mamba_none_s0_L128", CLASS_COLORS[2], "selective SSM", "s--")]:
        p = os.path.join(RESULTS, f"attack_axis_{ck}.json")
        if os.path.exists(p):
            d = json.load(open(p))["strength_sweep"]
            ax.plot([r["level_sigma"] for r in d],
                    [r["det_rate"] for r in d], mk, color=col, lw=1.4,
                    ms=4, label=disp)
    ax.axhline(0.06, color=INK2, ls=":", lw=1.0,
               label="BDD (stealth, $\\approx$FAR)")
    ax.set_xlabel(r"attack magnitude $\|a\|_\infty\ /\ \sigma$")
    ax.set_ylabel("FDIA detection rate")
    ax.legend()
    style_ax(ax)
    save(fig, "fig_strength")


def fig_eff():
    """Efficiency scan: latency + peak memory vs window length.

    L = 32..1024 is an efficiency-only sweep; the accuracy study uses
    L = 16..256 (attack episodes are 10-30 min, so longer windows
    straddle them).
    """
    setup()
    import torch
    import time as _time
    from models.nets import TwoBranchDetector
    dev = torch.device("cuda")
    Ls = [32, 64, 128, 256, 512, 1024]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for enc, col, mk, disp in [
            ("mamba", "#1baf7a", "s", "selective SSM (Mamba)"),
            ("transformer", "#eb6834", "^", "Transformer"),
            ("gru", "#2a78d6", "o", "BiGRU")]:
        lat, mem = [], []
        for L in Ls:
            model = TwoBranchDetector(m=131, encoder_name=enc).to(dev).eval()
            x = torch.randn(64, L, 131, device=dev)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(3):
                    model(x)
                torch.cuda.synchronize()
                t0 = _time.perf_counter()
                for _ in range(10):
                    model(x)
                torch.cuda.synchronize()
                lat.append((_time.perf_counter() - t0) / 10 * 1e3)
            mem.append(torch.cuda.max_memory_allocated() / 2**30)
            del model, x
        axes[0].plot(Ls, lat, mk + "-", color=col, lw=1.4, ms=4.5,
                     label=disp)
        axes[1].plot(Ls, mem, mk + "-", color=col, lw=1.4, ms=4.5)
    for ax, ttl, yl in [
            (axes[0], "batch of 64 (RTX 5090D)", "latency (ms)"),
            (axes[1], "", "peak memory (GiB)")]:
        ax.set_xlabel("window length $L$")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ls)
        ax.set_xticklabels([str(x) for x in Ls], rotation=45,
                           ha="right", rotation_mode="anchor")
        if ttl:
            ax.set_title(ttl, fontsize=7.5)
        ax.set_ylabel(yl)
        style_ax(ax)
    axes[0].legend(fontsize=6.5)
    save(fig, "fig_eff")


def fig_conf():
    """Confusion matrix of the reference model (GRU)."""
    setup()
    from sklearn.metrics import confusion_matrix
    p = os.path.join(RESULTS, f"logits_gru_none_s0_L{LREF}.npz")
    if not os.path.exists(p):
        print("skip fig_conf (no logits)")
        return
    d = np.load(p)
    cm = confusion_matrix(d["y"], d["logits"].argmax(1))
    cm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(5), CLASS_NAMES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "fig_conf")


def fig_gen_quality():
    """Real vs synthetic FDIA windows, strongest-carrier channel,
    adaptive y-range from the real pool, bias reference line."""
    setup()
    import torch
    from models.generative import CondDDPM, CondGenerator
    from scripts.common import CKPT_DIR, device, load_split
    from scripts.train_gen import subsample_attacks
    dev = device()
    X, y = load_split("train", LREF)
    Xs, ys = subsample_attacks(X, y, 0.1, seed=0)
    real_pool = Xs[ys == 2]
    ddpm = CondDDPM(m=131, L=LREF).to(dev)
    wgan = CondGenerator(131, L=LREF).to(dev)
    pd_ = os.path.join(CKPT_DIR, "ddpm_f0.1_s0_L128.pt")
    pw = os.path.join(CKPT_DIR, "wgan_f0.1_s0_L128.pt")
    yy = torch.full((3,), 2, dtype=torch.long, device=dev)
    with torch.no_grad():
        syn_d = (ddpm.ddim_sample(3, yy, LREF, dev, steps=50).cpu().numpy()
                 if os.path.exists(pd_) else np.full((3, LREF, 131), np.nan))
        syn_w = (wgan(torch.randn(3, wgan.z_fc.in_features, device=dev),
                      torch.nn.functional.one_hot(yy, 5).float()).cpu().numpy()
                 if os.path.exists(pw) else np.full((3, LREF, 131), np.nan))
    rng = np.random.default_rng(4)
    real = real_pool[rng.choice(len(real_pool), 3, replace=False)]
    # pick the channel with the strongest mean level shift in the real pool
    Lh = LREF // 2
    shift = np.abs(real_pool[:, Lh:, :].mean(1)
                   - real_pool[:, :Lh, :].mean(1)).mean(0)
    ch = int(np.argmax(shift))
    bias = float(shift[ch])
    lo, hi = np.percentile(real_pool[:, :, ch], [1, 99])
    pad = 0.15 * (hi - lo + 1e-6)
    lo, hi = lo - pad, hi + pad
    fid = {}
    fp = os.path.join(RESULTS, "gen_fidelity.json")
    if os.path.exists(fp):
        fid = json.load(open(fp))
    fig, axes = plt.subplots(3, 3, figsize=(6.8, 4.8), sharex=True,
                             sharey=True)
    rows = [(real, "real FDIA (60-window subset)"),
            (syn_d, "CDDM (diffusion) samples"),
            (syn_w, "WGAN-GP samples")]
    for r, (W, ttl) in enumerate(rows):
        for k in range(3):
            ax = axes[r, k]
            ax.plot(W[k, :, ch], lw=0.7, color=CLASS_COLORS[r])
            ax.axhline(0, color="#bbbbbb", lw=0.6, ls=":")
            ax.set_ylim(lo, hi)
            if r == 0:
                ax.set_title(f"sample {k + 1}", fontsize=7)
            if k == 0:
                w1 = ""
                if r == 1 and "diffusion" in fid:
                    w1 = "\n$W_1$=" + \
                        f"{fid['diffusion']['mean_channel_w1']:.2f}"
                if r == 2 and "wgan" in fid:
                    w1 = "\n$W_1$=" + \
                        f"{fid['wgan']['mean_channel_w1']:.2f}"
                ax.set_ylabel(ttl + w1, fontsize=6.8)
            style_ax(ax)
            ax.set_xlabel("step (5 s)", fontsize=6.8)
    fig.suptitle(f"channel {ch} (strongest FDIA carrier in the subset; "
                 f"standardized; dotted line = pre-attack baseline; "
                 f"mean level shift = {bias:.2f}); "
                 f"real-resample $W_1$ floor "
                 f"{fid.get('diffusion', {}).get('real_bootstrap_w1_floor', 0):.2f}",
                 fontsize=7.5, y=1.03)
    save(fig, "fig_gen")


def fig_length():
    """Macro-F1 vs window length (3 colors + markers; GRU@128 = mean+-std)."""
    setup()
    import json
    Ls = [16, 32, 64, 128, 256]
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    gru_m = gru_s = None
    vals = []
    for sd in (0, 1, 2):
        m = load(f"gru_none_s{sd}_L128")
        if m:
            vals.append(m["macro_f1"])
    if vals:
        gru_m, gru_s = float(np.mean(vals)), float(np.std(vals))
    series = [
        ("gru", "BiGRU", "#2a78d6", "o", Ls[:4]),
        ("mamba", "selective SSM (Mamba)", "#1baf7a", "s", Ls[:4]),
        ("transformer", "Transformer", "#eb6834", "^", [64, 128, 256]),
    ]
    for enc, disp, col, mk, ok in series:
        xs, f1 = [], []
        for L in ok:
            m = load(f"{enc}_none_s0_L{L}")
            if m:
                xs.append(L)
                f1.append(m["macro_f1"])
        ax.plot(xs, f1, marker=mk, color=col, lw=1.4, ms=4.5, ls="-",
                label=disp)
        if enc == "gru" and gru_m is not None and 128 in xs:
            i = xs.index(128)
            ax.errorbar([128], [gru_m], yerr=[gru_s], fmt="none",
                        ecolor=col, capsize=3, lw=1.2)
            f1[i] = gru_m
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(x) for x in Ls])
    ax.set_xlabel("window length $L$ (steps of 5 s)")
    ax.set_ylabel("macro-F1")
    ax.legend()
    style_ax(ax)
    save(fig, "fig_length")


def fig_abl_conf():
    """Confusion matrices: full vs no-difference-stem."""
    setup()
    from sklearn.metrics import confusion_matrix
    names = ["Normal", "Natural", "FDIA", "Replay", "RFDI"]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.1))
    for ax, tag, ttl in [
            (axes[0], "gru_none_s0_L128", "full model"),
            (axes[1], "gru_none_s0_L128_nodiff",
             "w/o difference stem")]:
        p = os.path.join(RESULTS, f"logits_{tag}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        cm = confusion_matrix(d["y"], d["logits"].argmax(1),
                              normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5), names, rotation=30, ha="right", fontsize=6.5)
        ax.set_yticks(range(5), names, fontsize=6.5)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if cm[i, j] > 0.5 else INK)
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("predicted", fontsize=7)
        ax.grid(False)
    axes[0].set_ylabel("true", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.03)
    save(fig, "fig_abl_conf")


def fig_strength():
    """Detection rate vs FDIA strength; training band shaded."""
    setup()
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for ck, col, disp, mk in [
            ("gru_none_s0_L128", "#e34948", "BiGRU", "o-"),
            ("mamba_none_s0_L128", "#1baf7a", "selective SSM", "s--")]:
        p = os.path.join(RESULTS, f"attack_axis_{ck}.json")
        if os.path.exists(p):
            d = json.load(open(p))["strength_sweep"]
            ax.plot([r["level_sigma"] for r in d],
                    [r["det_rate"] for r in d], mk, color=col, lw=1.5,
                    ms=4.5, label=disp)
    ax.axhspan(0.0, 0.067, color="#d9d8d4", alpha=0.5, lw=0)
    # BDD never alarms on stealth FDIA (Ma=0 by construction); the band
    # is its measured false-alarm level on the normal class (6.7%).
    ax.text(1.55, 0.042, "BDD false-alarm level (normal class, 6.7%)",
            fontsize=6.5, color=INK2)
    ax.axvspan(4, 10, color="#fdf3d0", lw=0)
    ax.text(4.15, 0.55, "training range (4-10$\\sigma$)", fontsize=6.5,
            color="#8a6d1a")
    ax.set_xticks([1.5, 2, 3, 4, 5, 6, 8])
    ax.set_xlabel(r"attack magnitude $\|a\|_\infty\,/\,\sigma$",
                  fontsize=9)
    ax.set_ylabel("FDIA detection rate", fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    style_ax(ax)
    save(fig, "fig_strength")


if __name__ == "__main__":
    which = sys.argv[1:] or ["arch", "sim", "bdd", "aug", "strength", "eff",
                             "conf", "gen_quality", "length", "abl_conf"]
    fns = {"arch": fig_arch, "sim": fig_sim, "bdd": fig_bdd, "aug": fig_aug,
           "strength": fig_strength, "eff": fig_eff, "conf": fig_conf,
           "gen_quality": fig_gen_quality, "length": fig_length,
           "abl_conf": fig_abl_conf}
    for w in which:
        try:
            fns[w]()
        except Exception as e:
            print(f"[{w}] FAILED: {e}")
