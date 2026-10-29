"""Generate LaTeX tables from results/*.json into paper/tables/.

Revision history:
  - baselines.tex  : Table 2  (BDD alarm rates + shallow unsupervised)
  - main_results   : Table 3  (deep detectors; GRU reference = mean+-std
                     over 3 seeds everywhere)
  - aug/length/ablation tables use the same 3-seed GRU reference numbers
  - subtype.tex    : natural-disturbance subtype recall (appendix)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
LREF = 128

NAMES = ["正常 (Normal)", "自然 (Natural)", "FDIA", "重放 (Replay)", "随机虚假 (RFDI)"]
KEYS = ["macro_f1", "det_f1", "det_auc", "rec_1", "rec_2", "rec_3", "rec_4"]


def load(tag):
    p = os.path.join(RESULTS, f"det_{tag}.json")
    return json.load(open(p))["metrics"] if os.path.exists(p) else None


def seed_stats(model, seeds, suffix=""):
    vals = [load(f"{model}_none_s{s}{suffix}") for s in seeds]
    vals = [v for v in vals if v]
    if not vals:
        return None, None, 0
    mean = {k: float(np.mean([v[k] for v in vals])) for k in KEYS}
    std = {k: float(np.std([v[k] for v in vals])) for k in KEYS}
    return mean, std, len(vals)


def fmt(v, best=False):
    if v is None:
        return "--"
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if best else s


def fmt_pm(mean, std, best=False):
    if mean is None:
        return "--"
    s = f"{mean:.3f}$\\pm${std:.3f}"
    return f"\\textbf{{{s}}}" if best else s


# ------------------------------------------------------------- #
def baselines_table():
    cl = json.load(open(os.path.join(RESULTS, "classical.json")))
    meta = json.load(open(os.path.join(RESULTS, "sim_meta.json")))
    bdd = meta["bdd_alarm_rates_test"]
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "检测器 & 正常类告警率 & 自然类告警率 & FDIA告警率 & "
             "重放告警率 & RFDI告警率 & 检测F1 \\\\", "\\midrule"]
    lines.append(
        f"BDD($\\chi^2$检验) & {fmt(bdd.get('0'))} & {fmt(bdd.get('1'))} & "
        f"{fmt(bdd.get('2'))} & {fmt(bdd.get('3'))} & {fmt(bdd.get('4'))} "
        "& -- \\\\")
    # Huber-IRLS robust WLS + LNR (external baseline)
    p = os.path.join(RESULTS, "external_baselines.json")
    if os.path.exists(p):
        hl = json.load(open(p)).get("huber_lnr", {}).get("alarm_rates", {})
        if hl:
            lines.append(
                f"Huber鲁棒WLS+LNR & {fmt(hl.get('0'))} & {fmt(hl.get('1'))}"
                f" & {fmt(hl.get('2'))} & {fmt(hl.get('3'))} & "
                f"{fmt(hl.get('4'))} & -- \\\\")
    for name, key in [("孤立森林", "IF"), ("单类SVM", "OCSVM"),
                      ("PCA重构", "PCA-AE")]:
        m = cl[key]
        lines.append(f"{name} & -- & -- & -- & -- & -- & "
                     f"{fmt(m['det_f1'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "baselines.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("baselines.tex written")


# ------------------------------------------------------------- #
def main_table():
    grum, grus, grun = seed_stats("gru", (0, 1, 2), suffix=f"_L{LREF}")
    deep = []
    deep.append(("GRU(推荐参考配置)", grum, grus, grun))
    for model, seeds, disp in [
            ("lstm", (0, 1, 2), "LSTM"),
            ("cnn", (0, 1, 2), "1-D CNN"),
            ("transformer", (0, 1, 2), "Transformer"),
            ("mamba", (0, 1, 2), "选择性SSM(Mamba)"),
            ("grumamba", (0, 1, 2), "GRU+SSM混合")]:
        m, s, n = seed_stats(model, seeds, suffix=f"_L{LREF}")
        deep.append((disp, m, s, n))
    # external published baselines (He et al. 2017 style)
    eb_p = os.path.join(RESULTS, "external_baselines.json")
    if os.path.exists(eb_p):
        eb = json.load(open(eb_p))
        if "he_mlp_snapshot" in eb:
            deep.append(("He式MLP(快照设定)", eb["he_mlp_snapshot"], None, 1))
        if "he_mlp_window" in eb:
            deep.append(("He式MLP(窗口设定)", eb["he_mlp_window"], None, 1))
    best = {c: max((d[1][c] for d in deep if d[1]), default=0) for c in KEYS}

    lines = ["\\begin{tabular}{lccccccc}", "\\toprule",
             "检测器 & Macro-F1 & Det-F1 & Det-AUC & R$_{nat}$ & "
             "R$_{fdia}$ & R$_{rep}$ & R$_{rfdi}$ \\\\", "\\midrule"]
    for disp, m, s, n in deep:
        if m is None:
            lines.append(f"{disp} & -- & -- & -- & -- & -- & -- & -- \\\\")
            continue
        cells = []
        for c in KEYS:
            if s is not None and n >= 3:
                cells.append(fmt_pm(m[c], s[c]))
            else:
                cells.append(fmt(m[c]))
        lines.append(f"{disp} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "main_results.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("main_results.tex written")


# ------------------------------------------------------------- #
def aug_table():
    grum, grus, _ = seed_stats("gru", (0, 1, 2), suffix=f"_L{LREF}")
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             "骨干 & 真实攻击窗口数 & 无增强 & WGAN-GP & CDDM(扩散) \\\\",
             "\\midrule"]
    for f in ["1.0", "0.25", "0.1"]:
        n_atk = {"1.0": 600, "0.25": 150, "0.1": 60}[f]
        cells = []
        for aug in ["none", "wgan", "diffusion"]:
            fracpart = "" if f == "1.0" else f
            if aug == "none" and f == "1.0" and grum:
                cells.append(fmt_pm(grum["macro_f1"], grus["macro_f1"]))
                continue
            m = load(f"gru_{aug}{fracpart}_s0_L{LREF}")
            cells.append(fmt(m["macro_f1"] if m else None))
        lines.append(f"GRU & {n_atk} & " + " & ".join(cells) + " \\\\")
    m = load(f"mamba_diffusion0.1_s0_L{LREF}")
    mn = load("mamba_none0.1_s0_L128")
    lines.append(f"SSM(仅扩散) & 60 & {fmt(mn['macro_f1'] if mn else None)} & -- & "
                 f"{fmt(m['macro_f1'] if m else None)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "aug_results.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("aug_results.tex written")


# ------------------------------------------------------------- #
def length_table():
    grum, grus, _ = seed_stats("gru", (0, 1, 2), suffix=f"_L{LREF}")
    Ls = ["16", "32", "64", "128", "256"]
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "骨干 & $L{=}16$ & $L{=}32$ & $L{=}64$ & $L{=}128$ & "
             "$L{=}256$ \\\\", "\\midrule"]
    for enc, disp, ok in [("gru", "GRU macro-F1", Ls[:4]),
                          ("gru", "GRU R$_{fdia}$", Ls[:4]),
                          ("mamba", "SSM macro-F1", Ls[:4]),
                          ("mamba", "SSM R$_{fdia}$", Ls[:4]),
                          ("transformer", "Transformer macro-F1",
                           ["64", "128", "256"])]:
        key = "rec_2" if "R$_" in disp else "macro_f1"
        cells = []
        for Ltxt in Ls:
            if Ltxt not in ok:
                cells.append("--")
                continue
            if enc == "gru" and Ltxt == "128":
                cells.append(fmt_pm(grum[key], grus[key]))
                continue
            m = load(f"{enc}_none_s0_L{Ltxt}")
            cells.append(fmt(m[key] if m else None))
        lines.append(f"{disp} & " + " & ".join(cells) + " \\\\")
        if disp in ("GRU R$_{fdia}$", "SSM R$_{fdia}$"):
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "length_results.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("length_results.tex written")


# ------------------------------------------------------------- #
def ablation_table():
    fullm, fulls, _ = seed_stats("gru", (0, 1, 2), suffix=f"_L{LREF}")
    rows = [("完整\\method{}(GRU)", fullm, fulls)]
    for suffix, disp in [("_nophys", "去除物理一致性损失"),
                         ("_nodiff", "去除时序差分输入层")]:
        m, s, n = seed_stats("gru", (0, 1, 2), suffix=f"_L{LREF}{suffix}")
        rows.append((disp, m, s))
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "变体 & Macro-F1 & Det-F1 & R$_{fdia}$ & FDIA精确率 & "
             "R$_{rep}$ & R$_{rfdi}$ \\\\", "\\midrule"]
    # FDIA precision needs logits; read from stored jsons where present
    import json as _json
    _pj = os.path.join(RESULTS, "ablation_precision.json")
    _pv = _json.load(open(_pj)) if os.path.exists(_pj) else {}
    prec = {"full": str(_pv.get("full", {}).get("fdia_precision", "--")),
            "_nophys": "--",
            "_nodiff": str(_pv.get("nodiff", {}).get("fdia_precision", "--"))}
    pkeys = ["full", "_nophys", "_nodiff"]
    for (disp, m, s), pk in zip(rows, pkeys):
        if m is None:
            lines.append(f"{disp} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        cells = [fmt_pm(m['macro_f1'], s['macro_f1']),
                 fmt_pm(m['det_f1'], s['det_f1']),
                 fmt_pm(m['rec_2'], s['rec_2']), prec[pk],
                 fmt_pm(m['rec_3'], s['rec_3']),
                 fmt_pm(m['rec_4'], s['rec_4'])]
        lines.append(f"{disp} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "ablation.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("ablation.tex written")


def ablation_precision():
    """FDIA precision for full vs nodiff (from logits), for the text."""
    from sklearn.metrics import confusion_matrix
    out = {}
    for tag in ["gru_none_s0_L128", "gru_none_s0_L128_nodiff"]:
        p = os.path.join(RESULTS, f"logits_{tag}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        cm = confusion_matrix(d["y"], d["logits"].argmax(1))
        out[tag] = {"fdia_precision": round(cm[2, 2] / cm[:, 2].sum(), 3),
                    "false_fdia": int(cm[0, 2] + cm[1, 2])}
    print("ablation precision:", out)
    return out


# ------------------------------------------------------------- #
def subtype_table():
    """Natural-disturbance subtype recall: 3-seed mean+-std per encoder.

    Source: results/subtype_metrics.json (written by backfill_subtype.py
    from the saved logits of each run; hit-set intersection stats included).
    """
    src = json.load(open(os.path.join(RESULTS, "subtype_metrics.json")))
    sm = src["summary"]
    B1, B2 = chr(92), chr(92) * 2
    lines = [B1 + "begin{tabular}{lccc}", B1 + "toprule",
             "检测器 & R$_{load" + B1 + "_step}$ & R$_{gen" + B1 + "_trip}$ & "
             "R$_{line" + B1 + "_outage}$ " + B2, B1 + "midrule"]
    for enc, disp in [("gru", "GRU"), ("lstm", "LSTM"), ("cnn", "1-D CNN"),
                      ("transformer", "Transformer"),
                      ("mamba", "选择性SSM"), ("grumamba", "GRU+SSM")]:
        cells = []
        for kind in ["load_step", "gen_trip", "line_outage"]:
            v = sm.get(enc, {}).get(kind)
            cells.append(fmt_pm(v["recall_mean"], v["recall_std"]) if v else "--")
        lines.append(f"{disp} & " + " & ".join(cells) + " " + B2)
    lines += [B1 + "bottomrule", B1 + "end{tabular}"]
    with open(os.path.join(OUT, "subtype.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("subtype.tex written (3-seed mean+-std)")


# ------------------------------------------------------------- #
def dataset_table():
    meta = json.load(open(os.path.join(RESULTS, "sim_meta.json")))
    cnt = meta["splits"]["test"]["counts"]
    tr = meta["splits"]["train"]["counts"]
    va = meta["splits"]["val"]["counts"]
    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "类别 & 训练 & 验证 & 测试 \\\\", "\\midrule"]
    for c, name in enumerate(NAMES):
        lines.append(f"{name} & {tr.get(str(c), 0)} & {va.get(str(c), 0)} & "
                     f"{cnt.get(str(c), 0)} \\\\")
    lines += ["\\midrule",
              f"合计 & {sum(tr.values())} & {sum(va.values())} & "
              f"{sum(cnt.values())} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "dataset.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("dataset.tex written")


def delay_table():
    p = os.path.join(RESULTS, "attack_axis_gru_none_s0_L128.json")
    if not os.path.exists(p):
        print("delay table skipped")
        return
    d = json.load(open(p))["delay_steps"]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "攻击 & 均值 (s) & 中位数 (s) & Q1--Q3 (s) & 检出率 \\\\",
             "\\midrule"]
    for k, disp in [("fdia", "FDIA"), ("replay", "重放"), ("rfdi", "RFDI")]:
        item = d[k]
        mean, med, n = item[0], item[1], item[2]
        iqr = item[3] if len(item) > 3 else (None, None)
        if mean is None:
            lines.append(f"{disp} & -- & -- & -- & {n}/8 \\\\")
        else:
            iqr_s = (f"{iqr[0] * 5:.0f}--{iqr[1] * 5:.0f}"
                     if iqr[0] is not None else "--")
            lines.append(f"{disp} & {mean * 5:.0f} & {med * 5:.0f} & "
                         f"{iqr_s} & {n}/8 \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "delay.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("delay.tex written")


if __name__ == "__main__":
    baselines_table()
    main_table()
    aug_table()
    length_table()
    ablation_table()
    ablation_precision()
    subtype_table()
    dataset_table()
    delay_table()
