# -*- coding: utf-8 -*-
"""Generate the English tables used in the manuscript (Tables 1-6)
from results/*.json into tables_en/. No hand-written numbers.

B1 = single backslash (command prefix, e.g. begin)
B2 = double backslash (LaTeX row terminator)
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
B1 = chr(92)
B2 = chr(92) * 2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "tables_en")
os.makedirs(OUT, exist_ok=True)
KEYS = ["macro_f1", "det_f1", "det_auc", "rec_1", "rec_2", "rec_3", "rec_4"]


def load(name):
    p = os.path.join(RESULTS, name + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["metrics"]


def seeds(model, ss=(0, 1, 2), suffix=""):
    vals = [load(f"det_{model}_none_s{s}_L128{suffix}") for s in ss]
    vals = [v for v in vals if v]
    if not vals:
        return None, None
    mean = {k: float(np.mean([v[k] for v in vals])) for k in KEYS}
    std = {k: float(np.std([v[k] for v in vals])) for k in KEYS}
    return mean, std


def pm(m, s):
    return f"{m:.3f}$" + B1 + f"pm${s:.3f}" if s is not None else f"{m:.3f}"


def head(cols):
    return [B1 + "begin{tabular}{" + cols + "}", B1 + "toprule"]


def row(cells):
    return " & ".join(cells) + " " + B2


def write(name, lines):
    open(os.path.join(OUT, name), "w", encoding="utf-8",
         newline="\n").write("\n".join(lines) + "\n")


def t1():
    meta = json.load(open(os.path.join(RESULTS, "sim_meta.json")))
    eb = json.load(open(os.path.join(RESULTS, "external_baselines.json")))
    cl = json.load(open(os.path.join(RESULTS, "classical.json")))
    bdd = meta["bdd_alarm_rates_test"]
    h = eb["huber_lnr"]["alarm_rates"]
    rows = [
        ("BDD ($" + B1 + "chi^2$)", [bdd["0"], bdd["1"], bdd["2"],
                                      bdd["3"], bdd["4"]], None),
        ("Huber WLS+LNR", [h["0"], h["1"], h["2"], h["3"], h["4"]], None),
        ("Isolation Forest", None, cl["IF"]["det_f1"]),
        ("OCSVM", None, cl["OCSVM"]["det_f1"]),
        ("PCA-AE", None, cl["PCA-AE"]["det_f1"]),
    ]
    lines = head("lcccccc")
    lines.append(row(["Detector", "FAR$_{norm}$", "FAR$_{nat}$",
                      "FAR$_{fdia}$", "FAR$_{rep}$", "FAR$_{rfdi}$",
                      "Det-F1"]))
    lines.append(B1 + "midrule")
    for name, alarms, det in rows:
        cells = ([f"{a:.3f}" for a in alarms] + ["--"] if alarms
                 else ["--"] * 5 + [f"{det:.3f}"])
        lines.append(row([name] + cells))
    lines += [B1 + "bottomrule", B1 + "end{tabular}"]
    write("baselines.tex", lines)
    print("T1 baselines.tex")


def t2():
    disp = [("GRU (reference)", "gru"), ("LSTM", "lstm"), ("1-D CNN", "cnn"),
            ("Transformer", "transformer"), ("SSM (Mamba)", "mamba"),
            ("GRU+SSM hybrid", "grumamba")]
    lines = head("lccccccc")
    lines.append(row(["Detector", "Macro-F1", "Det-F1", "Det-AUC",
                      "R$_{nat}$", "R$_{fdia}$", "R$_{rep}$", "R$_{rfdi}$"]))
    lines.append(B1 + "midrule")
    for name, m in disp:
        mean, std = seeds(m)
        lines.append(row([name] + [pm(mean[k], std[k]) for k in KEYS]))
    eb = json.load(open(os.path.join(RESULTS, "external_baselines.json")))
    for name, key in [("He-style MLP (snapshot)", "he_mlp_snapshot"),
                      ("He-style MLP (window)", "he_mlp_window")]:
        v = eb[key]
        lines.append(row([name] + [f"{v[k]:.3f}" for k in KEYS]))
    lines += [B1 + "bottomrule", B1 + "end{tabular}"]
    write("main_results.tex", lines)
    print("T2 main_results.tex")


def t3():
    rows = [("GRU", "600", None, load("det_gru_wgan_s0_L128"),
             load("det_gru_diffusion_s0_L128")),
            ("GRU", "150", load("det_gru_none0.25_s0_L128"),
             load("det_gru_wgan0.25_s0_L128"),
             load("det_gru_diffusion0.25_s0_L128")),
            ("GRU", "60", load("det_gru_none0.1_s0_L128"),
             load("det_gru_wgan0.1_s0_L128"),
             load("det_gru_diffusion0.1_s0_L128")),
            ("SSM (diff." + B1 + " only)", "60", load("det_mamba_none0.1_s0_L128"),
             None, load("det_mamba_diffusion0.1_s0_L128"))]
    gm, gs = seeds("gru")
    lines = head("llccc")
    lines.append(row(["Backbone", "Real windows", "None", "WGAN-GP",
                      "CDDM"]))
    lines.append(B1 + "midrule")
    for name, w, none_v, wg, dd in rows:
        if w == "600":
            n = pm(gm["macro_f1"], gs["macro_f1"])
        else:
            n = f"{none_v['macro_f1']:.3f}" if none_v else "--"
        wv = f"{wg['macro_f1']:.3f}" if wg else "--"
        dv = f"{dd['macro_f1']:.3f}" if dd else "--"
        lines.append(row([name, w, n, wv, dv]))
    lines += [B1 + "bottomrule", B1 + "end{tabular}"]
    write("aug_results.tex", lines)
    print("T3 aug_results.tex")


def t4():
    def one(tag):
        v = load(tag)
        return f"{v['macro_f1']:.3f}" if v else "--"

    def rec(tag):
        v = load(tag)
        return f"{v['rec_2']:.3f}" if v else "--"

    gm, gs = seeds("gru")
    sm = load("det_mamba_none_s0_L128")
    rows = [
        ("GRU macro-F1", [one("det_gru_none_s0_L16"),
                          one("det_gru_none_s0_L32"),
                          one("det_gru_none_s0_L64"),
                          pm(gm["macro_f1"], gs["macro_f1"]), "--"], False),
        ("GRU R$_{fdia}$", [rec("det_gru_none_s0_L16"),
                            rec("det_gru_none_s0_L32"),
                            rec("det_gru_none_s0_L64"),
                            pm(gm["rec_2"], gs["rec_2"]), "--"], True),
        ("SSM macro-F1", [one("det_mamba_none_s0_L16"),
                          one("det_mamba_none_s0_L32"),
                          one("det_mamba_none_s0_L64"),
                          f"{sm['macro_f1']:.3f}", "--"], False),
        ("SSM R$_{fdia}$", [rec("det_mamba_none_s0_L16"),
                            rec("det_mamba_none_s0_L32"),
                            rec("det_mamba_none_s0_L64"),
                            f"{sm['rec_2']:.3f}", "--"], True),
        ("Transformer macro-F1", ["--", "--",
                                  one("det_transformer_none_s0_L64"),
                                  one("det_transformer_none_s0_L128"),
                                  one("det_transformer_none_s0_L256")], False),
    ]
    lines = head("lccccc")
    lines.append(row(["Backbone", "$L{=}16$", "$L{=}32$", "$L{=}64$",
                      "$L{=}128$", "$L{=}256$"]))
    lines.append(B1 + "midrule")
    for name, cells, mid_after in rows:
        lines.append(row([name] + cells))
        if mid_after:
            lines.append(B1 + "midrule")
    lines += [B1 + "bottomrule", B1 + "end{tabular}"]
    write("length_results.tex", lines)
    print("T4 length_results.tex")


def t5():
    d = json.load(open(os.path.join(
        RESULTS, "attack_axis_gru_none_s0_L128.json")))["delay_steps"]
    lines = head("lcccc")
    lines.append(row(["Attack", "Mean (s)", "Median (s)", "Q1--Q3 (s)",
                      "Detected"]))
    lines.append(B1 + "midrule")
    for k, tot in (("fdia", 8), ("replay", 8), ("rfdi", 8)):
        mean, med, ndet, (q1, q3) = d[k]
        lines.append(row([k.upper(), f"{mean * 5:.0f}", f"{med * 5:.0f}",
                          f"{q1 * 5:.0f}--{q3 * 5:.0f}", f"{ndet}/{tot}"]))
    lines += [B1 + "bottomrule", B1 + "end{tabular}"]
    write("delay.tex", lines)
    print("T5 delay.tex")


def t6():
    prec = json.load(open(os.path.join(RESULTS, "ablation_precision.json")))
    rows = [("Full StealthGuard (GRU)", seeds("gru"),
             prec.get("full", {}).get("fdia_precision")),
            ("w/o physics loss", seeds("gru", suffix="_nophys"), None),
            ("w/o diff." + B1 + " stem", seeds("gru", suffix="_nodiff"),
             prec.get("nodiff", {}).get("fdia_precision"))]
    lines = head("lcccccc")
    lines.append(row(["Variant", "Macro-F1", "Det-F1", "R$_{fdia}$",
                      "FDIA prec.", "R$_{rep}$", "R$_{rfdi}$"]))
    lines.append(B1 + "midrule")
    for name, (m, s), p in rows:
        if m is None:
            continue
        cells = [pm(m[k], s[k]) for k in ["macro_f1", "det_f1", "rec_2"]]
        cells.append(f"{p:.3f}" if p else "--")
        cells += [pm(m[k], s[k]) for k in ["rec_3", "rec_4"]]
        lines.append(row([name] + cells))
    lines += [B1 + "bottomrule", B1 + "end{tabular}"]
    write("ablation.tex", lines)
    print("T6 ablation.tex")


if __name__ == "__main__":
    t1(); t2(); t3(); t4(); t5(); t6()
    print("tables_en/ regenerated from results/*.json")
