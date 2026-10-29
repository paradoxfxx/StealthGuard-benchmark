"""
Classical and reconstruction-based detectors on the test split:
  - BDD (chi-square residual test of WLS state estimation)
  - Isolation Forest, OCSVM (on flattened windows / feature summary)
  - PCA reconstruction error
Results feed the main comparison table alongside the deep detectors.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import OneClassSVM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import ATTACK_CLASSES, RESULTS, load_split  # noqa: E402


def binary_metrics(y, score):
    yt = np.isin(y, ATTACK_CLASSES).astype(int)
    pred = (score > np.quantile(score, 1 - yt.mean())).astype(int)
    return {"det_f1": f1_score(yt, pred), "det_auc": roc_auc_score(yt, score)}


def summary_features(X):
    """Cheap per-window features for shallow detectors."""
    return np.concatenate(
        [X.mean(1), X.std(1), np.abs(np.diff(X, axis=1)).mean(1)], axis=1)


def main():
    Xtr, ytr = load_split("train")
    Xte, yte = load_split("test")
    Ftr = summary_features(Xtr)
    Fte = summary_features(Xte)
    tr_norm = Ftr[ytr == 0]
    out = {}

    # ---- Isolation Forest -------------------------------------------
    for name, model in [
        ("if", IsolationForest(n_estimators=300, contamination=0.15,
                               random_state=0)),
        ("ocsvm", OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")),
        ("pca", None),
    ]:
        if name == "pca":
            p = PCA(n_components=0.95, random_state=0).fit(tr_norm)
            err = ((Fte - p.inverse_transform(p.transform(Fte))) ** 2).mean(1)
            # threshold calibrated for 5% false alarms on train normal
            th = np.quantile(((Ftr[ytr == 0]
                               - p.inverse_transform(p.transform(Ftr[ytr == 0])))
                              ** 2).mean(1), 0.95)
            score = err
            m = binary_metrics(yte, score)
            # classification-style metrics: none (binary only)
            out["PCA-AE"] = {**m, "det_pred_f1": f1_score(
                np.isin(yte, ATTACK_CLASSES).astype(int),
                (err > th).astype(int))}
        else:
            model.fit(tr_norm)
            score = -model.score_samples(Fte)   # higher = more anomalous
            m = binary_metrics(yte, score)
            out[name.upper()] = {**m, "macro_f1": None}
        print(name, m)

    # ---- BDD from simulation meta ------------------------------------
    meta = json.load(open(os.path.join(RESULTS, "sim_meta.json")))
    bdd = meta["bdd_alarm_rates_test"]
    yt = np.isin(yte, ATTACK_CLASSES).astype(int)
    # BDD per-window: simulate per-window alarm by re-running on the last
    # step is stored already; approximate window-level detection with the
    # mean alarm over the window via stored rate only (report as is).
    out["BDD"] = {"det_auc": None, "det_f1": None,
                  "alarm_rate_by_class": bdd,
                  "note": "per-step chi2 test at window end, from sim_meta"}

    with open(os.path.join(RESULTS, "classical.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
