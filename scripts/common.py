"""Shared utilities for training / evaluation scripts."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(ROOT, "ckpt")
os.makedirs(CKPT_DIR, exist_ok=True)

ATTACK_CLASSES = (2, 3, 4)
CLASS_NAMES = ["normal", "natural", "fdia", "replay", "rfdi"]


def load_meta():
    with open(os.path.join(RESULTS, "sim_meta.json")) as fh:
        return json.load(fh)


def load_split(split: str, L: int | None = None, suffix: str = ""):
    """Returns standardized float32 tensors X (N,L,m), y (N,)."""
    meta0 = json.load(open(os.path.join(
        RESULTS, f"sim_meta{suffix}.json")))
    base_L = meta0["L"]
    L = L or base_L
    if L <= base_L:
        fname = f"sim_{split}{suffix}.npz"
    else:
        fname = f"sim_{split}_L{L}.npz" if not suffix else             f"sim_{split}{suffix}.npz"
    d = np.load(os.path.join(RESULTS, fname))
    X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
    mu = np.array(meta0["chan_mean"], dtype=np.float32)
    sd = np.array(meta0["chan_std"], dtype=np.float32)
    X = (X - mu) / sd
    if L < X.shape[1]:
        X = X[:, -L:, :]          # keep the most recent steps
    return X, y


def class_weights(y: np.ndarray, n_classes: int = 5, mode: str = "inv") -> torch.Tensor:
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    if mode == "sqrt":
        w = np.sqrt(cnt.sum() / (n_classes * np.maximum(cnt, 1)))
    else:                                   # inverse frequency
        w = cnt.sum() / (n_classes * np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def physics_projector():
    """Dense (m, m) WLS residual projector M = I - H G^-1 H' W and sigma."""
    sys.path.insert(0, os.path.join(ROOT, "sim"))
    from grid39 import Grid39
    g = Grid39()
    M = (np.eye(g.m) - (g.H @ g.K).toarray())
    return M.astype(np.float32), g.sigma, np.array(g.chi2_th)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epochs_since(t0):
    return f"[{time.time()-t0:6.0f}s]"


def macro_f1(y_true, y_pred, n=5):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", labels=list(range(n)))


def detection_metrics(y_true, logits):
    """Binary attack-detection metrics (attacks = classes 2,3,4)."""
    from sklearn.metrics import f1_score, roc_auc_score
    p = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    score = p[:, 2:5].sum(1)
    yt = np.isin(y_true, ATTACK_CLASSES).astype(int)
    ypred = np.isin(np.argmax(p, 1), ATTACK_CLASSES).astype(int)
    return {"det_f1": f1_score(yt, ypred),
            "det_auc": roc_auc_score(yt, score)}
