"""C-1 external baselines:

(a) He et al. 2017-style deep feedforward detector (their CDBN reduced to
    a modern MLP of comparable role), evaluated in BOTH regimes:
      - snapshot regime: input is the LAST SCADA frame of the window
        (the setting of the original paper);
      - window regime: input is the flattened window (our setting).
(b) Huber-robust WLS (IRLS) + largest-normalized-residual test: the
    classical robust-SE detector matching our DC model and the Ma=0
    argument. Per-class alarm rates only (no multiclass output).

Writes results/external_baselines.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.stdout.reconfigure(encoding="utf-8")
from scripts.common import (RESULTS, class_weights, device,  # noqa: E402
                            detection_metrics, load_split, macro_f1,
                            set_seed)


class HeMLP(nn.Module):
    """Feed-forward detector in the role of He et al. 2017's CDBN."""

    def __init__(self, d_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 5))

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xva, yva, Xte, yte, tag, epochs=60):
    dev = device()
    set_seed(0)
    model = HeMLP(Xtr.shape[1]).to(dev)
    cw = class_weights(ytr).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=256, shuffle=True)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.cross_entropy(model(xb), yb, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lg = torch.cat([model(torch.tensor(Xva[s:s + 512]).to(dev))
                            for s in range(0, len(Xva), 512)]).cpu().numpy()
        f1 = macro_f1(yva, lg.argmax(1))
        if f1 > best:
            best, best_state = f1, {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lg = torch.cat([model(torch.tensor(Xte[s:s + 512]).to(dev))
                        for s in range(0, len(Xte), 512)]).cpu().numpy()
    pred = lg.argmax(1)
    out = {"macro_f1": float(macro_f1(yte, pred)),
           **{f"rec_{c}": float((pred[yte == c] == c).mean())
              for c in range(5)}}
    out.update(detection_metrics(yte, lg))
    print(tag, {k: round(v, 3) for k, v in out.items()
                if isinstance(v, float)})
    return out


def huber_wls_alarm(grid, X, y, clip=1.345, n_iter=20, per_class=300):
    """Huber IRLS-WLS + largest normalized residual test.

    Alarm if max |r_i| / (c * sigma) exceeds the 95th percentile of the
    same statistic on training normal windows (empirical calibration).
    """
    # calibrate threshold on training normal windows
    Xtr, ytr = load_split("train")
    norm = Xtr[ytr == 0][:2000]
    def lnr_stat(z):
        from scipy.sparse import diags
        w = np.ones_like(z)
        for _ in range(n_iter):
            Wd = w / grid.sigma ** 2
            G = (grid.H.T @ diags(Wd) @ grid.H).toarray()
            x = np.linalg.solve(G, grid.H.T @ (Wd * z))
            r = z - (grid.H @ x)
            s = np.median(np.abs(r)) / 0.6745 + 1e-9
            w = np.minimum(1.0, clip * s / np.maximum(np.abs(r), 1e-12))
        return np.abs(r).max() / grid.sigma
    stats = np.array([lnr_stat(norm[i, -1].astype(np.float64))
                      for i in range(len(norm))])
    th = np.percentile(stats, 95)
    out = {}
    for c in range(5):
        idx = np.where(y == c)[0][:per_class]
        alarms = sum(lnr_stat(X[i, -1].astype(np.float64)) > th
                     for i in idx)
        out[str(c)] = round(alarms / len(idx), 3)
    return out, th


def main():
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    Xte, yte = load_split("test")
    res = {}

    # (a) He-2017-style MLP, snapshot regime (last frame only)
    res["he_mlp_snapshot"] = train_mlp(Xtr[:, -1, :], ytr, Xva[:, -1, :],
                                       yva, Xte[:, -1, :], yte,
                                       "He-MLP(snapshot)")

    # (a') same architecture, window regime (flattened window)
    flat = lambda X: X.reshape(len(X), -1)
    res["he_mlp_window"] = train_mlp(flat(Xtr), ytr, flat(Xva), yva,
                                     flat(Xte), yte, "He-MLP(window)")

    # (b) Huber-IRLS WLS + LNR test
    from grid39 import Grid39
    grid = Grid39()
    rates, th = huber_wls_alarm(grid, Xte, yte)
    res["huber_lnr"] = {"alarm_rates": rates, "threshold": float(th)}
    print("Huber-LNR alarm rates:", rates)

    with open(os.path.join(RESULTS, "external_baselines.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved external_baselines.json")


if __name__ == "__main__":
    main()
