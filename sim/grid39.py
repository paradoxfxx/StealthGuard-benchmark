"""
IEEE 39-bus New England test system (data: MATPOWER case39).

Builds the DC measurement model used by the simulation and by the
classical bad-data detector (BDD):

    z = H * theta + e,   e ~ N(0, diag(sigma^2))

Measurement set (131 channels, 100 MVA base, all in pu):
    [ P_flow(f->t) on 46 branches,
      P_flow(t->f) on 46 branches,
      P_injection at 39 buses ]

State: 39 bus voltage angles, slack (bus 31) removed -> 38 states.
WLS estimator (linear, DC):  x_hat = G^-1 H' W z,  G = H' W H
Residual:  r = z - H x_hat = M z,  M = I - H G^-1 H' W  (projection)
BDD:  J = r' W r  vs  chi2(m - n, 1 - alpha)
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import sparse
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))


class Grid39:
    def __init__(self, case_json: str | None = None, sigma: float = 0.02):
        case_json = case_json or os.path.join(HERE, "case39.json")
        with open(case_json, "r", encoding="utf-8") as fh:
            case = json.load(fh)

        self.base = float(case["baseMVA"])
        self.bus_ids = np.array([int(b[0]) for b in case["bus"]], dtype=int)
        self.nb = len(self.bus_ids)
        self.idx = {b: i for i, b in enumerate(self.bus_ids)}  # bus id -> 0-based
        self.Pd0 = np.array([b[2] / self.base for b in case["bus"]])  # pu
        self.Qd0 = np.array([b[3] / self.base for b in case["bus"]])

        # generators
        self.gen_bus = np.array([self.idx[int(g[0])] for g in case["gen"]], dtype=int)
        self.Pg0 = np.array([g[1] / self.base for g in case["gen"]])
        self.Pmax = np.array([g[8] / self.base for g in case["gen"]])
        # AGC participation factors (proportional to headroom, normalised)
        head = np.clip(self.Pmax - self.Pg0, 0.05, None)
        self.alpha = head / head.sum()
        # slack bus of the case (type 3)
        self.slack = [self.idx[int(b[0])] for b in case["bus"] if int(b[1]) == 3][0]

        # branches: f, t, r, x, b, ... tap at index 8
        self.f = np.array([self.idx[int(b[0])] for b in case["branch"]], dtype=int)
        self.t = np.array([self.idx[int(b[1])] for b in case["branch"]], dtype=int)
        self.x = np.array([b[3] for b in case["branch"]], dtype=float)
        tap = np.array([b[8] if b[8] else 1.0 for b in case["branch"]], dtype=float)
        self.tap = tap
        self.nl = len(self.x)
        self.branch_ids = [(int(b[0]), int(b[1])) for b in case["branch"]]

        self.sigma = sigma  # pu, measurement std of every channel
        self._build()

    # ------------------------------------------------------------------ #
    # topology handling
    # ------------------------------------------------------------------ #
    def _build(self, out_lines: tuple[int, ...] = ()):
        """Rebuild B (DC), H and derived matrices for the current topology."""
        keep = np.ones(self.nl, bool)
        for k in out_lines:
            keep[k] = False
        f, t, x, tap = self.f[keep], self.t[keep], self.x[keep], self.tap[keep]
        b_series = 1.0 / x  # series susceptance (DC)

        nb, nl = self.nb, len(f)
        # node-branch incidence C (nb x nl): C[f,k]=+1/tap, C[t,k]=-1/tap
        # branch flow f->t:  P_ft = b_k * (C.T theta)_k
        C = sparse.csr_matrix(
            (np.concatenate([np.ones(nl), -np.ones(nl)]) / np.concatenate([tap, tap]),
             (np.concatenate([f, t]), np.tile(np.arange(nl), 2))),
            shape=(nb, nl))
        # DC bus matrix B = C diag(b) C' ; injections = B theta
        Bb = C @ sparse.diags(b_series) @ C.T
        self.B = Bb
        self.keep = keep
        self.f_k, self.t_k, self.x_k, self.tap_k, self.b_series = f, t, x, tap, b_series
        self.branch_index_map = np.where(keep)[0]  # position -> global branch id

        # state vector: all angles except slack
        state_idx = [i for i in range(nb) if i != self.slack]
        self.state_idx = np.array(state_idx)

        # ---- measurement model H  (m x n) -------------------------------
        rows = []
        # flows f->t and t->f on active branches (global order preserved)
        Hf = sparse.diags(b_series) @ C.T          # P_ft (nl x nb)
        Ht = -Hf                                   # P_tf = -P_ft
        rows += [Hf, Ht]
        H_inj = Bb                                 # P_inj at bus = (B theta)_bus
        rows.append(H_inj)
        Hfull = sparse.vstack(rows, format="csr")  # (nb*? , nb)
        self.H = Hfull[:, state_idx].tocsr()       # drop slack column

        # channel bookkeeping
        self.n_flow = 2 * nl
        self.m = Hfull.shape[0]
        self.n = self.H.shape[1]
        # channel type flags: 0 flow, 1 injection
        self.chan_type = np.concatenate(
            [np.zeros(2 * nl, int), np.ones(nb, int)])

        # WLS matrices (linear DC)
        W = sparse.diags(1.0 / self.sigma ** 2 * np.ones(self.m))
        G = (self.H.T @ W @ self.H).tocsc()
        self.Ginv = sparse.linalg.inv(G).tocsc()
        self.W = W
        self.K = (self.Ginv @ self.H.T @ W).tocsr()      # gain -> x_hat = K z
        self.M = (sparse.identity(self.m, format="csr") - self.H @ self.K).tocsr()
        self.chi2_th = chi2.ppf(0.95, self.m - self.n)

    def set_topology(self, out_lines: tuple[int, ...] = ()):
        self._build(out_lines)

    @property
    def outable_lines(self) -> np.ndarray:
        """Line indices whose removal keeps the graph connected
        (excludes bridges such as (16,19) and (23,36) in this case)."""
        if not hasattr(self, "_outable"):
            from collections import deque
            safe = []
            for k in range(self.nl):
                if self.tap[k] != 1.0:
                    continue
                adj = {i: set() for i in range(self.nb)}
                for j in range(self.nl):
                    if j == k:
                        continue
                    adj[self.f[j]].add(self.t[j])
                    adj[self.t[j]].add(self.f[j])
                seen, dq = {0}, deque([0])
                while dq:
                    u = dq.popleft()
                    for v in adj[u]:
                        if v not in seen:
                            seen.add(v)
                            dq.append(v)
                if len(seen) == self.nb:
                    safe.append(k)
            self._outable = np.array(safe, dtype=int)
        return self._outable

    # ------------------------------------------------------------------ #
    # physics
    # ------------------------------------------------------------------ #
    def solve_dc(self, Pinj: np.ndarray) -> np.ndarray:
        """Solve DC power flow for full bus injection vector (pu, slack=unused)."""
        rhs = Pinj[self.state_idx] - self.B[self.state_idx, self.slack].toarray().ravel() * 0.0
        th_s = np.linalg.solve(self.B[self.state_idx][:, self.state_idx].toarray(), rhs)
        theta = np.zeros(self.nb)
        theta[self.slack] = 0.0
        theta[self.state_idx] = th_s
        return theta

    def measure(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """z = H theta + e."""
        z = self.H @ theta[self.state_idx] + rng.normal(0.0, self.sigma, self.m)
        return z.astype(np.float32)

    def wls(self, z: np.ndarray):
        """Linear WLS estimate + residual + objective."""
        x_hat = self.K @ z
        r = self.M @ z
        J = float(np.sum((r / self.sigma) ** 2))
        return x_hat, r, J

    def bdd(self, z: np.ndarray) -> bool:
        """True => alarm (bad data detected)."""
        return self.wls(z)[2] > self.chi2_th

    # ------------------------------------------------------------------ #
    # attack construction
    # ------------------------------------------------------------------ #
    def fdia_vector(self, buses: list[int], dth: float) -> np.ndarray:
        """Stealth attack vector a = H c  (c = dth on chosen non-slack buses)."""
        c = np.zeros(self.nb)
        for b in buses:
            c[self.idx[b]] = dth
        return (self.H @ c[self.state_idx]).astype(np.float64)


if __name__ == "__main__":
    g = Grid39()
    print(f"buses={g.nb} branches={g.nl} gens={len(g.gen_bus)}")
    print(f"measurements m={g.m} states n={g.n} redundancy={g.m/g.n:.2f}")
    print(f"sigma={g.sigma} pu   chi2_95({g.m-g.n})={g.chi2_th:.1f}")
    # base-case solve: dispatch = Pg0, load = Pd0
    Pinj = np.zeros(g.nb)
    Pinj[g.gen_bus] = g.Pg0
    Pinj -= g.Pd0
    Pinj[g.slack] = 0.0  # slack balances
    theta = g.solve_dc(Pinj)
    z = g.measure(theta, np.random.default_rng(0))
    x, r, J = g.wls(z)
    print(f"base-case: max|theta|={np.abs(theta).max():.4f} rad, J={J:.1f} "
          f"(threshold {g.chi2_th:.1f}) -> {'ALARM' if g.bdd(z) else 'pass'}")
    a = g.fdia_vector([4, 5, 6], 0.03)
    za = z + a
    x2, r2, J2 = g.wls(za)
    print(f"FDIA +0.03 rad @bus4-6: J={J2:.1f} -> {'ALARM' if g.bdd(za) else 'pass'} "
          f"(state shift max |dx|={np.abs(x2-x).max():.4f} rad)")
