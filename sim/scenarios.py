"""
Temporal scenario generator for the IEEE 39-bus DC measurement stream.

Simulates quasi-steady-state operation at Ts = 5 s (SCADA cadence):

  load      Pd_i(t) = Pd0_i * s_sys(t) * s_i(t)      (daily curve x AR(1) drift)
  dispatch  Pg_i(t) = Pg0_i + alpha_i * dL(t) + AR(1) regulation noise,
            slack absorbs the remainder
  state     theta(t) = B^-1 Pinj(t)   (batched dense solve per topology)
  measurements  z(t) = H theta(t) + e,  e ~ N(0, sigma^2 I)

Scenario classes
  0  normal           no event
  1  natural          load step / generator trip / line outage (benign)
  2  fdia             stealth false-data injection  a = H c  (bypasses BDD)
  3  replay           recorded normal data replayed over the live stream
  4  rfdi             random (non-stealth) false data / scaling errors
"""
from __future__ import annotations

import numpy as np

from grid39 import Grid39

TS = 5.0                     # sampling period [s]
DAY = int(86400 / TS)        # 17280 steps per day

CLASS_NAMES = ["normal", "natural", "fdia", "replay", "rfdi"]
NATURAL_KINDS = ["load_step", "gen_trip", "line_outage"]


class ScenarioConfig:
    """Parameters that control scenario richness (paper Tab. 'simulation setup')."""
    # daily load curve
    daily_amp1 = 0.14          # fundamental daily swing
    daily_amp2 = 0.05          # half-daily harmonic
    ar_load_a = 0.995          # bus-level AR(1) coefficient  (~10 min corr)
    ar_load_sig = 0.0008       # bus-level AR(1) innovation std (~0.5% level)
    ar_gen_sig = 0.0004        # generator regulation noise std (pu)
    # natural events
    load_step_frac = (0.10, 0.30)      # fraction of the bus load, signed
    gen_trip_frac = (0.20, 0.55)       # fraction of unit output
    outage_minutes = (10, 50)
    # attacks
    fdia_k_bus = (2, 5)                # number of compromised state variables
    fdia_bias_sig = (4.0, 10.0)         # |a|_inf in units of sigma
    fdia_onset_steps = (0, 60)         # ramp-in length (0 = abrupt)
    attack_minutes = (10, 30)
    rfdi_frac_channels = (0.08, 0.35)
    rfdi_bias_sig = (4.0, 10.0)
    replay_lag_minutes = (12, 90)


def _ar1(n, a, sig, rng):
    x = np.empty(n)
    x[0] = rng.normal(0, sig / np.sqrt(1 - a * a))
    innov = rng.normal(0, sig, n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + innov[i]
    return x


def _smooth_step(n, width, rng=None):
    """0->1 ramp of integer length `width` (width=0 -> step)."""
    if width <= 0:
        return np.ones(n)
    w = min(width, n)
    s = 0.5 - 0.5 * np.cos(np.pi * np.arange(w) / w)
    return np.concatenate([s, np.ones(n - w)])


class ScenarioRunner:
    def __init__(self, grid: Grid39, cfg: ScenarioConfig | None = None, seed: int = 0):
        self.g = grid
        self.cfg = cfg or ScenarioConfig()
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- #
    def run(self, T: int, events: list[dict] | None = None):
        """Simulate T steps; returns z (T,m) float32, labels (T,), log(list)."""
        rng, cfg, g = self.rng, self.cfg, self.g
        events = sorted(events or [], key=lambda e: e["t0"])

        # ---------- loads ----------
        t = np.arange(T) / DAY
        ph1, ph2 = rng.uniform(0, 1, 2)
        s_sys = (1.0 + cfg.daily_amp1 * np.sin(2 * np.pi * (t + ph1))
                 + cfg.daily_amp2 * np.sin(4 * np.pi * (t + ph2)))
        S = np.array([_ar1(T, cfg.ar_load_a, cfg.ar_load_sig, rng)
                      for _ in range(g.nb)])                 # (nb, T)
        Pd = g.Pd0[:, None] * s_sys[None, :] * np.exp(S)     # log-normal-ish drift

        # ---------- dispatch ----------
        dL = Pd.sum(0) - g.Pd0.sum()
        Gnoise = np.array([_ar1(T, 0.999, cfg.ar_gen_sig, rng)
                           for _ in range(len(g.gen_bus))])
        Pg = g.Pg0[:, None] + g.alpha[:, None] * dL[None, :] + Gnoise

        # ---------- natural events alter physics -----------------------
        for ev in events:
            if ev["kind"] == "load_step":
                b = g.idx[ev["bus"]]
                Pd[b, ev["t0"]:ev["t0"] + ev["dur"]] *= (
                    1 + ev["mag"] * _smooth_step(ev["dur"], ev.get("onset", 0))[None, :])[0]
            elif ev["kind"] == "gen_trip":
                j = ev["unit"]
                Pg[j, ev["t0"]:ev["t0"] + ev["dur"]] *= (
                    1 - ev["frac"] * _smooth_step(ev["dur"], ev.get("onset", 12))[None, :])[0]

        Pd = Pd.T                 # (T, nb)
        Pg = Pg.T                 # (T, n_gen)

        # ---------- power flow per topology segment --------------------
        Pinj = -Pd.copy()
        for j, b in enumerate(g.gen_bus):
            Pinj[:, b] += Pg[:, j]
        theta = np.zeros((T, g.nb))
        top_ev = [(ev["t0"], ev["t1"]) for ev in events if ev["kind"] == "line_outage"]
        cuts = sorted({0, T} | {t0 for t0, _ in top_ev} | {t1 for _, t1 in top_ev})
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            out = tuple(ev["line"] for ev in events
                        if ev["kind"] == "line_outage"
                        and ev["t0"] <= s0 < ev["t1"])
            if out:
                self.g.set_topology(out)
            Bred = self.g.B[self.g.state_idx][:, self.g.state_idx].toarray()
            rhs = Pinj[s0:s1, self.g.state_idx].T
            theta[s0:s1, self.g.state_idx] = np.linalg.solve(Bred, rhs).T
        if not np.isfinite(theta).all() or np.abs(theta).max() > 3.0:
            raise RuntimeError("power-flow solve diverged (islanding?)")
        if top_ev:
            self.g.set_topology(())   # restore

        # ---------- measurements ----------
        z = (theta[:, self.g.state_idx] @ self.g.H.T.toarray()
             + rng.normal(0, self.g.sigma, (T, self.g.m))).astype(np.float32)

        # ---------- cyber attacks (measurement-only) -------------------
        labels = np.zeros(T, dtype=np.int8)
        for ev in events:
            if ev["kind"] == "fdia":
                a_vec = ev["a_vec"]                    # (m,) constant bias
                ramp = _smooth_step(ev["dur"], ev["onset"])
                z[ev["t0"]:ev["t0"] + ev["dur"]] += (a_vec * ramp[:, None]).astype(np.float32)
                labels[ev["t0"]:ev["t0"] + ev["dur"]] = 2
            elif ev["kind"] == "rfdi":
                ch = ev["channels"]
                bias = ev["bias"]
                ramp = _smooth_step(ev["dur"], ev.get("onset", 0))
                z[ev["t0"]:ev["t0"] + ev["dur"], ch] += (
                    bias * ramp[:, None]).astype(np.float32)
                labels[ev["t0"]:ev["t0"] + ev["dur"]] = 4
            elif ev["kind"] == "replay":
                lag, t0, dur = ev["lag"], ev["t0"], ev["dur"]
                z[t0:t0 + dur] = z[t0 - lag:t0 - lag + dur]
                labels[t0:t0 + dur] = 3
            elif ev["kind"] in NATURAL_KINDS:
                labels[ev["t0"]:ev["t0"] + ev["dur"]] = 1

        return z, labels, events

    # ---------------------------------------------------------------- #
    def sample_events(self, T: int, rng: np.random.Generator,
                      n_natural: int, n_attack: int):
        """Random event/attack schedule for one run (mutually non-overlapping)."""
        cfg, g = self.cfg, self.g
        slots = []
        events = []
        def free(t0, dur):
            return all(t0 + dur < s or s + d < t0 for s, d in slots) and t0 - 7200 > 0 \
                and t0 + dur + 600 < T

        guard = 0
        while len([e for e in events if e["kind"] in NATURAL_KINDS]) < n_natural and guard < 200:
            guard += 1
            kind = NATURAL_KINDS[rng.integers(0, 3)]
            dur = int(rng.uniform(*map(lambda m: m * 60 / TS, cfg.outage_minutes)))
            t0 = int(rng.integers(2 * 3600 // TS, T - dur - 1))
            if not free(t0, dur):
                continue
            ev = {"kind": kind, "t0": t0, "dur": dur, "t1": t0 + dur}
            if kind == "load_step":
                ev["bus"] = int(g.bus_ids[rng.integers(0, g.nb)])
                ev["mag"] = float(rng.choice([-1, 1]) * rng.uniform(*cfg.load_step_frac))
                ev["onset"] = int(rng.integers(0, 24))
            elif kind == "gen_trip":
                ev["unit"] = int(rng.integers(0, len(g.gen_bus)))
                ev["frac"] = float(rng.uniform(*cfg.gen_trip_frac))
                ev["onset"] = int(rng.integers(6, 30))
            else:  # line outage: only lines whose removal keeps the grid connected
                cand = g.outable_lines
                ev["line"] = int(cand[rng.integers(0, len(cand))])
            slots.append((t0, dur)); events.append(ev)

        guard = 0
        while len([e for e in events if e["kind"] in ("fdia", "replay", "rfdi")]) < n_attack and guard < 200:
            guard += 1
            kind = ("fdia", "replay", "rfdi")[rng.integers(0, 3)]
            dur = int(rng.uniform(*cfg.attack_minutes) * 60 / TS)
            t0 = int(rng.integers(3 * 3600 // TS, T - dur - 1))
            if not free(t0, dur):
                continue
            ev = {"kind": kind, "t0": t0, "dur": dur, "t1": t0 + dur}
            if kind == "fdia":
                k = int(rng.integers(*cfg.fdia_k_bus))
                buses = sorted(rng.choice(
                    [b for b in g.bus_ids if b != 31], size=k, replace=False).tolist())
                a = g.fdia_vector(buses, 0.02)            # direction only
                # scale so that |a|_inf hits the intended sigma multiple
                a = a * (rng.uniform(*cfg.fdia_bias_sig) * g.sigma / np.abs(a).max())
                ev.update(buses=buses, a_vec=a, onset=int(rng.integers(*cfg.fdia_onset_steps)))
            elif kind == "rfdi":
                n_ch = int(rng.uniform(*cfg.rfdi_frac_channels) * g.m)
                ch = rng.choice(g.m, size=n_ch, replace=False)
                mag = rng.uniform(*cfg.rfdi_bias_sig) * g.sigma
                bias = rng.choice([-1.0, 1.0], (n_ch,)) * mag * rng.uniform(0.4, 1.0, (n_ch,))
                ev.update(channels=ch, bias=bias,
                          onset=int(rng.integers(0, 30)))
            else:
                ev["lag"] = int(rng.uniform(*cfg.replay_lag_minutes) * 60 / TS)
            slots.append((t0, dur)); events.append(ev)
        return events


def make_run(grid: Grid39, seed: int, T: int, n_natural: int, n_attack: int,
             cfg: ScenarioConfig | None = None):
    """One simulated day; RNG inside ScenarioRunner drives load/dispatch noise,
    a separate RNG drives the event schedule (kept independent for analysis)."""
    runner = ScenarioRunner(grid, cfg, seed=seed)
    ev_rng = np.random.default_rng(seed + 1_000_000)
    events = runner.sample_events(T, ev_rng, n_natural, n_attack)
    z, labels, _ = runner.run(T, events)
    return z, labels, events


if __name__ == "__main__":
    g = Grid39()
    z, lab, ev = make_run(g, seed=7, T=DAY, n_natural=2, n_attack=2)
    print("z", z.shape, "labels:", dict(zip(*np.unique(lab, return_counts=True))))
    for e in ev:
        print(" ", {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in e.items()})
    # BDD on a stealth fdia window
    for e in ev:
        if e["kind"] == "fdia":
            seg = z[e["t0"] + e["dur"] // 2 - 50: e["t0"] + e["dur"] // 2 + 50]
            Js = [g.wls(zz)[2] for zz in seg]
            print("FDIA window J: mean", np.mean(Js).round(1),
                  "alarm rate", np.mean([g.bdd(zz) for zz in seg]).round(3))
