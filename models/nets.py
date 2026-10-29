"""
Model zoo for power-grid cyber-attack detection on measurement windows.

Input windows: (B, L, m) standardised SCADA measurements, L = 64, m = 131.

Own model (`MambaGuard`): bidirectional selective state-space encoder with
  (a) a reconstruction branch trained only on attack-free windows,
  (b) a classification branch over 5 classes,
  (c) a physics-informed consistency loss: the DC-feasible residual
      r = M z_pu of the reconstructed window is penalised (M = I - H G^-1 H' W,
      the fixed weighted-least-squares residual projector of the grid).

Baselines share the same two-branch architecture, swapping the encoder:
  GRU / LSTM / Transformer / 1D-CNN autoencoders.
Classical detectors (OCSVM, IsolationForest, BDD) live in scripts/evaluate.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# selective state-space (Mamba-style) building blocks
# --------------------------------------------------------------------- #
class MambaBlock(nn.Module):
    """Input-dependent (selective) SSM block, ZOH discretisation.

    d: model dim, d_state: SSM state size, expand: inner expansion.
    Follows Gu & Dao (2023) with a pure-PyTorch sequential scan.
    """

    def __init__(self, d: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d, self.d_state = d, d_state
        d_inner = expand * d
        self.d_inner = d_inner
        dt_rank = max(8, d // 16)

        self.in_proj = nn.Linear(d, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3,
                                groups=d_inner)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # dt init: softplus^{-1}(0.01..0.05)-ish bias
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001))
        inv = dt + torch.log(-torch.expm1(-dt))       # softplus inverse
        with torch.no_grad():
            self.dt_proj.weight.zero_()
            self.dt_proj.bias.copy_(inv)
        A = -torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer("A", A.repeat(d_inner, 1), persistent=False)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d, bias=False)

    def forward(self, x):                      # x: (B, L, d)
        B, L, _ = x.shape
        xz = self.in_proj(x)                   # (B, L, 2*d_inner)
        u, z = xz.chunk(2, dim=-1)             # path, gate
        u = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)
        # official detail: dt/B/C projected from the pre-activation signal
        x_dbl = self.x_proj(u)                 # (B, L, dt_rank + 2N)
        dt_rank = self.dt_proj.in_features
        dt, Bp, Cp = torch.split(x_dbl, [dt_rank, self.d_state, self.d_state],
                                 dim=-1)
        u = F.silu(u)
        dt = F.softplus(self.dt_proj(dt)).clamp(max=0.1)   # dt_limit = 0.1

        A = self.A                             # (d_inner, N), negative
        h = u.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)            # ZOH
            dBu = (dA - 1.0) / A * Bp[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            h = dA * h + dBu
            ys.append(torch.einsum("bdn,bn->bd", h, Cp[:, t]))
        y = torch.stack(ys, dim=1)             # (B, L, d_inner)
        y = y + u * self.D
        y = y * F.silu(z)                      # gate
        return self.out_proj(y)


class BiMambaLayer(nn.Module):
    """Residual bidirectional Mamba layer with pre-norm."""

    def __init__(self, d, d_state=16, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fwd = MambaBlock(d, d_state, expand)
        self.bwd = MambaBlock(d, d_state, expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        y = self.fwd(h) + self.bwd(h.flip(1)).flip(1)
        return x + self.drop(y)


# --------------------------------------------------------------------- #
# baseline encoders
# --------------------------------------------------------------------- #
class RNNEncoder(nn.Module):
    def __init__(self, kind: str, d=128, layers=2, dropout=0.1):
        super().__init__()
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = rnn(d, d, num_layers=layers, batch_first=True,
                       bidirectional=True, dropout=dropout)

    def forward(self, x):
        y, _ = self.rnn(x)
        return y                                  # (B, L, 2d)


class TransformerEncoder(nn.Module):
    def __init__(self, d=128, layers=4, heads=8, dropout=0.1, max_len=2048):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout,
                                           batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        return self.enc(x + self.pos[:, : x.shape[1]])


class CNNEncoder(nn.Module):
    def __init__(self, d=128, dropout=0.1):
        super().__init__()
        chs = [d, d, 2 * d, 2 * d]
        layers = []
        for cin, cout in zip(chs[:-1], chs[1:]):
            layers += [nn.Conv1d(cin, cout, 5, padding=2), nn.BatchNorm1d(cout),
                       nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                          # (B, L, d) -> (B, L, 2d)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


# --------------------------------------------------------------------- #
# two-branch detector (shared by our model and baselines)
# --------------------------------------------------------------------- #
class TwoBranchDetector(nn.Module):
    """encoder_name in {mamba, gru, lstm, transformer, cnn}.

    out_dim: encoder output feature dim (reconstruction decoder mirrors it).
    """

    def __init__(self, m: int, n_classes: int = 5, d: int = 128,
                 enc_layers: int = 2, encoder_name: str = "mamba",
                 recon: bool = True, diff_input: bool = True):
        super().__init__()
        self.encoder_name = encoder_name
        self.recon = recon
        self.diff_input = diff_input
        self.in_proj = nn.Linear(3 * m if diff_input else m, d)
        # local-temporal depthwise conv stem (transient sharpening) for the
        # SSM encoder path
        self.conv_stem = (nn.Sequential(
            nn.Conv1d(d, d, 5, padding=2, groups=d), nn.BatchNorm1d(d),
            nn.GELU()) if encoder_name == "mamba" else None)
        if encoder_name == "mamba":
            self.encoder = nn.Sequential(
                *[BiMambaLayer(d) for _ in range(enc_layers)])
            enc_dim = d
        elif encoder_name == "grumamba":
            # hybrid: gated recurrence (robust integration) + selective SSM
            # (input-dependent memory) over the same projected input
            self.encoder_rnn = RNNEncoder("gru", d, enc_layers)
            self.encoder_ssm = nn.Sequential(
                *[BiMambaLayer(d) for _ in range(enc_layers)])
            self.encoder_mix = nn.Linear(3 * d, d)
            enc_dim = d
        elif encoder_name in ("gru", "lstm"):
            self.encoder = RNNEncoder(encoder_name, d, enc_layers)
            enc_dim = 2 * d
        elif encoder_name == "transformer":
            self.encoder = TransformerEncoder(d, enc_layers + 2)
            enc_dim = d
        elif encoder_name == "cnn":
            self.encoder = CNNEncoder(d)
            enc_dim = 2 * d
        else:
            raise ValueError(encoder_name)

        # classification head (mean+max pooling)
        self.cls = nn.Sequential(
            nn.Linear(2 * enc_dim, d), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d, n_classes))
        # reconstruction decoder (mirror depth)
        if recon:
            dec = []
            din = enc_dim
            for _ in range(2):
                dec += [nn.Linear(din, d), nn.GELU()]
                din = d
            dec += [nn.Linear(d, m)]
            self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        if self.diff_input:
            # temporal-difference stem: fast onset transients (dz) and
            # slow level shifts against the pre-window baseline (b)
            d = torch.cat([x[:, :1], x[:, 1:] - x[:, :-1]], dim=1)
            q = max(1, x.shape[1] // 4)
            b = x - x[:, :q].mean(dim=1, keepdim=True)
            x = torch.cat([x, d, b], dim=-1)
        h = self.in_proj(x)
        if self.conv_stem is not None:
            h = self.conv_stem(h.transpose(1, 2)).transpose(1, 2)
        if self.encoder_name == "grumamba":
            return self.encoder_mix(torch.cat(
                [self.encoder_rnn(h), self.encoder_ssm(h)], dim=-1))
        return self.encoder(h)

    def forward(self, x):
        h = self.encode(x)                          # (B, L, enc_dim)
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=-1)
        logits = self.cls(pooled)
        xr = self.decoder(h) if self.recon else None
        return logits, xr


# --------------------------------------------------------------------- #
# losses (incl. physics-informed projector)
# --------------------------------------------------------------------- #
class DetectorLoss(nn.Module):
    """CE (class-weighted) + reconstruction on attack-free windows +
    physics consistency of the reconstruction under the DC model."""

    def __init__(self, M_proj: torch.Tensor, chan_mean: torch.Tensor,
                 chan_std: torch.Tensor, sigma: float,
                 lam_recon: float = 1.0, lam_phys: float = 0.1):
        super().__init__()
        self.register_buffer("M", M_proj)           # (m, m) residual projector
        self.register_buffer("mu", chan_mean)
        self.register_buffer("sd", chan_std)
        self.sigma = sigma
        self.lam_recon, self.lam_phys = lam_recon, lam_phys

    def forward(self, logits, xr, x, y, class_weights):
        ce = F.cross_entropy(logits, y, weight=class_weights)
        out = {"ce": ce.item()}
        loss = ce
        if xr is not None:
            clean = (y == 0) | (y == 1)             # no attacks in the window
            if clean.any():
                mse = F.mse_loss(xr[clean], x[clean])
                loss = loss + self.lam_recon * mse
                out["recon"] = mse.item()
                # physics: DC residual of the de-standardised reconstruction
                xpu = xr[clean] * self.sd + self.mu          # (Bc, L, m) pu
                r = torch.einsum("blm,mk->blk", xpu, self.M) / self.sigma
                phys = (r ** 2).mean()
                loss = loss + self.lam_phys * phys
                out["phys"] = phys.item()
        out["loss"] = loss
        return loss, out
