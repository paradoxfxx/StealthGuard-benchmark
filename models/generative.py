"""
Generative augmentation models for scarce attack windows.

CDDM  : conditional denoising diffusion model (DDPM, epsilon-prediction,
        cosine schedule, class-conditioned by embedding). Generates
        synthetic attack windows to rebalance the training set.
WGAN  : Wasserstein GAN with gradient penalty, conditional — the
        augmentation strategy of the earlier AGNet-style work, kept as
        the head-to-head baseline.

Both operate on standardised windows (B, L, m).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# conditional DDPM
# --------------------------------------------------------------------- #
def cosine_alpha_bar(T, s=0.008):
    t = torch.linspace(0, 1, T + 1)
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return torch.clip(ab[1:], 1e-4, 1.0)


class SinusoidalTE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):                          # t: (B,) in [0,1)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(
            half, device=t.device) / half)
        ang = t[:, None] * freqs[None] * 1000.0
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(cin)
        self.conv1 = nn.Conv1d(cin, cout, 5, padding=2)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 5, padding=2)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class CondUNet1d(nn.Module):
    """Small 1D UNet for (B, m, L) signals with time+class conditioning."""

    def __init__(self, m, base=64, emb_dim=128, n_classes=5):
        super().__init__()
        self.temb = SinusoidalTE(emb_dim)
        self.tmlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                  nn.Linear(emb_dim, emb_dim))
        self.cemb = nn.Embedding(n_classes, emb_dim)
        ch = [base, base * 2, base * 4]
        self.in_conv = nn.Conv1d(m, ch[0], 1)
        self.d1 = ResBlock1d(ch[0], ch[0], emb_dim)
        self.down12 = nn.Conv1d(ch[0], ch[1], 4, 2, 1)
        self.d2 = ResBlock1d(ch[1], ch[1], emb_dim)
        self.down23 = nn.Conv1d(ch[1], ch[2], 4, 2, 1)
        self.d3 = ResBlock1d(ch[2], ch[2], emb_dim)
        self.up23 = nn.ConvTranspose1d(ch[2], ch[1], 4, 2, 1)
        self.u2 = ResBlock1d(ch[1] * 2, ch[1], emb_dim)
        self.up12 = nn.ConvTranspose1d(ch[1], ch[0], 4, 2, 1)
        self.u1 = ResBlock1d(ch[0] * 2, ch[0], emb_dim)
        self.out = nn.Conv1d(ch[0], m, 1)

    def forward(self, x, t, y):                     # x: (B, m, L)
        emb = self.tmlp(self.temb(t)) + self.cemb(y)
        h1 = self.in_conv(x)
        h1 = self.d1(h1, emb)                       # (B, c0, L)
        h2 = self.d2(self.down12(h1), emb)          # (B, c1, L/2)
        h3 = self.d3(self.down23(h2), emb)          # (B, c2, L/4)
        u2 = self.u2(torch.cat([self.up23(h3), h2], 1), emb)
        u1 = self.u1(torch.cat([self.up12(u2), h1], 1), emb)
        return self.out(F.silu(u1))


class CondDDPM(nn.Module):
    def __init__(self, m, L, T=400, base=64, n_classes=5):
        super().__init__()
        self.net = CondUNet1d(m, base, n_classes=n_classes)
        self.T = T
        ab = cosine_alpha_bar(T)
        self.register_buffer("alpha_bar", ab)
        self.register_buffer("alpha", ab / torch.cat(
            [torch.ones(1), ab[:-1]]))
        self.register_buffer("beta", 1 - self.alpha)

    def forward(self, x0, y):                       # x0: (B, L, m)
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        ab = self.alpha_bar[t][:, None, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        pred = self.net(xt.transpose(1, 2), t / self.T, y).transpose(1, 2)
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def ddim_sample(self, n, y, L, device, steps=50):
        x = torch.randn(n, L, self.net.in_conv.in_channels * 0 + self.alpha_bar.shape[0] * 0 + self.net.out.out_channels, device=device) \
            if False else torch.randn(n, L, self.net.out.out_channels, device=device)
        ts = torch.linspace(self.T - 1, 0, steps).long()
        for i, t in enumerate(ts):
            tb = torch.full((n,), t.item(), device=device, dtype=torch.long)
            tt = tb.float() / self.T
            eps = self.net(x.transpose(1, 2), tt, y).transpose(1, 2)
            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < steps else torch.ones((), device=device)
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0 = x0.clamp(-6, 6)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x


# --------------------------------------------------------------------- #
# conditional WGAN-GP
# --------------------------------------------------------------------- #
class CondGenerator(nn.Module):
    def __init__(self, m, nz=64, base=64, n_classes=5, L=64):
        super().__init__()
        self.L = L
        self.z_fc = nn.Linear(nz, base * (L // 4))
        self.y_fc = nn.Linear(n_classes, base * (L // 4))
        def up(cin, cout):
            return nn.Sequential(nn.ConvTranspose1d(cin, cout, 4, 2, 1),
                                 nn.BatchNorm1d(cout), nn.ReLU())
        self.net = nn.Sequential(up(base, base), up(base, base // 2))
        self.refine = nn.Sequential(
            nn.Conv1d(base // 2, base // 2, 5, padding=2),
            nn.BatchNorm1d(base // 2), nn.ReLU(),
            nn.Conv1d(base // 2, base // 2, 5, padding=2))
        self.out = nn.Conv1d(base // 2, m, 1)

    def forward(self, z, y):                        # z: (B, nz), y one-hot
        h = (self.z_fc(z) + self.y_fc(y)).reshape(z.shape[0], -1, self.L // 4)
        h = self.net(h)
        return self.out(h + self.refine(h)).transpose(1, 2)   # (B, L, m)


class CondCritic(nn.Module):
    def __init__(self, m, base=64, n_classes=5, L=64):
        super().__init__()
        self.y_fc = nn.Linear(n_classes, base)
        self.n_pool = 3 if L <= 128 else 4      # enough downsampling at 256
        def blk(cin, cout):
            return nn.Sequential(nn.Conv1d(cin, cout, 5, 2, 2),
                                 nn.LeakyReLU(0.2))
        blocks = [blk(m + base, base), blk(base, base * 2)]
        ch = base * 2
        for _ in range(self.n_pool - 2):
            blocks.append(blk(ch, ch * 2)); ch *= 2
        self.net = nn.Sequential(*blocks)
        self.out = nn.Linear(ch * max(1, L // 2 ** self.n_pool), 1)

    def forward(self, x, y):                        # x: (B, L, m)
        h = torch.cat([x.transpose(1, 2), self.y_fc(y).unsqueeze(-1).expand(-1, -1, x.shape[1])], 1)
        return self.out(self.net(h).flatten(1))


def wgan_gp_loss(critic, real, fake, y_onehot, lam=10.0):
    B = real.shape[0]
    d_real = critic(real, y_onehot).mean()
    d_fake = critic(fake, y_onehot).mean()
    eps = torch.rand(B, 1, 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_i = critic(interp, y_onehot)
    grad = torch.autograd.grad(d_i.sum(), interp, create_graph=True)[0]
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return d_fake - d_real + lam * gp, d_real.item() - d_fake.item()
