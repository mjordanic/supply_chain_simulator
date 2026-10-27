"""Multi-product JEPA demand forecaster for track T1 (hidden shop-wide regimes).

:mod:`src.perfect_supplier.jepa` sees one product at a time, so it cannot read
a regime that only shows in how all products move together. Here one example
is the whole shop on one decision day ``t``:

- context: the last ``window`` days of ``[sales / scale, sold_out]`` of every
  product, ``scale`` each product's mean context sales plus one, and the
  weekday of the day after the window;
- long-run level: per product, ``log`` of its context mean over its mean
  since the history began (plus their average), which says whether the shop
  is busier than usual;
- future: the next ``window`` days of every product, in the context's form.

The JEPA target is the EMA encoder's embedding of the joint future window;
the predictor maps the context embedding and the long-run level to it, with
VICReg terms against collapse. The head is a regime mixture: ``n_components``
components, each a weight and a per-(day, product) Gaussian on scaled demand.
Its loss is the censored, discretised mixture likelihood of the whole future
window, so one component must explain all days and products at once. On a
sold-out day only "demand >= sale" is scored. Sampling one component per path
gives trajectory-coherent paths for the planner.

``jepa=False`` drops the target encoder, the predictor, and the latent loss:
the direct ablation.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.perfect_supplier.forecast import HORIZON, QUANTILES
from src.perfect_supplier.world import HISTORY_DAYS

SIGMA_MIN = 0.02


@dataclass
class JointConfig:
    window: int = HORIZON  # the context and the future window are the same length, so they share the encoder
    n_products: int = 4
    embed_dim: int = 64
    hidden: int = 256
    n_components: int = 3
    jepa: bool = True
    latent_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.04
    ema_decay: float = 0.996
    lr: float = 1e-3
    batch_size: int = 512
    steps: int = 20_000
    eval_every: int = 2_000
    seed: int = 0


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


def _weekday_onehot(weekday: torch.Tensor) -> torch.Tensor:
    return nn.functional.one_hot(weekday.long() % 7, 7).float()


class JointModel(nn.Module):
    def __init__(self, config: JointConfig):
        super().__init__()
        self.config = config
        w, P, d, h, K = config.window, config.n_products, config.embed_dim, config.hidden, config.n_components
        n_level = P + 1
        self.encoder = _mlp(2 * w * P + 7, h, d)
        head_in = d + n_level + 7
        if config.jepa:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
            self.predictor = _mlp(d + n_level + 7, h, d)
            head_in += d
        self.head = _mlp(head_in, h, K + 2 * K * w * P)

    def embed(self, window: torch.Tensor, weekday_after: torch.Tensor, target: bool = False) -> torch.Tensor:
        enc = self.target_encoder if target else self.encoder
        return enc(torch.cat([window.flatten(1), _weekday_onehot(weekday_after)], dim=1))

    def forward(self, context: torch.Tensor, level: torch.Tensor, weekday_next: torch.Tensor):
        """``(log weights (B, K), mu, sigma (B, K, H, P), z, z_hat or None)``, all in scaled units."""
        c = self.config
        K, H, P = c.n_components, c.window, c.n_products
        z = self.embed(context, weekday_next)
        wd = _weekday_onehot(weekday_next)
        if c.jepa:
            z_hat = self.predictor(torch.cat([z, level, wd], dim=1))
            x = torch.cat([z, z_hat, level, wd], dim=1)
        else:
            z_hat = None
            x = torch.cat([z, level, wd], dim=1)
        raw = self.head(x)
        logw = torch.log_softmax(raw[:, :K], dim=1)
        rest = raw[:, K:].view(-1, K, 2, H, P)
        mu = rest[:, :, 0]
        sigma = nn.functional.softplus(rest[:, :, 1]) + SIGMA_MIN
        return logw, mu, sigma, z, z_hat

    @torch.no_grad()
    def update_target(self) -> None:
        if not self.config.jepa:
            return
        tau = self.config.ema_decay
        for t, o in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            t.mul_(tau).add_(o.detach(), alpha=1 - tau)


def mixture_nll(logw, mu, sigma, y, censored, half_step) -> torch.Tensor:
    """Mean negative log-likelihood per (day, product) of the joint future window.

    Demand is integer, so a sale ``y`` (scaled) has the mass of its rounding
    interval ``y +- half_step``; zero takes everything below ``half_step``; a
    sold-out day takes everything above ``y - half_step``.
    """
    normal = torch.distributions.Normal(mu, sigma)
    yy, hs = y.unsqueeze(1), half_step.unsqueeze(1)  # (B, 1, H, P), (B, 1, 1, P)
    upper = normal.cdf(yy + hs)
    lower = torch.where(yy > 0, normal.cdf(yy - hs), torch.zeros_like(upper))
    exact = upper - lower
    above = 1 - normal.cdf(yy - hs)
    p = torch.where(censored.unsqueeze(1), above, exact).clamp_min(1e-8)
    per_component = torch.log(p).sum(dim=(2, 3))  # (B, K)
    return -(torch.logsumexp(logw + per_component, dim=1)).mean() / (y.shape[1] * y.shape[2])


def _vicreg(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.relu(1 - std).mean()
    cov = (z.T @ z) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss, (off**2).sum() / z.shape[1]


# ---------------------------------------------------------------------------
# Training data: one shop series per world, windows cut on the fly.
# ---------------------------------------------------------------------------


@dataclass
class ShopBank:
    """Sales, sold-out flags, and true demand of many shops, ``(S, L, P)``; row 0 is history day 0.

    ``demand`` is never shown to the model; it only scores forecasts.
    """

    sales: np.ndarray
    censored: np.ndarray
    demand: np.ndarray
    world_seed: np.ndarray  # (S,)

    def __post_init__(self):
        self.cumsum = np.concatenate([np.zeros_like(self.sales[:, :1]), np.cumsum(self.sales, axis=1)], axis=1)

    def anchors(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        """Every (shop, end-of-context row) from the first decision day on, with a full future window."""
        L = self.sales.shape[1]
        ends = np.arange(HISTORY_DAYS, L - window)
        return np.repeat(np.arange(len(self.sales)), len(ends)), np.tile(ends, len(self.sales))

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, sales=self.sales, censored=self.censored, demand=self.demand,
                            world_seed=self.world_seed)

    @classmethod
    def load(cls, path: str | Path) -> ShopBank:
        z = np.load(path)
        return cls(z["sales"], z["censored"], z["demand"], z["world_seed"])


def features(sales: np.ndarray, censored: np.ndarray, cumsum_end: np.ndarray, end: np.ndarray, window: int):
    """Context ``(B, P, W, 2)``, level ``(B, P + 1)``, and scale ``(B, P)`` (numpy).

    ``sales``/``censored`` are the context windows ``(B, W, P)``; ``cumsum_end``
    the summed sales of rows ``0..end`` ``(B, P)``; ``end`` the context's last row.
    """
    scale = sales.mean(axis=1) + 1.0
    context = np.stack([sales / scale[:, None, :], censored.astype(np.float32)], axis=-1).transpose(0, 2, 1, 3)
    long_run = cumsum_end / (end[:, None] + 1.0)
    level = np.log(scale / (long_run + 1.0))
    return context.astype(np.float32), np.concatenate([level, level.mean(axis=1, keepdims=True)], 1), scale


def make_batch(bank: ShopBank, s: np.ndarray, e: np.ndarray, window: int) -> dict[str, torch.Tensor]:
    ctx_idx = e[:, None] + np.arange(-window + 1, 1)[None, :]
    fut_idx = e[:, None] + np.arange(1, window + 1)[None, :]
    context, level, scale = features(bank.sales[s[:, None], ctx_idx], bank.censored[s[:, None], ctx_idx],
                                     bank.cumsum[s, e + 1], e, window)
    fut_s = bank.sales[s[:, None], fut_idx] / scale[:, None, :]
    fut_c = bank.censored[s[:, None], fut_idx]
    future = np.stack([fut_s, fut_c.astype(np.float32)], axis=-1).transpose(0, 2, 1, 3)

    def t(x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype)

    return {
        "context": t(context), "level": t(level), "future": t(future),
        "weekday_next": t((e + 1) % 7, torch.long), "weekday_after_future": t((e + window + 1) % 7, torch.long),
        "y": t(fut_s), "censored": t(fut_c, torch.bool),
        "demand": t(bank.demand[s[:, None], fut_idx] / scale[:, None, :]),
        "scale": t(scale), "half_step": t(0.5 / scale[:, None, :]),
    }


def _losses(model: JointModel, batch: dict) -> dict[str, torch.Tensor]:
    logw, mu, sigma, z, z_hat = model(batch["context"], batch["level"], batch["weekday_next"])
    out = {"nll": mixture_nll(logw, mu, sigma, batch["y"], batch["censored"], batch["half_step"])}
    total = out["nll"]
    if model.config.jepa:
        with torch.no_grad():
            z_tgt = model.embed(batch["future"], batch["weekday_after_future"], target=True)
        latent = nn.functional.mse_loss(z_hat, z_tgt)
        var_l, cov_l = _vicreg(z)
        out.update(latent=latent, var=var_l, cov=cov_l)
        c = model.config
        total = total + c.latent_weight * latent + c.var_weight * var_l + c.cov_weight * cov_l
    out["total"] = total
    return out


def sample(logw, mu, sigma, n: int, generator: torch.Generator, coherent: bool = True) -> torch.Tensor:
    """``(B, n, H, P)`` scaled demand draws; one component per path, or per (day, product) if not coherent."""
    B, K, H, P = mu.shape
    w = logw.exp()
    shape = (B, n) if coherent else (B, n * H * P)
    k = torch.multinomial(w, shape[1], replacement=True, generator=generator).view(B, n, *(() if coherent else (H, P)))
    b = torch.arange(B)[:, None]
    if coherent:
        m, s = mu[b, k], sigma[b, k]  # (B, n, H, P)
    else:
        hh = torch.arange(H)[None, None, :, None]
        pp = torch.arange(P)[None, None, None, :]
        m, s = mu[b[:, :, None, None], k, hh, pp], sigma[b[:, :, None, None], k, hh, pp]
    return (m + s * torch.randn(m.shape, generator=generator)).clamp_min(0)


@torch.no_grad()
def evaluate(model: JointModel, bank: ShopBank, n: int = 4_000, seed: int = 1) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    s_all, e_all = bank.anchors(model.config.window)
    pick = rng.choice(len(s_all), size=min(n, len(s_all)), replace=False)
    out = {k: float(v) for k, v in _losses(model, make_batch(bank, s_all[pick], e_all[pick],
                                                             model.config.window)).items()}
    model.train()
    return out


def fit(train: ShopBank, val: ShopBank, config: JointConfig, log_path: str | Path | None = None) -> JointModel:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = JointModel(config)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.lr, total_steps=config.steps, pct_start=0.05)
    s_all, e_all = train.anchors(config.window)
    history, best, best_state = [], math.inf, None
    for step in range(1, config.steps + 1):
        pick = rng.integers(0, len(s_all), config.batch_size)
        losses = _losses(model, make_batch(train, s_all[pick], e_all[pick], config.window))
        opt.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        model.update_target()
        if step % config.eval_every == 0 or step == config.steps:
            metrics = {"step": step, "train_total": float(losses["total"].detach()), **evaluate(model, val)}
            history.append(metrics)
            print(f"step {step} train {metrics['train_total']:.4f} val_nll {metrics['nll']:.4f}", flush=True)
            # Early stopping on the forecast loss alone, the same rule for JEPA and direct.
            if metrics["nll"] < best:
                best, best_state = metrics["nll"], copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps({"config": asdict(config), "history": history}, indent=1))
    model.eval()
    return model


def save(model: JointModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state": model.state_dict(), "type": "joint"}, path)


def load(path: str | Path) -> JointModel:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = JointModel(JointConfig(**blob["config"]))
    model.load_state_dict(blob["state"])
    model.eval()
    return model


class JointForecaster:
    """Planner forecaster over a trained :class:`JointModel`, in demand units.

    :meth:`paths` samples one mixture component per path (trajectory-coherent);
    calling it returns the marginal quantiles, which a copula sampler can use.
    """

    def __init__(self, model: JointModel, name: str = "jepa"):
        self.model, self.name = model, name

    @torch.no_grad()
    def _head(self, obs):
        w = self.model.config.window
        sales = obs.sales[-w:].astype(np.float32)[None]
        cens = obs.censored[-w:][None]
        end = np.array([len(obs.sales) - 1])
        context, level, scale = features(sales, cens, obs.sales.sum(axis=0)[None].astype(np.float64), end, w)
        out = self.model(torch.as_tensor(context), torch.as_tensor(level, dtype=torch.float32),
                         torch.as_tensor((end + 1) % 7))
        return out[:3], scale[0]

    def _draw(self, obs, n: int, seed: int, coherent: bool) -> np.ndarray:
        (logw, mu, sigma), scale = self._head(obs)
        g = torch.Generator().manual_seed(int(seed))
        return sample(logw, mu, sigma, n, g, coherent)[0].numpy() * scale[None, None, :]  # (n, H, P)

    def paths(self, obs, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        return np.rint(self._draw(obs, n_paths, int(rng.integers(2**62)), coherent=True))

    def __call__(self, obs) -> np.ndarray:
        draws = self._draw(obs, 512, obs.day, coherent=False)
        return np.quantile(draws, QUANTILES, axis=0).transpose(1, 2, 0)  # (H, P, Q)
