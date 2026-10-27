"""JEPA demand forecaster for the perfect-supplier shop.

Inventory mechanics are exact in the planner, so the only thing to learn is
demand, and the shop only sees censored sales. One example is one product on
one decision day ``t``:

- context: the last ``WINDOW`` days of ``[sales / scale, sold_out]`` ending on ``t``
- future: the next ``WINDOW`` days, days ``t + 1 .. t + WINDOW``, in the same form

``scale`` is the mean context sales plus one, so the model is scale-free.
Each window also carries the weekday of the day after it, which fixes the
weekday of every day inside it.

The online encoder embeds the context. An EMA copy embeds the future window
and is the JEPA target; the predictor maps the context embedding to it.
VICReg variance and covariance terms keep the context embedding from
collapsing. A quantile head reads the context embedding and the predicted
future embedding and emits ``len(config.quantiles)`` quantiles of demand for each
future day. Its loss is censored pinball: on a sold-out day the sale is only
a lower bound on demand, so only predictions below it are penalised.
The head's gradient also trains the encoder and predictor.

``jepa=False`` drops the target encoder, the predictor, and the latent loss;
the head reads the context embedding alone. That is the direct ablation.
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

WINDOW = HORIZON


@dataclass
class JepaConfig:
    window: int = WINDOW
    embed_dim: int = 64
    hidden: int = 256
    jepa: bool = True
    latent_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.04
    ema_decay: float = 0.996
    lr: float = 1e-3
    batch_size: int = 1024
    steps: int = 20_000
    eval_every: int = 2_000
    seed: int = 0
    quantiles: tuple[float, ...] = tuple(QUANTILES.tolist())


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


def _weekday_onehot(weekday: torch.Tensor) -> torch.Tensor:
    return nn.functional.one_hot(weekday.long() % 7, 7).float()


class JepaModel(nn.Module):
    def __init__(self, config: JepaConfig):
        super().__init__()
        self.config = config
        w, d, h = config.window, config.embed_dim, config.hidden
        n_q = len(config.quantiles)
        self.encoder = _mlp(2 * w + 7, h, d)
        head_in = d + 7
        if config.jepa:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
            self.predictor = _mlp(d + 7, h, d)
            head_in += d
        self.head = _mlp(head_in, h, HORIZON * n_q)
        self.n_q = n_q

    def embed(self, window: torch.Tensor, weekday_after: torch.Tensor, target: bool = False) -> torch.Tensor:
        enc = self.target_encoder if target else self.encoder
        x = torch.cat([window.flatten(1), _weekday_onehot(weekday_after)], dim=1)
        return enc(x)

    def forward(self, context: torch.Tensor, weekday_next: torch.Tensor):
        """Return ``(scaled quantiles (B, HORIZON, Q), z_context, z_future_pred or None)``."""
        z = self.embed(context, weekday_next)
        wd = _weekday_onehot(weekday_next)
        if self.config.jepa:
            z_hat = self.predictor(torch.cat([z, wd], dim=1))
            head_x = torch.cat([z, z_hat, wd], dim=1)
        else:
            z_hat = None
            head_x = torch.cat([z, wd], dim=1)
        raw = self.head(head_x).view(-1, HORIZON, self.n_q)
        # Monotone in the quantile level and non-negative.
        base = nn.functional.softplus(raw[..., :1]) - 0.5
        steps = nn.functional.softplus(raw[..., 1:])
        q = torch.cat([base, base + torch.cumsum(steps, dim=-1)], dim=-1).clamp_min(0.0)
        return q, z, z_hat

    @torch.no_grad()
    def update_target(self) -> None:
        if not self.config.jepa:
            return
        tau = self.config.ema_decay
        for t, o in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            t.mul_(tau).add_(o.detach(), alpha=1 - tau)


def censored_pinball(q: torch.Tensor, y: torch.Tensor, censored: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss; on censored days demand >= y, so over-prediction is free."""
    diff = y.unsqueeze(-1) - q  # > 0 when the prediction is below the sale
    loss = torch.maximum(levels * diff, (levels - 1) * diff)
    loss = torch.where(censored.unsqueeze(-1) & (diff < 0), torch.zeros_like(loss), loss)
    return loss.mean()


def _vicreg(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.relu(1 - std).mean()
    cov = (z.T @ z) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss, (off**2).sum() / z.shape[1]


# ---------------------------------------------------------------------------
# Training data: one series per (world, product), windows cut on the fly.
# ---------------------------------------------------------------------------


@dataclass
class SeriesBank:
    """Sales, sold-out flags, and true demand for many (world, product) series.

    ``demand`` is never shown to the model; it is only used to score forecasts.
    ``weekday0`` is the weekday of each series' first row.
    """

    sales: np.ndarray  # (S, L) float32
    censored: np.ndarray  # (S, L) bool
    demand: np.ndarray  # (S, L) float32
    weekday0: np.ndarray  # (S,) int
    world_seed: np.ndarray  # (S,) int
    product: np.ndarray  # (S,) int
    first_day: int  # first row index that may end a context window

    def anchors(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        """Every (series, end-of-context row) with a full future window."""
        L = self.sales.shape[1]
        ends = np.arange(max(self.first_day, window - 1), L - window)
        s = np.repeat(np.arange(len(self.sales)), len(ends))
        e = np.tile(ends, len(self.sales))
        return s, e


def bank_from_timeseries(frame, history: dict[int, np.ndarray]) -> SeriesBank:
    """Build a bank from a training-log frame and each world's pre-episode history."""
    from src.perfect_supplier.world import HISTORY_DAYS

    frame = frame.sort_values(["world_seed", "product", "day"])
    sales, cens, dem, wd0, seeds, products = [], [], [], [], [], []
    for (seed, product), g in frame.groupby(["world_seed", "product"], sort=False):
        hist = history[int(seed)][:, int(product)]
        sales.append(np.concatenate([hist, g["sales"].to_numpy()]))
        cens.append(np.concatenate([np.zeros(HISTORY_DAYS, bool), g["censored"].to_numpy(bool)]))
        dem.append(np.concatenate([hist, g["demand"].to_numpy()]))
        wd0.append(0)  # weekday of history row 0 is 0 by construction in world.py
        seeds.append(int(seed))
        products.append(int(product))
    return SeriesBank(
        sales=np.asarray(sales, np.float32),
        censored=np.asarray(cens, bool),
        demand=np.asarray(dem, np.float32),
        weekday0=np.asarray(wd0),
        world_seed=np.asarray(seeds),
        product=np.asarray(products),
        first_day=HISTORY_DAYS,
    )


def make_batch(bank: SeriesBank, s: np.ndarray, e: np.ndarray, window: int, device="cpu"):
    """Context, future, and scale tensors for anchors ``(s, e)``."""
    ctx_idx = e[:, None] + np.arange(-window + 1, 1)[None, :]
    fut_idx = e[:, None] + np.arange(1, window + 1)[None, :]
    ctx_s = bank.sales[s[:, None], ctx_idx]
    ctx_c = bank.censored[s[:, None], ctx_idx]
    scale = ctx_s.mean(axis=1, keepdims=True) + 1.0
    fut_s = bank.sales[s[:, None], fut_idx]
    fut_c = bank.censored[s[:, None], fut_idx]
    fut_d = bank.demand[s[:, None], fut_idx]
    weekday_next = (bank.weekday0[s] + e + 1) % 7
    weekday_after_future = (bank.weekday0[s] + e + window + 1) % 7

    def t(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device)

    context = torch.stack([t(ctx_s / scale), t(ctx_c)], dim=-1)
    future = torch.stack([t(fut_s / scale), t(fut_c)], dim=-1)
    return {
        "context": context,
        "future": future,
        "weekday_next": t(weekday_next, torch.long),
        "weekday_after_future": t(weekday_after_future, torch.long),
        "y": t(fut_s / scale),
        "censored": t(fut_c, torch.bool),
        "demand": t(fut_d / scale),
        "scale": t(scale),
    }


def _losses(model: JepaModel, batch: dict, levels: torch.Tensor) -> dict[str, torch.Tensor]:
    q, z, z_hat = model(batch["context"], batch["weekday_next"])
    out = {"pinball": censored_pinball(q, batch["y"], batch["censored"], levels)}
    total = out["pinball"]
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


@torch.no_grad()
def evaluate(model: JepaModel, bank: SeriesBank, n: int = 20_000, seed: int = 1) -> dict[str, float]:
    """Censored pinball on sales, and pinball and coverage against true demand."""
    model.eval()
    rng = np.random.default_rng(seed)
    s_all, e_all = bank.anchors(model.config.window)
    pick = rng.choice(len(s_all), size=min(n, len(s_all)), replace=False)
    batch = make_batch(bank, s_all[pick], e_all[pick], model.config.window)
    levels = torch.as_tensor(model.config.quantiles, dtype=torch.float32)
    q, _, _ = model(batch["context"], batch["weekday_next"])
    losses = _losses(model, batch, levels)
    true = batch["demand"]
    pin_true = censored_pinball(q, true, torch.zeros_like(batch["censored"]), levels)
    coverage = (true.unsqueeze(-1) <= q).float().mean(dim=(0, 1)).numpy()
    model.train()
    out = {k: float(v) for k, v in losses.items()}
    out["pinball_true_demand"] = float(pin_true)
    out["coverage"] = [float(c) for c in coverage]
    return out


def reference_metrics(bank: SeriesBank, split: str, window: int = WINDOW, n: int = 20_000, seed: int = 1) -> dict:
    """True-demand pinball and coverage for the moving-average and oracle forecasters.

    Same anchors and the same scale as :func:`evaluate`, so the numbers line up.
    """
    from scipy.stats import norm

    from src.perfect_supplier.world import demand_mean_path, sample_world, uniform_half_width

    rng = np.random.default_rng(seed)
    s_all, e_all = bank.anchors(window)
    pick = rng.choice(len(s_all), size=min(n, len(s_all)), replace=False)
    s, e = s_all[pick], e_all[pick]
    batch = make_batch(bank, s, e, window)
    levels = torch.as_tensor(QUANTILES, dtype=torch.float32)
    true = batch["demand"]
    scale = batch["scale"].numpy()
    out = {}
    # Moving average: last 28 days of sales, normal, flat over the horizon.
    ctx = bank.sales[s[:, None], e[:, None] + np.arange(-27, 1)[None, :]]
    mu, sd = ctx.mean(axis=1), ctx.std(axis=1, ddof=1)
    q = np.clip(mu[:, None] + sd[:, None] * norm.ppf(QUANTILES)[None, :], 0, None) / scale
    q = torch.as_tensor(np.repeat(q[:, None, :], window, axis=1), dtype=torch.float32)
    out["moving_average"] = q
    # Oracle: the world's expected path and uniform noise.
    cache = {}
    oq = np.empty((len(s), window, len(QUANTILES)), np.float32)
    for i, (si, ei) in enumerate(zip(s, e)):
        ws = int(bank.world_seed[si])
        if ws not in cache:
            w = sample_world(ws, split)
            cache[ws] = (demand_mean_path(w), uniform_half_width(w.cv))
        mean, half = cache[ws]
        m = mean[ei + 1 : ei + 1 + window, bank.product[si]]
        oq[i] = m[:, None] * (1 - half + 2 * half * QUANTILES)[None, :] / scale[i]
    out["oracle"] = torch.as_tensor(oq)
    zeros = torch.zeros_like(batch["censored"])
    return {
        name: {
            "pinball_true_demand": float(censored_pinball(q, true, zeros, levels)),
            "coverage": [float(c) for c in (true.unsqueeze(-1) <= q).float().mean(dim=(0, 1)).numpy()],
        }
        for name, q in out.items()
    }


def fit(train: SeriesBank, val: SeriesBank, config: JepaConfig, log_path: str | Path | None = None) -> JepaModel:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = JepaModel(config)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.lr, total_steps=config.steps, pct_start=0.05)
    levels = torch.as_tensor(config.quantiles, dtype=torch.float32)
    s_all, e_all = train.anchors(config.window)
    history, best, best_state = [], math.inf, None
    for step in range(1, config.steps + 1):
        pick = rng.integers(0, len(s_all), config.batch_size)
        batch = make_batch(train, s_all[pick], e_all[pick], config.window)
        losses = _losses(model, batch, levels)
        opt.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        model.update_target()
        if step % config.eval_every == 0 or step == config.steps:
            metrics = evaluate(model, val)
            metrics["step"] = step
            metrics["train_total"] = float(losses["total"].detach())
            history.append(metrics)
            print(f"step {step} train {float(losses['total'].detach()):.4f} val_pinball {metrics['pinball']:.4f} "
                  f"val_true {metrics['pinball_true_demand']:.4f}", flush=True)
            if metrics["pinball"] < best:
                best = metrics["pinball"]
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps({"config": asdict(config), "history": history}, indent=1))
    model.eval()
    return model


def save(model: JepaModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state": model.state_dict()}, path)


def load(path: str | Path) -> JepaModel:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = JepaModel(JepaConfig(**blob["config"]))
    model.load_state_dict(blob["state"])
    model.eval()
    return model


class JepaForecaster:
    """Forecaster interface over a trained model; quantiles in demand units."""

    def __init__(self, model: JepaModel, name: str = "jepa"):
        self.model, self.name = model, name
        self.levels = np.asarray(model.config.quantiles)

    @torch.no_grad()
    def __call__(self, obs) -> np.ndarray:
        w = self.model.config.window
        sales = obs.sales[-w:].astype(np.float32).T  # (P, w)
        cens = obs.censored[-w:].astype(np.float32).T
        scale = sales.mean(axis=1, keepdims=True) + 1.0
        context = torch.as_tensor(np.stack([sales / scale, cens], axis=-1))
        weekday_next = torch.full((len(sales),), (obs.weekday + 1) % 7, dtype=torch.long)
        q, _, _ = self.model(context, weekday_next)
        return (q.numpy() * scale[:, :, None]).transpose(1, 0, 2)  # (H, P, Q)
