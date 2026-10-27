"""Paired statistics for arm comparisons. The unit of analysis is the world.

- :func:`bca_ci`: bias-corrected and accelerated bootstrap interval for any
  vectorised statistic of one or more paired arrays (Efron 1987).
- :func:`wilcoxon_p`: two-sided Wilcoxon signed-rank p-value.
- :func:`holm` and :func:`bh`: adjusted p-values (Holm 1979; Benjamini and
  Hochberg 1995).
- :func:`tost`: two one-sided paired t-tests against ``+-sesoi`` (Lakens 2017).
- :func:`two_level_ci`: worlds and training seeds both resampled.
- :func:`compare` and :func:`decide`: one comparison and its protocol label.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


def mean_stat(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=-1)


def cost_reduction_stat(arm: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``1 - mean(1 - s_arm) / mean(1 - s_ref)``: the share of the reference's lost margin the arm removes."""
    return 1 - (1 - arm).mean(axis=-1) / (1 - ref).mean(axis=-1)


def bca_ci(
    data: np.ndarray | Sequence[np.ndarray],
    statistic: Callable[..., np.ndarray] = mean_stat,
    *,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    batch: int = 1000,
) -> tuple[float, float]:
    """BCa interval of ``statistic`` over paired samples.

    ``data`` is one array or a sequence of equal-length arrays resampled
    together (pairs stay paired). ``statistic(*arrays)`` must reduce the last
    axis, so it scores a whole ``(B, n)`` batch of resamples at once.
    """
    arrays = [np.asarray(data, float)] if isinstance(data, np.ndarray) else [np.asarray(a, float) for a in data]
    n = len(arrays[0])
    theta = float(statistic(*arrays))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples)
    for start in range(0, n_resamples, batch):
        idx = rng.integers(0, n, size=(min(batch, n_resamples - start), n))
        boot[start : start + len(idx)] = statistic(*(a[idx] for a in arrays))
    # Bias correction; ties count half so a degenerate sample does not run off to infinity.
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / n_resamples
    z0 = sps.norm.ppf(np.clip(prop, 1 / n_resamples, 1 - 1 / n_resamples))
    # Acceleration from the jackknife.
    loo = np.array([np.delete(np.arange(n), i) for i in range(n)]) if n <= 2000 else None
    if loo is not None:
        jack = statistic(*(a[loo] for a in arrays))
    else:
        jack = np.array([float(statistic(*(np.delete(a, i) for a in arrays))) for i in range(n)])
    d = jack.mean() - jack
    denom = 6 * (d**2).sum() ** 1.5
    a = (d**3).sum() / denom if denom > 0 else 0.0
    alpha = (1 - level) / 2
    z = sps.norm.ppf([alpha, 1 - alpha])
    adj = sps.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boot, adj)
    return float(lo), float(hi)


def wilcoxon_p(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    if not np.any(d != 0):
        return 1.0
    return float(sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def ttest_p(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    if d.std() == 0:
        return 1.0 if d.mean() == 0 else 0.0
    return float(sps.ttest_1samp(d, 0.0).pvalue)


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def bh(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (q-values), in the input order."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def tost(d: np.ndarray, sesoi: float) -> float:
    """TOST p-value for ``|mean(d)| < sesoi`` (paired t); equivalence when below alpha."""
    d = np.asarray(d, float)
    n, se = len(d), d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        return 0.0 if abs(d.mean()) < sesoi else 1.0
    p_low = sps.t.sf((d.mean() + sesoi) / se, n - 1)
    p_high = sps.t.cdf((d.mean() - sesoi) / se, n - 1)
    return float(max(p_low, p_high))


def two_level_ci(
    m: np.ndarray, *, level: float = 0.95, n_resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile interval of a ``(worlds, seeds)`` matrix of paired differences.

    Each resample draws worlds with replacement and, independently, training
    seeds with replacement, then averages the drawn cells. Training-seed
    variance therefore widens the interval; with one seed it reduces to the
    world bootstrap. Returns ``(mean, low, high)``.
    """
    m = np.asarray(m, float)
    n, k = m.shape
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples)
    for b in range(n_resamples):
        boot[b] = m[np.ix_(rng.integers(0, n, n), rng.integers(0, k, k))].mean()
    alpha = (1 - level) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return float(m.mean()), float(lo), float(hi)


def cvar(d: np.ndarray, frac: float = 0.10) -> float:
    """Mean of the worst ``frac`` of per-world differences."""
    d = np.sort(np.asarray(d, float))
    return float(d[: max(1, int(np.ceil(frac * len(d))))].mean())


@dataclass
class Comparison:
    arm: str
    ref: str
    n: int
    diff: float              # mean paired difference (score units)
    ci: tuple[float, float]  # BCa at the protocol level
    ci90: tuple[float, float]
    win_rate: float
    p_t: float
    p_wilcoxon: float
    p_tost: float
    cost_reduction: float
    cost_reduction_ci: tuple[float, float]
    cvar10: float
    p_adj: float = float("nan")
    ci_adj: tuple[float, float] | None = None
    label: str = ""


def compare(arm: np.ndarray, ref: np.ndarray, names: tuple[str, str], sesoi: float, *,
            level: float = 0.95, n_resamples: int = 10_000, seed: int = 0) -> Comparison:
    arm, ref = np.asarray(arm, float), np.asarray(ref, float)
    d = arm - ref
    return Comparison(
        arm=names[0], ref=names[1], n=len(d), diff=float(d.mean()),
        ci=bca_ci(d, level=level, n_resamples=n_resamples, seed=seed),
        ci90=bca_ci(d, level=0.90, n_resamples=n_resamples, seed=seed),
        win_rate=float((d > 0).mean()), p_t=ttest_p(d), p_wilcoxon=wilcoxon_p(d), p_tost=tost(d, sesoi),
        cost_reduction=float(cost_reduction_stat(arm, ref)),
        cost_reduction_ci=bca_ci([arm, ref], cost_reduction_stat, level=level, n_resamples=n_resamples, seed=seed),
        cvar10=cvar(d),
    )


def decide(family: list[Comparison], diffs: list[np.ndarray], sesoi: float, *, alpha: float = 0.05,
           n_resamples: int = 10_000, seed: int = 0) -> list[Comparison]:
    """Holm across ``family`` and the protocol label of each comparison.

    The k-th smallest p-value (paired t) gets a BCa interval at level
    ``1 - alpha / (m - k + 1)``, Holm's step-down level; the step-down stops
    at the first comparison whose interval covers 0.

    - superior: adjusted lower bound > 0
    - practically superior: superior and the estimate is at least the SESOI
    - equivalent: the 90% interval lies inside +-SESOI (TOST)
    - inferior: adjusted upper bound < 0
    - inconclusive: anything else
    """
    m = len(family)
    p_adj = holm([c.p_t for c in family])
    order = np.argsort([c.p_t for c in family], kind="stable")
    stopped = False
    for k, i in enumerate(order):
        c = family[i]
        c.p_adj = float(p_adj[i])
        c.ci_adj = bca_ci(diffs[i], level=1 - alpha / (m - k), n_resamples=n_resamples, seed=seed)
        lo, hi = c.ci_adj
        rejected = not stopped and (lo > 0 or hi < 0)
        stopped = stopped or not rejected
        if rejected and lo > 0:
            c.label = "practically superior" if c.diff >= sesoi else "superior"
        elif rejected and hi < 0:
            c.label = "inferior"
        elif -sesoi < c.ci90[0] and c.ci90[1] < sesoi:
            c.label = "equivalent"
        else:
            c.label = "inconclusive"
    return family
