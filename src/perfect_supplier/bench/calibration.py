"""Lead-time calibration of a planner arm's imagined demand.

The planner sums sampled daily demand into lead-time demand. This runs the arm
closed-loop, records the forecaster's quantiles every ``every`` days, draws the
planner's paths from them, and checks how often realised demand over the next
``lag`` and ``lag + 7`` days exceeds the empirical ``q``-quantile of the summed
paths. Calibrated paths exceed it at rate ``1 - q``; the PIT of the realised
sum is reported too (uniform when calibrated).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.perfect_supplier.bench import log
from src.perfect_supplier.bench.arms import make_policy
from src.perfect_supplier.bench.evaluate import DEFAULT_FAMILY
from src.perfect_supplier.bench.protocol import Protocol, config_sha256
from src.perfect_supplier.planner import sample_paths
from src.perfect_supplier.sim import simulate
from src.perfect_supplier.world import HISTORY_DAYS, demand_path, sample_world

LEVELS = (0.90, 0.95, 0.99)


class _Recorder:
    """Wraps the arm's forecaster; keeps its quantiles, or its own paths when it draws them."""

    def __init__(self, forecaster, every: int):
        self.forecaster, self.every, self.name = forecaster, every, forecaster.name
        self.seen: dict[int, np.ndarray] = {}
        self.draws_paths = hasattr(forecaster, "paths")
        if self.draws_paths:
            self.paths = self._paths

    def _paths(self, obs, n_paths, rng):
        p = self.forecaster.paths(obs, n_paths, rng)
        if obs.day % self.every == 0:
            self.seen[obs.day] = p
        return p

    def __call__(self, obs):
        q = self.forecaster(obs)
        if obs.day % self.every == 0:
            self.seen[obs.day] = q
        return q


def world_coverage(config: dict, seed: int, split: str, variant: str, every: int = 5,
                   family: str = DEFAULT_FAMILY, regimes: dict | None = None) -> list[dict]:
    world = sample_world(seed, split, variant=variant, family=family, regimes=regimes)
    policy = make_policy(config, world)
    if not hasattr(policy, "forecaster"):
        raise ValueError(f"arm {config['name']!r} has no forecaster to calibrate")
    policy.forecaster = recorder = _Recorder(policy.forecaster, every)
    simulate(world, policy)
    demand = demand_path(world)
    rng = np.random.default_rng([seed, 7])
    rows = []
    for day, q in recorder.seen.items():
        paths = q if recorder.draws_paths else sample_paths(q, policy.n_paths, rng)  # (M, H, P)
        M = len(paths)
        for span_name, span in (("lag", world.delivery_lag), ("lag+7", world.delivery_lag + 7)):
            if day + span >= world.n_days:
                continue
            sims = np.sort(paths[:, :span, :].sum(axis=1), axis=0)  # (M, P)
            real = demand[HISTORY_DAYS + day + 1 : HISTORY_DAYS + day + 1 + span].sum(axis=0)
            row = {"world_seed": seed, "day": day, "span": span_name,
                   "pit": ((sims < real).sum(axis=0) + 0.5 * (sims == real).sum(axis=0)) / M}
            for lv in LEVELS:
                pos = lv * (M - 1)
                lo = int(np.floor(pos))
                qv = sims[lo] + (pos - lo) * (sims[min(lo + 1, M - 1)] - sims[lo])
                row[f"exceed_{lv}"] = real > qv
            for p in range(world.n_products):
                rows.append({k: (v[p] if isinstance(v, np.ndarray) else v) for k, v in row.items()})
    return rows


def coverage_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Exceedance rate per span and level, against the nominal rate, plus PIT spread."""
    out = rows.groupby("span")[[f"exceed_{lv}" for lv in LEVELS]].mean()
    out.columns = [f"q{lv} exceed (nominal {1 - lv:.2f})" for lv in LEVELS]
    out["PIT sd (uniform 0.289)"] = rows.groupby("span")["pit"].std()
    out["n"] = rows.groupby("span").size()
    return out


def calibrate(config: dict, split: str, n_worlds: int, variant: str, protocol: Protocol, track: str,
              out: str | Path, *, offset: int = 0, every: int = 5, family: str = DEFAULT_FAMILY,
              log_dir: str | Path = log.LOG_DIR) -> pd.DataFrame:
    seeds = protocol.seeds(split, n_worlds, offset)
    if protocol.is_sealed(split):
        raise ValueError("calibration is a dev diagnostic; it never runs on the sealed split")
    out = Path(out)
    entry = {"kind": "diagnostic", "arm": config["name"], "world_family": family, "variant": variant,
             "split": split, "seeds": {"first": seeds[0], "count": len(seeds)},
             "config_sha256": config_sha256(config), "note": f"lead-time calibration, a decision every {every} days"}
    with log.logged_run(track, entry, protocol, log_dir):
        parts = Parallel(n_jobs=protocol.workers())(
            delayed(world_coverage)(config, s, split, variant, every, family, protocol.world_params(family))
            for s in seeds)
        rows = pd.DataFrame([r for p in parts for r in p])
        out.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(out / f"calibration-{config['name']}.parquet", index=False)
        table = coverage_table(rows)
        entry["result"] = {"metric": "lead_time_coverage", "table": table.round(4).to_dict(orient="index")}
        entry["artifacts"] = [{"path": str(out / f"calibration-{config['name']}.parquet"),
                               "sha256": log.file_sha256(out / f"calibration-{config['name']}.parquet")}]
    return table
