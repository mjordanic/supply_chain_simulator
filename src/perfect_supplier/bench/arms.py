"""Arm registry: one JSON-able config per arm, turned into a policy per world.

A config is ``{"name", "kind", "variant", "params", ...}``:

- textbook kinds: the four engine rules (``order_up_to``, ``reorder_point``,
  ``periodic_order_up_to``, ``periodic_reorder``), ``eoq``, ``eoq_joint``, and
  ``param_eoq_joint`` (joint EOQ whose ``z`` is a parametric map of the
  critical ratio). Adding ``"cell_keys"`` and ``"cells"`` makes any of them a
  per-cell contextual arm: the params of the world's cell are used.
- ``never_order``.
- ``planner``: the exact-recursion planner with ``params["forecaster"]`` one of
  ``oracle``, ``moving_average``, ``jepa``, ``direct``. ``jepa`` and ``direct``
  name a model file in ``config["model"]`` (path and sha256).
- ``seasonal_ets``: the planner with :class:`SeasonalEtsForecaster`.

Hashing a config (:func:`~src.perfect_supplier.bench.protocol.config_sha256`)
identifies the arm for the test gate and the log.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.perfect_supplier.experiment import NeverOrder
from src.perfect_supplier.forecast import HORIZON, QUANTILES, MovingAverageForecaster, OracleForecaster
from src.perfect_supplier.planner import Planner
from src.perfect_supplier.textbook import FAMILIES, EoqParams, EoqPolicy, TextbookParams, TextbookPolicy
from src.perfect_supplier.world import World

TEXTBOOK_KINDS = (*FAMILIES, "eoq", "eoq_joint", "param_eoq_joint")
KINDS = (*TEXTBOOK_KINDS, "never_order", "planner", "seasonal_ets")


class SeasonalEtsForecaster:
    """Additive seasonal exponential smoothing, ETS(A,N,A) with a 7-day season, on sales.

    Level and weekday terms start from the first 28 days of history and are
    smoothed with ``alpha`` and ``gamma``. The ``h``-step quantiles are normal
    around the point forecast with sd ``sigma * sqrt(1 + (h - 1) * alpha**2)``,
    ``sigma`` the one-step residual sd over the last 28 days. A minimal
    classical arm; its knobs are tuned like any other arm's.
    """

    name = "seasonal_ets"

    def __init__(self, alpha: float = 0.1, gamma: float = 0.1, horizon: int = HORIZON):
        self.alpha, self.gamma, self.horizon = alpha, gamma, horizon

    def __call__(self, obs) -> np.ndarray:
        y = obs.sales.astype(float)
        n = len(y)
        level = y[:28].mean(axis=0)
        season = np.stack([y[:28][np.arange(28) % 7 == k].mean(axis=0) - level for k in range(7)])
        resid = np.empty_like(y)
        for t in range(n):
            e = y[t] - level - season[t % 7]
            resid[t] = e
            level = level + self.alpha * e
            season[t % 7] = season[t % 7] + self.gamma * e
        sigma = resid[-28:].std(axis=0, ddof=1)
        h = np.arange(self.horizon)
        mu = level[None, :] + season[(n + h) % 7]  # (H, P)
        sd = sigma[None, :] * np.sqrt(1 + h * self.alpha**2)[:, None]
        q = mu[:, :, None] + sd[:, :, None] * norm.ppf(QUANTILES)[None, None, :]
        return np.clip(q, 0, None)


def cell_of(world: World, keys: list[str]) -> str:
    return str(tuple(round(float(getattr(world, k)), 6) for k in keys))


def _eoq_params(kind: str, params: dict, world: World) -> EoqParams:
    p = dict(params)
    if kind == "param_eoq_joint":
        m = world.price - world.unit_cost
        h = world.holding_rate * world.unit_cost
        p["service_z"] = p.pop("z_a") + p.pop("z_b") * float(norm.ppf(m / (m + h)))
    return EoqParams(**p, joint=kind != "eoq", family="eoq" if kind == "eoq" else "eoq_joint")


_MODELS: dict[str, object] = {}


def _model(spec: dict):
    """Load a model once per process; refuse a file whose hash changed."""
    path = spec["path"]
    if path not in _MODELS:
        import torch

        from src.perfect_supplier import jepa
        from src.perfect_supplier.bench.log import file_sha256

        if spec.get("sha256") and file_sha256(path) != spec["sha256"]:
            raise ValueError(f"{path}: sha256 does not match the arm config")
        torch.set_num_threads(1)
        _MODELS[path] = jepa.load(path)
    return _MODELS[path]


def make_policy(config: dict, world: World):
    kind, name = config["kind"], config["name"]
    params = dict(config.get("params", {}))
    if "cell_keys" in config:
        params = dict(config["cells"][cell_of(world, config["cell_keys"])])
    if kind == "never_order":
        return NeverOrder()
    if kind in FAMILIES:
        return TextbookPolicy(TextbookParams(family=kind, **params), name=name)
    if kind in ("eoq", "eoq_joint", "param_eoq_joint"):
        return EoqPolicy(_eoq_params(kind, params, world), name=name)
    if kind == "seasonal_ets":
        n_paths = params.pop("n_paths", 64)
        return Planner(SeasonalEtsForecaster(**params), name=name, n_paths=n_paths)
    if kind == "planner":
        fc = params.pop("forecaster")
        if fc == "oracle":
            forecaster = OracleForecaster(world)
        elif fc == "moving_average":
            forecaster = MovingAverageForecaster()
        elif fc in ("jepa", "direct"):
            from src.perfect_supplier.jepa import JepaForecaster

            forecaster = JepaForecaster(_model(config["model"]), name=fc)
        else:
            raise ValueError(f"unknown forecaster {fc!r}")
        return Planner(forecaster, name=name, **params)
    raise ValueError(f"unknown arm kind {kind!r}")


def model_spec(path: str | Path) -> dict:
    from src.perfect_supplier.bench.log import file_sha256

    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def legacy_configs(root: str | Path, variant: str, model_root: str | Path | None = None) -> list[dict]:
    """Arm configs for the stored arms of a ``runs/perfect-supplier*`` root.

    Textbook params come from ``tuning/best_params.json`` (with their tune
    scores); ``jepa`` and ``direct`` from ``models/`` under ``model_root``.
    """
    root = Path(root)
    model_root = Path(model_root or root)
    out = []
    best = json.loads((root / "tuning" / "best_params.json").read_text())
    scores = json.loads((root / "tuning" / "best_scores.json").read_text())
    for family, p in best.items():
        p = {k: v for k, v in p.items() if k not in ("family", "joint")}
        out.append({"name": f"tuned_{family}", "kind": family, "variant": variant, "params": p,
                    "tune_score": scores[family]["tune_score"],
                    "budget": {"trials": scores[family]["n_trials"], "worlds_per_trial": scores[family]["n_worlds"]}})
    for fc in ("jepa", "direct"):
        out.append({"name": fc, "kind": "planner", "variant": variant, "params": {"forecaster": fc},
                    "model": model_spec(model_root / "models" / f"{fc}.pt")})
    for fc in ("oracle", "moving_average"):
        out.append({"name": f"planner_{fc}", "kind": "planner", "variant": variant, "params": {"forecaster": fc}})
    out.append({"name": "never_order", "kind": "never_order", "variant": variant, "params": {}})
    return out
