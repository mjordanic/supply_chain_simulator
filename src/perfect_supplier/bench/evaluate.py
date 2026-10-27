"""Paired evaluation: every arm closed-loop on its own copy of the same worlds.

:func:`evaluate` runs each (arm, world) episode on a process pool and writes
``scores.parquet`` (one row per world and arm: the score, the cost breakdown,
and the world's parameters) plus ``arms.json`` (the configs) under ``out``.
Worlds come from :meth:`Protocol.seeds`, and a sealed split passes
:func:`check_test_gate` first. The run is one log line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from src.perfect_supplier.bench import log
from src.perfect_supplier.bench.arms import make_policy
from src.perfect_supplier.bench.protocol import Protocol, arm_sha256, check_test_gate, config_sha256
from src.perfect_supplier.sim import simulate
from src.perfect_supplier.world import sample_world

WORLD_COLUMNS = ["world_seed", "variant", "delivery_lag", "holding_rate", "cv", "weekly_amp", "order_fee"]
SUMMARY_COLUMNS = ["score", "total_demand", "lost_units", "fees", "holding_cost", "order_days",
                   "fill_rate_demand", "mean_on_hand", "price", "unit_cost"]


def episode(config: dict, seed: int, split: str, variant: str) -> dict:
    world = sample_world(seed, split, variant=variant)
    summary = simulate(world, make_policy(config, world)).summary
    return {"arm": config["name"], "split": split, **{k: summary[k] for k in WORLD_COLUMNS + SUMMARY_COLUMNS}}


def run_episodes(configs: list[dict], seeds: list[int], split: str, variant: str, workers: int) -> pd.DataFrame:
    tasks = [(c, s) for c in configs for s in seeds]
    rows = Parallel(n_jobs=workers)(delayed(episode)(c, s, split, variant) for c, s in tasks)
    return pd.DataFrame(rows)


def evaluate(
    configs: list[dict],
    split: str,
    n_worlds: int,
    variant: str,
    out: str | Path,
    protocol: Protocol,
    track: str,
    *,
    offset: int = 0,
    rerun_reason: str | None = None,
    note: str = "",
    log_dir: str | Path = log.LOG_DIR,
    frozen_path: str | Path | None = None,
) -> pd.DataFrame:
    seeds = protocol.seeds(split, n_worlds, offset)
    kind = "test-eval" if protocol.is_sealed(split) else f"{split}-eval"
    gate_kw = {"frozen_path": frozen_path} if frozen_path else {}
    previous = [r for r in log.read(log_dir=log_dir) if r.get("kind") == "test-eval"]
    for config in configs:
        if config.get("variant", variant) != variant:
            raise ValueError(f"arm {config['name']!r} was configured for variant {config['variant']!r}, not {variant!r}")
        try:
            check_test_gate(protocol, split, [config], variant, previous, rerun_reason, **gate_kw)
        except Exception as exc:
            log.append(track, {"kind": kind, "arm": config["name"], "variant": variant, "split": split,
                               "config_sha256": config_sha256(config), "status": "refused", "note": str(exc)},
                       protocol, log_dir)
            raise
    out = Path(out)
    entry = {
        "kind": kind, "arms": [c["name"] for c in configs], "world_family": "perfect_supplier",
        "variant": variant, "split": split, "seeds": {"first": seeds[0], "count": len(seeds)},
        "config": {c["name"]: config_sha256(c) for c in configs},
        "arm_sha256": {c["name"]: arm_sha256(c) for c in configs}, "rerun_reason": rerun_reason, "note": note,
    }
    with log.logged_run(track, entry, protocol, log_dir):
        frame = run_episodes(configs, seeds, split, variant, protocol.workers())
        out.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out / "scores.parquet", index=False)
        (out / "arms.json").write_text(json.dumps(configs, indent=1))
        means = frame.groupby("arm")["score"].mean()
        entry["result"] = {"metric": "margin_capture", "means": means.round(6).to_dict(), "n": len(seeds)}
        entry["artifacts"] = [{"path": str(out / f), "sha256": log.file_sha256(out / f)}
                              for f in ("scores.parquet", "arms.json")]
    return frame


def score_matrix(frame: pd.DataFrame, metric: str = "score") -> pd.DataFrame:
    """Worlds x arms; fails if an arm misses a world."""
    m = frame.pivot(index="world_seed", columns="arm", values=metric)
    if m.isna().any().any():
        raise ValueError(f"arms missing worlds: {m.isna().sum()[m.isna().sum() > 0].to_dict()}")
    return m
