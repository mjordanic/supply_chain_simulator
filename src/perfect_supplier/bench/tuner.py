"""Budgeted tuner: sequential TPE, each trial's worlds on a process pool.

Optuna's thread mode is GIL-bound on these episodes, so trials run one after
another (seeded TPE, reproducible) and the worlds inside a trial run on a
joblib process pool that stays up for the whole study. The trial count and
worlds per trial come from the protocol budget of the arm's kind; the search
space from its ``search_spaces`` entry. With ``cell_keys`` every cell of those
world fields is tuned on its own tune worlds and the result is a contextual
arm: a textbook arm gets its full budget in every cell, a learned arm spreads
its budget over the cells, so it never spends more than the learned budget.

:func:`select` scores tuned textbook arms on one common block of tune worlds
that no tuner used, so the reference is chosen on equal footing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from src.perfect_supplier.bench import log
from src.perfect_supplier.bench.arms import cell_of
from src.perfect_supplier.bench.evaluate import episode, run_episodes
from src.perfect_supplier.bench.protocol import LEARNED_KINDS, Protocol, config_sha256
from src.perfect_supplier.world import sample_world

def suggest(trial: optuna.Trial, space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        if spec[0] == "float":
            out[name] = trial.suggest_float(name, spec[1], spec[2], log=len(spec) > 3 and spec[3] == "log")
        elif spec[0] == "int":
            out[name] = trial.suggest_int(name, spec[1], spec[2])
        elif spec[0] == "categorical":
            out[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"unknown search type {spec[0]!r} for {name}")
    return out


def tune_worlds(protocol: Protocol, n: int, variant: str, cell_keys: list[str] | None) -> dict[str, list[int]]:
    """Tune seeds per cell, in block order; one cell ``"all"`` without ``cell_keys``."""
    if not cell_keys:
        return {"all": protocol.seeds("tune", n)}
    block = protocol.raw["seed_blocks"]["tune"]["size"]
    cells: dict[str, list[int]] = {}
    for seed in protocol.seeds("tune", block):
        c = cells.setdefault(cell_of(sample_world(seed, "tune", variant=variant), cell_keys), [])
        if len(c) < n:
            c.append(seed)
    short = {c: len(s) for c, s in cells.items() if len(s) < n}
    if short:
        raise ValueError(f"tune block too small for {n} worlds per cell: {short}")
    return dict(sorted(cells.items()))


def tune(
    kind: str,
    name: str,
    variant: str,
    protocol: Protocol,
    track: str,
    out: str | Path,
    *,
    cell_keys: list[str] | None = None,
    base_config: dict | None = None,
    log_dir: str | Path = log.LOG_DIR,
) -> dict:
    """Tune one arm; write ``<name>.json`` (the arm config) and the trial tables under ``out``."""
    budget = protocol.budget(kind)
    space = protocol.search_space(kind)
    base = {"params": {}, **(base_config or {}), "name": name, "kind": kind, "variant": variant}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cells = tune_worlds(protocol, budget["worlds_per_trial"], variant, cell_keys)
    n_trials = budget["trials"] // len(cells) if kind in LEARNED_KINDS else budget["trials"]
    entry = {"kind": "tune", "arm": name, "world_family": "perfect_supplier", "variant": variant, "split": "tune",
             "config": {"kind": kind, "cell_keys": cell_keys, "space": space, "base": base},
             "seeds": {c: {"first": s[0], "count": len(s)} for c, s in cells.items()},
             "budget": {**budget, "trials_per_cell": n_trials,
                        "episodes": n_trials * budget["worlds_per_trial"] * len(cells), "cells": len(cells)}}
    best: dict[str, dict] = {}
    with log.logged_run(track, entry, protocol, log_dir), Parallel(n_jobs=protocol.workers()) as pool:
        for i, (cell, seeds) in enumerate(cells.items()):

            def objective(trial, seeds=seeds):
                config = {**base, "params": {**base["params"], **suggest(trial, space)}}
                scores = pool(delayed(episode)(config, s, "tune", variant) for s in seeds)
                return float(np.mean([r["score"] for r in scores]))

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
            trials = study.trials_dataframe()
            trials["best_so_far"] = trials["value"].cummax()  # the tuning curve
            trials["cell"] = cell
            tag = "" if cell == "all" else f"-cell{i}"
            trials.to_parquet(out / f"{name}{tag}-trials.parquet", index=False)
            best[cell] = {"params": study.best_params, "tune_score": study.best_value}
            print(f"{name} {cell}: {study.best_value:.4f} {study.best_params}", flush=True)
        if cell_keys:
            config = {**base, "cell_keys": cell_keys,
                      "cells": {c: {**base["params"], **b["params"]} for c, b in best.items()},
                      "tune_score": float(np.mean([b["tune_score"] for b in best.values()]))}
        else:
            config = {**base, "params": {**base["params"], **best["all"]["params"]},
                      "tune_score": best["all"]["tune_score"]}
        config["budget"] = entry["budget"]
        (out / f"{name}.json").write_text(json.dumps(config, indent=1))
        entry["config_sha256"] = config_sha256(config)
        entry["result"] = {"metric": "margin_capture", "tune_score": config["tune_score"],
                           "cells": {c: b["tune_score"] for c, b in best.items()}}
        entry["artifacts"] = [{"path": str(p), "sha256": log.file_sha256(p)} for p in sorted(out.glob(f"{name}*"))]
    return config


def select(
    configs: list[dict],
    protocol: Protocol,
    track: str,
    variant: str,
    *,
    offset: int,
    n_worlds: int,
    log_dir: str | Path = log.LOG_DIR,
) -> list[dict]:
    """Each config with ``select_score``: its mean score on ``n_worlds`` tune worlds from ``offset``.

    Tune scores come from each arm's own tuning worlds (per cell for a
    contextual arm), so they do not compare across arms; the report picks
    the reference by ``select_score`` when every candidate has one. Pick an
    ``offset`` past every seed a tuner used.
    """
    seeds = protocol.seeds("tune", n_worlds, offset)
    entry = {"kind": "select", "arms": [c["name"] for c in configs], "world_family": "perfect_supplier",
             "variant": variant, "split": "tune", "seeds": {"first": seeds[0], "count": len(seeds)},
             "config": {c["name"]: config_sha256(c) for c in configs}}
    with log.logged_run(track, entry, protocol, log_dir):
        means = run_episodes(configs, seeds, "tune", variant, protocol.workers()).groupby("arm")["score"].mean()
        out = [{**c, "select_score": float(means[c["name"]]),
                "select": {"split": "tune", "first": seeds[0], "count": len(seeds)}} for c in configs]
        entry["result"] = {"metric": "margin_capture", "select_scores": {c["name"]: c["select_score"] for c in out}}
    return out


def tuning_curve(out: str | Path, name: str) -> pd.DataFrame:
    """Best tune score so far by trial, one column per cell."""
    frames = [pd.read_parquet(p) for p in sorted(Path(out).glob(f"{name}*-trials.parquet"))]
    t = pd.concat(frames)
    return t.pivot(index="number", columns="cell", values="best_so_far")
