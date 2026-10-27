"""Train one learned forecaster (``jepa`` or ``direct``) for one training seed, as a logged run.

The behaviour log under ``root`` (``python -m src.perfect_supplier collect``)
must hold exactly the protocol's train block. The validation worlds for early
stopping are the same 10% for every seed, as in ``experiment.fit``; the seed
sets the initial weights and the batches. The run writes ``<arm>-s<seed>.pt``,
its training history, and ``<arm>-s<seed>.json``: an arm config (planner kind,
model path and hash, ``seed_group`` = the arm) without a variant, which the
tuner completes per variant.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.perfect_supplier import data
from src.perfect_supplier.bench import log
from src.perfect_supplier.bench.arms import model_spec
from src.perfect_supplier.bench.protocol import GateError, Protocol
from src.perfect_supplier.forecast import QUANTILES, TAIL_QUANTILES


def fit_model(
    root: str | Path,
    arm: str,
    seed: int,
    protocol: Protocol,
    track: str,
    out: str | Path,
    *,
    steps: int = 20_000,
    tail_quantiles: bool = True,
    val_fraction: float = 0.1,
    log_dir: str | Path = log.LOG_DIR,
) -> dict:
    import torch

    from src.perfect_supplier import jepa
    from src.perfect_supplier.experiment import _bank

    if arm not in ("jepa", "direct"):
        raise ValueError(f"unknown learned arm {arm!r}")
    root, out = Path(root), Path(out)
    summary = data.load_summary(root)
    seeds = sorted(summary.loc[summary["split"] == "train", "world_seed"].unique().tolist())
    block = protocol.seeds("train", protocol.raw["seed_blocks"]["train"]["size"])
    if seeds != block:
        raise GateError(f"{root}: behaviour log holds {len(seeds)} worlds, not the protocol's train block")
    rng = np.random.default_rng(0)
    val = set(rng.choice(seeds, size=max(1, int(len(seeds) * val_fraction)), replace=False).tolist())
    levels = TAIL_QUANTILES if tail_quantiles else QUANTILES
    config = jepa.JepaConfig(jepa=arm == "jepa", steps=steps, seed=seed, quantiles=tuple(levels.tolist()))
    stem = f"{arm}-s{seed}"
    entry = {"kind": "fit", "arm": stem, "world_family": "perfect_supplier", "split": "train",
             "seeds": {"first": seeds[0], "count": len(seeds), "val": len(val)}, "config": asdict(config)}
    with log.logged_run(track, entry, protocol, log_dir):
        torch.set_num_threads(protocol.workers())
        train_bank = _bank(root, [s for s in seeds if s not in val], "base")
        val_bank = _bank(root, sorted(val), "base")
        out.mkdir(parents=True, exist_ok=True)
        model = jepa.fit(train_bank, val_bank, config, log_path=out / f"{stem}_fit.json")
        jepa.save(model, out / f"{stem}.pt")
        history = json.loads((out / f"{stem}_fit.json").read_text())["history"]
        best = min(history, key=lambda h: h["pinball"])
        arm_config = {"name": f"{arm}_s{seed}", "kind": "planner", "params": {"forecaster": arm},
                      "model": model_spec(out / f"{stem}.pt"), "seed_group": arm}
        (out / f"{stem}.json").write_text(json.dumps(arm_config, indent=1))
        entry["result"] = {"metric": "val_censored_pinball", "best": best["pinball"], "step": best["step"],
                           "pinball_true_demand": best["pinball_true_demand"], "coverage": best["coverage"]}
        entry["artifacts"] = [{"path": str(p), "sha256": log.file_sha256(p)}
                              for p in (out / f"{stem}.pt", out / f"{stem}_fit.json", out / f"{stem}.json")]
    return arm_config
