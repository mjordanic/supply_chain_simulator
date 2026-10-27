"""Backfill the results that predate the harness as ``kind: historical`` log lines.

Stored perfect-supplier runs are re-scored from a read-only snapshot of the
captain's ``runs/`` with the harness statistics (BCa intervals, cost
reduction). Results that live only in prose (the engine line, earlier
commits, the audit) are recorded as reported, with their source. None of
them is held-out: every eval world here is in a burnt block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.perfect_supplier import data
from src.perfect_supplier.bench import log, stats
from src.perfect_supplier.bench.arms import legacy_configs
from src.perfect_supplier.bench.protocol import Protocol
from src.perfect_supplier.bench.report import best_textbook

VARIANT_ROOTS = {"base": "perfect-supplier", "one_product": "perfect-supplier-one_product",
                 "no_fee": "perfect-supplier-no_fee"}
AUDIT = "firstmate data/sc-jepa-audit/report.md"

# Results recorded in prose only: (hypothesis, variant, arm, result, note).
REPORTED = [
    ("engine", None, "jepa_engine_v1", {"metric": "net_profit_over_initial_cash", "mean": 0.364, "vs": "tuned_order_up_to",
                                        "ref_mean": 3.031},
     "Engine line v1, first published run; both pass rules failed. Source: docs/paper/01-introduction.md."),
    ("engine", None, "jepa_engine_v2", {"metric": "net_profit_over_initial_cash", "mean": 0.381458,
                                        "vs": "tuned_order_up_to", "ref_mean": 1.021222, "diff": -0.639764,
                                        "ci95": [-1.185105, -0.184952], "n": 32},
     "Engine line v2 at holding rate 0.01; the learned buyer buys about 1.7% of the anchor's units. "
     "Source: docs/paper/03-results.md."),
    ("lean", "base", "jepa", {"metric": "margin_capture", "mean": 0.9592, "vs": "tuned_periodic_order_up_to",
                              "diff_pts": 2.43, "ci95_pts": [2.36, 2.49], "n": 500},
     "Lean line first result (aafc80f) against the engine rules; mostly the shared order fee. Superseded by cd804b8."),
    ("lean", "base", "jepa", {"metric": "margin_capture", "mean": 0.9592, "vs": "tuned_eoq_joint (fee split per product)",
                              "ref_mean": 0.9551, "diff_pts": 0.40, "ci95_pts": [0.37, 0.43], "n": 500},
     "Per-product-fee joint EOQ (cd804b8/0fa8c66). Superseded by the common EOQ cycle in b88e898 (+0.24)."),
    ("lean", "base", "jepa", {"metric": "margin_capture", "note": "service grid with 0.999 added: no change"},
     "ADR 0033 tail-sensitivity check (not kept in the repo). With 64 paths 0.995 and 0.999 are the same order "
     "statistic. Source: ADR 0033 via the audit section 3.2(c)."),
    ("audit", "base", "strata", {"metric": "margin_capture", "jepa_vs_tuned_eoq_joint_by_holding_rate_pts":
                                 [0.102, 0.216, 0.426], "cost_reduction": 0.056},
     f"Audit strata.py on the stored base eval; ratio-of-means cost reduction 5.6%. Source: {AUDIT} A.1."),
    ("audit", "base", "ctx_eoq_joint", {"metric": "margin_capture", "vs": "jepa", "diff_pts": -0.226,
                                        "ci95_pts": [-0.249, -0.202], "tuned_eoq_joint_minus_ctx_pts": -0.016,
                                        "budget": {"trials": 200, "worlds_per_trial": 64, "cells": 3}},
     f"Audit contextual_tune.py: joint EOQ tuned per holding rate, scored on the stored eval worlds. Source: {AUDIT} A.2."),
    ("audit", "no_fee", "ctx_eoq_joint", {"metric": "margin_capture", "vs": "jepa", "diff_pts": 0.182,
                                          "ci95_pts": [0.146, 0.219], "tuned_eoq_joint_minus_ctx_pts": -0.131,
                                          "budget": {"trials": 200, "worlds_per_trial": 64, "cells": 3}},
     f"Audit contextual_tune.py, no_fee. Source: {AUDIT} A.2."),
    ("audit", "one_product", "ctx_eoq_joint", {"metric": "margin_capture", "vs": "jepa", "diff_pts": -0.061,
                                               "ci95_pts": [-0.109, -0.012],
                                               "budget": {"trials": 200, "worlds_per_trial": 64, "cells": 3}},
     f"Audit contextual_tune.py, one_product. Source: {AUDIT} A.2."),
    ("audit", "base", "jepa", {"metric": "lead_time_exceedance", "lag+7": {"q0.99": 0.066}, "lag": {"q0.99": 0.040},
                               "oracle_q0.99": 0.02, "n_worlds": 60},
     f"Audit leadtime_calibration.py on stored JEPA eval trajectories; imagined lead-time demand too narrow. "
     f"Source: {AUDIT} 3.2(e)."),
    ("audit", "base", "jepa_rho0.0", {"metric": "margin_capture", "vs": "tuned_eoq_joint", "diff_pts": 0.263,
                                      "ci95_pts": [0.226, 0.301], "n": 200, "seeds": [3200000, 3200199],
                                      "rho_star": 0.0, "seasonal_minus_tuned_eoq_joint_pts": -0.018},
     f"Audit copula_planner.py, base: coherent paths do not help (rho* = 0). Source: {AUDIT} A.4."),
    ("audit", "no_fee", "jepa_rho0.75", {"metric": "margin_capture", "vs": "tuned_eoq_joint", "diff_pts": 0.137,
                                         "ci95_pts": [0.097, 0.179], "n": 200, "seeds": [3200000, 3200199],
                                         "vs_ctx_eoq_joint_pts": 0.040, "vs_jepa_rho0_pts": 0.211},
     f"Audit copula_planner_v.py, no_fee: tuned path correlation turns a loss into a lead. Source: {AUDIT} A.4."),
]


def scores_from_summary(root: str | Path, split: str = "eval") -> pd.DataFrame:
    """A stored run's summary in the harness ``scores.parquet`` shape."""
    s = data.load_summary(root)
    return s[s["split"] == split].rename(columns={"method": "arm"})


def backfill(snapshot: str | Path, protocol: Protocol, track: str = "historical",
             log_dir: str | Path = log.LOG_DIR, n_resamples: int | None = None) -> list[dict]:
    snapshot = Path(snapshot)
    n_resamples = n_resamples or protocol.stats["resamples"]
    rows = []
    for variant, sub in VARIANT_ROOTS.items():
        root = snapshot / sub
        frame = scores_from_summary(root)
        m = frame.pivot(index="world_seed", columns="arm", values="score")
        ref = best_textbook(legacy_configs(root, variant, snapshot / "perfect-supplier"))
        for arm in ("jepa", "direct", "planner_moving_average", "planner_oracle"):
            c = stats.compare(m[arm].to_numpy(), m[ref].to_numpy(), (arm, ref), protocol.sesoi,
                              n_resamples=n_resamples)
            rows.append(log.append(track, {
                "kind": "historical", "hypothesis": "lean", "arm": arm, "world_family": "perfect_supplier",
                "variant": variant, "split": "eval (burnt)", "seeds": {"first": int(m.index.min()), "count": len(m)},
                "budget": {"textbook": {"trials": 200, "worlds_per_trial": 64}, "learned": {"trials": 0}},
                "result": {"metric": "margin_capture", "mean": float(m[arm].mean()), "vs": ref,
                           "ref_mean": float(m[ref].mean()), "diff_pts": 100 * c.diff,
                           "ci95_pts": [100 * c.ci[0], 100 * c.ci[1]], "ci_method": "BCa",
                           "cost_reduction": c.cost_reduction, "cost_reduction_ci95": list(c.cost_reduction_ci),
                           "win_rate": c.win_rate, "p_wilcoxon": c.p_wilcoxon, "n": c.n},
                "artifacts": [{"path": f"captain runs/{sub}/summary"}],
                "status": "kept",
                "note": "Lean line b88e898 stored eval (common-cycle joint EOQ), re-scored by the harness.",
            }, protocol, log_dir))
    pre = snapshot / "perfect-supplier" / "tuning_pre_review" / "best_params.json"
    rows.append(log.append(track, {
        "kind": "historical", "hypothesis": "lean", "arm": "tuned_textbook (pre-review)", "variant": "base",
        "split": "tune", "result": {"families": sorted(json.loads(pre.read_text()))},
        "artifacts": [{"path": "captain runs/perfect-supplier/tuning_pre_review/", "sha256": log.file_sha256(pre)}],
        "status": "superseded",
        "note": "Textbook tuning before the cd804b8 review fixes; engine rules moved under 0.2 pts after re-tuning.",
    }, protocol, log_dir))
    dev = scores_from_summary(snapshot / "perfect-supplier-dev")
    rows.append(log.append(track, {
        "kind": "historical", "hypothesis": "lean", "arm": "dev-run", "variant": "base", "split": "eval (burnt)",
        "seeds": {"first": int(dev["world_seed"].min()), "count": int(dev["world_seed"].nunique())},
        "result": {"metric": "margin_capture", "means": dev.groupby("arm")["score"].mean().round(4).to_dict()},
        "artifacts": [{"path": "captain runs/perfect-supplier-dev/"}], "status": "abandoned",
        "note": "Early 16-world development run with its own forecaster fit (runs/perfect-supplier-dev/models).",
    }, protocol, log_dir))
    for hypothesis, variant, arm, result, note in REPORTED:
        rows.append(log.append(track, {"kind": "historical", "hypothesis": hypothesis, "arm": arm, "variant": variant,
                                       "result": result, "status": "superseded" if "Superseded" in note else "kept",
                                       "note": note}, protocol, log_dir))
    return rows
