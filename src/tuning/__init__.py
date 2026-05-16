"""Public API for the policy hyperparameter tuning module.

Exported in this slice (issues 02–04):
    TuningConfig               — immutable study-configuration dataclass.
    evaluate_policy_normalised — single-policy CRN evaluator (no Optuna).
    order_up_to_space          — OrderUpToPolicy trial-callback factory.
    reorder_point_space        — ReorderPointPolicy trial-callback factory.
    periodic_order_up_to_space — PeriodicOrderUpToPolicy trial-callback factory.
    periodic_reorder_space     — PeriodicReorderPolicy trial-callback factory.
    run_study                  — Optuna study orchestration + artifact writer.

Later slices will add:
    confirm_top_k (issue 05), CLI entry point (issue 06).
"""

from src.tuning.config import TuningConfig
from src.tuning.evaluator import evaluate_policy_normalised
from src.tuning.search_spaces import (
    order_up_to_space,
    periodic_order_up_to_space,
    periodic_reorder_space,
    reorder_point_space,
)
from src.tuning.study import run_study

__all__ = [
    "TuningConfig",
    "evaluate_policy_normalised",
    "order_up_to_space",
    "reorder_point_space",
    "periodic_order_up_to_space",
    "periodic_reorder_space",
    "run_study",
]
