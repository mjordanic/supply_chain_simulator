"""Public API for the policy hyperparameter tuning module.

Exported in this slice (issues 02–03):
    TuningConfig               — immutable study-configuration dataclass.
    evaluate_policy_normalised — single-policy CRN evaluator (no Optuna).
    order_up_to_space          — OrderUpToPolicy trial-callback factory.
    reorder_point_space        — ReorderPointPolicy trial-callback factory.
    periodic_order_up_to_space — PeriodicOrderUpToPolicy trial-callback factory.
    periodic_reorder_space     — PeriodicReorderPolicy trial-callback factory.

Later slices will add:
    study exports (issues 04/05): run_study, confirm_top_k.
"""

from src.tuning.config import TuningConfig
from src.tuning.evaluator import evaluate_policy_normalised
from src.tuning.search_spaces import (
    order_up_to_space,
    periodic_order_up_to_space,
    periodic_reorder_space,
    reorder_point_space,
)

__all__ = [
    "TuningConfig",
    "evaluate_policy_normalised",
    "order_up_to_space",
    "reorder_point_space",
    "periodic_order_up_to_space",
    "periodic_reorder_space",
]
