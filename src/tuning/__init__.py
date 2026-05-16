"""Public API for the policy hyperparameter tuning module.

Self-contained: no imports from ``src.rl``. The tuning module owns its own
``TuningConfig``, episode sampler, rollout primitives, and world loader.

Exported:
    TuningConfig               — immutable study-configuration dataclass.
    TuningEpisodeSpec          — pure-data description of one episode.
    sample_episode             — deterministic episode-spec builder.
    evaluate_policy_normalised — single-policy CRN evaluator.
    order_up_to_space          — OrderUpToPolicy trial-callback factory.
    reorder_point_space        — ReorderPointPolicy trial-callback factory.
    periodic_order_up_to_space — PeriodicOrderUpToPolicy trial-callback factory.
    periodic_reorder_space     — PeriodicReorderPolicy trial-callback factory.
    run_study                  — Optuna study orchestration + artifact writer.
    confirm_top_k              — Top-K re-evaluation on held-out seeds.
    load_world                 — Resolve catalog + base template from world cache.
"""

from src.tuning.config import TuningConfig
from src.tuning.episode import TuningEpisodeSpec, sample_episode
from src.tuning.evaluator import evaluate_policy_normalised
from src.tuning.search_spaces import (
    order_up_to_space,
    periodic_order_up_to_space,
    periodic_reorder_space,
    reorder_point_space,
)
from src.tuning.study import confirm_top_k, run_study
from src.tuning.world_loader import load_world

__all__ = [
    "TuningConfig",
    "TuningEpisodeSpec",
    "sample_episode",
    "evaluate_policy_normalised",
    "order_up_to_space",
    "reorder_point_space",
    "periodic_order_up_to_space",
    "periodic_reorder_space",
    "run_study",
    "confirm_top_k",
    "load_world",
]
