"""Public API for the policy hyperparameter tuning module.

Self-contained: no imports from ``src.rl``. The tuning module owns its own
``TuningConfig``, episode sampler (Config-adapter over sim), rollout
primitives, and world loader.

Exported:
    TuningConfig                     — immutable study-configuration dataclass.
    TuningEpisodeSpec                — graph-mode episode descriptor (capacity + balance).
    EpisodeSpec                      — backward-compatible alias for TuningEpisodeSpec.
    sample_episode                   — deterministic episode-spec builder.
    load_catalog_and_market_from_setup — load catalog + market from a setup directory.
    make_synthetic_catalog           — build a synthetic n-product catalog (no LLM).
    evaluate_policy_normalised       — single-policy CRN evaluator.
    order_up_to_space                — OrderUpToPolicy trial-callback factory.
    reorder_point_space              — ReorderPointPolicy trial-callback factory.
    periodic_order_up_to_space       — PeriodicOrderUpToPolicy trial-callback factory.
    periodic_reorder_space           — PeriodicReorderPolicy trial-callback factory.
    run_study                        — Optuna study orchestration + artifact writer.
    confirm_top_k                    — Top-K re-evaluation on held-out seeds.
    load_world                       — Resolve catalog + base template from world cache.
"""

from src.tuning.config import TuningConfig
from src.tuning.episode import (
    EpisodeSpec,
    TuningEpisodeSpec,
    load_catalog_and_market_from_setup,
    make_synthetic_catalog,
    sample_episode,
)
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
    "EpisodeSpec",
    "sample_episode",
    "load_catalog_and_market_from_setup",
    "make_synthetic_catalog",
    "evaluate_policy_normalised",
    "order_up_to_space",
    "reorder_point_space",
    "periodic_order_up_to_space",
    "periodic_reorder_space",
    "run_study",
    "confirm_top_k",
    "load_world",
]
