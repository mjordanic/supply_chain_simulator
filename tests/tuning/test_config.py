"""Tests for TuningConfig dataclass.

Coverage:
  - test_tuning_config_defaults: documented defaults exist with documented types.
  - test_seed_offsets_are_disjoint: TuningConfig.seed_offset and
    holdout_seed_offset are disjoint from RLConfig.eval_seed_offset and
    from the two-scale eval ranges.
"""

from __future__ import annotations

import pytest

from src.tuning.config import TuningConfig
from src.rl.configs.default import RLConfig


class TestTuningConfigDefaults:
    """Sanity check the documented defaults exist with the documented types."""

    def test_tuning_config_defaults(self):
        """TuningConfig() constructs with all documented defaults."""
        cfg = TuningConfig()

        assert cfg.n_trials == 150
        assert cfg.n_search_seeds == 16
        assert cfg.n_holdout_seeds == 32
        assert cfg.episode_length == 365
        assert cfg.seed_offset == 12_000_000
        assert cfg.holdout_seed_offset == 13_000_000
        assert cfg.sampler_seed == 42
        assert cfg.top_k_for_holdout == 5

    def test_tuning_config_field_types(self):
        """All TuningConfig fields have the documented types."""
        cfg = TuningConfig()

        assert isinstance(cfg.n_trials, int)
        assert isinstance(cfg.n_search_seeds, int)
        assert isinstance(cfg.n_holdout_seeds, int)
        assert isinstance(cfg.episode_length, int)
        assert isinstance(cfg.seed_offset, int)
        assert isinstance(cfg.holdout_seed_offset, int)
        assert isinstance(cfg.sampler_seed, int)
        assert isinstance(cfg.top_k_for_holdout, int)

    def test_tuning_config_is_frozen(self):
        """TuningConfig instances are immutable (frozen=True)."""
        cfg = TuningConfig()
        with pytest.raises((TypeError, AttributeError)):
            cfg.n_trials = 50  # type: ignore[misc]

    def test_tuning_config_allows_overrides(self):
        """TuningConfig fields can be overridden at construction time."""
        cfg = TuningConfig(n_trials=3, n_search_seeds=2, episode_length=10)
        assert cfg.n_trials == 3
        assert cfg.n_search_seeds == 2
        assert cfg.episode_length == 10
        # Other fields retain defaults
        assert cfg.n_holdout_seeds == 32


class TestSeedOffsetsAreDisjoint:
    """Assert that TuningConfig seed offsets are disjoint from all other seed ranges."""

    def test_seed_offsets_are_disjoint(self):
        """TuningConfig seed ranges must not overlap RLConfig eval or two-scale eval ranges.

        Disjointness invariant (from ADR 0009 and project conventions):
          - Training seeds: [0, total_env_steps) — effectively [0, 1_000_000)
          - CRN paired-eval: [eval_seed_offset, eval_seed_offset + n_eval_seeds)
            = [10_000_000, 10_000_032)
          - Two-scale eval: [eval_seed_offset + 1_000_000, ...)
            = [11_000_000, ...)
          - Tuning search: [seed_offset, seed_offset + n_search_seeds)
            = [12_000_000, 12_000_016)
          - Tuning holdout: [holdout_seed_offset, holdout_seed_offset + n_holdout_seeds)
            = [13_000_000, 13_000_032)

        This test checks that none of these ranges overlap.
        """
        rl_cfg = RLConfig()
        t_cfg = TuningConfig()

        # Define all seed ranges as (start, end) pairs (end is exclusive)
        training_end = 1_000_000  # generous upper bound on training seeds
        eval_start = rl_cfg.eval_seed_offset
        eval_end = eval_start + rl_cfg.n_eval_seeds

        two_scale_start = rl_cfg.eval_seed_offset + 1_000_000
        two_scale_end = two_scale_start + 100  # generous bound

        tuning_start = t_cfg.seed_offset
        tuning_end = tuning_start + t_cfg.n_search_seeds

        holdout_start = t_cfg.holdout_seed_offset
        holdout_end = holdout_start + t_cfg.n_holdout_seeds

        ranges = {
            "training": (0, training_end),
            "crn_eval": (eval_start, eval_end),
            "two_scale_eval": (two_scale_start, two_scale_end),
            "tuning_search": (tuning_start, tuning_end),
            "tuning_holdout": (holdout_start, holdout_end),
        }

        # Check all pairs for overlap.
        range_names = list(ranges.keys())
        for i in range(len(range_names)):
            for j in range(i + 1, len(range_names)):
                name_a = range_names[i]
                name_b = range_names[j]
                start_a, end_a = ranges[name_a]
                start_b, end_b = ranges[name_b]
                # Ranges overlap iff max(start_a, start_b) < min(end_a, end_b)
                overlap_start = max(start_a, start_b)
                overlap_end = min(end_a, end_b)
                assert overlap_start >= overlap_end, (
                    f"Seed ranges '{name_a}' [{start_a}, {end_a}) and "
                    f"'{name_b}' [{start_b}, {end_b}) overlap at "
                    f"[{overlap_start}, {overlap_end})"
                )

    def test_tuning_seed_offset_above_two_scale_range(self):
        """TuningConfig.seed_offset is above the two-scale eval range."""
        rl_cfg = RLConfig()
        t_cfg = TuningConfig()

        two_scale_start = rl_cfg.eval_seed_offset + 1_000_000
        assert t_cfg.seed_offset > two_scale_start, (
            f"seed_offset {t_cfg.seed_offset} should be above two-scale "
            f"eval start {two_scale_start}"
        )

    def test_holdout_offset_above_search_offset(self):
        """holdout_seed_offset > seed_offset + n_search_seeds (fully disjoint)."""
        cfg = TuningConfig()
        search_end = cfg.seed_offset + cfg.n_search_seeds
        assert cfg.holdout_seed_offset >= search_end, (
            f"holdout_seed_offset {cfg.holdout_seed_offset} must be >= "
            f"search end {search_end}"
        )
