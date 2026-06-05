"""Tests for TuningConfig dataclass."""

from __future__ import annotations

import pytest

from src.tuning.config import TuningConfig


class TestTuningConfigDefaults:
    """Sanity check the documented defaults exist with the documented types."""

    def test_tuning_config_defaults(self):
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
        cfg = TuningConfig()
        with pytest.raises((TypeError, AttributeError)):
            cfg.n_trials = 50  # type: ignore[misc]

    def test_tuning_config_allows_overrides(self):
        cfg = TuningConfig(n_trials=3, n_search_seeds=2, episode_length=10)
        assert cfg.n_trials == 3
        assert cfg.n_search_seeds == 2
        assert cfg.episode_length == 10
        assert cfg.n_holdout_seeds == 32


class TestSeedOffsetsAreDisjoint:
    """Assert tuning seed ranges are well separated from each other and from
    other ranges used elsewhere in the project."""

    # Project-wide seed-range conventions (mirrored here so the tuning tests
    # do not need to import RL config to assert disjointness).
    _TRAINING_END = 1_000_000
    _CRN_EVAL_START = 10_000_000
    _CRN_EVAL_END = 10_000_032
    _TWO_SCALE_START = 11_000_000
    _TWO_SCALE_END = 11_000_100

    def test_seed_offsets_are_disjoint(self):
        t_cfg = TuningConfig()

        tuning_start = t_cfg.seed_offset
        tuning_end = tuning_start + t_cfg.n_search_seeds

        holdout_start = t_cfg.holdout_seed_offset
        holdout_end = holdout_start + t_cfg.n_holdout_seeds

        ranges = {
            "training": (0, self._TRAINING_END),
            "crn_eval": (self._CRN_EVAL_START, self._CRN_EVAL_END),
            "two_scale_eval": (self._TWO_SCALE_START, self._TWO_SCALE_END),
            "tuning_search": (tuning_start, tuning_end),
            "tuning_holdout": (holdout_start, holdout_end),
        }

        range_names = list(ranges.keys())
        for i in range(len(range_names)):
            for j in range(i + 1, len(range_names)):
                name_a = range_names[i]
                name_b = range_names[j]
                start_a, end_a = ranges[name_a]
                start_b, end_b = ranges[name_b]
                overlap_start = max(start_a, start_b)
                overlap_end = min(end_a, end_b)
                assert overlap_start >= overlap_end, (
                    f"Seed ranges '{name_a}' [{start_a}, {end_a}) and "
                    f"'{name_b}' [{start_b}, {end_b}) overlap"
                )

    def test_tuning_seed_offset_above_two_scale_range(self):
        t_cfg = TuningConfig()
        assert t_cfg.seed_offset > self._TWO_SCALE_START

    def test_holdout_offset_above_search_offset(self):
        cfg = TuningConfig()
        search_end = cfg.seed_offset + cfg.n_search_seeds
        assert cfg.holdout_seed_offset >= search_end
