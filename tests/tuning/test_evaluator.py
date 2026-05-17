"""Tests for src/tuning/evaluator.py — evaluate_policy_normalised.

Coverage (per issue-02 spec):
  - test_evaluate_policy_normalised_2_seeds: Basic smoke test on 2 seeds.
    Asserts all expected keys present, list lengths == 2, mean computation.
  - test_evaluate_policy_normalised_crn_determinism: Same call twice → bit-identical.
  - test_normalisation_is_dimensionless: Two specs at 10× capacity/balance ratio
    on a no-op policy yield approximately equal per-seed normalised returns.
"""

from __future__ import annotations

import pytest

from src.sim.distributions import Constant
from src.sim.policy import NoopPolicy, OrderUpToPolicy
from src.sim.scenario import StoreTemplate, load_catalog
from src.tuning.config import TuningConfig
from src.tuning.episode import sample_episode
from src.tuning.evaluator import evaluate_policy_normalised


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_catalog(n: int = 15) -> list:
    return load_catalog(
        [
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": 10.0 + i,
                "unit_cost": 4.0 + i * 0.3,
                "seasonality": "all_season",
            }
            for i in range(n)
        ]
    )


def _make_base_template(
    capacity: int = 200,
    init_balance: float = 20_000.0,
) -> StoreTemplate:
    return StoreTemplate(
        id="tuning_eval_test",
        region="US",
        capacity=capacity,
        init_balance=init_balance,
        init_stock_pct=0.0,
        delivery_lag=3,
        holding_rate=0.01,
        order_fee=50.0,
        init_active_count=5,
    )


def _make_short_config(episode_length: int = 5) -> TuningConfig:
    """Return a TuningConfig with a very short episode for fast tests.

    init_stock_pct pinned to 0.0 so NoopPolicy yields zero profit (used by
    the dimensionless-normalisation tests below).
    """
    return TuningConfig(
        episode_length=episode_length,
        K_active=5,
        n_trials=1,
        n_search_seeds=2,
        init_stock_pct_dist=Constant(0.0),
    )


def _make_specs(n: int = 2, seed_offset: int = 12_000_000, episode_length: int = 5):
    catalog = _make_catalog()
    template = _make_base_template()
    config = _make_short_config(episode_length=episode_length)
    return [
        sample_episode(catalog, template, config, episode_seed=seed_offset + i)
        for i in range(n)
    ]


_EXPECTED_KEYS = {
    "mean_normalised_return",
    "per_seed_net_profit",
    "per_seed_initial_cash",
    "per_seed_capacity",
    "per_seed_service_level",
    "per_seed_stockout_rate",
    "per_seed_inventory_turnover",
    "per_seed_revenue",
    "per_seed_mean_price_pct_of_msrp",
    "mean_net_profit",
    "mean_initial_cash",
    "mean_capacity",
    "mean_service_level",
    "mean_stockout_rate",
    "mean_inventory_turnover",
    "mean_revenue",
    "mean_mean_price_pct_of_msrp",
}


class TestEvaluatePolicyNormalised2Seeds:
    """evaluate_policy_normalised works correctly on a 2-seed episode list."""

    def test_all_expected_keys_present(self):
        specs = _make_specs(n=2)
        result = evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs)
        for key in _EXPECTED_KEYS:
            assert key in result, f"Missing expected key {key!r}"

    def test_per_seed_lists_have_length_2(self):
        specs = _make_specs(n=2)
        result = evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs)
        per_seed_keys = [k for k in result if k.startswith("per_seed_")]
        assert per_seed_keys, "No per_seed_* keys found"
        for key in per_seed_keys:
            assert len(result[key]) == 2

    def test_mean_normalised_return_matches_manual_computation(self):
        specs = _make_specs(n=2)
        result = evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs)
        net_profits = result["per_seed_net_profit"]
        initial_cashes = result["per_seed_initial_cash"]
        per_seed_norm = [p / max(1e-9, c) for p, c in zip(net_profits, initial_cashes)]
        expected_mean = sum(per_seed_norm) / len(per_seed_norm)

        assert result["mean_normalised_return"] == pytest.approx(expected_mean, rel=1e-9)

    def test_mean_scalars_match_per_seed_means(self):
        specs = _make_specs(n=2)
        result = evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs)
        per_seed_to_mean = {
            "per_seed_net_profit": "mean_net_profit",
            "per_seed_initial_cash": "mean_initial_cash",
            "per_seed_capacity": "mean_capacity",
            "per_seed_service_level": "mean_service_level",
            "per_seed_stockout_rate": "mean_stockout_rate",
            "per_seed_inventory_turnover": "mean_inventory_turnover",
            "per_seed_revenue": "mean_revenue",
            "per_seed_mean_price_pct_of_msrp": "mean_mean_price_pct_of_msrp",
        }
        for per_key, mean_key in per_seed_to_mean.items():
            vals = result[per_key]
            expected = sum(vals) / len(vals)
            assert result[mean_key] == pytest.approx(expected, rel=1e-9)

    def test_no_optuna_import(self):
        """evaluate_policy_normalised does not import optuna at runtime."""
        import inspect
        import src.tuning.evaluator as ev_mod
        source = inspect.getsource(ev_mod)
        assert "import optuna" not in source

    def test_no_rl_import(self):
        """evaluator must not depend on any src.rl module."""
        import inspect
        import src.tuning.evaluator as ev_mod
        source = inspect.getsource(ev_mod)
        assert "src.rl" not in source


class TestEvaluatePolicyNormalisedCrnDeterminism:
    """Same call on identical inputs returns bit-identical numbers."""

    def test_crn_determinism_order_up_to(self):
        specs = _make_specs(n=2)
        result1 = evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs)
        result2 = evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs)

        assert result1["mean_normalised_return"] == result2["mean_normalised_return"]
        assert result1["per_seed_net_profit"] == result2["per_seed_net_profit"]
        assert result1["per_seed_service_level"] == result2["per_seed_service_level"]

    def test_crn_determinism_noop_policy(self):
        specs = _make_specs(n=2)
        result1 = evaluate_policy_normalised(lambda: NoopPolicy(), specs)
        result2 = evaluate_policy_normalised(lambda: NoopPolicy(), specs)

        assert result1["per_seed_net_profit"] == result2["per_seed_net_profit"]
        assert result1["mean_normalised_return"] == result2["mean_normalised_return"]


class TestNormalisationIsDimensionless:
    """Per-seed normalised return is approximately equal between two specs at 10×
    capacity/balance when using a no-op policy.

    With init_stock_pct=0.0 a NoopPolicy yields net_profit ≈ 0 → normalised return
    ≈ 0 on both scales. Confirms the normalisation cancels scale.
    """

    def test_normalisation_is_dimensionless_noop(self):
        catalog = _make_catalog()
        config = _make_short_config(episode_length=5)
        template_small = _make_base_template(capacity=200, init_balance=20_000.0)
        template_large = _make_base_template(capacity=2000, init_balance=200_000.0)

        spec_small = sample_episode(catalog, template_small, config, episode_seed=12_000_000)
        spec_large = sample_episode(catalog, template_large, config, episode_seed=12_000_001)

        result_small = evaluate_policy_normalised(lambda: NoopPolicy(), [spec_small])
        result_large = evaluate_policy_normalised(lambda: NoopPolicy(), [spec_large])

        norm_small = result_small["per_seed_net_profit"][0] / max(
            1e-9, result_small["per_seed_initial_cash"][0]
        )
        norm_large = result_large["per_seed_net_profit"][0] / max(
            1e-9, result_large["per_seed_initial_cash"][0]
        )

        assert abs(norm_small) <= 1e-6
        assert abs(norm_large) <= 1e-6

    def test_normalisation_is_dimensionless_noop_two_seeds(self):
        catalog = _make_catalog()
        config = _make_short_config(episode_length=5)
        template_small = _make_base_template(capacity=100, init_balance=10_000.0)
        template_large = _make_base_template(capacity=1000, init_balance=100_000.0)

        spec_small = sample_episode(catalog, template_small, config, episode_seed=12_000_200)
        spec_large = sample_episode(catalog, template_large, config, episode_seed=12_000_201)

        result_small = evaluate_policy_normalised(lambda: NoopPolicy(), [spec_small])
        result_large = evaluate_policy_normalised(lambda: NoopPolicy(), [spec_large])

        norm_small = result_small["per_seed_net_profit"][0] / max(
            1e-9, result_small["per_seed_initial_cash"][0]
        )
        norm_large = result_large["per_seed_net_profit"][0] / max(
            1e-9, result_large["per_seed_initial_cash"][0]
        )

        max_abs = max(abs(norm_small), abs(norm_large))
        if max_abs < 1e-9:
            assert True
        else:
            rel_diff = abs(norm_small - norm_large) / max_abs
            assert rel_diff <= 0.01
