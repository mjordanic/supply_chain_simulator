"""Tests for src/tuning/study.py — run_study() and confirm_top_k().

Coverage (per issue-04 spec):

  - test_run_study_smoke_3_trials:
      Run a 3-trial study with n_search_seeds=2, episode_length=10.
      Assert: returns optuna.Study with 3 trials; all three artifact files
      exist with correct schemas (column names + row counts).

  - test_run_study_artifacts_load_with_pandas_only:
      After running the smoke study, re-open the three artifact files using
      only pandas and json (no optuna import in the test body).
      Assert the schemas are usable without Optuna at read time.

  - test_run_study_crn_at_trial_level:
      Run two 2-trial studies with a policy_space that always returns a fixed
      OrderUpToPolicy() regardless of trial params.  Assert the per_seed.parquet
      rows for trial 0 in study A and trial 0 in study B are bit-identical
      (same eval_specs + same policy → same trajectory).

  - test_run_study_writes_git_sha:
      Assert study.json["git_sha"] is either a 40-char hex string or the
      empty string.

Coverage (per issue-05 spec):

  - test_confirm_top_k_smoke:
      Run a 5-trial study with n_holdout_seeds=2 and top_k_for_holdout=2;
      call confirm_top_k.  Assert holdout.parquet has 6 rows, labels are
      correct, holdout_summary.json has all documented fields.

  - test_confirm_top_k_seeds_disjoint_from_search:
      Assert seed values in holdout.parquet are disjoint from per_seed.parquet.

  - test_confirm_top_k_default_is_published:
      Assert "default"-labelled rows reproduce by directly evaluating
      OrderUpToPolicy() on the holdout specs.

  - test_confirm_top_k_paired_crn:
      Assert that for each seed, tuned_rank_1 and default rows share the
      same capacity and initial_cash (paired evaluation on identical worlds).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import pytest

from src.rl.configs.default import RLConfig
from src.sim.policy import OrderUpToPolicy
from src.sim.scenario import StoreTemplate, load_catalog
from src.tuning.config import TuningConfig
from src.tuning.study import confirm_top_k, run_study


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_catalog(n: int = 15) -> list:
    """Build a minimal n-product catalog."""
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
        id="study_test",
        region="US",
        capacity=capacity,
        init_balance=init_balance,
        init_stock_pct=0.0,
        delivery_lag=3,
        holding_rate=0.01,
        order_fee=50.0,
        init_active_count=5,
    )


def _make_short_config(episode_length: int = 10) -> RLConfig:
    """Return an RLConfig with a very short episode for fast tests."""
    return RLConfig(
        episode_length=episode_length,
        K_active=5,
        n_eval_seeds=2,
        eval_seed_offset=10_000_000,
    )


def _make_smoke_tuning_config(
    n_trials: int = 3,
    n_search_seeds: int = 2,
    episode_length: int = 10,
) -> TuningConfig:
    return TuningConfig(
        n_trials=n_trials,
        n_search_seeds=n_search_seeds,
        episode_length=episode_length,
    )


def _order_up_to_space(trial: optuna.Trial) -> OrderUpToPolicy:
    """Minimal trial-callback factory used in tests."""
    from src.tuning.search_spaces import order_up_to_space
    return order_up_to_space(trial)


# ---------------------------------------------------------------------------
# Expected column schemas
# ---------------------------------------------------------------------------

_TRIALS_REQUIRED_COLUMNS = {
    "trial_id",
    "state",
    "datetime_start",
    "datetime_complete",
    "duration_s",
    "mean_normalised_return",
    "mean_net_profit",
    "mean_service_level",
    "mean_stockout_rate",
    "mean_inventory_turnover",
    "mean_revenue",
    "mean_mean_price_pct_of_msrp",
}

_PER_SEED_REQUIRED_COLUMNS = {
    "trial_id",
    "seed",
    "capacity",
    "initial_cash",
    "net_profit",
    "normalised_return",
    "service_level",
    "stockout_rate",
    "inventory_turnover",
    "revenue",
    "mean_price_pct_of_msrp",
}

_STUDY_JSON_REQUIRED_FIELDS = {
    "study_name",
    "n_trials",
    "n_search_seeds",
    "n_holdout_seeds",
    "episode_length",
    "seed_offset",
    "holdout_seed_offset",
    "sampler_seed",
    "search_space",
    "best_trial_id",
    "best_params",
    "best_value",
    "started_at",
    "finished_at",
    "git_sha",
    "policy_class_name",
}


# ---------------------------------------------------------------------------
# Test: smoke — 3 trials, 2 seeds, 10-tick episodes
# ---------------------------------------------------------------------------


class TestRunStudySmoke3Trials:
    """A 3-trial smoke study runs to completion with the correct artifact schemas."""

    @pytest.fixture(scope="class")
    def smoke_study_result(self, tmp_path_factory):
        """Run the smoke study once; share results across test methods."""
        tmp_path = tmp_path_factory.mktemp("smoke")
        catalog = _make_catalog()
        template = _make_base_template()
        rl_config = _make_short_config()
        tuning_config = _make_smoke_tuning_config()

        study = run_study(
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_name="smoke",
            output_dir=str(tmp_path),
        )
        study_dir = tmp_path / "smoke"
        return {"study": study, "study_dir": study_dir, "tmp_path": tmp_path}

    def test_returns_optuna_study(self, smoke_study_result):
        """run_study returns an optuna.Study instance."""
        assert isinstance(smoke_study_result["study"], optuna.Study)

    def test_study_has_3_trials(self, smoke_study_result):
        """Returned study has exactly n_trials == 3 completed trials."""
        study = smoke_study_result["study"]
        assert len(study.trials) == 3

    def test_trials_parquet_exists(self, smoke_study_result):
        """trials.parquet is written under {output_dir}/{study_name}/."""
        trials_path = smoke_study_result["study_dir"] / "trials.parquet"
        assert trials_path.exists(), f"trials.parquet not found at {trials_path}"

    def test_trials_parquet_has_3_rows(self, smoke_study_result):
        """trials.parquet has exactly 3 rows (one per trial)."""
        trials_path = smoke_study_result["study_dir"] / "trials.parquet"
        df = pd.read_parquet(trials_path)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

    def test_trials_parquet_has_required_columns(self, smoke_study_result):
        """trials.parquet contains all documented columns."""
        trials_path = smoke_study_result["study_dir"] / "trials.parquet"
        df = pd.read_parquet(trials_path)
        missing = _TRIALS_REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"trials.parquet missing columns: {missing}"

    def test_per_seed_parquet_exists(self, smoke_study_result):
        """per_seed.parquet is written under {output_dir}/{study_name}/."""
        per_seed_path = smoke_study_result["study_dir"] / "per_seed.parquet"
        assert per_seed_path.exists(), f"per_seed.parquet not found at {per_seed_path}"

    def test_per_seed_parquet_has_6_rows(self, smoke_study_result):
        """per_seed.parquet has n_trials × n_search_seeds = 3 × 2 = 6 rows."""
        per_seed_path = smoke_study_result["study_dir"] / "per_seed.parquet"
        df = pd.read_parquet(per_seed_path)
        assert len(df) == 6, f"Expected 6 rows (3 trials × 2 seeds), got {len(df)}"

    def test_per_seed_parquet_has_required_columns(self, smoke_study_result):
        """per_seed.parquet contains all documented columns."""
        per_seed_path = smoke_study_result["study_dir"] / "per_seed.parquet"
        df = pd.read_parquet(per_seed_path)
        missing = _PER_SEED_REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"per_seed.parquet missing columns: {missing}"

    def test_study_json_exists(self, smoke_study_result):
        """study.json is written under {output_dir}/{study_name}/."""
        json_path = smoke_study_result["study_dir"] / "study.json"
        assert json_path.exists(), f"study.json not found at {json_path}"

    def test_study_json_has_all_required_fields(self, smoke_study_result):
        """study.json contains all documented fields."""
        json_path = smoke_study_result["study_dir"] / "study.json"
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        missing = _STUDY_JSON_REQUIRED_FIELDS - set(meta.keys())
        assert not missing, f"study.json missing fields: {missing}"

    def test_study_json_n_trials_matches(self, smoke_study_result):
        """study.json n_trials field matches the tuning_config value."""
        json_path = smoke_study_result["study_dir"] / "study.json"
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["n_trials"] == 3

    def test_study_json_n_search_seeds_matches(self, smoke_study_result):
        """study.json n_search_seeds field matches the tuning_config value."""
        json_path = smoke_study_result["study_dir"] / "study.json"
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["n_search_seeds"] == 2

    def test_storage_is_in_memory(self, smoke_study_result):
        """No SQLite file is created in the output directory."""
        study_dir = smoke_study_result["study_dir"]
        sqlite_files = list(study_dir.glob("*.db")) + list(study_dir.glob("*.sqlite"))
        assert not sqlite_files, f"Unexpected SQLite files: {sqlite_files}"


# ---------------------------------------------------------------------------
# Test: artifacts load with pandas only (no optuna in test body)
# ---------------------------------------------------------------------------


class TestRunStudyArtifactsLoadWithPandasOnly:
    """Artifact files load cleanly with only pandas and json — no optuna needed."""

    @pytest.fixture(scope="class")
    def artifacts(self, tmp_path_factory):
        """Run a smoke study and return paths to the artifact files."""
        tmp_path = tmp_path_factory.mktemp("pandas_only")
        catalog = _make_catalog()
        template = _make_base_template()
        rl_config = _make_short_config()
        tuning_config = _make_smoke_tuning_config()

        run_study(
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_name="pandas_only",
            output_dir=str(tmp_path),
        )
        study_dir = tmp_path / "pandas_only"
        return {
            "trials": study_dir / "trials.parquet",
            "per_seed": study_dir / "per_seed.parquet",
            "study_json": study_dir / "study.json",
        }

    def test_trials_parquet_loads_with_pandas(self, artifacts):
        """pd.read_parquet(trials.parquet) works without importing optuna."""
        # NB: the test body does NOT import optuna.
        df = pd.read_parquet(artifacts["trials"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Essential columns
        assert "trial_id" in df.columns
        assert "mean_normalised_return" in df.columns

    def test_per_seed_parquet_loads_with_pandas(self, artifacts):
        """pd.read_parquet(per_seed.parquet) works without importing optuna."""
        df = pd.read_parquet(artifacts["per_seed"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "trial_id" in df.columns
        assert "seed" in df.columns
        assert "net_profit" in df.columns

    def test_study_json_loads_with_stdlib(self, artifacts):
        """json.load(study.json) works without importing optuna."""
        with open(artifacts["study_json"], encoding="utf-8") as fh:
            meta = json.load(fh)
        assert isinstance(meta, dict)
        assert "study_name" in meta
        assert "best_trial_id" in meta

    def test_trials_trial_ids_match_per_seed(self, artifacts):
        """trial_id values in trials.parquet match those in per_seed.parquet."""
        trials_df = pd.read_parquet(artifacts["trials"])
        per_seed_df = pd.read_parquet(artifacts["per_seed"])
        trial_ids_from_trials = set(trials_df["trial_id"].unique())
        trial_ids_from_per_seed = set(per_seed_df["trial_id"].unique())
        assert trial_ids_from_trials == trial_ids_from_per_seed


# ---------------------------------------------------------------------------
# Test: CRN at the trial level — same policy + same specs → same trajectory
# ---------------------------------------------------------------------------


class TestRunStudyCRNAtTrialLevel:
    """Same eval_specs + same policy → bit-identical per-seed rows across trials."""

    def test_crn_at_trial_level(self, tmp_path):
        """Two trials with the same fixed policy produce identical per-seed rows.

        This verifies the CRN guarantee: the same eval_specs list is reused
        for every trial, so a policy that ignores trial params always sees the
        same world trajectory.
        """
        catalog = _make_catalog()
        template = _make_base_template()
        rl_config = _make_short_config()
        tuning_config = _make_smoke_tuning_config(n_trials=2, n_search_seeds=2)

        # A policy_space that always returns the same fixed policy (ignores trial params).
        def fixed_policy_space(trial: optuna.Trial) -> OrderUpToPolicy:
            # We must call at least one suggest_* so the trial has distributions.
            _ = trial.suggest_float("dummy", 0.0, 1.0)
            return OrderUpToPolicy()

        run_study(
            fixed_policy_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_name="crn_test",
            output_dir=str(tmp_path),
        )

        per_seed_df = pd.read_parquet(tmp_path / "crn_test" / "per_seed.parquet")

        # There should be 2 trials × 2 seeds = 4 rows.
        assert len(per_seed_df) == 4

        # Extract rows for trial 0 and trial 1.
        trial_0 = per_seed_df[per_seed_df["trial_id"] == 0].sort_values("seed").reset_index(drop=True)
        trial_1 = per_seed_df[per_seed_df["trial_id"] == 1].sort_values("seed").reset_index(drop=True)

        numeric_cols = [
            "capacity", "initial_cash", "net_profit", "normalised_return",
            "service_level", "stockout_rate", "inventory_turnover", "revenue",
            "mean_price_pct_of_msrp",
        ]
        for col in numeric_cols:
            vals_0 = trial_0[col].tolist()
            vals_1 = trial_1[col].tolist()
            assert vals_0 == vals_1, (
                f"CRN violation in column '{col}': "
                f"trial_0={vals_0} != trial_1={vals_1}"
            )


# ---------------------------------------------------------------------------
# Test: git_sha field
# ---------------------------------------------------------------------------


class TestRunStudyWritesGitSha:
    """study.json['git_sha'] is either a 40-char hex string or empty string."""

    def test_git_sha_is_valid(self, tmp_path):
        """study.json['git_sha'] is a 40-char hex string or empty string."""
        catalog = _make_catalog()
        template = _make_base_template()
        rl_config = _make_short_config()
        tuning_config = _make_smoke_tuning_config(n_trials=1, n_search_seeds=1)

        run_study(
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_name="git_sha_test",
            output_dir=str(tmp_path),
        )

        json_path = tmp_path / "git_sha_test" / "study.json"
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        git_sha = meta["git_sha"]
        assert isinstance(git_sha, str), f"git_sha must be a string, got {type(git_sha)}"

        if git_sha:
            # Must be a 40-char lowercase hex string.
            assert len(git_sha) == 40, (
                f"git_sha has length {len(git_sha)}, expected 40: {git_sha!r}"
            )
            assert all(c in "0123456789abcdef" for c in git_sha.lower()), (
                f"git_sha contains non-hex characters: {git_sha!r}"
            )
        # If empty string, that's also acceptable (e.g. git not available).


# ---------------------------------------------------------------------------
# Shared fixture helpers for confirm_top_k tests
# ---------------------------------------------------------------------------


def _make_holdout_tuning_config(
    n_trials: int = 5,
    n_search_seeds: int = 2,
    n_holdout_seeds: int = 2,
    episode_length: int = 10,
    top_k_for_holdout: int = 2,
) -> TuningConfig:
    """Return a TuningConfig sized for fast confirm_top_k tests."""
    return TuningConfig(
        n_trials=n_trials,
        n_search_seeds=n_search_seeds,
        n_holdout_seeds=n_holdout_seeds,
        episode_length=episode_length,
        top_k_for_holdout=top_k_for_holdout,
    )


_HOLDOUT_SUMMARY_REQUIRED_FIELDS = {
    "tuned_best_label",
    "tuned_best_mean_net_profit",
    "tuned_best_ci_low",
    "tuned_best_ci_high",
    "default_mean_net_profit",
    "default_ci_low",
    "default_ci_high",
    "headline_uplift",
    "headline_uplift_ci_low",
    "headline_uplift_ci_high",
    "n_holdout_seeds",
}


# ---------------------------------------------------------------------------
# Test: confirm_top_k smoke
# ---------------------------------------------------------------------------


class TestConfirmTopKSmoke:
    """confirm_top_k writes holdout.parquet (6 rows) + holdout_summary.json."""

    @pytest.fixture(scope="class")
    def holdout_result(self, tmp_path_factory):
        """Run a 5-trial smoke study then call confirm_top_k; share results."""
        tmp_path = tmp_path_factory.mktemp("holdout_smoke")
        catalog = _make_catalog()
        template = _make_base_template()
        rl_config = _make_short_config()
        tuning_config = _make_holdout_tuning_config()

        study = run_study(
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_name="holdout_smoke",
            output_dir=str(tmp_path),
        )
        study_dir = str(tmp_path / "holdout_smoke")

        summary = confirm_top_k(
            study,
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_dir=study_dir,
        )
        return {
            "study": study,
            "study_dir": tmp_path / "holdout_smoke",
            "summary": summary,
            "tmp_path": tmp_path,
            "tuning_config": tuning_config,
        }

    def test_holdout_parquet_exists(self, holdout_result):
        """holdout.parquet is written under study_dir."""
        holdout_path = holdout_result["study_dir"] / "holdout.parquet"
        assert holdout_path.exists(), f"holdout.parquet not found at {holdout_path}"

    def test_holdout_parquet_has_6_rows(self, holdout_result):
        """holdout.parquet has (top_k + 1) × n_holdout_seeds = 3 × 2 = 6 rows."""
        holdout_path = holdout_result["study_dir"] / "holdout.parquet"
        df = pd.read_parquet(holdout_path)
        assert len(df) == 6, f"Expected 6 rows, got {len(df)}"

    def test_holdout_parquet_has_correct_labels(self, holdout_result):
        """holdout.parquet label column contains tuned_rank_1, tuned_rank_2, default."""
        holdout_path = holdout_result["study_dir"] / "holdout.parquet"
        df = pd.read_parquet(holdout_path)
        labels = set(df["label"].unique())
        expected = {"tuned_rank_1", "tuned_rank_2", "default"}
        assert labels == expected, f"Expected labels {expected}, got {labels}"

    def test_holdout_summary_json_exists(self, holdout_result):
        """holdout_summary.json is written under study_dir."""
        summary_path = holdout_result["study_dir"] / "holdout_summary.json"
        assert summary_path.exists(), f"holdout_summary.json not found at {summary_path}"

    def test_holdout_summary_has_all_fields(self, holdout_result):
        """holdout_summary.json contains all documented fields."""
        summary_path = holdout_result["study_dir"] / "holdout_summary.json"
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        missing = _HOLDOUT_SUMMARY_REQUIRED_FIELDS - set(summary.keys())
        assert not missing, f"holdout_summary.json missing fields: {missing}"

    def test_holdout_summary_tuned_best_label_is_valid(self, holdout_result):
        """holdout_summary.json['tuned_best_label'] is one of tuned_rank_1 ... tuned_rank_K."""
        summary = holdout_result["summary"]
        top_k = holdout_result["tuning_config"].top_k_for_holdout
        valid_labels = {f"tuned_rank_{r}" for r in range(1, top_k + 1)}
        assert summary["tuned_best_label"] in valid_labels, (
            f"tuned_best_label '{summary['tuned_best_label']}' not in {valid_labels}"
        )

    def test_holdout_summary_n_holdout_seeds_matches(self, holdout_result):
        """holdout_summary.json['n_holdout_seeds'] matches tuning_config."""
        summary = holdout_result["summary"]
        expected = holdout_result["tuning_config"].n_holdout_seeds
        assert summary["n_holdout_seeds"] == expected


# ---------------------------------------------------------------------------
# Test: holdout seeds are disjoint from search seeds
# ---------------------------------------------------------------------------


class TestConfirmTopKSeedsDisjointFromSearch:
    """Seeds in holdout.parquet are disjoint from seeds in per_seed.parquet."""

    def test_seeds_disjoint(self, tmp_path):
        """No seed appears in both holdout.parquet and per_seed.parquet."""
        catalog = _make_catalog()
        template = _make_base_template()
        rl_config = _make_short_config()
        tuning_config = _make_holdout_tuning_config()

        study = run_study(
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_name="disjoint_test",
            output_dir=str(tmp_path),
        )
        study_dir = str(tmp_path / "disjoint_test")

        confirm_top_k(
            study,
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_dir=study_dir,
        )

        per_seed_df = pd.read_parquet(tmp_path / "disjoint_test" / "per_seed.parquet")
        holdout_df = pd.read_parquet(tmp_path / "disjoint_test" / "holdout.parquet")

        search_seeds = set(per_seed_df["seed"].unique())
        holdout_seeds = set(holdout_df["seed"].unique())

        overlap = search_seeds & holdout_seeds
        assert not overlap, (
            f"Seeds appear in both search and holdout sets: {overlap}"
        )


# ---------------------------------------------------------------------------
# Test: "default" rows reproduce direct evaluation
# ---------------------------------------------------------------------------


class TestConfirmTopKDefaultIsPublished:
    """The 'default'-labelled rows are produced by OrderUpToPolicy() with no kwargs."""

    def test_default_rows_match_direct_eval(self, tmp_path):
        """Per-seed net_profit for the 'default' label matches a direct evaluation.

        Re-evaluates OrderUpToPolicy() on the same holdout specs and asserts
        bit-identical net_profit values.
        """
        from src.rl.episode_sampler import sample_episode
        from src.tuning.evaluator import evaluate_policy_normalised

        catalog = _make_catalog()
        template = _make_base_template()
        rl_config = _make_short_config()
        tuning_config = _make_holdout_tuning_config()

        study = run_study(
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_name="default_repro",
            output_dir=str(tmp_path),
        )
        study_dir = str(tmp_path / "default_repro")

        confirm_top_k(
            study,
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_dir=study_dir,
        )

        holdout_df = pd.read_parquet(tmp_path / "default_repro" / "holdout.parquet")
        default_rows = holdout_df[holdout_df["label"] == "default"].sort_values("seed").reset_index(drop=True)

        # Rebuild holdout specs the same way confirm_top_k does.
        import dataclasses as _dc
        rl_config_with_length = _dc.replace(rl_config, episode_length=tuning_config.episode_length)
        holdout_specs = []
        for i in range(tuning_config.n_holdout_seeds):
            seed = tuning_config.holdout_seed_offset + i
            spec = sample_episode(
                catalog=catalog,
                base_template=template,
                config=rl_config_with_length,
                episode_seed=seed,
            )
            holdout_specs.append(spec)

        # Direct evaluation of OrderUpToPolicy() on the holdout specs.
        direct_kpis = evaluate_policy_normalised(
            lambda: OrderUpToPolicy(),
            holdout_specs,
            config=rl_config,
        )

        for i, row in default_rows.iterrows():
            expected_profit = direct_kpis["per_seed_net_profit"][i]
            actual_profit = row["net_profit"]
            assert actual_profit == expected_profit, (
                f"Row {i}: default net_profit {actual_profit} != direct eval {expected_profit}"
            )


# ---------------------------------------------------------------------------
# Test: paired CRN — same seed → same capacity and initial_cash across labels
# ---------------------------------------------------------------------------


class TestConfirmTopKPairedCRN:
    """Same seed in holdout.parquet → same capacity and initial_cash for all labels."""

    def test_paired_crn_same_world_per_seed(self, tmp_path):
        """For each holdout seed, tuned_rank_1 and default share capacity and initial_cash.

        This asserts the CRN invariant: all K+1 policies see the same world
        for each seed (the world is determined by the seed, not by the policy).
        """
        catalog = _make_catalog()
        template = _make_base_template()
        rl_config = _make_short_config()
        tuning_config = _make_holdout_tuning_config()

        study = run_study(
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_name="crn_paired",
            output_dir=str(tmp_path),
        )
        study_dir = str(tmp_path / "crn_paired")

        confirm_top_k(
            study,
            _order_up_to_space,
            catalog=catalog,
            base_template=template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_dir=study_dir,
        )

        holdout_df = pd.read_parquet(tmp_path / "crn_paired" / "holdout.parquet")

        for seed in holdout_df["seed"].unique():
            seed_rows = holdout_df[holdout_df["seed"] == seed]
            capacities = seed_rows["capacity"].unique()
            cash_vals = seed_rows["initial_cash"].unique()

            assert len(capacities) == 1, (
                f"seed={seed}: multiple capacity values across labels: {capacities.tolist()}"
            )
            assert len(cash_vals) == 1, (
                f"seed={seed}: multiple initial_cash values across labels: {cash_vals.tolist()}"
            )
