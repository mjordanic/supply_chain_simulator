"""Tests for src/tuning/study.py — run_study() and confirm_top_k()."""

from __future__ import annotations

import json
import subprocess
import sys

import optuna
import pandas as pd
import pytest

from src.sim.policy import OrderUpToPolicy
from src.sim.scenario import load_catalog
from src.tuning.config import TuningConfig
from src.tuning.study import confirm_top_k, run_study


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


def _make_smoke_tuning_config(
    n_trials: int = 3,
    n_search_seeds: int = 2,
    episode_length: int = 10,
) -> TuningConfig:
    return TuningConfig(
        n_trials=n_trials,
        n_search_seeds=n_search_seeds,
        episode_length=episode_length,
        K_active=5,
    )


def _order_up_to_space(trial: optuna.Trial) -> OrderUpToPolicy:
    from src.tuning.search_spaces import order_up_to_space
    return order_up_to_space(trial)


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


class TestRunStudySmoke3Trials:
    """A 3-trial smoke study runs to completion with the correct artifact schemas."""

    @pytest.fixture(scope="class")
    def smoke_study_result(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("smoke")
        catalog = _make_catalog()
        tuning_config = _make_smoke_tuning_config()

        study = run_study(
            _order_up_to_space,
            catalog=catalog,
            tuning_config=tuning_config,
            study_name="smoke",
            output_dir=str(tmp_path),
        )
        study_dir = tmp_path / "smoke"
        return {"study": study, "study_dir": study_dir, "tmp_path": tmp_path}

    def test_returns_optuna_study(self, smoke_study_result):
        assert isinstance(smoke_study_result["study"], optuna.Study)

    def test_study_has_3_trials(self, smoke_study_result):
        study = smoke_study_result["study"]
        assert len(study.trials) == 3

    def test_trials_parquet_exists(self, smoke_study_result):
        trials_path = smoke_study_result["study_dir"] / "trials.parquet"
        assert trials_path.exists()

    def test_trials_parquet_has_3_rows(self, smoke_study_result):
        trials_path = smoke_study_result["study_dir"] / "trials.parquet"
        df = pd.read_parquet(trials_path)
        assert len(df) == 3

    def test_trials_parquet_has_required_columns(self, smoke_study_result):
        trials_path = smoke_study_result["study_dir"] / "trials.parquet"
        df = pd.read_parquet(trials_path)
        missing = _TRIALS_REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"trials.parquet missing columns: {missing}"

    def test_per_seed_parquet_exists(self, smoke_study_result):
        per_seed_path = smoke_study_result["study_dir"] / "per_seed.parquet"
        assert per_seed_path.exists()

    def test_per_seed_parquet_has_6_rows(self, smoke_study_result):
        per_seed_path = smoke_study_result["study_dir"] / "per_seed.parquet"
        df = pd.read_parquet(per_seed_path)
        assert len(df) == 6

    def test_per_seed_parquet_has_required_columns(self, smoke_study_result):
        per_seed_path = smoke_study_result["study_dir"] / "per_seed.parquet"
        df = pd.read_parquet(per_seed_path)
        missing = _PER_SEED_REQUIRED_COLUMNS - set(df.columns)
        assert not missing

    def test_study_json_exists(self, smoke_study_result):
        json_path = smoke_study_result["study_dir"] / "study.json"
        assert json_path.exists()

    def test_study_json_has_all_required_fields(self, smoke_study_result):
        json_path = smoke_study_result["study_dir"] / "study.json"
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        missing = _STUDY_JSON_REQUIRED_FIELDS - set(meta.keys())
        assert not missing

    def test_study_json_n_trials_matches(self, smoke_study_result):
        json_path = smoke_study_result["study_dir"] / "study.json"
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["n_trials"] == 3

    def test_study_json_n_search_seeds_matches(self, smoke_study_result):
        json_path = smoke_study_result["study_dir"] / "study.json"
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["n_search_seeds"] == 2

    def test_storage_is_in_memory(self, smoke_study_result):
        study_dir = smoke_study_result["study_dir"]
        sqlite_files = list(study_dir.glob("*.db")) + list(study_dir.glob("*.sqlite"))
        assert not sqlite_files


class TestRunStudyArtifactsLoadWithPandasOnly:
    """Artifact files load cleanly with only pandas and json."""

    @pytest.fixture(scope="class")
    def artifacts(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("pandas_only")
        catalog = _make_catalog()
        tuning_config = _make_smoke_tuning_config()

        run_study(
            _order_up_to_space,
            catalog=catalog,
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
        df = pd.read_parquet(artifacts["trials"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "trial_id" in df.columns
        assert "mean_normalised_return" in df.columns

    def test_per_seed_parquet_loads_with_pandas(self, artifacts):
        df = pd.read_parquet(artifacts["per_seed"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "trial_id" in df.columns
        assert "seed" in df.columns
        assert "net_profit" in df.columns

    def test_study_json_loads_with_stdlib(self, artifacts):
        with open(artifacts["study_json"], encoding="utf-8") as fh:
            meta = json.load(fh)
        assert isinstance(meta, dict)
        assert "study_name" in meta
        assert "best_trial_id" in meta

    def test_trials_trial_ids_match_per_seed(self, artifacts):
        trials_df = pd.read_parquet(artifacts["trials"])
        per_seed_df = pd.read_parquet(artifacts["per_seed"])
        trial_ids_from_trials = set(trials_df["trial_id"].unique())
        trial_ids_from_per_seed = set(per_seed_df["trial_id"].unique())
        assert trial_ids_from_trials == trial_ids_from_per_seed


class TestRunStudyCRNAtTrialLevel:
    """Same eval_specs + same policy → bit-identical per-seed rows across trials."""

    def test_crn_at_trial_level(self, tmp_path):
        catalog = _make_catalog()
        tuning_config = _make_smoke_tuning_config(n_trials=2, n_search_seeds=2)

        def fixed_policy_space(trial: optuna.Trial) -> OrderUpToPolicy:
            _ = trial.suggest_float("dummy", 0.0, 1.0)
            return OrderUpToPolicy()

        run_study(
            fixed_policy_space,
            catalog=catalog,
            tuning_config=tuning_config,
            study_name="crn_test",
            output_dir=str(tmp_path),
        )

        per_seed_df = pd.read_parquet(tmp_path / "crn_test" / "per_seed.parquet")
        assert len(per_seed_df) == 4

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
            assert vals_0 == vals_1


class TestRunStudyWritesGitSha:
    """study.json['git_sha'] is either a 40-char hex string or empty."""

    def test_git_sha_is_valid(self, tmp_path):
        catalog = _make_catalog()
        tuning_config = _make_smoke_tuning_config(n_trials=1, n_search_seeds=1)

        run_study(
            _order_up_to_space,
            catalog=catalog,
            tuning_config=tuning_config,
            study_name="git_sha_test",
            output_dir=str(tmp_path),
        )

        json_path = tmp_path / "git_sha_test" / "study.json"
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        git_sha = meta["git_sha"]
        assert isinstance(git_sha, str)

        if git_sha:
            assert len(git_sha) == 40
            assert all(c in "0123456789abcdef" for c in git_sha.lower())


def _make_holdout_tuning_config(
    n_trials: int = 5,
    n_search_seeds: int = 2,
    n_holdout_seeds: int = 2,
    episode_length: int = 10,
    top_k_for_holdout: int = 2,
) -> TuningConfig:
    return TuningConfig(
        n_trials=n_trials,
        n_search_seeds=n_search_seeds,
        n_holdout_seeds=n_holdout_seeds,
        episode_length=episode_length,
        top_k_for_holdout=top_k_for_holdout,
        K_active=5,
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


class TestConfirmTopKSmoke:
    """confirm_top_k writes holdout.parquet (6 rows) + holdout_summary.json."""

    @pytest.fixture(scope="class")
    def holdout_result(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("holdout_smoke")
        catalog = _make_catalog()
        tuning_config = _make_holdout_tuning_config()

        study = run_study(
            _order_up_to_space,
            catalog=catalog,
            tuning_config=tuning_config,
            study_name="holdout_smoke",
            output_dir=str(tmp_path),
        )
        study_dir = str(tmp_path / "holdout_smoke")

        summary = confirm_top_k(
            study,
            _order_up_to_space,
            catalog=catalog,
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
        holdout_path = holdout_result["study_dir"] / "holdout.parquet"
        assert holdout_path.exists()

    def test_holdout_parquet_has_6_rows(self, holdout_result):
        holdout_path = holdout_result["study_dir"] / "holdout.parquet"
        df = pd.read_parquet(holdout_path)
        assert len(df) == 6

    def test_holdout_parquet_has_correct_labels(self, holdout_result):
        holdout_path = holdout_result["study_dir"] / "holdout.parquet"
        df = pd.read_parquet(holdout_path)
        labels = set(df["label"].unique())
        expected = {"tuned_rank_1", "tuned_rank_2", "default"}
        assert labels == expected

    def test_holdout_summary_json_exists(self, holdout_result):
        summary_path = holdout_result["study_dir"] / "holdout_summary.json"
        assert summary_path.exists()

    def test_holdout_summary_has_all_fields(self, holdout_result):
        summary_path = holdout_result["study_dir"] / "holdout_summary.json"
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        missing = _HOLDOUT_SUMMARY_REQUIRED_FIELDS - set(summary.keys())
        assert not missing

    def test_holdout_summary_tuned_best_label_is_valid(self, holdout_result):
        summary = holdout_result["summary"]
        top_k = holdout_result["tuning_config"].top_k_for_holdout
        valid_labels = {f"tuned_rank_{r}" for r in range(1, top_k + 1)}
        assert summary["tuned_best_label"] in valid_labels

    def test_holdout_summary_n_holdout_seeds_matches(self, holdout_result):
        summary = holdout_result["summary"]
        expected = holdout_result["tuning_config"].n_holdout_seeds
        assert summary["n_holdout_seeds"] == expected


class TestConfirmTopKSeedsDisjointFromSearch:
    """Seeds in holdout.parquet are disjoint from seeds in per_seed.parquet."""

    def test_seeds_disjoint(self, tmp_path):
        catalog = _make_catalog()
        tuning_config = _make_holdout_tuning_config()

        study = run_study(
            _order_up_to_space,
            catalog=catalog,
            tuning_config=tuning_config,
            study_name="disjoint_test",
            output_dir=str(tmp_path),
        )
        study_dir = str(tmp_path / "disjoint_test")

        confirm_top_k(
            study,
            _order_up_to_space,
            catalog=catalog,
            tuning_config=tuning_config,
            study_dir=study_dir,
        )

        per_seed_df = pd.read_parquet(tmp_path / "disjoint_test" / "per_seed.parquet")
        holdout_df = pd.read_parquet(tmp_path / "disjoint_test" / "holdout.parquet")

        search_seeds = set(per_seed_df["seed"].unique())
        holdout_seeds = set(holdout_df["seed"].unique())

        overlap = search_seeds & holdout_seeds
        assert not overlap


class TestConfirmTopKDefaultIsPublished:
    """The 'default'-labelled rows are produced by OrderUpToPolicy() with no kwargs."""

    def test_default_rows_match_direct_eval(self, tmp_path):
        from src.tuning.episode import sample_episode
        from src.tuning.evaluator import evaluate_policy_normalised

        catalog = _make_catalog()
        tuning_config = _make_holdout_tuning_config()

        study = run_study(
            _order_up_to_space,
            catalog=catalog,
            tuning_config=tuning_config,
            study_name="default_repro",
            output_dir=str(tmp_path),
        )
        study_dir = str(tmp_path / "default_repro")

        confirm_top_k(
            study,
            _order_up_to_space,
            catalog=catalog,
            tuning_config=tuning_config,
            study_dir=study_dir,
        )

        holdout_df = pd.read_parquet(tmp_path / "default_repro" / "holdout.parquet")
        default_rows = holdout_df[holdout_df["label"] == "default"].sort_values("seed").reset_index(drop=True)

        holdout_specs = []
        for i in range(tuning_config.n_holdout_seeds):
            seed = tuning_config.holdout_seed_offset + i
            spec = sample_episode(
                catalog=catalog,
                config=tuning_config,
                episode_seed=seed,
            )
            holdout_specs.append(spec)

        direct_kpis = evaluate_policy_normalised(lambda: OrderUpToPolicy(), holdout_specs)

        for i, row in default_rows.iterrows():
            expected_profit = direct_kpis["per_seed_net_profit"][i]
            actual_profit = row["net_profit"]
            assert actual_profit == expected_profit


class TestConfirmTopKPairedCRN:
    """Same seed → same capacity and initial_cash for all labels."""

    def test_paired_crn_same_world_per_seed(self, tmp_path):
        catalog = _make_catalog()
        tuning_config = _make_holdout_tuning_config()

        study = run_study(
            _order_up_to_space,
            catalog=catalog,
            tuning_config=tuning_config,
            study_name="crn_paired",
            output_dir=str(tmp_path),
        )
        study_dir = str(tmp_path / "crn_paired")

        confirm_top_k(
            study,
            _order_up_to_space,
            catalog=catalog,
            tuning_config=tuning_config,
            study_dir=study_dir,
        )

        holdout_df = pd.read_parquet(tmp_path / "crn_paired" / "holdout.parquet")

        for seed in holdout_df["seed"].unique():
            seed_rows = holdout_df[holdout_df["seed"] == seed]
            capacities = seed_rows["capacity"].unique()
            cash_vals = seed_rows["initial_cash"].unique()

            assert len(capacities) == 1
            assert len(cash_vals) == 1


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLISmokeFullPipeline:
    """Invoke the CLI for a full pipeline run and assert all artifacts exist."""

    def test_cli_smoke_full_pipeline(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable, "-m", "src.tuning.study",
                "--policy", "order_up_to",
                "--trials", "3",
                "--study-name", "cli_smoke",
                "--n-search-seeds", "2",
                "--n-holdout-seeds", "2",
                "--top-k", "2",
                "--episode-length", "10",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        study_dir = tmp_path / "cli_smoke"
        for artifact in [
            "trials.parquet",
            "per_seed.parquet",
            "study.json",
            "holdout.parquet",
            "holdout_summary.json",
        ]:
            assert (study_dir / artifact).exists()

    def test_cli_stdout_summary_line(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable, "-m", "src.tuning.study",
                "--policy", "order_up_to",
                "--trials", "2",
                "--study-name", "cli_stdout",
                "--n-search-seeds", "1",
                "--n-holdout-seeds", "1",
                "--top-k", "1",
                "--episode-length", "5",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        stdout = result.stdout
        assert "study=cli_stdout" in stdout
        assert "n_trials=" in stdout
        assert "best_mean_normalised_return=" in stdout
        assert "best_params=" in stdout
        assert "output=" in stdout


class TestCLISkipHoldout:
    """--skip-holdout writes only the three search-phase artifacts."""

    def test_cli_skip_holdout(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable, "-m", "src.tuning.study",
                "--policy", "order_up_to",
                "--trials", "2",
                "--study-name", "cli_skip",
                "--n-search-seeds", "1",
                "--episode-length", "5",
                "--skip-holdout",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        study_dir = tmp_path / "cli_skip"
        for artifact in ["trials.parquet", "per_seed.parquet", "study.json"]:
            assert (study_dir / artifact).exists()

        assert not (study_dir / "holdout.parquet").exists()
        assert not (study_dir / "holdout_summary.json").exists()


class TestCLIPolicyDispatch:
    """Each of the four --policy values runs to exit 0 and records the right class name."""

    _POLICY_CLASS_MAP = {
        "order_up_to": "OrderUpToPolicy",
        "reorder_point": "ReorderPointPolicy",
        "periodic_order_up_to": "PeriodicOrderUpToPolicy",
        "periodic_reorder": "PeriodicReorderPolicy",
    }

    @pytest.mark.parametrize("policy_name,expected_class", list(_POLICY_CLASS_MAP.items()))
    def test_cli_policy_dispatch(self, policy_name, expected_class, tmp_path):
        study_name = f"dispatch_{policy_name}"
        result = subprocess.run(
            [
                sys.executable, "-m", "src.tuning.study",
                "--policy", policy_name,
                "--trials", "1",
                "--study-name", study_name,
                "--n-search-seeds", "1",
                "--episode-length", "5",
                "--skip-holdout",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"CLI exited {result.returncode} for --policy {policy_name}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        study_json_path = tmp_path / study_name / "study.json"
        assert study_json_path.exists()

        with open(study_json_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        assert meta["policy_class_name"] == expected_class
