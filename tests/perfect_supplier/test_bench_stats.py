import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from src.perfect_supplier.bench import stats


def test_bca_matches_the_t_interval_for_a_normal_mean():
    x = np.random.default_rng(1).normal(0.3, 1.0, 400)
    lo, hi = stats.bca_ci(x, n_resamples=20_000)
    t_lo, t_hi = sps.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=sps.sem(x))
    assert lo == pytest.approx(t_lo, abs=0.01) and hi == pytest.approx(t_hi, abs=0.01)


def test_bca_matches_scipy_on_skewed_data():
    x = np.random.default_rng(2).exponential(1.0, 60)
    lo, hi = stats.bca_ci(x, n_resamples=20_000)
    ref = sps.bootstrap((x,), np.mean, method="BCa", n_resamples=20_000, random_state=0).confidence_interval
    assert lo == pytest.approx(ref.low, abs=0.02) and hi == pytest.approx(ref.high, abs=0.02)
    # Skewed data: BCa shifts the interval right of the symmetric percentile one.
    assert hi - x.mean() > x.mean() - lo


def test_bca_of_a_paired_ratio_matches_scipy():
    rng = np.random.default_rng(3)
    ref = rng.uniform(0.90, 0.98, 200)
    arm = ref + rng.normal(0.003, 0.004, 200)
    lo, hi = stats.bca_ci([arm, ref], stats.cost_reduction_stat, n_resamples=20_000)
    want = sps.bootstrap((arm, ref), stats.cost_reduction_stat, paired=True, method="BCa",
                         n_resamples=20_000, random_state=0).confidence_interval
    assert lo == pytest.approx(want.low, abs=0.005) and hi == pytest.approx(want.high, abs=0.005)


def test_cost_reduction_is_the_share_of_lost_margin_removed():
    assert stats.cost_reduction_stat(np.array([0.96]), np.array([0.95])) == pytest.approx(0.2)


def test_wilcoxon_exact_small_case():
    # All five differences positive: exact two-sided p = 2 / 2**5.
    assert stats.wilcoxon_p(np.array([1.0, 2, 3, 4, 5])) == pytest.approx(0.0625)
    assert stats.wilcoxon_p(np.zeros(10)) == 1.0


def test_holm_known_case_and_statsmodels():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(stats.holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(stats.holm(p), multipletests(p, method="holm")[1])


def test_bh_known_case_and_statsmodels():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(stats.bh(p), [0.02, 0.04, 0.04, 0.02])
    q = np.random.default_rng(4).uniform(0, 0.2, 25)
    np.testing.assert_allclose(stats.bh(q), multipletests(q, method="fdr_bh")[1])


def test_tost_separates_equivalent_from_shifted():
    rng = np.random.default_rng(5)
    assert stats.tost(rng.normal(0.0, 0.005, 500), sesoi=0.001) < 0.05
    assert stats.tost(rng.normal(0.002, 0.005, 500), sesoi=0.001) > 0.5


def test_two_level_reduces_to_the_world_bootstrap_with_one_seed_and_widens_with_seed_noise():
    rng = np.random.default_rng(6)
    d = rng.normal(0.002, 0.003, 300)
    mean, lo, hi = stats.two_level_ci(np.repeat(d[:, None], 3, axis=1), n_resamples=4000)
    assert mean == pytest.approx(d.mean())
    b_lo, b_hi = stats.bca_ci(d, n_resamples=4000)
    assert lo == pytest.approx(b_lo, abs=3e-4) and hi == pytest.approx(b_hi, abs=3e-4)
    # A per-seed offset shared by every world is invisible to the world bootstrap.
    noisy = d[:, None] + np.array([-0.002, 0.0, 0.002])[None, :]
    _, n_lo, n_hi = stats.two_level_ci(noisy, n_resamples=4000)
    assert n_hi - n_lo > 2 * (hi - lo)


def test_cvar_is_the_mean_of_the_worst_tenth():
    assert stats.cvar(np.arange(100.0)) == pytest.approx(4.5)


def test_decide_labels_and_holm_step_down():
    rng = np.random.default_rng(7)
    ref = rng.uniform(0.9, 0.98, 500)
    sesoi = 0.001
    deltas = {"big": 0.003, "small": 0.0004, "zero": 0.0, "worse": -0.003}
    arms = {k: ref + v + rng.normal(0, 0.002, 500) for k, v in deltas.items()}
    fam = [stats.compare(arms[k], ref, (k, "ref"), sesoi, n_resamples=2000) for k in deltas]
    stats.decide(fam, [arms[k] - ref for k in deltas], sesoi, n_resamples=2000)
    got = {c.arm: c.label for c in fam}
    assert got["big"] == "practically superior"
    assert got["small"] == "superior"
    assert got["zero"] == "equivalent"
    assert got["worse"] == "inferior"
    # Holm-adjusted intervals are wider than the unadjusted ones.
    assert all(c.ci_adj[0] <= c.ci[0] + 1e-12 for c in fam)
