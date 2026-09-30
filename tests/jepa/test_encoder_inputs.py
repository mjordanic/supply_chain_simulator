"""Encoder inputs for the latent replenishment model.

Seams, from the issue acceptance criteria:

1. Sales window. Its length is the encoder's own knob, longer than the
   longest delivery lead time the episode sampler draws.
2. Feature dimension. One width for every sampled delivery lag, so one
   fitted model can read every episode.
3. Delivery lead time. The nominal lag is a feature, the same contract
   the textbook buyer reads.
4. Arrival-age profile. In-transit units keep their ages. Held-out
   next-in-transit accuracy on that profile clears the 0.75 ceiling the
   summed scalar imposes.
5. Hidden state. The Market multiplier, the disruption identity, and the
   calendar are not features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.jepa.dynamics import (
    ARRIVAL_AGE_SLOTS,
    SALES_WINDOW,
    DirectDynamics,
    feature_dim,
    log_row_features,
    sku_features,
)
from src.jepa.plan import ReplenishmentPlanner, fit_direct
from src.sim.central_table import CentralTable, Offer
from src.tuning.episode import _LAG_MAX, _LEAD_OFFSETS


def _vector(**overrides) -> list[float]:
    values = dict(
        on_hand=4,
        in_transit=[0, 5, 1],
        sales_window=[8, 9],
        cash=80.0,
        capacity=30,
        delivery_lag=2,
    )
    values.update(overrides)
    return sku_features(**values)


def test_sales_window_is_its_own_knob_above_the_delivery_lead_time():
    longest_edge = _LAG_MAX + max(_LEAD_OFFSETS)
    assert SALES_WINDOW > longest_edge
    assert SALES_WINDOW > _LAG_MAX

    short = _vector(sales_window_length=SALES_WINDOW, delivery_lag=2)
    other = _vector(sales_window_length=SALES_WINDOW, delivery_lag=6)
    assert len(short) == len(other) == feature_dim()
    knob = _vector(sales_window_length=SALES_WINDOW + 4, delivery_lag=2)
    assert len(knob) == feature_dim(SALES_WINDOW + 4)
    assert len(knob) != len(short)

    sales = short[1 + ARRIVAL_AGE_SLOTS : 1 + ARRIVAL_AGE_SLOTS + SALES_WINDOW]
    assert sales[-2:] == [8.0, 9.0]
    assert sales[:-2] == [0.0] * (SALES_WINDOW - 2)
    assert len(sales) > 6


def test_feature_dimension_does_not_follow_the_sampled_delivery_lag():
    lag_two = _vector(delivery_lag=2, in_transit=[0, 3], sales_window=[1, 1])
    lag_six = _vector(
        delivery_lag=6,
        in_transit=[0, 1, 2, 3, 4, 5],
        sales_window=[1, 2, 3, 4, 5, 6],
    )
    assert len(lag_two) == len(lag_six) == feature_dim()
    assert lag_two[-1] == 2.0
    assert lag_six[-1] == 6.0
    assert DirectDynamics(delivery_lag=2).x_mean.shape[0] == feature_dim()
    assert DirectDynamics(delivery_lag=6).x_mean.shape[0] == feature_dim()


def test_one_fitted_model_reads_every_episode_lag():
    def rows(lag: int, episode: int) -> list[dict]:
        built = []
        for tick in range(lag + 2):
            built.append(
                {
                    "episode_seed": episode,
                    "product_id": "A",
                    "tick": tick,
                    "on_hand": float(tick),
                    "in_transit": [0, tick, 0],
                    "sales_window": [1, tick],
                    "cash": 50.0,
                    "capacity": 80,
                    "filled_qty": 1.0,
                    "next_on_hand": float(tick),
                    "next_in_transit": [0, 1, 0],
                    "next_sales": 1.0,
                    "delivery_lag": lag,
                }
            )
        return built

    frame = pd.DataFrame(rows(2, 1) + rows(6, 2))
    fitted = fit_direct(frame, delivery_lag=2, epochs=1, seed=0, budget=1, policy_seed=0)
    first = log_row_features(frame.iloc[0].to_dict(), delivery_lag=2)
    later = log_row_features(frame.iloc[-1].to_dict(), delivery_lag=2)
    assert len(first) == len(later) == feature_dim()
    assert first[-1] == 2.0
    assert later[-1] == 6.0
    predicted = fitted.dynamics.predict_next([first, later], [1.0, 1.0])
    assert len(predicted) == 2


class _CaptureRollout:
    def step(self, granted: list[float]) -> list[tuple[float, float, float]]:
        return [(0.0, 0.0, 0.0) for _ in granted]


class _CaptureDynamics:
    """Records the vector ``decide`` encodes. Does not fit."""

    def __init__(self) -> None:
        self.delivery_lag = 2
        self.features: list[list[list[float]]] = []

    def open_rollout(self, features: list[list[float]]) -> _CaptureRollout:
        self.features.append(features)
        return _CaptureRollout()


def test_decide_uses_the_episode_lead_time_not_the_fit_lag():
    """One fitted planner serves a later episode whose nominal lag differs."""
    captured = _CaptureDynamics()
    planner = ReplenishmentPlanner(
        captured,  # type: ignore[arg-type]
        delivery_lag=2,
        budget=1,
        policy_seed=0,
        order_fee=0.0,
    )
    table = CentralTable()
    table.publish("F_A", "A", Offer(available_qty=100, list_price=4.0, min_order=0))
    obs = {
        "tick": 1,
        "inventory": {"A": 0},
        "pending": {},
        "observed_sales": {"A": 0},
        "base_prices": {"A": 10.0},
        "list_prices": {"A": 10.0},
        "capacity": 100,
        "cash": 1_000.0,
        "direct_supplier_ids": {"F_A"},
        "delivery_lags": {"A": 6},
    }
    planner.decide(obs, table)
    first = list(captured.features)
    assert first
    assert all(batch[0][-1] == 6.0 for batch in first)

    bare = dict(obs)
    del bare["delivery_lags"]
    planner.decide(bare, table)
    second = captured.features[len(first):]
    assert second
    assert all(batch[0][-1] == 2.0 for batch in second)


def test_delivery_lead_time_and_arrival_ages_are_features():
    young = _vector(in_transit=[0, 9, 0], delivery_lag=4)
    old = _vector(in_transit=[0, 0, 9], delivery_lag=4)
    assert young != old
    assert sum(young[1 : 1 + ARRIVAL_AGE_SLOTS]) == 9.0
    assert sum(old[1 : 1 + ARRIVAL_AGE_SLOTS]) == 9.0
    assert young[1 : 1 + ARRIVAL_AGE_SLOTS][:3] == [0.0, 9.0, 0.0]
    assert old[1 : 1 + ARRIVAL_AGE_SLOTS][:3] == [0.0, 0.0, 9.0]
    assert young[-3:] == [80.0, 30.0, 4.0]


def test_market_disruption_and_calendar_are_not_encoder_inputs():
    row = {
        "on_hand": 4,
        "in_transit": [0, 5, 1],
        "sales_window": [8, 9],
        "cash": 80.0,
        "capacity": 30,
        "delivery_lag": 4,
        "tick": 3,
    }
    poisoned = dict(row)
    poisoned["market_multiplier"] = 123456.0
    poisoned["disruption"] = "pandemic"
    poisoned["tick"] = 987654
    poisoned["simulation_date"] = "2026-09-30"
    plain = log_row_features(row, delivery_lag=2)
    marked = log_row_features(poisoned, delivery_lag=2)
    assert plain == marked
    assert 123456.0 not in marked
    assert 987654.0 not in marked
    assert row["delivery_lag"] == marked[-1]


def _held_out_r2(features: list[list[float]], target: list[float], episodes: list[int]) -> float:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    eps = np.asarray(episodes)
    test = eps >= 30

    def design(matrix: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(matrix)), matrix])

    coef, *_ = np.linalg.lstsq(design(x[~test]), y[~test], rcond=None)
    pred = design(x[test]) @ coef
    y_test = y[test]
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    ss_res = float(np.sum((y_test - pred) ** 2))
    return 1.0 - ss_res / ss_tot


def test_arrival_age_profile_lifts_held_out_next_in_transit_above_the_scalar_ceiling():
    """A summed pipeline cannot say which cohort lands next.

    With a lead of three and iid order sizes, ordinary least squares on
    the sum plus the filled quantity has a population R² of 0.75. The
    arrival-age profile makes the same target linear. Held-out accuracy
    has to clear that ceiling.
    """
    rng = np.random.default_rng(0)
    lead = 3
    profile_rows: list[list[float]] = []
    scalar_rows: list[list[float]] = []
    target: list[float] = []
    episodes: list[int] = []
    for episode in range(40):
        orders = rng.integers(0, 40, size=28)
        for tick in range(lead, 28):
            profile = [0, int(orders[tick - 1]), int(orders[tick - 2])]
            filled = int(orders[tick])
            features = sku_features(
                on_hand=0,
                in_transit=profile,
                sales_window=[0],
                cash=10.0,
                capacity=100,
                delivery_lag=lead,
            )
            profile_rows.append([*features, float(filled)])
            scalar_rows.append(
                [float(sum(profile)), float(filled), 10.0, 100.0, float(lead)]
            )
            target.append(float(profile[1] + filled))
            episodes.append(episode)

    profile_r2 = _held_out_r2(profile_rows, target, episodes)
    scalar_r2 = _held_out_r2(scalar_rows, target, episodes)
    assert profile_r2 > 0.75
    assert profile_r2 > scalar_r2 + 0.1
