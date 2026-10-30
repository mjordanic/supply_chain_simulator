"""Streamlit page — opens on results and round-trips knobs through the URL."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_APP = Path(__file__).resolve().parent.parent.parent / "src" / "web" / "app.py"


def _app() -> AppTest:
    return AppTest.from_file(str(_APP), default_timeout=60)


def test_first_load_shows_scorecard_for_all_four_policies_with_r_3():
    at = _app().run()
    assert not at.exception
    card = at.dataframe[0].value
    assert len(card) == 4
    assert at.slider(key="review_interval").value == 3
    # No periodic death spiral at the default R: every policy actually sells.
    assert (card["Fill rate"] > 0.5).all()
    assert at.query_params["R"] == ["3"]
    assert at.query_params["preset"] == ["three_node_chain"]


def test_link_knobs_drive_the_run_and_bad_values_fall_back_or_clamp():
    at = _app()
    at.query_params["preset"] = "two_factories_two_shops"
    at.query_params["seed"] = "7"
    at.query_params["steps"] = "999"
    at.query_params["R"] = "oops"
    at.query_params["policies"] = "order_up_to,periodic_reorder"
    at.run()
    assert not at.exception
    assert at.selectbox(key="preset").value == "Two shops compete (6 nodes)"
    assert at.number_input(key="world_seed").value == 7
    assert at.slider(key="n_steps").value == 60
    assert at.slider(key="review_interval").value == 3
    assert len(at.dataframe[0].value) == 2
    assert at.query_params["policies"] == ["order_up_to,periodic_reorder"]
    assert at.query_params["steps"] == ["60"]


def test_compare_reruns_with_new_knobs_and_updates_the_link():
    at = _app().run()
    at.slider(key="review_interval").set_value(7)
    at.button[0].click().run()
    assert not at.exception
    assert at.query_params["R"] == ["7"]
    card = at.dataframe[0].value.set_index("Policy")
    # The known periodic-review starvation at long R is now visible, not 1.0.
    assert card.loc["(R, S) periodic order-up-to", "Fill rate"] < 0.5
