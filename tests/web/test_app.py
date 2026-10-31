"""Streamlit page — opens on results and round-trips knobs through the URL."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_APP = Path(__file__).resolve().parent.parent.parent / "src" / "web" / "app.py"


def _app(**params: str) -> AppTest:
    at = AppTest.from_file(str(_APP), default_timeout=60)
    for key, value in params.items():
        at.query_params[key] = value
    return at


def _card(at: AppTest):
    return at.dataframe[0].value.set_index("Policy")


def test_first_load_shows_scorecard_for_all_four_policies_with_r_3():
    at = _app().run()
    assert not at.exception
    card = _card(at)
    assert len(card) == 4
    assert at.slider(key="review_interval").value == 3
    # No periodic death spiral at the default R: every policy actually sells.
    assert (card["Fill rate"] > 0.5).all()
    assert at.query_params["R"] == ["3"]
    assert at.query_params["preset"] == ["three_node_chain"]
    assert not at.warning


def test_changing_a_knob_reruns_at_once_and_updates_the_link():
    at = _app().run()
    at.slider(key="review_interval").set_value(7).run()
    assert not at.exception
    assert at.query_params["R"] == ["7"]
    # The known periodic-review starvation at long R is visible, not 1.0.
    assert _card(at).loc["(R, S)", "Fill rate"] < 0.5


def test_link_knobs_drive_the_run():
    at = _app(
        preset="two_factories_two_shops",
        seed="7",
        policies="order_up_to,periodic_reorder",
    ).run()
    assert not at.exception
    assert at.selectbox(key="preset").value == "Two shops compete (6 nodes)"
    assert at.number_input(key="world_seed").value == 7
    assert list(_card(at).index) == ["(s, S)", "(R, s, S)"]
    assert at.query_params["policies"] == ["order_up_to,periodic_reorder"]


def test_bad_link_values_are_replaced_and_named_in_a_notice():
    at = _app(
        preset="nope",
        steps="0",
        R="oops",
        seed="99999999999",
        policies="order_up_to,bogus",
    ).run()
    assert not at.exception
    notice = at.warning[0].value
    for named in ("nope", "steps=0", "R=oops", "seed=99999999999", "bogus"):
        assert named in notice
    assert at.slider(key="n_steps").value == 1
    assert at.number_input(key="world_seed").value == 2**31 - 1
    assert at.query_params["steps"] == ["1"]


def test_crafted_link_values_show_as_plain_text_not_links_or_images():
    at = _app(
        preset="[Your account is locked](https://evil.example)",
        policies="![x](https://tracker.example/p.png)",
    ).run()
    assert not at.exception
    notice = at.warning[0].value
    assert "](" not in notice and "https://" not in notice
    assert r"\[Your account is locked\]" in notice
    assert r"\!\[x\]" in notice


@pytest.mark.parametrize(
    "value", ["https://evil.example/login", "www.evil.example", "me@evil.example"]
)
def test_bare_url_link_values_are_not_autolinked(value):
    notice = _app(preset=value).run().warning[0].value
    # GFM autolinks these literal forms; the visitor still reads the value.
    for form in ("https:", "www.", "@evil", ".example"):
        assert form not in notice.replace("\\", "")
    assert value in notice.replace("\u2060", "").replace("\\", "")


def test_two_shops_row_is_the_same_whatever_else_is_ticked():
    link = dict(preset="two_factories_two_shops", seed="42", steps="30")
    together = _card(_app(**link).run()).loc["(R, S)", "Net profit"]
    alone = _card(_app(**link, policies="periodic_order_up_to").run())
    assert alone.loc["(R, S)", "Net profit"] == together


def test_two_shops_shows_shop_2_results():
    at = _app(preset="two_factories_two_shops").run()
    rival = at.dataframe[1].value
    assert "shop-2 fill rate" in rival.columns
    assert len(rival) == 4


def test_network_starts_on_the_best_profit_policy():
    at = _app().run()
    card = _card(at)
    best_short = card["Net profit"].idxmax()
    shown = at.radio(key="picture")
    assert shown.format_func(shown.value) == best_short


def test_delta_column_shows_a_dash_when_s_s_is_not_ticked():
    at = _app(policies="reorder_point").run()
    # The grid draws a null cell as "None" whatever the Styler says, so the
    # data the grid receives must already hold the dash.
    assert _card(at)["Δ vs (s, S)"].tolist() == ["—"]


def test_identical_rows_are_called_out():
    at = _app(R="14", policies="periodic_order_up_to,periodic_reorder").run()
    assert any("came out identical" in c.value for c in at.caption)
