"""Small pieces of the P3 report; tests/integration/test_p3_pipeline covers the whole report."""

import numpy as np

from hyperdime.reports.p3 import _bars, _spread, _ticks
from hyperdime.reports.p3_controls import _meets
from hyperdime.reports.p3_shared import cohort_positions


def test_ticks_enclose_the_data_on_round_values() -> None:
    assert _ticks(0.0, 0.92) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ticks = _ticks(-0.003, 0.0123)
    assert ticks[0] <= -0.003 and ticks[-1] >= 0.0123


def test_a_spread_without_values_reads_as_not_available() -> None:
    assert _spread({"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}) == "n/a"
    assert _spread({"n": 2, "mean": 0.5, "p10": 0.25, "p50": 0.5, "p90": 0.75}) == (
        "0.500 (0.250 to 0.750)"
    )


def test_a_bar_with_no_value_is_left_out_rather_than_drawn_at_zero() -> None:
    svg = _bars("t", "s", "x", "y", ["a", "b"], {"one": [0.5, None], "two": [0.25, 0.75]})
    assert svg.count("<path") == 3


def test_a_curve_that_dips_is_not_described_as_staying_above_a_limit() -> None:
    assert _meets([0.9, 0.7, 0.9, None], 0.8) == "at 2 of the 3 defined k (first 1, last 3)"
    assert _meets([0.1, None], 0.8) == "at none of the 1 defined k"


def test_common_cohort_rows_select_the_same_queries_from_cells_of_different_eligibility() -> None:
    counts = np.array([2, 4, 40, 3, 50])
    assert cohort_positions(counts, 2, 40).tolist() == [2, 4]  # the cell keeps all five queries
    assert cohort_positions(counts, 4, 40).tolist() == [1, 2]  # it keeps 4, 40, 50
    assert cohort_positions(counts, 40, 40).tolist() == [0, 1]  # it keeps 40, 50
