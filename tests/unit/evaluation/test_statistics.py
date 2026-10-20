"""The one-sided non-inferiority sign-flip test (F9 research D7)."""

import itertools

import pytest

from hyperdime.evaluation.statistics import noninferiority_test, paired_randomization_test


def _exact(diffs: list[float], margin: float) -> float:
    """The share of all 2^n sign flips of ``d + margin`` whose mean reaches the observed one."""
    shifted = [d + margin for d in diffs]
    observed = sum(shifted) / len(shifted)
    flips = list(itertools.product((-1.0, 1.0), repeat=len(shifted)))
    hits = sum(sum(s * v for s, v in zip(signs, shifted, strict=True)) / len(shifted)
               >= observed - 1e-12 for signs in flips)  # fmt: skip
    return hits / len(flips)


@pytest.mark.parametrize(
    ("diffs", "margin", "share"),
    [
        # all four shifted differences are 0.1: only the all-plus flip reaches 0.1, 1 of 16
        ([0.1, 0.1, 0.1, 0.1], 0.0, 1 / 16),
        # shifted [0, 0.04, 0.02, 0.02]: the 0's sign is free, the rest must be plus, 2 of 16
        ([-0.01, 0.03, 0.01, 0.01], 0.01, 2 / 16),
    ],
)
def test_the_p_value_is_the_share_of_sign_flips_reaching_the_shifted_mean(
    diffs: list[float], margin: float, share: float
) -> None:
    assert _exact(diffs, margin) == share
    p = noninferiority_test(diffs, [0.0] * len(diffs), margin)
    assert p == pytest.approx(share, abs=0.01)


def test_with_no_margin_it_is_the_upper_half_of_the_two_sided_test() -> None:
    a, b = [0.5, 0.6, 0.7, 0.4, 0.55, 0.65], [0.5, 0.55, 0.6, 0.45, 0.5, 0.6]
    one_sided, two_sided = noninferiority_test(a, b, 0.0), paired_randomization_test(a, b)
    assert one_sided <= two_sided
    assert one_sided == pytest.approx(two_sided / 2, abs=0.02)


def test_a_clearly_better_candidate_is_non_inferior_and_a_clearly_worse_one_is_not() -> None:
    reference = [0.3 + 0.01 * i for i in range(40)]
    better = [v + 0.02 for v in reference]
    worse = [v - 0.02 for v in reference]
    assert noninferiority_test(better, reference, 0.005) < 0.001
    assert noninferiority_test(worse, reference, 0.005) > 0.99


def test_unpaired_values_are_refused() -> None:
    with pytest.raises(ValueError, match="paired"):
        noninferiority_test([0.1, 0.2], [0.1], 0.005)
