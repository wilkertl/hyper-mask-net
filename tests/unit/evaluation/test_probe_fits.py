"""E2 probes: ridge, GCV, the shift and direct probes, permutations, and the probe rows
(F10 research D7, D9, D10)."""

import numpy as np
import pytest

from hyperdime.evaluation import descriptor_probe as probe


def test_ridge_on_three_rows_and_one_feature_by_hand() -> None:
    x, y = np.array([[1.0], [2.0], [-1.0]]), np.array([2.0, 3.0, -1.0])
    # beta = sum(xy) / (sum(x^2) + lambda) = 9 / (6 + 3)
    assert probe.ridge_fit(x, y, 3.0) == pytest.approx([1.0])
    assert probe.ridge_predict(x, np.array([1.0])) == pytest.approx([1.0, 2.0, -1.0])


def test_gcv_picks_the_hand_computed_minimum_on_four_rows() -> None:
    # x = 1..4 (sum x^2 = 30), y = (1, 1, 2, 1) (sum xy = 13, sum y^2 = 7):
    # GCV(l) = 4 * (7 - 2b*13 + 30 b^2) / (4 - 30 / (30 + l))^2 with b = 13 / (30 + l);
    # 0.6061, 0.5971, 0.6509, 1.3233, 1.6937 for l = 0.1, 1, 10, 100, 1000.
    x, y = np.arange(1.0, 5.0)[:, None], np.array([1.0, 1.0, 2.0, 1.0])
    assert probe.gcv_penalty(x, y) == 1.0
    beta = 13 / 31
    assert 4 * (7 - 26 * beta + 30 * beta**2) / (4 - 30 / 31) ** 2 == pytest.approx(
        0.5971, abs=1e-4
    )


def test_standardization_uses_the_training_rows_and_drops_a_flat_feature() -> None:
    a = np.array([[1.0, 5.0], [3.0, 5.0]])
    b = np.array([[5.0, 5.0], [7.0, 5.0]])
    mean, std, kept = probe.standardizer([a, b])
    assert mean.tolist() == [4.0, 5.0] and std[0] == pytest.approx(5**0.5)
    assert kept.tolist() == [True, False]


DIM = 256


def _planted(
    signal: float = 1.0, related: bool = True
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Three environments: a shared profile plus, if ``related``, a term linear in descriptor
    column 0 (which differs per environment), plus noise; columns 1-2 are unrelated."""
    rng = np.random.default_rng(7)
    shared = rng.normal(size=DIM)
    descriptors, targets = {}, {}
    for env in ("a", "b", "c"):
        descriptors[env] = rng.normal(size=(DIM, 3))
        term = descriptors[env][:, 0] if related else rng.normal(size=DIM)
        targets[env] = shared + signal * term + 0.3 * rng.normal(size=DIM)
    return descriptors, targets


def test_the_shift_probe_finds_a_planted_signal_and_beats_both_baselines() -> None:
    descriptors, targets = _planted()
    rng = np.random.default_rng(3)
    for heldout in targets:
        fit = probe.shift_probe(descriptors, targets, heldout, ["x0", "x1", "x2"])
        b1 = probe.spearman(np.mean([t for e, t in targets.items() if e != heldout], axis=0),
                            targets[heldout])  # fmt: skip
        b2 = probe.spearman(rng.normal(size=DIM), targets[heldout])
        assert fit.spearman > b1 + 0.1 and fit.spearman > b2
        assert abs(fit.coefficients["x0"]) > 5 * max(abs(fit.coefficients[c]) for c in ("x1", "x2"))


def test_the_direct_probe_finds_the_planted_signal() -> None:
    descriptors, targets = _planted(signal=3.0)
    for heldout in targets:
        fit = probe.direct_probe(descriptors, targets, heldout, ["x0", "x1", "x2"])
        assert fit.spearman > 0.5 and fit.penalty in probe.PENALTIES
        assert max(fit.coefficients, key=lambda c: abs(fit.coefficients[c])) == "x0"


def test_an_unrelated_target_does_not_beat_b1_beyond_noise() -> None:
    descriptors, targets = _planted(related=False)
    for heldout in targets:
        fit = probe.shift_probe(descriptors, targets, heldout)
        b1 = probe.spearman(np.mean([t for e, t in targets.items() if e != heldout], axis=0),
                            targets[heldout])  # fmt: skip
        assert fit.spearman < b1 + 0.05


def test_with_beta_zero_the_shift_prediction_ranks_as_b1(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptors, targets = _planted()
    monkeypatch.setattr(probe, "ridge_fit", lambda x, y, penalty: np.zeros(x.shape[1]))
    fit = probe.shift_probe(descriptors, targets, "c")
    b1 = probe.spearman((targets["a"] + targets["b"]) / 2, targets["c"])
    assert fit.spearman == pytest.approx(b1)


@pytest.mark.parametrize("form", ["shift", "direct"])
def test_the_held_out_environment_never_enters_the_fit(form: str) -> None:
    descriptors, targets = _planted()
    fit = probe.PROBES[form](descriptors, targets, "b", ["x0", "x1", "x2"])
    changed_targets = {**targets, "b": -targets["b"]}
    changed_descriptors = {**descriptors, "b": descriptors["b"] * 10 + 3}
    for d, t in ((descriptors, changed_targets), (changed_descriptors, targets)):
        other = probe.PROBES[form](d, t, "b", ["x0", "x1", "x2"])
        assert other.penalty == fit.penalty
        assert other.coefficients == pytest.approx(fit.coefficients)


def test_the_permutation_p_counts_a_seeded_null_that_refits_every_time() -> None:
    descriptors, targets = _planted()
    calls: list[float] = []

    def counting(d: dict[str, np.ndarray], t: dict[str, np.ndarray], h: str) -> probe.ProbeFit:
        fit = probe.shift_probe(d, t, h)
        calls.append(fit.penalty or 0.0)
        return fit

    null = probe.permutation_null(counting, descriptors, targets, "a", 20, seed=0)
    assert len(calls) == 20  # the whole fit, GCV included, once per permutation
    rng = np.random.default_rng(0)
    for i in range(20):
        shuffled = {e: v[rng.permutation(DIM)] for e, v in sorted(descriptors.items())}
        assert null[i] == probe.shift_probe(shuffled, targets, "a").spearman
    observed = probe.shift_probe(descriptors, targets, "a").spearman
    assert probe.permutation_p(observed, null) == (1 + int(np.sum(null >= observed))) / 21
    assert probe.permutation_p(0.5, np.array([0.1, 0.5, 0.9])) == 0.75


def _profiles(related: bool = True) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    descriptors, targets = _planted(related=related)
    rng = np.random.default_rng(11)
    profiles: dict[str, dict] = {}
    for env, target in targets.items():
        seeds = {s: target + 0.05 * rng.normal(size=DIM) for s in (0, 1, 2)}
        found = {f"s1/seed-{s}/mean": v for s, v in seeds.items()}
        found |= {f"s1/seed-{s}/budget-{k}": v for s, v in seeds.items() for k in (10, 20)}
        found |= {"s1/mean": probe.seed_mean(seeds), "s1/bias": rng.normal(size=DIM),
                  "s4_correction/mean": rng.normal(size=DIM),
                  "variance/mean": rng.normal(size=DIM)}  # fmt: skip
        profiles[env] = found
    # Columns named as the ladder's set 1 ("doc_std") and two set-2 columns.
    full = {e: np.hstack([d, np.zeros((DIM, 6))]) for e, d in descriptors.items()}
    return full, profiles


NAMES = ["doc_std", "doc_mean", "doc_mean_abs", "doc_rms",
         "doc_q05", "doc_q25", "doc_q50", "doc_q75", "doc_q95"]  # fmt: skip


def test_e2_probes_rows_baselines_and_holm_within_fold_and_form() -> None:
    descriptors, profiles = _profiles()
    rows, coefficients = probe.e2_probes(descriptors, NAMES, profiles, [0, 1, 2], [10, 20], 10)
    primary = [r for r in rows if r["target"] == "s1" and r["seed"] == "mean"]
    assert {r["set"] for r in primary} == {"B1", "B2", "1", "2"}  # sets 3-6 need missing columns
    for heldout in ("a", "b", "c"):
        for form in ("shift", "direct"):
            fold = [r for r in primary if r["heldout"] == heldout and r["form"] == form]
            assert len(fold) == 2 and all(r["n_permutations"] == 10 for r in fold)
            expected = probe.holm({r["set"]: r["p"] for r in fold})
            assert {r["set"]: r["p_holm"] for r in fold} == expected
        assert set(coefficients[heldout]) == {"variance", "A"}
    secondary = [r for r in rows if r["seed"] != "mean" or r["target"] != "s1"]
    assert all(r["p"] is None and r["n_permutations"] == 0 for r in secondary)
    assert {r["target"] for r in rows} == {"s1", "s4_correction", "s1/bias", "s1/budget-10",
                                          "s1/budget-20"}  # fmt: skip
    assert {r["seed"] for r in rows if r["target"] == "s1"} == {"mean", "0", "1", "2"}
    shift = [r for r in primary if r["set"] == "1" and r["form"] == "shift"]
    b1 = [r for r in primary if r["set"] == "B1"]
    assert all(s["spearman"] > b["spearman"] for s, b in zip(shift, b1, strict=True))
