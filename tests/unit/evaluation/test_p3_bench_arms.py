"""F12 arm helpers: nine-budget target, nesting, global mask ties, checkpoint mapping (T030)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hyperdime.cli.p3_bench_arms import (
    INCOMPLETE,
    Checkpoint,
    _linear,
    _nesting,
    _statistics,
    _target,
    _teacher_names,
    family_table,
    search_lower_bound_hours,
)
from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.validation import ContractError
from hyperdime.evaluation.benchmark import PRIMARY
from hyperdime.evaluation.benchmark_stats import holm
from hyperdime.evaluation.teachers import top_k_dims


def test_target_averages_uniform_distributions_over_budgets() -> None:
    masks = np.zeros((1, 2, 4), dtype=bool)
    masks[0, 0, [0]] = True  # k=1
    masks[0, 1, [0, 2]] = True  # k=2
    target = _target(masks)
    assert torch.allclose(target, torch.tensor([[0.75, 0.0, 0.25, 0.0]]))
    assert torch.isclose(target.sum(), torch.tensor(1.0))


def test_nesting_share_of_the_smaller_mask_inside_the_larger() -> None:
    masks = np.zeros((2, 2, 4), dtype=bool)
    masks[0, 0, [0, 1]] = True
    masks[0, 1, [0, 1, 2]] = True  # nested
    masks[1, 0, [0, 3]] = True
    masks[1, 1, [0, 1, 2]] = True  # half inside
    assert _nesting(masks) == [0.75]


def test_global_mask_ties_go_to_the_lower_dimension() -> None:
    counts = np.array([2.0, 5.0, 5.0, 1.0, 5.0])
    assert top_k_dims(counts, 2).tolist() == [1, 2]


def test_linear_maps_official_and_s1_checkpoints_to_the_same_layer() -> None:
    weight, bias = torch.randn(4, 4), torch.randn(4)
    a = _linear({"net.0.weight": weight, "net.0.bias": bias})
    b = _linear({"projection.weight": weight, "projection.bias": bias})
    x = torch.randn(3, 4)
    assert torch.equal(a(x), b(x)) and torch.allclose(a(x), x @ weight.T + bias)


# --- escalation and family (ADR-0018, "Escalation" and "Multiplicity") -------------------------


def _record(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _resources(tmp_path: Path, cap: int | None) -> Path:
    return _record(
        tmp_path / "resources.json",
        {"kind": "resources", "corpus": "nfcorpus", "realized": {"nfcorpus": {"step_cap": cap}}},
    )


def test_teacher_names_keep_all_three_when_the_gate_set_a_cap(tmp_path: Path) -> None:
    assert _teacher_names(_resources(tmp_path, 8), None, "nfcorpus") == (
        "positive",
        "contrastive",
        "rank-guided",
    )


def test_escalated_gate_needs_a_matching_removal_record(tmp_path: Path) -> None:
    resources = _resources(tmp_path, None)
    with pytest.raises(ContractError):
        _teacher_names(resources, None, "nfcorpus")
    other = _record(
        tmp_path / "other.json",
        {
            "kind": "escalation",
            "corpus": "nfcorpus",
            "resource_record_sha256": "0",
            "option": "remove-rank-guided",
        },
    )
    with pytest.raises(ContractError):
        _teacher_names(resources, other, "nfcorpus")
    base = {
        "kind": "escalation",
        "corpus": "nfcorpus",
        "resource_record_sha256": file_sha256(resources),
    }
    not_run = _record(tmp_path / "not-run.json", {**base, "option": "not-run"})
    with pytest.raises(ContractError):
        _teacher_names(resources, not_run, "nfcorpus")
    removal = _record(tmp_path / "removal.json", {**base, "option": "remove-rank-guided"})
    assert _teacher_names(resources, removal, "nfcorpus") == ("positive", "contrastive")


def test_search_lower_bound_uses_one_scan_per_search_at_the_fastest_scan() -> None:
    searches = [{"step_seconds": [2.0, 1.5]}, {"step_seconds": [3.0]}]
    # 1.5 s x 18 searches x 100 queries x 1.5 safety = 4,050 s
    assert search_lower_bound_hours(searches, 18, 100, 1.5) == pytest.approx(4050 / 3600)


def _rows(arm_values: dict[str, list[float]]) -> list[dict]:
    rows = []
    for arm, values in arm_values.items():
        for qi, value in enumerate(values):
            for k in PRIMARY:
                rows.append({"query": f"q{qi}", "k": k, "arm": arm, "ndcg@10": value})
    return rows


def test_family_holm_and_bonferroni_use_the_family_size() -> None:
    table = _rows(
        {"l2s-official@42": [0.5] * 6, "positive@42": [0.6] * 6, "contrastive@42": [0.4] * 6}
    )
    rows = family_table(
        {"nfcorpus": table}, [("nfcorpus", "positive"), ("nfcorpus", "contrastive")]
    )
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["positive"]["D"] == pytest.approx(0.1)
    assert by_arm["contrastive"]["D"] == pytest.approx(-0.1)
    # n = 6: exact sign flips; all-equal positive diffs give p = 1/64, negative ones p = 1
    assert by_arm["positive"]["p_one_sided"] == pytest.approx(1 / 64)
    assert by_arm["positive"]["holm_p"] == pytest.approx(2 / 64)
    assert by_arm["contrastive"]["holm_p"] == 1.0
    assert by_arm["contrastive"]["claim_level"] == "not superior"
    assert by_arm["positive"]["claim_level"] == "established minimum gain"


def test_holm_step_down_is_monotone() -> None:
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_statistics_skip_arms_that_were_not_trained() -> None:
    table = _rows(
        {
            "l2s-official@42": [0.5, 0.6],
            "positive@42": [0.4, 0.5],
            "positive@0": [0.4, 0.5],
            "positive@1": [0.4, 0.5],
            "positive@2": [0.4, 0.5],
            "variance": [0.3, 0.3],
            "full": [0.5, 0.5],
        }
    )
    summary = _statistics(table, "nfcorpus", ["q0", "q1"], family_size=8)
    assert list(summary["primary"]) == ["positive"]
    assert set(summary["secondary"]) == {"variance", "full"}
    assert summary["family_size"] == 8


# --- large-corpus paths equal the direct paths ------------------------------------------------


def _large_case() -> tuple[np.ndarray, np.ndarray, list[tuple[list[int], list[float]]]]:
    rng = np.random.default_rng(11)
    docs = rng.standard_normal((300, 1024)).astype(np.float16)
    queries = rng.standard_normal((3, 1024)).astype(np.float16)
    positives = [([4, 250], [1.0, 2.0]), ([17], [1.0]), ([299, 0, 8], [1.0, 1.0, 1.0])]
    return docs, queries, positives


def test_tune_score_large_equals_the_direct_score() -> None:
    from hyperdime.cli.p3_bench_arms import _tune_score, _tune_score_large
    from hyperdime.evaluation.benchmark import CorpusScorer

    docs, queries, positives = _large_case()
    torch.manual_seed(0)
    model = torch.nn.Linear(1024, 1024)
    direct = _tune_score(
        model, queries, positives, torch.from_numpy(docs.astype(np.float64)), torch.device("cpu")
    )
    scorer = CorpusScorer(docs, "cpu", chunk_rows=64, resident=False)
    assert _tune_score_large(model, queries, positives, scorer) == direct


def test_evaluate_large_rows_equal_the_direct_rows() -> None:
    from hyperdime.cli.p3_bench_arms import _evaluate_large
    from hyperdime.evaluation.benchmark import (
        DECILES,
        CorpusScorer,
        masked_scores,
        positive_ranks,
        query_measures,
    )

    docs, queries, positives = _large_case()
    full_docs = torch.from_numpy(docs.astype(np.float64))
    importances = {"a@42": torch.randn(3, 1024), "b@42": torch.randn(3, 1024)}
    variance = full_docs.var(0, unbiased=False)
    scorer = CorpusScorer(docs, "cpu", chunk_rows=64)
    for qi, (rows, grades) in enumerate(positives):
        q = torch.from_numpy(queries[qi].astype(np.float64))
        pos, gains = torch.tensor(rows), torch.tensor(grades, dtype=torch.float64)
        got = _evaluate_large(
            scorer, f"q{qi}", q, pos, gains, importances, qi, q * q * variance, None
        )
        expected = []
        full = query_measures(positive_ranks(masked_scores(full_docs, q, None), pos), gains)
        for k in DECILES:
            expected.append({"query": f"q{qi}", "k": k, "arm": "full", **full})
            arms = {n: torch.topk(imp[qi], k).indices for n, imp in importances.items()}
            arms["variance"] = torch.topk(q * q * variance, k).indices
            arms["mrl-prefix"] = torch.arange(k)
            for name, dims in arms.items():
                ranks = positive_ranks(masked_scores(full_docs, q, dims), pos)
                expected.append(
                    {"query": f"q{qi}", "k": k, "arm": name, **query_measures(ranks, gains)}
                )
        assert got == expected


def test_checkpoint_round_trips_blocks_in_order(tmp_path: Path) -> None:
    ckpt = Checkpoint(tmp_path, "teachers", "scifact", {"run": "a"}, None)
    ckpt.save(0, 2, {"m": np.array([[True], [False]])}, {"rows": [1, 2]})
    ckpt.save(2, 3, {"m": np.array([[True]])}, {"rows": [3]})
    blocks = Checkpoint(tmp_path, "teachers", "scifact", {"run": "a"}, None).blocks()
    assert [(a, b) for a, b, _, _ in blocks] == [(0, 2), (2, 3)]
    assert np.concatenate([arr["m"] for _, _, arr, _ in blocks]).ravel().tolist() == [
        True,
        False,
        True,
    ]
    assert [r for *_, meta in blocks for r in meta["rows"]] == [1, 2, 3]


def test_checkpoint_never_resumes_another_configuration(tmp_path: Path) -> None:
    Checkpoint(tmp_path, "evaluate", "scifact", {"lock": "a"}, None).save(0, 1, {}, {"rows": []})
    assert Checkpoint(tmp_path, "evaluate", "scifact", {"lock": "b"}, None).blocks() == []


def test_checkpoint_refuses_a_gap(tmp_path: Path) -> None:
    ckpt = Checkpoint(tmp_path, "evaluate", "scifact", {}, None)
    ckpt.save(0, 2, {}, {"rows": []})
    ckpt.save(3, 4, {}, {"rows": []})
    with pytest.raises(ContractError, match="gap"):
        ckpt.blocks()


def test_checkpoint_expires_only_with_a_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert not Checkpoint(tmp_path, "e", "scifact", {}, None).expired()
    stopped = Checkpoint(tmp_path, "e", "scifact", {}, 0.0)
    assert stopped.expired()
    assert stopped.stop(50, 120) == INCOMPLETE == 75
    assert json.loads(capsys.readouterr().out)["done"] == 50


# --- S1 selection, statistics batches, parameter counts (T057) ----------------------------------


def _cell(index: int, score: float, guard: bool = True) -> dict:
    return {"cell": index, "tune_mean_ndcg10": score, "passes_guard": guard}


def test_s1_selection_takes_the_best_eligible_cell_and_ties_go_to_the_lower_index() -> None:
    from hyperdime.cli.p3_bench_arms import select_cell

    cells = [_cell(0, 0.5), _cell(1, 0.7), _cell(2, 0.7), _cell(3, 0.9, guard=False)]
    assert select_cell(cells)["cell"] == 1  # cell 3 is best but fails the soft-mass guard


def test_s1_selection_refuses_when_no_cell_passes_the_guard() -> None:
    from hyperdime.cli.p3_bench_arms import select_cell

    with pytest.raises(ContractError, match="soft-mass guard"):
        select_cell([_cell(0, 0.5, guard=False)])


def test_s1_step_0_is_a_candidate_and_a_tie_keeps_it() -> None:
    """With lr = 0 every checkpoint ties, so the kept state is step 0's, the starting selector."""
    from hyperdime.training.examples import RetrievalExamples
    from hyperdime.training.retrieval_trainer import RetrievalConfig, train_retrieval_candidates

    gen = torch.Generator().manual_seed(0)
    n, dim = 16, 8
    queries = torch.nn.functional.normalize(torch.randn(n, dim, generator=gen), dim=-1)
    documents = torch.cat([queries + 0.1 * torch.randn(n, dim, generator=gen)] * 2)
    examples = RetrievalExamples(
        queries,
        documents,
        torch.arange(n)[:, None],
        torch.ones(n, 1),
        (torch.arange(n) + n)[:, None],
        skipped=0,
    )
    config = RetrievalConfig(
        budgets=(4,), eval_budgets=(4,), lr=0.0, steps=6, batch_size=4, eval_every=3
    )
    start = {"projection.weight": torch.eye(dim), "projection.bias": torch.zeros(dim)}
    model, history, _ = train_retrieval_candidates(examples, examples, config, start)
    assert history[0]["step"] == 0 and len(history) == 3
    assert torch.equal(model.state_dict()["projection.weight"], torch.eye(dim))


@pytest.mark.parametrize("n", [1, 121, 388, 7_500, 10_000_000])
def test_statistics_batches_stay_within_ten_million_elements(n: int) -> None:
    from hyperdime.evaluation.benchmark_stats import DRAWS, MAX_ELEMENTS, _batch

    assert 1 <= _batch(n, DRAWS) <= DRAWS and _batch(n, DRAWS) * n <= MAX_ELEMENTS


def test_teacher_arms_and_l2s_official_have_the_same_parameter_count() -> None:
    official = {"net.0.weight": torch.zeros(1024, 1024), "net.0.bias": torch.zeros(1024)}
    s1 = {"projection.weight": torch.zeros(1024, 1024), "projection.bias": torch.zeros(1024)}
    counts = {sum(p.numel() for p in _linear(state).parameters()) for state in (official, s1)}
    assert counts == {1024 * 1024 + 1024}


# --- generated report: framing, provenance, byte-identical regeneration (T046, T053, T054) -------


def _provenance() -> dict:
    return {
        "split": "p3-query-split-v1",
        "split_manifest_sha256": "e" * 64,
        "split_fingerprints": {"fit": "a" * 64, "tune": "b" * 64, "evaluation": "c" * 64},
        "l2s_commit": "55f010f5f3e82a6d171ce0909793e72f9c066ec7",
        "transfer_flag": "τ = 0.10 and 50 epochs were published elsewhere.",
    }


def test_the_report_regenerates_byte_identically_and_carries_scope_and_provenance() -> None:
    from hyperdime.cli.p3_bench_arms import _report
    from hyperdime.reports.p3_bench import FRAMING

    values = [0.5, 0.6, 0.55]
    table = _rows(
        {
            "l2s-official@42": values,
            **{f"positive@{s}": [v - 0.1 for v in values] for s in (42, 0, 1, 2)},
            "mrl-prefix": [0.4, 0.4, 0.4],
            "full": [0.5, 0.5, 0.5],
        }
    )
    summary = _statistics(table, "msmarco", ["q0", "q1", "q2"], family_size=8)
    summary["provenance"] = _provenance()
    stored = json.loads(json.dumps(summary, sort_keys=True))  # as written to summary.json
    first, second = _report(stored, "msmarco", "run"), _report(stored, "msmarco", "run")
    assert first == second
    assert FRAMING in first and "a" * 64 in first and "55f010f5" in first
    assert "Transfer flag (US1/AC4)" in first
    assert "| mrl-prefix |" in first  # the prefix-k control is a secondary row


def test_a_summary_without_provenance_says_so() -> None:
    from hyperdime.cli.p3_bench_arms import _report

    table = _rows({"l2s-official@42": [0.5], **{f"positive@{s}": [0.4] for s in (42, 0, 1, 2)}})
    summary = _statistics(table, "scifact", ["q0"], family_size=8)
    assert "Provenance: not stored in this summary" in _report(summary, "scifact", "run")


# --- resource gates (T051) ----------------------------------------------------------------------


def _large_probe(step_seconds: dict[int, float], steps: dict[int, list[int]]) -> dict:
    """A large-corpus probe whose only cost is the rank-guided search, so sizes follow from it."""
    searches = [
        {"shortlist": n, "steps": s, "seconds": s * step_seconds[n], "step_seconds":
         [step_seconds[n]] * s}
        for n in (10_000, 1_000)
        for s in steps[n]
    ]  # fmt: skip
    return {
        "labels_run_summary": {"seconds": {"labels": 0.0, "train": 0.0}, "n_fit": 1, "n_valid": 1},
        "scoring": {"seconds_per_query_mask": 0.0, "seconds_per_top_pass": 0.0},
        "s1": {"seconds": 0.0, "steps": 1},
        "search": {"queries": len(steps[1_000]), "searches": searches},
    }


def test_large_plan_sizes_follow_the_priority_order_and_take_the_first_fitting_shortlist() -> None:
    from hyperdime.cli.p3_bench import large_resource_plan

    # budget after the probe: 69 h / 1.5 = 165,600 s. Shortlist 1,000: 15 s per query, so
    # n_rg 8,000 fits (120,000 s) and 16,000 does not; shortlist 10,000 (30 s) fits nothing.
    probe = _large_probe({1_000: 5.0, 10_000: 10.0}, {1_000: [3] * 10, 10_000: [3] * 10})
    plan = large_resource_plan(probe, 40_000, 8_000, 8_000, (3.0, 72.0))
    realized = plan["realized"]["msmarco"]
    assert (realized["n_eval"], realized["n_rg"], realized["n_tune"]) == (7_500, 8_000, 7_500)
    assert realized["shortlist"] == 1_000 and realized["step_cap"] == 8
    assert plan["rule"] == "smallest cap with >= 95% of searches stopping"
    assert plan["realized_without_rank_guided"]["n_rg"] == 35_000


def test_large_plan_takes_the_smallest_cap_meeting_95_percent() -> None:
    from hyperdime.cli.p3_bench import large_resource_plan

    # one search in ten runs to 8 steps: 90% stop before 8, all before 32
    steps = {1_000: [3] * 9 + [8], 10_000: [3] * 9 + [8]}
    probe = _large_probe({1_000: 0.001, 10_000: 0.001}, steps)
    realized = large_resource_plan(probe, 40_000, 8_000, 8_000, (3.0, 72.0))["realized"]["msmarco"]
    assert realized["shortlist"] == 10_000 and realized["step_cap"] == 32


def test_large_plan_escalates_below_minimum_coverage() -> None:
    from hyperdime.cli.p3_bench import large_resource_plan

    probe = _large_probe({1_000: 100.0, 10_000: 100.0}, {1_000: [3] * 4, 10_000: [3] * 4})
    plan = large_resource_plan(probe, 40_000, 8_000, 8_000, (3.0, 72.0))
    assert plan["realized"]["msmarco"]["step_cap"] is None
    assert plan["rule"] == "escalate: minimum coverage over budget"
    # without search the same order applies, at the largest sizes that fit
    assert plan["realized_without_rank_guided"]["n_rg"] == 35_000
    assert plan["realized"]["msmarco"]["n_rg"] == 35_000


def _small_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: list[int], step_s: float):
    import argparse

    from hyperdime.cli import p3_bench
    from hyperdime.data import query_split

    split = tmp_path / query_split.SPLIT_MANIFEST
    split.parent.mkdir(parents=True)
    ids = {"fit": sorted(f"f{i:03d}" for i in range(100)), "tune": ["t0"], "evaluation": ["e0"]}
    split.write_text(
        json.dumps(
            {
                "seed": 20261112,
                "corpora": {
                    "scifact": {
                        "source_dataset_run": "run",
                        "source_queries_sha256": "a" * 64,
                        "excluded_test_overlap_ids": [],
                        "counts": {k: len(v) for k, v in ids.items()},
                        "query_ids": ids,
                    }
                },
            }
        )
    )
    monkeypatch.setitem(query_split.load_query_split.__kwdefaults__, "expected_sha256",
                        file_sha256(split))  # fmt: skip
    probe_dir = tmp_path / "evaluation" / "P3-bench" / "probe" / "scifact" / "p"
    probe_dir.mkdir(parents=True)
    (probe_dir / "manifest.json").write_text("{}")
    searches = [{"steps": s, "seconds": s * step_s, "step_seconds": [step_s] * s} for s in steps]
    (probe_dir / "probe.json").write_text(
        json.dumps(
            {
                "scoring": {"seconds_per_query_mask": 0.0},
                "s1": {"seconds": 0.0, "steps": 1},
                "search": {"queries": 1, "searches": searches},
            }
        )
    )
    out = tmp_path / "resources.json"
    args = argparse.Namespace(root=tmp_path, corpus="scifact", probe_run="p", out=out)
    code = p3_bench.cmd_freeze_resources(args)
    return code, json.loads(out.read_text())


def test_small_gate_takes_the_smallest_cap_where_95_percent_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, record = _small_gate(tmp_path, monkeypatch, [3] * 19 + [10], 0.01)  # 95% < 8
    assert code == 0 and record["realized"]["scifact"]["step_cap"] == 8
    assert record["rule"] == "smallest cap with >= 95% of searches stopping before it"


def test_small_gate_escalates_when_no_cap_fits_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 100 fit queries x 20 searches x 3 steps x 10 s = 16.7 h, over SciFact's 8 h at every cap
    code, record = _small_gate(tmp_path, monkeypatch, [3] * 20, 10.0)
    assert code == 3 and record["realized"]["scifact"]["step_cap"] is None
    assert record["rule"] == "escalate: minimum coverage over budget"


def test_freeze_records_are_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from hyperdime.cli import p3_bench

    _small_gate(tmp_path, monkeypatch, [3] * 20, 0.01)
    again = argparse.Namespace(
        root=tmp_path, corpus="scifact", probe_run="p", out=tmp_path / "resources.json"
    )
    with pytest.raises(ContractError, match="never overwritten"):
        p3_bench.cmd_freeze_resources(again)
