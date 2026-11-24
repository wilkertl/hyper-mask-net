"""P3-bench report pieces: cost, provenance, teacher report (F12 T044-T046, T048, T052)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperdime.baselines.l2s_official import transfer_flag
from hyperdime.evaluation.benchmark import PRIMARY
from hyperdime.reports.p3_bench import (
    FRAMING,
    PREFIX_MISSING,
    S1_LIMITATION,
    arm_costs,
    cost_lines,
    provenance_lines,
    search_gains,
    selector_forward_seconds,
    teacher_report,
)


def _run(root: Path, relative: str, files: dict[str, object], seconds: float = 10.0) -> None:
    run = root / relative
    run.mkdir(parents=True)
    manifest = {
        "run_id": run.name,
        "device": "cuda:0",
        "started_at": "2026-11-12T00:00:00+00:00",
        "finished_at": f"2026-11-12T00:00:{int(seconds):02d}+00:00",
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    for name, payload in files.items():
        (run / name).write_text(json.dumps(payload))


@pytest.fixture
def runs(tmp_path: Path) -> dict[str, str]:
    c = "scifact"
    _run(tmp_path, f"selectors/l2s-official/{c}/o", {"summary.json": {
        "seconds": {"labels": 0.5, "train": 1.5, "total": 2.0}}})  # fmt: skip
    _run(tmp_path, f"evaluation/P3-bench/teachers/{c}/t", {"summary.json": {"seconds": 100.0}})
    _run(tmp_path, f"selectors/p3-teacher-arms/{c}/a", {"summary.json": {"seconds": 2.5}})
    cells = [{"seconds": 3.0}] * 8
    _run(tmp_path, f"selectors/p3-s1/{c}/scratch/s", {"selection.json": {
        "cells": cells, "selection_candidates": 168}}, seconds=40)  # fmt: skip
    return {"official": "o", "teachers": "t", "arm:positive@42": "a", "s1:scratch": "s"}


def test_costs_come_from_the_run_files(tmp_path: Path, runs: dict[str, str]) -> None:
    probe = {"scoring": {"seconds_per_query_mask": 0.002}}
    arms = ["l2s-official@42", "positive@42", "s1-scratch@42", "full", "variance", "mrl-prefix"]
    costs = arm_costs(tmp_path, "scifact", runs, arms, probe, 0.0001, "cpu")
    offline = costs["offline"]
    assert offline["l2s-official@42"]["training_seconds"] == 1.5
    assert offline["l2s-official@42"]["upstream_seconds"] == 0.5
    assert offline["positive@42"] == {
        "training_seconds": 2.5,
        "upstream_seconds": 100.0,
        "upstream": "teacher mask generation (shared by every teacher arm)",
        "wall_seconds": 10.0,
        "device": "cuda:0",
        "selection_candidates": 1,
    }
    assert offline["s1-scratch"]["training_seconds"] == 24.0
    assert offline["s1-scratch"]["selection_candidates"] == 168
    assert offline["s1-scratch"]["wall_seconds"] == 40.0
    assert offline["mrl-prefix"]["selection_candidates"] == 0
    assert "global-mask" not in offline  # only arms that were evaluated
    assert costs["online"]["masked_scoring_seconds_per_query"] == 0.002


def test_cost_lines_separate_offline_and_online_and_state_the_s1_limitation(
    tmp_path: Path, runs: dict[str, str]
) -> None:
    costs = arm_costs(tmp_path, "scifact", runs, ["full"], None, 0.0001, "cpu")
    text = "\n".join(cost_lines(costs))
    assert "## Offline cost" in text and "## Online cost per query" in text
    assert S1_LIMITATION in text
    assert "| s1-scratch | cuda:0 | 24.0 | — | 40.0 | 168 |" in text
    assert "not measured by this corpus's probe" in text
    assert PREFIX_MISSING in text  # no mrl-prefix arm was evaluated here


def test_provenance_lines_name_the_split_commit_and_transfer() -> None:
    provenance = {
        "split": "p3-query-split-v1",
        "split_manifest_sha256": "e" * 64,
        "split_fingerprints": {"fit": "f" * 64, "tune": "t" * 64, "evaluation": "v" * 64},
        "l2s_commit": "55f010f5f3e82a6d171ce0909793e72f9c066ec7",
        "transfer_flag": transfer_flag("msmarco"),
    }
    text = "\n".join(provenance_lines(provenance))
    assert "e" * 64 in text and "v" * 64 in text and "55f010f5" in text
    assert "50k-passage subset" in text and "0.10" in text
    assert transfer_flag("scifact") is None
    assert "not stored" in provenance_lines(None)[0]
    assert "not a comparison with published numbers" in FRAMING


def test_search_gains_count_chosen_searches_and_their_cost() -> None:
    searches = [
        {"chosen": True, "start_value": [0.5, 0.1], "end_value": [0.6, 0.1], "seconds": 1.0},
        {"chosen": False, "start_value": [0.4, 0.1], "end_value": [0.4, 0.1], "seconds": 1.0},
        {"chosen": True, "start_value": [0.5, 0.1], "end_value": [0.5, 0.3], "seconds": 2.0},
    ]
    gains = search_gains(searches)
    assert gains["chosen_searches"] == 2 and gains["improved"] == 2
    assert gains["mean_delta_ndcg10"] == pytest.approx(0.05)
    assert gains["search_seconds_total"] == 4.0
    assert search_gains([]) == {"chosen_searches": 0}


def test_teacher_report_shows_nesting_gain_and_demotions_by_kind() -> None:
    summary = {
        "teachers": ["positive", "contrastive", "rank-guided"],
        "queries": 4,
        "step_cap": 8,
        "seconds": 12.5,
        "device": "cpu",
        "nesting": {t: [1.0] * (len(PRIMARY) - 1) for t in ("positive", "contrastive")}
        | {"rank-guided": [0.99] * (len(PRIMARY) - 1)},
        "searches": 72,
        "searches_improved": 3,
        "searches_hit_cap": 0,
        "searches_rejected": 0,
        "rank_guided_from": {"positive": 30, "contrastive": 6},
        "demotions": {"unjudged": 5, "known_negative": 2},
    }
    gains = {"chosen_searches": 36, "improved": 3, "mean_delta_ndcg10": 0.01,
             "max_delta_ndcg10": 0.2, "mean_delta_gdr": 0.001, "search_seconds_total": 9.0,
             "search_seconds_mean": 0.125}  # fmt: skip
    text = teacher_report(summary, "scifact", "run", gains)
    assert text == teacher_report(summary, "scifact", "run", gains)
    assert "| rank-guided | 0.9900 |" in text and "102→204" in text
    assert "| 5 | 2 |" in text  # unjudged, known negatives
    assert "| 36 | 3 | +0.01000 |" in text
    without = teacher_report({**summary, "teachers": ["positive", "contrastive"]}, "c", "r")
    assert "No rank-guided search ran" in without


def test_selector_forward_time_is_measured() -> None:
    assert 0 < selector_forward_seconds("cpu", repeats=3) < 1
