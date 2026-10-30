"""The E2 report from a synthetic probe-descriptors run (F10 research D0)."""

import json
from pathlib import Path

import numpy as np
import pytest

from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.reports import e2

ENVS = ("scifact", "nfcorpus", "msmarco")


def _fold(b1: float, ceiling: float) -> dict[str, float]:
    return {"b1": b1, "seed_spearman": ceiling / (3 - 2 * ceiling), "ceiling": ceiling,
            "gap": ceiling - b1}  # fmt: skip


def write_check_run(root: Path, folds: dict[str, dict[str, float]], no_shift: bool) -> Path:
    inputs = {"environments": list(ENVS), "targets_only": True, "seeds": [0, 1, 2]}
    inputs |= {env: {"validation_queries": 100} for env in ENVS}
    check = {"folds": folds, "no_shift": no_shift, "threshold": {"gap": 0.05, "min_folds": 2}}
    manifest = start_manifest("probe-descriptors", {}, seed=0)
    with ArtifactWriter(root, "evaluation/E2/all", manifest) as w:
        for name, payload in (("inputs.json", inputs), ("check.json", check)):
            w.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return w.commit()


def test_a_targets_only_run_gives_the_shift_check_and_says_descriptors_were_not_evaluated(
    tmp_path: Path,
) -> None:
    folds = {"scifact": _fold(0.80, 0.95), "nfcorpus": _fold(0.70, 0.90),
             "msmarco": _fold(0.93, 0.95)}  # fmt: skip
    write_check_run(tmp_path / "artifacts", folds, no_shift=False)
    readme = e2.write(tmp_path / "artifacts", tmp_path / "E2")
    text = readme.read_text()
    rows = (tmp_path / "E2" / "tables" / "check.csv").read_text().splitlines()
    assert rows[0] == "held-out,B1 spearman,seed spearman,ceiling,gap,room (gap >= 0.05)"
    assert rows[1] == "scifact,0.8000,0.8636,0.9500,+0.1500,yes"
    assert rows[3].endswith(",+0.0200,no")
    assert "there is an environment-specific shift for descriptors to explain" in text
    assert "Descriptors were not evaluated in this run" in text
    assert "exploratory" in text and "no full or prefix rows" in text


def test_the_report_states_no_shift_and_regenerates_identically(tmp_path: Path) -> None:
    folds = {env: _fold(0.94, 0.95) for env in ENVS}
    write_check_run(tmp_path / "artifacts", folds, no_shift=True)
    first = e2.write(tmp_path / "artifacts", tmp_path / "E2").read_bytes()
    assert b"no environment-specific shift beyond seed noise" in first
    second = e2.write(tmp_path / "artifacts", tmp_path / "E2").read_bytes()
    assert first == second


SETS = {"1": ("variance", "P0"), "2": ("A", "P0"), "3": ("A+E", "P0"), "4": ("A+E+tails", "P0")}
DIM = 8


def _row(env: str, target: str, seed: str, number: str, form: str, rho: float,
         p_holm: float | None = None) -> dict:  # fmt: skip
    baselines = {"B1": ("dimension identity", "—"), "B2": ("variance", "—")}
    name, protocol = {**SETS, **baselines}[number]
    return {"heldout": env, "target": target, "seed": seed, "set": number, "name": name,
            "protocol": protocol, "form": form, "spearman": rho,
            "penalty": None if form == "—" else 1.0, "p": p_holm, "p_holm": p_holm,
            "n_permutations": 0 if p_holm is None else 100}  # fmt: skip


def probe_rows(passing: dict[str, set[str]], failing_seed: tuple[str, str] | None = None) -> list:
    """B1 0.8 and B2 0.5 everywhere; a set in ``passing[env]`` reaches 0.9 with Holm p 0.01 on
    every target and seed, the others 0.7 with p 0.5; ``failing_seed`` = (env, seed) drops the
    passing sets of that seed to 0.75."""
    rows = []
    for env in ENVS:
        for target, seed in [
            ("s1", "mean"),
            *(("s1", s) for s in "012"),
            ("s4_correction", "mean"),
        ]:
            rows += [
                _row(env, target, seed, "B1", "—", 0.8),
                _row(env, target, seed, "B2", "—", 0.5),
            ]
            for form in ("shift", "direct"):
                for number in SETS:
                    good = number in passing.get(env, set())
                    rho = 0.9 if good else 0.7
                    if (env, seed) == failing_seed and good:
                        rho = 0.75
                    p = (0.01 if good else 0.5) if (target, seed) == ("s1", "mean") else None
                    rows.append(_row(env, target, seed, number, form, rho, p))
    return rows


def write_full_run(root: Path, rows: list, no_shift: bool = False) -> Path:
    inputs = {"environments": list(ENVS), "targets_only": False, "seeds": [0, 1, 2],
              "descriptors": {e: {"run": f"desc-{e}", "protocol": "P0"} for e in ENVS}}  # fmt: skip
    inputs |= {env: {"validation_queries": 100} for env in ENVS}
    folds = {env: _fold(0.8, 0.95) for env in ENVS}
    check = {"folds": folds, "no_shift": no_shift, "threshold": {"gap": 0.05, "min_folds": 2}}
    rng = np.random.default_rng(0)
    context = {e: {"participation_ratio": 50.0, "mean_energy": 0.25, "s4_prior_weights": [1.0, 0.5],
                   "descriptor_means": {"doc_std": 0.03, "doc_mean": 0.0},
                   "e1_degradation": {o: 0.01 for o in ENVS if o != e}} for e in ENVS}  # fmt: skip
    coefficients = {e: {"A+E": {"shift": {"doc_std": 0.5, "red_corr_top8": -0.25}}} for e in ENVS}
    manifest = start_manifest("probe-descriptors", {}, seed=0)
    with ArtifactWriter(root, "evaluation/E2/all", manifest) as w:
        payloads = {"inputs.json": inputs, "check.json": check, "context.json": context,
                    "coefficients.json": coefficients}  # fmt: skip
        for name, payload in payloads.items():
            w.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        w.path("probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for env in ENVS:
            keys = [f"{t}/seed-{s}/mean" for t in ("s1", "s4_correction") for s in range(3)]
            np.save(w.path(f"targets-{env}.npy"), rng.normal(size=(len(keys), DIM)))
            w.path(f"targets-{env}.json").write_text(json.dumps({"keys": keys}))
        return w.commit()


def test_sc001_counts_folds_where_one_set_passes_and_sc002_names_its_protocol(
    tmp_path: Path,
) -> None:
    write_full_run(tmp_path, probe_rows({"scifact": {"3"}, "nfcorpus": {"3"}}))
    found = {c["name"]: c for c in e2.criteria(e2.load(tmp_path))}
    assert found["SC-001"]["value"] == 2.0 and found["SC-001"]["met"]
    assert found["SC-002"]["shown"] == "P0 (set 3, A+E)" and found["SC-002"]["met"]
    rows = e2.criteria_table(e2.load(tmp_path))
    assert rows[0][-4:] == ["Holm p", "above both", "every seed above both", "passes"]
    assert ["scifact", "3", "A+E", "P0", "0.9000", "0.8000", "0.5000", "0.0100", "0.0100",
            "yes", "yes", "yes"] in rows  # fmt: skip


def test_different_sets_in_different_folds_do_not_add_up(tmp_path: Path) -> None:
    write_full_run(tmp_path, probe_rows({"scifact": {"2"}, "nfcorpus": {"3"}}))
    found = {c["name"]: c for c in e2.criteria(e2.load(tmp_path))}
    assert found["SC-001"]["value"] == 1.0 and not found["SC-001"]["met"]
    assert not found["SC-002"]["met"] and found["SC-002"]["shown"] == "no set passes SC-001"


def test_a_seed_below_a_baseline_fails_the_fold(tmp_path: Path) -> None:
    rows = probe_rows({"scifact": {"3"}, "nfcorpus": {"3"}}, failing_seed=("nfcorpus", "1"))
    write_full_run(tmp_path, rows)
    found = {c["name"]: c for c in e2.criteria(e2.load(tmp_path))}
    assert found["SC-001"]["value"] == 1.0 and not found["SC-001"]["met"]


def test_the_full_report_lists_its_tables_and_regenerates_identically(tmp_path: Path) -> None:
    write_full_run(tmp_path / "artifacts", probe_rows({"scifact": {"3"}, "nfcorpus": {"3"}}))
    first = e2.write(tmp_path / "artifacts", tmp_path / "E2").read_bytes()
    tables = {p.stem for p in (tmp_path / "E2" / "tables").glob("*.csv")}
    assert tables == {"check", "criteria", "ladder", "seeds", "secondary", "ceiling",
                      "coefficients", "context"}  # fmt: skip
    text = first.decode()
    assert "sets evaluated: 1 `variance` (P0), 2 `A` (P0), 3 `A+E` (P0), 4 `A+E+tails` (P0)" in text
    assert "not evaluated (no such columns): 5 `+B+C`, 6 `+D`" in text
    assert "no full or prefix rows" in text and "exploratory" in text
    assert "**SC-001**: set 3 (A+E) passes 2 of 3" in text
    coefficients = (tmp_path / "E2" / "tables" / "coefficients.csv").read_text().splitlines()
    assert "scifact,A+E,shift,red_corr_top8,-0.2500" in coefficients
    assert e2.write(tmp_path / "artifacts", tmp_path / "E2").read_bytes() == first


def test_the_g3_decision_and_its_proposed_adr(tmp_path: Path) -> None:
    write_full_run(tmp_path, probe_rows({"scifact": {"3"}, "nfcorpus": {"3"}}))
    results = e2.load(tmp_path)
    adr = tmp_path / "ADR-0013.md"
    record = e2.decision(results, adr)
    assert (record.gate, record.outcome, record.tag) == ("G3", "pass", "g3-space-signal")
    assert [c.name for c in record.criteria] == ["SC-001", "SC-002"]
    assert set(record.context) == {"check", "ladder", "ceiling", "direct", "test_set_reuse"}
    assert record.evidence[1:] == sorted(f"desc-{e}" for e in ENVS)
    text = e2.write_adr(results, record, adr).read_text()
    assert "- **Status:** proposed" in text and "G3 reads **pass**" in text


def test_with_no_shift_sc001_reads_not_supported(tmp_path: Path) -> None:
    folds = {env: _fold(0.94, 0.95) for env in ENVS}
    write_check_run(tmp_path, folds, no_shift=True)
    results = e2.load(tmp_path)
    record = e2.decision(results, tmp_path / "ADR-0013.md")
    assert record.outcome == "fail" and set(record.context) == {"check", "test_set_reuse"}
    text = e2.proposed_adr(results, record)
    assert "not supported: no shift to explain" in text


def test_a_targets_only_run_with_a_shift_cannot_be_decided(tmp_path: Path) -> None:
    write_check_run(tmp_path, {env: _fold(0.5, 0.95) for env in ENVS}, no_shift=False)
    with pytest.raises(ValueError, match="run the probes"):
        e2.decision(e2.load(tmp_path), tmp_path / "ADR-0013.md")


def test_an_adr_no_longer_proposed_is_not_overwritten(tmp_path: Path) -> None:
    write_full_run(tmp_path, probe_rows({"scifact": {"3"}, "nfcorpus": {"3"}}))
    adr = tmp_path / "ADR-0013.md"
    adr.write_text("# ADR-0013\n\n- **Status:** accepted\n")
    results = e2.load(tmp_path)
    with pytest.raises(ValueError, match="no longer proposed"):
        e2.write_adr(results, e2.decision(results, adr), adr)
    assert "accepted" in adr.read_text()
