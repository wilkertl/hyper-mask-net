"""The E2 report: do space descriptors predict how a selector shifts (gate G3)? Generated from the
latest ``probe-descriptors --experiment E2`` run only (F10 research D0-D15).

A ``--targets-only`` run gives the shift check alone: per held-out environment, how well the
profile the other environments share predicts its selector target, against the seed noise ceiling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.reports.e0 import _csv, _fmt, _markdown, _yes

DEFAULT_ADR = Path("docs/decisions/ADR-0013-g3-space-signal.md")
REUSE = (
    "S1-S4 were shaped on these three environments and the targets come from selectors tuned on "
    "them, so E2's reading is exploratory; G3 is confirmed by rerunning this spec unchanged on "
    "new environments (#60)."
)


@dataclass
class E2Results:
    run: str
    commit: str
    inputs: dict[str, Any]
    check: dict[str, Any]

    @property
    def environments(self) -> list[str]:
        return list(self.inputs["environments"])

    @property
    def targets_only(self) -> bool:
        return bool(self.inputs["targets_only"])


def load(root: Path) -> E2Results:
    directory = latest_artifact(root, "evaluation/E2", "all")
    return E2Results(
        directory.name,
        read_manifest(directory).git_commit,
        _json(directory / "inputs.json"),
        _json(directory / "check.json"),
    )


def check_table(results: E2Results) -> list[list[str]]:
    gap = results.check["threshold"]["gap"]
    table = [["held-out", "B1 spearman", "seed spearman", "ceiling", "gap", f"room (gap >= {gap})"]]
    for env in results.environments:
        fold = results.check["folds"][env]
        table.append([env, _fmt(fold["b1"]), _fmt(fold["seed_spearman"]), _fmt(fold["ceiling"]),
                      _fmt(fold["gap"], "+"), _yes(fold["gap"] >= gap)])  # fmt: skip
    return table


def render(root: Path) -> tuple[str, dict[str, str]]:
    results = load(root)
    tables = {"check": check_table(results)}
    threshold = results.check["threshold"]
    reading = (
        "no environment-specific shift beyond seed noise: the shared profile reaches the ceiling"
        if results.check["no_shift"]
        else "there is an environment-specific shift for descriptors to explain"
    )
    lines = [
        "# E2: do space descriptors predict selector shift? (gate G3)",
        "",
        "Unit: the dimension; each environment is held out in turn. Target: the S1 selector's",
        "selection frequency per dimension on validation queries (embeddings only, no judgments),",
        "divided by k and averaged over the 10%-90% budgets, then over seeds. E2 scores no",
        "retrieval, so it has no full or prefix rows.",
        "",
        f"- probe-descriptors `{results.run}` (commit `{results.commit[:12]}`)",
        f"- seeds {', '.join(str(s) for s in results.inputs['seeds'])}; validation queries per "
        "environment: "
        + ", ".join(f"{e} {results.inputs[e]['validation_queries']}" for e in results.environments),
        f"- {REUSE}",
        "",
        "## Shift check (research D0)",
        "",
        "B1 predicts the held-out environment by the mean of the other environments' seed-averaged",
        "profiles; the ceiling is the mean pairwise Spearman between seeds stepped up to the",
        "three-seed average by Spearman-Brown. The gap is what a descriptor could explain. Rule",
        f"(fixed before any target): no shift when the gap is below {threshold['gap']} in at least",
        f"{threshold['min_folds']} environments.",
        "",
        *_markdown(tables["check"]),
        "",
        f"Reading: {reading}.",
    ]
    if results.targets_only:
        lines += ["", "Descriptors were not evaluated in this run (`--targets-only`)."]
    return "\n".join(lines) + "\n", {name: _csv(table) for name, table in tables.items()}


def write(root: Path, output: Path) -> Path:
    text, csvs = render(root)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(text, encoding="utf-8")
    for name, content in csvs.items():
        (output / "tables" / f"{name}.csv").write_text(content, encoding="utf-8")
    return output / "README.md"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
