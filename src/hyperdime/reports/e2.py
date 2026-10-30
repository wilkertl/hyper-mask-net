"""The E2 report: do space descriptors predict how a selector shifts (gate G3)? Generated from the
latest ``probe-descriptors --experiment E2`` run only (F10 research D0-D15).

A ``--targets-only`` run gives the shift check alone: per held-out environment, how well the
profile the other environments share predicts its selector target, against the seed noise ceiling.
A full run adds the descriptor ladder in both probe forms, the per-seed and secondary targets,
the coefficients, the environment-level context, and the G3 criteria (SC-001, SC-002).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.schemas import Criterion, GateDecision
from hyperdime.evaluation.descriptor_probe import LADDER, spearman, spearman_brown
from hyperdime.reports import s1
from hyperdime.reports.e0 import _csv, _fmt, _markdown, _yes

DEFAULT_ADR = Path("docs/decisions/ADR-0013-g3-space-signal.md")
TAG = "g3-space-signal"
ALPHA = 0.05
MIN_FOLDS = 2  # SC-001: at least 2 of the 3 held-out environments
NO_SHIFT = "not supported: no shift to explain"
REUSE = (
    "S1-S4 were shaped on these three environments and the targets come from selectors tuned on "
    "them, so E2's reading is exploratory; G3 is confirmed by rerunning this spec unchanged on "
    "new environments (#60)."
)
CEILING_TARGETS = ("s1", "s4_correction")


@dataclass
class E2Results:
    run: str
    commit: str
    finished: str
    inputs: dict[str, Any]
    check: dict[str, Any]
    rows: list[dict[str, Any]]
    coefficients: dict[str, Any]
    context: dict[str, Any]
    targets: dict[str, dict[str, np.ndarray]]

    @property
    def environments(self) -> list[str]:
        return list(self.inputs["environments"])

    @property
    def targets_only(self) -> bool:
        return bool(self.inputs["targets_only"])

    @property
    def seeds(self) -> list[str]:
        return [str(s) for s in self.inputs["seeds"]]

    def select(self, **fields: str) -> list[dict[str, Any]]:
        return [r for r in self.rows if all(r[k] == v for k, v in fields.items())]

    def one(self, **fields: str) -> dict[str, Any]:
        (found,) = self.select(**fields)
        return found

    @property
    def sets(self) -> list[tuple[str, str, str]]:
        """``(number, name, protocol)`` of the descriptor sets evaluated, in ladder order."""
        found = {(r["set"], r["name"], r["protocol"]) for r in self.rows if r["form"] != "—"}
        return sorted(found, key=lambda s: int(s[0]))


def load(root: Path) -> E2Results:
    directory = latest_artifact(root, "evaluation/E2", "all")
    manifest = read_manifest(directory)
    inputs = _json(directory / "inputs.json")
    probe = directory / "probe.jsonl"
    rows = [json.loads(line) for line in probe.read_text(encoding="utf-8").splitlines()] if (
        probe.is_file()
    ) else []  # fmt: skip
    targets = {}
    for env in inputs["environments"]:
        if not (directory / f"targets-{env}.json").is_file():
            continue
        keys = _json(directory / f"targets-{env}.json")["keys"]
        values = np.load(directory / f"targets-{env}.npy")
        targets[env] = dict(zip(keys, values, strict=True))
    optional = {name: _json(directory / f"{name}.json") if (directory / f"{name}.json").is_file()
                else {} for name in ("coefficients", "context")}  # fmt: skip
    return E2Results(
        directory.name,
        manifest.git_commit,
        f"{manifest.finished_at:%Y-%m-%d}" if manifest.finished_at else "",
        inputs,
        _json(directory / "check.json"),
        rows,
        optional["coefficients"],
        optional["context"],
        targets,
    )


def check_table(results: E2Results) -> list[list[str]]:
    gap = results.check["threshold"]["gap"]
    table = [["held-out", "B1 spearman", "seed spearman", "ceiling", "gap", f"room (gap >= {gap})"]]
    for env in results.environments:
        fold = results.check["folds"][env]
        table.append([env, _fmt(fold["b1"]), _fmt(fold["seed_spearman"]), _fmt(fold["ceiling"]),
                      _fmt(fold["gap"], "+"), _yes(fold["gap"] >= gap)])  # fmt: skip
    return table


def fold_passes(results: E2Results, number: str, heldout: str) -> dict[str, Any]:
    """SC-001 for one set and held-out environment: the shift probe above B1 and B2 on the
    seed-averaged target with Holm p <= ALPHA, and above both for every seed's target."""
    shift = results.one(heldout=heldout, target="s1", seed="mean", set=number, form="shift")
    b1 = results.one(heldout=heldout, target="s1", seed="mean", set="B1")["spearman"]
    b2 = results.one(heldout=heldout, target="s1", seed="mean", set="B2")["spearman"]
    seeds = all(
        results.one(heldout=heldout, target="s1", seed=s, set=number, form="shift")["spearman"]
        > max(results.one(heldout=heldout, target="s1", seed=s, set=b)["spearman"]
              for b in ("B1", "B2"))
        for s in results.seeds
    )  # fmt: skip
    above = shift["spearman"] > b1 and shift["spearman"] > b2
    significant = shift["p_holm"] is not None and shift["p_holm"] <= ALPHA
    return {"spearman": shift["spearman"], "b1": b1, "b2": b2, "p": shift["p"],
            "p_holm": shift["p_holm"], "above": above, "significant": significant,
            "seeds": seeds, "passes": above and significant and seeds}  # fmt: skip


def set_summary(results: E2Results) -> list[dict[str, Any]]:
    """Per set: the folds it passes and its mean held-out shift Spearman."""
    found = []
    for number, name, protocol in results.sets:
        folds = {env: fold_passes(results, number, env) for env in results.environments}
        found.append({
            "set": number, "name": name, "protocol": protocol, "folds": folds,
            "passed": sum(f["passes"] for f in folds.values()),
            "mean": float(np.mean([f["spearman"] for f in folds.values()])),
        })  # fmt: skip
    return found


def criteria(results: E2Results) -> list[dict[str, Any]]:
    """SC-001 and SC-002 (spec); with no shift, SC-001 is not supported and nothing is probed."""
    if results.check["no_shift"]:
        return [
            {"name": "SC-001", "value": 0.0, "shown": NO_SHIFT,
             "threshold": f">= {MIN_FOLDS} of {len(results.environments)} held-out environments",
             "met": False},
            {"name": "SC-002", "value": 0.0, "shown": "no set evaluated",
             "threshold": "P0 if a set 1-4 passes SC-001, else P1 if set 5 or 6 does",
             "met": False},
        ]  # fmt: skip
    if results.targets_only:
        raise ValueError("the latest E2 run is targets-only and found a shift; run the probes")
    summary = set_summary(results)
    best = max(summary, key=lambda s: (s["passed"], s["mean"]))
    passing = [s for s in summary if s["passed"] >= MIN_FOLDS]
    protocol = "P0" if any(s["protocol"] == "P0" for s in passing) else "P1" if passing else ""
    chosen = max((s for s in passing if s["protocol"] == protocol), key=lambda s: s["mean"],
                 default=None)  # fmt: skip
    return [
        {"name": "SC-001", "value": float(best["passed"]),
         "shown": f"set {best['set']} ({best['name']}) passes {best['passed']} of "
                  f"{len(results.environments)} held-out environments",
         "threshold": f">= {MIN_FOLDS} of {len(results.environments)} held-out environments",
         "met": bool(passing)},
        {"name": "SC-002", "value": float(chosen["set"]) if chosen else 0.0,
         "shown": f"{protocol} (set {chosen['set']}, {chosen['name']})" if chosen
                  else "no set passes SC-001",
         "threshold": "P0 if a set 1-4 passes SC-001, else P1 if set 5 or 6 does",
         "met": chosen is not None},
    ]  # fmt: skip


def criteria_table(results: E2Results) -> list[list[str]]:
    table = [["held-out", "set", "name", "protocol", "shift spearman", "B1", "B2", "p",
              "Holm p", "above both", "every seed above both", "passes"]]  # fmt: skip
    for summary in set_summary(results):
        for env, fold in summary["folds"].items():
            table.append([env, summary["set"], summary["name"], summary["protocol"],
                          _fmt(fold["spearman"]), _fmt(fold["b1"]), _fmt(fold["b2"]),
                          _opt(fold["p"]), _opt(fold["p_holm"]), _yes(fold["above"]),
                          _yes(fold["seeds"]), _yes(fold["passes"])])  # fmt: skip
    return table


def ladder_table(results: E2Results) -> list[list[str]]:
    table = [["held-out", "set", "name", "protocol", "form", "spearman", "vs B1", "vs B2",
              "penalty", "p", "Holm p"]]  # fmt: skip
    for env in results.environments:
        b1 = results.one(heldout=env, target="s1", seed="mean", set="B1")["spearman"]
        b2 = results.one(heldout=env, target="s1", seed="mean", set="B2")["spearman"]
        table.append([env, "B1", "dimension identity", "—", "—", _fmt(b1), "", "", "", "", ""])
        table.append([env, "B2", "variance", "P0", "—", _fmt(b2), "", "", "", "", ""])
        for form in ("shift", "direct"):
            for row in results.select(heldout=env, target="s1", seed="mean", form=form):
                table.append([env, row["set"], row["name"], row["protocol"], form,
                              _fmt(row["spearman"]), _fmt(row["spearman"] - b1, "+"),
                              _fmt(row["spearman"] - b2, "+"), _opt(row["penalty"]),
                              _opt(row["p"]), _opt(row["p_holm"])])  # fmt: skip
    return table


def _against_baselines(results: E2Results, rows: list[dict[str, Any]]) -> list[list[str]]:
    table = []
    for row in rows:
        same = {"heldout": row["heldout"], "target": row["target"], "seed": row["seed"]}
        b1 = results.one(**same, set="B1")["spearman"]
        b2 = results.one(**same, set="B2")["spearman"]
        table.append([row["heldout"], row["target"], row["seed"], row["set"], row["name"],
                      row["form"], _fmt(row["spearman"]), _fmt(b1), _fmt(b2)])  # fmt: skip
    return table


def seeds_table(results: E2Results) -> list[list[str]]:
    rows = [r for r in results.select(target="s1") if r["seed"] != "mean" and r["form"] != "—"]
    head = ["held-out", "target", "seed", "set", "name", "form", "spearman", "B1", "B2"]
    return [head, *_against_baselines(results, rows)]


def secondary_table(results: E2Results) -> list[list[str]]:
    rows = [r for r in results.rows if r["target"] != "s1" and r["form"] != "—"]
    head = ["held-out", "target", "seed", "set", "name", "form", "spearman", "B1", "B2"]
    return [head, *_against_baselines(results, rows)]


def ceiling_table(results: E2Results) -> list[list[str]]:
    """Per environment and seeded target: the mean pairwise seed Spearman and its step-up."""
    table = [["environment", "target", "seed spearman", "ceiling"]]
    for env in results.environments:
        for target in CEILING_TARGETS:
            seeds = [results.targets[env][f"{target}/seed-{s}/mean"] for s in results.seeds]
            pairs = itertools.combinations(seeds, 2)
            rho = float(np.mean([spearman(a, b) for a, b in pairs]))
            table.append([env, target, _fmt(rho), _fmt(spearman_brown(rho, len(seeds)))])
    return table


def coefficients_table(results: E2Results) -> list[list[str]]:
    table = [["held-out", "set", "form", "feature", "standardized coefficient"]]
    for env in results.environments:
        for name, forms in results.coefficients.get(env, {}).items():
            for form, found in forms.items():
                table += [[env, name, form, f, _fmt(v, "+")] for f, v in found.items()]
    return table


def context_table(results: E2Results) -> list[list[str]]:
    names = list(next(iter(results.context.values()))["descriptor_means"])
    table = [["environment", "participation ratio", "mean energy", "mean S4 a_k",
              "E1 degradation as test (train: nDCG@10)",
              *(f"mean {n}" for n in names)]]  # fmt: skip
    for env in results.environments:
        found = results.context[env]
        degradation = "; ".join(f"{t}: {_fmt(v, '+')}" for t, v in
                                sorted(found["e1_degradation"].items()) if t != env)  # fmt: skip
        table.append([env, _fmt(found["participation_ratio"]), _fmt(found["mean_energy"]),
                      _fmt(float(np.mean(found["s4_prior_weights"]))), degradation or "—",
                      *(_fmt(found["descriptor_means"][n]) for n in names)])  # fmt: skip
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
        f"- seeds {', '.join(results.seeds)}; validation queries per "
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
    tables |= {
        "criteria": criteria_table(results),
        "ladder": ladder_table(results),
        "seeds": seeds_table(results),
        "secondary": secondary_table(results),
        "ceiling": ceiling_table(results),
        "coefficients": coefficients_table(results),
        "context": context_table(results),
    }
    descriptors = results.inputs["descriptors"]
    evaluated = ", ".join(f"{n} `{name}` ({p})" for n, name, p in results.sets)
    missing = [f"{i} `{s.name}`" for i, s in enumerate(LADDER, 1)
               if str(i) not in {n for n, _, _ in results.sets}]  # fmt: skip
    permutations = max(r["n_permutations"] for r in results.rows)
    lines += [
        "",
        "## Descriptors and probes",
        "",
        "- descriptors: "
        + ", ".join(
            f"{e} `{descriptors[e]['run']}` ({descriptors[e]['protocol']})"
            for e in results.environments
        ),  # fmt: skip
        f"- sets evaluated: {evaluated}"
        + (f"; not evaluated (no such columns): {', '.join(missing)}" if missing else ""),
        "- shift probe (decisive, research D10): the rank of B1 plus a ridge correction fitted on",
        "  the training environments' deviations from their mean, penalty by GCV; direct probe",
        "  (secondary, D7): ridge from standardized descriptors to target ranks, penalty by",
        "  leave-one-environment-out on the training environments. Standardization and penalties",
        "  use the training environments only. B2 is the held-out environment's `variance`",
        "  selection frequency.",
        f"- permutation tests: {permutations} per fold, set, and form on the seed-averaged",
        "  primary target (descriptor rows permuted within each environment, the fit redone);",
        f"  Holm over the sets evaluated within each fold and form; alpha {ALPHA}.",
        "",
        "## G3 criteria",
        "",
        *[
            f"- **{c['name']}**: {c['shown']}; threshold {c['threshold']}; "
            f"{'met' if c['met'] else 'not met'}."
            for c in criteria(results)
        ],  # fmt: skip
        "",
        "Per held-out environment and set, shift form (`tables/criteria.csv`):",
        "",
        *_markdown(tables["criteria"]),
        "",
        "## Ladder (seed-averaged primary target)",
        "",
        *_markdown(tables["ladder"]),
        "",
        "## Noise ceilings",
        "",
        *_markdown(tables["ceiling"]),
        "",
        "## Environment-level context (descriptive, research D11)",
        "",
        *_markdown([row[:5] for row in tables["context"]]),
        "",
        "Per-seed targets: `tables/seeds.csv`; per-budget, S1 bias, and S4 correction targets:",
        "`tables/secondary.csv`; standardized coefficients: `tables/coefficients.csv`; descriptor",
        "column means per environment: `tables/context.csv`.",
    ]
    return "\n".join(lines) + "\n", {name: _csv(table) for name, table in tables.items()}


def write(root: Path, output: Path) -> Path:
    text, csvs = render(root)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(text, encoding="utf-8")
    for name, content in csvs.items():
        (output / "tables" / f"{name}.csv").write_text(content, encoding="utf-8")
    return output / "README.md"


def decision(results: E2Results, adr: Path) -> GateDecision:
    computed = criteria(results)
    check = "; ".join(
        f"{e} B1 {_fmt(f['b1'])}, ceiling {_fmt(f['ceiling'])}, gap {_fmt(f['gap'], '+')}"
        for e, f in results.check["folds"].items()
    )
    context = {"check": check, "test_set_reuse": REUSE}
    if not results.targets_only:
        context["ladder"] = "; ".join(
            f"set {s['set']} {s['name']}: "
            + ", ".join(f"{e} {_fmt(f['spearman'])} (B1 {_fmt(f['b1'])}, B2 {_fmt(f['b2'])}, "
                        f"Holm p {_opt(f['p_holm'])})" for e, f in s["folds"].items())
            for s in set_summary(results)
        )  # fmt: skip
        context["ceiling"] = "; ".join(f"{r[0]} {r[1]} {r[3]}" for r in ceiling_table(results)[1:])
        context["direct"] = "; ".join(
            f"{r['heldout']} set {r['set']} {_fmt(r['spearman'])} (Holm p {_opt(r['p_holm'])})"
            for r in results.select(target="s1", seed="mean", form="direct")
        )
    return GateDecision(
        gate="G3",
        outcome="pass" if all(c["met"] for c in computed) else "fail",
        criteria=[
            Criterion(name=c["name"], value=c["value"], threshold=c["threshold"], met=c["met"])
            for c in computed
        ],
        context=context,
        evidence=[
            results.run,
            *sorted(r["run"] for r in results.inputs.get("descriptors", {}).values()),
        ],  # fmt: skip
        adr=adr.as_posix(),
        tag=TAG,
    )


def proposed_adr(results: E2Results, record: GateDecision) -> str:
    lines = [
        "# ADR-0013: Gate G3 — does the space's geometry explain selector shift?",
        "",
        "- **Status:** proposed",
        f"- **Date:** {results.finished}",
        "- **Issue:** F10 (`specs/011-descriptor-probe`), experiment E2",
        "",
        "## Context",
        "",
        "H2 claims that per-dimension statistics of an environment's embedding space predict how",
        "its selector departs from the profile every environment shares. E2 tests it with the",
        "dimension as the unit and each environment held out in turn; the decisive form is the",
        "shift probe, which starts from that shared profile (B1). This draft is generated by",
        "`hyperdime decide G3` from the result files of `reports/E2/README.md`.",
        "",
        "## Evidence",
        "",
        f"From probe-descriptors `{results.run}` (commit `{results.commit[:12]}`); every value is",
        "in `reports/E2/README.md`.",
        "",
        *[
            f"- **{c['name']}**: {c['shown']}; threshold {c['threshold']}; "
            f"{'met' if c['met'] else 'not met'}."
            for c in criteria(results)
        ],  # fmt: skip
        "",
        "Context, not decisive:",
        "",
        *[f"- **{key}**: {value}" for key, value in sorted(record.context.items())],
        "",
        "## Decision",
        "",
        f"G3 reads **{record.outcome}** (exploratory: these environments shaped S1-S4). The",
        "decision is the researcher's: accept this reading, or amend it here with the reasons.",
        "",
        "## Consequences",
        "",
        "- If G3 holds: the descriptor is an input the selector generator can condition on;",
        "  confirm by rerunning this spec unchanged on the added environments (#60).",
        "- If G3 fails: the selector is not predicted from geometry at this granularity; the",
        "  P1 groups (stage 2) and the added environments are the next evidence.",
    ]
    return "\n".join(lines) + "\n"


def write_adr(results: E2Results, record: GateDecision, path: Path) -> Path:
    if path.exists():
        found = s1._STATUS.search(path.read_text(encoding="utf-8"))
        if found is None or found.group(1) != "proposed":
            raise ValueError(f"{path} is no longer proposed; it is not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(proposed_adr(results, record), encoding="utf-8")
    return path


def _opt(value: float | None, sign: str = "") -> str:
    return "" if value is None else _fmt(value, sign)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
