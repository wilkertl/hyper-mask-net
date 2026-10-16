"""Synthetic `transfer-matrix` result files for the E1 report and G2 draft tests."""

import json
from collections.abc import Sequence
from pathlib import Path

import torch

from hyperdime.contracts.hashing import NO_DATASET
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.schemas import EvaluationResult, dump_records
from hyperdime.evaluation.runs import METRICS
from hyperdime.evaluation.transfer import degradation_tests, seed_average, transfer_cell

ENVS = ("msmarco", "nfcorpus", "scifact")
BUDGETS = (102, 307, 512)
BASELINES = {"prefix": 0.15, "random": 0.2, "global-importance": 0.1, "global-selector": 0.05}


def _result(
    env: str, set_name: str, method: str, k: int, values: Sequence[float]
) -> EvaluationResult:
    per_query = {f"q{i}": dict.fromkeys(METRICS, v) for i, v in enumerate(values)}
    means = {m: sum(values) / len(values) for m in METRICS}
    protocol = "transfer" if ":" in method and method.split(":")[1] != env else "in-domain"
    return EvaluationResult(
        environment=env, protocol=protocol, split="test", selector=method, k=k,
        metrics=means, per_query=per_query, num_queries=len(values),
    )  # fmt: skip


def build(
    root: Path,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    shift: float = 0.1,
    within: float = 0.8,
    cross: float = 0.3,
    p_value: float = 6 / 1680,
) -> Path:
    """Selectors from another environment score ``shift`` lower per query; battery overlaps are
    ``within`` inside an environment and ``cross`` across. Returns the evaluation directory."""
    sources: dict[str, object] = {"global": "global-run"}
    for env in ENVS:
        sources[env] = {}
        for seed in seeds:
            run = f"seeded-{env}-{seed}"
            directory = root / "selectors" / "seeded" / env / f"seed-{seed}" / run
            directory.mkdir(parents=True)
            selected = "global-regularized" if env == "msmarco" else "wide-grid"
            record = {
                "seed": seed,
                "validation_ndcg@10": 0.5,
                "source": {
                    "run": f"candidates-{env}",
                    "cell": "wide-grid-t0.01-e50",
                    "selected": selected,
                    "family_scores": {
                        "wide-grid": 0.5,
                        "global-regularized": 0.51,
                        "official": 0.49,
                    },
                },
            }
            (directory / "selection.json").write_text(json.dumps(record))
            sources[env][f"seed-{seed}"] = run  # type: ignore[index]
    base = (torch.rand(20, generator=torch.Generator().manual_seed(0)) * 0.6 + 0.3).tolist()
    noise = (torch.rand(20, generator=torch.Generator().manual_seed(1)) * 0.02).tolist()
    manifest = start_manifest(
        "transfer-matrix", {"synthetic": True}, dataset_hash=NO_DATASET, seed=0
    )
    with ArtifactWriter(root, "evaluation/E1/all", manifest) as writer:
        averaged = {}
        for test in ENVS:
            for set_name in ("test", "trec-dl-2019") if test == "msmarco" else ("test",):
                results = [_result(test, set_name, "full", 1024, [v + 0.02 for v in base])]
                for k in BUDGETS:
                    step = (k - 307) / 1024  # each budget scores a little differently
                    results += [
                        _result(test, set_name, m, k, [v - d + step for v in base])
                        for m, d in BASELINES.items()
                    ]
                    for train in ENVS:
                        drop = 0.0 if train == test else shift
                        values = [v - drop + e + step for v, e in zip(base, noise, strict=True)]
                        results += [
                            _result(test, set_name, f"selector:{train}:s{s}", k, values)
                            for s in seeds
                        ]
                        results.append(
                            _result(
                                test, set_name, f"fixed:{train}", k, [v - 0.05 + step for v in base]
                            )
                        )
                dump_records(results, writer.path(f"results-{test}-{set_name}.jsonl"))
                by_name = {r.selector: r for r in results if r.k == 307}
                for train in ENVS:
                    averaged[(train, test, set_name)] = seed_average(
                        [by_name[f"selector:{train}:s{s}"].per_query for s in seeds]
                    )
        scores = {k: {q: v["ndcg@10"] for q, v in a.items()} for k, a in averaged.items()}
        cells = [
            transfer_cell(
                row,
                {
                    m: sum(v[m] for v in averaged[(row.train, row.test, row.query_set)].values())
                    / 20
                    for m in METRICS
                },
                within if row.train == row.test else cross,
                seeds,
                307,
                [f"selector:{row.train}:s{s}" for s in seeds],
            )
            for row in degradation_tests(scores)
        ]
        dump_records(cells, writer.path("cells.jsonl"))
        names = [f"selector:{env}:s{s}" for env in ENVS for s in seeds]
        pairs = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                same = a.split(":")[1] == b.split(":")[1]
                value = within if same else cross
                pairs.append({
                    "a": a, "b": b, "within": same, "overlap": value, "js": 1 - value,
                    "kl_ab": 1 - value, "kl_ba": 1 - value, "spearman_importance": value,
                    "spearman_frequency": value,
                })  # fmt: skip
        behavior = {
            "k": 307,
            "pairs": pairs,
            "statistic": within - cross,
            "p_value": p_value,
            "assignments": 1680,
            "holds": within > cross and p_value <= 0.05,
            "environment_pairs": {
                f"{a}|{b}": {"cross": cross, f"within_{a}": within, f"within_{b}": within}
                for i, a in enumerate(ENVS)
                for b in ENVS[i + 1 :]
            },
        }
        fit = {
            env: {
                "mean_gain": 0.1,
                "p_value": 0.001,
                "num_queries": 20,
                "p_holm": 0.003,
                "fits": True,
            }
            for env in ENVS
        }
        for name, payload in (
            ("behavior.json", behavior),
            ("fit.json", fit),
            ("inputs.json", sources),
            ("battery.json", {"ids": {}, "n": 20, "seed": 0, "sha256": "ab" * 32}),
        ):
            writer.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return writer.commit()
