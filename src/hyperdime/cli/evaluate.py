"""``hyperdime evaluate``: every E0 method at every budget on test and evaluation sets (F3 US4).

This is the only F3 step that reads test and evaluation-set judgments, and only to score. The
oracle mines negatives for the evaluated queries from their own labels: it is a non-deployable
upper bound that never feeds training or selection.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.methods import (
    global_importance,
    oracle_scores,
    prefix_importance,
    random_importance,
    selector_importance,
)
from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.schemas import dump_records
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ContractError
from hyperdime.data.environments import spec_for
from hyperdime.data.negatives import sample_from_rankings
from hyperdime.data.prepare import load_environment
from hyperdime.embeddings.cache import load_embeddings
from hyperdime.evaluation.runs import QuerySet, evaluate_methods, ratio_budgets
from hyperdime.oracle.targets import oracle_pairs, targets_at
from hyperdime.retrieval.exact import exact_search_streamed
from hyperdime.training.inputs import judgments, load_inputs
from hyperdime.training.selection import load_selector

COMMAND = "evaluate"
TREC_THRESHOLD = 2
ORACLE_DEPTH, ORACLE_COUNT, ORACLE_SEED = 1000, 64, 0
G1_K = 256


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="evaluate every E0 method at every budget")
    add_config_arguments(parser)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(
        args,
        lambda config, output: _evaluate(config, output, args),
        validate=lambda config: spec_for(config.data),
    )


def _evaluate(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    environment = spec_for(config.data).environment
    inputs = load_inputs(output, environment)
    bundle = load_environment(inputs.dataset_dir, include_test_qrels=True)
    device = resolve_device(config.run.device)
    official_dir = latest_artifact(output, "selectors/official", environment)
    official = json.loads((official_dir / "selection.json").read_text(encoding="utf-8"))
    temperature = official["cells"][official["chosen"]]["cell"]["temperature"]
    mean_target = targets_at(*inputs.supervision.train, temperature).mean(dim=0)
    selectors = {"l2s": load_selector(official_dir, "official")[1]}
    sources = {"official": official_dir.name}
    try:
        candidates_dir = latest_artifact(output, "selectors/candidates", environment)
        selectors["selected"] = load_selector(candidates_dir, "selected")[1]
        sources["candidates"] = candidates_dir.name
    except ContractError:
        pass  # candidates not trained yet: E0 reports the official selector only
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=bundle.manifest.fingerprint,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    sets = [("test", "queries", bundle.query_ids("test"), judgments(bundle, "test"), 1, 100)]
    for name, set_queries in sorted(bundle.eval_queries.items()):
        sets.append(
            (
                name,
                f"queries-{name}",
                sorted(set_queries),
                judgments(bundle, name),
                TREC_THRESHOLD,
                1000,
            )
        )
    budgets = sorted(set(ratio_budgets()) | {G1_K})
    checks: dict[str, Any] = {}
    with ArtifactWriter(output, f"evaluation/{environment}", manifest) as writer:
        for name, matrix, ids, labels, threshold, depth in sets:
            record, vectors = load_embeddings(inputs.embeddings_dir, matrix)
            rows = {qid: i for i, qid in enumerate(record.ids)}
            queries = torch.from_numpy(
                np.asarray(vectors[[rows[q] for q in ids]], dtype=np.float32)
            )
            methods: dict[str, Tensor | None] = {
                "full": None,
                "prefix": prefix_importance(queries),
                "random": random_importance(queries, seed=0),
                "global-importance": global_importance(queries, mean_target),
                **{m: selector_importance(queries, model) for m, model in selectors.items()},
                "oracle": _oracle(queries, ids, labels, vectors, rows, inputs, device),
            }
            pairs = [("full", 1024)] + [(m, k) for m in methods if m != "full" for k in budgets]
            evaluation = evaluate_methods(
                QuerySet(name, ids, queries, labels, threshold, depth),
                methods, budgets, inputs.documents, inputs.doc_ids,
                environment=environment, device=device, chunk_size=args.chunk_size,
                keep_k=(1024,), pairs=pairs,
            )  # fmt: skip
            dump_records(evaluation.results, writer.path(f"results-{name}.jsonl"))
            full = evaluation.rankings[("full", 1024)]
            checks[name] = {
                "k1024_identical": {m: evaluation.rankings[(m, 1024)] == full for m in methods},
                "excluded_queries": evaluation.excluded_queries,
                "relevance_threshold": threshold,
                "depth": depth,
            }
        writer.path("checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
        writer.path("inputs.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
        return writer.commit()


def _oracle(
    queries: Tensor,
    ids: Sequence[str],
    labels: Mapping[str, Mapping[str, int]],
    vectors: np.ndarray,
    rows: Mapping[str, int],
    inputs: Any,
    device: str,
) -> Tensor:
    """Raw oracle scores from the evaluated queries' own labels and freshly sampled negatives."""
    doc_rows = {doc_id: i for i, doc_id in enumerate(inputs.doc_ids)}
    depth = min(ORACLE_DEPTH, len(inputs.doc_ids))
    ranked = exact_search_streamed(queries, inputs.documents, depth, device=device)
    excluded = [{d for d, g in labels.get(q, {}).items() if g > 0} | {q} for q in ids]
    pools = sample_from_rankings(
        ranked, inputs.doc_ids, excluded, ORACLE_COUNT, torch.Generator().manual_seed(ORACLE_SEED)
    )
    judged = {q: dict(labels.get(q, {})) for q in ids}
    has_positive = [q for q in ids if any(g > 0 and d in doc_rows for d, g in judged[q].items())]
    scores = torch.zeros_like(queries)
    if has_positive:
        q, p, n = oracle_pairs(
            has_positive, vectors, rows, inputs.documents, doc_rows, judged,
            dict(zip(ids, pools, strict=True)),
        )  # fmt: skip
        index = {qid: i for i, qid in enumerate(ids)}
        scores[[index[qid] for qid in has_positive]] = oracle_scores(q, p, n)
    return scores
