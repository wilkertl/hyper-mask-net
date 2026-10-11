"""F0 diagnostics (research R20): oracle-level shift, trainer limits, and a first H2 probe.

The decision rules are the ones fixed in research R20 before this step first ran. Oracle
profiles, every model trained here, and the probe use training labels only; evaluation labels
are read only to score retrieval, as in ``analyze``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from hyperdime.evaluation.selector_shift import (
    half_split_profiles,
    mean_importance,
    overlap_at_k,
    profile_permutation_test,
    selection_frequency,
    spearman,
)
from hyperdime.evaluation.statistics import holm, paired_bootstrap_ci, paired_randomization_test
from hyperdime.oracle.importance import oracle_scores
from hyperdime.oracle.targets import aggregate_pairs
from hyperdime.selection.topk import top_k_mask
from hyperdime.space.marginal import marginal_stats
from hyperdime.spike import pipeline
from hyperdime.spike.manifest import read_json, require_complete, write_json, write_manifest

K = pipeline.HEADLINE_K
SEED = 0
CURVE_FRACTIONS = (0.25, 0.5, 1.0)
# One step past the edge of the selection grid, per research R20 D2(e).
TAU_BELOW = (0.001, 0.002)
TAU_ABOVE = (0.5, 1.0)
EPOCHS_ABOVE = (500,)
EPOCHS_BELOW = (3,)
GRID_MARGIN = 0.005
RIDGE_ALPHA = 1.0
# Rule T: a check must hold on at least this many environments.
MIN_LIMITED_ENVS = 2

PerQuery = pipeline.PerQuery


def oracle_scores_for(data: pipeline.EnvData, qids: Sequence[str]) -> Tensor:
    """``e_q * (p - n)`` from training labels and the stored training negatives only."""
    return _raw_oracle(data, qids, data.splits["train_labels"], data.negatives)


def diagnose(
    envs: Sequence[str],
    root: Path,
    seeds: Sequence[int],
    *,
    permutations: int = 1000,
    half_splits: int = 200,
) -> dict[str, Any]:
    """Run D1-D3 of research R20 and write ``diagnostics/diagnostics.json`` with a manifest."""
    require_complete(root / "analysis")
    for env in envs:
        require_complete(root / env)
        require_complete(root / env / "selectors")
    config = {
        "envs": list(envs),
        "seeds": list(seeds),
        "k": K,
        "permutations": permutations,
        "half_splits": half_splits,
        "seed": SEED,
        "curve_fractions": list(CURVE_FRACTIONS),
        "grid_extensions": {
            "tau_below": list(TAU_BELOW),
            "tau_above": list(TAU_ABOVE),
            "epochs_above": list(EPOCHS_ABOVE),
            "epochs_below": list(EPOCHS_BELOW),
            "margin": GRID_MARGIN,
        },
        "ridge_alpha": RIDGE_ALPHA,
        "alpha": pipeline.ALPHA,
    }
    directory = root / "diagnostics"
    write_manifest(directory, "diagnose", config, status="running")
    datasets = {env: pipeline.load_env(root, env) for env in envs}
    models = {
        env: [pipeline._load_selector(root / env / "selectors" / f"seed-{s}.pt") for s in seeds]
        for env in envs
    }
    scores = {env: oracle_scores_for(data, _labelled(data)) for env, data in datasets.items()}
    oracle_shift = _oracle_shift(scores, permutations, half_splits)
    trainer = _trainer(datasets, models, scores, root)
    probe = _probe(datasets, models, scores)
    result = {
        "config": config,
        "oracle_shift": oracle_shift,
        "trainer": trainer,
        "probe": probe,
        "recommendation": recommendation(oracle_shift["o1"], trainer["t"]),
    }
    write_json(directory / "diagnostics.json", result)
    write_manifest(directory, "diagnose", config)
    return result


def recommendation(o1: bool, t: bool) -> str:
    """Research R20's table: no oracle-level shift stops the roadmap whatever the trainer does."""
    if not o1:
        return "stop_or_pivot"
    return "continue_strengthen_selector" if t else "continue"


def loeo_probe(
    features: Mapping[str, Tensor], targets: Mapping[str, Tensor], alpha: float = RIDGE_ALPHA
) -> dict[str, Any]:
    """Leave one environment out: ridge from per-dimension features to a per-dimension target.

    Features ``[D, F]`` are z-scored with statistics of the training environments of each fold;
    the score is the Spearman correlation with the held-out target, and the baseline is the mean
    target of the other environments.
    """
    folds: dict[str, dict[str, Any]] = {}
    for held in features:
        train = [env for env in features if env != held]
        x = torch.cat([features[env] for env in train]).double()
        y = torch.cat([targets[env] for env in train]).double()
        mean, std = x.mean(dim=0), x.std(dim=0, correction=0)
        std = torch.where(std > 0, std, torch.ones_like(std))
        z = (x - mean) / std
        y_mean = y.mean()
        gram = z.T @ z + alpha * torch.eye(z.shape[1], dtype=torch.float64)
        weights = torch.linalg.solve(gram, z.T @ (y - y_mean))
        prediction = (features[held].double() - mean) / std @ weights + y_mean
        target = targets[held].double()
        baseline = torch.stack([targets[env].double() for env in train]).mean(dim=0)
        score, base = spearman(prediction, target), spearman(baseline, target)
        folds[held] = {
            "score": score,
            "baseline": base,
            "gain": score - base,
            "zscore": {"mean": mean.tolist(), "std": std.tolist()},
        }
    gains = [fold["gain"] for fold in folds.values()]
    return {"folds": folds, "mean_gain": sum(gains) / len(gains)}


def probe_permutation_test(
    features: Mapping[str, Tensor], targets: Mapping[str, Tensor], alpha: float = RIDGE_ALPHA
) -> tuple[float, int]:
    """Exact test over every assignment of descriptors to environments; returns ``(p, count)``.

    The statistic is the probe's mean gain; the identity assignment is one of those counted.
    """
    envs = list(features)
    observed = loeo_probe(features, targets, alpha)["mean_gain"]
    gains = [
        loeo_probe(
            {env: features[source] for env, source in zip(envs, order, strict=True)},
            targets,
            alpha,
        )["mean_gain"]
        for order in itertools.permutations(envs)
    ]
    hits = sum(gain >= observed - 1e-12 for gain in gains)
    return hits / len(gains), len(gains)


def _labelled(data: pipeline.EnvData) -> list[str]:
    return list(data.splits["train"]) + list(data.splits["validation"])


def _raw_oracle(
    data: pipeline.EnvData,
    qids: Sequence[str],
    labels: Mapping[str, Mapping[str, int]],
    negatives: Mapping[str, Sequence[str]],
) -> Tensor:
    # Top-k on raw scores: at small tau the softmax underflows to ties.
    positives, pools = pipeline._supervision(data, qids, labels, negatives)
    queries = data.query_rows(qids).float()
    aggregated_p, aggregated_n = aggregate_pairs(queries, data.docs, positives, pools)
    return oracle_scores(queries, aggregated_p, aggregated_n)


def _oracle_shift(
    scores: Mapping[str, Tensor], permutations: int, half_splits: int
) -> dict[str, Any]:
    """D1: do environments' oracle profiles differ more than random halves of one environment?"""
    halves = {env: half_split_profiles(s, K, half_splits, SEED) for env, s in scores.items()}
    profiles: dict[str, dict[str, Any]] = {
        env: {
            "k": K,
            "labels": "train",
            "num_queries": int(s.shape[0]),
            "frequency": selection_frequency(s, K).tolist(),
            "split_half_spearman": _mean(
                [spearman(a, b) for a, b in zip(*halves[env], strict=True)]
            ),
        }
        for env, s in scores.items()
    }
    pairs: dict[str, dict[str, float]] = {}
    for x, y in itertools.combinations(scores, 2):
        rho = [spearman(a, b) for a, b in zip(halves[x][0], halves[y][0], strict=True)]
        pairs[f"{x}|{y}"] = {
            "rho_cross": _mean(rho),
            "p": profile_permutation_test(scores[x], scores[y], K, permutations, SEED),
        }
    _adjust(pairs)
    significant = [name for name, pair in pairs.items() if pair["p_holm"] < pipeline.ALPHA]
    rho_within = _mean([profile["split_half_spearman"] for profile in profiles.values()])
    rho_cross = _mean([pair["rho_cross"] for pair in pairs.values()])
    return {
        "profiles": profiles,
        "pairs": pairs,
        "significant_pairs": significant,
        "rho_within_mean": rho_within,
        "rho_cross_mean": rho_cross,
        "o1": len(significant) >= math.ceil(len(pairs) / 2) and rho_cross < rho_within,
    }


def _trainer(
    datasets: Mapping[str, pipeline.EnvData],
    models: Mapping[str, Sequence[Any]],
    scores: Mapping[str, Tensor],
    root: Path,
) -> dict[str, Any]:
    """D2: learning curve, seed ensemble, fit gap, recovery, and grid edge per environment."""
    per_env = {
        env: _trainer_env(data, models[env], scores[env], root / env / "selectors")
        for env, data in datasets.items()
    }
    for test in ("curve_test", "ensemble_test"):
        tests = {env: r[test] for env, r in per_env.items() if r[test] is not None}
        _adjust(tests)
    limited = {
        "curve": [env for env, r in per_env.items() if _wins(r["curve_test"])],
        "ensemble": [env for env, r in per_env.items() if _wins(r["ensemble_test"])],
        "grid_edge": [env for env, r in per_env.items() if r["grid_edge"]["limited"]],
    }
    return {
        "envs": per_env,
        "limited": limited,
        "t": any(len(found) >= MIN_LIMITED_ENVS for found in limited.values()),
    }


def _trainer_env(
    data: pipeline.EnvData, models: Sequence[Any], oracle: Tensor, selectors: Path
) -> dict[str, Any]:
    chosen = read_json(selectors / "temperature.json")
    temperature, epochs = float(chosen["temperature"]), int(chosen["epochs"])
    eval_qids, eval_labels = data.splits["evaluation"], data.splits["eval_labels"]
    eval_queries = data.query_rows(eval_qids)

    # (a) learning curve on nested subsets of the training queries, seed 0
    train_qids = list(data.splits["train"])
    order = torch.randperm(len(train_qids), generator=torch.Generator().manual_seed(SEED))
    curve: dict[str, PerQuery] = {}
    for fraction in CURVE_FRACTIONS:
        count = max(1, math.ceil(fraction * len(train_qids)))
        subset = [train_qids[i] for i in sorted(order[:count].tolist())]
        model, _ = pipeline._fit(data, {**data.splits, "train": subset}, temperature, SEED, epochs)
        importance = pipeline._importance(model, eval_queries)
        curve[str(round(fraction * 100))] = pipeline._evaluate(
            importance, K, data, eval_qids, eval_labels
        )

    # (b) mean of the seeds' log-importances against the per-query mean of single seeds
    log_importances = [pipeline._importance(m, eval_queries) for m in models]
    single = pipeline._run(log_importances, K, data, eval_qids, eval_labels)
    ensemble = pipeline._evaluate(
        torch.stack(log_importances).mean(dim=0), K, data, eval_qids, eval_labels
    )

    # (c) fit gap: retrieval and agreement with the oracle on training vs evaluation queries
    train_queries = data.query_rows(train_qids)
    train_importances = [pipeline._importance(m, train_queries) for m in models]
    train_scores = pipeline._run(
        train_importances, K, data, train_qids, data.splits["train_labels"]
    )
    # Evaluation-side oracle: a measurement only, built like analyze's oracle ceiling.
    eval_negatives = pipeline._mine(eval_queries, data.docs, data.doc_ids, eval_qids, eval_labels)
    oracle_masks = {
        "train": top_k_mask(oracle_scores_for(data, train_qids), K),
        "evaluation": top_k_mask(_raw_oracle(data, eval_qids, eval_labels, eval_negatives), K),
    }
    importances = {"train": train_importances, "evaluation": log_importances}
    oracle_overlap = {
        name: _mean(
            [float(overlap_at_k(top_k_mask(li, K), oracle_masks[name]).mean()) for li in lis]
        )
        for name, lis in importances.items()
    }

    # (d) recovery of the environment's oracle profile on its labelled queries
    labelled_queries = data.query_rows(_labelled(data))
    profile = selection_frequency(oracle, K)
    recovery = _mean(
        [
            spearman(selection_frequency(pipeline._importance(m, labelled_queries), K), profile)
            for m in models
        ]
    )

    return {
        "curve": {name: pipeline._mean(pq, "ndcg@10") for name, pq in curve.items()},
        "curve_test": _paired(curve["100"], curve["50"]),
        "ensemble": pipeline._mean(ensemble, "ndcg@10"),
        "single": pipeline._mean(single, "ndcg@10"),
        "ensemble_test": _paired(ensemble, single) if len(models) > 1 else None,
        "fit_gap": {
            "train_ndcg@10": pipeline._mean(train_scores, "ndcg@10"),
            "evaluation_ndcg@10": pipeline._mean(single, "ndcg@10"),
            "train_oracle_overlap": oracle_overlap["train"],
            "evaluation_oracle_overlap": oracle_overlap["evaluation"],
        },
        "recovery": recovery,
        "grid_edge": _grid_edge(data, temperature, epochs, chosen["validation_ndcg@10"]),
    }


def _grid_edge(
    data: pipeline.EnvData,
    temperature: float,
    epochs: int,
    grid_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """(e) One step past the grid edge, scored on validation queries only."""
    temperatures, budgets = pipeline.TEMPERATURES, pipeline.EPOCH_BUDGETS
    extra_taus = (TAU_BELOW if temperature == min(temperatures) else ()) + (
        TAU_ABOVE if temperature == max(temperatures) else ()
    )
    extra_epochs = (EPOCHS_BELOW if epochs == min(budgets) else ()) + (
        EPOCHS_ABOVE if epochs == max(budgets) else ()
    )
    cells = [
        (tau, budget)
        for tau in (temperature, *extra_taus)
        for budget in (epochs, *extra_epochs)
        if (tau, budget) != (temperature, epochs)
    ]
    chosen = float(grid_scores[str(temperature)][str(epochs)])
    extended = {
        f"{tau}|{budget}": pipeline._validation_ndcg(
            pipeline._fit(data, data.splits, tau, SEED, budget)[0], data
        )
        for tau, budget in cells
    }
    best = max(extended.values(), default=chosen)
    return {
        "chosen": chosen,
        "extended": extended,
        "best_extended": best if extended else None,
        "limited": bool(extended) and best - chosen > GRID_MARGIN,
    }


def _probe(
    datasets: Mapping[str, pipeline.EnvData],
    models: Mapping[str, Sequence[Any]],
    scores: Mapping[str, Tensor],
) -> dict[str, Any]:
    """D3: corpus-only group-A statistics predicting the oracle profile, leave one env out."""
    features = {env: marginal_stats(data.docs) for env, data in datasets.items()}
    targets = {env: selection_frequency(s, K) for env, s in scores.items()}
    main = loeo_probe(features, targets)
    p, count = probe_permutation_test(features, targets)
    secondary_targets = {
        env: torch.stack(
            [
                mean_importance(pipeline._importance(m, data.query_rows(_labelled(data))))
                for m in models[env]
            ]
        ).mean(dim=0)
        for env, data in datasets.items()
    }
    secondary = loeo_probe(features, secondary_targets)
    return {
        "target": "oracle_selection_frequency",
        "folds": _without_zscore(main["folds"]),
        "mean_gain": main["mean_gain"],
        "p": p,
        "num_permutations": count,
        "s": main["mean_gain"] > 0 and p <= pipeline.ALPHA,
        "secondary": {
            "target": "mean_importance",
            "folds": _without_zscore(secondary["folds"]),
            "mean_gain": secondary["mean_gain"],
        },
    }


def _without_zscore(folds: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {env: {k: v for k, v in fold.items() if k != "zscore"} for env, fold in folds.items()}


def _paired(better: PerQuery, worse: PerQuery) -> dict[str, Any]:
    qids = sorted(better)
    xs = [better[q]["ndcg@10"] for q in qids]
    ys = [worse[q]["ndcg@10"] for q in qids]
    return {
        "diff": _mean(xs) - _mean(ys),
        "p": paired_randomization_test(xs, ys),
        "ci": list(paired_bootstrap_ci(xs, ys)),
    }


def _wins(test: Mapping[str, Any] | None) -> bool:
    return test is not None and test["diff"] > 0 and test["p_holm"] < pipeline.ALPHA


def _adjust(tests: Mapping[str, dict[str, Any]]) -> None:
    adjusted = holm({name: t["p"] for name, t in tests.items()})
    for name, t in tests.items():
        t["p_holm"] = adjusted[name]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)
