"""``hyperdime p3-target``: the P3 target-encoding study (ADR-0019; exploratory/development).

- ``diagnose``: stage D on F12 artifacts of one small corpus. Fit queries (teachers privileged,
  students in-sample) and tune queries (students out of sample); no training.
- ``ablate``: stage A, train one arm (E0-E4, one seed, optional λ) on fit queries.
- ``score``: exact tune-A / tune-B nDCG@10 of trained arms, with paired bootstrap against E0.

Only fit and tune labels are read (``qrels-train`` and ``qrels-validation`` filtered to those ids);
F12 evaluation ids and official test labels are never opened. Small corpora only (SciFact,
NFCorpus): scores use a float64 copy of the corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.baselines.l2s_official import DEFAULT_CLONE, PUBLISHED, SEED, import_official
from hyperdime.cli.p3_bench import MODEL_REVISION, _positives, _write_json, load_fit_inputs
from hyperdime.cli.p3_bench_arms import (
    SECONDARY_SEEDS,
    TEACHERS,
    _linear,
    _normalized,
    _official_dir,
    _query_vectors,
    _split_qrels,
)
from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, read_manifest, start_manifest
from hyperdime.contracts.validation import ContractError
from hyperdime.data.query_split import hash_order, id_fingerprint, load_query_split
from hyperdime.evaluation.benchmark import PRIMARY, ideal_dcg, ndcg10
from hyperdime.evaluation.benchmark_stats import bootstrap_means
from hyperdime.evaluation.target_study import (
    LAMBDAS,
    jaccard,
    kl_rows,
    loss_budget_input,
    loss_hybrid,
    loss_kl,
    loss_membership,
    margins,
    mask_average,
    ranks_from_scores,
    top_k_mask,
    train,
)
from hyperdime.evaluation.teachers import positive_mask, positive_mean

COMMAND = "p3-target"
SMALL = ("scifact", "nfcorpus")
SEEDS = (SEED, *SECONDARY_SEEDS)
PILOT_SEEDS = (42, 0, 1)
BOOTSTRAP = 10_000


def _manifest(root: Path, name: str, config: dict[str, Any], device: str, seed: int) -> Any:
    return start_manifest(
        f"{COMMAND} {name}",
        config,
        dataset_hash=file_sha256(root / "p3-query-split-v1" / "manifest.json"),
        model_revision=MODEL_REVISION,
        seed=seed,
        device=device,
    )


def _tune_ids(
    root: Path, corpus: str, doc_rows: dict[str, int]
) -> tuple[list[str], dict[str, dict[str, int]]]:
    split = load_query_split(root, corpus)
    ids = hash_order(corpus, "tune", split.ids["tune"])
    qrels = _split_qrels(root, corpus, set(ids))
    kept = [q for q in ids if any(r > 0 and d in doc_rows for d, r in qrels.get(q, {}).items())]
    return kept, qrels


def _measure(
    docs: Tensor, query: Tensor, masks: Tensor, rows: list[int], grades: list[float]
) -> tuple[Tensor, Tensor]:
    """nDCG@10 and margin of each boolean mask row ``[B, D]`` for one query (exact float64)."""
    masked = query[None, :] * masks.to(torch.float64)  # zeros outside the mask add nothing
    scores = docs @ masked.T  # [N, B]
    pos = torch.tensor(rows)
    gains = torch.tensor(grades, dtype=torch.float64)
    ranks = ranks_from_scores(scores, pos)  # [P, B]
    return ndcg10(ranks.T, gains, ideal_dcg(grades)), margins(scores, pos)


def _students(
    root: Path, corpus: str, runs: dict[str, str], teachers: list[str]
) -> dict[str, dict[str, Tensor]]:
    states = {
        "l2s-official": torch.load(_official_dir(root, corpus, runs["official"]) / "state_dict.pt")
    }
    for teacher in teachers:
        for seed in SEEDS:
            run = runs[f"arm:{teacher}@{seed}"]
            path = root / "selectors" / "p3-teacher-arms" / corpus / run / "state_dict.pt"
            states[f"{teacher}@{seed}"] = torch.load(path)
    return states


# ----------------------------------------------------------------------------- diagnose


def cmd_diagnose(args: argparse.Namespace) -> int:
    corpus = args.corpus
    if corpus not in SMALL:
        raise ContractError("the diagnostic runs on small corpora only")
    torch.set_num_threads(args.threads)
    runs = json.loads(args.runs.read_text())
    official = _official_dir(args.root, corpus, runs["official"])
    inputs = load_fit_inputs(args.root, corpus, read_manifest(official).config["n_fit"])
    valid = json.loads((official / "valid_query_ids.json").read_text())
    teachers_dir = args.root / "evaluation" / "P3-bench" / "teachers" / corpus / runs["teachers"]
    teachers = [t for t in TEACHERS if (teachers_dir / f"masks_{t}.npy").exists()]
    masks = {t: torch.from_numpy(np.load(teachers_dir / f"masks_{t}.npy")) for t in teachers}
    averages = {t: mask_average(masks[t]) for t in teachers}
    t_off = torch.load(official / "targets.pt").to(torch.float64)
    states = _students(args.root, corpus, runs, teachers)
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    qindex = {q: i for i, q in enumerate(inputs.query_ids)}
    docs = torch.from_numpy(np.asarray(inputs.documents).astype(np.float64))
    x_raw = torch.from_numpy(inputs.queries[[qindex[q] for q in valid]].astype(np.float32))
    cpu = torch.device("cpu")
    layers = {name: _linear(state) for name, state in states.items()}
    with torch.no_grad():
        logits_fit = {n: layer(_normalized(x_raw.numpy(), cpu)) for n, layer in layers.items()}
        logp_raw = {n: torch.log_softmax(layer(x_raw), -1) for n, layer in layers.items()}
    targets_of = {
        "l2s-official": t_off,
        **{f"{t}@{s}": averages[t] for t in teachers for s in SEEDS},
    }
    began = time.perf_counter()
    rows_out: list[dict[str, Any]] = []

    def add(
        split: str, qid: str, npos: int, k: int, quantity: str, name: str, value: float | Tensor
    ) -> None:
        rows_out.append(
            {
                "split": split,
                "query": qid,
                "positives": npos,
                "k": k,
                "quantity": quantity,
                "name": name,
                "value": float(value),
            }
        )

    for qi, qid in enumerate(valid):
        rows, grades = _positives(inputs, qid, doc_rows)
        q = torch.from_numpy(inputs.queries[qindex[qid]].astype(np.float64))
        npos = len(rows)
        for name, logp in logp_raw.items():
            add(
                "fit",
                qid,
                npos,
                0,
                "kl_target_student",
                name,
                kl_rows(targets_of[name][qi], logp[qi]),
            )
        for b, k in enumerate(PRIMARY):
            named: dict[str, Tensor] = {}
            for t in teachers:
                named[f"teacher:{t}"] = masks[t][qi, b]
                named[f"decoded:{t}"] = top_k_mask(averages[t][qi], k)
            named["decoded:l2s-official"] = top_k_mask(t_off[qi], k)
            for name, z in logits_fit.items():
                student = torch.zeros(z.shape[1], dtype=torch.bool)
                student[torch.topk(z[qi], k).indices] = True
                named[f"student:{name}"] = student
            stacked = torch.stack(list(named.values()))
            ndcg, margin = _measure(docs, q, stacked, rows, grades)
            for (name, _), nd, mg in zip(
                named.items(), ndcg.tolist(), margin.tolist(), strict=True
            ):
                add("fit", qid, npos, k, "ndcg10", name, nd)
                add("fit", qid, npos, k, "margin", name, mg)
            for t in teachers:
                add(
                    "fit",
                    qid,
                    npos,
                    k,
                    "jaccard_decoded_teacher",
                    t,
                    jaccard(named[f"decoded:{t}"], named[f"teacher:{t}"]),
                )
                seeds = [named[f"student:{t}@{s}"] for s in SEEDS]
                for s in SEEDS:
                    add(
                        "fit",
                        qid,
                        npos,
                        k,
                        "jaccard_student_decoded",
                        f"{t}@{s}",
                        jaccard(named[f"student:{t}@{s}"], named[f"decoded:{t}"]),
                    )
                add("fit", qid, npos, k, "jaccard_seeds", t, _pairwise(seeds))
            if "contrastive" in teachers:
                add(
                    "fit",
                    qid,
                    npos,
                    k,
                    "jaccard_decoded_teacher",
                    "l2s-official-vs-contrastive",
                    jaccard(named["decoded:l2s-official"], named["teacher:contrastive"]),
                )
            add(
                "fit",
                qid,
                npos,
                k,
                "jaccard_student_decoded",
                "l2s-official",
                jaccard(named["student:l2s-official"], named["decoded:l2s-official"]),
            )

    # tune queries: students out of sample against the positive teacher built from tune positives
    tune_ids, tune_qrels = _tune_ids(args.root, corpus, doc_rows)
    tune_vectors = _query_vectors(args.root, corpus, tune_ids, inputs)
    with torch.no_grad():
        logits_tune = {n: layer(_normalized(tune_vectors, cpu)) for n, layer in layers.items()}
    for ti, qid in enumerate(tune_ids):
        judged = tune_qrels[qid]
        rows = [doc_rows[d] for d, r in judged.items() if r > 0 and d in doc_rows]
        grades = [float(judged[inputs.doc_ids[r]]) for r in rows]
        q64 = tune_vectors[ti].astype(np.float64)
        p_bar = positive_mean(
            np.asarray(inputs.documents)[rows].astype(np.float64), np.asarray(grades)
        )
        q = torch.from_numpy(q64)
        for k in PRIMARY:
            named = {}
            teacher = torch.zeros(1024, dtype=torch.bool)
            teacher[torch.from_numpy(positive_mask(q64, p_bar, k))] = True
            named["teacher:positive"] = teacher
            for name, z in logits_tune.items():
                student = torch.zeros(1024, dtype=torch.bool)
                student[torch.topk(z[ti], k).indices] = True
                named[f"student:{name}"] = student
            ndcg, margin = _measure(docs, q, torch.stack(list(named.values())), rows, grades)
            for (name, _), nd, mg in zip(
                named.items(), ndcg.tolist(), margin.tolist(), strict=True
            ):
                add("tune", qid, len(rows), k, "ndcg10", name, nd)
                add("tune", qid, len(rows), k, "margin", name, mg)
            for name in logits_tune:
                add(
                    "tune",
                    qid,
                    len(rows),
                    k,
                    "jaccard_student_teacher_positive",
                    name,
                    jaccard(named[f"student:{name}"], teacher),
                )
            for t in teachers:
                add(
                    "tune",
                    qid,
                    len(rows),
                    k,
                    "jaccard_seeds",
                    t,
                    _pairwise([named[f"student:{t}@{s}"] for s in SEEDS]),
                )

    search = _search_effect(teachers_dir, masks) if "rank-guided" in teachers else None
    nest = {t: _nesting_mean(masks[t]) for t in teachers}
    summary = _summarize(rows_out)
    manifest = _manifest(
        args.root,
        "diagnose",
        {
            "corpus": corpus,
            "runs": runs,
            "teachers": teachers,
            "tune_ids_sha256": id_fingerprint(tune_ids),
        },
        "cpu",
        20261114,
    )
    with ArtifactWriter(args.root, f"evaluation/P3-target/diagnose/{corpus}", manifest) as writer:
        _write_csv(writer.path("per_query.csv"), rows_out)
        _write_csv(writer.path("summary.csv"), summary)
        _write_json(
            writer.path("summary.json"),
            {
                "corpus": corpus,
                "fit_queries": len(valid),
                "tune_queries": len(tune_ids),
                "teachers": teachers,
                "nesting": nest,
                "search_effect": search,
                "seconds": time.perf_counter() - began,
            },
        )
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "fit": len(valid),
                "tune": len(tune_ids),
                "search_effect": search,
                "nesting": nest,
            }
        )
    )
    return 0


def _pairwise(masks: list[Tensor]) -> float:
    values = [
        float(jaccard(masks[i], masks[j]))
        for i in range(len(masks))
        for j in range(i + 1, len(masks))
    ]
    return float(np.mean(values))


def _nesting_mean(masks: Tensor) -> list[float]:
    small, large = masks[:, :-1], masks[:, 1:]
    share = (small & large).sum(-1).double() / small.sum(-1).clamp(min=1)
    return [float(v) for v in share.mean(0)]


def _search_effect(teachers_dir: Path, masks: dict[str, Tensor]) -> dict[str, Any]:
    """How much the rank-guided search changed its chosen start and improved the objective."""
    searches = json.loads((teachers_dir / "searches.json").read_text())
    chosen = [s for s in searches if s["chosen"]]
    changed, gains_ndcg, gains_gdr, overlap = 0, [], [], []
    for s in chosen:
        b = PRIMARY.index(s["k"])
        rg, start = masks["rank-guided"][s["query"], b], masks[s["start"]][s["query"], b]
        overlap.append(float(jaccard(rg, start)))
        changed += int(not torch.equal(rg, start))
        gains_ndcg.append(s["end_value"][0] - s["start_value"][0])
        gains_gdr.append(s["end_value"][1] - s["start_value"][1])
    return {
        "searches": len(searches),
        "chosen": len(chosen),
        "chosen_changed": changed,
        "chosen_from_contrastive": sum(s["start"] == "contrastive" for s in chosen),
        "mean_jaccard_to_start": float(np.mean(overlap)),
        "mean_gain_ndcg10": float(np.mean(gains_ndcg)),
        "mean_gain_gdr": float(np.mean(gains_gdr)),
        "share_improved_ndcg10": float(np.mean([g > 0 for g in gains_ndcg])),
        "steps_mean": float(np.mean([s["steps"] for s in searches])),
        "hit_cap": sum(s["hit_cap"] for s in searches),
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean per (split, k, quantity, name) over all queries and by single/multi positive cohort
    (secondary). Every query weighs equally."""
    groups: dict[tuple[Any, ...], list[float]] = {}
    for r in rows:
        cohort = "single" if r["positives"] == 1 else "multi"
        for c in ("all", cohort):
            groups.setdefault((r["split"], r["k"], r["quantity"], r["name"], c), []).append(
                r["value"]
            )
    return [
        {
            "split": s,
            "k": k,
            "quantity": q,
            "name": n,
            "cohort": c,
            "queries": len(v),
            "mean": float(np.mean(v)),
        }
        for (s, k, q, n, c), v in sorted(groups.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        out = csv.DictWriter(handle, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


# ----------------------------------------------------------------------------- ablate


def _ablation_inputs(root: Path, corpus: str, runs: dict[str, str]) -> dict[str, Any]:
    official = _official_dir(root, corpus, runs["official"])
    inputs = load_fit_inputs(root, corpus, read_manifest(official).config["n_fit"])
    valid = json.loads((official / "valid_query_ids.json").read_text())
    teachers_dir = root / "evaluation" / "P3-bench" / "teachers" / corpus / runs["teachers"]
    if json.loads((teachers_dir / "valid_query_ids.json").read_text()) != valid:
        raise ContractError("teacher queries differ from the official reference's")
    masks = torch.from_numpy(np.load(teachers_dir / "masks_contrastive.npy"))
    qindex = {q: i for i, q in enumerate(inputs.query_ids)}
    return {
        "inputs": inputs,
        "valid": valid,
        "x": torch.from_numpy(inputs.queries[[qindex[q] for q in valid]].astype(np.float32)),
        "official": torch.load(official / "targets.pt"),
        "masks": masks,
        "average": mask_average(masks).to(torch.float32),
        "sources": {
            "targets_sha256": file_sha256(official / "targets.pt"),
            "masks_sha256": file_sha256(teachers_dir / "masks_contrastive.npy"),
        },
    }


def _arm_payload(arm: str, data: dict[str, Any], lam: float | None) -> tuple[Tensor, Any, int]:
    """(payload, loss, budgets) for one arm; the payload is indexed with the queries."""
    n, b, d = data["masks"].shape
    if arm == "E0":
        return data["official"], lambda m, x, t, _b: loss_kl(m, x, t), 0
    if arm == "E1":
        return data["average"], lambda m, x, t, _b: loss_kl(m, x, t), 0
    if arm == "E2":
        return data["masks"], lambda m, x, t, beta: loss_membership(m, x, t, beta, PRIMARY), 0
    if arm == "E3":
        per_budget = data["masks"].to(torch.float32)
        per_budget = per_budget / per_budget.sum(-1, keepdim=True)
        return per_budget, lambda m, x, t, _b: loss_budget_input(m, x, t), b
    if arm == "E3c":
        repeated = data["average"][:, None, :].expand(n, b, d).contiguous()
        return repeated, lambda m, x, t, _b: loss_budget_input(m, x, t), b
    if arm == "E4":
        if lam is None:
            raise ContractError("E4 needs --lam")
        both = torch.stack([data["official"], data["average"]], 1)
        return both, lambda m, x, t, _b: loss_hybrid(m, x, t[:, 0], t[:, 1], lam), 0
    raise ContractError(f"unknown arm {arm}")


def cmd_ablate(args: argparse.Namespace) -> int:
    corpus = args.corpus
    if corpus not in SMALL:
        raise ContractError("the pilot runs on small corpora only")
    if args.lam is not None and args.lam not in LAMBDAS:
        raise ContractError(f"λ must be one of {LAMBDAS}")
    runs = json.loads(args.runs.read_text())
    data = _ablation_inputs(args.root, corpus, runs)
    payload, loss, budgets = _arm_payload(args.arm, data, args.lam)
    _, epochs = PUBLISHED[corpus]
    module = import_official(args.clone)
    manifest = _manifest(
        args.root,
        "ablate",
        {
            "corpus": corpus,
            "arm": args.arm,
            "seed": args.seed,
            "lam": args.lam,
            "epochs": epochs,
            "runs": runs,
            **data["sources"],
        },
        args.device,
        args.seed,
    )
    result = train(
        data["x"],
        payload,
        loss,
        epochs=epochs,
        seed=args.seed,
        device=args.device,
        budgets=budgets,
        set_seed=module.set_random_seed,
    )
    state = {k: v.detach().cpu().clone() for k, v in result.model.state_dict().items()}
    summary: dict[str, Any] = {"final_loss": result.final_loss, "steps": result.steps}
    for name, reference in (("E0", "official"), ("E1", "arm:contrastive@42")):
        if args.arm == name and args.seed == SEED and reference in runs:
            path = (
                _official_dir(args.root, corpus, runs[reference]) / "state_dict.pt"
                if reference == "official"
                else args.root
                / "selectors"
                / "p3-teacher-arms"
                / corpus
                / runs[reference]
                / "state_dict.pt"
            )
            ref = torch.load(path)
            summary["equals_f12_checkpoint"] = all(
                torch.equal(state[f"net.0.{p}"], ref[f"net.0.{p}"]) for p in ("weight", "bias")
            )
    with ArtifactWriter(args.root, f"selectors/p3-target/{corpus}/{args.arm}", manifest) as writer:
        torch.save(state, writer.path("state_dict.pt"))
        _write_json(writer.path("summary.json"), summary)
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "arm": args.arm,
                "seed": args.seed,
                "lam": args.lam,
                **summary,
            }
        )
    )
    return 0


# ----------------------------------------------------------------------------- score


def _scores_of(state: dict[str, Tensor], x: Tensor, k_index: int) -> Tensor:
    weight, bias = state["net.0.weight"], state["net.0.bias"]
    z = x @ weight.T + bias
    if "offsets" in state:
        z = z + state["offsets"][k_index]
    return z


def cmd_score(args: argparse.Namespace) -> int:
    """Tune-A and tune-B nDCG@10 of every listed checkpoint; paired bootstrap vs E0 on tune-B."""
    corpus = args.corpus
    torch.set_num_threads(args.threads)
    listing = json.loads(args.checkpoints.read_text())  # {"E0@42": run, "E4[0.3]@42": run, ...}
    inputs = load_fit_inputs(args.root, corpus, 1)
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    tune_ids, qrels = _tune_ids(args.root, corpus, doc_rows)
    half = len(tune_ids) // 2
    halves = {q: ("A" if i < half else "B") for i, q in enumerate(tune_ids)}
    vectors = _query_vectors(args.root, corpus, tune_ids, inputs)
    x = _normalized(vectors, torch.device("cpu"))
    docs = torch.from_numpy(np.asarray(inputs.documents).astype(np.float64))
    states = {}
    for key, run in listing.items():
        arm = key.split("@")[0].split("[")[0]
        states[key] = torch.load(
            args.root / "selectors" / "p3-target" / corpus / arm / run / "state_dict.pt"
        )
    rows_out = []
    for qi, qid in enumerate(tune_ids):
        rows = [doc_rows[d] for d, r in qrels[qid].items() if r > 0 and d in doc_rows]
        grades = [float(qrels[qid][inputs.doc_ids[r]]) for r in rows]
        q = torch.from_numpy(vectors[qi].astype(np.float64))
        for b, k in enumerate(PRIMARY):
            names, stacked = [], []
            for key, state in states.items():
                mask = torch.zeros(1024, dtype=torch.bool)
                mask[torch.topk(_scores_of(state, x[qi : qi + 1], b)[0], k).indices] = True
                names.append(key)
                stacked.append(mask)
            ndcg, _ = _measure(docs, q, torch.stack(stacked), rows, grades)
            for key, value in zip(names, ndcg.tolist(), strict=True):
                rows_out.append(
                    {"half": halves[qid], "query": qid, "k": k, "arm": key, "ndcg10": value}
                )
    summary = _score_summary(rows_out, tune_ids, halves)
    manifest = _manifest(
        args.root,
        "score",
        {"corpus": corpus, "checkpoints": listing, "tune_ids_sha256": id_fingerprint(tune_ids)},
        "cpu",
        20261114,
    )
    with ArtifactWriter(args.root, f"evaluation/P3-target/score/{corpus}", manifest) as writer:
        _write_csv(writer.path("per_query.csv"), rows_out)
        _write_json(writer.path("summary.json"), summary)
    print(json.dumps({"run_id": manifest.run_id, "summary": summary}))
    return 0


def _score_summary(
    rows: list[dict[str, Any]], ids: list[str], halves: dict[str, str]
) -> dict[str, Any]:
    table: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        table.setdefault((r["arm"], r["query"]), []).append(r["ndcg10"])
    arms = sorted({r["arm"] for r in rows})
    means = {(a, q): float(np.mean(v)) for (a, q), v in table.items()}
    out: dict[str, Any] = {"queries": {"A": 0, "B": 0}, "arms": {}}
    for half in ("A", "B"):
        out["queries"][half] = sum(1 for q in ids if halves[q] == half)
    for arm in arms:
        entry: dict[str, Any] = {}
        for half in ("A", "B"):
            values = np.array([means[(arm, q)] for q in ids if halves[q] == half])
            entry[f"tune_{half}_mean"] = float(values.mean())
        reference = f"E0@{arm.split('@')[1]}"
        if reference in arms and reference != arm:
            diffs = np.array(
                [means[(arm, q)] - means[(reference, q)] for q in ids if halves[q] == "B"]
            )
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261114)))
            boot = bootstrap_means(diffs, rng, draws=BOOTSTRAP)
            low, high = np.quantile(boot, [0.025, 0.975])
            entry["tune_B_minus_E0_same_seed"] = {
                "D": float(diffs.mean()),
                "low": float(low),
                "high": float(high),
            }
        out["arms"][arm] = entry
    return out


# ----------------------------------------------------------------------------- report


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _table(
    summary: list[dict[str, str]], split: str, quantity: str, names: list[str], cohort: str = "all"
) -> list[str]:
    values = {
        (r["k"], r["name"]): float(r["mean"])
        for r in summary
        if r["split"] == split and r["quantity"] == quantity and r["cohort"] == cohort
    }
    present = [n for n in names if any((str(k), n) in values for k in PRIMARY)]
    lines = ["| k | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for k in PRIMARY:
        lines.append(
            f"| {k} | "
            + " | ".join(f"{values.get((str(k), n), float('nan')):.4f}" for n in present)
            + " |"
        )
    means = [
        np.mean([values[(str(k), n)] for k in PRIMARY if (str(k), n) in values]) for n in present
    ]
    lines.append("| mean | " + " | ".join(f"{m:.4f}" for m in means) + " |")
    return lines


def cmd_report(args: argparse.Namespace) -> int:
    """README with every table generated from the diagnose and score runs listed in ``--runs``."""
    listing = json.loads(args.runs.read_text())  # {"diagnose": {corpus: run}, "score": {...}}
    out = [
        "# P3-target: target-encoding diagnostic and ablation (exploratory)",
        "",
        "Generated by `hyperdime p3-target report`; protocol [ADR-0019]"
        "(../../docs/decisions/ADR-0019-p3-target-encoding-study.md). Development data only: fit",
        "queries train, tune queries diagnose and select (tune-A) and report (tune-B). No F12",
        "evaluation label or official test label is read. Differences locate where ranking is",
        "lost; without the controlled ablation they do not identify a cause.",
        "",
    ]
    for corpus, run in listing.get("diagnose", {}).items():
        base = args.root / "evaluation" / "P3-target" / "diagnose" / corpus / run
        summary = _read_csv(base / "summary.csv")
        info = json.loads((base / "summary.json").read_text())
        teachers = info["teachers"]
        out += [
            f"## Diagnostic, {corpus} (`{run}`)",
            "",
            f"Fit queries: {info['fit_queries']} (teachers privileged, students in-sample); "
            f"tune queries: {info['tune_queries']} (students out of sample).",
            "",
        ]
        out += ["### (a) teacher masks and (b) their decoded targets, fit nDCG@10", ""]
        names = [f"teacher:{t}" for t in teachers] + [f"decoded:{t}" for t in teachers]
        out += _table(summary, "fit", "ndcg10", names + ["decoded:l2s-official"]) + [""]
        out += ["### (b) decode agreement, Jaccard(top-k of the target, teacher mask)", ""]
        out += _table(
            summary, "fit", "jaccard_decoded_teacher", teachers + ["l2s-official-vs-contrastive"]
        ) + [""]
        out += ["### (c) students on fit (in-sample) nDCG@10, seed 42", ""]
        students = ["student:l2s-official"] + [f"student:{t}@42" for t in teachers]
        out += _table(summary, "fit", "ndcg10", students) + [""]
        out += ["### (c) student agreement with its decoded target, fit Jaccard", ""]
        out += _table(
            summary,
            "fit",
            "jaccard_student_decoded",
            ["l2s-official"] + [f"{t}@42" for t in teachers],
        ) + [""]
        out += ["### Margins (best positive minus best non-positive masked score), fit", ""]
        out += _table(summary, "fit", "margin", names[: len(teachers)] + students) + [""]
        out += ["### Tune (out of sample): nDCG@10", ""]
        out += _table(summary, "tune", "ndcg10", ["teacher:positive"] + students) + [""]
        out += ["### Tune: agreement with the tune positive mask, and seed stability", ""]
        out += _table(
            summary,
            "tune",
            "jaccard_student_teacher_positive",
            ["l2s-official"] + [f"{t}@42" for t in teachers],
        ) + [""]
        out += _table(summary, "tune", "jaccard_seeds", teachers) + [""]
        kl = {
            r["name"]: float(r["mean"])
            for r in summary
            if r["quantity"] == "kl_target_student" and r["cohort"] == "all"
        }
        out += [
            "KL(target ‖ student) on fit, mean over queries: "
            + ", ".join(
                f"{n} {v:.4f}" for n, v in sorted(kl.items()) if n.endswith(("42", "official"))
            ),
            "",
        ]
        out += [
            "Teacher nesting (share of the k-mask inside the next): "
            + "; ".join(
                f"{t} " + ", ".join(f"{v:.3f}" for v in vals) for t, vals in info["nesting"].items()
            ),
            "",
        ]
        if info.get("search_effect"):
            se = info["search_effect"]
            out += [
                "Rank-guided search: "
                + ", ".join(
                    f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in se.items()
                ),
                "",
            ]
        out += [
            "### Secondary: single- vs multi-positive fit queries, nDCG@10 (mean of budgets)",
            "",
        ]
        for cohort in ("single", "multi"):
            rows = [
                r
                for r in summary
                if r["split"] == "fit"
                and r["quantity"] == "ndcg10"
                and r["cohort"] == cohort
                and r["name"] in names + students
            ]
            if rows:
                means: dict[str, list[float]] = {}
                for r in rows:
                    means.setdefault(r["name"], []).append(float(r["mean"]))
                count = rows[0]["queries"]
                out += [
                    f"- {cohort} ({count} queries): "
                    + ", ".join(f"{n} {np.mean(v):.4f}" for n, v in means.items())
                ]
        out += [""]
    for corpus, run in listing.get("score", {}).items():
        base = args.root / "evaluation" / "P3-target" / "score" / corpus / run
        summary_json = json.loads((base / "summary.json").read_text())
        arms = summary_json["arms"]
        out += [
            f"## Ablation pilot, {corpus} (`{run}`)",
            "",
            f"tune-A {summary_json['queries']['A']} queries (λ selection), tune-B "
            f"{summary_json['queries']['B']} queries (report). Mean nDCG@10 over the nine budgets; "
            "D is the paired difference to E0 at the same seed on tune-B, 95% bootstrap.",
            "",
            "| arm@seed | tune-A | tune-B | D vs E0 (tune-B) | 95% |",
            "|---|---|---|---|---|",
        ]
        for arm, entry in arms.items():
            d = entry.get("tune_B_minus_E0_same_seed")
            dtext = f"{d['D']:+.4f} | [{d['low']:+.4f}, {d['high']:+.4f}]" if d else "— | —"
            out.append(
                f"| {arm} | {entry['tune_A_mean']:.4f} | {entry['tune_B_mean']:.4f} | {dtext} |"
            )
        groups: dict[str, list[float]] = {}
        for arm, entry in arms.items():
            groups.setdefault(arm.split("@")[0], []).append(entry["tune_B_mean"])
        out += [
            "",
            "Seed spread on tune-B (range over seeds): "
            + ", ".join(f"{g} {max(v) - min(v):.4f} (n={len(v)})" for g, v in groups.items()),
            "",
        ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "README.md").write_text("\n".join(out) + "\n")
    print(json.dumps({"written": str(args.out / "README.md")}))
    return 0


# ----------------------------------------------------------------------------- registration


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="P3 target-encoding study (ADR-0019, exploratory)")
    sub = parser.add_subparsers(dest="p3_target_command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", type=Path, default=Path("artifacts"))
        p.add_argument("--corpus", required=True, choices=SMALL)
        p.add_argument("--threads", type=int, default=4)

    p = sub.add_parser("diagnose")
    common(p)
    p.add_argument("--runs", type=Path, required=True, help="F12 runs record of the corpus")
    p.set_defaults(handler=cmd_diagnose)

    p = sub.add_parser("ablate")
    common(p)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--arm", required=True, choices=("E0", "E1", "E2", "E3", "E3c", "E4"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lam", type=float)
    p.add_argument("--device", default="cpu")
    p.add_argument("--clone", type=Path, default=DEFAULT_CLONE)
    p.set_defaults(handler=cmd_ablate)

    p = sub.add_parser("score")
    common(p)
    p.add_argument("--checkpoints", type=Path, required=True, help='JSON {"E0@42": run, ...}')
    p.set_defaults(handler=cmd_score)

    p = sub.add_parser("report")
    p.add_argument("--root", type=Path, default=Path("artifacts"))
    p.add_argument("--runs", type=Path, required=True, help='{"diagnose": {...}, "score": {...}}')
    p.add_argument("--out", type=Path, default=Path("reports/P3-target"))
    p.set_defaults(handler=cmd_report)


__all__ = ["register"]
