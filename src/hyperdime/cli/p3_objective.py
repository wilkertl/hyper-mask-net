"""``hyperdime p3-objective``: target versus loss versus direct ranking (ADR-0020; exploratory).

- ``train``: one arm, one learning rate, one seed on the F12 fit queries of a small corpus.
- ``score``: exact full-corpus tune-A/tune-B metrics of listed checkpoints at the ten deciles
  (1024 is the parity budget), with fit diagnostics (Jaccard with the contrastive masks, KL to
  ``t_off``).
- ``select``: the learning rate of every arm from tune-A (seed 42) of a grid score run.
- ``report``: README with every table generated from the final score runs and training summaries.

Only fit labels (training) and tune labels (selection on tune-A, reporting on tune-B) are read.
F12 evaluation ids, official test labels and corpus text are never opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.baselines.l2s_official import DEFAULT_CLONE, PUBLISHED, SEED, import_official
from hyperdime.cli.p3_bench import (
    MODEL_REVISION,
    FitInputs,
    _embedding_dir,
    _jsonl,
    _write_json,
)
from hyperdime.cli.p3_bench_arms import _official_dir, _query_vectors, _split_qrels
from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, read_manifest, start_manifest
from hyperdime.contracts.validation import ContractError
from hyperdime.data.query_split import hash_order, id_fingerprint, load_query_split
from hyperdime.evaluation.benchmark import DECILES, PRIMARY, ideal_dcg
from hyperdime.evaluation.objective_study import (
    CONTRASTS,
    INITIALIZED,
    LEARNING_RATES,
    MEANING,
    OFFICIAL_LR,
    POST_HOC_RATES,
    SEEDS,
    TRAINED,
    ArmData,
    arm_payload,
    contrast,
    family_holm,
    parameters,
    reading,
    select_lr,
)
from hyperdime.evaluation.target_study import jaccard, kl_rows, top_k_mask, train
from hyperdime.training.examples import retrieval_examples

COMMAND = "p3-objective"
SMALL = ("scifact", "nfcorpus")
BOOTSTRAP = 10_000
STREAM = 20261115


def device_record(device: str) -> dict[str, Any]:
    """Host, GPU model and UUID of the run (recorded in every manifest config)."""
    record: dict[str, Any] = {"host": socket.gethostname(), "device": device}
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device(device))
        record["gpu"] = props.name
        record["gpu_uuid"] = f"GPU-{props.uuid}"
    return record


def _manifest(root: Path, name: str, config: dict[str, Any], device: str, seed: int) -> Any:
    return start_manifest(
        f"{COMMAND} {name}",
        {**config, "compute": device_record(device)},
        dataset_hash=file_sha256(root / "p3-query-split-v1" / "manifest.json"),
        model_revision=MODEL_REVISION,
        seed=seed,
        device=device,
    )


def fit_inputs(root: Path, corpus: str) -> FitInputs:
    """The F12 fit queries without the corpus text (lengths are not used here)."""
    split = load_query_split(root, corpus)
    order = hash_order(corpus, "fit", split.ids["fit"])
    wanted = set(order)
    dataset = root / "datasets" / corpus / split.source_dataset_run
    emb = _embedding_dir(root, corpus)
    doc_ids = json.loads((emb / "documents.json").read_text())["ids"]
    rows = {q: i for i, q in enumerate(json.loads((emb / "queries.json").read_text())["ids"])}
    qrels: dict[str, dict[str, int]] = {}
    for name in ("train", "validation"):  # never qrels-test
        for record in _jsonl(dataset / f"qrels-{name}.jsonl"):
            if record["query_id"] in wanted:
                qrels.setdefault(record["query_id"], {})[record["doc_id"]] = int(
                    record["relevance"]
                )
    queries = np.load(emb / "queries.npy", mmap_mode="r")
    return FitInputs(
        corpus=corpus,
        query_ids=order,
        queries=np.asarray(queries[[rows[q] for q in order]]),
        doc_ids=doc_ids,
        documents=np.load(emb / "documents.npy", mmap_mode="r"),
        lengths=[],
        qrels=qrels,
        dataset_run=split.source_dataset_run,
        embedding_run=emb.name,
    )


def load_arm_data(root: Path, corpus: str, runs: dict[str, str]) -> tuple[ArmData, dict[str, Any]]:
    official = _official_dir(root, corpus, runs["official"])
    inputs = fit_inputs(root, corpus)
    valid = json.loads((official / "valid_query_ids.json").read_text())
    teachers = root / "evaluation" / "P3-bench" / "teachers" / corpus / runs["teachers"]
    if json.loads((teachers / "valid_query_ids.json").read_text()) != valid:
        raise ContractError("teacher queries differ from the official reference's")
    masks = torch.from_numpy(np.load(teachers / "masks_contrastive.npy"))
    if bool((masks[:, :-1] & ~masks[:, 1:]).any()):
        raise ContractError("contrastive masks are not nested; KS and MS are not equivalent")
    rows = {q: i for i, q in enumerate(inputs.query_ids)}
    x = torch.from_numpy(inputs.queries[[rows[q] for q in valid]].astype(np.float32))
    negatives = np.load(official / "negatives_f2_rows.npy")
    pools = {q: [inputs.doc_ids[r] for r in negatives[i]] for i, q in enumerate(valid)}
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    examples = retrieval_examples(
        valid, inputs.queries, rows, inputs.documents, doc_rows, inputs.qrels, pools
    )
    if examples.skipped or not torch.equal(examples.queries, x):
        raise ContractError("ranking examples do not align with the fit queries")
    data = ArmData(x, torch.load(official / "targets.pt"), masks, examples, tuple(PRIMARY))
    sources = {
        "targets_sha256": file_sha256(official / "targets.pt"),
        "masks_sha256": file_sha256(teachers / "masks_contrastive.npy"),
        "negatives_sha256": file_sha256(official / "negatives_f2_rows.npy"),
        "fit_ids_sha256": id_fingerprint(valid),
    }
    return data, sources


# ----------------------------------------------------------------------------- train


def cmd_train(args: argparse.Namespace) -> int:
    corpus = args.corpus
    if args.lr not in (*LEARNING_RATES, *POST_HOC_RATES):
        raise ContractError(f"lr must be one of {LEARNING_RATES} or post hoc {POST_HOC_RATES}")
    runs = json.loads(args.runs.read_text())
    data, sources = load_arm_data(args.root, corpus, runs)
    payload, loss, budgets = arm_payload(args.arm, data)
    official_dir = _official_dir(args.root, corpus, runs["official"])
    reference = torch.load(official_dir / "state_dict.pt")
    start = (
        {"net.0.weight": reference["net.0.weight"], "net.0.bias": reference["net.0.bias"]}
        if args.arm in INITIALIZED
        else None
    )
    _, epochs = PUBLISHED[corpus]
    module = import_official(args.clone)
    manifest = _manifest(
        args.root,
        "train",
        {
            "corpus": corpus,
            "arm": args.arm,
            "lr": args.lr,
            "grid": "primary" if args.lr in LEARNING_RATES else "post-hoc A1",
            "seed": args.seed,
            "epochs": epochs,
            "runs": runs,
            "init": "l2s-official" if start else "seeded",
            **sources,
        },
        args.device,
        args.seed,
    )
    tick = time.perf_counter()
    result = train(
        data.x,
        payload,
        loss,
        epochs=epochs,
        seed=args.seed,
        device=args.device,
        budgets=budgets,
        lr=args.lr,
        set_seed=module.set_random_seed,
        start=start,
    )
    seconds = time.perf_counter() - tick
    state = {k: v.detach().cpu().clone() for k, v in result.model.state_dict().items()}
    summary: dict[str, Any] = {
        "final_loss": result.final_loss,
        "steps": result.steps,
        "seconds": seconds,
        "parameters": parameters(args.arm),
    }
    if args.arm == "KG" and args.lr == OFFICIAL_LR and args.seed == SEED:
        gaps = [
            float((state[f"net.0.{p}"] - reference[f"net.0.{p}"]).abs().max())
            for p in ("weight", "bias")
        ]
        summary["parity_with_l2s_official"] = {
            "bit_equal": all(
                torch.equal(state[f"net.0.{p}"], reference[f"net.0.{p}"])
                for p in ("weight", "bias")
            ),
            "max_abs_difference": max(gaps),
        }
    with ArtifactWriter(args.root, f"selectors/p3-objective/{corpus}/{args.arm}", manifest) as w:
        torch.save(state, w.path("state_dict.pt"))
        _write_json(w.path("summary.json"), summary)
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "arm": args.arm,
                "lr": args.lr,
                "seed": args.seed,
                **summary,
            }
        )
    )
    return 0


# ----------------------------------------------------------------------------- score


def tune_queries(
    root: Path, corpus: str, doc_rows: dict[str, int]
) -> tuple[list[str], dict[str, str], dict[str, dict[str, int]]]:
    """Tune ids with a positive in the corpus, in hash order, with their tune-A/tune-B half."""
    split = load_query_split(root, corpus)
    ids = hash_order(corpus, "tune", split.ids["tune"])
    qrels = _split_qrels(root, corpus, set(ids))
    kept = [q for q in ids if any(r > 0 and d in doc_rows for d, r in qrels.get(q, {}).items())]
    half = len(kept) // 2
    return kept, {q: ("A" if i < half else "B") for i, q in enumerate(kept)}, qrels


def column_ranks(scores: Tensor, positives: Tensor, chunk: int = 64) -> Tensor:
    """Ranks ``[P, M]`` of ``positives`` under score columns ``[N, M]`` (lower-row ties)."""
    rows = torch.arange(scores.shape[0], device=scores.device)
    out = []
    for start in range(0, scores.shape[1], chunk):
        s = scores[:, start : start + chunk]
        own = s[positives]  # [P, m]
        higher = (s[None] > own[:, None]).sum(1)
        tied = ((s[None] == own[:, None]) & (rows[None, :, None] < positives[:, None, None])).sum(1)
        out.append(1 + higher + tied)
    return torch.cat(out, 1)


def measures(ranks: Tensor, gains: Tensor, idcg: float) -> dict[str, Tensor]:
    """nDCG@10 (linear gains), MRR@10 and Recall@100 per column from ranks ``[P, M]``."""
    r = ranks.to(torch.float64)
    first = r.min(0).values
    return {
        "ndcg10": ((gains[:, None] / torch.log2(1 + r)) * (r <= 10)).sum(0) / idcg,
        "mrr10": torch.where(first <= 10, 1.0 / first, torch.zeros_like(first)),
        "recall100": (r <= 100).to(torch.float64).mean(0),
    }


def _scores(state: dict[str, Tensor], x: Tensor, budget: int | None) -> Tensor:
    z = x @ state["net.0.weight"].T.to(x) + state["net.0.bias"].to(x)
    if "offsets" in state and budget is not None:
        z = z + state["offsets"][budget].to(x)
    return z


def _state_path(root: Path, corpus: str, key: str, run: str, runs: dict[str, str]) -> Path:
    arm, _, seed = key.split("|")
    if arm == "REF" and seed == str(SEED):  # the F12 checkpoint itself
        return _official_dir(root, corpus, runs["official"]) / "state_dict.pt"
    if arm == "REF":  # the official recipe at another seed is KG at lr 1e-4
        arm = "KG"
    return root / "selectors" / "p3-objective" / corpus / arm / run / "state_dict.pt"


def cmd_score(args: argparse.Namespace) -> int:
    corpus = args.corpus
    device = torch.device(args.device)
    runs = json.loads(args.runs.read_text())
    listing: dict[str, str] = json.loads(args.checkpoints.read_text())  # {"KG|0.0001|42": run}
    inputs = fit_inputs(args.root, corpus)
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    ids, halves, qrels = tune_queries(args.root, corpus, doc_rows)
    vectors = _query_vectors(args.root, corpus, ids, inputs)
    x = torch.from_numpy(vectors.astype(np.float32)).to(device)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    docs = torch.from_numpy(np.asarray(inputs.documents).astype(np.float64)).to(device)
    states = {k: torch.load(_state_path(args.root, corpus, k, r, runs)) for k, r in listing.items()}
    keys = list(states)
    budget_index = {k: (PRIMARY.index(k) if k in PRIMARY else None) for k in DECILES}
    manifest = _manifest(
        args.root,
        "score",
        {
            "corpus": corpus,
            "checkpoints": listing,
            "runs": runs,
            "tune_ids_sha256": id_fingerprint(ids),
            "budgets": list(DECILES),
        },
        str(device),
        STREAM,
    )
    tick = time.perf_counter()
    importance = {
        (key, k): _scores(states[key], x, budget_index[k]) for key in keys for k in DECILES
    }
    rows_out = []
    for qi, qid in enumerate(ids):
        pos_rows = [doc_rows[d] for d, r in qrels[qid].items() if r > 0 and d in doc_rows]
        grades = [float(qrels[qid][inputs.doc_ids[r]]) for r in pos_rows]
        q = torch.from_numpy(vectors[qi].astype(np.float64)).to(device)
        columns, names = [], []
        for key in keys:
            for k in DECILES:
                mask = top_k_mask(importance[(key, k)][qi], k)
                columns.append(q * mask.to(torch.float64))
                names.append((key, k))
        scores = docs @ torch.stack(columns, 1)  # [N, M] exact float64
        ranks = column_ranks(scores, torch.tensor(pos_rows, device=device))
        found = measures(
            ranks, torch.tensor(grades, dtype=torch.float64, device=device), ideal_dcg(grades)
        )
        values = {m: v.cpu().tolist() for m, v in found.items()}
        for j, (key, k) in enumerate(names):
            arm, lr, seed = key.split("|")
            rows_out.append(
                {
                    "half": halves[qid],
                    "query": qid,
                    "k": k,
                    "arm": arm,
                    "lr": lr,
                    "seed": seed,
                    **{m: values[m][j] for m in values},
                }
            )
    diagnostics = _fit_diagnostics(args.root, corpus, runs, states, device, args.fit_ndcg)
    seconds = time.perf_counter() - tick
    with ArtifactWriter(args.root, f"evaluation/P3-objective/score/{corpus}", manifest) as w:
        _write_csv(w.path("per_query.csv"), rows_out)
        _write_json(w.path("diagnostics.json"), diagnostics)
        _write_json(
            w.path("summary.json"),
            {
                "queries": {h: sum(1 for q in ids if halves[q] == h) for h in "AB"},
                "checkpoints": len(keys),
                "seconds": seconds,
            },
        )
    print(json.dumps({"run_id": manifest.run_id, "checkpoints": len(keys), "seconds": seconds}))
    return 0


def _fit_diagnostics(
    root: Path,
    corpus: str,
    runs: dict[str, str],
    states: dict[str, Any],
    device: torch.device,
    fit_ndcg: bool = False,
) -> dict[str, Any]:
    """Per checkpoint, on fit queries (in sample): Jaccard of the student top-k with the
    contrastive mask per budget, and KL(t_off ‖ softmax z) on the raw query (training input).
    With ``fit_ndcg``, seed-42 checkpoints also get fit nDCG@10 per budget, and the decoded
    contrastive mask its own (the regret is the difference)."""
    official = _official_dir(root, corpus, runs["official"])
    teachers = root / "evaluation" / "P3-bench" / "teachers" / corpus / runs["teachers"]
    valid = json.loads((official / "valid_query_ids.json").read_text())
    inputs = fit_inputs(root, corpus)
    rows = {q: i for i, q in enumerate(inputs.query_ids)}
    raw = torch.from_numpy(inputs.queries[[rows[q] for q in valid]].astype(np.float32)).to(device)
    unit = raw / (raw.norm(dim=1, keepdim=True) + 1e-12)
    masks = torch.from_numpy(np.load(teachers / "masks_contrastive.npy")).to(device)
    target = torch.load(official / "targets.pt").to(device)
    out: dict[str, Any] = {}
    for key, state in states.items():
        jac = []
        for b, k in enumerate(PRIMARY):
            jac.append(float(jaccard(top_k_mask(_scores(state, unit, b), k), masks[:, b]).mean()))
        log_p = torch.log_softmax(_scores(state, raw, None).to(torch.float64), -1)
        out[key] = {"jaccard": jac, "kl_t_off": float(kl_rows(target, log_p).mean())}
    if not fit_ndcg:
        return out
    chosen = [key for key in states if key.split("|")[2] == str(SEED)]
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    docs = torch.from_numpy(np.asarray(inputs.documents).astype(np.float64)).to(device)
    sums = {key: np.zeros(len(PRIMARY)) for key in [*chosen, "decoded:contrastive"]}
    for qi, qid in enumerate(valid):
        pos = [doc_rows[d] for d, r in inputs.qrels[qid].items() if r > 0 and d in doc_rows]
        grades = [float(inputs.qrels[qid][inputs.doc_ids[r]]) for r in pos]
        q = raw[qi].to(torch.float64)
        cols = [masks[qi]] + [
            top_k_mask(_scores(states[key], unit[qi : qi + 1], b)[0], k)[None]
            for key in chosen
            for b, k in enumerate(PRIMARY)
        ]
        stacked = torch.cat([c.reshape(-1, c.shape[-1]) for c in cols])  # [B + |chosen|·B, D]
        scores = docs @ (q[None] * stacked.to(torch.float64)).T
        ranks = column_ranks(scores, torch.tensor(pos, device=device))
        nd = (
            measures(
                ranks, torch.tensor(grades, dtype=torch.float64, device=device), ideal_dcg(grades)
            )["ndcg10"]
            .cpu()
            .numpy()
        )
        b = len(PRIMARY)
        sums["decoded:contrastive"] += nd[:b]
        for j, key in enumerate(chosen):
            sums[key] += nd[b * (j + 1) : b * (j + 2)]
    for key, total in sums.items():
        out.setdefault(key, {})["fit_ndcg10"] = list(total / len(valid))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------------------- select


def per_query_means(
    rows: list[dict[str, str]],
    half: str,
    metric: str = "ndcg10",
    budgets: tuple[int, ...] = PRIMARY,
) -> dict[tuple[str, str, str], dict[str, float]]:
    """``{(arm, lr, seed): {query: mean over budgets}}`` for one half."""
    acc: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    wanted = {str(k) for k in budgets}
    for r in rows:
        if r["half"] == half and r["k"] in wanted:
            acc.setdefault((r["arm"], r["lr"], r["seed"]), {}).setdefault(r["query"], []).append(
                float(r[metric])
            )
    return {key: {q: float(np.mean(v)) for q, v in queries.items()} for key, queries in acc.items()}


def cmd_select(args: argparse.Namespace) -> int:
    score_dir = args.root / "evaluation" / "P3-objective" / "score" / args.corpus / args.score_run
    rows = _read_csv(score_dir / "per_query.csv")
    means = per_query_means(rows, "A")  # tune-A only
    listing = read_manifest(score_dir).config["checkpoints"]
    selection: dict[str, Any] = {
        "kind": "p3-objective-selection",
        "corpus": args.corpus,
        "score_run": args.score_run,
        "rule": "max tune-A mean nDCG@10 over 102..921 at seed 42; ties to the smaller lr",
        "arms": {},
    }
    for arm in TRAINED:
        table = {
            float(lr): float(np.mean(list(v.values())))
            for (a, lr, seed), v in means.items()
            if a == arm and seed == str(SEED)
        }
        rates = (*LEARNING_RATES, *POST_HOC_RATES) if args.extended else LEARNING_RATES
        table = {lr: v for lr, v in table.items() if lr in rates}
        if set(table) != set(rates):
            raise ContractError(f"{arm}: grid incomplete {sorted(table)}")
        lr = select_lr(table)
        selection["arms"][arm] = {
            "tune_A": {str(k): v for k, v in sorted(table.items())},
            "selected_lr": lr,
            "seed42_run": listing[f"{arm}|{lr}|{SEED}"],
        }
    args.out.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps({a: v["selected_lr"] for a, v in selection["arms"].items()}))
    return 0


# ----------------------------------------------------------------------------- report


def _arrays(
    means: dict[tuple[str, str, str], dict[str, float]], arm: str, lr: str, queries: list[str]
) -> np.ndarray:
    return np.array([[means[(arm, lr, str(s))][q] for q in queries] for s in SEEDS])


def corpus_results(rows: list[dict[str, str]], selection: dict[str, Any]) -> dict[str, Any]:
    """Arm table, contrasts and curves of one corpus from the final score rows (tune-B)."""
    lrs = {arm: str(v["selected_lr"]) for arm, v in selection["arms"].items()}
    lrs["REF"] = str(OFFICIAL_LR)
    out: dict[str, Any] = {"arms": {}, "contrasts": {}, "curves": {}, "lr": lrs}
    for metric in ("ndcg10", "mrr10", "recall100"):
        means = per_query_means(rows, "B", metric)
        queries = sorted(next(iter(means.values())))
        arrays = {arm: _arrays(means, arm, lr, queries) for arm, lr in lrs.items()}
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(STREAM)))
        for arm, values in arrays.items():
            entry = out["arms"].setdefault(arm, {})
            entry[metric] = float(values.mean())
            if metric == "ndcg10" and arm != "REF":
                entry["vs_REF"] = contrast(values, arrays["REF"], rng, len(CONTRASTS), BOOTSTRAP)
        if metric == "ndcg10":
            out["queries"] = len(queries)
            for name, a, b in CONTRASTS:
                out["contrasts"][name] = contrast(
                    arrays[a], arrays[b], rng, len(CONTRASTS), BOOTSTRAP
                )
            family_holm(out["contrasts"])
    for arm, lr in lrs.items():
        for k in DECILES:
            curve = [
                float(r["ndcg10"])
                for r in rows
                if r["half"] == "B" and r["arm"] == arm and r["lr"] == lr and r["k"] == str(k)
            ]
            out["curves"].setdefault(arm, {})[k] = float(np.mean(curve))
    full = {
        r["query"]: float(r["ndcg10"])
        for r in rows
        if r["k"] == str(DECILES[-1]) and r["arm"] == "REF"
    }
    out["parity_1024_max_gap"] = max(
        abs(float(r["ndcg10"]) - full[r["query"]]) for r in rows if r["k"] == str(DECILES[-1])
    )
    return out


def cmd_report(args: argparse.Namespace) -> int:
    listing = json.loads(args.runs.read_text())  # {corpus: {"score": run, "selection": path, ...}}
    results, costs, diags = {}, {}, {}
    for corpus, entry in listing.items():
        score_dir = args.root / "evaluation" / "P3-objective" / "score" / corpus / entry["score"]
        selection = json.loads((args.runs.parent / entry["selection"]).read_text())
        results[corpus] = corpus_results(_read_csv(score_dir / "per_query.csv"), selection)
        diags[corpus] = json.loads((score_dir / "diagnostics.json").read_text())
        trained = json.loads((args.runs.parent / entry["trainings"]).read_text())
        costs[corpus] = _costs(args.root, corpus, trained)
    text = _render(results, costs, diags, listing)
    if args.label:
        heading, rest = text.split("\n", 1)
        text = f"{heading} [{args.label}]\n\n> {args.label}: see ADR-0020 amendment A1.\n{rest}"
    (args.out / "README.md").write_text(text)
    _write_json(args.out / "results.json", results)
    print(json.dumps({"written": str(args.out / "README.md")}))
    return 0


def _costs(root: Path, corpus: str, trained: dict[str, str]) -> dict[str, Any]:
    """Per arm: runs, total and mean training seconds, and devices, from training summaries."""
    out: dict[str, Any] = {}
    for key, run in trained.items():
        arm = key.split("|")[0]
        base = root / "selectors" / "p3-objective" / corpus / arm / run
        summary = json.loads((base / "summary.json").read_text())
        compute = read_manifest(base).config["compute"]
        entry = out.setdefault(arm, {"runs": 0, "seconds": 0.0, "devices": set()})
        entry["runs"] += 1
        entry["seconds"] += summary["seconds"]
        entry["devices"].add(f"{compute['host']} {compute.get('gpu_uuid', compute['device'])}")
        if "parity_with_l2s_official" in summary:
            out["parity"] = summary["parity_with_l2s_official"]
    for arm, entry in out.items():
        if arm != "parity":
            entry["devices"] = sorted(entry["devices"])
    return out


def _fmt(c: dict[str, Any]) -> str:
    return f"{c['D']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"


def _render(
    results: dict[str, Any], costs: dict[str, Any], diags: dict[str, Any], listing: dict[str, Any]
) -> str:
    corpora = list(results)
    arms = ["REF", *TRAINED]
    out = [
        "# P3-objective: target versus loss versus direct ranking (exploratory)",
        "",
        "Generated by `hyperdime p3-objective report`; protocol [ADR-0020]"
        "(../../docs/decisions/ADR-0020-p3-objective-study.md). Development data: fit queries",
        "train, tune-A selects the learning rate, tune-B reports. Tune-B was already observed in",
        "ADR-0019, so nothing here is confirmatory. Exact full-corpus float64 scoring; query-only",
        "binary masks; no renormalization. Values are means over seeds 42, 0, 1 (paired).",
        "",
        "## Ranking (tune-B, mean nDCG@10 over k = 102…921, minus REF with 95% paired bootstrap)",
        "",
        "| arm | lr | " + " | ".join(f"{c} nDCG@10 | {c} − REF" for c in corpora) + " |",
        "|---|---|" + "---|---|" * len(corpora),
    ]
    order = sorted(arms, key=lambda a: -np.mean([results[c]["arms"][a]["ndcg10"] for c in corpora]))
    for arm in order:
        cells = []
        for c in corpora:
            entry = results[c]["arms"][arm]
            cells += [f"{entry['ndcg10']:.4f}", _fmt(entry["vs_REF"]) if "vs_REF" in entry else "—"]
        lrs = sorted({results[c]["lr"][arm] for c in corpora})
        out.append(f"| {arm} | {', '.join(lrs)} | " + " | ".join(cells) + " |")
    out += [
        "",
        "## Contrasts (tune-B; D, 95% interval, Bonferroni interval at 0.05/10, Holm p, seed "
        "spread)",
        "",
        "| ID | contrast | meaning | " + " | ".join(corpora) + " | reading |",
        "|---|---|---|" + "---|" * len(corpora) + "---|",
    ]
    for name, a, b in CONTRASTS:
        cells = []
        for c in corpora:
            r = results[c]["contrasts"][name]
            cells.append(
                f"{_fmt(r)}; Bonf [{r['bonferroni_low']:+.4f}, {r['bonferroni_high']:+.4f}]; "
                f"Holm p {r['holm_p']:.3f}; spread {r['spread']:.4f}"
            )
        verdict = reading({c: results[c]["contrasts"][name] for c in corpora})
        out.append(
            f"| {name} | {a} − {b} | {MEANING[name]} | " + " | ".join(cells) + f" | {verdict} |"
        )
    out += ["", "C5 (target × loss interaction) = C2 − C1:"]
    for c in corpora:
        r = results[c]["contrasts"]
        out.append(f"- {c}: {r['C2']['D'] - r['C1']['D']:+.4f}")
    out += ["", "## Secondary (tune-B, mean over k = 102…921)", ""]
    out += [
        "| arm | " + " | ".join(f"{c} MRR@10 | {c} Recall@100" for c in corpora) + " |",
        "|---|" + "---|---|" * len(corpora),
    ]
    for arm in arms:
        cells = []
        for c in corpora:
            e = results[c]["arms"][arm]
            cells += [f"{e['mrr10']:.4f}", f"{e['recall100']:.4f}"]
        out.append(f"| {arm} | " + " | ".join(cells) + " |")
    for c in corpora:
        out += ["", f"### nDCG@10 per budget, {c} (tune-B; 1024 is the parity budget)", ""]
        out += ["| k | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for k in DECILES:
            out.append(
                f"| {k} | " + " | ".join(f"{results[c]['curves'][a][k]:.4f}" for a in arms) + " |"
            )
        out.append("")
        out.append(
            f"Parity at k = 1024: largest gap to REF {results[c]['parity_1024_max_gap']:.2e}."
        )
    out += ["", "## Diagnostics (fit queries, in sample; not win criteria)", ""]
    out += [
        "| arm | "
        + " | ".join(f"{c} Jaccard with m_k | {c} KL(t_off ‖ p)" for c in corpora)
        + " |",
        "|---|" + "---|---|" * len(corpora),
    ]
    for arm in arms:
        cells = []
        for c in corpora:
            lr = results[c]["lr"][arm]
            vals = [
                v for k, v in diags[c].items() if k.split("|")[0] == arm and k.split("|")[1] == lr
            ]
            cells += [
                f"{np.mean([np.mean(v['jaccard']) for v in vals]):.3f}",
                f"{np.mean([v['kl_t_off'] for v in vals]):.4f}",
            ]
        out.append(f"| {arm} | " + " | ".join(cells) + " |")
    out += [
        "",
        "## Cost",
        "",
        "Online, per query: every arm is one 1024 × 1024 linear map "
        "(1,049,600 parameters, ≈ 2.1 MFLOP), top-k, masked exact scoring; KBk and KBa add "
        "9,216 offset parameters and one vector addition.",
        "",
    ]
    out += [
        "| arm | parameters | " + " | ".join(f"{c} runs | {c} training s" for c in corpora) + " |",
        "|---|---|" + "---|---|" * len(corpora),
    ]
    for arm in TRAINED:
        cells = []
        for c in corpora:
            e = costs[c].get(arm, {"runs": 0, "seconds": 0.0})
            cells += [str(e["runs"]), f"{e['seconds']:.0f}"]
        out.append(f"| {arm} | {parameters(arm):,} | " + " | ".join(cells) + " |")
    out += [""]
    for c in corpora:
        p = costs[c].get("parity")
        devices = sorted({d for a, e in costs[c].items() if a != "parity" for d in e["devices"]})
        out.append(
            f"- {c}: trainer parity with `l2s-official` (KG, lr 1e-4, seed 42): {p}; "
            f"devices: {'; '.join(devices)}; tune-B queries {results[c]['queries']}."
        )
    out += ["", "Runs: " + json.dumps(listing, sort_keys=True), ""]
    return "\n".join(out)


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="P3 objective study (ADR-0020, exploratory)")
    sub = parser.add_subparsers(dest="p3_objective_command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", type=Path, default=Path("artifacts"))
        p.add_argument("--corpus", required=True, choices=SMALL)

    p = sub.add_parser("train")
    common(p)
    p.add_argument("--runs", type=Path, required=True, help="F12 runs record of the corpus")
    p.add_argument("--arm", required=True, choices=TRAINED)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--clone", type=Path, default=DEFAULT_CLONE)
    p.set_defaults(handler=cmd_train)

    p = sub.add_parser("score")
    common(p)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True, help='JSON {"KG|0.0001|42": run}')
    p.add_argument("--device", default="cpu")
    p.add_argument("--fit-ndcg", action="store_true", help="fit nDCG@10 of seed-42 checkpoints")
    p.set_defaults(handler=cmd_score)

    p = sub.add_parser("select")
    common(p)
    p.add_argument("--score-run", required=True)
    p.add_argument("--extended", action="store_true", help="amendment A1: six-point grid")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=cmd_select)

    p = sub.add_parser("report")
    p.add_argument("--root", type=Path, default=Path("artifacts"))
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default="", help="e.g. 'post hoc A1' (printed in the heading)")
    p.set_defaults(handler=cmd_report)
