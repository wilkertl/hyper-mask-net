"""The authors' Learning2Select as F12's single official reference (ADR-0018, research B3).

The authors' functions are imported from their clone at a pinned commit and called unchanged, in
the order of their ``main`` and ``run_single_experiment``: seed once per process, label generation
(top 1000 by unmasked score, then known positives removed, 64 draws with replacement or the
fallback), ``DataLoader`` with shuffling, model initialization, training, final-epoch weights.

The adapter adds two things without changing the authors' random stream: it reorders our F2
documents into the authors' corpus order (``len(title + text)`` descending, stable), and it records
each query's negatives by observing the ``torch.randint`` draws and rebuilding the pool. The
rebuilt negatives must reproduce the authors' target for every query bit for bit.

On corpora too large for a second normalized copy (MS MARCO), ``record_pools`` also observes the
authors' own ``torch.topk`` candidates instead of recomputing them; the check is the same: the
recorded negatives, combined with the positives through the authors' operations on the normalized
rows they touch, must reproduce every target bit for bit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import Tensor

L2S_COMMIT = "55f010f5f3e82a6d171ce0909793e72f9c066ec7"
DEFAULT_CLONE = Path(os.environ.get("HYPERDIME_L2S_CLONE", "/home/wilker/Code/learning2select"))
SEED = 42
PUBLISHED = {"scifact": (0.05, 200), "nfcorpus": (0.05, 75), "msmarco": (0.10, 50)}
PUBLISHED_SOURCE = {"msmarco": "msmarco_subset_50k (transferred unchanged to the full corpus)"}


def transfer_flag(corpus: str) -> str | None:
    """US1/AC4: the sentence every report carries when τ and epochs were published elsewhere."""
    if corpus not in PUBLISHED_SOURCE:
        return None
    temperature, epochs = PUBLISHED[corpus]
    return (
        f"τ = {temperature:.2f} and {epochs} epochs were published for the authors' 50k-passage "
        "subset (msmarco_subset_50k) and are applied unchanged to the full corpus; they were not "
        "retuned."
    )


_RAN = False


def clone_commit(clone: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def import_official(clone: Path, expected: str = L2S_COMMIT) -> ModuleType:
    """Import ``train_and_eval`` from the clone after checking its commit."""
    commit = clone_commit(clone)
    if commit != expected:
        raise RuntimeError(f"{clone} is at {commit}, not {expected}")
    if str(clone) not in sys.path:
        sys.path.insert(0, str(clone))
    spec = importlib.util.spec_from_file_location("l2s_train_and_eval", clone / "train_and_eval.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authors_order(lengths: Sequence[int]) -> list[int]:
    """F2 rows in the authors' corpus order: length descending, file order among ties."""
    return sorted(range(len(lengths)), key=lambda row: lengths[row], reverse=True)


@dataclass
class OfficialResult:
    state_dict: dict[str, Tensor]
    targets: Tensor  # [Q_valid, D]
    valid_query_ids: list[str]
    negatives: np.ndarray  # [Q_valid, M] indices in the authors' corpus order
    fallback: np.ndarray  # [Q_valid] bool, the pool had fewer than M documents
    rng_after_labels: dict[str, Any]
    final_train_loss: float | None
    seconds: dict[str, float]


def run_official(
    module: ModuleType,
    query_ids: Sequence[str],
    query_vectors: Tensor,
    corpus_ids: Sequence[str],
    documents: Tensor,
    qrels: Mapping[str, Mapping[str, int]],
    *,
    temperature: float,
    epochs: int,
    device: str,
    seed: int = SEED,
    lr: float = 1e-4,
    batch_size: int = 32,
    hard_negative_topk: int = 1000,
    num_negatives: int = 64,
    train: bool = True,
    record_pools: bool = False,
) -> OfficialResult:
    """One official run; ``corpus_ids`` and ``documents`` authors' order, once per process."""
    global _RAN
    if _RAN:
        raise RuntimeError("the authors seed once per process; run one corpus per process")
    _RAN = True
    module.set_random_seed(seed)
    draws: list[Tensor] = []
    tops: list[Tensor] = []
    original = torch.randint
    original_topk = torch.topk

    def observed(*args: Any, **kwargs: Any) -> Tensor:
        out = original(*args, **kwargs)
        draws.append(out.detach().clone())
        return out

    def observed_topk(*args: Any, **kwargs: Any) -> Any:
        out = original_topk(*args, **kwargs)
        tops.append(out.indices.detach().clone())
        return out

    tick = time.perf_counter()
    torch.randint = observed
    if record_pools:
        torch.topk = observed_topk
    try:
        x_train, y_train = module.generate_soft_oracle_labels_multilevel(
            query_ids=list(query_ids),
            corpus_ids=list(corpus_ids),
            q_emb=query_vectors,
            d_emb=documents,
            qrels=qrels,
            temperature=temperature,
            K=hard_negative_topk,
            M=num_negatives,
            device=device,
        )
    finally:
        torch.randint = original
        torch.topk = original_topk
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    label_seconds = time.perf_counter() - tick
    rng: dict[str, Any] = {"cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        rng["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    valid, negatives, fallback = _rebuild_negatives(
        module,
        query_ids,
        query_vectors,
        corpus_ids,
        documents,
        qrels,
        draws,
        y_train,
        temperature=temperature,
        k_hard=hard_negative_topk,
        m=num_negatives,
        device=device,
        recorded_tops=tops if record_pools else None,
    )
    state: dict[str, Tensor] = {}
    loss = None
    train_seconds = 0.0
    if train:
        tick = time.perf_counter()
        dataset = module.PredictorDataset(x_train, y_train)
        loader = module.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        predictor = module.DimensionPredictor(input_dim=query_vectors.shape[1]).to(device)
        predictor, loss = module.train_predictor(
            model=predictor, train_loader=loader, epochs=epochs, device=device, lr=lr
        )
        state = {
            name: value.detach().cpu().clone() for name, value in predictor.state_dict().items()
        }
        train_seconds = time.perf_counter() - tick
    return OfficialResult(
        state_dict=state,
        targets=y_train,
        valid_query_ids=valid,
        negatives=negatives,
        fallback=fallback,
        rng_after_labels=rng,
        final_train_loss=None if loss is None else float(loss),
        seconds={"labels": label_seconds, "train": train_seconds},
    )


def _rebuild_negatives(
    module: ModuleType,
    query_ids: Sequence[str],
    query_vectors: Tensor,
    corpus_ids: Sequence[str],
    documents: Tensor,
    qrels: Mapping[str, Mapping[str, int]],
    draws: list[Tensor],
    targets: Tensor,
    *,
    temperature: float,
    k_hard: int,
    m: int,
    device: str,
    recorded_tops: list[Tensor] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Rebuild each query's pool with the authors' operations and combine it with its draw.

    With ``recorded_tops`` (the authors' own top-``k_hard`` candidates, observed during label
    generation) no full normalized copy of the corpus is made: only the rows a target touches are
    normalized, row by row as in the authors' code, and the target check is unchanged."""
    q_mat = module.l2_normalize_rows(query_vectors.to(device).float())
    n_docs = documents.shape[0]
    if recorded_tops is None:
        d_mat = module.l2_normalize_rows(documents.to(device).float())

        def rows_of(index: Tensor) -> Tensor:
            picked: Tensor = d_mat[index]
            return picked
    else:
        if len(recorded_tops) != len(draws):
            raise RuntimeError(f"{len(recorded_tops)} recorded pools for {len(draws)} draws")

        def rows_of(index: Tensor) -> Tensor:
            picked = documents[index.cpu()].to(device).float()
            normalized: Tensor = module.l2_normalize_rows(picked)
            return normalized

    index = module.build_id_index(list(corpus_ids))
    valid: list[str] = []
    negatives: list[np.ndarray] = []
    fallback: list[bool] = []
    for qi, qid in enumerate(query_ids):
        rows, grades = [], []
        for doc_id, rel in qrels.get(qid, {}).items():
            if rel > 0 and doc_id in index and index[doc_id] < n_docs:
                rows.append(index[doc_id])
                grades.append(float(rel))
        if not rows:
            continue
        pos = torch.tensor(rows, device=device, dtype=torch.long)
        draw = draws[len(valid)].to(device)
        if recorded_tops is None:
            top = torch.topk(d_mat @ q_mat[qi], k=min(k_hard, n_docs)).indices
        else:
            top = recorded_tops[len(valid)].to(device)
        pool = top[~torch.isin(top, pos)]
        if len(pool) >= m:
            neg = pool[draw]
        elif len(pool) > 0:
            neg = torch.cat([pool, draw])
        else:
            neg = draw
        weights = torch.pow(2.0, torch.tensor(grades, device=device)) - 1.0
        weights = (weights / (weights.sum() + 1e-9)).unsqueeze(1)
        delta = (rows_of(pos) * weights).sum(0) - rows_of(neg).mean(0)
        probs = torch.softmax(delta * q_mat[qi] / temperature, dim=0)
        if not torch.equal(probs.cpu(), targets[len(valid)]):
            raise RuntimeError(f"rebuilt negatives do not reproduce the official target of {qid}")
        valid.append(qid)
        negatives.append(neg.cpu().numpy())
        fallback.append(len(pool) < m)
    if len(valid) != len(draws):
        raise RuntimeError(f"{len(draws)} draws for {len(valid)} queries with positives")
    return valid, np.stack(negatives), np.array(fallback)


def _run_spec(spec_path: Path, out: Path) -> None:
    """Entry for one isolated process: inputs in F2 order from an ``.npz`` and a JSON spec."""
    spec = json.loads(spec_path.read_text())
    arrays = np.load(spec["arrays"])
    order = authors_order(spec["lengths"])
    module = import_official(Path(spec["clone"]))
    corpus_ids = [spec["doc_ids"][row] for row in order]
    documents = torch.from_numpy(np.asarray(arrays["documents"], dtype=np.float32)[order])
    queries = torch.from_numpy(np.asarray(arrays["queries"], dtype=np.float32))
    result = run_official(
        module,
        spec["query_ids"],
        queries,
        corpus_ids,
        documents,
        spec["qrels"],
        temperature=spec["temperature"],
        epochs=spec["epochs"],
        device=spec.get("device", "cpu"),
        record_pools=spec.get("record_pools", False),
    )
    out.mkdir(parents=True, exist_ok=True)
    torch.save(result.state_dict, out / "state_dict.pt")
    torch.save(result.targets, out / "targets.pt")
    np.save(out / "negatives.npy", result.negatives)
    np.save(out / "fallback.npy", result.fallback)
    (out / "valid.json").write_text(json.dumps(result.valid_query_ids))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One isolated official L2S run (F12 parity).")
    parser.add_argument("spec", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    _run_spec(args.spec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
