"""The five spike environments and their leakage-safe query splits."""

from __future__ import annotations

import random
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from hyperdime.data.loaders import BeirDataset, Qrels
from hyperdime.data.splits import split_ids

SPLIT_SEED = 0
VALIDATION_FRACTION = 0.1
HALF_SPLIT_FRACTION = 0.5


@dataclass(frozen=True)
class EnvSpec:
    """A BEIR dataset as an environment.

    ``train_split`` names the qrels split that supplies training labels; ``None`` means the
    dataset only has test labels, so a fixed half of its judged test queries is used for training
    and the other half for evaluation.
    """

    name: str
    instruction: str
    train_split: str | None
    eval_split: str = "test"


# Task instructions follow the Qwen3-Embedding evaluation prompts for these BEIR tasks.
ENVIRONMENTS: dict[str, EnvSpec] = {
    "scifact": EnvSpec(
        "scifact",
        "Given a scientific claim, retrieve documents that support or refute the claim",
        "train",
    ),
    "nfcorpus": EnvSpec(
        "nfcorpus",
        "Given a question, retrieve relevant documents that best answer the question",
        "train",
    ),
    "fiqa": EnvSpec(
        "fiqa",
        "Given a financial question, retrieve user replies that best answer the question",
        "train",
    ),
    "arguana": EnvSpec("arguana", "Given a claim, find documents that refute the claim", None),
    "scidocs": EnvSpec(
        "scidocs",
        "Given a scientific paper title, retrieve paper abstracts that are cited by the "
        "given paper",
        None,
    ),
}


@dataclass(frozen=True)
class QuerySplits:
    """Disjoint query sets; ``train_labels`` covers train and validation, never evaluation."""

    train: list[str]
    validation: list[str]
    evaluation: list[str]
    train_labels: dict[str, dict[str, int]]
    eval_labels: dict[str, dict[str, int]]


def make_splits(spec: EnvSpec, dataset: BeirDataset) -> QuerySplits:
    """Split judged queries so that no evaluation label reaches training."""
    eval_qrels = dataset.qrels[spec.eval_split]
    eval_judged = _usable(eval_qrels, dataset.queries, dataset.corpus)
    if spec.train_split is None:
        pool, evaluation = split_ids(eval_judged, HALF_SPLIT_FRACTION, SPLIT_SEED)
        train_qrels: Qrels = {qid: eval_qrels[qid] for qid in pool}
    else:
        train_qrels = dataset.qrels[spec.train_split]
        pool, evaluation = _usable(train_qrels, dataset.queries, dataset.corpus), eval_judged
    if set(pool) & set(evaluation):
        raise ValueError(f"{spec.name}: training and evaluation queries overlap.")
    train, validation = split_ids(pool, VALIDATION_FRACTION, SPLIT_SEED)
    return QuerySplits(
        train=train,
        validation=validation,
        evaluation=evaluation,
        train_labels={qid: dict(train_qrels[qid]) for qid in pool},
        eval_labels={qid: dict(eval_qrels[qid]) for qid in evaluation},
    )


def sample_corpus(
    dataset: BeirDataset, splits: QuerySplits, cap: int, seed: int = SPLIT_SEED
) -> list[str]:
    """Document IDs judged for any used query, filled with seeded random documents up to ``cap``.

    Every system in an environment is compared on the same sampled corpus, so relative results
    are preserved; absolute scores are not comparable with full-corpus numbers. ``cap <= 0``
    keeps the full corpus.
    """
    if cap <= 0 or cap >= len(dataset.corpus):
        return sorted(dataset.corpus)
    judged = {
        doc
        for labels in (*splits.train_labels.values(), *splits.eval_labels.values())
        for doc in labels
        if doc in dataset.corpus
    }
    rest = sorted(set(dataset.corpus) - judged)
    random.Random(seed).shuffle(rest)
    return sorted(judged) + sorted(rest[: max(0, cap - len(judged))])


def _usable(
    qrels: Mapping[str, Mapping[str, int]], queries: Collection[str], corpus: Collection[str]
) -> list[str]:
    """Sorted queries that exist and have a positive judgment on a document in the corpus."""
    return sorted(
        qid
        for qid, labels in qrels.items()
        if qid in queries and any(g > 0 and doc in corpus for doc, g in labels.items())
    )
