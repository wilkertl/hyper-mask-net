"""Training inputs from an environment's published F2 artifacts (training labels only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.schemas import NegativePool, load_records
from hyperdime.contracts.validation import ContractError
from hyperdime.data.prepare import DatasetBundle, load_environment
from hyperdime.embeddings.cache import load_embeddings
from hyperdime.oracle.targets import oracle_pairs
from hyperdime.training.selection import Supervision


@dataclass
class EnvironmentInputs:
    environment: str
    dataset_dir: Path
    embeddings_dir: Path
    negatives_dir: Path
    bundle: DatasetBundle
    documents: np.ndarray
    doc_ids: list[str]
    query_vectors: np.ndarray
    query_rows: dict[str, int]
    supervision: Supervision


def judgments(bundle: DatasetBundle, name: str) -> dict[str, dict[str, int]]:
    found: dict[str, dict[str, int]] = {}
    for record in bundle.qrels[name]:
        found.setdefault(record.query_id, {})[record.doc_id] = record.relevance
    return found


def load_inputs(root: Path, environment: str) -> EnvironmentInputs:
    """Latest dataset, embeddings, and negatives of ``environment``; all from the same dataset.

    The dataset is loaded without test or evaluation-set judgments, so nothing built here can
    see a test label.
    """
    dataset_dir = latest_artifact(root, "datasets", environment)
    bundle = load_environment(dataset_dir)
    fingerprint = bundle.manifest.fingerprint
    embeddings_dir = latest_artifact(root, "embeddings", environment)
    negatives_dir = latest_artifact(root, "negatives", environment)
    for directory in (embeddings_dir, negatives_dir):
        if read_manifest(directory).dataset_hash != fingerprint:
            raise ContractError(f"{directory} was not made from {dataset_dir}")
    documents_record, documents = load_embeddings(embeddings_dir, "documents")
    queries_record, query_vectors = load_embeddings(embeddings_dir, "queries")
    doc_rows = {doc_id: i for i, doc_id in enumerate(documents_record.ids)}
    query_rows = {qid: i for i, qid in enumerate(queries_record.ids)}
    pools = {
        p.query_id: p.doc_ids for p in load_records(NegativePool, negatives_dir / "negatives.jsonl")
    }
    splits = {}
    for split in ("train", "validation"):
        ids = bundle.query_ids(split)
        labels = judgments(bundle, split)
        splits[split] = (
            ids,
            labels,
            oracle_pairs(ids, query_vectors, query_rows, documents, doc_rows, labels, pools),
        )
    supervision = Supervision(
        train=splits["train"][2],
        validation=splits["validation"][2],
        validation_ids=splits["validation"][0],
        validation_judgments=splits["validation"][1],
    )
    return EnvironmentInputs(
        environment,
        dataset_dir,
        embeddings_dir,
        negatives_dir,
        bundle,
        documents,
        list(documents_record.ids),
        query_vectors,
        query_rows,
        supervision,
    )
