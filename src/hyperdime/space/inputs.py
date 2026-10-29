"""An environment's embeddings and query splits, loaded without opening any judgments file.

Protocols P0 and P1 must not read qrels (constitution II), and ``training.inputs.load_inputs``
builds oracle pairs from training and validation judgments. This loader reads only the dataset
manifest, ``queries.jsonl`` (each query carries its split), and the embeddings (F10 research D2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.schemas import DatasetManifest, QueryRecord, load_record, load_records
from hyperdime.contracts.validation import ContractError
from hyperdime.data.prepare import MANIFEST, QUERIES
from hyperdime.embeddings.cache import load_embeddings


@dataclass
class UnlabeledInputs:
    environment: str
    dataset_dir: Path
    embeddings_dir: Path
    documents: np.ndarray  # [N, D] float16, memory-mapped
    doc_ids: list[str]
    query_vectors: np.ndarray  # [Q, D] float16, memory-mapped; every split
    query_rows: dict[str, int]
    splits: dict[str, list[str]]  # sorted query ids per split

    def queries(self, ids: Sequence[str]) -> Tensor:
        """``[len(ids), D]`` float32 query embeddings in the order of ``ids``."""
        rows = [self.query_rows[qid] for qid in ids]
        return torch.from_numpy(np.asarray(self.query_vectors[rows], dtype=np.float32))


def load_unlabeled(root: Path, environment: str) -> UnlabeledInputs:
    """The latest dataset and embeddings of ``environment``; refuses embeddings made from
    another dataset."""
    dataset_dir = latest_artifact(root, "datasets", environment)
    manifest = load_record(DatasetManifest, dataset_dir / MANIFEST)
    embeddings_dir = latest_artifact(root, "embeddings", environment)
    if read_manifest(embeddings_dir).dataset_hash != manifest.fingerprint:
        raise ContractError(f"{embeddings_dir} was not made from {dataset_dir}")
    splits: dict[str, list[str]] = {}
    for record in load_records(QueryRecord, dataset_dir / QUERIES):
        splits.setdefault(record.split, []).append(record.query_id)
    documents_record, documents = load_embeddings(embeddings_dir, "documents")
    queries_record, query_vectors = load_embeddings(embeddings_dir, "queries")
    return UnlabeledInputs(
        environment,
        dataset_dir,
        embeddings_dir,
        documents,
        list(documents_record.ids),
        query_vectors,
        {qid: i for i, qid in enumerate(queries_record.ids)},
        {split: sorted(ids) for split, ids in sorted(splits.items())},
    )
