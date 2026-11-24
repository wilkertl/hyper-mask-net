"""Record how P3's published F2 inputs differ from the F0 spike-derived inputs of its earlier runs.

    python scripts/checks/p3_input_differences.py artifacts artifacts/p3-spike

Writes ``<f2_root>/evaluation/P3-inputs/all/<run_id>/differences.json`` with a run manifest, for
``hyperdime report P3`` to print. Reads embeddings and training judgments only. Compares
identifiers, counts, hashes, and vectors over shared ids; it stores no id and no text, and an
environment with no spike inputs is recorded as such rather than skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.cli.analyze_representations import load_training_judgments
from hyperdime.contracts.hashing import NO_DATASET, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.schemas import DatasetManifest, EmbeddingArtifact, load_record
from hyperdime.contracts.validation import ContractError
from hyperdime.data.prepare import MANIFEST, qrels_file
from hyperdime.space.inputs import UnlabeledInputs, load_unlabeled

CHUNK = 8192


def _sets(left: set[str], right: set[str]) -> dict[str, int]:
    return {
        "shared": len(left & right),
        "only_f2": len(left - right),
        "only_spike": len(right - left),
    }


def _cosines(
    left: np.ndarray, left_rows: dict[str, int], right: np.ndarray, right_rows: dict[str, int]
) -> dict[str, Any]:
    """Cosine and largest coordinate difference between the two vectors of every shared id."""
    shared = sorted(left_rows.keys() & right_rows.keys())
    low, total, largest = 1.0, 0.0, 0.0
    for start in range(0, len(shared), CHUNK):
        ids = shared[start : start + CHUNK]
        a = np.asarray(left[[left_rows[i] for i in ids]], dtype=np.float64)
        b = np.asarray(right[[right_rows[i] for i in ids]], dtype=np.float64)
        cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        low = min(low, float(cosine.min()))
        total += float(cosine.sum())
        largest = max(largest, float(np.abs(a - b).max()))
    if not shared:
        return {
            "rows": 0,
            "min_cosine": None,
            "mean_cosine": None,
            "max_coordinate_difference": None,
        }
    return {
        "rows": len(shared),
        "min_cosine": low,
        "mean_cosine": total / len(shared),
        "max_coordinate_difference": largest,
    }


def _judged_pairs(inputs: UnlabeledInputs) -> set[tuple[str, str, int]]:
    judgments = load_training_judgments(inputs.dataset_dir, inputs.splits.get("train", []))
    return {(q, d, g) for q, labels in judgments.items() for d, g in labels.items()}


def compare(f2_root: Path, spike_root: Path, env: str) -> dict[str, Any]:
    """Differences for one environment; ``spike_inputs`` is false when only F2 has it."""
    f2 = load_unlabeled(f2_root, env)
    try:
        spike = load_unlabeled(spike_root, env)
    except ContractError:
        return {"spike_inputs": False}
    records = {}
    for name, inputs in (("f2", f2), ("spike", spike)):
        records[name] = {
            "dataset": load_record(DatasetManifest, inputs.dataset_dir / MANIFEST),
            "documents": load_record(EmbeddingArtifact, inputs.embeddings_dir / "documents.json"),
            "queries": load_record(EmbeddingArtifact, inputs.embeddings_dir / "queries.json"),
        }
    f2_pairs, spike_pairs = _judged_pairs(f2), _judged_pairs(spike)
    f2_rows = {d: i for i, d in enumerate(f2.doc_ids)}
    spike_rows = {d: i for i, d in enumerate(spike.doc_ids)}
    return {
        "spike_inputs": True,
        "same": {
            "dataset_fingerprint": records["f2"]["dataset"].fingerprint
            == records["spike"]["dataset"].fingerprint,
            "train_judgments_file": file_sha256(f2.dataset_dir / qrels_file("train"))
            == file_sha256(spike.dataset_dir / qrels_file("train")),
            "documents_payload": records["f2"]["documents"].payload.sha256
            == records["spike"]["documents"].payload.sha256,
            "queries_payload": records["f2"]["queries"].payload.sha256
            == records["spike"]["queries"].payload.sha256,
            "model_revision": records["f2"]["documents"].model_revision
            == records["spike"]["documents"].model_revision,
            "instruction": records["f2"]["queries"].instruction
            == records["spike"]["queries"].instruction,
            "split_seed": records["f2"]["dataset"].split_seed
            == records["spike"]["dataset"].split_seed,
            "validation_fraction": records["f2"]["dataset"].validation_fraction
            == records["spike"]["dataset"].validation_fraction,
        },
        "train_queries": _sets(set(f2.splits.get("train", [])), set(spike.splits.get("train", []))),
        "validation_queries": _sets(
            set(f2.splits.get("validation", [])), set(spike.splits.get("validation", []))
        ),
        "documents": _sets(set(f2.doc_ids), set(spike.doc_ids)),
        "train_judgments": {
            "shared": len(f2_pairs & spike_pairs),
            "only_f2": len(f2_pairs - spike_pairs),
            "only_spike": len(spike_pairs - f2_pairs),
        },
        "document_vectors": _cosines(f2.documents, f2_rows, spike.documents, spike_rows),
        "query_vectors": _cosines(
            f2.query_vectors, f2.query_rows, spike.query_vectors, spike.query_rows
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("f2_root", type=Path)
    parser.add_argument("spike_root", type=Path)
    parser.add_argument("--envs", nargs="+", default=["scifact", "nfcorpus", "msmarco"])
    args = parser.parse_args(argv)
    manifest = start_manifest(
        "p3-input-differences",
        {"f2_root": str(args.f2_root), "spike_root": str(args.spike_root), "envs": args.envs},
        seed=0,
        dataset_hash=NO_DATASET,
    )
    with ArtifactWriter(args.f2_root, "evaluation/P3-inputs/all", manifest) as writer:
        found = {env: compare(args.f2_root, args.spike_root, env) for env in args.envs}
        text = json.dumps(found, indent=2, sort_keys=True, allow_nan=False) + "\n"
        writer.path("differences.json").write_text(text, encoding="utf-8")
        print(writer.commit())
    return 0


if __name__ == "__main__":
    sys.exit(main())
