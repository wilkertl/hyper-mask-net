"""Convert local F0 spike caches into an F2-layout root, for an offline P3 pilot only.

    python scripts/checks/p3_spike_inputs.py artifacts/spike artifacts/p3-spike --envs scifact

Use this only when the published F2 artifacts are out of reach. It loads no model and downloads
nothing. It re-derives the training split with F2's own rule, from the BEIR ``train.tsv`` (seed 0,
10% validation), and writes only the training split's judgments: validation queries' rows are
dropped while reading, and ``test.tsv`` and the spike's ``splits.json``, which holds evaluation
labels, are never opened. Query text is withheld. The dataset's ``spec`` carries a ``provenance``
line that the P3 report prints.

A run on this root is a pilot, not a substitute for the published artifacts: the caches are the
same model revision, but F2 agreement is what ``f0_agreement.py`` checks, not this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from hyperdime.contracts.hashing import config_hash, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.schemas import (
    DatasetManifest,
    EmbeddingArtifact,
    PayloadRef,
    QrelsRecord,
    QueryRecord,
    dump_records,
)
from hyperdime.data.prepare import MANIFEST, QUERIES, SPLIT_SEED, qrels_file
from hyperdime.data.splits import split_ids

VALIDATION_FRACTION = 0.1
PROVENANCE = "converted from the F0 spike caches by scripts/checks/p3_spike_inputs.py (pilot)"


def convert(spike_root: Path, output: Path, env: str) -> None:
    source = spike_root / env
    settings = json.loads((source / "corpus" / "settings.json").read_text(encoding="utf-8"))
    revision = settings["model"].rpartition("@")[2]
    document_ids = json.loads((source / "corpus" / "ids.json").read_text(encoding="utf-8"))
    query_ids = json.loads((source / "queries" / "ids.json").read_text(encoding="utf-8"))
    known_documents, known_queries = set(document_ids), set(query_ids)

    # F2's cleaning order: drop judgments of missing documents, keep queries with a positive.
    judged: dict[str, dict[str, int]] = {}
    path = spike_root / "beir" / env / "qrels" / "train.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for query, document, grade in reader:
            if document in known_documents:
                judged.setdefault(query, {})[document] = int(grade)
    eligible = [
        q for q, labels in judged.items() if q in known_queries and max(labels.values()) > 0
    ]
    train, validation = split_ids(eligible, VALIDATION_FRACTION, SPLIT_SEED)

    instruction = json.loads((source / "manifest.json").read_text(encoding="utf-8"))["config"][
        "instruction"
    ]
    spec = {"environment": env, "provenance": PROVENANCE, "eval_query_sets": []}
    with ArtifactWriter(
        output,
        f"datasets/{env}",
        start_manifest("p3-spike-inputs", {"environment": env}, seed=SPLIT_SEED),
    ) as writer:
        records = [
            QueryRecord(
                query_id=q, text="<withheld>", instruction=instruction, environment=env, split=split
            )  # fmt: skip
            for split, ids in (("train", train), ("validation", validation))
            for q in ids
        ]
        dump_records(records, writer.path(QUERIES))
        qrels = [
            QrelsRecord(query_id=q, doc_id=d, relevance=g, environment=env, split="train")
            for q in train
            for d, g in sorted(judged[q].items())
        ]
        dump_records(qrels, writer.path(qrels_file("train")))
        files = {name: file_sha256(writer.path(name)) for name in (QUERIES, qrels_file("train"))}
        fingerprint = config_hash(files)
        writer.write_record(
            MANIFEST,
            DatasetManifest(
                environment=env,
                spec=spec,
                num_documents=len(document_ids),
                dropped_documents=0,
                num_queries={"train": len(train), "validation": len(validation)},
                num_qrels={"train": len(qrels)},
                dropped_qrels=0,
                excluded_queries={},
                split_seed=SPLIT_SEED,
                validation_fraction=VALIDATION_FRACTION,
                files=files,
                fingerprint=fingerprint,
            ),
        )
        writer.commit()
    with ArtifactWriter(
        output,
        f"embeddings/{env}",
        start_manifest(
            "p3-spike-inputs",
            {"environment": env},
            seed=SPLIT_SEED,
            dataset_hash=fingerprint,
            model_revision=revision,
            dtype="float16",
        ),
    ) as writer:
        for name, directory, ids, role, text in (
            ("documents", "corpus", document_ids, "documents", None),
            ("queries", "queries", query_ids, "queries", instruction),
        ):
            vectors = np.load(source / directory / "embeddings.npy", mmap_mode="r")
            if vectors.dtype != np.float16:
                raise SystemExit(f"{source / directory}: expected float16, found {vectors.dtype}")
            target = writer.path(f"{name}.npy")
            np.save(target, vectors)
            writer.write_record(
                f"{name}.json",
                EmbeddingArtifact(
                    role=role,
                    environment=env,
                    ids=ids,
                    num_rows=len(ids),
                    dtype="float16",
                    normalized=True,
                    model_revision=revision,
                    instruction=text,
                    payload=PayloadRef(
                        path=f"{name}.npy", sha256=file_sha256(target), format="npy"
                    ),
                ),
            )
        writer.commit()
    print(f"{env}: {len(train)} train queries, {len(qrels)} judgments", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spike_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--envs", nargs="+", default=["scifact", "nfcorpus"])
    args = parser.parse_args()
    for env in args.envs:
        convert(args.spike_root, args.output, env)


if __name__ == "__main__":
    main()
