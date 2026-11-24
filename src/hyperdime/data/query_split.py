"""The F12 query split `p3-query-split-v1` (ADR-0018, research B1).

The split manifest is read as it is; ids are never rematerialized. Loading checks the file hash,
the seed, and that the three splits are sorted and disjoint. ``verify_against_source`` repeats the
researcher's checks against the F2 queries file: source hash, counts, no normalized-text group
across splits, and no kept query equal to an official test query's normalized text. Test judgments
are never opened here; only test query text is compared.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.validation import ContractError

SPLIT_DIR = "p3-query-split-v1"
SPLIT_MANIFEST = f"{SPLIT_DIR}/manifest.json"
MANIFEST_SHA256 = "e4b1fa65c7f04a480d8cd9845e63e055ac6777c4c0ff95f1227e8f6551ff4896"
SEED = 20261112
SPLITS = ("fit", "tune", "evaluation")
ORDER_PREFIX = "p3-bench-v1"


def normalize(text: str) -> str:
    """NFKC, then casefold, then collapse whitespace (the split's grouping key)."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def id_fingerprint(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def hash_order(corpus: str, split: str, ids: Iterable[str], seed: int = SEED) -> list[str]:
    """ADR-0018 query order: ascending SHA-256 of ``p3-bench-v1|<seed>|<corpus>|<split>|<id>``.

    Any bound n takes the first n ids, so smaller bounds are prefixes of larger ones."""

    def key(query_id: str) -> str:
        return hashlib.sha256(
            f"{ORDER_PREFIX}|{seed}|{corpus}|{split}|{query_id}".encode()
        ).hexdigest()

    return sorted(ids, key=key)


@dataclass(frozen=True)
class QuerySplit:
    corpus: str
    ids: Mapping[str, tuple[str, ...]]
    source_dataset_run: str
    source_queries_sha256: str
    excluded_test_overlap_ids: tuple[str, ...]
    manifest_sha256: str

    def fingerprints(self) -> dict[str, str]:
        return {split: id_fingerprint(self.ids[split]) for split in SPLITS}


def load_query_split(
    root: Path, corpus: str, *, expected_sha256: str = MANIFEST_SHA256
) -> QuerySplit:
    path = root / SPLIT_MANIFEST
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ContractError(f"{path}: SHA-256 {digest} differs from the frozen {expected_sha256}")
    manifest = json.loads(path.read_text())
    if manifest.get("seed") != SEED:
        raise ContractError(f"{path}: seed {manifest.get('seed')} is not {SEED}")
    if corpus not in manifest["corpora"]:
        raise ContractError(f"{path}: no corpus {corpus!r}")
    entry = manifest["corpora"][corpus]
    ids = {split: tuple(entry["query_ids"][split]) for split in SPLITS}
    seen: set[str] = set()
    for split in SPLITS:
        if list(ids[split]) != sorted(ids[split]):
            raise ContractError(f"{corpus}/{split}: ids are not sorted")
        if seen & set(ids[split]):
            raise ContractError(f"{corpus}/{split}: ids overlap another split")
        seen |= set(ids[split])
        if entry["counts"][split] != len(ids[split]):
            raise ContractError(f"{corpus}/{split}: count differs from the manifest's counts")
    return QuerySplit(
        corpus=corpus,
        ids=ids,
        source_dataset_run=entry["source_dataset_run"],
        source_queries_sha256=entry["source_queries_sha256"],
        excluded_test_overlap_ids=tuple(entry["excluded_test_overlap_ids"]),
        manifest_sha256=digest,
    )


def verify_against_source(
    split: QuerySplit, queries: Sequence[Mapping[str, Any]], queries_sha256: str
) -> dict[str, Any]:
    """Checks against F2 query records (``query_id``, ``text``, ``split``); returns no text."""
    by_split: dict[str, set[str]] = defaultdict(set)
    text = {}
    for record in queries:
        by_split[record["split"]].add(record["query_id"])
        text[record["query_id"]] = normalize(record["text"])
    development = by_split["train"] | by_split["validation"]
    test = by_split["test"]
    kept = {qid: name for name in SPLITS for qid in split.ids[name]}
    groups: dict[str, set[str]] = defaultdict(set)
    for qid, name in kept.items():
        groups[text[qid]].add(name)
    test_text = {text[qid] for qid in test}
    dropped = development - set(kept)
    checks = {
        "source_queries_sha256": queries_sha256 == split.source_queries_sha256,
        "kept_ids_are_development": set(kept) <= development,
        "no_test_id_kept": not (set(kept) & test),
        "dropped_equals_excluded": dropped == set(split.excluded_test_overlap_ids),
        "no_group_crosses_splits": all(len(names) == 1 for names in groups.values()),
        "no_kept_text_matches_test": not any(text[qid] in test_text for qid in kept),
        "dropped_match_test": all(
            qid in test or text[qid] in test_text for qid in split.excluded_test_overlap_ids
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "counts": {name: len(split.ids[name]) for name in SPLITS},
        "development_pool": len(development),
        "dropped": len(dropped),
        "groups": len(groups),
        "fingerprints": split.fingerprints(),
    }
