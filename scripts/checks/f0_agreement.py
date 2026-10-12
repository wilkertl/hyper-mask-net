"""SC-004: F2 vectors agree with the F0 spike's vectors for the same texts and model revision.

    python scripts/checks/f0_agreement.py \\
        artifacts/spike/scifact artifacts/embeddings/scifact/<run_id>

Compares documents and queries over the ids both contain; exits 1 if any cosine is below the
threshold (default 0.999). Reads only cached vectors; loads no model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hyperdime.embeddings.cache import load_cached, load_embeddings


def agreement(spike_dir: Path, name: str, f2_dir: Path, f2_name: str) -> tuple[int, float, float]:
    """``(shared rows, min cosine, mean cosine)`` between one spike cache and one F2 matrix."""
    spike_ids, spike = load_cached(spike_dir / name)
    record, vectors = load_embeddings(f2_dir, f2_name)
    rows = {doc_id: i for i, doc_id in enumerate(record.ids)}
    shared = [(i, rows[doc_id]) for i, doc_id in enumerate(spike_ids) if doc_id in rows]
    if not shared:
        raise SystemExit(f"no shared ids between {spike_dir / name} and {f2_dir / f2_name}")
    left = spike.numpy()[[i for i, _ in shared]].astype(np.float64)
    right = np.asarray(vectors[[j for _, j in shared]], dtype=np.float64)
    cosine = (left * right).sum(1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))
    return len(shared), float(cosine.min()), float(cosine.mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spike_env_dir", type=Path)
    parser.add_argument("f2_embeddings_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.999)
    args = parser.parse_args(argv)
    ok = True
    for spike_name, f2_name in (("corpus", "documents"), ("queries", "queries")):
        count, low, mean = agreement(
            args.spike_env_dir, spike_name, args.f2_embeddings_dir, f2_name
        )
        passed = low >= args.threshold
        ok &= passed
        print(
            f"{f2_name}: {count} shared rows, min cosine {low:.6f}, mean {mean:.6f}"
            f" -> {'ok' if passed else 'BELOW ' + str(args.threshold)}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
