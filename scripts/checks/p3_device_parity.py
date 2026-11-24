"""Compare two P3 runs made on different devices or batch sizes, array by array.

    python scripts/checks/p3_device_parity.py artifacts <cpu_run_id> <gpu_run_id>

Writes ``<root>/evaluation/P3-parity/all/<run_id>/parity.json`` with a run manifest. Orderings and
reference draws must be identical; every other array must agree within ``--tolerance``. A difference
fails the check. Names and numbers only: nothing here reads a query, a document, or a judgment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.contracts.hashing import NO_DATASET
from hyperdime.contracts.manifests import ArtifactWriter, read_manifest, start_manifest

EXACT = ("pair-order.npy", "reference-documents.npz", "matched-rows.npz", "tracked.npz")


def _arrays(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npy":
        return {"": np.load(path, mmap_mode="r")}
    with np.load(path) as found:
        return {name: found[name] for name in found.files}


def _gap(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_differs": True, "max_abs": None, "nan_differs": None}
    if left.dtype.kind in "iub":
        same = np.array_equal(left, right)
        return {"max_abs": 0.0 if same else float(np.abs(left.astype(float) - right).max())}
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    undefined = np.isnan(a) != np.isnan(b)
    both = np.isfinite(a) & np.isfinite(b)
    return {
        "max_abs": float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0,
        "nan_differs": int(undefined.sum()),
    }


def compare(left: Path, right: Path, tolerance: float) -> dict[str, Any]:
    """Every array file both environments' runs share, with the worst gap of each."""
    rows: dict[str, Any] = {}
    failed: list[str] = []
    for path in sorted(left.glob("*/*.np[yz]")):
        name = f"{path.parent.name}/{path.name}"
        other = _arrays(right / path.parent.name / path.name)
        for key, array in _arrays(path).items():
            gap = _gap(array, other[key])
            label = f"{name}:{key}" if key else name
            rows[label] = gap
            limit = 0.0 if path.name in EXACT else tolerance
            bad = gap.get("shape_differs") or gap.get("nan_differs") or gap["max_abs"] > limit
            if bad:
                failed.append(label)
    return {"arrays": rows, "failed": failed, "tolerance": tolerance}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--experiment", default="P3")
    args = parser.parse_args(argv)
    directory = args.root / "evaluation" / args.experiment / "all"
    left, right = directory / args.left, directory / args.right
    manifest = start_manifest(
        "p3-device-parity",
        {"left": args.left, "right": args.right, "tolerance": args.tolerance},
        seed=0,
        dataset_hash=NO_DATASET,
    )
    found = compare(left, right, args.tolerance)
    found["devices"] = {
        run: read_manifest(directory / run).device for run in (args.left, args.right)
    }
    worst = max((v["max_abs"] or 0.0 for v in found["arrays"].values()), default=0.0)
    found["worst_max_abs"] = worst
    with ArtifactWriter(args.root, "evaluation/P3-parity/all", manifest) as writer:
        text = json.dumps(found, indent=2, sort_keys=True, allow_nan=False) + "\n"
        writer.path("parity.json").write_text(text, encoding="utf-8")
        print(writer.commit())
    print(json.dumps({k: found[k] for k in ("devices", "worst_max_abs", "failed")}))
    return 1 if found["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
