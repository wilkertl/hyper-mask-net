"""Map every P3-bench run to the code snapshot it most likely ran on (F12 T043, Constitution III).

P3-bench runs used code snapshots exported from git bundles that are not on any branch, so their
manifests say ``git_commit: "unknown"``. This script rebuilds the mapping from files only. For each
run manifest under ``--runs`` it finds the bundled snapshot committed most recently before the run
started.

The match is an inference, and the output says so. A run that started after the newest bundle may
have used a later snapshot whose bundle is not in ``--bundles``. RUNS.md names two such snapshots
for MS MARCO, ``5347538`` and ``42bf508``. Those runs are marked ``unverified``. Runs whose manifest
already records a commit keep it, marked ``recorded``.

Usage:
    python scripts/checks/p3_bench_snapshot_map.py \
        --runs reports/P3-bench/runs --bundles artifacts/p3-bench/code-snapshots \
        --out reports/P3-bench/snapshots.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def bundle_commits(bundles: Path) -> list[tuple[datetime, str]]:
    """(commit time, commit id) of every bundle's head, oldest first."""
    found = []
    with tempfile.TemporaryDirectory() as scratch:
        subprocess.run(["git", "init", "-q", scratch], check=True)
        for bundle in sorted(bundles.glob("*.bundle")):
            heads = subprocess.run(
                ["git", "bundle", "list-heads", str(bundle)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
            commit = heads[0]
            subprocess.run(
                ["git", "-C", scratch, "fetch", "-q", str(bundle.resolve()), commit],
                check=True,
                capture_output=True,
            )
            when = subprocess.run(
                ["git", "-C", scratch, "log", "-1", "--format=%cI", commit],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            found.append((datetime.fromisoformat(when), commit))
    return sorted(found)


def snapshot_map(manifests: list[dict[str, Any]], snapshots: list[tuple[datetime, str]]) -> dict:
    newest = snapshots[-1][0] if snapshots else None
    runs = {}
    for manifest in sorted(manifests, key=lambda m: m["run_id"]):
        started = datetime.fromisoformat(manifest["started_at"])
        entry: dict[str, Any] = {
            "command": manifest["command"],
            "started_at": manifest["started_at"],
        }
        if manifest.get("git_commit", "unknown") != "unknown":
            entry.update(snapshot=manifest["git_commit"], method="recorded")
        else:
            before = [commit for when, commit in snapshots if when <= started]
            entry["snapshot"] = before[-1] if before else None
            entry["method"] = "inferred: newest bundled snapshot committed before the run started"
            if newest is not None and started > newest:
                entry["unverified"] = (
                    "started after the newest bundle; a later snapshot without a local bundle may "
                    "have been used"
                )
            if not before:
                entry["unverified"] = "no bundled snapshot predates this run"
        runs[manifest["run_id"]] = entry
    return {
        "bundles": [{"commit": c, "committed_at": w.isoformat()} for w, c in snapshots],
        "runs": runs,
        "unverified": sorted(r for r, e in runs.items() if "unverified" in e),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("reports/P3-bench/runs"))
    parser.add_argument("--bundles", type=Path, default=Path("artifacts/p3-bench/code-snapshots"))
    parser.add_argument("--out", type=Path, default=Path("reports/P3-bench/snapshots.json"))
    args = parser.parse_args(argv)
    manifests = [json.loads(p.read_text()) for p in sorted(args.runs.rglob("manifest.json"))]
    manifests = [m for m in manifests if "run_id" in m and "started_at" in m]
    result = snapshot_map(manifests, bundle_commits(args.bundles))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"out": str(args.out), "runs": len(result["runs"]), "unverified": result["unverified"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
