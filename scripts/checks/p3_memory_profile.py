"""Split one P3 run's peak RSS into file-backed and anonymous memory, stage by stage.

    python scripts/checks/p3_memory_profile.py <root> output.json run.device=cuda --envs msmarco

``<root>`` receives a complete run: use one holding only links to ``datasets`` and ``embeddings``,
so that it cannot become the latest canonical run. The rest of the line goes to
``analyze-representations``. Linux only; a diagnostic, and nothing reads its output but people.
"""

from __future__ import annotations

import functools
import json
import resource
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hyperdime.cli.analyze_representations as command
import hyperdime.evaluation.representation_controls as controls
from hyperdime.cli.__main__ import main as cli

STAGES = (
    (command, ("load_unlabeled", "verify_inputs", "load_training_judgments", "analyze_environment",
               "analyze_controls", "_write_json")),
    (controls, ("coordinate_scale", "_paired", "_half_range", "_across", "_frequency")),
)  # fmt: skip
FIELDS = {"VmRSS": "rss", "RssAnon": "anonymous", "RssFile": "file"}


def memory_mib() -> dict[str, int]:
    found = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key in FIELDS:
            found[FIELDS[key]] = int(value.split()[0]) // 1024
    return found


def main(argv: list[str]) -> int:
    root, output, *rest = argv
    current = ["startup"]
    peaks: dict[str, dict[str, int]] = {}
    stages: dict[str, dict[str, float]] = {}

    def sample() -> None:
        while True:
            peak = peaks.setdefault(current[0], {})
            for name, value in memory_mib().items():
                peak[name] = max(peak.get(name, 0), value)
            time.sleep(0.1)

    def timed(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def run(*args: Any, **kwargs: Any) -> Any:
            previous, current[0] = current[0], name
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                row = stages.setdefault(name, {"calls": 0, "seconds": 0.0})
                row["calls"] += 1
                row["seconds"] += time.perf_counter() - started
                current[0] = previous

        return run

    for module, names in STAGES:
        for name in names:
            setattr(module, name, timed(name, getattr(module, name)))
    threading.Thread(target=sample, daemon=True).start()
    started = time.perf_counter()
    code = cli(["analyze-representations", "--output", root, *rest])
    found = {
        "exit": code,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        "stages": stages,
        "peak_mib_by_stage": peaks,
    }
    Path(output).write_text(json.dumps(found, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
