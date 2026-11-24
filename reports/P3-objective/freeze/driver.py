"""ADR-0020 driver on the second host: grid, select, seeds, final score, report.

Resumable: every finished training is recorded in ledger-<corpus>.json and skipped on rerun.
Refuses to start if the claim directory exists (no duplicate launches) or the GPU is in use.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

D = Path("/home/aluno_iago/hyper-mask-net-p3-rank-20261109")
O = D / "objective"
PY = str(D / "venv" / "bin" / "python")
ENV = {**os.environ, "PYTHONPATH": str(D / "code-objective" / "src"),
       "HYPERDIME_L2S_CLONE": str(D / "learning2select")}
ARMS = ("KG", "KS", "MS", "MG", "KBk", "KBa", "R0", "RI", "KGI")
LRS = ("3e-05", "0.0001", "0.0003", "0.001")
CORPORA = ("scifact", "nfcorpus")
DEVICE = "cuda:0"


def log(msg: str) -> None:
    with (O / "driver.log").open("a") as h:
        h.write(msg + "\n")


def cli(*args: str) -> dict:
    out = subprocess.run([PY, "-m", "hyperdime.cli", *args], env=ENV, cwd=D, check=True,
                         capture_output=True, text=True)
    with (O / "cli.err").open("a") as h:
        h.write(out.stderr)
    line = [x for x in out.stdout.splitlines() if x.startswith("{")][-1]
    return json.loads(line)


def gpu_busy() -> bool:
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    return bool(apps)


def ledger(corpus: str) -> tuple[Path, dict]:
    path = O / f"ledger-{corpus}.json"
    return path, (json.loads(path.read_text()) if path.exists() else {})


def ensure(corpus: str, arm: str, lr: str, seed: int) -> str:
    path, book = ledger(corpus)
    key = f"{arm}|{float(lr)}|{seed}"
    if key in book:
        return book[key]
    result = cli("p3-objective", "train", "--root", "artifacts", "--corpus", corpus,
                 "--runs", str(O / f"runs-{corpus}.json"), "--arm", arm, "--lr", lr,
                 "--seed", str(seed), "--device", DEVICE)
    book[key] = result["run_id"]
    path.write_text(json.dumps(book, indent=1, sort_keys=True) + "\n")
    log(json.dumps({"corpus": corpus, "key": key, **result}))
    return result["run_id"]


def main() -> int:
    claim = O / "claim"
    claim.mkdir()  # raises if a driver already holds it
    (claim / "owner").write_text(f"pid {os.getpid()}\n")
    if gpu_busy():
        log("GPU in use at start; stopping without launching")
        return 3
    report_runs: dict = {}
    for corpus in CORPORA:
        grid = {f"{a}|{float(lr)}|42": ensure(corpus, a, lr, 42) for a in ARMS for lr in LRS}
        (O / f"grid-{corpus}.json").write_text(json.dumps(grid, indent=1) + "\n")
        sel_path = O / f"selection-{corpus}.json"
        if not sel_path.exists():
            score = cli("p3-objective", "score", "--root", "artifacts", "--corpus", corpus,
                        "--runs", str(O / f"runs-{corpus}.json"),
                        "--checkpoints", str(O / f"grid-{corpus}.json"), "--device", DEVICE)
            log(json.dumps({"corpus": corpus, "grid_score": score}))
            chosen = cli("p3-objective", "select", "--root", "artifacts", "--corpus", corpus,
                         "--score-run", score["run_id"], "--out", str(sel_path))
            log(json.dumps({"corpus": corpus, "selected": chosen}))
        selection = json.loads(sel_path.read_text())
        final: dict = {}
        for arm in ARMS:
            lr = repr(selection["arms"][arm]["selected_lr"])
            for seed in (42, 0, 1):
                final[f"{arm}|{float(lr)}|{seed}"] = ensure(corpus, arm, lr, seed)
        final["REF|0.0001|42"] = "l2s-official"
        for seed in (0, 1):
            final[f"REF|0.0001|{seed}"] = ensure(corpus, "KG", "0.0001", seed)
        (O / f"final-{corpus}.json").write_text(json.dumps(final, indent=1) + "\n")
        score = cli("p3-objective", "score", "--root", "artifacts", "--corpus", corpus,
                    "--runs", str(O / f"runs-{corpus}.json"),
                    "--checkpoints", str(O / f"final-{corpus}.json"), "--device", DEVICE,
                    "--fit-ndcg")
        log(json.dumps({"corpus": corpus, "final_score": score}))
        _, book = ledger(corpus)
        (O / f"trainings-{corpus}.json").write_text(json.dumps(book, indent=1) + "\n")
        report_runs[corpus] = {"score": score["run_id"], "selection": sel_path.name,
                               "trainings": f"trainings-{corpus}.json"}
    (O / "report-runs.json").write_text(json.dumps(report_runs, indent=1) + "\n")
    (O / "report").mkdir(exist_ok=True)
    cli("p3-objective", "report", "--root", "artifacts", "--runs", str(O / "report-runs.json"),
        "--out", str(O / "report"))
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
