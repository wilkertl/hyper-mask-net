# Quickstart: validating F12 P3-bench

## 1. Offline checks (any machine, CPU)

```bash
uv run pytest -q tests/unit/data/test_query_split.py tests/unit/baselines/test_l2s_official.py \
  tests/unit/evaluation/test_benchmark.py tests/unit/evaluation/test_benchmark_stats.py \
  tests/unit/evaluation/test_teachers.py tests/unit/evaluation/test_p3_bench_arms.py
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest -q tests/unit/reports/test_p3_bench_report.py tests/integration/test_p3_bench.py  # ~2 min
```

The integration test runs every command on a tiny CPU corpus with the authors' clone, and is
skipped without it. It also checks:
- replay;
- leakage (tune, evaluation and test labels replaced → byte-identical reference, teachers and
  teacher arm);
- contrastive masks built from the recorded negatives;
- that no stage opens `qrels-test.jsonl`.

Expected: all pass. The GPU regression test for the streaming scorer is skipped without CUDA.
These tests cover:
- bit-for-bit parity with the authors' script on the fixture;
- k = 1024 equal to full scoring;
- the brute-force optimum of the search;
- statistics on hand cases;
- the resume blocks.

## 2. Run sequence for one corpus (host with F2 artifacts and the authors' clone)

Prerequisites:
- F2 `datasets/` and `embeddings/` under `--root`;
- `artifacts/p3-query-split-v1/manifest.json`;
- the authors' clone at `55f010f5`;
- ≥ 20 GB free GPU memory and no other compute process before each GPU step.

The commands follow [contracts/cli.md](contracts/cli.md). With `C=scifact` and `P="python -m
hyperdime.cli p3-bench"`:

```bash
$P split-provenance --corpus $C
$P freeze-protocol                                   # once, before any probe
$P official --corpus $C --purpose probe --labels-only --limit 256
$P official --corpus $C --purpose probe --limit <grid size>
$P probe --corpus $C --labels-run <run> --protocol-record reports/P3-bench/freeze/protocol.json --device cuda
$P freeze-resources --corpus $C --probe-run <run> --out reports/P3-bench/freeze/resources-$C.json
#   if it escalates: $P escalate --corpus $C --option <option> --reason "..." --out .../escalation-$C.json
$P official --corpus $C --purpose reference --resource-record <resources>
$P official-parity --corpus $C --official-run <run> --resource-record <resources>   # diagnostic
$P train-arm --corpus $C --official-run <run> --target official --seed 42          # replay control
$P teachers  --corpus $C --official-run <run> --resource-record <resources> --device cuda [--max-hours 11]
$P train-arm --corpus $C --official-run <run> --teachers-run <run> --target <teacher> --seed {42,0,1,2}
$P train-s1  --corpus $C --official-run <run> --arm {scratch,official-init} --device cuda
$P evaluation-lock --corpus $C --runs reports/P3-bench/freeze/runs-$C.json \
   --family reports/P3-bench/freeze/family.json --out reports/P3-bench/freeze/evaluation-lock-$C.json
$P evaluate --corpus $C --lock-record <lock> --device cuda [--max-hours 11]   # exit 75 → rerun
$P family --family reports/P3-bench/freeze/family.json --evaluations <{corpus: run}>  # after all corpora
$P report --corpus $C --evaluation-run <run> --lock-record <lock> --resource-record <resources>
$P teacher-report --corpus $C --teachers-run <run>
```

On a host where the code is an exported snapshot rather than a git checkout, export
`HYPERDIME_CODE_COMMIT=<40-hex snapshot commit>` first, so every manifest records it.

## 3. Expected outcomes

| Check | Where | Expected |
|---|---|---|
| Split provenance | `verification.json` | `passed: true` for every check |
| Replay control | arm `summary.json` | `replay_equals_official: true` |
| Recorded negatives | `official` run | it fails if the rebuilt negatives don't reproduce the targets |
| 100% parity | evaluation `summary.json` | `full_vector_parity_at_1024` holds for every arm |
| Lock | `evaluate` | refuses if any arm hash differs from the lock |
| Report | evaluation `README.md` | generated from `summary.json`; nothing edited by hand |

Current results and run ids: [reports/P3-bench/RUNS.md](../../reports/P3-bench/RUNS.md).
SciFact and NFCorpus are complete; MS MARCO and `family` are pending.
