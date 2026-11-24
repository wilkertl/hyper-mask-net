# Contract: `p3-bench` CLI

Entry point: `python -m hyperdime.cli p3-bench <command>`, registered in `cli/__main__.py`. Every
command that writes an artifact writes a run directory with a `manifest.json`
([data-model.md](../data-model.md)). Errors that break the contract (`ContractError`) exit with a
non-zero code. Resumable commands exit with **75** when `--max-hours` stops them at a block
boundary; rerun the same command to continue.

Options shared by most commands: `--root PATH` (default `artifacts`) and
`--corpus {msmarco,nfcorpus,scifact}`. The `teachers`, `train-*`, `evaluation-lock` and `evaluate`
commands also take `--device` (default `cpu`) and `--resource-record PATH`.

| Command | Required | Optional | Reads | Writes | Refuses when |
|---|---|---|---|---|---|
| `split-provenance` | `--corpus` | `--root` | split manifest, F2 sources, official test text | `p3-query-split-v1/provenance/<c>/` | a hash, count, group or test-overlap check fails |
| `freeze-protocol` | — | `--repo`, `--out` (default `reports/P3-bench/freeze/protocol.json`) | ADR-0018, spec, plan, research, tasks | protocol record | the output exists (freeze records are never overwritten) |
| `official` | `--corpus`, `--purpose {probe,reference}` | `--limit`, `--labels-only`, `--resource-record`, `--device`, `--clone` | fit ids and fit labels, embeddings, authors' clone | `selectors/l2s-official/<c>/` or `evaluation/P3-bench/probe-official/<c>/` | the clone isn't at `55f010f5`; rebuilt negatives don't reproduce the targets; GPU label generation on a large corpus |
| `probe` | `--corpus`, `--labels-run`, `--protocol-record` | `--repo`, `--device`, `--search-share` (0.7) | fit only | `evaluation/P3-bench/probe/<c>/` | any protocol file hash changed |
| `freeze-resources` | `--corpus`, `--probe-run`, `--out` | — | probe output | resource record (`realized`, or escalate) | the output exists |
| `escalate` | `--corpus`, `--option`, `--reason`, `--out` | `--resource-record` | resource record | escalation record | the output exists; this corpus's resource gate didn't escalate |
| `official-parity` | `--corpus`, `--official-run`, `--resource-record` | `--device`, `--clone` | fit labels, reference | `evaluation/P3-bench/official-parity/<c>/` (diagnostic) | — |
| `teachers` | `--corpus`, `--official-run` | `--escalation-record`, `--max-hours` | fit labels, recorded negatives | `evaluation/P3-bench/teachers/<c>/` | missing resource record; escalated without an escalation record, or with one for another resource record or an option that trains no teacher |
| `train-arm` | `--corpus`, `--official-run`, `--target {official,<teacher>}`, `--seed` | `--teachers-run`, `--clone` | targets or teacher masks, the RNG state | `selectors/p3-teacher-arms/<c>/` | teacher queries differ from the reference's (`official` writes `replay_equals_official`) |
| `train-s1` | `--corpus`, `--official-run`, `--arm {scratch,official-init}` | — | fit, recorded negatives, tune labels (selection only) | `selectors/p3-s1/<c>/<arm>/` | no cell passes the soft-mass guard |
| `evaluation-lock` | `--corpus`, `--runs`, `--out` | `--family` | arm files (hashes), evaluation ids | lock record | the output exists; family arms differ from the trained arms |
| `evaluate` | `--corpus`, `--lock-record` | `--max-hours` | locked arms, then evaluation labels | `evaluation/P3-bench/evaluation/<c>/` | any arm file changed since the lock; an arm at 1024 differs from the full vector |
| `report` | `--corpus`, `--evaluation-run`, `--lock-record`, `--resource-record` | `--root`, `--device` (selector timing) | the evaluation's `summary.json` and manifest, the runs named in the lock, the probe | `evaluation/P3-bench/report/<c>/` (`README.md`, `costs.json`, `provenance.json`); never reads judgments | the evaluation wasn't made under that lock record |
| `teacher-report` | `--corpus`, `--teachers-run` | `--root` | the teachers' `summary.json`, `searches.json` if present | `evaluation/P3-bench/teacher-report/<c>/` (`README.md`, `gains.json`) | — |
| `family` | `--family`, `--evaluations` | `--root` | every corpus's `per_query.csv` | `evaluation/P3-bench/family/` | unknown or duplicate member; a corpus without an evaluation run |

Order constraints (ADR-0018 freeze gates):

```text
freeze-protocol < probe < freeze-resources [< escalate] < teachers, train-*, tune metrics
< evaluation-lock < evaluate (once per corpus) < family
```

Output fields added on 2026-11-24 (T043–T048, T054):
- every manifest records the code commit, from `HYPERDIME_CODE_COMMIT` when the code isn't a git
  checkout;
- `official` on MS MARCO writes `transfer_flag`;
- `evaluate` adds the `mrl-prefix` control and stores `provenance` (split fingerprints, the
  authors' commit, the transfer flag) in `summary.json`;
- the `evaluate` and `family` READMEs carry the FR-013 scope sentence.

Cost tables come from `report`, not from `evaluate`.
