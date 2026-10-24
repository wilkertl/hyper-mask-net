# Implementation Plan: F10 Descriptor Probe (experiment E2, gate G3)

**Branch**: `agent/F10-descriptor-probe` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)

## Summary

Build a per-dimension space descriptor `R_T` (`1024 × h`) for each environment without labels:
document marginals and tails, and redundancy (P0); query marginals, query–document products, and
pseudo-relevance contrast from unlabeled training queries (P1) (research D2–D5). Derive per-
dimension targets from existing selectors on validation queries without their labels: the S1
selector's selection frequency averaged over budgets (primary), per-budget, S1-bias, and
S4-correction variants (secondary) (D6). A shift check runs first, from targets alone: if the
shared profile already reaches the seed noise ceiling, there is nothing to explain (D0). The
decisive shift probe starts from the shared profile and learns, leave-one-environment-out, how
descriptor differences predict target differences; a direct probe from descriptors alone is
secondary. Both run over a six-step descriptor ladder against two baselines (dimension identity,
`variance`), with a refitting permutation test and Holm (D1, D7–D10). Delivered in two stages:
P0 first (sets 1–4, a complete G3 reading under P0), then P1 (sets 5–6) (D13). Experiment
E2 ends with `reports/E2/` and a drafted ADR-0013 reading G3 as exploratory (D14). Stacked on F9;
runs on DataPolicy-4090.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.12) · **Dependencies**: `space.marginal`,
`space.geometry` (streamed Gram), `retrieval.exact.exact_search_streamed`,
`evaluation.selector_shift.selection_frequency`, `baselines.methods.variance_importance`, the
S1/S4 seeded-selector loading of `cli/compare_selectors.py`, `contracts` (`SpaceDescriptor`,
`GateDecision` with `"G3"`, `ArtifactWriter`); torch, numpy; nothing new · **Storage**:
`artifacts/{descriptors/<env>,evaluation/E2/all,decisions/G3}/`, `reports/E2/` (versioned) ·
**Testing**: pytest on CPU; hand-computed tests for every statistic (moments, tails, correlations,
PC share, products, contrast), the ridge and its inner penalty choice, ranks with ties, the
permutation p, and fold isolation; qrels-free loading tested by deleting and corrupting qrels ·
**Platform**: DataPolicy-4090 (shared GPU) · **Performance**: three descriptors ≤ 3 h, probe
≤ 1 h (SC-003; research D13) · **Constraints**: no qrels read by `describe-space`; validation
judgments not read by `probe-descriptors`; normalization from training environments only; E0,
E1, P1, S1–S4 outputs byte-identical (FR-008).

No open clarifications: sample sizes, K, r, the ridge grid, permutations, and the criterion are
fixed in research D3–D9.

## Constitution Check

| Principle | Status | Evidence |
|---|---|---|
| I | ✅ | embeddings only read; no selection is changed; the targets read existing checkpoints |
| II | ✅ | `describe-space` uses a loader that never opens qrels (D2, tested); calibration queries are training-split queries without labels, targets use validation queries without judgments (D5, D6); normalization and penalty use the fold's training environments only (D7); P0 descriptors read no queries (`SpaceDescriptor` validator) |
| III | ✅ | `SpaceDescriptor` (schema 0, provisional) with payloads and manifests; `GateDecision` already has `"G3"`; no contract change; new public names in [contracts/cli.md](contracts/cli.md) |
| IV | ✅ | this is G3's probe (M5), before any space encoder or Hyper Head; lowest-capacity probe (ridge); no G4 code |
| V | ✅ | descriptors use exact retrieval; generated report and decision; protocol per row; permutation p with Holm; test-set reuse stated (D14). Full and prefix rows do not apply: E2 scores no retrieval, and the report says so |
| VI | ✅ | hand-computed tests for every descriptor statistic and the probe; descriptor invariance to document and query order, equivariance to dimension permutation |
| VII | ✅ | descriptors in `hyperdime.space`, probe in `evaluation.descriptor_probe`, commands thin; the selector is not touched |

**Result**: PASS (re-checked after Phase 1 design: unchanged). Dependency: F9 merged before this
branch, or the PR stacks on it.

## Project Structure

### Documentation (this feature)

```text
specs/011-descriptor-probe/
├── spec.md · plan.md · research.md · data-model.md · quickstart.md
├── contracts/cli.md
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks
```

### Source code

```text
src/hyperdime/space/inputs.py              # new: load_unlabeled, qrels-free (D2)
src/hyperdime/space/streaming.py           # new: streamed moments, tails, Gram (D3, D4)
src/hyperdime/space/redundancy.py          # new: correlation and PC-share features (D4)
src/hyperdime/space/interaction.py         # new: groups B, C, D from calibration queries (D5)
src/hyperdime/space/descriptor.py          # new: builds P0/P1 SpaceDescriptors, feature names
src/hyperdime/evaluation/descriptor_probe.py  # new: ranks, ridge, folds, baselines, shift check, probes, permutation (D0, D1, D7–D10)
src/hyperdime/evaluation/targets.py        # new: selector target profiles, noise ceiling (D6)
src/hyperdime/cli/describe_space.py        # new command
src/hyperdime/cli/probe_descriptors.py     # new command
src/hyperdime/cli/__main__.py              # register both
src/hyperdime/cli/report.py, cli/decide.py # + E2, + G3
src/hyperdime/reports/e2.py                # new: E2 report, G3 criteria, ADR-0013 draft
tests/unit/test_space_*.py, tests/unit/evaluation/, tests/unit/reports/
tests/integration/test_f10_pipeline.py     # synthetic environments end to end
docs/architecture.md, docs/experiment_registry.md, README.md
```

**Structure Decision**: single package `hyperdime`; descriptor code in `hyperdime.space` (plan
M5), probe and targets in `hyperdime.evaluation`; existing modules only gain registrations.

## Risks

- **R2, identity**: dimensions keep their identity across environments, so descriptors can
  predict targets by encoding which dimension it is; the shift probe starts from B1 and can only
  move away from it through descriptor differences (D10).
- **No shift**: S1's profiles may be mostly shared; the shift check (D0) reads that first, from
  existing checkpoints, before any descriptor code runs.
- **Two training environments per fold**: the shift probe learns from one contrast (training
  environment 1 against 2), so it generalizes only if descriptor–target co-variation is shared
  across pairs; that is H2's claim, and three folds test it three times.
- **Three folds**: SC-001 counts 2 of 3; one odd environment (MS MARCO's scale) decides a lot.
  The report shows each fold and the noise ceiling, so a low ceiling explains a low score.
- **Pooled normalization**: corpora differ in scale (MS MARCO vs SciFact); pooled training
  statistics may shift the held-out environment's features. Spearman is invariant to a constant
  per-feature shift of the prediction but not to rescaling; the report includes the fold's
  feature means for context.
- **Circularity in D**: pseudo-relevance uses the full query, which the selector approximates;
  D is P1, labeled, and set 6 is the last rung, so P0 sets are read first (SC-002).
- **Re-used environments**: exploratory until #60 (D14).
- **Shared GPU**: SC-003 is read from the run's own timer.

## Complexity Tracking

None.
