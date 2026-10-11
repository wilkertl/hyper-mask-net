# ADR-0003: H1 spike verdict and what F3 must change

- **Status:** proposed (the researcher accepts or amends the decision below before merge)
- **Date:** 2026-10-11
- **Issue:** F0 spike, tasks T025–T026 in `specs/002-hyperdime-roadmap/tasks.md`

## Context

The F0 spike tests H1 (the best dimension selector differs across environments) on SciFact,
NFCorpus, FiQA-2018, ArguAna, and SciDocs with the frozen Qwen3-Embedding-0.6B (research R0). Its
first run returned "not supported" because in-domain selectors did not beat random dimensions;
ADR-0002 then aligned training with the official Learning-to-Select recipe, and the rerun returned
"supported", with criteria C1 and C3 passing exactly at their thresholds. Because the recipe
changed after a negative result, research R20 fixed diagnostic rules (commit `a72a3c9`) before
any diagnostic ran, to separate weak selector shift from a selector that shows little of it.

## Decision

Proposed, following R20's table (O1 = yes, T = yes, recommendation
`continue_strengthen_selector`):

1. **H1 is supported for continuing the roadmap**, as exploratory evidence. Start F1 (task T028).
2. **F3 adds sample-efficient per-environment selector candidates**, chosen on validation only, and
   F4 retests H1 with the selected one. The D2 evidence points to data limits and overfitting, not
   to missing capacity or seed variance (see Evidence), so the candidates are, in this order:
   regularization toward the global selector (the `W_0, b_0` base of research R12/R13), shorter or
   validation-stopped training, and a wider temperature-by-epoch grid. A higher-capacity selector
   is not a candidate unless an ablation shows the linear one underfits (constitution IV).
3. **The H2 signal from D3 (S = yes) is recorded as early support only.** G3 still decides H2.

## Evidence

All values are in the generated report `reports/F0/README.md` (diagnostics manifest
`artifacts/spike/diagnostics/manifest.json`: complete, commit `7fdf1e5`, clean tree), from
`artifacts/spike/analysis/analysis.json` and `artifacts/spike/diagnostics/diagnostics.json`.

- **R0 verdict** (report, "Verdict" and "Headroom"): supported; C1 and C3 at their thresholds;
  per-domain headroom over the global selector is significant on SciFact and NFCorpus only.
- **D1, shift in the oracle targets** (report, "D1"): every environment pair differs at the
  smallest p-value the permutation test can produce, and cross-environment profile similarity is
  below every environment's split-half reliability. Selector shift exists in the targets
  themselves, independently of any trainer.
- **D2, trainer limits** (report, "D2"):
  - the learning curve still rises from 50% to 100% of the training queries on SciFact, NFCorpus,
    and SciDocs (rule T);
  - the seed ensemble never beats single seeds, so seed variance is not the limit;
  - nDCG@10 and oracle-mask agreement are clearly higher on training than on evaluation queries;
  - on FiQA the grid-edge check favors fewer epochs than the grid allowed;
  - recovery of each environment's oracle profile is high: selectors learn the
    environment-level profile and lose on per-query selection.
- **D3, first H2 probe** (report, "D3"): corpus-only group-A statistics predict the held-out
  environment's oracle profile better than the other environments' mean in all five folds, at the
  smallest attainable exact p over 120 assignments.

## Consequences

- F1 starts. F3's spec must include the selector candidates above, and F4's H1 test must use the
  selected one; the F0 numbers stay exploratory and are not reported as results.
- The hypernetwork's case now rests on two facts to recheck at G2: the shift is real at the
  target level, but its retrieval-level payoff over the global selector was small in F0 (two of
  five environments).
- Correction recorded with this ADR: R20's note that split-half reliability is a "ceiling" for D2
  recovery was wrong (it compares half-sample profiles); the report wording is fixed and no rule
  used that value.
- Alternatives if this proposal is amended: `continue` without changing F3 (ignores rule T), or
  stop / pivot to a universal selector (contradicts D1).
