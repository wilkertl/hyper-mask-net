# ADR-0010: Experiment S2 — a retrieval selector started from the geometric prior

- **Status:** proposed
- **Date:** 2026-10-15
- **Issue:** F7 (`specs/008-geometric-prior`), experiment S2

## Context

P1 found that the label-free `variance` method approximates the full query best
(MS MARCO) while learned selectors reshape the ranking toward relevance (SciFact). S2
asks whether S1's retrieval selector, started from that geometric prior, gets both. This
draft is generated
by `hyperdime decide S2` from the result files of `reports/S2/README.md`.

## Evidence

From compare-selectors `20261015T161858Z-d0c633dc` (commit `13d6580d0320`); every value
is in `reports/S2/README.md`.

- **SC-001 better than the S1 selector (mean nDCG@10, 10%-90%, Holm, alpha 0.05)**: 2 of 3 environments; threshold >= 2 of 3; met.
- **SC-002 worse than the variance method in no environment (same test)**: 0 of 3 environments; threshold 0; met.

Context, not decisive:

- **against_l2s**: msmarco +0.0154 (Holm p 0.0003), nfcorpus +0.0103 (Holm p 0.0122), scifact -0.0197 (Holm p 0.0004)
- **against_s1_selector**: msmarco +0.0102 (Holm p 0.0003), nfcorpus +0.0103 (Holm p 0.0122), scifact -0.0313 (Holm p 0.0003)
- **against_variance**: msmarco +0.0000 (Holm p 1.0000), nfcorpus +0.0137 (Holm p 0.0004), scifact +0.1006 (Holm p 0.0003)
- **note_msmarco**: E0 does not reproduce Learning-to-Select on MS MARCO (ADR-0006). E1 scores MS MARCO dev against the full 8.8M-passage corpus; the paper evaluates the `test` split of an unpublished `msmarco_subset_50k` folder, with an ideal DCG taken from the retrieved list (F4 research D9).
- **note_nfcorpus**: NFCorpus's selector does not fit its oracle distribution as a calibrated distribution (ADR-0006); read its shift with its ranking-fit result (F4 research D2).
- **selected**: msmarco fixed-307-prior-lr0.0001-ts0.02-b0.1, nfcorpus multi-budget-prior-lr0.001-ts0.05-b0.1, scifact multi-budget-prior-lr0.001-ts0.05-b0.02
- **unchanged_msmarco**: the selected cell starts from the geometric prior and every seed keeps its step-0 checkpoint (no training step improved validation), so `geometric:msmarco` is the `variance` method itself.

## Decision

The S2 criteria read **pass**. The decision is the researcher's: accept
this reading, or amend it here with the reasons.

## Consequences

- If S2 passes: the geometric selector replaces the S1 selector as the one later work
  builds on, and the prior makes it usable on a new corpus before any label exists.
- If S2 fails: the prior and the learned correction are kept as separate methods, chosen
  per environment.
