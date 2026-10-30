# E2: do space descriptors predict selector shift? (gate G3)

Unit: the dimension; each environment is held out in turn. Target: the S1 selector's
selection frequency per dimension on validation queries (embeddings only, no judgments),
divided by k and averaged over the 10%-90% budgets, then over seeds. E2 scores no
retrieval, so it has no full or prefix rows.

- probe-descriptors `20261029T184544Z-77a32eff` (commit `3995a62e4e94`)
- seeds 0, 1, 2; validation queries per environment: scifact 81, nfcorpus 259, msmarco 5000
- S1-S4 were shaped on these three environments and the targets come from selectors tuned on them, so E2's reading is exploratory; G3 is confirmed by rerunning this spec unchanged on new environments (#60).

## Shift check (research D0)

B1 predicts the held-out environment by the mean of the other environments' seed-averaged
profiles; the ceiling is the mean pairwise Spearman between seeds stepped up to the
three-seed average by Spearman-Brown. The gap is what a descriptor could explain. Rule
(fixed before any target): no shift when the gap is below 0.05 in at least
2 environments.

| held-out | B1 spearman | seed spearman | ceiling | gap | room (gap >= 0.05) |
|---|---|---|---|---|---|
| scifact | 0.4354 | 0.6022 | 0.8195 | +0.3841 | yes |
| nfcorpus | 0.5590 | 0.9221 | 0.9726 | +0.4136 | yes |
| msmarco | 0.5391 | 0.9950 | 0.9983 | +0.4592 | yes |

Reading: there is an environment-specific shift for descriptors to explain.

## Descriptors and probes

- descriptors: scifact `20261029T183731Z-9a436bfe` (P1), nfcorpus `20261029T183738Z-7925d5a0` (P1), msmarco `20261029T183753Z-1874aa0a` (P1)
- sets evaluated: 1 `variance` (P0), 2 `A` (P0), 3 `A+E` (P0), 4 `A+E+tails` (P0), 5 `+B+C` (P1), 6 `+D` (P1)
- shift probe (decisive, research D10): the rank of B1 plus a ridge correction fitted on
  the training environments' deviations from their mean, penalty by GCV; direct probe
  (secondary, D7): ridge from standardized descriptors to target ranks, penalty by
  leave-one-environment-out on the training environments. Standardization and penalties
  use the training environments only. B2 is the held-out environment's `variance`
  selection frequency.
- permutation tests: 1000 per fold, set, and form on the seed-averaged
  primary target (descriptor rows permuted within each environment, the fit redone);
  Holm over the sets evaluated within each fold and form; alpha 0.05.

## G3 criteria

- **SC-001**: set 1 (variance) passes 0 of 3 held-out environments; threshold >= 2 of 3 held-out environments; not met.
- **SC-002**: no set passes SC-001; threshold P0 if a set 1-4 passes SC-001, else P1 if set 5 or 6 does; not met.

Per held-out environment and set, shift form (`tables/criteria.csv`):

| held-out | set | name | protocol | shift spearman | B1 | B2 | p | Holm p | above both | every seed above both | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact | 1 | variance | P0 | 0.4376 | 0.4354 | 0.5381 | 0.0010 | 0.0060 | no | no | no |
| nfcorpus | 1 | variance | P0 | 0.5634 | 0.5590 | 0.6246 | 0.0010 | 0.0060 | no | no | no |
| msmarco | 1 | variance | P0 | 0.5657 | 0.5391 | 0.6687 | 0.0010 | 0.0060 | no | no | no |
| scifact | 2 | A | P0 | 0.4391 | 0.4354 | 0.5381 | 0.0020 | 0.0060 | no | no | no |
| nfcorpus | 2 | A | P0 | 0.5622 | 0.5590 | 0.6246 | 0.0010 | 0.0060 | no | no | no |
| msmarco | 2 | A | P0 | 0.5641 | 0.5391 | 0.6687 | 0.0010 | 0.0060 | no | no | no |
| scifact | 3 | A+E | P0 | 0.4420 | 0.4354 | 0.5381 | 0.0010 | 0.0060 | no | no | no |
| nfcorpus | 3 | A+E | P0 | 0.5543 | 0.5590 | 0.6246 | 0.9660 | 1.0000 | no | no | no |
| msmarco | 3 | A+E | P0 | 0.5639 | 0.5391 | 0.6687 | 0.0010 | 0.0060 | no | no | no |
| scifact | 4 | A+E+tails | P0 | 0.4413 | 0.4354 | 0.5381 | 0.0010 | 0.0060 | no | no | no |
| nfcorpus | 4 | A+E+tails | P0 | 0.5540 | 0.5590 | 0.6246 | 0.9760 | 1.0000 | no | no | no |
| msmarco | 4 | A+E+tails | P0 | 0.5648 | 0.5391 | 0.6687 | 0.0010 | 0.0060 | no | no | no |
| scifact | 5 | +B+C | P1 | 0.4472 | 0.4354 | 0.5381 | 0.0010 | 0.0060 | no | no | no |
| nfcorpus | 5 | +B+C | P1 | 0.5120 | 0.5590 | 0.6246 | 1.0000 | 1.0000 | no | no | no |
| msmarco | 5 | +B+C | P1 | 0.5732 | 0.5391 | 0.6687 | 0.0010 | 0.0060 | no | no | no |
| scifact | 6 | +D | P1 | 0.4623 | 0.4354 | 0.5381 | 0.0010 | 0.0060 | no | no | no |
| nfcorpus | 6 | +D | P1 | 0.5071 | 0.5590 | 0.6246 | 1.0000 | 1.0000 | no | no | no |
| msmarco | 6 | +D | P1 | 0.5356 | 0.5391 | 0.6687 | 0.9570 | 0.9570 | no | no | no |

## Ladder (seed-averaged primary target)

| held-out | set | name | protocol | form | spearman | vs B1 | vs B2 | penalty | p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|
| scifact | B1 | dimension identity | — | — | 0.4354 |  |  |  |  |  |
| scifact | B2 | variance | P0 | — | 0.5381 |  |  |  |  |  |
| scifact | 1 | variance | P0 | shift | 0.4376 | +0.0022 | -0.1005 | 10.0000 | 0.0010 | 0.0060 |
| scifact | 2 | A | P0 | shift | 0.4391 | +0.0037 | -0.0990 | 0.1000 | 0.0020 | 0.0060 |
| scifact | 3 | A+E | P0 | shift | 0.4420 | +0.0066 | -0.0962 | 0.1000 | 0.0010 | 0.0060 |
| scifact | 4 | A+E+tails | P0 | shift | 0.4413 | +0.0059 | -0.0968 | 0.1000 | 0.0010 | 0.0060 |
| scifact | 5 | +B+C | P1 | shift | 0.4472 | +0.0118 | -0.0909 | 0.1000 | 0.0010 | 0.0060 |
| scifact | 6 | +D | P1 | shift | 0.4623 | +0.0269 | -0.0759 | 0.1000 | 0.0010 | 0.0060 |
| scifact | 1 | variance | P0 | direct | 0.5366 | +0.1012 | -0.0015 | 0.1000 | 0.0010 | 0.0060 |
| scifact | 2 | A | P0 | direct | 0.5541 | +0.1187 | +0.0160 | 1000.0000 | 0.0010 | 0.0060 |
| scifact | 3 | A+E | P0 | direct | 0.5475 | +0.1121 | +0.0093 | 1000.0000 | 0.0010 | 0.0060 |
| scifact | 4 | A+E+tails | P0 | direct | 0.4998 | +0.0644 | -0.0383 | 0.1000 | 0.0010 | 0.0060 |
| scifact | 5 | +B+C | P1 | direct | 0.4995 | +0.0641 | -0.0386 | 0.1000 | 0.0010 | 0.0060 |
| scifact | 6 | +D | P1 | direct | 0.5189 | +0.0835 | -0.0192 | 0.1000 | 0.0010 | 0.0060 |
| nfcorpus | B1 | dimension identity | — | — | 0.5590 |  |  |  |  |  |
| nfcorpus | B2 | variance | P0 | — | 0.6246 |  |  |  |  |  |
| nfcorpus | 1 | variance | P0 | shift | 0.5634 | +0.0044 | -0.0612 | 10.0000 | 0.0010 | 0.0060 |
| nfcorpus | 2 | A | P0 | shift | 0.5622 | +0.0031 | -0.0624 | 0.1000 | 0.0010 | 0.0060 |
| nfcorpus | 3 | A+E | P0 | shift | 0.5543 | -0.0047 | -0.0703 | 0.1000 | 0.9660 | 1.0000 |
| nfcorpus | 4 | A+E+tails | P0 | shift | 0.5540 | -0.0050 | -0.0705 | 0.1000 | 0.9760 | 1.0000 |
| nfcorpus | 5 | +B+C | P1 | shift | 0.5120 | -0.0471 | -0.1126 | 0.1000 | 1.0000 | 1.0000 |
| nfcorpus | 6 | +D | P1 | shift | 0.5071 | -0.0520 | -0.1175 | 0.1000 | 1.0000 | 1.0000 |
| nfcorpus | 1 | variance | P0 | direct | 0.6815 | +0.1224 | +0.0569 | 0.1000 | 0.0010 | 0.0060 |
| nfcorpus | 2 | A | P0 | direct | 0.6873 | +0.1283 | +0.0628 | 1000.0000 | 0.0010 | 0.0060 |
| nfcorpus | 3 | A+E | P0 | direct | 0.6974 | +0.1384 | +0.0728 | 1000.0000 | 0.0010 | 0.0060 |
| nfcorpus | 4 | A+E+tails | P0 | direct | 0.6939 | +0.1349 | +0.0693 | 1000.0000 | 0.0010 | 0.0060 |
| nfcorpus | 5 | +B+C | P1 | direct | 0.6528 | +0.0938 | +0.0282 | 0.1000 | 0.0010 | 0.0060 |
| nfcorpus | 6 | +D | P1 | direct | 0.6619 | +0.1029 | +0.0373 | 0.1000 | 0.0010 | 0.0060 |
| msmarco | B1 | dimension identity | — | — | 0.5391 |  |  |  |  |  |
| msmarco | B2 | variance | P0 | — | 0.6687 |  |  |  |  |  |
| msmarco | 1 | variance | P0 | shift | 0.5657 | +0.0266 | -0.1030 | 1.0000 | 0.0010 | 0.0060 |
| msmarco | 2 | A | P0 | shift | 0.5641 | +0.0250 | -0.1046 | 10.0000 | 0.0010 | 0.0060 |
| msmarco | 3 | A+E | P0 | shift | 0.5639 | +0.0248 | -0.1048 | 10.0000 | 0.0010 | 0.0060 |
| msmarco | 4 | A+E+tails | P0 | shift | 0.5648 | +0.0257 | -0.1039 | 10.0000 | 0.0010 | 0.0060 |
| msmarco | 5 | +B+C | P1 | shift | 0.5732 | +0.0341 | -0.0954 | 10.0000 | 0.0010 | 0.0060 |
| msmarco | 6 | +D | P1 | shift | 0.5356 | -0.0036 | -0.1331 | 1.0000 | 0.9570 | 0.9570 |
| msmarco | 1 | variance | P0 | direct | 0.7864 | +0.2473 | +0.1177 | 0.1000 | 0.0010 | 0.0060 |
| msmarco | 2 | A | P0 | direct | 0.7441 | +0.2050 | +0.0754 | 100.0000 | 0.0010 | 0.0060 |
| msmarco | 3 | A+E | P0 | direct | 0.7464 | +0.2073 | +0.0777 | 100.0000 | 0.0010 | 0.0060 |
| msmarco | 4 | A+E+tails | P0 | direct | 0.7450 | +0.2059 | +0.0763 | 100.0000 | 0.0010 | 0.0060 |
| msmarco | 5 | +B+C | P1 | direct | 0.7418 | +0.2027 | +0.0731 | 1000.0000 | 0.0010 | 0.0060 |
| msmarco | 6 | +D | P1 | direct | 0.7492 | +0.2101 | +0.0806 | 1000.0000 | 0.0010 | 0.0060 |

## Noise ceilings

| environment | target | seed spearman | ceiling |
|---|---|---|---|
| scifact | s1 | 0.6022 | 0.8195 |
| scifact | s4_correction | 0.9178 | 0.9710 |
| nfcorpus | s1 | 0.9221 | 0.9726 |
| nfcorpus | s4_correction | 0.7223 | 0.8864 |
| msmarco | s1 | 0.9950 | 0.9983 |
| msmarco | s4_correction | 0.9202 | 0.9719 |

## Environment-level context (descriptive, research D11)

| environment | participation ratio | mean energy | mean S4 a_k | E1 degradation as test (train: nDCG@10) |
|---|---|---|---|---|
| scifact | 98.8111 | 0.2847 | 0.0555 | msmarco: +0.1609; nfcorpus: +0.1882 |
| nfcorpus | 90.8628 | 0.3081 | 0.0250 | msmarco: +0.0152; scifact: +0.0303 |
| msmarco | 223.8985 | 0.1635 | 0.8352 | nfcorpus: +0.0124; scifact: +0.0153 |

Per-seed targets: `tables/seeds.csv`; per-budget, S1 bias, and S4 correction targets:
`tables/secondary.csv`; standardized coefficients: `tables/coefficients.csv`; descriptor
column means per environment: `tables/context.csv`.
