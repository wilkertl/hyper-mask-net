# Experiment registry

One row per experiment from plan §6. The report column links to generated reports and manifests;
numbers are never typed into this file.

| ID | Question | Protocol | Gate | Issues | Status | Report |
|---|---|---|---|---|---|---|
| E0 | Do the embeddings, oracle, and Learning-to-Select baseline reproduce the paper's trend? | in-domain | G1 | E0 | reported | [reports/E0](../reports/E0/README.md), [ADR-0006](decisions/ADR-0006-g1-baseline.md) |
| E1 | Does a selector trained on environment A degrade on B, and do its masks change systematically? | in-domain, transfer | G2 | M4.1, M4.2, M4.3, G2 | reported | [reports/E1](../reports/E1/README.md), [ADR-0007](decisions/ADR-0007-g2-selector-shift.md) |
| S1 | Does the same linear selector, trained on masked retrieval, beat Learning-to-Select over the 10%–90% budgets? | in-domain | — (ADR-0008) | F5 | reported | [reports/S1](../reports/S1/README.md), [ADR-0009](decisions/ADR-0009-s1-selector-objective.md) |
| E2 | Do environment descriptors predict properties of specialized selectors? | P0, P1 descriptors | G3 | M5.4 | planned | — |
| E3 | Does HyperDIME narrow the global vs per-domain gap on unseen environments? | P0, P1 | G4 | E3 | planned | — |
| E4 | Which descriptor groups and design choices matter? | P0, P1 | — | E4/E5 | planned | — |
| E5 | How sensitive are results to seeds, sample sizes, τ, negatives, rank, and k? | P0, P1 | — | E4/E5 | planned | — |
| E6 | Is offline adaptation amortizable, and is the online overhead acceptable? | — | G5 | M12.2 | planned | — |

S-rows are selector experiments (ADR-0008); they are not gates and do not renumber the plan's
E-series. Status values: `planned`, `running`, `reported`, `superseded`. When a result is superseded, keep
the row and link the ADR or pull request that invalidated it.
