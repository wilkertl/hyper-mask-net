# Experiment registry

One row per experiment from plan §6. The report column links to generated reports and manifests;
numbers are never typed into this file.

| ID | Question | Protocol | Gate | Issues | Status | Report |
|---|---|---|---|---|---|---|
| E0 | Do the embeddings, oracle, and Learning-to-Select baseline reproduce the paper's trend? | in-domain | G1 | E0 | reported | [reports/E0](../reports/E0/README.md), [ADR-0006](decisions/ADR-0006-g1-baseline.md) |
| E1 | Does a selector trained on environment A degrade on B, and do its masks change systematically? | in-domain, transfer | G2 | M4.1, M4.2, M4.3, G2 | reported | [reports/E1](../reports/E1/README.md), [ADR-0007](decisions/ADR-0007-g2-selector-shift.md) |
| S1 | Does the same linear selector, trained on masked retrieval, beat Learning-to-Select over the 10%–90% budgets? | in-domain | — (ADR-0008) | F5 | reported | [reports/S1](../reports/S1/README.md), [ADR-0009](decisions/ADR-0009-s1-selector-objective.md) |
| P1 | How much of the selector's gain does a label-free geometric score `q_d² · Var(x_d)` recover, and does effective dimensionality order the best budget? | in-domain | — (probe for S2) | F6 | reported | [reports/P1](../reports/P1/README.md) |
| S2 | Does S1's retrieval selector, started from the geometric prior `q_d² · Var(x_d)`, beat the S1 selector without losing the prior's strength? | in-domain | — (ADR-0008) | F7 | reported | [reports/S2](../reports/S2/README.md), [ADR-0010](decisions/ADR-0010-s2-geometric-prior.md) |
| S3 | Does one learned weight on the geometric prior give a selector never worse than the S1 selector, the S2 selector, or `variance`? | in-domain | — (ADR-0008) | F8, #59 | reported | [reports/S3](../reports/S3/README.md), [ADR-0011](decisions/ADR-0011-s3-prior-weight.md) |
| S4 | Is a selector with one prior weight per budget non-inferior (δ = 0.005) to aligned S3, the S2 selector, and `variance`, and better than aligned S3 somewhere? | in-domain | — (ADR-0008) | F9 | reported | [reports/S4](../reports/S4/README.md), [ADR-0012](decisions/ADR-0012-s4-budget-prior.md) (proposed; exploratory until rerun on new environments, #60) |
| E2 | Do environment descriptors predict properties of specialized selectors? | P0, P1 descriptors | G3 | M5.4, F10 | reported (stages 1-2: P0, P1) | [reports/E2](../reports/E2/README.md), [ADR-0013](decisions/ADR-0013-g3-space-signal.md) (proposed; exploratory until rerun on new environments, #60) |
| E3 | Does HyperDIME narrow the global vs per-domain gap on unseen environments? | P0, P1 | G4 | E3 | planned | — |
| E4 | Which descriptor groups and design choices matter? | P0, P1 | — | E4/E5 | planned | — |
| E5 | How sensitive are results to seeds, sample sizes, τ, negatives, rank, and k? | P0, P1 | — | E4/E5 | planned | — |
| E6 | Is offline adaptation amortizable, and is the online overhead acceptable? | — | G5 | M12.2 | planned | — |

S-rows are selector experiments and P-rows probes that decide whether to run one (ADR-0008); they are not gates and do not renumber the plan's
E-series. Status values: `planned`, `running`, `reported`, `superseded`. When a result is superseded, keep
the row and link the ADR or pull request that invalidated it.
