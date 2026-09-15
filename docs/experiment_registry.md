# Experiment registry

One row per experiment from plan §6. The report column links to generated reports and manifests;
numbers are never typed into this file.

| ID | Question | Protocol | Gate | Issues | Status | Report |
|---|---|---|---|---|---|---|
| E0 | Do the embeddings, oracle, and Learning-to-Select baseline reproduce the paper's trend? | in-domain | G1 | E0 | planned | — |
| E1 | Does a selector trained on environment A degrade on B, and do its masks change systematically? | in-domain, transfer | G2 | M4.1, M4.2, M4.3, G2 | planned | — |
| E2 | Do environment descriptors predict properties of specialized selectors? | P0, P1 descriptors | G3 | M5.4 | planned | — |
| E3 | Does HyperDIME narrow the global vs per-domain gap on unseen environments? | P0, P1 | G4 | E3 | planned | — |
| E4 | Which descriptor groups and design choices matter? | P0, P1 | — | E4/E5 | planned | — |
| E5 | How sensitive are results to seeds, sample sizes, τ, negatives, rank, and k? | P0, P1 | — | E4/E5 | planned | — |
| E6 | Is offline adaptation amortizable, and is the online overhead acceptable? | — | G5 | M12.2 | planned | — |

Status values: `planned`, `running`, `reported`, `superseded`. When a result is superseded, keep
the row and link the ADR or pull request that invalidated it.
