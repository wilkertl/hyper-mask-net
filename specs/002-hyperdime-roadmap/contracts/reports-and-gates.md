# Contract: reports, gate decisions, and repository markers

## Reports (versioned in Git)

```text
reports/
├── E0/   baseline reproduction      → G1
├── E1/   selector shift             → G2
├── E2/   space signal (probes)      → G3
├── E3/   main result (meta-test)    → G4
├── E4/   ablations
├── E5/   sensitivity
└── E6/   efficiency                 → G5
    ├── README.md                    # generated: tables, figures, protocol label, manifest run_ids
    ├── tables/*.csv                 # generated
    └── figures/*.svg                # generated
```

- Every table states its protocol (P0, P1, P2, in-domain) and scoring (exact, or FAISS for
  deployment-only tables in E6).
- Every table with a method comparison includes full-1024 and MRL prefix-k rows; hypernetwork tables
  also include the global selector (risks R5, R6).
- Significance columns: paired randomization p-value (Holm-adjusted) and 95% bootstrap CI.

## Gate decisions

| Gate | Report | Criteria source | ADR | Tag |
|---|---|---|---|---|
| G1 | E0 | research R7 | `ADR-XXXX-g1-baseline.md` | `g1-baseline-valid` |
| G2 | E1 | research R8 | `ADR-XXXX-g2-selector-shift.md` | `g2-selector-shift` |
| G3 | E2 | research R10 | `ADR-XXXX-g3-space-signal.md` | `g3-space-signal` |
| — | meta-val | research R14 | `ADR-XXXX-meta-validation.md` | — |
| G4 | E3 | spec SC-006 | `ADR-XXXX-g4-hypernetwork.md` | `g4-hypernetwork` |
| G5 | E6 | spec SC-007 | `ADR-XXXX-g5-cost.md` | `g5-cost` |

- `hyperdime decide <gate>` drafts the ADR and `GateDecision` from result files; the researcher
  reviews and merges it. The tag is applied to the merge commit, pass or fail.
- A failed gate: later milestone issues close as *not planned*; later features are not specified.

## Other ADRs required by the roadmap

- `ADR-XXXX-meta-split.md` — fixes `MetaSplit` (research R3) before F5 fits normalization.
- `ADR-XXXX-environment-pool.md` — records the reference-paper datasets (research R19), their
  revisions, shared corpora, and whether Robust04 was available.

## Comparability tables

Reports that cover a dataset used by a reference paper include a table in that paper's format:
Learning to Select (SciFact, NFCorpus, MS MARCO; nDCG@10, 30% ratio row) in E0; DIME family
(DL'19, DL'20, DL-HARD, Robust04; nDCG@10 and AP over retained-dimension ratios) in E0 and E3;
Unveiling DIME (13 BEIR tasks; nDCG@10 at 0.2/0.4/0.6/0.8 retained) in E3. Published numbers are
cited, never copied into result files.
