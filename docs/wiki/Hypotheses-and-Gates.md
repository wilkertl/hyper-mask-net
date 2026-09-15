# Hypotheses and Gates

The project tests a causal chain, one link at a time:

`distribution shift → embedding geometry shift → optimal selector shift → hypernetwork adaptation`

| Hypothesis | Claim | If it fails |
|---|---|---|
| **H1** Selector shift | The optimal selector for the same Qwen model changes systematically across environments | The hypernetwork is unnecessary; build a universal selector |
| **H2** Geometry explains the shift | Statistics of the embedding space predict part of that change | The space representation is insufficient; iterate on the descriptor |
| **H3** Hypernetwork adaptation | A Hyper Head maps environment statistics to selector parameters that beat a global selector cross-domain | The problem is the Hyper Head's parameterization or training |

## Gates

| Gate | Passes when | Decided by | Tag |
|---|---|---|---|
| **G1** Valid baseline | Learning-to-Select converges, masking is mathematically correct, exact retrieval ranks as expected | `[E0]` | `g1-baseline-valid` |
| **G2** H1 supported | Selectors differ in behavior beyond seed noise and transfer degrades retrieval | `[G2]` | `g2-selector-shift` |
| **G3** H2 supported | Descriptors predict selector or transfer properties above a trivial baseline, robustly | `[M5.4]` | `g3-space-signal` |
| **G4** H3 supported | The hypernetwork narrows the global vs per-domain gap on unseen environments | `[E3]` | `g4-hypernetwork` |
| **G5** Acceptable cost | Offline adaptation is amortizable; online overhead keeps the dimension-reduction benefit | `[M12.2]` | `g5-cost` |

Every gate decision is an ADR in `docs/decisions/`. A failed gate is recorded as evidence, and work
does not continue by adding complexity.

## Main risks

| Risk | Mitigation |
|---|---|
| R1 Selector shift is small | Proven or refuted first, at G2 |
| R2 Descriptor learns dataset identity instead of geometry | Truly separate meta-test environments; permutation and probe controls |
| R3 Pseudo-feedback is circular | P0 corpus-only control; P1 always labeled; no qrels in meta-test descriptors |
| R4 Unstable generated parameters | Residual low-rank, near-zero deltas, clipping, delta regularization |
| R5 MRL truncation already solves it | MRL prefix-k in every table |
| R6 Gains come from supervision, not the hypernetwork | Global multi-domain selector with the same labels and capacity |
| R7 ANN hides the effect | Scientific results use exact scoring |
