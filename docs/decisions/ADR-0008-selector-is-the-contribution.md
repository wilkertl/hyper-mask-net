# ADR-0008: The per-environment selector is the contribution; the hypernetwork waits

- **Status:** proposed
- **Date:** 2026-10-14
- **Issue:** F5 (`specs/006-selector-objective`), experiment S1

## Context

The plan (ADR-0001) builds, after G2, a space descriptor (Milestone 3, gate G3) and then a
hypernetwork that maps it to a per-environment selector (Milestone 4, gate G4). The hypernetwork
only automates producing a selector for a new environment; whatever it produces can be no better
than the per-environment selector it imitates. E0 and E1 show that selector is the weak link:

- The oracle mask retrieves far better than the Learning-to-Select (L2S) selector at the same
  budget in every environment (`reports/E0/tables/g1.csv`, criterion c).
- The ADR-0003 recipe candidates improved significantly on the official recipe only on SciFact
  (`reports/E0/tables/selection.csv`, ADR-0006): tuning L2S's hyperparameters is exhausted.
- On MS MARCO, the environment with the most training data, the in-domain selector stays below
  full 1024 at every budget below 100% and does not beat the global selector at most budgets
  (`reports/E1/tables/progression.csv`, from the budget-progression run of task T037).
- L2S is trained to match the oracle's importance distribution (KL), but retrieval uses only its
  top-k ranking; on NFCorpus the chosen target is close to uniform (ADR-0006).

With three environments, a descriptor-to-selector map (G3, G4) also has too few points to fit or
test, so Milestones 3 and 4 would need many more environments before they could pass, while the
selector they would generalize is not yet good where it matters.

## Decision

Proposed to the researcher, who stated on 2026-10-14 that the selector is the project's
contribution and the hypernetwork a mechanism to automate it:

1. **The per-environment selector is the primary contribution.** The next feature, F5
   (`specs/006-selector-objective`), trains the same linear selector for retrieval through a
   relaxed top-k mask, fixed-budget and multi-budget, and experiment S1 compares it with L2S at
   every 10% budget. S1's criteria are in the F5 spec (SC-001, SC-002).
2. **Milestones 3 and 4 (space descriptor, hypernetwork) are deferred, not cancelled.** They are
   re-planned after S1 is decided, around the selector S1 selects, and once more environments are
   prepared.
3. **No constitution amendment is needed for F5.** It keeps the linear selector (constitution IV:
   lowest capacity first) and adds no stage beyond G2. Higher capacity, a per-query budget policy,
   or soft re-weighting would each need S1's evidence that the linear selector or the fixed budget
   is limiting.
4. **Selector experiments get their own series, S1, S2, ….** The plan's E-series keeps its
   numbering (E2 is still the space-signal experiment of gate G3), so the first selector
   experiment is S1; it is not a gate, and its decision record uses `GateDecision(gate="S1")`.
5. **L2S stays the reproduced baseline.** Its trainer, checkpoints, and the E0 and E1 results are
   not changed; every S1 table compares against it.

## Evidence

Every value referenced above is in a generated table: `reports/E0/tables/g1.csv`,
`reports/E0/tables/selection.csv`, `reports/E1/tables/methods.csv`,
`reports/E1/tables/progression.csv` (budget-progression run, transfer-matrix
`20261015T004735Z-4e76ec6e`). No number is restated here.

## Consequences

- `docs/wiki/Roadmap.md` and `docs/wiki/Hypotheses-and-Gates.md` gain an S1 step between G2 and
  G3 on acceptance; Milestone 3 and 4 issues are labeled `status:blocked` on S1.
- S1's decision is recorded in its own ADR. If the retrieval-trained selector meets S1's criteria,
  it replaces L2S as the selector later work builds on; if not, the next candidates are higher
  capacity or a budget policy, each justified by S1's diagnostics.
- More BEIR environments remain needed, now to show a selector improvement holds across domains;
  they are a separate feature.
- Depends on ADR-0007 (G2) being accepted; the hypothesis order H1 → H2 → H3 is unchanged, only
  delayed.
