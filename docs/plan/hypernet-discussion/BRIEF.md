# Multi-agent discussion: a hypernetwork for corpus-generalizing dimension selectors

## Problem
HyperDIME selects k of D=1024 dimensions of a frozen Qwen3-Embedding-0.6B query embedding;
score(d) = <q ⊙ mask, d>, documents/index untouched, masked query not re-normalized.
A selector trained on one corpus transfers poorly to another (E1: in-domain beats global by
+0.112 nDCG@10 on SciFact, +0.017 NFCorpus, −0.002 MS MARCO at k=307). The question:
**what hypernetwork (a map from an unseen corpus, using no meta-test labels, to selector
parameters or corrections) should the project build, and how should it be tested?**

## Participants
- **Astra** (GPT-6.2-Astra) — architecture lead: model design, parameterization, training.
- **Opus-E** (Claude Opus 5.5) — evidence lead: what project results and the literature
  actually support; constraints and prior art.
- **Opus-M** (Claude Opus 5.5) — methods lead / red team: evaluation design, statistics,
  falsification, leakage, gates.
Every participant proposes a full design; the role is the lens you lead with, not a limit.

## Required reading (read before writing)
1. Project: `AGENTS.md`, `docs/architecture.md`, `docs/protocols.md`,
   `docs/plan/Qwen3_HyperDIME_Implementation_Plan.en.md`, `docs/experiment_registry.md`,
   `docs/references.md`, `docs/decisions/ADR-0007` … `ADR-0020`, and `reports/` summaries
   (E1, E2, S1–S4, P3*). `src/hyperdime/contracts/` for interfaces.
2. Sources: PDFs in `/home/wilker/Papers/` (ignore `requerimento_passaporte.pdf`, a personal
   document — do not open it). Priority: Learning to Select, DIME, Eclipse, Statistical
   Foundations of DIME, Unveiling DIME, When Reducing Representations Improves, HyperNetworks,
   Principled Weight Initialization, Hypencoder, Revisited Hypencoder, Generalization Bounds via
   Meta-Learned Model Representations, Search-Adapter, AdaWidth, Align then Train, SPIN,
   A Survey of Controllable Learning. Extract text with `pdftotext` (or python) into /tmp, never
   into the repo. Cite page/section for every claim taken from a paper.
3. Scientific skills (read the SKILL.md and use its method): `/home/wilker/.claude/skills/`
   `scientific-critical-thinking`, `hypothesis-generation`, `experimental-design`,
   `statistical-power`, `scientific-brainstorming`, `peer-review`, `literature-review`.

## Hard constraints (from AGENTS.md)
Qwen3 frozen, D=1024; never use test qrels or any meta-test label in training, adaptation,
negative mining, descriptor construction, or normalization; selection never modifies document
embeddings/index; exact scoring for scientific results; offline adaptation cost separate from
online per-query cost; S1–S4/E2/P3 were shaped on SciFact/NFCorpus/MS MARCO, so results on
those are exploratory.

## Working rules
- Read-only on the repository except your own output file in this folder. No code edits, no
  commits, no downloads, no GPU jobs.
- Separate data (with file path + number) from interpretation. Flag every number you did not
  verify yourself.
- Be specific: equations, tensor shapes, parameter counts, training loop, loss, gates with
  thresholds, comparators, the experimental unit, and what result would kill the idea.

## Rounds
1. Independent proposals → `round1-<name>.md` (do not read others' files in round 1).
2. Cross-critique → `round2-<name>.md` (read all round-1 files plus `round0-seed.md`).
3. Synthesis (Astra) → `round3-synthesis.md`.
4. Sign-off / dissent (Opus-E, Opus-M) → `round4-<name>.md`.
Final consolidated proposal → `PROPOSAL.md` (orchestrator).
