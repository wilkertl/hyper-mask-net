# Council review — hypernetwork for corpus-generalizing selectors

Members: Gemini 4 Neon, Claude Opus 5.5, GPT-6.2-Astra (high reasoning). Independent, parallel.

## Input ("what was done until now")
- `BRIEF.md` — problem, constraints, sources, skills.
- `round0-seed.md` — orchestrator's initial proposal.
- `round1-{astra,opus-e,opus-m}.md` — independent proposals.
- `round2-{astra,opus-e,opus-m}.md` — cross-critiques; convergence on a gated ladder.
- IGNORE `round3-synthesis.md` if present: it is being written by an in-flight agent and may be partial.

## Current consensus after round 2 (verify, do not trust)
- Comparator G+V: corpus-balanced global selector + target-corpus variance prior with global
  per-budget weights. E1 global was a plain union, 93.6% MS MARCO (45,000/48,059), and includes the
  target's own queries (`src/hyperdime/.../train_global.py:54-60` per Opus-E).
- Ladder: G+V → HyperPrior (corpus stats → prior weight / budget profile) → per-dimension bias →
  rank-4 correction; each rung unlocked only by incremental held-out-family gain.
- Prior weight varies ~2.7× on a scale-free share (0.95/0.43/0.36 at k=307), not orders of magnitude;
  budget profile differs; MS MARCO S3 weight never left init.
- Rulings: reference selector trained on a meta-test corpus's training labels is inadmissible;
  payoff census on meta-train only after split freeze; P2 needs its own pool.
- Power: 0.005 gain needs ≥25 families (unavailable); 8–12 meta-test families power ≈0.010.
- Open: (1) first generated output; (2) what a failed G3 rerun kills; (3) test-only-label corpora
  policy; (4) confirmatory vs exploratory at 8–12 families; (5) primary endpoint and comparator.

## Rules
Read-only on the repository and sources except your own output file
`council-<member>.md` in this folder. No code edits, commits, downloads, or training runs.
Papers: /home/wilker/Papers (never open requerimento_passaporte.pdf). Skills:
/home/wilker/.claude/skills/{scientific-critical-thinking,peer-review,experimental-design,statistical-power}/SKILL.md.
AGENTS.md hard constraints apply.
