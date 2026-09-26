# ADR-0002: Selector training follows the official Learning-to-Select code

- **Status:** accepted
- **Date:** 2026-09-26
- **Issue:** F0 spike (branch `agent/F0-h1-spike`)

## Context

The first full F0 run (`reports/F0/README.md` at commit `94bbefc`) returned "H1 not supported", but
on nfcorpus, fiqa, arguana, and scidocs the in-domain selector did not beat random dimensions at
k = 307, so the run could not measure selector shift. Training histories
(`artifacts/spike/*/selectors/history-seed-*.json`) showed validation KL bottoming out after 2–4
epochs and early stopping triggered by an absolute `1e-6` tolerance on losses of order 1e-5–1e-3.

The official implementation (`train_and_eval.py` of the Learning-to-Select release) differs from
ours in its label generation and optimization recipe.

## Decision

Match the official recipe, keeping our leakage-safe protocol:

- Oracle positives are weighted by the exponential gain `2**rel - 1`.
- Hard negatives: 64 documents sampled with replacement from the top-1000 non-positives of the
  full embedding (seeded), instead of the 8 hardest.
- Optimizer: AdamW, lr 1e-4, weight decay 1e-4, batch 32, cosine decay over a fixed number of
  epochs; the state with the lowest validation KL is kept; no early stopping.
- The official code tunes temperature and epochs per (model, dataset). We choose both per
  environment on **validation** nDCG@10 at the headline k over temperatures
  {0.005, 0.01, 0.02, 0.05, 0.1, 0.2} x epochs {10, 50, 200}, with seed 0, using training labels
  only. The global selector gets the same epoch tuning on mean validation nDCG@10, so the
  C3 comparison stays fair.

## Evidence

A seed-0 check on the cached nfcorpus embeddings, outside the pipeline, moved the in-domain
selector above random and prefix; the rerun's generated report is the authoritative evidence.

## Consequences

- All F0 selectors, the analysis, and the report must be regenerated; `prepare` reruns to write
  the new negatives but reuses cached embeddings.
- Training cost grows by the 18-cell grid (about 10 minutes per mid-sized environment on CPU).
- Validation sets of 50–81 queries (scidocs, arguana, scifact) make the grid choice noisy;
  choices at the grid edge suggest widening it.
