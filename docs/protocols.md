# Experimental protocols

Rules that decide which data each stage may see and how results are reported. Code must enforce
them with tests wherever possible.

## Environments

An environment is `T = (corpus, query distribution, instruction, frozen Qwen3-Embedding-0.6B)`.
The initial environments are BEIR SciFact, BEIR NFCorpus, and MS MARCO with a reproducibly sampled
corpus. Extra environments may come from subdomains, instructions, or semantically coherent shards;
reports must state that shards of one dataset are not unseen domains.

## Splits

Two levels, never mixed:

| Level | Unit | Sets | Rule |
|---|---|---|---|
| Environment | environment | meta-train, meta-validation, meta-test | Gradients for `G_ψ`, `H_φ`, `θ_0` and descriptor normalization use meta-train only. Architecture and hyperparameters are chosen on meta-validation. Meta-test is evaluated once, after the meta-validation gate. |
| Query | query in an environment | train, validation, test | Oracle targets, selectors, and hard-negative pools use train. Early stopping uses validation. BEIR test qrels are used only for final evaluation. |

## Adaptation protocols

| Protocol | Inputs when adapting to a new environment | Descriptor groups |
|---|---|---|
| **P0** corpus-only | document embeddings | A (document marginal), E (redundancy) |
| **P1** unlabeled calibration | documents and a small unlabeled query set `Q_calib` | A, B, C, E, and D through pseudo-feedback only |
| **P2** few-label calibration | P1 inputs plus a small labeled calibration set | all groups, D supervised on those labels |

P0 is the strongest zero-shot claim. P1 is expected to be the main protocol. P2 is a separate
scenario and is never reported as zero-shot. Once `θ_T` exists, each query needs only
encode → selector → mask → retrieval.

## Leakage rules

1. No code path used by P0 or P1 may read qrels.
2. Descriptor normalization statistics are fit on meta-train environments only.
3. The meta-trainer refuses to load meta-test environments (config validation).
4. Test qrels never reach training, hard-negative mining, oracle targets, or descriptors.
5. Hard-negative pools come from training queries only.

## Reporting

- **Primary metrics:** nDCG@10, MRR@10, MRR@100, Recall@100, with exact scoring. nDCG uses linear
  gains, as trec_eval does.
- **Selection metrics:** retained dimensions, overlap@k, entropy of the predicted importance, KL to
  the oracle, per-dimension usage frequency.
- **Efficiency:** Qwen encoding, selector, mask and scoring, and search time reported separately;
  offline adaptation cost reported separately from online cost; size of `θ_T`.
- **Mandatory comparisons:** full 1024 dimensions and MRL prefix-k in every table (risk R5); a
  global multi-domain selector trained with the same labels and similar capacity whenever the
  hypernetwork is evaluated (risk R6).
- **Significance:** paired per-query tests, multiple-comparison correction for families of
  hypotheses, and confidence intervals rather than p-values alone.
- **Provenance:** every table states its protocol (P0, P1, P2, or in-domain) and is generated from
  result files that have run manifests.
