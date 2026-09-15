# ADR-0001: Restart the repository as HyperDIME-Qwen

- **Status:** accepted
- **Date:** 2026-09-15

## Context

The previous prototype, Hyper-Mask-Net (commits `00fb5db`–`65fa90f`), generated a new Mask-Net for
each query from that query's Qwen token states, in the style of Hypencoder. On BEIR SciFact, learned
masks kept more retrieval quality than random masks, but the per-query hypernetwork did not beat the
unmasked embedding, nor the linear Learning-to-Select predictor at 30% of dimensions.

The [implementation plan](../plan/Qwen3_HyperDIME_Implementation_Plan.en.md) replaces that design.
The hypernetwork is conditioned on a description of the retrieval environment, produces a low-rank
modification of a linear selector once per environment, and is built only after gates show that
selector shift exists and can be predicted from the embedding space.

## Decision

- Remove the previous code, without compatibility with its checkpoints or data files.
- Adopt the plan's `src/hyperdime` layout, gated milestones, and agent workflow.
- Keep only functions whose math the new architecture needs, rewritten with unit tests:

| Previous code | New location | Change |
|---|---|---|
| `prepare_embeddings.format_query` | `embeddings/qwen.py` | none |
| `prepare_embeddings.last_token_pool` | `embeddings/qwen.py` | Handles left and right padding. The old version assumed right padding, which matched the tokenizer default but would silently break under left padding. |
| `train.oracle_importance` | `oracle/importance.py` | Split into `oracle_scores` and `oracle_importance`. |
| `models.LinearMaskPredictor` | `baselines/learning_to_select.py` | Log-softmax output as in the paper; no fixed prediction temperature. |
| `train.kl_loss` | `training/losses.py` | Takes log-probabilities instead of clamping and taking the log of probabilities. |
| `models.select_top_k` | `selection/topk.py` | Split into `top_k_mask` and `apply_mask`; ratios use `floor(ρD)` as in the plan instead of rounding. |
| `evaluate_scifact.metric_summary` | `evaluation/metrics.py` | Linear-gain nDCG as in trec_eval (identical on binary qrels, different on graded ones such as NFCorpus); several cutoffs; per-query values for paired tests. |
| FAISS search in `evaluate_scifact.retrieve` | `retrieval/exact.py` | Exact batched dot product; FAISS deferred to deployment analysis. |
| `prepare_scifact.split_query_ids` | `data/splits.py` | Standard-library RNG; generic IDs. |
| Negative mining in `prepare_scifact.build_triples` | `data/negatives.py` | Returns pools of several negatives using exact search. |

## Consequences

- Earlier SciFact numbers came from a different pipeline and are not reused as baselines. E0
  reproduces all baselines under the new pipeline.
- The per-query design stays retrievable from git history at `65fa90f`.
- Local `data/` and `runs/` directories from the prototype are gitignored and unused.
