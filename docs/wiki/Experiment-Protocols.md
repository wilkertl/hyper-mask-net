# Experiment Protocols

Canonical version: [docs/protocols.md](https://github.com/wilkertl/hyper-mask-net/blob/master/docs/protocols.md).

## Environments

BEIR SciFact, BEIR NFCorpus, and MS MARCO with a sampled corpus. Extra environments may come from
subdomains, instructions, or coherent shards, but shards of one dataset are not unseen domains.

## Two levels of splits

- **Environments:** meta-train (gradients and descriptor normalization), meta-validation (model
  selection), meta-test (evaluated once).
- **Queries within an environment:** train (targets, selectors, negatives), validation (early
  stopping), test (final evaluation only).

## Adaptation protocols

| Protocol | Allowed inputs for a new environment | Claim |
|---|---|---|
| **P0** corpus-only | Document embeddings | Strongest zero-shot |
| **P1** unlabeled calibration | Documents + a few unlabeled queries | Expected main protocol |
| **P2** few-label | P1 + a small labeled set | Never reported as zero-shot |

## Leakage rules

1. P0 and P1 code paths never read qrels.
2. Descriptor normalization is fit on meta-train environments only.
3. The meta-trainer cannot load meta-test environments.
4. Test qrels never reach training, negative mining, oracle targets, or descriptors.

## Reporting

nDCG@10, MRR@10/100 and Recall@100 with exact scoring; full 1024 and MRL prefix-k always included;
paired per-query tests with confidence intervals; every table labeled with its protocol and
generated from result files.
