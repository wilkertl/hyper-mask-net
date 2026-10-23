# S4: a prior weight per budget

Protocol: in-domain. Scoring: exact inner product over the full corpus, hard top-k mask.
The budget selector scores `a_k · log(q_d² · Var(x_d) + ε) + (W q + b)_d` with one weight
`a_k >= 0` per 10%-90% budget and the linear map shared by every budget (at step 0 it is
the `variance` method at every budget). Its control, aligned S3, is S3's selector with
one weight, trained on the same 8 cells. Both keep the checkpoint with the best
full-corpus validation nDCG@10 (step 0 included) on a fixed validation subset.
Selected cells (budget / aligned S3): msmarco `multi-budget-budget-lr0.0001-ts0.02-b0.1` / `multi-budget-weighted-aligned-lr0.0001-ts0.02-b0.1`, nfcorpus `multi-budget-budget-lr0.001-ts0.05-b0.02` / `multi-budget-weighted-aligned-lr0.001-ts0.05-b0.1`, scifact `multi-budget-budget-lr0.001-ts0.02-b0.1` / `multi-budget-weighted-aligned-lr0.001-ts0.02-b0.1`; seeds
0, 1, 2, averaged per query before every paired test.

- compare-selectors `20261021T211322Z-0cdb96ab` (commit `855d31baf5bb`)
- Non-inferiority margin δ = 0.005 nDCG@10 (mean over budgets), fixed before any
  S4 run.
- S1-S3's results on these test sets informed the S4 proposal, so S4's reading on them is exploratory; the confirmation reruns this spec unchanged on new environments (#60).

## Criteria

| criterion | value | threshold | met |
|---|---|---|---|
| SC-001 non-inferior to aligned S3, S2 selector, variance (margin 0.005, one-sided, Holm, 0.025) | none fails | none fails | yes |
| SC-002 better than aligned S3 (mean nDCG@10, 10%-90%, Holm, 0.05) | 2 of 3 environments | >= 1 of 3 | yes |

## Non-inferiority (SC-001)

One-sided sign-flip test of `mean(S4 − reference) <= −0.005`, Holm over the
environments for each reference, met at p_holm <= 0.025.

| test | against | diff | ci_low | ci_high | margin | p (one-sided) | p_holm | non-inferior |
|---|---|---|---|---|---|---|---|---|
| msmarco | aligned S3 | +0.0001 | -0.0001 | +0.0003 | 0.0050 | 0.0001 | 0.0003 | yes |
| nfcorpus | aligned S3 | +0.0111 | +0.0082 | +0.0141 | 0.0050 | 0.0001 | 0.0003 | yes |
| scifact | aligned S3 | +0.0094 | +0.0049 | +0.0141 | 0.0050 | 0.0001 | 0.0003 | yes |
| msmarco | S2 selector | +0.0001 | -0.0002 | +0.0004 | 0.0050 | 0.0001 | 0.0003 | yes |
| nfcorpus | S2 selector | +0.0047 | -0.0022 | +0.0117 | 0.0050 | 0.0033 | 0.0033 | yes |
| scifact | S2 selector | +0.0466 | +0.0341 | +0.0597 | 0.0050 | 0.0001 | 0.0003 | yes |
| msmarco | variance | +0.0001 | -0.0002 | +0.0004 | 0.0050 | 0.0001 | 0.0003 | yes |
| nfcorpus | variance | +0.0185 | +0.0087 | +0.0284 | 0.0050 | 0.0001 | 0.0003 | yes |
| scifact | variance | +0.1472 | +0.1152 | +0.1815 | 0.0050 | 0.0001 | 0.0003 | yes |

## Against every reference on the test sets

Two-sided tests, Holm over the environments for each pair, metric, and summary; the last
pair is the aligned control against S3. MRR@10 and Recall@100 are secondary.

| test | query set | metric | candidate | against | summary | reference | candidate value | diff | ci_low | ci_high | p | p_holm | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | test | ndcg@10 | budget | aligned S3 | mean 10%-90% | 0.3760 | 0.3761 | +0.0001 | -0.0001 | +0.0003 | 0.3574 | 0.3574 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | aligned S3 | mean 10%-90% | 0.3661 | 0.3772 | +0.0111 | +0.0082 | +0.0141 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | aligned S3 | mean 10%-90% | 0.8342 | 0.8436 | +0.0094 | +0.0049 | +0.0141 | 0.0002 | 0.0004 | — |
| msmarco | test | ndcg@10 | budget | aligned S3 | k = 307 | 0.3756 | 0.3761 | +0.0005 | -0.0001 | +0.0012 | 0.0788 | 0.1576 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | aligned S3 | k = 307 | 0.3642 | 0.3799 | +0.0157 | +0.0092 | +0.0227 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | aligned S3 | k = 307 | 0.8500 | 0.8522 | +0.0022 | -0.0053 | +0.0097 | 0.5616 | 0.5616 | — |
| msmarco | test | ndcg@10 | budget | S3 selector | mean 10%-90% | 0.3760 | 0.3761 | +0.0001 | -0.0002 | +0.0004 | 0.7107 | 0.7107 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | S3 selector | mean 10%-90% | 0.3674 | 0.3772 | +0.0098 | +0.0068 | +0.0128 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | S3 selector | mean 10%-90% | 0.8340 | 0.8436 | +0.0096 | +0.0051 | +0.0143 | 0.0001 | 0.0003 | — |
| msmarco | test | ndcg@10 | budget | S3 selector | k = 307 | 0.3765 | 0.3761 | -0.0003 | -0.0014 | +0.0007 | 0.5137 | 1.0000 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | S3 selector | k = 307 | 0.3681 | 0.3799 | +0.0118 | +0.0055 | +0.0182 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | S3 selector | k = 307 | 0.8517 | 0.8522 | +0.0005 | -0.0076 | +0.0087 | 0.9085 | 1.0000 | — |
| msmarco | test | ndcg@10 | budget | S2 selector | mean 10%-90% | 0.3760 | 0.3761 | +0.0001 | -0.0002 | +0.0004 | 0.7107 | 0.7107 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | S2 selector | mean 10%-90% | 0.3725 | 0.3772 | +0.0047 | -0.0022 | +0.0117 | 0.1754 | 0.3508 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | S2 selector | mean 10%-90% | 0.7970 | 0.8436 | +0.0466 | +0.0341 | +0.0597 | 0.0001 | 0.0003 | — |
| msmarco | test | ndcg@10 | budget | S2 selector | k = 307 | 0.3765 | 0.3761 | -0.0003 | -0.0014 | +0.0007 | 0.5137 | 0.7231 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | S2 selector | k = 307 | 0.3752 | 0.3799 | +0.0047 | -0.0054 | +0.0147 | 0.3616 | 0.7231 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | S2 selector | k = 307 | 0.8390 | 0.8522 | +0.0132 | -0.0001 | +0.0268 | 0.0522 | 0.1566 | — |
| msmarco | test | ndcg@10 | budget | variance | mean 10%-90% | 0.3760 | 0.3761 | +0.0001 | -0.0002 | +0.0004 | 0.7107 | 0.7107 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | variance | mean 10%-90% | 0.3587 | 0.3772 | +0.0185 | +0.0087 | +0.0284 | 0.0002 | 0.0004 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | variance | mean 10%-90% | 0.6964 | 0.8436 | +0.1472 | +0.1152 | +0.1815 | 0.0001 | 0.0003 | — |
| msmarco | test | ndcg@10 | budget | variance | k = 307 | 0.3765 | 0.3761 | -0.0003 | -0.0014 | +0.0007 | 0.5137 | 0.5137 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | variance | k = 307 | 0.3596 | 0.3799 | +0.0203 | +0.0078 | +0.0331 | 0.0018 | 0.0036 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | variance | k = 307 | 0.6973 | 0.8522 | +0.1549 | +0.1188 | +0.1930 | 0.0001 | 0.0003 | — |
| msmarco | test | ndcg@10 | budget | S1 selector | mean 10%-90% | 0.3658 | 0.3761 | +0.0103 | +0.0084 | +0.0122 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | S1 selector | mean 10%-90% | 0.3622 | 0.3772 | +0.0150 | +0.0089 | +0.0213 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | S1 selector | mean 10%-90% | 0.8282 | 0.8436 | +0.0153 | +0.0085 | +0.0223 | 0.0001 | 0.0003 | — |
| msmarco | test | ndcg@10 | budget | S1 selector | k = 307 | 0.3645 | 0.3761 | +0.0116 | +0.0083 | +0.0150 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | S1 selector | k = 307 | 0.3607 | 0.3799 | +0.0192 | +0.0087 | +0.0298 | 0.0004 | 0.0008 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | S1 selector | k = 307 | 0.8458 | 0.8522 | +0.0065 | -0.0036 | +0.0170 | 0.2106 | 0.2106 | — |
| msmarco | test | ndcg@10 | budget | L2S | mean 10%-90% | 0.3607 | 0.3761 | +0.0155 | +0.0132 | +0.0178 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | L2S | mean 10%-90% | 0.3622 | 0.3772 | +0.0150 | +0.0089 | +0.0213 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | L2S | mean 10%-90% | 0.8166 | 0.8436 | +0.0269 | +0.0173 | +0.0369 | 0.0001 | 0.0003 | — |
| msmarco | test | ndcg@10 | budget | L2S | k = 307 | 0.3572 | 0.3761 | +0.0189 | +0.0150 | +0.0230 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | ndcg@10 | budget | L2S | k = 307 | 0.3607 | 0.3799 | +0.0192 | +0.0087 | +0.0298 | 0.0004 | 0.0008 | see note: nfcorpus |
| scifact | test | ndcg@10 | budget | L2S | k = 307 | 0.8353 | 0.8522 | +0.0169 | +0.0033 | +0.0310 | 0.0161 | 0.0161 | — |
| msmarco | test | ndcg@10 | weighted-aligned | S3 selector | mean 10%-90% | 0.3760 | 0.3760 | -0.0000 | -0.0003 | +0.0002 | 0.8177 | 1.0000 | see note: msmarco |
| nfcorpus | test | ndcg@10 | weighted-aligned | S3 selector | mean 10%-90% | 0.3674 | 0.3661 | -0.0013 | -0.0035 | +0.0007 | 0.2431 | 0.7292 | see note: nfcorpus |
| scifact | test | ndcg@10 | weighted-aligned | S3 selector | mean 10%-90% | 0.8340 | 0.8342 | +0.0002 | -0.0012 | +0.0017 | 0.7563 | 1.0000 | — |
| msmarco | test | ndcg@10 | weighted-aligned | S3 selector | k = 307 | 0.3765 | 0.3756 | -0.0009 | -0.0017 | -0.0000 | 0.0352 | 0.1056 | see note: msmarco |
| nfcorpus | test | ndcg@10 | weighted-aligned | S3 selector | k = 307 | 0.3681 | 0.3642 | -0.0039 | -0.0087 | +0.0006 | 0.1002 | 0.2004 | see note: nfcorpus |
| scifact | test | ndcg@10 | weighted-aligned | S3 selector | k = 307 | 0.8517 | 0.8500 | -0.0017 | -0.0051 | +0.0016 | 0.3187 | 0.3187 | — |
| msmarco | test | mrr@10 | budget | aligned S3 | mean 10%-90% | 0.3133 | 0.3133 | +0.0001 | -0.0002 | +0.0003 | 0.6160 | 0.6160 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | aligned S3 | mean 10%-90% | 0.5534 | 0.5718 | +0.0183 | +0.0119 | +0.0249 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | aligned S3 | mean 10%-90% | 0.8059 | 0.8155 | +0.0097 | +0.0046 | +0.0148 | 0.0004 | 0.0008 | — |
| msmarco | test | mrr@10 | budget | aligned S3 | k = 307 | 0.3124 | 0.3130 | +0.0006 | -0.0001 | +0.0013 | 0.0705 | 0.1410 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | aligned S3 | k = 307 | 0.5417 | 0.5711 | +0.0294 | +0.0135 | +0.0459 | 0.0003 | 0.0009 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | aligned S3 | k = 307 | 0.8262 | 0.8251 | -0.0011 | -0.0097 | +0.0075 | 0.7958 | 0.7958 | — |
| msmarco | test | mrr@10 | budget | S3 selector | mean 10%-90% | 0.3132 | 0.3133 | +0.0001 | -0.0003 | +0.0004 | 0.6987 | 0.6987 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | S3 selector | mean 10%-90% | 0.5556 | 0.5718 | +0.0162 | +0.0101 | +0.0223 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | S3 selector | mean 10%-90% | 0.8057 | 0.8155 | +0.0098 | +0.0048 | +0.0150 | 0.0005 | 0.0010 | — |
| msmarco | test | mrr@10 | budget | S3 selector | k = 307 | 0.3132 | 0.3130 | -0.0003 | -0.0014 | +0.0009 | 0.6499 | 1.0000 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | S3 selector | k = 307 | 0.5486 | 0.5711 | +0.0224 | +0.0089 | +0.0362 | 0.0017 | 0.0051 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | S3 selector | k = 307 | 0.8279 | 0.8251 | -0.0028 | -0.0114 | +0.0058 | 0.5304 | 1.0000 | — |
| msmarco | test | mrr@10 | budget | S2 selector | mean 10%-90% | 0.3132 | 0.3133 | +0.0001 | -0.0003 | +0.0004 | 0.6987 | 1.0000 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | S2 selector | mean 10%-90% | 0.5745 | 0.5718 | -0.0028 | -0.0160 | +0.0103 | 0.6951 | 1.0000 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | S2 selector | mean 10%-90% | 0.7616 | 0.8155 | +0.0540 | +0.0396 | +0.0692 | 0.0001 | 0.0003 | — |
| msmarco | test | mrr@10 | budget | S2 selector | k = 307 | 0.3132 | 0.3130 | -0.0003 | -0.0014 | +0.0009 | 0.6499 | 1.0000 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | S2 selector | k = 307 | 0.5729 | 0.5711 | -0.0018 | -0.0219 | +0.0185 | 0.8672 | 1.0000 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | S2 selector | k = 307 | 0.8094 | 0.8251 | +0.0157 | -0.0011 | +0.0327 | 0.0703 | 0.2109 | — |
| msmarco | test | mrr@10 | budget | variance | mean 10%-90% | 0.3132 | 0.3133 | +0.0001 | -0.0003 | +0.0004 | 0.6987 | 0.6987 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | variance | mean 10%-90% | 0.5585 | 0.5718 | +0.0133 | -0.0030 | +0.0297 | 0.1230 | 0.2460 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | variance | mean 10%-90% | 0.6578 | 0.8155 | +0.1578 | +0.1233 | +0.1938 | 0.0001 | 0.0003 | — |
| msmarco | test | mrr@10 | budget | variance | k = 307 | 0.3132 | 0.3130 | -0.0003 | -0.0014 | +0.0009 | 0.6499 | 0.7513 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | variance | k = 307 | 0.5602 | 0.5711 | +0.0108 | -0.0125 | +0.0343 | 0.3757 | 0.7513 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | variance | k = 307 | 0.6591 | 0.8251 | +0.1660 | +0.1267 | +0.2072 | 0.0001 | 0.0003 | — |
| msmarco | test | mrr@10 | budget | S1 selector | mean 10%-90% | 0.3047 | 0.3133 | +0.0086 | +0.0066 | +0.0107 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | S1 selector | mean 10%-90% | 0.5487 | 0.5718 | +0.0231 | +0.0117 | +0.0344 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | S1 selector | mean 10%-90% | 0.8000 | 0.8155 | +0.0155 | +0.0082 | +0.0231 | 0.0001 | 0.0003 | — |
| msmarco | test | mrr@10 | budget | S1 selector | k = 307 | 0.3030 | 0.3130 | +0.0100 | +0.0063 | +0.0137 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | S1 selector | k = 307 | 0.5466 | 0.5711 | +0.0245 | +0.0054 | +0.0431 | 0.0121 | 0.0242 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | S1 selector | k = 307 | 0.8205 | 0.8251 | +0.0046 | -0.0066 | +0.0163 | 0.4338 | 0.4338 | — |
| msmarco | test | mrr@10 | budget | L2S | mean 10%-90% | 0.2998 | 0.3133 | +0.0135 | +0.0110 | +0.0160 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | L2S | mean 10%-90% | 0.5487 | 0.5718 | +0.0231 | +0.0117 | +0.0344 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | L2S | mean 10%-90% | 0.7863 | 0.8155 | +0.0292 | +0.0185 | +0.0404 | 0.0001 | 0.0003 | — |
| msmarco | test | mrr@10 | budget | L2S | k = 307 | 0.2965 | 0.3130 | +0.0165 | +0.0123 | +0.0210 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | mrr@10 | budget | L2S | k = 307 | 0.5466 | 0.5711 | +0.0245 | +0.0054 | +0.0431 | 0.0121 | 0.0242 | see note: nfcorpus |
| scifact | test | mrr@10 | budget | L2S | k = 307 | 0.8073 | 0.8251 | +0.0178 | +0.0021 | +0.0345 | 0.0307 | 0.0307 | — |
| msmarco | test | mrr@10 | weighted-aligned | S3 selector | mean 10%-90% | 0.3132 | 0.3133 | +0.0000 | -0.0002 | +0.0003 | 0.9150 | 1.0000 | see note: msmarco |
| nfcorpus | test | mrr@10 | weighted-aligned | S3 selector | mean 10%-90% | 0.5556 | 0.5534 | -0.0021 | -0.0063 | +0.0020 | 0.3125 | 0.9374 | see note: nfcorpus |
| scifact | test | mrr@10 | weighted-aligned | S3 selector | mean 10%-90% | 0.8057 | 0.8059 | +0.0001 | -0.0017 | +0.0020 | 0.8890 | 1.0000 | — |
| msmarco | test | mrr@10 | weighted-aligned | S3 selector | k = 307 | 0.3132 | 0.3124 | -0.0009 | -0.0018 | +0.0001 | 0.0692 | 0.2076 | see note: msmarco |
| nfcorpus | test | mrr@10 | weighted-aligned | S3 selector | k = 307 | 0.5486 | 0.5417 | -0.0070 | -0.0175 | +0.0034 | 0.1958 | 0.3916 | see note: nfcorpus |
| scifact | test | mrr@10 | weighted-aligned | S3 selector | k = 307 | 0.8279 | 0.8262 | -0.0017 | -0.0058 | +0.0024 | 0.4149 | 0.4149 | — |
| msmarco | test | recall@100 | budget | aligned S3 | mean 10%-90% | 0.8742 | 0.8741 | -0.0001 | -0.0003 | +0.0002 | 0.5712 | 1.0000 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | aligned S3 | mean 10%-90% | 0.3557 | 0.3592 | +0.0035 | -0.0004 | +0.0074 | 0.0779 | 0.2337 | see note: nfcorpus |
| scifact | test | recall@100 | budget | aligned S3 | mean 10%-90% | 0.9855 | 0.9856 | +0.0001 | -0.0022 | +0.0023 | 1.0000 | 1.0000 | — |
| msmarco | test | recall@100 | budget | aligned S3 | k = 307 | 0.8732 | 0.8734 | +0.0002 | -0.0007 | +0.0010 | 0.7364 | 1.0000 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | aligned S3 | k = 307 | 0.3707 | 0.3669 | -0.0038 | -0.0157 | +0.0071 | 0.5306 | 1.0000 | see note: nfcorpus |
| scifact | test | recall@100 | budget | aligned S3 | k = 307 | 0.9889 | 0.9856 | -0.0033 | -0.0089 | +0.0000 | 0.4911 | 1.0000 | — |
| msmarco | test | recall@100 | budget | S3 selector | mean 10%-90% | 0.8743 | 0.8741 | -0.0002 | -0.0006 | +0.0003 | 0.4685 | 1.0000 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | S3 selector | mean 10%-90% | 0.3577 | 0.3592 | +0.0014 | -0.0028 | +0.0059 | 0.5270 | 1.0000 | see note: nfcorpus |
| scifact | test | recall@100 | budget | S3 selector | mean 10%-90% | 0.9852 | 0.9856 | +0.0003 | -0.0015 | +0.0022 | 0.8080 | 1.0000 | — |
| msmarco | test | recall@100 | budget | S3 selector | k = 307 | 0.8723 | 0.8734 | +0.0011 | -0.0002 | +0.0023 | 0.1107 | 0.3321 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | S3 selector | k = 307 | 0.3727 | 0.3669 | -0.0058 | -0.0182 | +0.0054 | 0.3360 | 0.6719 | see note: nfcorpus |
| scifact | test | recall@100 | budget | S3 selector | k = 307 | 0.9878 | 0.9856 | -0.0022 | -0.0056 | +0.0000 | 0.4911 | 0.6719 | — |
| msmarco | test | recall@100 | budget | S2 selector | mean 10%-90% | 0.8743 | 0.8741 | -0.0002 | -0.0006 | +0.0003 | 0.4685 | 0.9369 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | S2 selector | mean 10%-90% | 0.3508 | 0.3592 | +0.0084 | -0.0002 | +0.0167 | 0.0525 | 0.1575 | see note: nfcorpus |
| scifact | test | recall@100 | budget | S2 selector | mean 10%-90% | 0.9838 | 0.9856 | +0.0017 | -0.0027 | +0.0065 | 0.5245 | 0.9369 | — |
| msmarco | test | recall@100 | budget | S2 selector | k = 307 | 0.8723 | 0.8734 | +0.0011 | -0.0002 | +0.0023 | 0.1107 | 0.3321 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | S2 selector | k = 307 | 0.3639 | 0.3669 | +0.0030 | -0.0124 | +0.0172 | 0.6977 | 1.0000 | see note: nfcorpus |
| scifact | test | recall@100 | budget | S2 selector | k = 307 | 0.9867 | 0.9856 | -0.0011 | -0.0100 | +0.0044 | 1.0000 | 1.0000 | — |
| msmarco | test | recall@100 | budget | variance | mean 10%-90% | 0.8743 | 0.8741 | -0.0002 | -0.0006 | +0.0003 | 0.4685 | 0.4685 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | variance | mean 10%-90% | 0.3257 | 0.3592 | +0.0335 | +0.0225 | +0.0449 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | recall@100 | budget | variance | mean 10%-90% | 0.9552 | 0.9856 | +0.0304 | +0.0140 | +0.0491 | 0.0003 | 0.0006 | — |
| msmarco | test | recall@100 | budget | variance | k = 307 | 0.8723 | 0.8734 | +0.0011 | -0.0002 | +0.0023 | 0.1107 | 0.1107 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | variance | k = 307 | 0.3222 | 0.3669 | +0.0447 | +0.0285 | +0.0604 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | test | recall@100 | budget | variance | k = 307 | 0.9500 | 0.9856 | +0.0356 | +0.0133 | +0.0589 | 0.0040 | 0.0080 | — |
| msmarco | test | recall@100 | budget | S1 selector | mean 10%-90% | 0.8636 | 0.8741 | +0.0105 | +0.0083 | +0.0128 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | S1 selector | mean 10%-90% | 0.3492 | 0.3592 | +0.0100 | +0.0021 | +0.0180 | 0.0112 | 0.0224 | see note: nfcorpus |
| scifact | test | recall@100 | budget | S1 selector | mean 10%-90% | 0.9831 | 0.9856 | +0.0025 | -0.0010 | +0.0064 | 0.2251 | 0.2251 | — |
| msmarco | test | recall@100 | budget | S1 selector | k = 307 | 0.8649 | 0.8734 | +0.0084 | +0.0045 | +0.0124 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | S1 selector | k = 307 | 0.3604 | 0.3669 | +0.0065 | -0.0104 | +0.0221 | 0.4478 | 0.8069 | see note: nfcorpus |
| scifact | test | recall@100 | budget | S1 selector | k = 307 | 0.9800 | 0.9856 | +0.0056 | -0.0033 | +0.0167 | 0.4035 | 0.8069 | — |
| msmarco | test | recall@100 | budget | L2S | mean 10%-90% | 0.8595 | 0.8741 | +0.0146 | +0.0120 | +0.0172 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | L2S | mean 10%-90% | 0.3492 | 0.3592 | +0.0100 | +0.0021 | +0.0180 | 0.0112 | 0.0134 | see note: nfcorpus |
| scifact | test | recall@100 | budget | L2S | mean 10%-90% | 0.9821 | 0.9856 | +0.0034 | +0.0010 | +0.0061 | 0.0067 | 0.0134 | — |
| msmarco | test | recall@100 | budget | L2S | k = 307 | 0.8598 | 0.8734 | +0.0136 | +0.0092 | +0.0181 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | recall@100 | budget | L2S | k = 307 | 0.3604 | 0.3669 | +0.0065 | -0.0104 | +0.0221 | 0.4478 | 0.8955 | see note: nfcorpus |
| scifact | test | recall@100 | budget | L2S | k = 307 | 0.9844 | 0.9856 | +0.0011 | -0.0033 | +0.0056 | 1.0000 | 1.0000 | — |
| msmarco | test | recall@100 | weighted-aligned | S3 selector | mean 10%-90% | 0.8743 | 0.8742 | -0.0001 | -0.0004 | +0.0003 | 0.6728 | 1.0000 | see note: msmarco |
| nfcorpus | test | recall@100 | weighted-aligned | S3 selector | mean 10%-90% | 0.3577 | 0.3557 | -0.0020 | -0.0055 | +0.0011 | 0.2321 | 0.6962 | see note: nfcorpus |
| scifact | test | recall@100 | weighted-aligned | S3 selector | mean 10%-90% | 0.9852 | 0.9855 | +0.0002 | -0.0005 | +0.0011 | 0.7606 | 1.0000 | — |
| msmarco | test | recall@100 | weighted-aligned | S3 selector | k = 307 | 0.8723 | 0.8732 | +0.0009 | -0.0002 | +0.0020 | 0.1030 | 0.3090 | see note: msmarco |
| nfcorpus | test | recall@100 | weighted-aligned | S3 selector | k = 307 | 0.3727 | 0.3707 | -0.0020 | -0.0097 | +0.0049 | 0.6040 | 1.0000 | see note: nfcorpus |
| scifact | test | recall@100 | weighted-aligned | S3 selector | k = 307 | 0.9878 | 0.9889 | +0.0011 | +0.0000 | +0.0033 | 1.0000 | 1.0000 | — |

## By budget: S4 against aligned S3

Holm over the budgets of each environment (secondary).

| test | k | weighted-aligned | budget | diff | ci_low | ci_high | p | p_holm |
|---|---|---|---|---|---|---|---|---|
| msmarco | 102 | 0.3583 | 0.3584 | +0.0001 | -0.0009 | +0.0011 | 0.8238 | 1.0000 |
| msmarco | 204 | 0.3725 | 0.3725 | -0.0000 | -0.0007 | +0.0007 | 0.9599 | 1.0000 |
| msmarco | 307 | 0.3756 | 0.3761 | +0.0005 | -0.0001 | +0.0012 | 0.0788 | 0.6303 |
| msmarco | 409 | 0.3779 | 0.3781 | +0.0002 | -0.0003 | +0.0006 | 0.5036 | 1.0000 |
| msmarco | 512 | 0.3793 | 0.3792 | -0.0000 | -0.0004 | +0.0004 | 0.9435 | 1.0000 |
| msmarco | 614 | 0.3804 | 0.3802 | -0.0002 | -0.0006 | +0.0001 | 0.1824 | 1.0000 |
| msmarco | 716 | 0.3797 | 0.3800 | +0.0003 | +0.0001 | +0.0006 | 0.0145 | 0.1305 |
| msmarco | 819 | 0.3802 | 0.3801 | -0.0000 | -0.0002 | +0.0002 | 0.7239 | 1.0000 |
| msmarco | 921 | 0.3804 | 0.3803 | -0.0000 | -0.0002 | +0.0001 | 0.5275 | 1.0000 |
| nfcorpus | 102 | 0.3211 | 0.3566 | +0.0355 | +0.0256 | +0.0457 | 0.0001 | 0.0009 |
| nfcorpus | 204 | 0.3535 | 0.3751 | +0.0217 | +0.0127 | +0.0307 | 0.0001 | 0.0009 |
| nfcorpus | 307 | 0.3642 | 0.3799 | +0.0157 | +0.0092 | +0.0227 | 0.0001 | 0.0009 |
| nfcorpus | 409 | 0.3744 | 0.3840 | +0.0096 | +0.0042 | +0.0152 | 0.0005 | 0.0030 |
| nfcorpus | 512 | 0.3799 | 0.3821 | +0.0022 | -0.0016 | +0.0061 | 0.2567 | 0.5133 |
| nfcorpus | 614 | 0.3786 | 0.3834 | +0.0048 | +0.0006 | +0.0090 | 0.0255 | 0.1055 |
| nfcorpus | 716 | 0.3784 | 0.3830 | +0.0046 | -0.0002 | +0.0094 | 0.0646 | 0.1938 |
| nfcorpus | 819 | 0.3762 | 0.3777 | +0.0015 | -0.0028 | +0.0056 | 0.4950 | 0.5133 |
| nfcorpus | 921 | 0.3688 | 0.3731 | +0.0043 | +0.0008 | +0.0083 | 0.0211 | 0.1055 |
| scifact | 102 | 0.8303 | 0.8515 | +0.0212 | +0.0094 | +0.0336 | 0.0005 | 0.0040 |
| scifact | 204 | 0.8479 | 0.8560 | +0.0080 | -0.0019 | +0.0182 | 0.1214 | 0.4856 |
| scifact | 307 | 0.8500 | 0.8522 | +0.0022 | -0.0053 | +0.0097 | 0.5616 | 1.0000 |
| scifact | 409 | 0.8485 | 0.8469 | -0.0016 | -0.0091 | +0.0061 | 0.6828 | 1.0000 |
| scifact | 512 | 0.8462 | 0.8509 | +0.0047 | -0.0019 | +0.0114 | 0.1770 | 0.5309 |
| scifact | 614 | 0.8422 | 0.8516 | +0.0094 | +0.0024 | +0.0167 | 0.0111 | 0.0777 |
| scifact | 716 | 0.8359 | 0.8462 | +0.0104 | +0.0007 | +0.0200 | 0.0370 | 0.1850 |
| scifact | 819 | 0.8255 | 0.8350 | +0.0095 | +0.0017 | +0.0177 | 0.0215 | 0.1290 |
| scifact | 921 | 0.7811 | 0.8017 | +0.0206 | +0.0106 | +0.0312 | 0.0001 | 0.0009 |

## The learned prior weights

At each seed's kept checkpoint; the control's one weight applies at every budget.

| environment | track | seed | kept step | a at k=102 | a at k=204 | a at k=307 | a at k=409 | a at k=512 | a at k=614 | a at k=716 | a at k=819 | a at k=921 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | weighted-aligned | 0 | 3000 | 0.6384 | 0.6384 | 0.6384 | 0.6384 | 0.6384 | 0.6384 | 0.6384 | 0.6384 | 0.6384 |
| msmarco | weighted-aligned | 1 | 1000 | 0.9160 | 0.9160 | 0.9160 | 0.9160 | 0.9160 | 0.9160 | 0.9160 | 0.9160 | 0.9160 |
| msmarco | weighted-aligned | 2 | 1000 | 0.9119 | 0.9119 | 0.9119 | 0.9119 | 0.9119 | 0.9119 | 0.9119 | 0.9119 | 0.9119 |
| msmarco | budget | 0 | 4000 | 0.8824 | 0.8585 | 0.8609 | 0.8397 | 0.8425 | 0.8281 | 0.8345 | 0.8330 | 0.8421 |
| msmarco | budget | 1 | 1000 | 0.9979 | 0.9825 | 0.9734 | 0.9632 | 0.9665 | 0.9685 | 0.9639 | 0.9626 | 0.9670 |
| msmarco | budget | 2 | 10000 | 0.7560 | 0.7027 | 0.6769 | 0.6738 | 0.6687 | 0.6690 | 0.6677 | 0.6717 | 0.6962 |
| nfcorpus | weighted-aligned | 0 | 7000 | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0029 |
| nfcorpus | weighted-aligned | 1 | 14000 | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0022 |
| nfcorpus | weighted-aligned | 2 | 4000 | 0.0055 | 0.0055 | 0.0055 | 0.0055 | 0.0055 | 0.0055 | 0.0055 | 0.0055 | 0.0055 |
| nfcorpus | budget | 0 | 8000 | 0.0652 | 0.0589 | 0.0465 | 0.0291 | 0.0154 | 0.0016 | 0.0000 | 0.0000 | 0.0000 |
| nfcorpus | budget | 1 | 12000 | 0.0613 | 0.0646 | 0.0491 | 0.0312 | 0.0157 | 0.0040 | 0.0000 | 0.0000 | 0.0000 |
| nfcorpus | budget | 2 | 10000 | 0.0684 | 0.0635 | 0.0495 | 0.0307 | 0.0163 | 0.0038 | 0.0000 | 0.0000 | 0.0000 |
| scifact | weighted-aligned | 0 | 17000 | 0.0186 | 0.0186 | 0.0186 | 0.0186 | 0.0186 | 0.0186 | 0.0186 | 0.0186 | 0.0186 |
| scifact | weighted-aligned | 1 | 7000 | 0.0183 | 0.0183 | 0.0183 | 0.0183 | 0.0183 | 0.0183 | 0.0183 | 0.0183 | 0.0183 |
| scifact | weighted-aligned | 2 | 13000 | 0.0204 | 0.0204 | 0.0204 | 0.0204 | 0.0204 | 0.0204 | 0.0204 | 0.0204 | 0.0204 |
| scifact | budget | 0 | 13000 | 0.1305 | 0.1262 | 0.0854 | 0.0625 | 0.0427 | 0.0343 | 0.0184 | 0.0000 | 0.0000 |
| scifact | budget | 1 | 15000 | 0.1330 | 0.1263 | 0.0761 | 0.0532 | 0.0407 | 0.0290 | 0.0114 | 0.0000 | 0.0000 |
| scifact | budget | 2 | 12000 | 0.1359 | 0.1314 | 0.0885 | 0.0712 | 0.0469 | 0.0358 | 0.0191 | 0.0000 | 0.0000 |

## How much the prior still decides (validation queries)

Prior share `σ(a_k π) / (σ(a_k π) + σ(W q + b))` per query, unchanged when `a`, `W`, `b`
are scaled together; top-k overlaps with `variance` and with the correction alone.

| environment | track | budget | prior share | overlap with variance | overlap with the correction alone | mean prior weight |
|---|---|---|---|---|---|---|
| msmarco | weighted-aligned | 102 | 0.9791 | 0.9703 | 0.1953 | 0.8221 |
| msmarco | weighted-aligned | 204 | 0.9791 | 0.9808 | 0.2748 | 0.8221 |
| msmarco | weighted-aligned | 307 | 0.9791 | 0.9863 | 0.3581 | 0.8221 |
| msmarco | weighted-aligned | 409 | 0.9791 | 0.9899 | 0.4449 | 0.8221 |
| msmarco | weighted-aligned | 512 | 0.9791 | 0.9925 | 0.5344 | 0.8221 |
| msmarco | weighted-aligned | 614 | 0.9791 | 0.9945 | 0.6243 | 0.8221 |
| msmarco | weighted-aligned | 716 | 0.9791 | 0.9962 | 0.7154 | 0.8221 |
| msmarco | weighted-aligned | 819 | 0.9791 | 0.9977 | 0.8089 | 0.8221 |
| msmarco | weighted-aligned | 921 | 0.9791 | 0.9989 | 0.9040 | 0.8221 |
| msmarco | budget | 102 | 0.9546 | 0.9367 | 0.2169 | 0.8788 |
| msmarco | budget | 204 | 0.9522 | 0.9557 | 0.2979 | 0.8479 |
| msmarco | budget | 307 | 0.9512 | 0.9675 | 0.3799 | 0.8371 |
| msmarco | budget | 409 | 0.9507 | 0.9754 | 0.4638 | 0.8256 |
| msmarco | budget | 512 | 0.9505 | 0.9816 | 0.5498 | 0.8259 |
| msmarco | budget | 614 | 0.9503 | 0.9864 | 0.6364 | 0.8219 |
| msmarco | budget | 716 | 0.9503 | 0.9906 | 0.7241 | 0.8220 |
| msmarco | budget | 819 | 0.9505 | 0.9943 | 0.8137 | 0.8225 |
| msmarco | budget | 921 | 0.9517 | 0.9974 | 0.9054 | 0.8351 |
| nfcorpus | weighted-aligned | 102 | 0.1070 | 0.1787 | 0.9297 | 0.0035 |
| nfcorpus | weighted-aligned | 204 | 0.1070 | 0.2729 | 0.9404 | 0.0035 |
| nfcorpus | weighted-aligned | 307 | 0.1070 | 0.3623 | 0.9479 | 0.0035 |
| nfcorpus | weighted-aligned | 409 | 0.1070 | 0.4488 | 0.9555 | 0.0035 |
| nfcorpus | weighted-aligned | 512 | 0.1070 | 0.5374 | 0.9620 | 0.0035 |
| nfcorpus | weighted-aligned | 614 | 0.1070 | 0.6257 | 0.9691 | 0.0035 |
| nfcorpus | weighted-aligned | 716 | 0.1070 | 0.7163 | 0.9755 | 0.0035 |
| nfcorpus | weighted-aligned | 819 | 0.1070 | 0.8094 | 0.9826 | 0.0035 |
| nfcorpus | weighted-aligned | 921 | 0.1070 | 0.9031 | 0.9904 | 0.0035 |
| nfcorpus | budget | 102 | 0.5008 | 0.2942 | 0.5743 | 0.0650 |
| nfcorpus | budget | 204 | 0.4905 | 0.4162 | 0.6368 | 0.0623 |
| nfcorpus | budget | 307 | 0.4281 | 0.4764 | 0.7308 | 0.0484 |
| nfcorpus | budget | 409 | 0.3198 | 0.5013 | 0.8369 | 0.0303 |
| nfcorpus | budget | 512 | 0.1971 | 0.5321 | 0.9263 | 0.0158 |
| nfcorpus | budget | 614 | 0.0467 | 0.5800 | 0.9879 | 0.0031 |
| nfcorpus | budget | 716 | 0.0000 | 0.6769 | 1.0000 | 0.0000 |
| nfcorpus | budget | 819 | 0.0000 | 0.7878 | 1.0000 | 0.0000 |
| nfcorpus | budget | 921 | 0.0000 | 0.8956 | 1.0000 | 0.0000 |
| scifact | weighted-aligned | 102 | 0.2367 | 0.2938 | 0.8624 | 0.0191 |
| scifact | weighted-aligned | 204 | 0.2367 | 0.3807 | 0.8680 | 0.0191 |
| scifact | weighted-aligned | 307 | 0.2367 | 0.4558 | 0.8731 | 0.0191 |
| scifact | weighted-aligned | 409 | 0.2367 | 0.5255 | 0.8853 | 0.0191 |
| scifact | weighted-aligned | 512 | 0.2367 | 0.5999 | 0.9013 | 0.0191 |
| scifact | weighted-aligned | 614 | 0.2367 | 0.6765 | 0.9167 | 0.0191 |
| scifact | weighted-aligned | 716 | 0.2367 | 0.7529 | 0.9347 | 0.0191 |
| scifact | weighted-aligned | 819 | 0.2367 | 0.8341 | 0.9530 | 0.0191 |
| scifact | weighted-aligned | 921 | 0.2367 | 0.9150 | 0.9740 | 0.0191 |
| scifact | budget | 102 | 0.4712 | 0.3588 | 0.6578 | 0.1331 |
| scifact | budget | 204 | 0.4614 | 0.4628 | 0.7008 | 0.1280 |
| scifact | budget | 307 | 0.3580 | 0.4857 | 0.7931 | 0.0833 |
| scifact | budget | 409 | 0.2937 | 0.5255 | 0.8545 | 0.0623 |
| scifact | budget | 512 | 0.2255 | 0.5699 | 0.9109 | 0.0434 |
| scifact | budget | 614 | 0.1811 | 0.6348 | 0.9444 | 0.0330 |
| scifact | budget | 716 | 0.0981 | 0.6997 | 0.9785 | 0.0163 |
| scifact | budget | 819 | 0.0000 | 0.7882 | 1.0000 | 0.0000 |
| scifact | budget | 921 | 0.0000 | 0.8955 | 1.0000 | 0.0000 |

Validation nDCG@10 with every prior weight set to 0 (context, not a criterion):

| environment | track | budget | kept | prior removed | removed - kept |
|---|---|---|---|---|---|
| msmarco | weighted-aligned | 102 | 0.3266 | 0.1238 | -0.2028 |
| msmarco | weighted-aligned | 204 | 0.3366 | 0.2596 | -0.0771 |
| msmarco | weighted-aligned | 307 | 0.3401 | 0.3013 | -0.0388 |
| msmarco | weighted-aligned | 409 | 0.3410 | 0.3205 | -0.0205 |
| msmarco | weighted-aligned | 512 | 0.3414 | 0.3310 | -0.0104 |
| msmarco | weighted-aligned | 614 | 0.3419 | 0.3364 | -0.0055 |
| msmarco | weighted-aligned | 716 | 0.3417 | 0.3415 | -0.0002 |
| msmarco | weighted-aligned | 819 | 0.3417 | 0.3428 | +0.0011 |
| msmarco | weighted-aligned | 921 | 0.3415 | 0.3430 | +0.0015 |
| msmarco | budget | 102 | 0.3257 | 0.1250 | -0.2007 |
| msmarco | budget | 204 | 0.3367 | 0.2607 | -0.0760 |
| msmarco | budget | 307 | 0.3406 | 0.3016 | -0.0390 |
| msmarco | budget | 409 | 0.3412 | 0.3209 | -0.0203 |
| msmarco | budget | 512 | 0.3412 | 0.3316 | -0.0096 |
| msmarco | budget | 614 | 0.3417 | 0.3365 | -0.0051 |
| msmarco | budget | 716 | 0.3417 | 0.3404 | -0.0013 |
| msmarco | budget | 819 | 0.3420 | 0.3432 | +0.0013 |
| msmarco | budget | 921 | 0.3415 | 0.3431 | +0.0016 |
| nfcorpus | weighted-aligned | 102 | 0.3208 | 0.3136 | -0.0072 |
| nfcorpus | weighted-aligned | 204 | 0.3496 | 0.3451 | -0.0045 |
| nfcorpus | weighted-aligned | 307 | 0.3655 | 0.3589 | -0.0066 |
| nfcorpus | weighted-aligned | 409 | 0.3696 | 0.3690 | -0.0005 |
| nfcorpus | weighted-aligned | 512 | 0.3663 | 0.3661 | -0.0002 |
| nfcorpus | weighted-aligned | 614 | 0.3653 | 0.3650 | -0.0003 |
| nfcorpus | weighted-aligned | 716 | 0.3627 | 0.3635 | +0.0008 |
| nfcorpus | weighted-aligned | 819 | 0.3614 | 0.3609 | -0.0005 |
| nfcorpus | weighted-aligned | 921 | 0.3559 | 0.3578 | +0.0019 |
| nfcorpus | budget | 102 | 0.3516 | 0.2927 | -0.0589 |
| nfcorpus | budget | 204 | 0.3661 | 0.3384 | -0.0278 |
| nfcorpus | budget | 307 | 0.3693 | 0.3579 | -0.0114 |
| nfcorpus | budget | 409 | 0.3666 | 0.3666 | -0.0000 |
| nfcorpus | budget | 512 | 0.3653 | 0.3670 | +0.0017 |
| nfcorpus | budget | 614 | 0.3638 | 0.3649 | +0.0011 |
| nfcorpus | budget | 716 | 0.3648 | 0.3648 | +0.0000 |
| nfcorpus | budget | 819 | 0.3615 | 0.3615 | +0.0000 |
| nfcorpus | budget | 921 | 0.3594 | 0.3594 | +0.0000 |
| scifact | weighted-aligned | 102 | 0.8195 | 0.8043 | -0.0151 |
| scifact | weighted-aligned | 204 | 0.8350 | 0.8199 | -0.0151 |
| scifact | weighted-aligned | 307 | 0.8358 | 0.8273 | -0.0084 |
| scifact | weighted-aligned | 409 | 0.8397 | 0.8297 | -0.0100 |
| scifact | weighted-aligned | 512 | 0.8513 | 0.8399 | -0.0114 |
| scifact | weighted-aligned | 614 | 0.8406 | 0.8400 | -0.0006 |
| scifact | weighted-aligned | 716 | 0.8352 | 0.8403 | +0.0051 |
| scifact | weighted-aligned | 819 | 0.8250 | 0.8316 | +0.0066 |
| scifact | weighted-aligned | 921 | 0.8015 | 0.8215 | +0.0200 |
| scifact | budget | 102 | 0.8245 | 0.7931 | -0.0315 |
| scifact | budget | 204 | 0.8284 | 0.8364 | +0.0080 |
| scifact | budget | 307 | 0.8457 | 0.8416 | -0.0041 |
| scifact | budget | 409 | 0.8589 | 0.8401 | -0.0188 |
| scifact | budget | 512 | 0.8509 | 0.8443 | -0.0066 |
| scifact | budget | 614 | 0.8461 | 0.8425 | -0.0036 |
| scifact | budget | 716 | 0.8415 | 0.8403 | -0.0011 |
| scifact | budget | 819 | 0.8347 | 0.8347 | +0.0000 |
| scifact | budget | 921 | 0.8124 | 0.8124 | +0.0000 |

## Kept checkpoints

| environment | track | seed | kept step | rerank-best step | subset nDCG@10 at kept step | soft mass at final temperature |
|---|---|---|---|---|---|---|
| msmarco | weighted-aligned | 0 | 3000 | 0 | 0.3550 | 0.9139 |
| msmarco | weighted-aligned | 1 | 1000 | 0 | 0.3550 | 0.9136 |
| msmarco | weighted-aligned | 2 | 1000 | 0 | 0.3546 | 0.9136 |
| msmarco | budget | 0 | 4000 | 0 | 0.3549 | 0.9138 |
| msmarco | budget | 1 | 1000 | 0 | 0.3549 | 0.9136 |
| msmarco | budget | 2 | 10000 | 0 | 0.3548 | 0.9148 |
| nfcorpus | weighted-aligned | 0 | 7000 | 7000 | 0.3584 | 0.9373 |
| nfcorpus | weighted-aligned | 1 | 14000 | 5000 | 0.3563 | 0.9378 |
| nfcorpus | weighted-aligned | 2 | 4000 | 1000 | 0.3576 | 0.9384 |
| nfcorpus | budget | 0 | 8000 | 8000 | 0.3636 | 0.9868 |
| nfcorpus | budget | 1 | 12000 | 12000 | 0.3623 | 0.9866 |
| nfcorpus | budget | 2 | 10000 | 11000 | 0.3636 | 0.9865 |
| scifact | weighted-aligned | 0 | 17000 | 17000 | 0.8323 | 0.9373 |
| scifact | weighted-aligned | 1 | 7000 | 5000 | 0.8326 | 0.9373 |
| scifact | weighted-aligned | 2 | 13000 | 12000 | 0.8297 | 0.9379 |
| scifact | budget | 0 | 13000 | 11000 | 0.8397 | 0.9370 |
| scifact | budget | 1 | 15000 | 8000 | 0.8382 | 0.9377 |
| scifact | budget | 2 | 12000 | 16000 | 0.8364 | 0.9372 |

## nDCG@10 by budget

| test | query set | method | 10% (k=102) | 20% (k=204) | 30% (k=307) | 40% (k=409) | 50% (k=512) | 60% (k=614) | 70% (k=716) | 80% (k=819) | 90% (k=921) | 100% (k=1024) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | test | full | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | — |
| msmarco | test | prefix | 0.2598 | 0.3180 | 0.3385 | 0.3528 | 0.3626 | 0.3702 | 0.3746 | 0.3777 | 0.3792 | 0.3800 | see note: msmarco |
| msmarco | test | random | 0.2397 | 0.3186 | 0.3425 | 0.3587 | 0.3643 | 0.3695 | 0.3746 | 0.3758 | 0.3778 | 0.3800 | see note: msmarco |
| msmarco | test | global-importance | 0.2435 | 0.3230 | 0.3524 | 0.3639 | 0.3714 | 0.3737 | 0.3774 | 0.3797 | 0.3798 | 0.3800 | see note: msmarco |
| msmarco | test | global-selector | 0.3064 | 0.3460 | 0.3591 | 0.3680 | 0.3742 | 0.3757 | 0.3767 | 0.3786 | 0.3797 | 0.3800 | see note: msmarco |
| msmarco | test | variance | 0.3584 | 0.3717 | 0.3765 | 0.3783 | 0.3791 | 0.3806 | 0.3792 | 0.3802 | 0.3804 | 0.3800 | see note: msmarco |
| msmarco | test | l2s:msmarco | 0.3047 | 0.3444 | 0.3572 | 0.3640 | 0.3702 | 0.3718 | 0.3752 | 0.3785 | 0.3799 | 0.3800 | see note: msmarco |
| msmarco | test | retrieval:msmarco | 0.3185 | 0.3516 | 0.3645 | 0.3704 | 0.3745 | 0.3762 | 0.3784 | 0.3790 | 0.3796 | 0.3800 | see note: msmarco |
| msmarco | test | geometric:msmarco | 0.3584 | 0.3717 | 0.3765 | 0.3783 | 0.3791 | 0.3806 | 0.3792 | 0.3802 | 0.3804 | 0.3800 | see note: msmarco |
| msmarco | test | weighted:msmarco | 0.3584 | 0.3717 | 0.3765 | 0.3783 | 0.3791 | 0.3806 | 0.3792 | 0.3802 | 0.3804 | 0.3800 | see note: msmarco |
| msmarco | test | weighted-aligned:msmarco | 0.3583 | 0.3725 | 0.3756 | 0.3779 | 0.3793 | 0.3804 | 0.3797 | 0.3802 | 0.3804 | 0.3800 | see note: msmarco |
| msmarco | test | budget:msmarco | 0.3584 | 0.3725 | 0.3761 | 0.3781 | 0.3792 | 0.3802 | 0.3800 | 0.3801 | 0.3803 | 0.3800 | see note: msmarco |
| msmarco | test | family:multi-budget-budget:msmarco | 0.3581 | 0.3730 | 0.3760 | 0.3776 | 0.3792 | 0.3804 | 0.3802 | 0.3799 | 0.3804 | 0.3800 | see note: msmarco |
| msmarco | trec-dl-2019 | full | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | — |
| msmarco | trec-dl-2019 | prefix | 0.5240 | 0.5803 | 0.5990 | 0.6270 | 0.6327 | 0.6466 | 0.6609 | 0.6620 | 0.6671 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | random | 0.4374 | 0.5820 | 0.6367 | 0.6472 | 0.6611 | 0.6559 | 0.6553 | 0.6563 | 0.6600 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | global-importance | 0.4592 | 0.5850 | 0.6165 | 0.6387 | 0.6522 | 0.6664 | 0.6583 | 0.6606 | 0.6589 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | global-selector | 0.5762 | 0.6410 | 0.6495 | 0.6472 | 0.6511 | 0.6648 | 0.6645 | 0.6651 | 0.6632 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | variance | 0.6273 | 0.6451 | 0.6540 | 0.6605 | 0.6628 | 0.6671 | 0.6687 | 0.6680 | 0.6659 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | l2s:msmarco | 0.5829 | 0.6342 | 0.6348 | 0.6482 | 0.6574 | 0.6636 | 0.6638 | 0.6696 | 0.6685 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | retrieval:msmarco | 0.5736 | 0.6359 | 0.6349 | 0.6458 | 0.6690 | 0.6636 | 0.6649 | 0.6611 | 0.6627 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | geometric:msmarco | 0.6273 | 0.6451 | 0.6540 | 0.6605 | 0.6628 | 0.6671 | 0.6687 | 0.6680 | 0.6659 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | weighted:msmarco | 0.6273 | 0.6451 | 0.6540 | 0.6605 | 0.6628 | 0.6671 | 0.6687 | 0.6680 | 0.6659 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | weighted-aligned:msmarco | 0.6292 | 0.6395 | 0.6526 | 0.6574 | 0.6637 | 0.6715 | 0.6678 | 0.6677 | 0.6664 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | budget:msmarco | 0.6281 | 0.6444 | 0.6511 | 0.6567 | 0.6632 | 0.6699 | 0.6675 | 0.6674 | 0.6668 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | family:multi-budget-budget:msmarco | 0.6311 | 0.6388 | 0.6499 | 0.6572 | 0.6643 | 0.6706 | 0.6675 | 0.6671 | 0.6673 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2020 | full | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | — |
| msmarco | trec-dl-2020 | prefix | 0.4900 | 0.5717 | 0.6234 | 0.6497 | 0.6525 | 0.6595 | 0.6616 | 0.6591 | 0.6678 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | random | 0.4936 | 0.6049 | 0.6478 | 0.6658 | 0.6668 | 0.6785 | 0.6732 | 0.6736 | 0.6687 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | global-importance | 0.4598 | 0.5977 | 0.6450 | 0.6580 | 0.6697 | 0.6714 | 0.6742 | 0.6747 | 0.6767 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | global-selector | 0.5698 | 0.6370 | 0.6507 | 0.6566 | 0.6661 | 0.6734 | 0.6757 | 0.6715 | 0.6769 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | variance | 0.6488 | 0.6673 | 0.6738 | 0.6723 | 0.6702 | 0.6732 | 0.6751 | 0.6733 | 0.6756 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | l2s:msmarco | 0.5939 | 0.6347 | 0.6534 | 0.6550 | 0.6590 | 0.6680 | 0.6721 | 0.6778 | 0.6803 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | retrieval:msmarco | 0.6058 | 0.6497 | 0.6522 | 0.6604 | 0.6650 | 0.6712 | 0.6733 | 0.6728 | 0.6791 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | geometric:msmarco | 0.6488 | 0.6673 | 0.6738 | 0.6723 | 0.6702 | 0.6732 | 0.6751 | 0.6733 | 0.6756 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | weighted:msmarco | 0.6488 | 0.6673 | 0.6738 | 0.6723 | 0.6702 | 0.6732 | 0.6751 | 0.6733 | 0.6756 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | weighted-aligned:msmarco | 0.6481 | 0.6688 | 0.6737 | 0.6734 | 0.6704 | 0.6730 | 0.6747 | 0.6732 | 0.6756 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | budget:msmarco | 0.6497 | 0.6689 | 0.6746 | 0.6738 | 0.6707 | 0.6731 | 0.6748 | 0.6737 | 0.6756 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | family:multi-budget-budget:msmarco | 0.6526 | 0.6712 | 0.6754 | 0.6740 | 0.6707 | 0.6731 | 0.6760 | 0.6726 | 0.6756 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-hard | full | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | — |
| msmarco | trec-dl-hard | prefix | 0.2813 | 0.2999 | 0.3363 | 0.3404 | 0.3435 | 0.3508 | 0.3586 | 0.3600 | 0.3707 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | random | 0.2088 | 0.2979 | 0.3385 | 0.3441 | 0.3543 | 0.3655 | 0.3734 | 0.3726 | 0.3678 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | global-importance | 0.2490 | 0.3157 | 0.3420 | 0.3469 | 0.3563 | 0.3610 | 0.3689 | 0.3704 | 0.3681 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | global-selector | 0.3146 | 0.3368 | 0.3462 | 0.3477 | 0.3557 | 0.3611 | 0.3577 | 0.3686 | 0.3700 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | variance | 0.3558 | 0.3724 | 0.3697 | 0.3700 | 0.3709 | 0.3698 | 0.3690 | 0.3692 | 0.3683 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | l2s:msmarco | 0.3252 | 0.3484 | 0.3509 | 0.3594 | 0.3772 | 0.3803 | 0.3781 | 0.3835 | 0.3729 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | retrieval:msmarco | 0.3009 | 0.3412 | 0.3478 | 0.3615 | 0.3663 | 0.3677 | 0.3750 | 0.3706 | 0.3724 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | geometric:msmarco | 0.3558 | 0.3724 | 0.3697 | 0.3700 | 0.3709 | 0.3698 | 0.3690 | 0.3692 | 0.3683 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | weighted:msmarco | 0.3558 | 0.3724 | 0.3697 | 0.3700 | 0.3709 | 0.3698 | 0.3690 | 0.3692 | 0.3683 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | weighted-aligned:msmarco | 0.3551 | 0.3665 | 0.3683 | 0.3689 | 0.3687 | 0.3702 | 0.3700 | 0.3696 | 0.3683 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | budget:msmarco | 0.3556 | 0.3668 | 0.3682 | 0.3677 | 0.3728 | 0.3692 | 0.3706 | 0.3697 | 0.3683 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | family:multi-budget-budget:msmarco | 0.3477 | 0.3624 | 0.3689 | 0.3674 | 0.3727 | 0.3700 | 0.3693 | 0.3687 | 0.3683 | 0.3681 | see note: msmarco |
| nfcorpus | test | full | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | — |
| nfcorpus | test | prefix | 0.2716 | 0.3165 | 0.3263 | 0.3394 | 0.3459 | 0.3492 | 0.3555 | 0.3609 | 0.3617 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | random | 0.2400 | 0.2987 | 0.3183 | 0.3345 | 0.3488 | 0.3546 | 0.3582 | 0.3614 | 0.3633 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | global-importance | 0.2471 | 0.3091 | 0.3358 | 0.3371 | 0.3501 | 0.3511 | 0.3540 | 0.3559 | 0.3620 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | global-selector | 0.3075 | 0.3304 | 0.3436 | 0.3495 | 0.3550 | 0.3555 | 0.3581 | 0.3572 | 0.3590 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | variance | 0.3443 | 0.3560 | 0.3596 | 0.3590 | 0.3603 | 0.3629 | 0.3630 | 0.3621 | 0.3615 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | l2s:nfcorpus | 0.3132 | 0.3480 | 0.3607 | 0.3682 | 0.3736 | 0.3747 | 0.3751 | 0.3751 | 0.3709 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | retrieval:nfcorpus | 0.3132 | 0.3480 | 0.3607 | 0.3682 | 0.3736 | 0.3747 | 0.3751 | 0.3751 | 0.3709 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | geometric:nfcorpus | 0.3540 | 0.3784 | 0.3752 | 0.3797 | 0.3787 | 0.3768 | 0.3749 | 0.3702 | 0.3644 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | weighted:nfcorpus | 0.3166 | 0.3535 | 0.3681 | 0.3769 | 0.3806 | 0.3823 | 0.3798 | 0.3779 | 0.3706 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | weighted-aligned:nfcorpus | 0.3211 | 0.3535 | 0.3642 | 0.3744 | 0.3799 | 0.3786 | 0.3784 | 0.3762 | 0.3688 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | budget:nfcorpus | 0.3566 | 0.3751 | 0.3799 | 0.3840 | 0.3821 | 0.3834 | 0.3830 | 0.3777 | 0.3731 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | family:multi-budget-budget:nfcorpus | 0.3567 | 0.3747 | 0.3782 | 0.3862 | 0.3852 | 0.3854 | 0.3843 | 0.3787 | 0.3736 | 0.3619 | see note: nfcorpus |
| scifact | test | full | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | — |
| scifact | test | prefix | 0.5520 | 0.6387 | 0.6640 | 0.6672 | 0.6895 | 0.6915 | 0.6921 | 0.6937 | 0.6974 | 0.7000 | — |
| scifact | test | random | 0.5285 | 0.6379 | 0.6556 | 0.6708 | 0.6838 | 0.6914 | 0.6871 | 0.6926 | 0.7012 | 0.7000 | — |
| scifact | test | global-importance | 0.5559 | 0.6671 | 0.6922 | 0.6988 | 0.7049 | 0.7055 | 0.7090 | 0.7037 | 0.7019 | 0.7000 | — |
| scifact | test | global-selector | 0.6735 | 0.7099 | 0.7238 | 0.7256 | 0.7252 | 0.7146 | 0.7089 | 0.7070 | 0.7061 | 0.7000 | — |
| scifact | test | variance | 0.6721 | 0.7011 | 0.6973 | 0.6975 | 0.6971 | 0.6978 | 0.7008 | 0.7027 | 0.7006 | 0.7000 | — |
| scifact | test | l2s:scifact | 0.8021 | 0.8270 | 0.8353 | 0.8369 | 0.8312 | 0.8247 | 0.8142 | 0.8033 | 0.7749 | 0.7000 | — |
| scifact | test | retrieval:scifact | 0.8060 | 0.8410 | 0.8458 | 0.8473 | 0.8453 | 0.8374 | 0.8301 | 0.8171 | 0.7842 | 0.7000 | — |
| scifact | test | geometric:scifact | 0.8358 | 0.8401 | 0.8390 | 0.8251 | 0.8149 | 0.7960 | 0.7684 | 0.7442 | 0.7093 | 0.7000 | — |
| scifact | test | weighted:scifact | 0.8288 | 0.8475 | 0.8517 | 0.8493 | 0.8458 | 0.8424 | 0.8350 | 0.8243 | 0.7807 | 0.7000 | — |
| scifact | test | weighted-aligned:scifact | 0.8303 | 0.8479 | 0.8500 | 0.8485 | 0.8462 | 0.8422 | 0.8359 | 0.8255 | 0.7811 | 0.7000 | — |
| scifact | test | budget:scifact | 0.8515 | 0.8560 | 0.8522 | 0.8469 | 0.8509 | 0.8516 | 0.8462 | 0.8350 | 0.8017 | 0.7000 | — |
| scifact | test | family:multi-budget-budget:scifact | 0.8536 | 0.8578 | 0.8503 | 0.8450 | 0.8534 | 0.8529 | 0.8451 | 0.8362 | 0.8014 | 0.7000 | — |

## Against full 1024

| test | query set | budget | full | variance - full | retrieval - full | geometric - full | weighted - full | weighted-aligned - full | budget - full |
|---|---|---|---|---|---|---|---|---|---|
| msmarco | test | 102 | 0.3800 | -0.0216 | -0.0616 | -0.0216 | -0.0216 | -0.0217 | -0.0216 |
| msmarco | test | 204 | 0.3800 | -0.0083 | -0.0284 | -0.0083 | -0.0083 | -0.0076 | -0.0076 |
| msmarco | test | 307 | 0.3800 | -0.0036 | -0.0155 | -0.0036 | -0.0036 | -0.0044 | -0.0039 |
| msmarco | test | 409 | 0.3800 | -0.0017 | -0.0097 | -0.0017 | -0.0017 | -0.0021 | -0.0020 |
| msmarco | test | 512 | 0.3800 | -0.0009 | -0.0056 | -0.0009 | -0.0009 | -0.0008 | -0.0008 |
| msmarco | test | 614 | 0.3800 | +0.0005 | -0.0038 | +0.0005 | +0.0005 | +0.0004 | +0.0002 |
| msmarco | test | 716 | 0.3800 | -0.0008 | -0.0017 | -0.0008 | -0.0008 | -0.0004 | -0.0000 |
| msmarco | test | 819 | 0.3800 | +0.0002 | -0.0010 | +0.0002 | +0.0002 | +0.0001 | +0.0001 |
| msmarco | test | 921 | 0.3800 | +0.0003 | -0.0004 | +0.0003 | +0.0003 | +0.0003 | +0.0003 |
| msmarco | trec-dl-2019 | 102 | 0.6654 | -0.0381 | -0.0918 | -0.0381 | -0.0381 | -0.0362 | -0.0373 |
| msmarco | trec-dl-2019 | 204 | 0.6654 | -0.0203 | -0.0295 | -0.0203 | -0.0203 | -0.0258 | -0.0210 |
| msmarco | trec-dl-2019 | 307 | 0.6654 | -0.0114 | -0.0305 | -0.0114 | -0.0114 | -0.0128 | -0.0143 |
| msmarco | trec-dl-2019 | 409 | 0.6654 | -0.0049 | -0.0196 | -0.0049 | -0.0049 | -0.0080 | -0.0087 |
| msmarco | trec-dl-2019 | 512 | 0.6654 | -0.0026 | +0.0036 | -0.0026 | -0.0026 | -0.0017 | -0.0022 |
| msmarco | trec-dl-2019 | 614 | 0.6654 | +0.0017 | -0.0018 | +0.0017 | +0.0017 | +0.0061 | +0.0045 |
| msmarco | trec-dl-2019 | 716 | 0.6654 | +0.0033 | -0.0005 | +0.0033 | +0.0033 | +0.0024 | +0.0021 |
| msmarco | trec-dl-2019 | 819 | 0.6654 | +0.0026 | -0.0043 | +0.0026 | +0.0026 | +0.0023 | +0.0021 |
| msmarco | trec-dl-2019 | 921 | 0.6654 | +0.0005 | -0.0027 | +0.0005 | +0.0005 | +0.0010 | +0.0014 |
| msmarco | trec-dl-2020 | 102 | 0.6767 | -0.0279 | -0.0708 | -0.0279 | -0.0279 | -0.0285 | -0.0270 |
| msmarco | trec-dl-2020 | 204 | 0.6767 | -0.0093 | -0.0269 | -0.0093 | -0.0093 | -0.0079 | -0.0078 |
| msmarco | trec-dl-2020 | 307 | 0.6767 | -0.0029 | -0.0245 | -0.0029 | -0.0029 | -0.0030 | -0.0021 |
| msmarco | trec-dl-2020 | 409 | 0.6767 | -0.0044 | -0.0162 | -0.0044 | -0.0044 | -0.0033 | -0.0029 |
| msmarco | trec-dl-2020 | 512 | 0.6767 | -0.0064 | -0.0117 | -0.0064 | -0.0064 | -0.0063 | -0.0060 |
| msmarco | trec-dl-2020 | 614 | 0.6767 | -0.0034 | -0.0054 | -0.0034 | -0.0034 | -0.0036 | -0.0036 |
| msmarco | trec-dl-2020 | 716 | 0.6767 | -0.0016 | -0.0034 | -0.0016 | -0.0016 | -0.0019 | -0.0018 |
| msmarco | trec-dl-2020 | 819 | 0.6767 | -0.0033 | -0.0039 | -0.0033 | -0.0033 | -0.0034 | -0.0029 |
| msmarco | trec-dl-2020 | 921 | 0.6767 | -0.0011 | +0.0024 | -0.0011 | -0.0011 | -0.0011 | -0.0011 |
| msmarco | trec-dl-hard | 102 | 0.3681 | -0.0124 | -0.0672 | -0.0124 | -0.0124 | -0.0130 | -0.0125 |
| msmarco | trec-dl-hard | 204 | 0.3681 | +0.0043 | -0.0269 | +0.0043 | +0.0043 | -0.0017 | -0.0013 |
| msmarco | trec-dl-hard | 307 | 0.3681 | +0.0016 | -0.0203 | +0.0016 | +0.0016 | +0.0001 | +0.0001 |
| msmarco | trec-dl-hard | 409 | 0.3681 | +0.0019 | -0.0067 | +0.0019 | +0.0019 | +0.0008 | -0.0004 |
| msmarco | trec-dl-hard | 512 | 0.3681 | +0.0028 | -0.0018 | +0.0028 | +0.0028 | +0.0006 | +0.0046 |
| msmarco | trec-dl-hard | 614 | 0.3681 | +0.0017 | -0.0004 | +0.0017 | +0.0017 | +0.0021 | +0.0011 |
| msmarco | trec-dl-hard | 716 | 0.3681 | +0.0009 | +0.0069 | +0.0009 | +0.0009 | +0.0019 | +0.0025 |
| msmarco | trec-dl-hard | 819 | 0.3681 | +0.0011 | +0.0024 | +0.0011 | +0.0011 | +0.0015 | +0.0016 |
| msmarco | trec-dl-hard | 921 | 0.3681 | +0.0002 | +0.0043 | +0.0002 | +0.0002 | +0.0002 | +0.0002 |
| nfcorpus | test | 102 | 0.3619 | -0.0176 | -0.0488 | -0.0079 | -0.0453 | -0.0408 | -0.0054 |
| nfcorpus | test | 204 | 0.3619 | -0.0060 | -0.0140 | +0.0165 | -0.0084 | -0.0085 | +0.0132 |
| nfcorpus | test | 307 | 0.3619 | -0.0024 | -0.0012 | +0.0132 | +0.0062 | +0.0022 | +0.0179 |
| nfcorpus | test | 409 | 0.3619 | -0.0029 | +0.0063 | +0.0177 | +0.0149 | +0.0125 | +0.0221 |
| nfcorpus | test | 512 | 0.3619 | -0.0016 | +0.0117 | +0.0167 | +0.0187 | +0.0179 | +0.0202 |
| nfcorpus | test | 614 | 0.3619 | +0.0009 | +0.0127 | +0.0149 | +0.0204 | +0.0167 | +0.0214 |
| nfcorpus | test | 716 | 0.3619 | +0.0011 | +0.0132 | +0.0130 | +0.0179 | +0.0165 | +0.0210 |
| nfcorpus | test | 819 | 0.3619 | +0.0001 | +0.0132 | +0.0082 | +0.0160 | +0.0142 | +0.0158 |
| nfcorpus | test | 921 | 0.3619 | -0.0004 | +0.0090 | +0.0024 | +0.0086 | +0.0068 | +0.0112 |
| scifact | test | 102 | 0.7000 | -0.0278 | +0.1061 | +0.1359 | +0.1289 | +0.1303 | +0.1515 |
| scifact | test | 204 | 0.7000 | +0.0012 | +0.1411 | +0.1401 | +0.1475 | +0.1480 | +0.1560 |
| scifact | test | 307 | 0.7000 | -0.0026 | +0.1458 | +0.1390 | +0.1518 | +0.1501 | +0.1523 |
| scifact | test | 409 | 0.7000 | -0.0024 | +0.1473 | +0.1251 | +0.1494 | +0.1485 | +0.1469 |
| scifact | test | 512 | 0.7000 | -0.0028 | +0.1454 | +0.1150 | +0.1458 | +0.1462 | +0.1509 |
| scifact | test | 614 | 0.7000 | -0.0021 | +0.1375 | +0.0961 | +0.1424 | +0.1423 | +0.1516 |
| scifact | test | 716 | 0.7000 | +0.0009 | +0.1301 | +0.0684 | +0.1351 | +0.1359 | +0.1463 |
| scifact | test | 819 | 0.7000 | +0.0027 | +0.1171 | +0.0443 | +0.1244 | +0.1256 | +0.1351 |
| scifact | test | 921 | 0.7000 | +0.0006 | +0.0842 | +0.0093 | +0.0808 | +0.0812 | +0.1018 |

## TREC DL sets (context, not in the criteria)

| test | query set | metric | candidate | against | summary | reference | candidate value | diff | ci_low | ci_high | p | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | trec-dl-2019 | ndcg@10 | budget | aligned S3 | mean 10%-90% | 0.6573 | 0.6572 | -0.0001 | -0.0011 | +0.0010 | 0.8939 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | budget | aligned S3 | k = 307 | 0.6526 | 0.6511 | -0.0014 | -0.0053 | +0.0021 | 0.4734 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | budget | S3 selector | mean 10%-90% | 0.6577 | 0.6572 | -0.0005 | -0.0021 | +0.0014 | 0.6321 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | budget | S3 selector | k = 307 | 0.6540 | 0.6511 | -0.0029 | -0.0093 | +0.0027 | 0.3754 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | budget | S2 selector | mean 10%-90% | 0.6577 | 0.6572 | -0.0005 | -0.0021 | +0.0014 | 0.6321 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | budget | S2 selector | k = 307 | 0.6540 | 0.6511 | -0.0029 | -0.0093 | +0.0027 | 0.3754 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | budget | variance | mean 10%-90% | 0.6577 | 0.6572 | -0.0005 | -0.0021 | +0.0014 | 0.6321 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | budget | variance | k = 307 | 0.6540 | 0.6511 | -0.0029 | -0.0093 | +0.0027 | 0.3754 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | budget | S1 selector | mean 10%-90% | 0.6457 | 0.6572 | +0.0115 | -0.0010 | +0.0244 | 0.0864 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | budget | S1 selector | k = 307 | 0.6349 | 0.6511 | +0.0162 | -0.0031 | +0.0361 | 0.1097 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | weighted-aligned | S3 selector | mean 10%-90% | 0.6577 | 0.6573 | -0.0004 | -0.0015 | +0.0009 | 0.5440 | see note: msmarco |
| msmarco | trec-dl-2019 | ndcg@10 | weighted-aligned | S3 selector | k = 307 | 0.6540 | 0.6526 | -0.0015 | -0.0061 | +0.0034 | 0.5601 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | budget | aligned S3 | mean 10%-90% | 0.6701 | 0.6705 | +0.0004 | -0.0003 | +0.0013 | 0.2961 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | budget | aligned S3 | k = 307 | 0.6737 | 0.6746 | +0.0009 | -0.0018 | +0.0037 | 0.5379 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | budget | S3 selector | mean 10%-90% | 0.6700 | 0.6705 | +0.0006 | -0.0007 | +0.0019 | 0.3785 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | budget | S3 selector | k = 307 | 0.6738 | 0.6746 | +0.0008 | -0.0037 | +0.0054 | 0.7280 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | budget | S2 selector | mean 10%-90% | 0.6700 | 0.6705 | +0.0006 | -0.0007 | +0.0019 | 0.3785 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | budget | S2 selector | k = 307 | 0.6738 | 0.6746 | +0.0008 | -0.0037 | +0.0054 | 0.7280 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | budget | variance | mean 10%-90% | 0.6700 | 0.6705 | +0.0006 | -0.0007 | +0.0019 | 0.3785 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | budget | variance | k = 307 | 0.6738 | 0.6746 | +0.0008 | -0.0037 | +0.0054 | 0.7280 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | budget | S1 selector | mean 10%-90% | 0.6588 | 0.6705 | +0.0117 | -0.0002 | +0.0240 | 0.0669 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | budget | S1 selector | k = 307 | 0.6522 | 0.6746 | +0.0224 | +0.0037 | +0.0433 | 0.0315 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | weighted-aligned | S3 selector | mean 10%-90% | 0.6700 | 0.6701 | +0.0002 | -0.0010 | +0.0013 | 0.8033 | see note: msmarco |
| msmarco | trec-dl-2020 | ndcg@10 | weighted-aligned | S3 selector | k = 307 | 0.6738 | 0.6737 | -0.0001 | -0.0040 | +0.0034 | 0.9734 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | budget | aligned S3 | mean 10%-90% | 0.3673 | 0.3677 | +0.0004 | -0.0008 | +0.0017 | 0.5562 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | budget | aligned S3 | k = 307 | 0.3683 | 0.3682 | -0.0000 | -0.0038 | +0.0044 | 0.9931 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | budget | S3 selector | mean 10%-90% | 0.3683 | 0.3677 | -0.0007 | -0.0026 | +0.0012 | 0.5060 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | budget | S3 selector | k = 307 | 0.3697 | 0.3682 | -0.0014 | -0.0074 | +0.0052 | 0.6634 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | budget | S2 selector | mean 10%-90% | 0.3683 | 0.3677 | -0.0007 | -0.0026 | +0.0012 | 0.5060 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | budget | S2 selector | k = 307 | 0.3697 | 0.3682 | -0.0014 | -0.0074 | +0.0052 | 0.6634 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | budget | variance | mean 10%-90% | 0.3683 | 0.3677 | -0.0007 | -0.0026 | +0.0012 | 0.5060 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | budget | variance | k = 307 | 0.3697 | 0.3682 | -0.0014 | -0.0074 | +0.0052 | 0.6634 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | budget | S1 selector | mean 10%-90% | 0.3559 | 0.3677 | +0.0117 | -0.0014 | +0.0250 | 0.0898 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | budget | S1 selector | k = 307 | 0.3478 | 0.3682 | +0.0205 | +0.0018 | +0.0396 | 0.0406 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | weighted-aligned | S3 selector | mean 10%-90% | 0.3683 | 0.3673 | -0.0011 | -0.0027 | +0.0006 | 0.2281 | see note: msmarco |
| msmarco | trec-dl-hard | ndcg@10 | weighted-aligned | S3 selector | k = 307 | 0.3697 | 0.3683 | -0.0014 | -0.0060 | +0.0032 | 0.5623 | see note: msmarco |

## Notes

- **msmarco**: E0 does not reproduce Learning-to-Select on MS MARCO (ADR-0006). E1 scores MS MARCO dev against the full 8.8M-passage corpus; the paper evaluates the `test` split of an unpublished `msmarco_subset_50k` folder, with an ideal DCG taken from the retrieved list (F4 research D9).
- **nfcorpus**: NFCorpus's selector does not fit its oracle distribution as a calibrated distribution (ADR-0006); read its shift with its ranking-fit result (F4 research D2).
