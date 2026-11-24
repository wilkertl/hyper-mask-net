# P3: exact positive masks beside Learning-to-Select

Privileged, paired, in-domain diagnostic on training queries and training judgments. L2S numbers are **in-sample**: L2S was fitted on these queries. Nothing here measures performance on unseen queries.

- Run `20261112T160013Z-462aa6c4` (commit `d362793ae9be`, dirty `False`, device `cuda:0`), environments ['nfcorpus'], query bound 128 (a bounded sample).
- Decile grid: [102, 204, 307, 409, 512, 614, 716, 819, 921, 1024], with the 30% point treated like every other.
- Exact float64 scoring of every corpus document for every arm. Ties go to the lower row. No document is excluded by id.

## Reading these tables

- The **all-positive oracle** saw the positives. It is a privileged ceiling for their relevance-weighted mean similarity. It is not a ranking ceiling: a mask that raises positives can raise other documents more, and the tables show where it falls below the full query or L2S.
- **Held-out transfer** is built from one positive and scored on another of the same query (multi-positive queries only). All-label metrics for this arm count the document that built it; use `paired-held` for what happened to the others.
- L2S was fitted on the held-out positives as well as the others. Held-out transfer was not, so `held minus L2S` measures an in-sample fit against an out-of-construction mask, not two equally informed methods.
- The **pair-optimal** mask depends on the target document, so it is a secondary diagnostic of that target's rank. For single-positive queries it is the oracle's mask.
- Single-positive and multi-positive queries are separate cohorts and are never pooled.
- Intervals are pointwise bootstrap intervals over queries and are exploratory. Competing documents are other documents, not confirmed irrelevant ones.

## Provenance of the L2S checkpoint

| environment | check | passed |
|---|---|---|
| nfcorpus | checkpoint fingerprint is the dataset's | True |
| nfcorpus | dataset is the F2 dataset read here | True |
| nfcorpus | embeddings are the F2 embeddings read here | True |
| nfcorpus | embeddings fingerprint is the dataset's | True |
| nfcorpus | environment matches | True |
| nfcorpus | model revision matches | True |
| nfcorpus | negatives are the latest F2 negatives | True |
| nfcorpus | negatives fingerprint is the dataset's | True |
| nfcorpus | official recipe | True |
| nfcorpus | one published checkpoint | True |
| nfcorpus | payload hash matches | True |
| nfcorpus | per-domain, 1024 wide | True |
| nfcorpus | published F2 dataset | True |
| nfcorpus | split ids are disjoint | True |
| nfcorpus | trained by train-selector | True |
| nfcorpus | trained from a clean tree | True |

## Coverage

| environment | training queries | with a positive | evaluated | single-positive | multi-positive | positives | documents | capped | weights all 1 |
|---|---|---|---|---|---|---|---|---|---|
| nfcorpus | 2331 | 2331 | 128 | 64 | 64 | 1749 | 3633 | True | False |

## Official metrics, full training judgments

| environment | cohort | k | ratio | arm | ndcg@10 | mrr@10 | mrr@100 | recall@100 |
|---|---|---|---|---|---|---|---|---|
| nfcorpus | single-positive | 102 | 10% | prefix-k | 0.1902 (0.1072 to 0.2742) | 0.1664 (0.0877 to 0.2459) | 0.1751 (0.0975 to 0.2530) | 0.5469 (0.4375 to 0.6719) |
| nfcorpus | single-positive | 102 | 10% | all-positive oracle (privileged) | 0.9885 (0.9712 to 1.0000) | 0.9844 (0.9609 to 1.0000) | 0.9844 (0.9609 to 1.0000) | 1.0000 (1.0000 to 1.0000) |
| nfcorpus | single-positive | 102 | 10% | L2S (in-sample) | 0.9786 (0.9416 to 1.0000) | 0.9766 (0.9375 to 1.0000) | 0.9767 (0.9380 to 1.0000) | 1.0000 (1.0000 to 1.0000) |
| nfcorpus | single-positive | 204 | 20% | prefix-k | 0.2581 (0.1690 to 0.3538) | 0.2292 (0.1419 to 0.3190) | 0.2364 (0.1502 to 0.3263) | 0.5312 (0.4062 to 0.6562) |
| nfcorpus | single-positive | 204 | 20% | all-positive oracle (privileged) | 0.9838 (0.9572 to 1.0000) | 0.9788 (0.9442 to 1.0000) | 0.9788 (0.9442 to 1.0000) | 1.0000 (1.0000 to 1.0000) |
| nfcorpus | single-positive | 204 | 20% | L2S (in-sample) | 0.9728 (0.9358 to 1.0000) | 0.9688 (0.9297 to 1.0000) | 0.9688 (0.9297 to 1.0000) | 0.9844 (0.9531 to 1.0000) |
| nfcorpus | single-positive | 307 | 30% | prefix-k | 0.2708 (0.1829 to 0.3638) | 0.2411 (0.1552 to 0.3328) | 0.2510 (0.1660 to 0.3404) | 0.5938 (0.4688 to 0.7188) |
| nfcorpus | single-positive | 307 | 30% | all-positive oracle (privileged) | 0.9728 (0.9358 to 1.0000) | 0.9688 (0.9297 to 1.0000) | 0.9702 (0.9325 to 1.0000) | 1.0000 (1.0000 to 1.0000) |
| nfcorpus | single-positive | 307 | 30% | L2S (in-sample) | 0.9728 (0.9358 to 1.0000) | 0.9688 (0.9297 to 1.0000) | 0.9688 (0.9297 to 1.0000) | 0.9844 (0.9531 to 1.0000) |
| nfcorpus | single-positive | 409 | 40% | prefix-k | 0.2925 (0.2009 to 0.3921) | 0.2654 (0.1734 to 0.3630) | 0.2739 (0.1830 to 0.3699) | 0.6406 (0.5156 to 0.7500) |
| nfcorpus | single-positive | 409 | 40% | all-positive oracle (privileged) | 0.9613 (0.9202 to 0.9886) | 0.9531 (0.9062 to 0.9846) | 0.9543 (0.9111 to 0.9857) | 1.0000 (1.0000 to 1.0000) |
| nfcorpus | single-positive | 409 | 40% | L2S (in-sample) | 0.9671 (0.9260 to 0.9942) | 0.9609 (0.9217 to 0.9922) | 0.9609 (0.9217 to 0.9922) | 0.9844 (0.9531 to 1.0000) |
| nfcorpus | single-positive | 512 | 50% | prefix-k | 0.2880 (0.1976 to 0.3854) | 0.2551 (0.1685 to 0.3512) | 0.2621 (0.1766 to 0.3561) | 0.6094 (0.4844 to 0.7344) |
| nfcorpus | single-positive | 512 | 50% | all-positive oracle (privileged) | 0.9555 (0.9128 to 0.9885) | 0.9453 (0.8984 to 0.9844) | 0.9462 (0.9002 to 0.9844) | 1.0000 (1.0000 to 1.0000) |
| nfcorpus | single-positive | 512 | 50% | L2S (in-sample) | 0.9613 (0.9202 to 0.9886) | 0.9531 (0.9062 to 0.9846) | 0.9531 (0.9062 to 0.9846) | 0.9844 (0.9531 to 1.0000) |
| nfcorpus | single-positive | 614 | 60% | prefix-k | 0.2951 (0.2053 to 0.3913) | 0.2600 (0.1715 to 0.3540) | 0.2665 (0.1792 to 0.3576) | 0.5938 (0.4688 to 0.7188) |
| nfcorpus | single-positive | 614 | 60% | all-positive oracle (privileged) | 0.9477 (0.9029 to 0.9827) | 0.9349 (0.8828 to 0.9766) | 0.9356 (0.8849 to 0.9766) | 1.0000 (1.0000 to 1.0000) |
| nfcorpus | single-positive | 614 | 60% | L2S (in-sample) | 0.9514 (0.9066 to 0.9864) | 0.9401 (0.8906 to 0.9818) | 0.9401 (0.8906 to 0.9818) | 0.9844 (0.9531 to 1.0000) |
| nfcorpus | single-positive | 716 | 70% | prefix-k | 0.3037 (0.2098 to 0.4017) | 0.2723 (0.1813 to 0.3709) | 0.2817 (0.1919 to 0.3789) | 0.6094 (0.4844 to 0.7344) |
| nfcorpus | single-positive | 716 | 70% | all-positive oracle (privileged) | 0.9301 (0.8811 to 0.9728) | 0.9115 (0.8516 to 0.9635) | 0.9118 (0.8523 to 0.9635) | 1.0000 (1.0000 to 1.0000) |
| nfcorpus | single-positive | 716 | 70% | L2S (in-sample) | 0.9184 (0.8676 to 0.9630) | 0.8964 (0.8338 to 0.9531) | 0.8964 (0.8338 to 0.9531) | 0.9844 (0.9531 to 1.0000) |
| nfcorpus | single-positive | 819 | 80% | prefix-k | 0.2994 (0.2059 to 0.3990) | 0.2711 (0.1786 to 0.3703) | 0.2814 (0.1917 to 0.3790) | 0.5938 (0.4688 to 0.7188) |
| nfcorpus | single-positive | 819 | 80% | all-positive oracle (privileged) | 0.8340 (0.7664 to 0.8978) | 0.7904 (0.7107 to 0.8672) | 0.7915 (0.7130 to 0.8672) | 1.0000 (1.0000 to 1.0000) |
| nfcorpus | single-positive | 819 | 80% | L2S (in-sample) | 0.8026 (0.7266 to 0.8722) | 0.7548 (0.6672 to 0.8398) | 0.7566 (0.6707 to 0.8414) | 0.9844 (0.9531 to 1.0000) |
| nfcorpus | single-positive | 921 | 90% | prefix-k | 0.3028 (0.2104 to 0.4000) | 0.2710 (0.1803 to 0.3692) | 0.2788 (0.1882 to 0.3746) | 0.5938 (0.4688 to 0.7188) |
| nfcorpus | single-positive | 921 | 90% | all-positive oracle (privileged) | 0.6718 (0.5709 to 0.7754) | 0.6379 (0.5315 to 0.7487) | 0.6467 (0.5436 to 0.7533) | 0.9531 (0.8906 to 1.0000) |
| nfcorpus | single-positive | 921 | 90% | L2S (in-sample) | 0.6415 (0.5373 to 0.7499) | 0.6118 (0.5069 to 0.7227) | 0.6226 (0.5216 to 0.7310) | 0.9531 (0.8906 to 1.0000) |
| nfcorpus | single-positive | 1024 | 100% | full query | 0.3144 (0.2208 to 0.4112) | 0.2807 (0.1887 to 0.3767) | 0.2872 (0.1969 to 0.3829) | 0.6094 (0.4844 to 0.7188) |
| nfcorpus | single-positive | 1024 | 100% | prefix-k | 0.3144 (0.2208 to 0.4112) | 0.2807 (0.1887 to 0.3767) | 0.2872 (0.1969 to 0.3829) | 0.6094 (0.4844 to 0.7188) |
| nfcorpus | single-positive | 1024 | 100% | all-positive oracle (privileged) | 0.3144 (0.2208 to 0.4112) | 0.2807 (0.1887 to 0.3767) | 0.2872 (0.1969 to 0.3829) | 0.6094 (0.4844 to 0.7188) |
| nfcorpus | single-positive | 1024 | 100% | L2S (in-sample) | 0.3144 (0.2208 to 0.4112) | 0.2807 (0.1887 to 0.3767) | 0.2872 (0.1969 to 0.3829) | 0.6094 (0.4844 to 0.7188) |
| nfcorpus | multi-positive | 102 | 10% | prefix-k | 0.2674 (0.2036 to 0.3293) | 0.4722 (0.3699 to 0.5707) | 0.4793 (0.3792 to 0.5756) | 0.2724 (0.2269 to 0.3242) |
| nfcorpus | multi-positive | 102 | 10% | all-positive oracle (privileged) | 0.5672 (0.4906 to 0.6389) | 0.8249 (0.7377 to 0.9071) | 0.8291 (0.7443 to 0.9080) | 0.5618 (0.5078 to 0.6186) |
| nfcorpus | multi-positive | 102 | 10% | held-out transfer (1-1) | 0.5419 (0.4937 to 0.5901) | 0.9716 (0.9532 to 0.9869) | 0.9716 (0.9532 to 0.9869) | 0.4615 (0.4097 to 0.5155) |
| nfcorpus | multi-positive | 102 | 10% | L2S (in-sample) | 0.6359 (0.5645 to 0.7082) | 0.8478 (0.7696 to 0.9232) | 0.8487 (0.7712 to 0.9233) | 0.6494 (0.6007 to 0.6990) |
| nfcorpus | multi-positive | 204 | 20% | prefix-k | 0.3043 (0.2290 to 0.3776) | 0.4751 (0.3690 to 0.5792) | 0.4821 (0.3783 to 0.5840) | 0.2878 (0.2338 to 0.3472) |
| nfcorpus | multi-positive | 204 | 20% | all-positive oracle (privileged) | 0.5841 (0.5126 to 0.6539) | 0.8098 (0.7268 to 0.8904) | 0.8115 (0.7296 to 0.8914) | 0.5704 (0.5159 to 0.6251) |
| nfcorpus | multi-positive | 204 | 20% | held-out transfer (1-1) | 0.5606 (0.5102 to 0.6102) | 0.9709 (0.9508 to 0.9875) | 0.9709 (0.9508 to 0.9875) | 0.4730 (0.4197 to 0.5277) |
| nfcorpus | multi-positive | 204 | 20% | L2S (in-sample) | 0.6590 (0.5847 to 0.7288) | 0.8508 (0.7729 to 0.9235) | 0.8515 (0.7746 to 0.9238) | 0.6508 (0.6056 to 0.6973) |
| nfcorpus | multi-positive | 307 | 30% | prefix-k | 0.3137 (0.2398 to 0.3873) | 0.4876 (0.3801 to 0.5925) | 0.4932 (0.3874 to 0.5962) | 0.2966 (0.2421 to 0.3579) |
| nfcorpus | multi-positive | 307 | 30% | all-positive oracle (privileged) | 0.5925 (0.5206 to 0.6613) | 0.8285 (0.7483 to 0.9042) | 0.8303 (0.7520 to 0.9047) | 0.5635 (0.5104 to 0.6169) |
| nfcorpus | multi-positive | 307 | 30% | held-out transfer (1-1) | 0.5652 (0.5152 to 0.6158) | 0.9653 (0.9420 to 0.9844) | 0.9653 (0.9420 to 0.9844) | 0.4697 (0.4175 to 0.5243) |
| nfcorpus | multi-positive | 307 | 30% | L2S (in-sample) | 0.6463 (0.5784 to 0.7132) | 0.8511 (0.7697 to 0.9256) | 0.8518 (0.7710 to 0.9256) | 0.6498 (0.6045 to 0.6959) |
| nfcorpus | multi-positive | 409 | 40% | prefix-k | 0.3324 (0.2539 to 0.4103) | 0.5236 (0.4058 to 0.6338) | 0.5314 (0.4172 to 0.6387) | 0.3114 (0.2555 to 0.3724) |
| nfcorpus | multi-positive | 409 | 40% | all-positive oracle (privileged) | 0.5867 (0.5156 to 0.6571) | 0.8234 (0.7433 to 0.9037) | 0.8251 (0.7462 to 0.9041) | 0.5602 (0.5066 to 0.6128) |
| nfcorpus | multi-positive | 409 | 40% | held-out transfer (1-1) | 0.5646 (0.5146 to 0.6156) | 0.9585 (0.9330 to 0.9800) | 0.9585 (0.9330 to 0.9800) | 0.4674 (0.4156 to 0.5219) |
| nfcorpus | multi-positive | 409 | 40% | L2S (in-sample) | 0.6388 (0.5669 to 0.7072) | 0.8620 (0.7817 to 0.9337) | 0.8640 (0.7847 to 0.9342) | 0.6249 (0.5738 to 0.6752) |
| nfcorpus | multi-positive | 512 | 50% | prefix-k | 0.3347 (0.2583 to 0.4105) | 0.5218 (0.4119 to 0.6265) | 0.5259 (0.4178 to 0.6287) | 0.3163 (0.2610 to 0.3763) |
| nfcorpus | multi-positive | 512 | 50% | all-positive oracle (privileged) | 0.5836 (0.5126 to 0.6533) | 0.8360 (0.7560 to 0.9117) | 0.8377 (0.7604 to 0.9118) | 0.5494 (0.4957 to 0.6042) |
| nfcorpus | multi-positive | 512 | 50% | held-out transfer (1-1) | 0.5651 (0.5138 to 0.6176) | 0.9408 (0.9094 to 0.9691) | 0.9408 (0.9094 to 0.9691) | 0.4672 (0.4159 to 0.5219) |
| nfcorpus | multi-positive | 512 | 50% | L2S (in-sample) | 0.6227 (0.5564 to 0.6866) | 0.8690 (0.7974 to 0.9335) | 0.8703 (0.7989 to 0.9340) | 0.6165 (0.5687 to 0.6647) |
| nfcorpus | multi-positive | 614 | 60% | prefix-k | 0.3542 (0.2776 to 0.4274) | 0.5669 (0.4545 to 0.6667) | 0.5695 (0.4587 to 0.6687) | 0.3175 (0.2619 to 0.3812) |
| nfcorpus | multi-positive | 614 | 60% | all-positive oracle (privileged) | 0.5725 (0.5016 to 0.6414) | 0.8233 (0.7380 to 0.9054) | 0.8251 (0.7422 to 0.9063) | 0.5402 (0.4868 to 0.5952) |
| nfcorpus | multi-positive | 614 | 60% | held-out transfer (1-1) | 0.5565 (0.5035 to 0.6098) | 0.9138 (0.8745 to 0.9492) | 0.9138 (0.8745 to 0.9492) | 0.4637 (0.4129 to 0.5191) |
| nfcorpus | multi-positive | 614 | 60% | L2S (in-sample) | 0.5919 (0.5252 to 0.6569) | 0.8401 (0.7604 to 0.9152) | 0.8412 (0.7626 to 0.9155) | 0.5882 (0.5364 to 0.6421) |
| nfcorpus | multi-positive | 716 | 70% | prefix-k | 0.3524 (0.2782 to 0.4254) | 0.5826 (0.4742 to 0.6798) | 0.5854 (0.4765 to 0.6816) | 0.3150 (0.2578 to 0.3763) |
| nfcorpus | multi-positive | 716 | 70% | all-positive oracle (privileged) | 0.5554 (0.4843 to 0.6253) | 0.8114 (0.7295 to 0.8978) | 0.8131 (0.7330 to 0.8979) | 0.5296 (0.4752 to 0.5853) |
| nfcorpus | multi-positive | 716 | 70% | held-out transfer (1-1) | 0.5454 (0.4901 to 0.6003) | 0.8885 (0.8421 to 0.9301) | 0.8885 (0.8421 to 0.9301) | 0.4594 (0.4078 to 0.5142) |
| nfcorpus | multi-positive | 716 | 70% | L2S (in-sample) | 0.5594 (0.4938 to 0.6264) | 0.8176 (0.7330 to 0.8986) | 0.8180 (0.7334 to 0.8989) | 0.5664 (0.5094 to 0.6242) |
| nfcorpus | multi-positive | 819 | 80% | prefix-k | 0.3552 (0.2844 to 0.4242) | 0.5827 (0.4725 to 0.6835) | 0.5862 (0.4779 to 0.6852) | 0.3173 (0.2603 to 0.3792) |
| nfcorpus | multi-positive | 819 | 80% | all-positive oracle (privileged) | 0.5169 (0.4453 to 0.5864) | 0.7872 (0.7019 to 0.8739) | 0.7891 (0.7044 to 0.8746) | 0.5024 (0.4464 to 0.5630) |
| nfcorpus | multi-positive | 819 | 80% | held-out transfer (1-1) | 0.5186 (0.4563 to 0.5783) | 0.8310 (0.7669 to 0.8916) | 0.8326 (0.7696 to 0.8925) | 0.4493 (0.3971 to 0.5057) |
| nfcorpus | multi-positive | 819 | 80% | L2S (in-sample) | 0.5077 (0.4432 to 0.5777) | 0.7898 (0.7060 to 0.8730) | 0.7914 (0.7095 to 0.8740) | 0.5277 (0.4701 to 0.5855) |
| nfcorpus | multi-positive | 921 | 90% | prefix-k | 0.3534 (0.2765 to 0.4232) | 0.5904 (0.4794 to 0.6920) | 0.5958 (0.4867 to 0.6964) | 0.3213 (0.2657 to 0.3823) |
| nfcorpus | multi-positive | 921 | 90% | all-positive oracle (privileged) | 0.4461 (0.3736 to 0.5170) | 0.7328 (0.6369 to 0.8241) | 0.7365 (0.6426 to 0.8262) | 0.4530 (0.3941 to 0.5131) |
| nfcorpus | multi-positive | 921 | 90% | held-out transfer (1-1) | 0.4477 (0.3797 to 0.5151) | 0.7241 (0.6419 to 0.8024) | 0.7274 (0.6458 to 0.8049) | 0.4214 (0.3682 to 0.4813) |
| nfcorpus | multi-positive | 921 | 90% | L2S (in-sample) | 0.4492 (0.3781 to 0.5212) | 0.7311 (0.6400 to 0.8200) | 0.7354 (0.6468 to 0.8226) | 0.4518 (0.3985 to 0.5091) |
| nfcorpus | multi-positive | 1024 | 100% | full query | 0.3579 (0.2822 to 0.4304) | 0.5828 (0.4730 to 0.6833) | 0.5870 (0.4784 to 0.6867) | 0.3173 (0.2612 to 0.3806) |
| nfcorpus | multi-positive | 1024 | 100% | prefix-k | 0.3579 (0.2822 to 0.4304) | 0.5828 (0.4730 to 0.6833) | 0.5870 (0.4784 to 0.6867) | 0.3173 (0.2612 to 0.3806) |
| nfcorpus | multi-positive | 1024 | 100% | all-positive oracle (privileged) | 0.3579 (0.2822 to 0.4304) | 0.5828 (0.4730 to 0.6833) | 0.5870 (0.4784 to 0.6867) | 0.3173 (0.2612 to 0.3806) |
| nfcorpus | multi-positive | 1024 | 100% | held-out transfer (1-1) | 0.3579 (0.2822 to 0.4304) | 0.5828 (0.4730 to 0.6833) | 0.5870 (0.4784 to 0.6867) | 0.3173 (0.2612 to 0.3806) |
| nfcorpus | multi-positive | 1024 | 100% | L2S (in-sample) | 0.3579 (0.2822 to 0.4304) | 0.5828 (0.4730 to 0.6833) | 0.5870 (0.4784 to 0.6867) | 0.3173 (0.2612 to 0.3806) |

## Paired differences, with queries better and worse

| environment | cohort | k | measure | comparison | queries | difference | 2.5% | 97.5% | share better | share worse |
|---|---|---|---|---|---|---|---|---|---|---|
| nfcorpus | single-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0099 | 0.0000 | 0.0296 | 0.016 | 0.000 |
| nfcorpus | single-positive | 102 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.6642 | 0.5626 | 0.7576 | 0.766 | 0.000 |
| nfcorpus | single-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.6741 | 0.5772 | 0.7657 | 0.781 | 0.000 |
| nfcorpus | single-positive | 102 | ndcg@10 | prefix-k minus full query | 64 | -0.1242 | -0.1962 | -0.0612 | 0.031 | 0.250 |
| nfcorpus | single-positive | 102 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0078 | 0.0000 | 0.0234 | 0.016 | 0.000 |
| nfcorpus | single-positive | 102 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.6959 | 0.5952 | 0.7879 | 0.766 | 0.000 |
| nfcorpus | single-positive | 102 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.7037 | 0.6065 | 0.7934 | 0.781 | 0.000 |
| nfcorpus | single-positive | 102 | mrr@10 | prefix-k minus full query | 64 | -0.1142 | -0.1905 | -0.0488 | 0.031 | 0.250 |
| nfcorpus | single-positive | 102 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0076 | 0.0000 | 0.0229 | 0.016 | 0.000 |
| nfcorpus | single-positive | 102 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.6895 | 0.5906 | 0.7813 | 0.781 | 0.000 |
| nfcorpus | single-positive | 102 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.6972 | 0.6015 | 0.7857 | 0.781 | 0.000 |
| nfcorpus | single-positive | 102 | mrr@100 | prefix-k minus full query | 64 | -0.1121 | -0.1859 | -0.0494 | 0.094 | 0.391 |
| nfcorpus | single-positive | 102 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| nfcorpus | single-positive | 102 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3906 | 0.2812 | 0.5156 | 0.391 | 0.000 |
| nfcorpus | single-positive | 102 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.3906 | 0.2812 | 0.5156 | 0.391 | 0.000 |
| nfcorpus | single-positive | 102 | recall@100 | prefix-k minus full query | 64 | -0.0625 | -0.1406 | 0.0000 | 0.016 | 0.078 |
| nfcorpus | single-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0110 | 0.0000 | 0.0277 | 0.031 | 0.000 |
| nfcorpus | single-positive | 204 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.6585 | 0.5559 | 0.7525 | 0.750 | 0.000 |
| nfcorpus | single-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.6694 | 0.5723 | 0.7606 | 0.781 | 0.000 |
| nfcorpus | single-positive | 204 | ndcg@10 | prefix-k minus full query | 64 | -0.0563 | -0.1121 | 0.0032 | 0.078 | 0.203 |
| nfcorpus | single-positive | 204 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0100 | 0.0000 | 0.0279 | 0.031 | 0.000 |
| nfcorpus | single-positive | 204 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.6881 | 0.5869 | 0.7827 | 0.750 | 0.000 |
| nfcorpus | single-positive | 204 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.6981 | 0.5993 | 0.7885 | 0.781 | 0.000 |
| nfcorpus | single-positive | 204 | mrr@10 | prefix-k minus full query | 64 | -0.0515 | -0.1129 | 0.0086 | 0.078 | 0.203 |
| nfcorpus | single-positive | 204 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0100 | 0.0000 | 0.0279 | 0.031 | 0.000 |
| nfcorpus | single-positive | 204 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.6816 | 0.5813 | 0.7746 | 0.750 | 0.000 |
| nfcorpus | single-positive | 204 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.6916 | 0.5941 | 0.7813 | 0.781 | 0.000 |
| nfcorpus | single-positive | 204 | mrr@100 | prefix-k minus full query | 64 | -0.0508 | -0.1100 | 0.0082 | 0.125 | 0.344 |
| nfcorpus | single-positive | 204 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0156 | 0.0000 | 0.0469 | 0.016 | 0.000 |
| nfcorpus | single-positive | 204 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3750 | 0.2656 | 0.4844 | 0.375 | 0.000 |
| nfcorpus | single-positive | 204 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.3906 | 0.2812 | 0.5156 | 0.391 | 0.000 |
| nfcorpus | single-positive | 204 | recall@100 | prefix-k minus full query | 64 | -0.0781 | -0.1562 | 0.0000 | 0.016 | 0.094 |
| nfcorpus | single-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| nfcorpus | single-positive | 307 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.6585 | 0.5559 | 0.7525 | 0.750 | 0.000 |
| nfcorpus | single-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.6585 | 0.5559 | 0.7525 | 0.750 | 0.000 |
| nfcorpus | single-positive | 307 | ndcg@10 | prefix-k minus full query | 64 | -0.0436 | -0.0883 | 0.0003 | 0.078 | 0.156 |
| nfcorpus | single-positive | 307 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| nfcorpus | single-positive | 307 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.6881 | 0.5869 | 0.7827 | 0.750 | 0.000 |
| nfcorpus | single-positive | 307 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.6881 | 0.5869 | 0.7827 | 0.750 | 0.000 |
| nfcorpus | single-positive | 307 | mrr@10 | prefix-k minus full query | 64 | -0.0395 | -0.0887 | 0.0062 | 0.078 | 0.156 |
| nfcorpus | single-positive | 307 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0014 | 0.0000 | 0.0043 | 0.016 | 0.000 |
| nfcorpus | single-positive | 307 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.6816 | 0.5813 | 0.7746 | 0.750 | 0.000 |
| nfcorpus | single-positive | 307 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.6830 | 0.5834 | 0.7757 | 0.766 | 0.000 |
| nfcorpus | single-positive | 307 | mrr@100 | prefix-k minus full query | 64 | -0.0362 | -0.0842 | 0.0085 | 0.156 | 0.266 |
| nfcorpus | single-positive | 307 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0156 | 0.0000 | 0.0469 | 0.016 | 0.000 |
| nfcorpus | single-positive | 307 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3750 | 0.2656 | 0.4844 | 0.375 | 0.000 |
| nfcorpus | single-positive | 307 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.3906 | 0.2812 | 0.5156 | 0.391 | 0.000 |
| nfcorpus | single-positive | 307 | recall@100 | prefix-k minus full query | 64 | -0.0156 | -0.0781 | 0.0312 | 0.016 | 0.031 |
| nfcorpus | single-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0058 | -0.0173 | 0.0000 | 0.000 | 0.016 |
| nfcorpus | single-positive | 409 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.6527 | 0.5521 | 0.7455 | 0.750 | 0.000 |
| nfcorpus | single-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.6469 | 0.5456 | 0.7399 | 0.750 | 0.000 |
| nfcorpus | single-positive | 409 | ndcg@10 | prefix-k minus full query | 64 | -0.0219 | -0.0735 | 0.0245 | 0.078 | 0.141 |
| nfcorpus | single-positive | 409 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0078 | -0.0234 | 0.0000 | 0.000 | 0.016 |
| nfcorpus | single-positive | 409 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.6803 | 0.5780 | 0.7751 | 0.750 | 0.000 |
| nfcorpus | single-positive | 409 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.6724 | 0.5717 | 0.7682 | 0.750 | 0.000 |
| nfcorpus | single-positive | 409 | mrr@10 | prefix-k minus full query | 64 | -0.0153 | -0.0630 | 0.0290 | 0.078 | 0.141 |
| nfcorpus | single-positive | 409 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0066 | -0.0234 | 0.0036 | 0.016 | 0.016 |
| nfcorpus | single-positive | 409 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.6737 | 0.5733 | 0.7674 | 0.750 | 0.000 |
| nfcorpus | single-positive | 409 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.6671 | 0.5684 | 0.7607 | 0.766 | 0.000 |
| nfcorpus | single-positive | 409 | mrr@100 | prefix-k minus full query | 64 | -0.0133 | -0.0580 | 0.0291 | 0.156 | 0.266 |
| nfcorpus | single-positive | 409 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0156 | 0.0000 | 0.0469 | 0.016 | 0.000 |
| nfcorpus | single-positive | 409 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3750 | 0.2656 | 0.4844 | 0.375 | 0.000 |
| nfcorpus | single-positive | 409 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.3906 | 0.2812 | 0.5156 | 0.391 | 0.000 |
| nfcorpus | single-positive | 409 | recall@100 | prefix-k minus full query | 64 | 0.0312 | -0.0312 | 0.0938 | 0.047 | 0.016 |
| nfcorpus | single-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0058 | -0.0173 | 0.0000 | 0.000 | 0.016 |
| nfcorpus | single-positive | 512 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.6469 | 0.5456 | 0.7399 | 0.750 | 0.000 |
| nfcorpus | single-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.6412 | 0.5405 | 0.7365 | 0.750 | 0.000 |
| nfcorpus | single-positive | 512 | ndcg@10 | prefix-k minus full query | 64 | -0.0264 | -0.0627 | 0.0083 | 0.062 | 0.125 |
| nfcorpus | single-positive | 512 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0078 | -0.0234 | 0.0000 | 0.000 | 0.016 |
| nfcorpus | single-positive | 512 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.6724 | 0.5717 | 0.7682 | 0.750 | 0.000 |
| nfcorpus | single-positive | 512 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.6646 | 0.5650 | 0.7606 | 0.750 | 0.000 |
| nfcorpus | single-positive | 512 | mrr@10 | prefix-k minus full query | 64 | -0.0256 | -0.0632 | 0.0077 | 0.062 | 0.125 |
| nfcorpus | single-positive | 512 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0069 | -0.0234 | 0.0026 | 0.016 | 0.016 |
| nfcorpus | single-positive | 512 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.6659 | 0.5672 | 0.7593 | 0.750 | 0.000 |
| nfcorpus | single-positive | 512 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.6590 | 0.5616 | 0.7545 | 0.766 | 0.000 |
| nfcorpus | single-positive | 512 | mrr@100 | prefix-k minus full query | 64 | -0.0251 | -0.0620 | 0.0069 | 0.141 | 0.234 |
| nfcorpus | single-positive | 512 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0156 | 0.0000 | 0.0469 | 0.016 | 0.000 |
| nfcorpus | single-positive | 512 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3750 | 0.2656 | 0.4844 | 0.375 | 0.000 |
| nfcorpus | single-positive | 512 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.3906 | 0.2812 | 0.5156 | 0.391 | 0.000 |
| nfcorpus | single-positive | 512 | recall@100 | prefix-k minus full query | 64 | 0.0000 | -0.0625 | 0.0625 | 0.031 | 0.031 |
| nfcorpus | single-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0037 | -0.0173 | 0.0061 | 0.016 | 0.016 |
| nfcorpus | single-positive | 614 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.6371 | 0.5369 | 0.7314 | 0.750 | 0.000 |
| nfcorpus | single-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.6334 | 0.5333 | 0.7296 | 0.750 | 0.000 |
| nfcorpus | single-positive | 614 | ndcg@10 | prefix-k minus full query | 64 | -0.0193 | -0.0502 | 0.0096 | 0.047 | 0.125 |
| nfcorpus | single-positive | 614 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0052 | -0.0234 | 0.0078 | 0.016 | 0.016 |
| nfcorpus | single-positive | 614 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.6594 | 0.5589 | 0.7552 | 0.750 | 0.000 |
| nfcorpus | single-positive | 614 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.6542 | 0.5534 | 0.7507 | 0.750 | 0.000 |
| nfcorpus | single-positive | 614 | mrr@10 | prefix-k minus full query | 64 | -0.0207 | -0.0487 | 0.0017 | 0.047 | 0.125 |
| nfcorpus | single-positive | 614 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0045 | -0.0228 | 0.0078 | 0.031 | 0.016 |
| nfcorpus | single-positive | 614 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.6529 | 0.5529 | 0.7483 | 0.750 | 0.000 |
| nfcorpus | single-positive | 614 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.6484 | 0.5500 | 0.7441 | 0.766 | 0.000 |
| nfcorpus | single-positive | 614 | mrr@100 | prefix-k minus full query | 64 | -0.0207 | -0.0479 | 0.0007 | 0.109 | 0.219 |
| nfcorpus | single-positive | 614 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0156 | 0.0000 | 0.0469 | 0.016 | 0.000 |
| nfcorpus | single-positive | 614 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3750 | 0.2656 | 0.4844 | 0.375 | 0.000 |
| nfcorpus | single-positive | 614 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.3906 | 0.2812 | 0.5156 | 0.391 | 0.000 |
| nfcorpus | single-positive | 614 | recall@100 | prefix-k minus full query | 64 | -0.0156 | -0.0781 | 0.0312 | 0.016 | 0.031 |
| nfcorpus | single-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0116 | -0.0136 | 0.0374 | 0.094 | 0.031 |
| nfcorpus | single-positive | 716 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.6041 | 0.5094 | 0.6980 | 0.750 | 0.000 |
| nfcorpus | single-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.6157 | 0.5181 | 0.7097 | 0.750 | 0.000 |
| nfcorpus | single-positive | 716 | ndcg@10 | prefix-k minus full query | 64 | -0.0107 | -0.0327 | 0.0096 | 0.031 | 0.109 |
| nfcorpus | single-positive | 716 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0151 | -0.0182 | 0.0495 | 0.094 | 0.031 |
| nfcorpus | single-positive | 716 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.6157 | 0.5184 | 0.7149 | 0.750 | 0.000 |
| nfcorpus | single-positive | 716 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.6308 | 0.5341 | 0.7253 | 0.750 | 0.000 |
| nfcorpus | single-positive | 716 | mrr@10 | prefix-k minus full query | 64 | -0.0084 | -0.0185 | 0.0009 | 0.031 | 0.109 |
| nfcorpus | single-positive | 716 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0155 | -0.0178 | 0.0503 | 0.109 | 0.031 |
| nfcorpus | single-positive | 716 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.6092 | 0.5127 | 0.7076 | 0.750 | 0.000 |
| nfcorpus | single-positive | 716 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.6246 | 0.5289 | 0.7186 | 0.766 | 0.000 |
| nfcorpus | single-positive | 716 | mrr@100 | prefix-k minus full query | 64 | -0.0055 | -0.0135 | 0.0014 | 0.141 | 0.188 |
| nfcorpus | single-positive | 716 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0156 | 0.0000 | 0.0469 | 0.016 | 0.000 |
| nfcorpus | single-positive | 716 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3750 | 0.2656 | 0.4844 | 0.375 | 0.000 |
| nfcorpus | single-positive | 716 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.3906 | 0.2812 | 0.5156 | 0.391 | 0.000 |
| nfcorpus | single-positive | 716 | recall@100 | prefix-k minus full query | 64 | 0.0000 | -0.0625 | 0.0625 | 0.031 | 0.031 |
| nfcorpus | single-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0314 | 0.0035 | 0.0625 | 0.156 | 0.062 |
| nfcorpus | single-positive | 819 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.4882 | 0.3976 | 0.5770 | 0.703 | 0.000 |
| nfcorpus | single-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.5196 | 0.4312 | 0.6088 | 0.734 | 0.000 |
| nfcorpus | single-positive | 819 | ndcg@10 | prefix-k minus full query | 64 | -0.0150 | -0.0351 | 0.0031 | 0.016 | 0.109 |
| nfcorpus | single-positive | 819 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0356 | 0.0024 | 0.0725 | 0.156 | 0.062 |
| nfcorpus | single-positive | 819 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.4741 | 0.3793 | 0.5722 | 0.703 | 0.000 |
| nfcorpus | single-positive | 819 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.5097 | 0.4140 | 0.6026 | 0.734 | 0.000 |
| nfcorpus | single-positive | 819 | mrr@10 | prefix-k minus full query | 64 | -0.0096 | -0.0188 | -0.0012 | 0.016 | 0.109 |
| nfcorpus | single-positive | 819 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0348 | 0.0015 | 0.0716 | 0.188 | 0.062 |
| nfcorpus | single-positive | 819 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.4694 | 0.3772 | 0.5646 | 0.734 | 0.000 |
| nfcorpus | single-positive | 819 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.5043 | 0.4110 | 0.5954 | 0.766 | 0.000 |
| nfcorpus | single-positive | 819 | mrr@100 | prefix-k minus full query | 64 | -0.0058 | -0.0131 | 0.0003 | 0.094 | 0.219 |
| nfcorpus | single-positive | 819 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0156 | 0.0000 | 0.0469 | 0.016 | 0.000 |
| nfcorpus | single-positive | 819 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3750 | 0.2656 | 0.4844 | 0.375 | 0.000 |
| nfcorpus | single-positive | 819 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.3906 | 0.2812 | 0.5156 | 0.391 | 0.000 |
| nfcorpus | single-positive | 819 | recall@100 | prefix-k minus full query | 64 | -0.0156 | -0.0781 | 0.0312 | 0.016 | 0.031 |
| nfcorpus | single-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0303 | 0.0065 | 0.0570 | 0.125 | 0.016 |
| nfcorpus | single-positive | 921 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.3272 | 0.2396 | 0.4205 | 0.484 | 0.000 |
| nfcorpus | single-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.3574 | 0.2705 | 0.4470 | 0.531 | 0.000 |
| nfcorpus | single-positive | 921 | ndcg@10 | prefix-k minus full query | 64 | -0.0116 | -0.0247 | -0.0027 | 0.000 | 0.109 |
| nfcorpus | single-positive | 921 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0261 | 0.0016 | 0.0553 | 0.125 | 0.016 |
| nfcorpus | single-positive | 921 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.3311 | 0.2358 | 0.4271 | 0.484 | 0.000 |
| nfcorpus | single-positive | 921 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.3572 | 0.2608 | 0.4564 | 0.531 | 0.000 |
| nfcorpus | single-positive | 921 | mrr@10 | prefix-k minus full query | 64 | -0.0097 | -0.0184 | -0.0026 | 0.000 | 0.109 |
| nfcorpus | single-positive | 921 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0241 | 0.0003 | 0.0529 | 0.250 | 0.062 |
| nfcorpus | single-positive | 921 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.3354 | 0.2447 | 0.4282 | 0.703 | 0.000 |
| nfcorpus | single-positive | 921 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.3595 | 0.2661 | 0.4550 | 0.703 | 0.000 |
| nfcorpus | single-positive | 921 | mrr@100 | prefix-k minus full query | 64 | -0.0084 | -0.0173 | -0.0014 | 0.109 | 0.203 |
| nfcorpus | single-positive | 921 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0000 | -0.0469 | 0.0469 | 0.016 | 0.016 |
| nfcorpus | single-positive | 921 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3438 | 0.2344 | 0.4531 | 0.344 | 0.000 |
| nfcorpus | single-positive | 921 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.3438 | 0.2344 | 0.4688 | 0.344 | 0.000 |
| nfcorpus | single-positive | 921 | recall@100 | prefix-k minus full query | 64 | -0.0156 | -0.0781 | 0.0312 | 0.016 | 0.031 |
| nfcorpus | multi-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0687 | -0.1160 | -0.0231 | 0.326 | 0.530 |
| nfcorpus | multi-positive | 102 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.2779 | 0.2198 | 0.3379 | 0.842 | 0.121 |
| nfcorpus | multi-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.2093 | 0.1611 | 0.2596 | 0.694 | 0.189 |
| nfcorpus | multi-positive | 102 | ndcg@10 | prefix-k minus full query | 64 | -0.0905 | -0.1302 | -0.0494 | 0.160 | 0.626 |
| nfcorpus | multi-positive | 102 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0229 | -0.1115 | 0.0616 | 0.122 | 0.162 |
| nfcorpus | multi-positive | 102 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.2650 | 0.1568 | 0.3747 | 0.459 | 0.083 |
| nfcorpus | multi-positive | 102 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.2421 | 0.1525 | 0.3395 | 0.405 | 0.098 |
| nfcorpus | multi-positive | 102 | mrr@10 | prefix-k minus full query | 64 | -0.1106 | -0.2033 | -0.0179 | 0.114 | 0.323 |
| nfcorpus | multi-positive | 102 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0196 | -0.1048 | 0.0629 | 0.122 | 0.162 |
| nfcorpus | multi-positive | 102 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.2617 | 0.1537 | 0.3698 | 0.459 | 0.083 |
| nfcorpus | multi-positive | 102 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.2421 | 0.1544 | 0.3386 | 0.429 | 0.098 |
| nfcorpus | multi-positive | 102 | mrr@100 | prefix-k minus full query | 64 | -0.1077 | -0.1998 | -0.0175 | 0.186 | 0.382 |
| nfcorpus | multi-positive | 102 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0876 | -0.1210 | -0.0556 | 0.091 | 0.595 |
| nfcorpus | multi-positive | 102 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3321 | 0.2835 | 0.3805 | 0.921 | 0.023 |
| nfcorpus | multi-positive | 102 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.2445 | 0.2034 | 0.2860 | 0.844 | 0.068 |
| nfcorpus | multi-positive | 102 | recall@100 | prefix-k minus full query | 64 | -0.0449 | -0.0776 | -0.0150 | 0.193 | 0.452 |
| nfcorpus | multi-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0749 | -0.1084 | -0.0393 | 0.167 | 0.693 |
| nfcorpus | multi-positive | 204 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.3011 | 0.2422 | 0.3618 | 0.855 | 0.083 |
| nfcorpus | multi-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.2262 | 0.1754 | 0.2793 | 0.770 | 0.151 |
| nfcorpus | multi-positive | 204 | ndcg@10 | prefix-k minus full query | 64 | -0.0537 | -0.0854 | -0.0230 | 0.199 | 0.532 |
| nfcorpus | multi-positive | 204 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0409 | -0.1112 | 0.0233 | 0.092 | 0.145 |
| nfcorpus | multi-positive | 204 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.2680 | 0.1660 | 0.3723 | 0.443 | 0.083 |
| nfcorpus | multi-positive | 204 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.2270 | 0.1348 | 0.3198 | 0.405 | 0.121 |
| nfcorpus | multi-positive | 204 | mrr@10 | prefix-k minus full query | 64 | -0.1077 | -0.1785 | -0.0395 | 0.069 | 0.291 |
| nfcorpus | multi-positive | 204 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0400 | -0.1100 | 0.0252 | 0.107 | 0.153 |
| nfcorpus | multi-positive | 204 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.2645 | 0.1627 | 0.3685 | 0.467 | 0.083 |
| nfcorpus | multi-positive | 204 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.2245 | 0.1333 | 0.3164 | 0.429 | 0.121 |
| nfcorpus | multi-positive | 204 | mrr@100 | prefix-k minus full query | 64 | -0.1049 | -0.1741 | -0.0383 | 0.140 | 0.356 |
| nfcorpus | multi-positive | 204 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0804 | -0.1069 | -0.0534 | 0.045 | 0.599 |
| nfcorpus | multi-positive | 204 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3335 | 0.2845 | 0.3833 | 0.921 | 0.023 |
| nfcorpus | multi-positive | 204 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.2531 | 0.2111 | 0.2968 | 0.874 | 0.023 |
| nfcorpus | multi-positive | 204 | recall@100 | prefix-k minus full query | 64 | -0.0295 | -0.0487 | -0.0104 | 0.152 | 0.454 |
| nfcorpus | multi-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0538 | -0.0880 | -0.0194 | 0.226 | 0.628 |
| nfcorpus | multi-positive | 307 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.2884 | 0.2316 | 0.3485 | 0.842 | 0.106 |
| nfcorpus | multi-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.2346 | 0.1817 | 0.2949 | 0.838 | 0.068 |
| nfcorpus | multi-positive | 307 | ndcg@10 | prefix-k minus full query | 64 | -0.0442 | -0.0745 | -0.0164 | 0.183 | 0.509 |
| nfcorpus | multi-positive | 307 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0227 | -0.0823 | 0.0442 | 0.083 | 0.176 |
| nfcorpus | multi-positive | 307 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.2684 | 0.1621 | 0.3784 | 0.437 | 0.083 |
| nfcorpus | multi-positive | 307 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.2457 | 0.1564 | 0.3456 | 0.413 | 0.076 |
| nfcorpus | multi-positive | 307 | mrr@10 | prefix-k minus full query | 64 | -0.0952 | -0.1633 | -0.0395 | 0.054 | 0.253 |
| nfcorpus | multi-positive | 307 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0215 | -0.0809 | 0.0450 | 0.083 | 0.191 |
| nfcorpus | multi-positive | 307 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.2648 | 0.1581 | 0.3741 | 0.452 | 0.083 |
| nfcorpus | multi-positive | 307 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.2433 | 0.1547 | 0.3424 | 0.437 | 0.076 |
| nfcorpus | multi-positive | 307 | mrr@100 | prefix-k minus full query | 64 | -0.0938 | -0.1619 | -0.0386 | 0.122 | 0.312 |
| nfcorpus | multi-positive | 307 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0863 | -0.1126 | -0.0595 | 0.053 | 0.629 |
| nfcorpus | multi-positive | 307 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3325 | 0.2857 | 0.3771 | 0.921 | 0.000 |
| nfcorpus | multi-positive | 307 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.2462 | 0.2049 | 0.2904 | 0.860 | 0.045 |
| nfcorpus | multi-positive | 307 | recall@100 | prefix-k minus full query | 64 | -0.0207 | -0.0433 | -0.0001 | 0.199 | 0.410 |
| nfcorpus | multi-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0521 | -0.0856 | -0.0188 | 0.265 | 0.595 |
| nfcorpus | multi-positive | 409 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.2808 | 0.2239 | 0.3397 | 0.855 | 0.083 |
| nfcorpus | multi-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.2288 | 0.1783 | 0.2845 | 0.815 | 0.106 |
| nfcorpus | multi-positive | 409 | ndcg@10 | prefix-k minus full query | 64 | -0.0256 | -0.0554 | 0.0016 | 0.275 | 0.449 |
| nfcorpus | multi-positive | 409 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0386 | -0.1016 | 0.0263 | 0.092 | 0.161 |
| nfcorpus | multi-positive | 409 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.2792 | 0.1753 | 0.3827 | 0.437 | 0.060 |
| nfcorpus | multi-positive | 409 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.2406 | 0.1612 | 0.3278 | 0.413 | 0.053 |
| nfcorpus | multi-positive | 409 | mrr@10 | prefix-k minus full query | 64 | -0.0592 | -0.1129 | -0.0101 | 0.047 | 0.247 |
| nfcorpus | multi-positive | 409 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0389 | -0.1019 | 0.0258 | 0.107 | 0.184 |
| nfcorpus | multi-positive | 409 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.2770 | 0.1737 | 0.3784 | 0.452 | 0.060 |
| nfcorpus | multi-positive | 409 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.2381 | 0.1597 | 0.3229 | 0.437 | 0.053 |
| nfcorpus | multi-positive | 409 | mrr@100 | prefix-k minus full query | 64 | -0.0556 | -0.1080 | -0.0087 | 0.135 | 0.294 |
| nfcorpus | multi-positive | 409 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0647 | -0.0833 | -0.0472 | 0.038 | 0.697 |
| nfcorpus | multi-positive | 409 | recall@100 | L2S (in-sample) minus full query | 64 | 0.3076 | 0.2616 | 0.3545 | 0.921 | 0.023 |
| nfcorpus | multi-positive | 409 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.2429 | 0.2014 | 0.2865 | 0.889 | 0.045 |
| nfcorpus | multi-positive | 409 | recall@100 | prefix-k minus full query | 64 | -0.0058 | -0.0262 | 0.0154 | 0.192 | 0.282 |
| nfcorpus | multi-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0391 | -0.0738 | -0.0048 | 0.356 | 0.498 |
| nfcorpus | multi-positive | 512 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.2648 | 0.2107 | 0.3219 | 0.833 | 0.060 |
| nfcorpus | multi-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.2257 | 0.1759 | 0.2824 | 0.876 | 0.045 |
| nfcorpus | multi-positive | 512 | ndcg@10 | prefix-k minus full query | 64 | -0.0233 | -0.0460 | -0.0033 | 0.221 | 0.414 |
| nfcorpus | multi-positive | 512 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0330 | -0.1072 | 0.0332 | 0.106 | 0.132 |
| nfcorpus | multi-positive | 512 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.2862 | 0.1867 | 0.3889 | 0.444 | 0.038 |
| nfcorpus | multi-positive | 512 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.2532 | 0.1684 | 0.3472 | 0.413 | 0.053 |
| nfcorpus | multi-positive | 512 | mrr@10 | prefix-k minus full query | 64 | -0.0609 | -0.1121 | -0.0146 | 0.047 | 0.224 |
| nfcorpus | multi-positive | 512 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0326 | -0.1044 | 0.0333 | 0.106 | 0.147 |
| nfcorpus | multi-positive | 512 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.2833 | 0.1856 | 0.3850 | 0.459 | 0.038 |
| nfcorpus | multi-positive | 512 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.2507 | 0.1666 | 0.3423 | 0.437 | 0.053 |
| nfcorpus | multi-positive | 512 | mrr@100 | prefix-k minus full query | 64 | -0.0611 | -0.1114 | -0.0158 | 0.088 | 0.303 |
| nfcorpus | multi-positive | 512 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0671 | -0.0891 | -0.0455 | 0.038 | 0.599 |
| nfcorpus | multi-positive | 512 | recall@100 | L2S (in-sample) minus full query | 64 | 0.2992 | 0.2539 | 0.3450 | 0.921 | 0.000 |
| nfcorpus | multi-positive | 512 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.2321 | 0.1906 | 0.2745 | 0.874 | 0.023 |
| nfcorpus | multi-positive | 512 | recall@100 | prefix-k minus full query | 64 | -0.0010 | -0.0230 | 0.0222 | 0.223 | 0.297 |
| nfcorpus | multi-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0194 | -0.0519 | 0.0126 | 0.371 | 0.466 |
| nfcorpus | multi-positive | 614 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.2340 | 0.1874 | 0.2848 | 0.838 | 0.045 |
| nfcorpus | multi-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.2146 | 0.1679 | 0.2684 | 0.860 | 0.060 |
| nfcorpus | multi-positive | 614 | ndcg@10 | prefix-k minus full query | 64 | -0.0037 | -0.0166 | 0.0108 | 0.216 | 0.396 |
| nfcorpus | multi-positive | 614 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0168 | -0.0898 | 0.0519 | 0.077 | 0.108 |
| nfcorpus | multi-positive | 614 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.2573 | 0.1619 | 0.3552 | 0.413 | 0.038 |
| nfcorpus | multi-positive | 614 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.2405 | 0.1546 | 0.3347 | 0.413 | 0.053 |
| nfcorpus | multi-positive | 614 | mrr@10 | prefix-k minus full query | 64 | -0.0159 | -0.0377 | 0.0019 | 0.055 | 0.176 |
| nfcorpus | multi-positive | 614 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0161 | -0.0863 | 0.0518 | 0.077 | 0.132 |
| nfcorpus | multi-positive | 614 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.2542 | 0.1598 | 0.3507 | 0.437 | 0.038 |
| nfcorpus | multi-positive | 614 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.2381 | 0.1541 | 0.3310 | 0.437 | 0.053 |
| nfcorpus | multi-positive | 614 | mrr@100 | prefix-k minus full query | 64 | -0.0175 | -0.0390 | -0.0009 | 0.159 | 0.209 |
| nfcorpus | multi-positive | 614 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0480 | -0.0690 | -0.0273 | 0.062 | 0.538 |
| nfcorpus | multi-positive | 614 | recall@100 | L2S (in-sample) minus full query | 64 | 0.2709 | 0.2262 | 0.3157 | 0.906 | 0.000 |
| nfcorpus | multi-positive | 614 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.2229 | 0.1832 | 0.2645 | 0.889 | 0.023 |
| nfcorpus | multi-positive | 614 | recall@100 | prefix-k minus full query | 64 | 0.0002 | -0.0140 | 0.0131 | 0.183 | 0.220 |
| nfcorpus | multi-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0040 | -0.0309 | 0.0266 | 0.351 | 0.419 |
| nfcorpus | multi-positive | 716 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.2015 | 0.1590 | 0.2468 | 0.800 | 0.068 |
| nfcorpus | multi-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.1975 | 0.1519 | 0.2490 | 0.823 | 0.083 |
| nfcorpus | multi-positive | 716 | ndcg@10 | prefix-k minus full query | 64 | -0.0055 | -0.0179 | 0.0078 | 0.282 | 0.368 |
| nfcorpus | multi-positive | 716 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0062 | -0.0755 | 0.0614 | 0.093 | 0.092 |
| nfcorpus | multi-positive | 716 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.2348 | 0.1485 | 0.3300 | 0.413 | 0.038 |
| nfcorpus | multi-positive | 716 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.2286 | 0.1484 | 0.3173 | 0.390 | 0.053 |
| nfcorpus | multi-positive | 716 | mrr@10 | prefix-k minus full query | 64 | -0.0002 | -0.0260 | 0.0263 | 0.084 | 0.111 |
| nfcorpus | multi-positive | 716 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0050 | -0.0728 | 0.0626 | 0.108 | 0.101 |
| nfcorpus | multi-positive | 716 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.2311 | 0.1464 | 0.3253 | 0.437 | 0.038 |
| nfcorpus | multi-positive | 716 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.2261 | 0.1464 | 0.3128 | 0.414 | 0.053 |
| nfcorpus | multi-positive | 716 | mrr@100 | prefix-k minus full query | 64 | -0.0016 | -0.0269 | 0.0241 | 0.161 | 0.161 |
| nfcorpus | multi-positive | 716 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0367 | -0.0565 | -0.0180 | 0.092 | 0.455 |
| nfcorpus | multi-positive | 716 | recall@100 | L2S (in-sample) minus full query | 64 | 0.2491 | 0.2032 | 0.2952 | 0.874 | 0.000 |
| nfcorpus | multi-positive | 716 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.2123 | 0.1713 | 0.2565 | 0.865 | 0.045 |
| nfcorpus | multi-positive | 716 | recall@100 | prefix-k minus full query | 64 | -0.0023 | -0.0176 | 0.0116 | 0.228 | 0.191 |
| nfcorpus | multi-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0092 | -0.0153 | 0.0355 | 0.507 | 0.343 |
| nfcorpus | multi-positive | 819 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.1498 | 0.1121 | 0.1898 | 0.775 | 0.060 |
| nfcorpus | multi-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.1590 | 0.1205 | 0.2031 | 0.791 | 0.106 |
| nfcorpus | multi-positive | 819 | ndcg@10 | prefix-k minus full query | 64 | -0.0027 | -0.0194 | 0.0180 | 0.228 | 0.361 |
| nfcorpus | multi-positive | 819 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0026 | -0.0656 | 0.0564 | 0.127 | 0.117 |
| nfcorpus | multi-positive | 819 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.2070 | 0.1258 | 0.3002 | 0.411 | 0.030 |
| nfcorpus | multi-positive | 819 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.2044 | 0.1255 | 0.2959 | 0.381 | 0.053 |
| nfcorpus | multi-positive | 819 | mrr@10 | prefix-k minus full query | 64 | -0.0001 | -0.0363 | 0.0372 | 0.084 | 0.096 |
| nfcorpus | multi-positive | 819 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0023 | -0.0637 | 0.0554 | 0.151 | 0.117 |
| nfcorpus | multi-positive | 819 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.2044 | 0.1252 | 0.2953 | 0.420 | 0.030 |
| nfcorpus | multi-positive | 819 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.2021 | 0.1244 | 0.2907 | 0.414 | 0.053 |
| nfcorpus | multi-positive | 819 | mrr@100 | prefix-k minus full query | 64 | -0.0008 | -0.0358 | 0.0349 | 0.155 | 0.152 |
| nfcorpus | multi-positive | 819 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0253 | -0.0483 | -0.0045 | 0.131 | 0.366 |
| nfcorpus | multi-positive | 819 | recall@100 | L2S (in-sample) minus full query | 64 | 0.2104 | 0.1671 | 0.2561 | 0.826 | 0.023 |
| nfcorpus | multi-positive | 819 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.1852 | 0.1445 | 0.2293 | 0.857 | 0.045 |
| nfcorpus | multi-positive | 819 | recall@100 | prefix-k minus full query | 64 | -0.0000 | -0.0143 | 0.0124 | 0.220 | 0.145 |
| nfcorpus | multi-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | -0.0031 | -0.0217 | 0.0151 | 0.404 | 0.318 |
| nfcorpus | multi-positive | 921 | ndcg@10 | L2S (in-sample) minus full query | 64 | 0.0913 | 0.0627 | 0.1218 | 0.643 | 0.076 |
| nfcorpus | multi-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus full query | 64 | 0.0882 | 0.0594 | 0.1188 | 0.679 | 0.090 |
| nfcorpus | multi-positive | 921 | ndcg@10 | prefix-k minus full query | 64 | -0.0045 | -0.0135 | 0.0049 | 0.226 | 0.269 |
| nfcorpus | multi-positive | 921 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0017 | -0.0593 | 0.0491 | 0.123 | 0.084 |
| nfcorpus | multi-positive | 921 | mrr@10 | L2S (in-sample) minus full query | 64 | 0.1483 | 0.0851 | 0.2178 | 0.355 | 0.030 |
| nfcorpus | multi-positive | 921 | mrr@10 | all-positive oracle (privileged) minus full query | 64 | 0.1500 | 0.0870 | 0.2178 | 0.331 | 0.038 |
| nfcorpus | multi-positive | 921 | mrr@10 | prefix-k minus full query | 64 | 0.0076 | -0.0073 | 0.0274 | 0.076 | 0.064 |
| nfcorpus | multi-positive | 921 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0011 | -0.0574 | 0.0474 | 0.147 | 0.117 |
| nfcorpus | multi-positive | 921 | mrr@100 | L2S (in-sample) minus full query | 64 | 0.1484 | 0.0855 | 0.2171 | 0.426 | 0.045 |
| nfcorpus | multi-positive | 921 | mrr@100 | all-positive oracle (privileged) minus full query | 64 | 0.1495 | 0.0880 | 0.2150 | 0.381 | 0.038 |
| nfcorpus | multi-positive | 921 | mrr@100 | prefix-k minus full query | 64 | 0.0088 | -0.0061 | 0.0283 | 0.168 | 0.123 |
| nfcorpus | multi-positive | 921 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 64 | 0.0013 | -0.0159 | 0.0170 | 0.243 | 0.221 |
| nfcorpus | multi-positive | 921 | recall@100 | L2S (in-sample) minus full query | 64 | 0.1345 | 0.1021 | 0.1670 | 0.764 | 0.045 |
| nfcorpus | multi-positive | 921 | recall@100 | all-positive oracle (privileged) minus full query | 64 | 0.1358 | 0.1004 | 0.1727 | 0.786 | 0.045 |
| nfcorpus | multi-positive | 921 | recall@100 | prefix-k minus full query | 64 | 0.0040 | -0.0040 | 0.0126 | 0.160 | 0.128 |

## Held-out positives of multi-positive queries

Same partitions for every arm. Measures use only the held-out positives.

| environment | cohort | k | measure | comparison | queries | difference | 2.5% | 97.5% | share better | share worse |
|---|---|---|---|---|---|---|---|---|---|---|
| nfcorpus | multi-positive | 102 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1119 | -0.1336 | -0.0898 | 0.166 | 0.834 |
| nfcorpus | multi-positive | 102 | held_rr | L2S (in-sample) minus full query | 64 | 0.1121 | 0.0876 | 0.1370 | 0.827 | 0.173 |
| nfcorpus | multi-positive | 102 | held_rr | held-out transfer (1-1) minus full query | 64 | 0.0003 | -0.0143 | 0.0159 | 0.513 | 0.487 |
| nfcorpus | multi-positive | 102 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1004 | 0.0817 | 0.1185 | 0.872 | 0.128 |
| nfcorpus | multi-positive | 102 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 64 | 1.3912 | 1.1795 | 1.6104 | 0.955 | 0.045 |
| nfcorpus | multi-positive | 102 | held_logrank | L2S (in-sample) minus full query | 64 | -1.8225 | -2.0876 | -1.5695 | 0.068 | 0.932 |
| nfcorpus | multi-positive | 102 | held_logrank | held-out transfer (1-1) minus full query | 64 | -0.4314 | -0.6322 | -0.2440 | 0.372 | 0.628 |
| nfcorpus | multi-positive | 102 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | -0.9411 | -1.1085 | -0.7900 | 0.090 | 0.910 |
| nfcorpus | multi-positive | 102 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1821 | -0.2279 | -0.1368 | 0.091 | 0.598 |
| nfcorpus | multi-positive | 102 | held_recall10 | L2S (in-sample) minus full query | 64 | 0.2338 | 0.1845 | 0.2846 | 0.540 | 0.038 |
| nfcorpus | multi-positive | 102 | held_recall10 | held-out transfer (1-1) minus full query | 64 | 0.0517 | 0.0202 | 0.0854 | 0.338 | 0.197 |
| nfcorpus | multi-positive | 102 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1316 | 0.0946 | 0.1721 | 0.443 | 0.038 |
| nfcorpus | multi-positive | 102 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.2385 | -0.2896 | -0.1890 | 0.045 | 0.751 |
| nfcorpus | multi-positive | 102 | held_recall100 | L2S (in-sample) minus full query | 64 | 0.3116 | 0.2518 | 0.3708 | 0.763 | 0.045 |
| nfcorpus | multi-positive | 102 | held_recall100 | held-out transfer (1-1) minus full query | 64 | 0.0732 | 0.0306 | 0.1158 | 0.449 | 0.183 |
| nfcorpus | multi-positive | 102 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1261 | 0.0902 | 0.1644 | 0.525 | 0.106 |
| nfcorpus | multi-positive | 204 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1148 | -0.1375 | -0.0915 | 0.113 | 0.887 |
| nfcorpus | multi-positive | 204 | held_rr | L2S (in-sample) minus full query | 64 | 0.1176 | 0.0917 | 0.1439 | 0.827 | 0.173 |
| nfcorpus | multi-positive | 204 | held_rr | held-out transfer (1-1) minus full query | 64 | 0.0028 | -0.0122 | 0.0184 | 0.515 | 0.485 |
| nfcorpus | multi-positive | 204 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0987 | 0.0803 | 0.1183 | 0.902 | 0.098 |
| nfcorpus | multi-positive | 204 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 64 | 1.3258 | 1.1399 | 1.5332 | 0.932 | 0.068 |
| nfcorpus | multi-positive | 204 | held_logrank | L2S (in-sample) minus full query | 64 | -1.8240 | -2.0819 | -1.5750 | 0.045 | 0.955 |
| nfcorpus | multi-positive | 204 | held_logrank | held-out transfer (1-1) minus full query | 64 | -0.4982 | -0.6885 | -0.3228 | 0.294 | 0.706 |
| nfcorpus | multi-positive | 204 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | -0.9297 | -1.0892 | -0.7899 | 0.090 | 0.910 |
| nfcorpus | multi-positive | 204 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1774 | -0.2222 | -0.1331 | 0.053 | 0.550 |
| nfcorpus | multi-positive | 204 | held_recall10 | L2S (in-sample) minus full query | 64 | 0.2295 | 0.1806 | 0.2809 | 0.537 | 0.038 |
| nfcorpus | multi-positive | 204 | held_recall10 | held-out transfer (1-1) minus full query | 64 | 0.0522 | 0.0212 | 0.0866 | 0.307 | 0.175 |
| nfcorpus | multi-positive | 204 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1387 | 0.0991 | 0.1788 | 0.464 | 0.060 |
| nfcorpus | multi-positive | 204 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1905 | -0.2419 | -0.1392 | 0.090 | 0.637 |
| nfcorpus | multi-positive | 204 | held_recall100 | L2S (in-sample) minus full query | 64 | 0.2921 | 0.2316 | 0.3516 | 0.702 | 0.045 |
| nfcorpus | multi-positive | 204 | held_recall100 | held-out transfer (1-1) minus full query | 64 | 0.1015 | 0.0594 | 0.1459 | 0.503 | 0.132 |
| nfcorpus | multi-positive | 204 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1281 | 0.0930 | 0.1638 | 0.561 | 0.076 |
| nfcorpus | multi-positive | 307 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1068 | -0.1276 | -0.0863 | 0.128 | 0.872 |
| nfcorpus | multi-positive | 307 | held_rr | L2S (in-sample) minus full query | 64 | 0.1150 | 0.0913 | 0.1387 | 0.849 | 0.151 |
| nfcorpus | multi-positive | 307 | held_rr | held-out transfer (1-1) minus full query | 64 | 0.0082 | -0.0060 | 0.0237 | 0.591 | 0.409 |
| nfcorpus | multi-positive | 307 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0914 | 0.0732 | 0.1117 | 0.872 | 0.128 |
| nfcorpus | multi-positive | 307 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 64 | 1.2597 | 1.0666 | 1.4669 | 0.955 | 0.045 |
| nfcorpus | multi-positive | 307 | held_logrank | L2S (in-sample) minus full query | 64 | -1.7741 | -2.0274 | -1.5240 | 0.045 | 0.955 |
| nfcorpus | multi-positive | 307 | held_logrank | held-out transfer (1-1) minus full query | 64 | -0.5144 | -0.6946 | -0.3431 | 0.279 | 0.721 |
| nfcorpus | multi-positive | 307 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | -0.8915 | -1.0516 | -0.7435 | 0.090 | 0.910 |
| nfcorpus | multi-positive | 307 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1670 | -0.2106 | -0.1245 | 0.038 | 0.476 |
| nfcorpus | multi-positive | 307 | held_recall10 | L2S (in-sample) minus full query | 64 | 0.2215 | 0.1759 | 0.2693 | 0.478 | 0.038 |
| nfcorpus | multi-positive | 307 | held_recall10 | held-out transfer (1-1) minus full query | 64 | 0.0546 | 0.0236 | 0.0856 | 0.269 | 0.129 |
| nfcorpus | multi-positive | 307 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1335 | 0.0943 | 0.1716 | 0.442 | 0.060 |
| nfcorpus | multi-positive | 307 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1996 | -0.2526 | -0.1508 | 0.068 | 0.623 |
| nfcorpus | multi-positive | 307 | held_recall100 | L2S (in-sample) minus full query | 64 | 0.2911 | 0.2349 | 0.3485 | 0.747 | 0.023 |
| nfcorpus | multi-positive | 307 | held_recall100 | held-out transfer (1-1) minus full query | 64 | 0.0914 | 0.0466 | 0.1373 | 0.479 | 0.138 |
| nfcorpus | multi-positive | 307 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1293 | 0.0887 | 0.1693 | 0.517 | 0.083 |
| nfcorpus | multi-positive | 409 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.0996 | -0.1187 | -0.0808 | 0.113 | 0.887 |
| nfcorpus | multi-positive | 409 | held_rr | L2S (in-sample) minus full query | 64 | 0.1082 | 0.0857 | 0.1306 | 0.849 | 0.151 |
| nfcorpus | multi-positive | 409 | held_rr | held-out transfer (1-1) minus full query | 64 | 0.0086 | -0.0049 | 0.0238 | 0.569 | 0.431 |
| nfcorpus | multi-positive | 409 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0895 | 0.0709 | 0.1098 | 0.872 | 0.128 |
| nfcorpus | multi-positive | 409 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 64 | 1.2268 | 1.0398 | 1.4240 | 0.977 | 0.023 |
| nfcorpus | multi-positive | 409 | held_logrank | L2S (in-sample) minus full query | 64 | -1.7311 | -1.9818 | -1.4864 | 0.045 | 0.955 |
| nfcorpus | multi-positive | 409 | held_logrank | held-out transfer (1-1) minus full query | 64 | -0.5043 | -0.6837 | -0.3386 | 0.348 | 0.652 |
| nfcorpus | multi-positive | 409 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | -0.8783 | -1.0398 | -0.7290 | 0.090 | 0.910 |
| nfcorpus | multi-positive | 409 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1694 | -0.2155 | -0.1237 | 0.023 | 0.460 |
| nfcorpus | multi-positive | 409 | held_recall10 | L2S (in-sample) minus full query | 64 | 0.2221 | 0.1734 | 0.2724 | 0.492 | 0.045 |
| nfcorpus | multi-positive | 409 | held_recall10 | held-out transfer (1-1) minus full query | 64 | 0.0527 | 0.0229 | 0.0834 | 0.254 | 0.114 |
| nfcorpus | multi-positive | 409 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1285 | 0.0891 | 0.1686 | 0.404 | 0.060 |
| nfcorpus | multi-positive | 409 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1745 | -0.2234 | -0.1274 | 0.121 | 0.600 |
| nfcorpus | multi-positive | 409 | held_recall100 | L2S (in-sample) minus full query | 64 | 0.2702 | 0.2158 | 0.3252 | 0.702 | 0.045 |
| nfcorpus | multi-positive | 409 | held_recall100 | held-out transfer (1-1) minus full query | 64 | 0.0956 | 0.0551 | 0.1388 | 0.464 | 0.138 |
| nfcorpus | multi-positive | 409 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1289 | 0.0884 | 0.1693 | 0.538 | 0.098 |
| nfcorpus | multi-positive | 512 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1010 | -0.1185 | -0.0836 | 0.090 | 0.910 |
| nfcorpus | multi-positive | 512 | held_rr | L2S (in-sample) minus full query | 64 | 0.1101 | 0.0881 | 0.1314 | 0.894 | 0.106 |
| nfcorpus | multi-positive | 512 | held_rr | held-out transfer (1-1) minus full query | 64 | 0.0091 | -0.0034 | 0.0227 | 0.608 | 0.392 |
| nfcorpus | multi-positive | 512 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0881 | 0.0713 | 0.1072 | 0.857 | 0.143 |
| nfcorpus | multi-positive | 512 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 64 | 1.1767 | 0.9892 | 1.3753 | 0.932 | 0.068 |
| nfcorpus | multi-positive | 512 | held_logrank | L2S (in-sample) minus full query | 64 | -1.6799 | -1.9276 | -1.4393 | 0.068 | 0.932 |
| nfcorpus | multi-positive | 512 | held_logrank | held-out transfer (1-1) minus full query | 64 | -0.5032 | -0.6726 | -0.3416 | 0.317 | 0.683 |
| nfcorpus | multi-positive | 512 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | -0.8370 | -0.9942 | -0.6963 | 0.113 | 0.887 |
| nfcorpus | multi-positive | 512 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1595 | -0.2076 | -0.1143 | 0.038 | 0.481 |
| nfcorpus | multi-positive | 512 | held_recall10 | L2S (in-sample) minus full query | 64 | 0.2126 | 0.1630 | 0.2627 | 0.467 | 0.015 |
| nfcorpus | multi-positive | 512 | held_recall10 | held-out transfer (1-1) minus full query | 64 | 0.0531 | 0.0228 | 0.0843 | 0.254 | 0.114 |
| nfcorpus | multi-positive | 512 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1260 | 0.0884 | 0.1648 | 0.427 | 0.060 |
| nfcorpus | multi-positive | 512 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1704 | -0.2174 | -0.1238 | 0.083 | 0.608 |
| nfcorpus | multi-positive | 512 | held_recall100 | L2S (in-sample) minus full query | 64 | 0.2692 | 0.2140 | 0.3239 | 0.710 | 0.023 |
| nfcorpus | multi-positive | 512 | held_recall100 | held-out transfer (1-1) minus full query | 64 | 0.0987 | 0.0590 | 0.1403 | 0.473 | 0.107 |
| nfcorpus | multi-positive | 512 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1135 | 0.0705 | 0.1584 | 0.455 | 0.121 |
| nfcorpus | multi-positive | 614 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.0905 | -0.1080 | -0.0731 | 0.136 | 0.864 |
| nfcorpus | multi-positive | 614 | held_rr | L2S (in-sample) minus full query | 64 | 0.1009 | 0.0809 | 0.1216 | 0.940 | 0.060 |
| nfcorpus | multi-positive | 614 | held_rr | held-out transfer (1-1) minus full query | 64 | 0.0104 | -0.0015 | 0.0234 | 0.601 | 0.399 |
| nfcorpus | multi-positive | 614 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0831 | 0.0662 | 0.1014 | 0.834 | 0.166 |
| nfcorpus | multi-positive | 614 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 64 | 1.0683 | 0.8898 | 1.2656 | 0.887 | 0.113 |
| nfcorpus | multi-positive | 614 | held_logrank | L2S (in-sample) minus full query | 64 | -1.5651 | -1.8134 | -1.3337 | 0.023 | 0.977 |
| nfcorpus | multi-positive | 614 | held_logrank | held-out transfer (1-1) minus full query | 64 | -0.4968 | -0.6586 | -0.3417 | 0.302 | 0.698 |
| nfcorpus | multi-positive | 614 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | -0.7986 | -0.9528 | -0.6574 | 0.136 | 0.864 |
| nfcorpus | multi-positive | 614 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1368 | -0.1787 | -0.0966 | 0.060 | 0.433 |
| nfcorpus | multi-positive | 614 | held_recall10 | L2S (in-sample) minus full query | 64 | 0.1904 | 0.1447 | 0.2372 | 0.428 | 0.015 |
| nfcorpus | multi-positive | 614 | held_recall10 | held-out transfer (1-1) minus full query | 64 | 0.0536 | 0.0241 | 0.0842 | 0.224 | 0.083 |
| nfcorpus | multi-positive | 614 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1208 | 0.0825 | 0.1593 | 0.373 | 0.060 |
| nfcorpus | multi-positive | 614 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1635 | -0.2107 | -0.1201 | 0.083 | 0.585 |
| nfcorpus | multi-positive | 614 | held_recall100 | L2S (in-sample) minus full query | 64 | 0.2557 | 0.2008 | 0.3098 | 0.672 | 0.023 |
| nfcorpus | multi-positive | 614 | held_recall100 | held-out transfer (1-1) minus full query | 64 | 0.0921 | 0.0546 | 0.1310 | 0.473 | 0.107 |
| nfcorpus | multi-positive | 614 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1088 | 0.0691 | 0.1516 | 0.463 | 0.143 |
| nfcorpus | multi-positive | 716 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.0719 | -0.0911 | -0.0536 | 0.204 | 0.796 |
| nfcorpus | multi-positive | 716 | held_rr | L2S (in-sample) minus full query | 64 | 0.0848 | 0.0633 | 0.1068 | 0.902 | 0.098 |
| nfcorpus | multi-positive | 716 | held_rr | held-out transfer (1-1) minus full query | 64 | 0.0128 | -0.0005 | 0.0288 | 0.617 | 0.383 |
| nfcorpus | multi-positive | 716 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0753 | 0.0589 | 0.0929 | 0.849 | 0.151 |
| nfcorpus | multi-positive | 716 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 64 | 0.9013 | 0.7371 | 1.0894 | 0.887 | 0.113 |
| nfcorpus | multi-positive | 716 | held_logrank | L2S (in-sample) minus full query | 64 | -1.3905 | -1.6180 | -1.1709 | 0.000 | 1.000 |
| nfcorpus | multi-positive | 716 | held_logrank | held-out transfer (1-1) minus full query | 64 | -0.4892 | -0.6466 | -0.3394 | 0.310 | 0.690 |
| nfcorpus | multi-positive | 716 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | -0.7593 | -0.9049 | -0.6259 | 0.136 | 0.864 |
| nfcorpus | multi-positive | 716 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1169 | -0.1531 | -0.0812 | 0.000 | 0.388 |
| nfcorpus | multi-positive | 716 | held_recall10 | L2S (in-sample) minus full query | 64 | 0.1611 | 0.1170 | 0.2061 | 0.353 | 0.000 |
| nfcorpus | multi-positive | 716 | held_recall10 | held-out transfer (1-1) minus full query | 64 | 0.0442 | 0.0173 | 0.0738 | 0.164 | 0.083 |
| nfcorpus | multi-positive | 716 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1247 | 0.0877 | 0.1619 | 0.373 | 0.000 |
| nfcorpus | multi-positive | 716 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1360 | -0.1791 | -0.0947 | 0.083 | 0.531 |
| nfcorpus | multi-positive | 716 | held_recall100 | L2S (in-sample) minus full query | 64 | 0.2272 | 0.1749 | 0.2800 | 0.640 | 0.023 |
| nfcorpus | multi-positive | 716 | held_recall100 | held-out transfer (1-1) minus full query | 64 | 0.0912 | 0.0571 | 0.1270 | 0.458 | 0.069 |
| nfcorpus | multi-positive | 716 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1024 | 0.0639 | 0.1422 | 0.424 | 0.106 |
| nfcorpus | multi-positive | 819 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.0490 | -0.0660 | -0.0315 | 0.196 | 0.804 |
| nfcorpus | multi-positive | 819 | held_rr | L2S (in-sample) minus full query | 64 | 0.0650 | 0.0458 | 0.0856 | 0.864 | 0.136 |
| nfcorpus | multi-positive | 819 | held_rr | held-out transfer (1-1) minus full query | 64 | 0.0161 | 0.0029 | 0.0318 | 0.653 | 0.347 |
| nfcorpus | multi-positive | 819 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0576 | 0.0426 | 0.0727 | 0.819 | 0.181 |
| nfcorpus | multi-positive | 819 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 64 | 0.7227 | 0.5843 | 0.8811 | 0.872 | 0.128 |
| nfcorpus | multi-positive | 819 | held_logrank | L2S (in-sample) minus full query | 64 | -1.1620 | -1.3609 | -0.9756 | 0.045 | 0.955 |
| nfcorpus | multi-positive | 819 | held_logrank | held-out transfer (1-1) minus full query | 64 | -0.4394 | -0.5819 | -0.3056 | 0.288 | 0.712 |
| nfcorpus | multi-positive | 819 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | -0.6645 | -0.7891 | -0.5461 | 0.113 | 0.887 |
| nfcorpus | multi-positive | 819 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.0817 | -0.1161 | -0.0488 | 0.030 | 0.307 |
| nfcorpus | multi-positive | 819 | held_recall10 | L2S (in-sample) minus full query | 64 | 0.1248 | 0.0838 | 0.1679 | 0.289 | 0.000 |
| nfcorpus | multi-positive | 819 | held_recall10 | held-out transfer (1-1) minus full query | 64 | 0.0431 | 0.0167 | 0.0733 | 0.161 | 0.068 |
| nfcorpus | multi-positive | 819 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.1025 | 0.0689 | 0.1389 | 0.326 | 0.000 |
| nfcorpus | multi-positive | 819 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.1148 | -0.1576 | -0.0758 | 0.090 | 0.462 |
| nfcorpus | multi-positive | 819 | held_recall100 | L2S (in-sample) minus full query | 64 | 0.1928 | 0.1418 | 0.2460 | 0.547 | 0.023 |
| nfcorpus | multi-positive | 819 | held_recall100 | held-out transfer (1-1) minus full query | 64 | 0.0780 | 0.0437 | 0.1148 | 0.403 | 0.069 |
| nfcorpus | multi-positive | 819 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0860 | 0.0501 | 0.1224 | 0.356 | 0.090 |
| nfcorpus | multi-positive | 921 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.0285 | -0.0424 | -0.0143 | 0.241 | 0.759 |
| nfcorpus | multi-positive | 921 | held_rr | L2S (in-sample) minus full query | 64 | 0.0409 | 0.0258 | 0.0587 | 0.842 | 0.158 |
| nfcorpus | multi-positive | 921 | held_rr | held-out transfer (1-1) minus full query | 64 | 0.0124 | 0.0001 | 0.0274 | 0.633 | 0.367 |
| nfcorpus | multi-positive | 921 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0291 | 0.0171 | 0.0416 | 0.789 | 0.211 |
| nfcorpus | multi-positive | 921 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 64 | 0.4699 | 0.3678 | 0.5814 | 0.864 | 0.136 |
| nfcorpus | multi-positive | 921 | held_logrank | L2S (in-sample) minus full query | 64 | -0.8009 | -0.9485 | -0.6621 | 0.068 | 0.932 |
| nfcorpus | multi-positive | 921 | held_logrank | held-out transfer (1-1) minus full query | 64 | -0.3310 | -0.4288 | -0.2343 | 0.293 | 0.707 |
| nfcorpus | multi-positive | 921 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | -0.4383 | -0.5279 | -0.3557 | 0.158 | 0.842 |
| nfcorpus | multi-positive | 921 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.0573 | -0.0849 | -0.0312 | 0.015 | 0.223 |
| nfcorpus | multi-positive | 921 | held_recall10 | L2S (in-sample) minus full query | 64 | 0.0875 | 0.0541 | 0.1240 | 0.248 | 0.000 |
| nfcorpus | multi-positive | 921 | held_recall10 | held-out transfer (1-1) minus full query | 64 | 0.0303 | 0.0091 | 0.0556 | 0.128 | 0.023 |
| nfcorpus | multi-positive | 921 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0503 | 0.0262 | 0.0769 | 0.214 | 0.023 |
| nfcorpus | multi-positive | 921 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 64 | -0.0762 | -0.1126 | -0.0413 | 0.083 | 0.292 |
| nfcorpus | multi-positive | 921 | held_recall100 | L2S (in-sample) minus full query | 64 | 0.1396 | 0.0951 | 0.1850 | 0.469 | 0.023 |
| nfcorpus | multi-positive | 921 | held_recall100 | held-out transfer (1-1) minus full query | 64 | 0.0635 | 0.0334 | 0.0963 | 0.325 | 0.047 |
| nfcorpus | multi-positive | 921 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 64 | 0.0543 | 0.0271 | 0.0841 | 0.254 | 0.068 |

## Positive ranks, and how scores moved against the full query

| environment | cohort | k | arm | rr | logrank | recall10 | recall100 | positive score change | at rank 1 change | at rank 10 change | at rank 100 change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nfcorpus | single-positive | 102 | prefix-k | 0.1757 (0.0984 to 0.2536) | 4.2610 (3.6192 to 4.8830) | 0.2656 (0.1562 to 0.3750) | 0.5469 (0.4375 to 0.6719) | -0.3506 (-0.3856 to -0.3149) | -0.4532 (-0.4764 to -0.4293) | -0.3625 (-0.3804 to -0.3452) | -0.2944 (-0.3078 to -0.2814) |
| nfcorpus | single-positive | 102 | all-positive oracle (privileged) | 0.9844 (0.9609 to 1.0000) | 0.0217 (0.0000 to 0.0542) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1170 (-0.1472 to -0.0861) | -0.2451 (-0.2650 to -0.2242) | -0.2239 (-0.2384 to -0.2098) | -0.1767 (-0.1880 to -0.1658) |
| nfcorpus | single-positive | 102 | L2S (in-sample) | 0.9767 (0.9380 to 1.0000) | 0.0815 (0.0000 to 0.2336) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.2009 (-0.2316 to -0.1689) | -0.3276 (-0.3491 to -0.3050) | -0.3091 (-0.3245 to -0.2945) | -0.2627 (-0.2760 to -0.2500) |
| nfcorpus | single-positive | 102 | pair-optimal, target only (secondary) | 0.9844 (0.9609 to 1.0000) | 0.0217 (0.0000 to 0.0542) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1170 (-0.1472 to -0.0861) | -0.2451 (-0.2650 to -0.2242) | -0.2239 (-0.2384 to -0.2098) | -0.1767 (-0.1880 to -0.1658) |
| nfcorpus | single-positive | 204 | prefix-k | 0.2374 (0.1516 to 0.3273) | 3.9075 (3.2288 to 4.6006) | 0.3438 (0.2344 to 0.4688) | 0.5312 (0.4062 to 0.6562) | -0.3142 (-0.3460 to -0.2818) | -0.4093 (-0.4300 to -0.3881) | -0.3256 (-0.3417 to -0.3100) | -0.2630 (-0.2748 to -0.2513) |
| nfcorpus | single-positive | 204 | all-positive oracle (privileged) | 0.9788 (0.9442 to 1.0000) | 0.0412 (0.0000 to 0.1129) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0063 (-0.0183 to 0.0320) | -0.1213 (-0.1407 to -0.1018) | -0.1390 (-0.1502 to -0.1284) | -0.1061 (-0.1148 to -0.0975) |
| nfcorpus | single-positive | 204 | L2S (in-sample) | 0.9689 (0.9299 to 1.0000) | 0.0978 (0.0000 to 0.2610) | 0.9844 (0.9531 to 1.0000) | 0.9844 (0.9531 to 1.0000) | -0.1175 (-0.1437 to -0.0906) | -0.2432 (-0.2634 to -0.2222) | -0.2565 (-0.2701 to -0.2439) | -0.2232 (-0.2350 to -0.2117) |
| nfcorpus | single-positive | 204 | pair-optimal, target only (secondary) | 0.9788 (0.9442 to 1.0000) | 0.0412 (0.0000 to 0.1129) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0063 (-0.0183 to 0.0320) | -0.1213 (-0.1407 to -0.1018) | -0.1390 (-0.1502 to -0.1284) | -0.1061 (-0.1148 to -0.0975) |
| nfcorpus | single-positive | 307 | prefix-k | 0.2518 (0.1668 to 0.3411) | 3.7364 (3.0345 to 4.4060) | 0.3594 (0.2500 to 0.4844) | 0.5938 (0.4688 to 0.7188) | -0.2795 (-0.3079 to -0.2504) | -0.3638 (-0.3822 to -0.3448) | -0.2883 (-0.3034 to -0.2740) | -0.2329 (-0.2439 to -0.2224) |
| nfcorpus | single-positive | 307 | all-positive oracle (privileged) | 0.9702 (0.9325 to 1.0000) | 0.0591 (0.0000 to 0.1449) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0766 (0.0556 to 0.0983) | -0.0506 (-0.0694 to -0.0315) | -0.0868 (-0.0963 to -0.0780) | -0.0635 (-0.0711 to -0.0560) |
| nfcorpus | single-positive | 307 | L2S (in-sample) | 0.9689 (0.9299 to 1.0000) | 0.1003 (0.0000 to 0.2683) | 0.9844 (0.9531 to 1.0000) | 0.9844 (0.9531 to 1.0000) | -0.0662 (-0.0892 to -0.0425) | -0.1912 (-0.2108 to -0.1711) | -0.2225 (-0.2349 to -0.2108) | -0.1962 (-0.2073 to -0.1859) |
| nfcorpus | single-positive | 307 | pair-optimal, target only (secondary) | 0.9702 (0.9325 to 1.0000) | 0.0591 (0.0000 to 0.1449) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0766 (0.0556 to 0.0983) | -0.0506 (-0.0694 to -0.0315) | -0.0868 (-0.0963 to -0.0780) | -0.0635 (-0.0711 to -0.0560) |
| nfcorpus | single-positive | 409 | prefix-k | 0.2743 (0.1835 to 0.3703) | 3.6598 (2.9618 to 4.3512) | 0.3750 (0.2656 to 0.5000) | 0.6406 (0.5156 to 0.7500) | -0.2322 (-0.2568 to -0.2075) | -0.3021 (-0.3184 to -0.2855) | -0.2383 (-0.2514 to -0.2261) | -0.1916 (-0.2013 to -0.1826) |
| nfcorpus | single-positive | 409 | all-positive oracle (privileged) | 0.9543 (0.9111 to 0.9857) | 0.0834 (0.0214 to 0.1820) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1156 (0.0975 to 0.1348) | -0.0109 (-0.0300 to 0.0084) | -0.0548 (-0.0635 to -0.0469) | -0.0372 (-0.0437 to -0.0308) |
| nfcorpus | single-positive | 409 | L2S (in-sample) | 0.9610 (0.9217 to 0.9922) | 0.1141 (0.0108 to 0.2989) | 0.9844 (0.9531 to 1.0000) | 0.9844 (0.9531 to 1.0000) | -0.0301 (-0.0508 to -0.0091) | -0.1544 (-0.1730 to -0.1350) | -0.1941 (-0.2056 to -0.1831) | -0.1733 (-0.1833 to -0.1639) |
| nfcorpus | single-positive | 409 | pair-optimal, target only (secondary) | 0.9543 (0.9111 to 0.9857) | 0.0834 (0.0214 to 0.1820) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1156 (0.0975 to 0.1348) | -0.0109 (-0.0300 to 0.0084) | -0.0548 (-0.0635 to -0.0469) | -0.0372 (-0.0437 to -0.0308) |
| nfcorpus | single-positive | 512 | prefix-k | 0.2628 (0.1775 to 0.3568) | 3.6668 (2.9795 to 4.3434) | 0.3906 (0.2812 to 0.5156) | 0.6094 (0.4844 to 0.7344) | -0.1887 (-0.2088 to -0.1678) | -0.2451 (-0.2589 to -0.2305) | -0.1924 (-0.2032 to -0.1821) | -0.1548 (-0.1628 to -0.1473) |
| nfcorpus | single-positive | 512 | all-positive oracle (privileged) | 0.9462 (0.9002 to 0.9844) | 0.0993 (0.0217 to 0.2113) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1354 (0.1191 to 0.1526) | 0.0098 (-0.0091 to 0.0291) | -0.0360 (-0.0439 to -0.0286) | -0.0213 (-0.0272 to -0.0156) |
| nfcorpus | single-positive | 512 | L2S (in-sample) | 0.9532 (0.9065 to 0.9846) | 0.1281 (0.0214 to 0.3193) | 0.9844 (0.9531 to 1.0000) | 0.9844 (0.9531 to 1.0000) | -0.0036 (-0.0221 to 0.0154) | -0.1269 (-0.1457 to -0.1077) | -0.1696 (-0.1797 to -0.1593) | -0.1534 (-0.1626 to -0.1447) |
| nfcorpus | single-positive | 512 | pair-optimal, target only (secondary) | 0.9462 (0.9002 to 0.9844) | 0.0993 (0.0217 to 0.2113) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1354 (0.1191 to 0.1526) | 0.0098 (-0.0091 to 0.0291) | -0.0360 (-0.0439 to -0.0286) | -0.0213 (-0.0272 to -0.0156) |
| nfcorpus | single-positive | 614 | prefix-k | 0.2674 (0.1801 to 0.3585) | 3.6170 (2.9366 to 4.3150) | 0.4062 (0.2969 to 0.5312) | 0.5938 (0.4688 to 0.7188) | -0.1478 (-0.1644 to -0.1311) | -0.1913 (-0.2028 to -0.1793) | -0.1512 (-0.1602 to -0.1429) | -0.1206 (-0.1271 to -0.1147) |
| nfcorpus | single-positive | 614 | all-positive oracle (privileged) | 0.9356 (0.8849 to 0.9766) | 0.1203 (0.0325 to 0.2463) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1425 (0.1276 to 0.1583) | 0.0176 (-0.0017 to 0.0366) | -0.0268 (-0.0345 to -0.0195) | -0.0139 (-0.0193 to -0.0085) |
| nfcorpus | single-positive | 614 | L2S (in-sample) | 0.9402 (0.8907 to 0.9818) | 0.1545 (0.0280 to 0.3623) | 0.9844 (0.9531 to 1.0000) | 0.9844 (0.9531 to 1.0000) | 0.0150 (-0.0018 to 0.0322) | -0.1067 (-0.1252 to -0.0872) | -0.1472 (-0.1569 to -0.1377) | -0.1347 (-0.1430 to -0.1266) |
| nfcorpus | single-positive | 614 | pair-optimal, target only (secondary) | 0.9356 (0.8849 to 0.9766) | 0.1203 (0.0325 to 0.2463) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1425 (0.1276 to 0.1583) | 0.0176 (-0.0017 to 0.0366) | -0.0268 (-0.0345 to -0.0195) | -0.0139 (-0.0193 to -0.0085) |
| nfcorpus | single-positive | 716 | prefix-k | 0.2824 (0.1930 to 0.3797) | 3.5619 (2.8834 to 4.2558) | 0.4062 (0.2969 to 0.5312) | 0.6094 (0.4844 to 0.7344) | -0.1137 (-0.1267 to -0.1007) | -0.1463 (-0.1557 to -0.1367) | -0.1154 (-0.1227 to -0.1087) | -0.0920 (-0.0975 to -0.0871) |
| nfcorpus | single-positive | 716 | all-positive oracle (privileged) | 0.9118 (0.8523 to 0.9635) | 0.1700 (0.0560 to 0.3141) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1414 (0.1275 to 0.1560) | 0.0185 (-0.0003 to 0.0371) | -0.0207 (-0.0278 to -0.0137) | -0.0094 (-0.0144 to -0.0045) |
| nfcorpus | single-positive | 716 | L2S (in-sample) | 0.8964 (0.8338 to 0.9531) | 0.2350 (0.0840 to 0.4627) | 0.9844 (0.9531 to 1.0000) | 0.9844 (0.9531 to 1.0000) | 0.0272 (0.0122 to 0.0425) | -0.0918 (-0.1105 to -0.0725) | -0.1254 (-0.1342 to -0.1166) | -0.1164 (-0.1239 to -0.1091) |
| nfcorpus | single-positive | 716 | pair-optimal, target only (secondary) | 0.9118 (0.8523 to 0.9635) | 0.1700 (0.0560 to 0.3141) | 0.9844 (0.9531 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1414 (0.1275 to 0.1560) | 0.0185 (-0.0003 to 0.0371) | -0.0207 (-0.0278 to -0.0137) | -0.0094 (-0.0144 to -0.0045) |
| nfcorpus | single-positive | 819 | prefix-k | 0.2823 (0.1929 to 0.3799) | 3.5733 (2.8853 to 4.2652) | 0.3906 (0.2812 to 0.5156) | 0.5938 (0.4688 to 0.7188) | -0.0732 (-0.0819 to -0.0643) | -0.0932 (-0.0998 to -0.0868) | -0.0734 (-0.0787 to -0.0686) | -0.0582 (-0.0620 to -0.0546) |
| nfcorpus | single-positive | 819 | all-positive oracle (privileged) | 0.7915 (0.7130 to 0.8672) | 0.4298 (0.2426 to 0.6507) | 0.9688 (0.9219 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1300 (0.1180 to 0.1427) | 0.0145 (-0.0024 to 0.0315) | -0.0153 (-0.0213 to -0.0097) | -0.0052 (-0.0095 to -0.0010) |
| nfcorpus | single-positive | 819 | L2S (in-sample) | 0.7566 (0.6708 to 0.8414) | 0.5527 (0.3178 to 0.8367) | 0.9531 (0.8906 to 1.0000) | 0.9844 (0.9531 to 1.0000) | 0.0329 (0.0205 to 0.0457) | -0.0765 (-0.0926 to -0.0595) | -0.1015 (-0.1086 to -0.0944) | -0.0942 (-0.1008 to -0.0881) |
| nfcorpus | single-positive | 819 | pair-optimal, target only (secondary) | 0.7915 (0.7130 to 0.8672) | 0.4298 (0.2426 to 0.6507) | 0.9688 (0.9219 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1300 (0.1180 to 0.1427) | 0.0145 (-0.0024 to 0.0315) | -0.0153 (-0.0213 to -0.0097) | -0.0052 (-0.0095 to -0.0010) |
| nfcorpus | single-positive | 921 | prefix-k | 0.2797 (0.1891 to 0.3754) | 3.5910 (2.9197 to 4.2642) | 0.4062 (0.2969 to 0.5312) | 0.5938 (0.4688 to 0.7188) | -0.0342 (-0.0391 to -0.0292) | -0.0441 (-0.0483 to -0.0399) | -0.0337 (-0.0369 to -0.0306) | -0.0255 (-0.0277 to -0.0236) |
| nfcorpus | single-positive | 921 | all-positive oracle (privileged) | 0.6470 (0.5439 to 0.7534) | 1.0655 (0.6927 to 1.4466) | 0.7812 (0.6719 to 0.8750) | 0.9531 (0.8906 to 1.0000) | 0.0991 (0.0905 to 0.1081) | 0.0073 (-0.0047 to 0.0193) | -0.0093 (-0.0133 to -0.0054) | -0.0018 (-0.0046 to 0.0013) |
| nfcorpus | single-positive | 921 | L2S (in-sample) | 0.6229 (0.5222 to 0.7312) | 1.1695 (0.7765 to 1.5753) | 0.7344 (0.6250 to 0.8438) | 0.9531 (0.8906 to 1.0000) | 0.0296 (0.0207 to 0.0387) | -0.0562 (-0.0679 to -0.0442) | -0.0706 (-0.0765 to -0.0649) | -0.0659 (-0.0709 to -0.0612) |
| nfcorpus | single-positive | 921 | pair-optimal, target only (secondary) | 0.6470 (0.5439 to 0.7534) | 1.0655 (0.6927 to 1.4466) | 0.7812 (0.6719 to 0.8750) | 0.9531 (0.8906 to 1.0000) | 0.0991 (0.0905 to 0.1081) | 0.0073 (-0.0047 to 0.0193) | -0.0093 (-0.0133 to -0.0054) | -0.0018 (-0.0046 to 0.0013) |
| nfcorpus | multi-positive | 102 | prefix-k | 0.0493 (0.0363 to 0.0635) | 5.6635 (5.4056 to 5.9077) | 0.0993 (0.0721 to 0.1262) | 0.2724 (0.2269 to 0.3242) | -0.2866 (-0.3040 to -0.2680) | -0.4856 (-0.5095 to -0.4618) | -0.4017 (-0.4184 to -0.3843) | -0.3114 (-0.3224 to -0.2995) |
| nfcorpus | multi-positive | 102 | all-positive oracle (privileged) | 0.1604 (0.1396 to 0.1815) | 4.0042 (3.7317 to 4.2675) | 0.2947 (0.2510 to 0.3402) | 0.5618 (0.5078 to 0.6186) | -0.1165 (-0.1308 to -0.1017) | -0.2879 (-0.3052 to -0.2702) | -0.2282 (-0.2408 to -0.2153) | -0.1636 (-0.1723 to -0.1545) |
| nfcorpus | multi-positive | 102 | L2S (in-sample) | 0.1789 (0.1565 to 0.2008) | 3.5769 (3.3396 to 3.8119) | 0.3521 (0.3101 to 0.3949) | 0.6494 (0.6007 to 0.6990) | -0.2378 (-0.2524 to -0.2227) | -0.4280 (-0.4468 to -0.4086) | -0.3623 (-0.3756 to -0.3481) | -0.2940 (-0.3063 to -0.2811) |
| nfcorpus | multi-positive | 102 | pair-optimal, target only (secondary) | 0.9048 (0.8683 to 0.9371) | 0.1570 (0.1004 to 0.2227) | 0.9977 (0.9933 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0606 (-0.0759 to -0.0451) | n/a | n/a | n/a |
| nfcorpus | multi-positive | 204 | prefix-k | 0.0543 (0.0404 to 0.0693) | 5.5721 (5.2806 to 5.8316) | 0.1134 (0.0854 to 0.1433) | 0.2878 (0.2338 to 0.3472) | -0.2558 (-0.2713 to -0.2389) | -0.4359 (-0.4573 to -0.4140) | -0.3610 (-0.3762 to -0.3449) | -0.2783 (-0.2883 to -0.2675) |
| nfcorpus | multi-positive | 204 | all-positive oracle (privileged) | 0.1622 (0.1415 to 0.1838) | 3.9532 (3.6900 to 4.2187) | 0.3080 (0.2651 to 0.3517) | 0.5704 (0.5159 to 0.6251) | -0.0258 (-0.0374 to -0.0140) | -0.1672 (-0.1808 to -0.1532) | -0.1242 (-0.1337 to -0.1146) | -0.0797 (-0.0868 to -0.0727) |
| nfcorpus | multi-positive | 204 | L2S (in-sample) | 0.1826 (0.1595 to 0.2052) | 3.5409 (3.3103 to 3.7694) | 0.3510 (0.3064 to 0.3947) | 0.6508 (0.6056 to 0.6973) | -0.1906 (-0.2041 to -0.1767) | -0.3595 (-0.3781 to -0.3402) | -0.3072 (-0.3202 to -0.2937) | -0.2559 (-0.2675 to -0.2435) |
| nfcorpus | multi-positive | 204 | pair-optimal, target only (secondary) | 0.8956 (0.8552 to 0.9317) | 0.1749 (0.1117 to 0.2448) | 0.9979 (0.9938 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0516 (0.0389 to 0.0646) | n/a | n/a | n/a |
| nfcorpus | multi-positive | 307 | prefix-k | 0.0558 (0.0424 to 0.0705) | 5.5411 (5.2523 to 5.8092) | 0.1234 (0.0936 to 0.1553) | 0.2966 (0.2421 to 0.3579) | -0.2269 (-0.2409 to -0.2117) | -0.3877 (-0.4073 to -0.3675) | -0.3211 (-0.3350 to -0.3064) | -0.2467 (-0.2562 to -0.2366) |
| nfcorpus | multi-positive | 307 | all-positive oracle (privileged) | 0.1625 (0.1413 to 0.1846) | 3.9643 (3.6923 to 4.2298) | 0.3039 (0.2619 to 0.3456) | 0.5635 (0.5104 to 0.6169) | 0.0263 (0.0167 to 0.0363) | -0.0940 (-0.1056 to -0.0819) | -0.0638 (-0.0719 to -0.0556) | -0.0309 (-0.0368 to -0.0251) |
| nfcorpus | multi-positive | 307 | L2S (in-sample) | 0.1817 (0.1594 to 0.2035) | 3.5770 (3.3402 to 3.8088) | 0.3501 (0.3069 to 0.3938) | 0.6498 (0.6045 to 0.6959) | -0.1564 (-0.1687 to -0.1434) | -0.3087 (-0.3260 to -0.2901) | -0.2659 (-0.2786 to -0.2525) | -0.2259 (-0.2370 to -0.2140) |
| nfcorpus | multi-positive | 307 | pair-optimal, target only (secondary) | 0.8625 (0.8161 to 0.9046) | 0.2360 (0.1603 to 0.3200) | 0.9974 (0.9927 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.1142 (0.1035 to 0.1253) | n/a | n/a | n/a |
| nfcorpus | multi-positive | 409 | prefix-k | 0.0637 (0.0471 to 0.0810) | 5.4551 (5.1578 to 5.7310) | 0.1243 (0.0940 to 0.1584) | 0.3114 (0.2555 to 0.3724) | -0.1871 (-0.1992 to -0.1744) | -0.3213 (-0.3382 to -0.3038) | -0.2671 (-0.2793 to -0.2541) | -0.2038 (-0.2119 to -0.1952) |
| nfcorpus | multi-positive | 409 | all-positive oracle (privileged) | 0.1601 (0.1389 to 0.1827) | 4.0026 (3.7319 to 4.2701) | 0.3042 (0.2615 to 0.3482) | 0.5602 (0.5066 to 0.6128) | 0.0559 (0.0477 to 0.0644) | -0.0504 (-0.0611 to -0.0392) | -0.0276 (-0.0347 to -0.0208) | -0.0016 (-0.0067 to 0.0035) |
| nfcorpus | multi-positive | 409 | L2S (in-sample) | 0.1758 (0.1544 to 0.1974) | 3.6312 (3.3969 to 3.8700) | 0.3414 (0.2983 to 0.3844) | 0.6249 (0.5738 to 0.6752) | -0.1280 (-0.1393 to -0.1161) | -0.2655 (-0.2818 to -0.2479) | -0.2304 (-0.2428 to -0.2178) | -0.1999 (-0.2101 to -0.1893) |
| nfcorpus | multi-positive | 409 | pair-optimal, target only (secondary) | 0.8297 (0.7793 to 0.8762) | 0.3035 (0.2154 to 0.3986) | 0.9920 (0.9844 to 0.9981) | 1.0000 (1.0000 to 1.0000) | 0.1479 (0.1386 to 0.1578) | n/a | n/a | n/a |
| nfcorpus | multi-positive | 512 | prefix-k | 0.0591 (0.0457 to 0.0736) | 5.4534 (5.1637 to 5.7277) | 0.1324 (0.1026 to 0.1659) | 0.3163 (0.2610 to 0.3763) | -0.1515 (-0.1615 to -0.1409) | -0.2623 (-0.2762 to -0.2484) | -0.2183 (-0.2283 to -0.2078) | -0.1652 (-0.1718 to -0.1584) |
| nfcorpus | multi-positive | 512 | all-positive oracle (privileged) | 0.1563 (0.1360 to 0.1785) | 4.0496 (3.7830 to 4.3147) | 0.2959 (0.2543 to 0.3378) | 0.5494 (0.4957 to 0.6042) | 0.0717 (0.0645 to 0.0793) | -0.0247 (-0.0346 to -0.0145) | -0.0061 (-0.0123 to -0.0003) | 0.0149 (0.0102 to 0.0197) |
| nfcorpus | multi-positive | 512 | L2S (in-sample) | 0.1723 (0.1525 to 0.1925) | 3.7054 (3.4644 to 3.9435) | 0.3314 (0.2891 to 0.3732) | 0.6165 (0.5687 to 0.6647) | -0.1039 (-0.1145 to -0.0929) | -0.2270 (-0.2432 to -0.2104) | -0.1993 (-0.2117 to -0.1874) | -0.1747 (-0.1849 to -0.1644) |
| nfcorpus | multi-positive | 512 | pair-optimal, target only (secondary) | 0.7835 (0.7301 to 0.8340) | 0.4010 (0.2967 to 0.5078) | 0.9854 (0.9757 to 0.9947) | 1.0000 (1.0000 to 1.0000) | 0.1640 (0.1554 to 0.1729) | n/a | n/a | n/a |
| nfcorpus | multi-positive | 614 | prefix-k | 0.0638 (0.0497 to 0.0788) | 5.4378 (5.1427 to 5.7147) | 0.1382 (0.1083 to 0.1714) | 0.3175 (0.2619 to 0.3812) | -0.1180 (-0.1265 to -0.1095) | -0.2062 (-0.2174 to -0.1945) | -0.1704 (-0.1789 to -0.1618) | -0.1290 (-0.1345 to -0.1236) |
| nfcorpus | multi-positive | 614 | all-positive oracle (privileged) | 0.1519 (0.1316 to 0.1742) | 4.1012 (3.8319 to 4.3667) | 0.2920 (0.2510 to 0.3339) | 0.5402 (0.4868 to 0.5952) | 0.0781 (0.0716 to 0.0850) | -0.0114 (-0.0206 to -0.0017) | 0.0041 (-0.0018 to 0.0098) | 0.0228 (0.0184 to 0.0272) |
| nfcorpus | multi-positive | 614 | L2S (in-sample) | 0.1598 (0.1403 to 0.1795) | 3.8302 (3.5820 to 4.0814) | 0.3123 (0.2706 to 0.3527) | 0.5882 (0.5364 to 0.6421) | -0.0829 (-0.0924 to -0.0735) | -0.1907 (-0.2052 to -0.1760) | -0.1697 (-0.1810 to -0.1584) | -0.1511 (-0.1605 to -0.1417) |
| nfcorpus | multi-positive | 614 | pair-optimal, target only (secondary) | 0.7362 (0.6808 to 0.7904) | 0.4984 (0.3839 to 0.6173) | 0.9728 (0.9580 to 0.9864) | 1.0000 (1.0000 to 1.0000) | 0.1687 (0.1608 to 0.1769) | n/a | n/a | n/a |
| nfcorpus | multi-positive | 716 | prefix-k | 0.0645 (0.0504 to 0.0795) | 5.4453 (5.1424 to 5.7299) | 0.1352 (0.1070 to 0.1682) | 0.3150 (0.2578 to 0.3763) | -0.0910 (-0.0981 to -0.0838) | -0.1578 (-0.1673 to -0.1481) | -0.1300 (-0.1374 to -0.1223) | -0.0990 (-0.1039 to -0.0939) |
| nfcorpus | multi-positive | 716 | all-positive oracle (privileged) | 0.1468 (0.1277 to 0.1678) | 4.1498 (3.8848 to 4.4132) | 0.2835 (0.2423 to 0.3264) | 0.5296 (0.4752 to 0.5853) | 0.0787 (0.0725 to 0.0851) | -0.0069 (-0.0158 to 0.0023) | 0.0078 (0.0021 to 0.0133) | 0.0248 (0.0206 to 0.0291) |
| nfcorpus | multi-positive | 716 | L2S (in-sample) | 0.1479 (0.1275 to 0.1684) | 3.9939 (3.7456 to 4.2532) | 0.2885 (0.2486 to 0.3288) | 0.5664 (0.5094 to 0.6242) | -0.0646 (-0.0729 to -0.0564) | -0.1557 (-0.1685 to -0.1424) | -0.1405 (-0.1509 to -0.1305) | -0.1277 (-0.1367 to -0.1194) |
| nfcorpus | multi-positive | 716 | pair-optimal, target only (secondary) | 0.6608 (0.6058 to 0.7169) | 0.6885 (0.5588 to 0.8169) | 0.9419 (0.9171 to 0.9656) | 1.0000 (1.0000 to 1.0000) | 0.1657 (0.1583 to 0.1735) | n/a | n/a | n/a |
| nfcorpus | multi-positive | 819 | prefix-k | 0.0670 (0.0509 to 0.0841) | 5.4333 (5.1329 to 5.7140) | 0.1346 (0.1031 to 0.1693) | 0.3173 (0.2603 to 0.3792) | -0.0577 (-0.0626 to -0.0526) | -0.1010 (-0.1077 to -0.0938) | -0.0834 (-0.0886 to -0.0779) | -0.0629 (-0.0665 to -0.0593) |
| nfcorpus | multi-positive | 819 | all-positive oracle (privileged) | 0.1317 (0.1113 to 0.1524) | 4.3102 (4.0341 to 4.5803) | 0.2557 (0.2167 to 0.2974) | 0.5024 (0.4464 to 0.5630) | 0.0734 (0.0679 to 0.0791) | -0.0031 (-0.0108 to 0.0050) | 0.0093 (0.0040 to 0.0145) | 0.0253 (0.0216 to 0.0289) |
| nfcorpus | multi-positive | 819 | L2S (in-sample) | 0.1292 (0.1090 to 0.1502) | 4.2336 (3.9736 to 4.4929) | 0.2437 (0.2048 to 0.2846) | 0.5277 (0.4701 to 0.5855) | -0.0460 (-0.0530 to -0.0393) | -0.1200 (-0.1315 to -0.1085) | -0.1096 (-0.1189 to -0.1007) | -0.1012 (-0.1093 to -0.0937) |
| nfcorpus | multi-positive | 819 | pair-optimal, target only (secondary) | 0.5189 (0.4677 to 0.5721) | 1.1672 (1.0092 to 1.3218) | 0.7880 (0.7358 to 0.8371) | 0.9970 (0.9928 to 0.9997) | 0.1510 (0.1446 to 0.1577) | n/a | n/a | n/a |
| nfcorpus | multi-positive | 921 | prefix-k | 0.0650 (0.0510 to 0.0800) | 5.4446 (5.1465 to 5.7219) | 0.1249 (0.0968 to 0.1546) | 0.3213 (0.2657 to 0.3823) | -0.0256 (-0.0282 to -0.0228) | -0.0467 (-0.0512 to -0.0420) | -0.0383 (-0.0414 to -0.0349) | -0.0277 (-0.0299 to -0.0255) |
| nfcorpus | multi-positive | 921 | all-positive oracle (privileged) | 0.1066 (0.0877 to 0.1262) | 4.6470 (4.3595 to 4.9231) | 0.1975 (0.1612 to 0.2359) | 0.4530 (0.3941 to 0.5131) | 0.0569 (0.0529 to 0.0611) | -0.0002 (-0.0056 to 0.0055) | 0.0097 (0.0055 to 0.0138) | 0.0205 (0.0176 to 0.0233) |
| nfcorpus | multi-positive | 921 | L2S (in-sample) | 0.1070 (0.0880 to 0.1272) | 4.6093 (4.3369 to 4.8719) | 0.2046 (0.1689 to 0.2430) | 0.4518 (0.3985 to 0.5091) | -0.0281 (-0.0334 to -0.0230) | -0.0801 (-0.0891 to -0.0712) | -0.0734 (-0.0802 to -0.0667) | -0.0687 (-0.0748 to -0.0627) |
| nfcorpus | multi-positive | 921 | pair-optimal, target only (secondary) | 0.2831 (0.2461 to 0.3246) | 2.4390 (2.2269 to 2.6533) | 0.4748 (0.4217 to 0.5290) | 0.8649 (0.8217 to 0.9043) | 0.1140 (0.1093 to 0.1187) | n/a | n/a | n/a |

## Agreement with the oracle mask

| environment | cohort | k | arm | Jaccard with oracle mask | score regret to ceiling / mass |
|---|---|---|---|---|---|
| nfcorpus | single-positive | 102 | prefix-k | 0.0812 (0.0773 to 0.0853) | 0.3307 (0.3257 to 0.3359) |
| nfcorpus | single-positive | 102 | L2S (in-sample) | 0.3678 (0.3544 to 0.3813) | 0.1193 (0.1138 to 0.1255) |
| nfcorpus | single-positive | 204 | prefix-k | 0.1155 (0.1128 to 0.1181) | 0.4538 (0.4485 to 0.4594) |
| nfcorpus | single-positive | 204 | L2S (in-sample) | 0.3952 (0.3830 to 0.4073) | 0.1762 (0.1683 to 0.1844) |
| nfcorpus | single-positive | 307 | prefix-k | 0.1713 (0.1682 to 0.1745) | 0.5042 (0.4983 to 0.5103) |
| nfcorpus | single-positive | 307 | L2S (in-sample) | 0.4216 (0.4112 to 0.4322) | 0.2030 (0.1950 to 0.2113) |
| nfcorpus | single-positive | 409 | prefix-k | 0.2512 (0.2479 to 0.2543) | 0.4932 (0.4869 to 0.4996) |
| nfcorpus | single-positive | 409 | L2S (in-sample) | 0.4555 (0.4482 to 0.4629) | 0.2072 (0.1999 to 0.2151) |
| nfcorpus | single-positive | 512 | prefix-k | 0.3347 (0.3313 to 0.3380) | 0.4602 (0.4548 to 0.4657) |
| nfcorpus | single-positive | 512 | L2S (in-sample) | 0.5043 (0.4981 to 0.5099) | 0.1975 (0.1906 to 0.2051) |
| nfcorpus | single-positive | 614 | prefix-k | 0.4322 (0.4287 to 0.4359) | 0.4132 (0.4058 to 0.4211) |
| nfcorpus | single-positive | 614 | L2S (in-sample) | 0.5718 (0.5659 to 0.5775) | 0.1813 (0.1750 to 0.1886) |
| nfcorpus | single-positive | 716 | prefix-k | 0.5399 (0.5372 to 0.5427) | 0.3640 (0.3546 to 0.3739) |
| nfcorpus | single-positive | 716 | L2S (in-sample) | 0.6590 (0.6520 to 0.6657) | 0.1626 (0.1561 to 0.1698) |
| nfcorpus | single-positive | 819 | prefix-k | 0.6687 (0.6668 to 0.6708) | 0.2910 (0.2803 to 0.3018) |
| nfcorpus | single-positive | 819 | L2S (in-sample) | 0.7618 (0.7550 to 0.7684) | 0.1384 (0.1328 to 0.1443) |
| nfcorpus | single-positive | 921 | prefix-k | 0.8188 (0.8175 to 0.8202) | 0.1915 (0.1819 to 0.2013) |
| nfcorpus | single-positive | 921 | L2S (in-sample) | 0.8736 (0.8694 to 0.8778) | 0.0991 (0.0945 to 0.1040) |
| nfcorpus | single-positive | 1024 | prefix-k | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| nfcorpus | single-positive | 1024 | L2S (in-sample) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| nfcorpus | multi-positive | 102 | prefix-k | 0.0866 (0.0832 to 0.0903) | 0.2484 (0.2426 to 0.2543) |
| nfcorpus | multi-positive | 102 | L2S (in-sample) | 0.1988 (0.1808 to 0.2187) | 0.1776 (0.1679 to 0.1868) |
| nfcorpus | multi-positive | 102 | held-out transfer (1-1) | 0.4734 (0.4614 to 0.4860) | n/a |
| nfcorpus | multi-positive | 204 | prefix-k | 0.1221 (0.1196 to 0.1246) | 0.3360 (0.3284 to 0.3433) |
| nfcorpus | multi-positive | 204 | L2S (in-sample) | 0.2380 (0.2218 to 0.2559) | 0.2413 (0.2297 to 0.2522) |
| nfcorpus | multi-positive | 204 | held-out transfer (1-1) | 0.5227 (0.5123 to 0.5333) | n/a |
| nfcorpus | multi-positive | 307 | prefix-k | 0.1759 (0.1727 to 0.1793) | 0.3699 (0.3615 to 0.3776) |
| nfcorpus | multi-positive | 307 | L2S (in-sample) | 0.2836 (0.2711 to 0.2973) | 0.2674 (0.2562 to 0.2780) |
| nfcorpus | multi-positive | 307 | held-out transfer (1-1) | 0.5607 (0.5512 to 0.5701) | n/a |
| nfcorpus | multi-positive | 409 | prefix-k | 0.2529 (0.2499 to 0.2559) | 0.3550 (0.3466 to 0.3626) |
| nfcorpus | multi-positive | 409 | L2S (in-sample) | 0.3370 (0.3266 to 0.3486) | 0.2692 (0.2590 to 0.2789) |
| nfcorpus | multi-positive | 409 | held-out transfer (1-1) | 0.5914 (0.5824 to 0.6005) | n/a |
| nfcorpus | multi-positive | 512 | prefix-k | 0.3377 (0.3350 to 0.3407) | 0.3263 (0.3186 to 0.3332) |
| nfcorpus | multi-positive | 512 | L2S (in-sample) | 0.3999 (0.3912 to 0.4089) | 0.2570 (0.2467 to 0.2673) |
| nfcorpus | multi-positive | 512 | held-out transfer (1-1) | 0.6212 (0.6122 to 0.6300) | n/a |
| nfcorpus | multi-positive | 614 | prefix-k | 0.4319 (0.4286 to 0.4353) | 0.2869 (0.2796 to 0.2937) |
| nfcorpus | multi-positive | 614 | L2S (in-sample) | 0.4815 (0.4752 to 0.4884) | 0.2359 (0.2262 to 0.2452) |
| nfcorpus | multi-positive | 614 | held-out transfer (1-1) | 0.6471 (0.6387 to 0.6553) | n/a |
| nfcorpus | multi-positive | 716 | prefix-k | 0.5405 (0.5378 to 0.5433) | 0.2483 (0.2418 to 0.2544) |
| nfcorpus | multi-positive | 716 | L2S (in-sample) | 0.5800 (0.5745 to 0.5856) | 0.2099 (0.2010 to 0.2189) |
| nfcorpus | multi-positive | 716 | held-out transfer (1-1) | 0.7055 (0.6988 to 0.7124) | n/a |
| nfcorpus | multi-positive | 819 | prefix-k | 0.6710 (0.6685 to 0.6734) | 0.1920 (0.1862 to 0.1978) |
| nfcorpus | multi-positive | 819 | L2S (in-sample) | 0.7009 (0.6964 to 0.7057) | 0.1751 (0.1674 to 0.1832) |
| nfcorpus | multi-positive | 819 | held-out transfer (1-1) | 0.7927 (0.7878 to 0.7979) | n/a |
| nfcorpus | multi-positive | 921 | prefix-k | 0.8185 (0.8173 to 0.8198) | 0.1210 (0.1163 to 0.1257) |
| nfcorpus | multi-positive | 921 | L2S (in-sample) | 0.8379 (0.8350 to 0.8410) | 0.1247 (0.1185 to 0.1315) |
| nfcorpus | multi-positive | 921 | held-out transfer (1-1) | 0.8856 (0.8826 to 0.8887) | n/a |
| nfcorpus | multi-positive | 1024 | prefix-k | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| nfcorpus | multi-positive | 1024 | L2S (in-sample) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| nfcorpus | multi-positive | 1024 | held-out transfer (1-1) | 1.0000 (1.0000 to 1.0000) | n/a |

## Relationship among a query's positives (multi-positive)

`pair_jaccard` is the mean overlap of pair-optimal masks. `sign_consensus` is the share of the oracle mask's coordinates on which all positives agree in sign, `positive_consensus` the share on which all are positive. `oracle_regret` is the regret of the all-positive mask against each positive's own optimum, `held_regret` of a mask built from a disjoint positive, each over mass.

| environment | k | queries | pair_jaccard | sign_consensus | positive_consensus | oracle_regret | held_regret | held_privileged_regret | held_oracle_jaccard |
|---|---|---|---|---|---|---|---|---|---|
| nfcorpus | 102 | 64 | 0.3281 (0.3179 to 0.3391) | 0.4022 (0.3697 to 0.4339) | 0.4022 (0.3697 to 0.4339) | 0.0824 (0.0785 to 0.0863) | 0.1561 (0.1496 to 0.1628) | 0.0850 (0.0808 to 0.0893) | 0.4734 (0.4614 to 0.4860) |
| nfcorpus | 204 | 64 | 0.3814 (0.3719 to 0.3909) | 0.3250 (0.2955 to 0.3554) | 0.3250 (0.2955 to 0.3554) | 0.1141 (0.1088 to 0.1194) | 0.2148 (0.2064 to 0.2234) | 0.1179 (0.1123 to 0.1235) | 0.5227 (0.5123 to 0.5333) |
| nfcorpus | 307 | 64 | 0.4245 (0.4160 to 0.4334) | 0.2766 (0.2494 to 0.3048) | 0.2766 (0.2494 to 0.3048) | 0.1297 (0.1238 to 0.1357) | 0.2434 (0.2341 to 0.2528) | 0.1344 (0.1281 to 0.1407) | 0.5607 (0.5512 to 0.5701) |
| nfcorpus | 409 | 64 | 0.4634 (0.4551 to 0.4719) | 0.2409 (0.2160 to 0.2666) | 0.2409 (0.2160 to 0.2666) | 0.1358 (0.1295 to 0.1421) | 0.2557 (0.2458 to 0.2657) | 0.1403 (0.1336 to 0.1472) | 0.5914 (0.5824 to 0.6005) |
| nfcorpus | 512 | 64 | 0.5013 (0.4937 to 0.5092) | 0.2133 (0.1909 to 0.2359) | 0.2133 (0.1909 to 0.2359) | 0.1361 (0.1297 to 0.1425) | 0.2565 (0.2465 to 0.2666) | 0.1405 (0.1336 to 0.1474) | 0.6212 (0.6122 to 0.6300) |
| nfcorpus | 614 | 64 | 0.5430 (0.5368 to 0.5495) | 0.1906 (0.1700 to 0.2114) | 0.1900 (0.1696 to 0.2110) | 0.1336 (0.1273 to 0.1400) | 0.2529 (0.2430 to 0.2629) | 0.1380 (0.1311 to 0.1449) | 0.6471 (0.6387 to 0.6553) |
| nfcorpus | 716 | 64 | 0.6294 (0.6245 to 0.6347) | 0.1708 (0.1525 to 0.1901) | 0.1656 (0.1477 to 0.1838) | 0.1284 (0.1223 to 0.1345) | 0.2421 (0.2326 to 0.2519) | 0.1326 (0.1260 to 0.1391) | 0.7055 (0.6988 to 0.7124) |
| nfcorpus | 819 | 64 | 0.7380 (0.7341 to 0.7419) | 0.1586 (0.1411 to 0.1768) | 0.1449 (0.1293 to 0.1608) | 0.1145 (0.1090 to 0.1198) | 0.2153 (0.2067 to 0.2241) | 0.1180 (0.1121 to 0.1237) | 0.7927 (0.7878 to 0.7979) |
| nfcorpus | 921 | 64 | 0.8563 (0.8541 to 0.8586) | 0.1533 (0.1359 to 0.1717) | 0.1289 (0.1150 to 0.1430) | 0.0841 (0.0802 to 0.0881) | 0.1568 (0.1507 to 0.1630) | 0.0869 (0.0825 to 0.0914) | 0.8856 (0.8826 to 0.8887) |
| nfcorpus | 1024 | 64 | 1.0000 (1.0000 to 1.0000) | 0.1558 (0.1378 to 0.1745) | 0.1159 (0.1034 to 0.1286) | 0.0000 (0.0000 to 0.0000) | 0.0000 (0.0000 to 0.0000) | 0.0000 (0.0000 to 0.0000) | 1.0000 (1.0000 to 1.0000) |

## Checks and cost

| environment | check | value |
|---|---|---|
| nfcorpus | boundary_note | "L2S ties at a mask boundary go to the lower dimension" |
| nfcorpus | documents_excluded_by_id | 0 |
| nfcorpus | float32_evaluation_gap | {"largest_ndcg10_gap": 0.0, "query_budget_pairs_above_1e-6": 0} |
| nfcorpus | ids_colliding_with_a_document | 0 |
| nfcorpus | k1024_list_mismatches | 0 |
| nfcorpus | k1024_rank_mismatches | 0 |
| nfcorpus | l2s_boundary_ties | 28 |
| nfcorpus | masks | 25418 |
| nfcorpus | masks_with_same_id_document | 0 |
| nfcorpus | oracle_ceiling_violations | 0 |
| nfcorpus | ranks_out_of_bounds | 0 |
| nfcorpus | same_id_list_inconsistencies | 0 |
| nfcorpus | single_positive_masks_identical_to_oracle | 64 |
| nfcorpus | single_positive_queries | 64 |
| nfcorpus | stored_run_parity | {"compared": 20521, "mismatched": 0, "run": "20261109T163931Z-9e9f5a0a"} |
| nfcorpus | tied_tracked_documents | 5659 |
| nfcorpus | tracked_ranks_charged_by_same_id_document | 0 |
| nfcorpus | total seconds | 3.6 |
| nfcorpus | device | cuda:0 |
