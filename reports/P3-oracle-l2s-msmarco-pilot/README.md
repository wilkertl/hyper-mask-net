# P3: exact positive masks beside Learning-to-Select

Privileged, paired, in-domain diagnostic on training queries and training judgments. L2S numbers are **in-sample**: L2S was fitted on these queries. Nothing here measures performance on unseen queries.

- Run `20261112T160235Z-298d5679` (commit `d362793ae9be`, dirty `False`, device `cuda:0`), environments ['msmarco'], query bound 32 (a bounded sample).
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
| msmarco | checkpoint fingerprint is the dataset's | True |
| msmarco | dataset is the F2 dataset read here | True |
| msmarco | embeddings are the F2 embeddings read here | True |
| msmarco | embeddings fingerprint is the dataset's | True |
| msmarco | environment matches | True |
| msmarco | model revision matches | True |
| msmarco | negatives are the latest F2 negatives | True |
| msmarco | negatives fingerprint is the dataset's | True |
| msmarco | official recipe | True |
| msmarco | one published checkpoint | True |
| msmarco | payload hash matches | True |
| msmarco | per-domain, 1024 wide | True |
| msmarco | published F2 dataset | True |
| msmarco | split ids are disjoint | True |
| msmarco | trained by train-selector | True |
| msmarco | trained from a clean tree | True |

## Coverage

| environment | training queries | with a positive | evaluated | single-positive | multi-positive | positives | documents | capped | weights all 1 |
|---|---|---|---|---|---|---|---|---|---|
| msmarco | 45000 | 45000 | 32 | 16 | 16 | 69 | 8841823 | True | False |

## Official metrics, full training judgments

| environment | cohort | k | ratio | arm | ndcg@10 | mrr@10 | mrr@100 | recall@100 |
|---|---|---|---|---|---|---|---|---|
| msmarco | single-positive | 102 | 10% | prefix-k | 0.1469 (0.0208 to 0.3017) | 0.1152 (0.0089 to 0.2590) | 0.1369 (0.0374 to 0.2801) | 0.7500 (0.5625 to 0.9375) |
| msmarco | single-positive | 102 | 10% | all-positive oracle (privileged) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) |
| msmarco | single-positive | 102 | 10% | L2S (in-sample) | 0.4154 (0.2558 to 0.5936) | 0.3147 (0.1544 to 0.5074) | 0.3180 (0.1591 to 0.5089) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 204 | 20% | prefix-k | 0.2377 (0.0911 to 0.4128) | 0.1799 (0.0417 to 0.3604) | 0.1900 (0.0551 to 0.3674) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 204 | 20% | all-positive oracle (privileged) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) |
| msmarco | single-positive | 204 | 20% | L2S (in-sample) | 0.3699 (0.2024 to 0.5493) | 0.2917 (0.1270 to 0.4792) | 0.3039 (0.1423 to 0.4884) | 0.9375 (0.8125 to 1.0000) |
| msmarco | single-positive | 307 | 30% | prefix-k | 0.2122 (0.0860 to 0.3609) | 0.1455 (0.0424 to 0.2903) | 0.1569 (0.0574 to 0.2995) | 0.7500 (0.5625 to 0.9375) |
| msmarco | single-positive | 307 | 30% | all-positive oracle (privileged) | 0.9769 (0.9308 to 1.0000) | 0.9688 (0.9062 to 1.0000) | 0.9688 (0.9062 to 1.0000) | 1.0000 (1.0000 to 1.0000) |
| msmarco | single-positive | 307 | 30% | L2S (in-sample) | 0.3978 (0.2202 to 0.5925) | 0.3293 (0.1488 to 0.5374) | 0.3372 (0.1578 to 0.5415) | 0.8750 (0.6875 to 1.0000) |
| msmarco | single-positive | 409 | 40% | prefix-k | 0.2483 (0.0992 to 0.4298) | 0.1929 (0.0526 to 0.3714) | 0.2069 (0.0708 to 0.3819) | 0.7500 (0.5625 to 0.9375) |
| msmarco | single-positive | 409 | 40% | all-positive oracle (privileged) | 0.9077 (0.8385 to 0.9769) | 0.8750 (0.7812 to 0.9688) | 0.8750 (0.7812 to 0.9688) | 1.0000 (1.0000 to 1.0000) |
| msmarco | single-positive | 409 | 40% | L2S (in-sample) | 0.3565 (0.1718 to 0.5582) | 0.3115 (0.1250 to 0.5250) | 0.3263 (0.1457 to 0.5339) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 512 | 50% | prefix-k | 0.2819 (0.1183 to 0.4654) | 0.2312 (0.0781 to 0.4062) | 0.2459 (0.1019 to 0.4183) | 0.8750 (0.6875 to 1.0000) |
| msmarco | single-positive | 512 | 50% | all-positive oracle (privileged) | 0.8612 (0.7537 to 0.9539) | 0.8146 (0.6729 to 0.9375) | 0.8146 (0.6729 to 0.9375) | 1.0000 (1.0000 to 1.0000) |
| msmarco | single-positive | 512 | 50% | L2S (in-sample) | 0.3942 (0.1864 to 0.6200) | 0.3625 (0.1562 to 0.5938) | 0.3791 (0.1795 to 0.6029) | 0.8750 (0.6875 to 1.0000) |
| msmarco | single-positive | 614 | 60% | prefix-k | 0.2816 (0.1179 to 0.4621) | 0.2312 (0.0833 to 0.4083) | 0.2445 (0.1010 to 0.4185) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 614 | 60% | all-positive oracle (privileged) | 0.8015 (0.6534 to 0.9308) | 0.7552 (0.5884 to 0.9062) | 0.7609 (0.5989 to 0.9064) | 1.0000 (1.0000 to 1.0000) |
| msmarco | single-positive | 614 | 60% | L2S (in-sample) | 0.3679 (0.1562 to 0.5938) | 0.3458 (0.1457 to 0.5750) | 0.3673 (0.1690 to 0.5916) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 716 | 70% | prefix-k | 0.3065 (0.1279 to 0.5093) | 0.2646 (0.0958 to 0.4625) | 0.2770 (0.1161 to 0.4697) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 716 | 70% | all-positive oracle (privileged) | 0.7483 (0.6024 to 0.8847) | 0.6844 (0.5125 to 0.8438) | 0.6875 (0.5249 to 0.8438) | 1.0000 (1.0000 to 1.0000) |
| msmarco | single-positive | 716 | 70% | L2S (in-sample) | 0.3399 (0.1622 to 0.5427) | 0.2896 (0.1166 to 0.4896) | 0.3065 (0.1413 to 0.4995) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 819 | 80% | prefix-k | 0.2978 (0.1207 to 0.4957) | 0.2537 (0.0923 to 0.4531) | 0.2680 (0.1133 to 0.4614) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 819 | 80% | all-positive oracle (privileged) | 0.6569 (0.4842 to 0.8207) | 0.5854 (0.4052 to 0.7730) | 0.5916 (0.4141 to 0.7761) | 1.0000 (1.0000 to 1.0000) |
| msmarco | single-positive | 819 | 80% | L2S (in-sample) | 0.3711 (0.1802 to 0.5817) | 0.3313 (0.1375 to 0.5437) | 0.3462 (0.1623 to 0.5540) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 921 | 90% | prefix-k | 0.2843 (0.1207 to 0.4702) | 0.2344 (0.0833 to 0.4115) | 0.2478 (0.1026 to 0.4197) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 921 | 90% | all-positive oracle (privileged) | 0.5326 (0.3477 to 0.7288) | 0.4659 (0.2678 to 0.6826) | 0.4754 (0.2818 to 0.6879) | 0.9375 (0.8125 to 1.0000) |
| msmarco | single-positive | 921 | 90% | L2S (in-sample) | 0.2987 (0.1332 to 0.4881) | 0.2521 (0.1000 to 0.4375) | 0.2697 (0.1251 to 0.4489) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 1024 | 100% | full query | 0.2790 (0.1249 to 0.4665) | 0.2277 (0.0833 to 0.4062) | 0.2430 (0.1055 to 0.4168) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 1024 | 100% | prefix-k | 0.2790 (0.1249 to 0.4665) | 0.2277 (0.0833 to 0.4062) | 0.2430 (0.1055 to 0.4168) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 1024 | 100% | all-positive oracle (privileged) | 0.2790 (0.1249 to 0.4665) | 0.2277 (0.0833 to 0.4062) | 0.2430 (0.1055 to 0.4168) | 0.8125 (0.6250 to 1.0000) |
| msmarco | single-positive | 1024 | 100% | L2S (in-sample) | 0.2790 (0.1249 to 0.4665) | 0.2277 (0.0833 to 0.4062) | 0.2430 (0.1055 to 0.4168) | 0.8125 (0.6250 to 1.0000) |
| msmarco | multi-positive | 102 | 10% | prefix-k | 0.1092 (0.0149 to 0.2472) | 0.1203 (0.0121 to 0.2916) | 0.1316 (0.0277 to 0.3013) | 0.4930 (0.2641 to 0.7222) |
| msmarco | multi-positive | 102 | 10% | all-positive oracle (privileged) | 0.4598 (0.3066 to 0.6193) | 0.5742 (0.3662 to 0.7913) | 0.5806 (0.3788 to 0.7973) | 0.9371 (0.8334 to 0.9997) |
| msmarco | multi-positive | 102 | 10% | held-out transfer (1-1) | 0.5891 (0.5188 to 0.6593) | 0.8663 (0.7710 to 0.9528) | 0.8663 (0.7710 to 0.9528) | 0.7717 (0.6574 to 0.8771) |
| msmarco | multi-positive | 102 | 10% | L2S (in-sample) | 0.1852 (0.0527 to 0.3444) | 0.1818 (0.0502 to 0.3496) | 0.1892 (0.0634 to 0.3557) | 0.5831 (0.3338 to 0.7912) |
| msmarco | multi-positive | 204 | 20% | prefix-k | 0.1539 (0.0545 to 0.2783) | 0.1519 (0.0418 to 0.3217) | 0.1632 (0.0554 to 0.3278) | 0.5624 (0.3336 to 0.7912) |
| msmarco | multi-positive | 204 | 20% | all-positive oracle (privileged) | 0.4315 (0.2869 to 0.5747) | 0.4038 (0.2406 to 0.5810) | 0.4078 (0.2474 to 0.5844) | 0.9371 (0.8334 to 0.9997) |
| msmarco | multi-positive | 204 | 20% | held-out transfer (1-1) | 0.5681 (0.4861 to 0.6616) | 0.8365 (0.7396 to 0.9272) | 0.8365 (0.7396 to 0.9272) | 0.8433 (0.7287 to 0.9470) |
| msmarco | multi-positive | 204 | 20% | L2S (in-sample) | 0.1623 (0.0524 to 0.2943) | 0.1639 (0.0445 to 0.3329) | 0.1718 (0.0561 to 0.3374) | 0.6664 (0.4582 to 0.8743) |
| msmarco | multi-positive | 307 | 30% | prefix-k | 0.1558 (0.0588 to 0.2731) | 0.1269 (0.0379 to 0.2391) | 0.1400 (0.0533 to 0.2471) | 0.6040 (0.3546 to 0.8329) |
| msmarco | multi-positive | 307 | 30% | all-positive oracle (privileged) | 0.3503 (0.2005 to 0.5040) | 0.3462 (0.1965 to 0.5048) | 0.3639 (0.2256 to 0.5109) | 0.9371 (0.8334 to 0.9997) |
| msmarco | multi-positive | 307 | 30% | held-out transfer (1-1) | 0.5314 (0.4494 to 0.6167) | 0.7622 (0.6562 to 0.8682) | 0.7640 (0.6597 to 0.8695) | 0.8225 (0.7082 to 0.9364) |
| msmarco | multi-positive | 307 | 30% | L2S (in-sample) | 0.1560 (0.0583 to 0.2664) | 0.1472 (0.0504 to 0.2581) | 0.1581 (0.0680 to 0.2636) | 0.7495 (0.5818 to 0.9159) |
| msmarco | multi-positive | 409 | 40% | prefix-k | 0.1406 (0.0340 to 0.2764) | 0.1430 (0.0291 to 0.3128) | 0.1601 (0.0495 to 0.3262) | 0.6040 (0.3960 to 0.8120) |
| msmarco | multi-positive | 409 | 40% | all-positive oracle (privileged) | 0.3089 (0.1768 to 0.4403) | 0.3005 (0.1594 to 0.4673) | 0.3127 (0.1820 to 0.4728) | 0.9371 (0.8334 to 0.9997) |
| msmarco | multi-positive | 409 | 40% | held-out transfer (1-1) | 0.4781 (0.3771 to 0.5768) | 0.6556 (0.4892 to 0.8151) | 0.6575 (0.4939 to 0.8151) | 0.8225 (0.7083 to 0.9365) |
| msmarco | multi-positive | 409 | 40% | L2S (in-sample) | 0.1354 (0.0413 to 0.2466) | 0.1278 (0.0307 to 0.2387) | 0.1434 (0.0546 to 0.2489) | 0.7495 (0.5818 to 0.9159) |
| msmarco | multi-positive | 512 | 50% | prefix-k | 0.1319 (0.0311 to 0.2567) | 0.1335 (0.0240 to 0.3003) | 0.1450 (0.0394 to 0.3103) | 0.6040 (0.3960 to 0.8120) |
| msmarco | multi-positive | 512 | 50% | all-positive oracle (privileged) | 0.2879 (0.1504 to 0.4251) | 0.2832 (0.1380 to 0.4505) | 0.3022 (0.1679 to 0.4644) | 0.9371 (0.8334 to 0.9997) |
| msmarco | multi-positive | 512 | 50% | held-out transfer (1-1) | 0.4263 (0.3204 to 0.5279) | 0.5643 (0.3978 to 0.7196) | 0.5695 (0.4078 to 0.7205) | 0.8225 (0.7082 to 0.9365) |
| msmarco | multi-positive | 512 | 50% | L2S (in-sample) | 0.1404 (0.0430 to 0.2542) | 0.1349 (0.0375 to 0.2459) | 0.1518 (0.0615 to 0.2562) | 0.7495 (0.5818 to 0.9159) |
| msmarco | multi-positive | 614 | 60% | prefix-k | 0.1247 (0.0355 to 0.2322) | 0.1068 (0.0286 to 0.2043) | 0.1186 (0.0440 to 0.2130) | 0.6039 (0.3959 to 0.8119) |
| msmarco | multi-positive | 614 | 60% | all-positive oracle (privileged) | 0.2969 (0.1496 to 0.4427) | 0.3169 (0.1377 to 0.5214) | 0.3327 (0.1626 to 0.5324) | 0.9162 (0.7915 to 0.9997) |
| msmarco | multi-positive | 614 | 60% | held-out transfer (1-1) | 0.3801 (0.2588 to 0.4990) | 0.4836 (0.3165 to 0.6475) | 0.4960 (0.3384 to 0.6521) | 0.8018 (0.6772 to 0.9164) |
| msmarco | multi-positive | 614 | 60% | L2S (in-sample) | 0.1498 (0.0443 to 0.2698) | 0.1681 (0.0349 to 0.3482) | 0.1862 (0.0587 to 0.3642) | 0.7495 (0.5818 to 0.9159) |
| msmarco | multi-positive | 716 | 70% | prefix-k | 0.1397 (0.0493 to 0.2453) | 0.1101 (0.0392 to 0.1971) | 0.1174 (0.0496 to 0.2021) | 0.6247 (0.4167 to 0.8326) |
| msmarco | multi-positive | 716 | 70% | all-positive oracle (privileged) | 0.3033 (0.1552 to 0.4470) | 0.3537 (0.1457 to 0.5828) | 0.3679 (0.1662 to 0.5904) | 0.9162 (0.7915 to 0.9997) |
| msmarco | multi-positive | 716 | 70% | held-out transfer (1-1) | 0.3447 (0.2166 to 0.4752) | 0.4202 (0.2550 to 0.5925) | 0.4337 (0.2745 to 0.5996) | 0.8018 (0.6773 to 0.9164) |
| msmarco | multi-positive | 716 | 70% | L2S (in-sample) | 0.1679 (0.0643 to 0.2856) | 0.1828 (0.0498 to 0.3583) | 0.1929 (0.0646 to 0.3675) | 0.7495 (0.5818 to 0.9159) |
| msmarco | multi-positive | 819 | 80% | prefix-k | 0.1389 (0.0504 to 0.2439) | 0.1073 (0.0366 to 0.1939) | 0.1134 (0.0459 to 0.1974) | 0.6249 (0.3754 to 0.8331) |
| msmarco | multi-positive | 819 | 80% | all-positive oracle (privileged) | 0.2665 (0.1232 to 0.4098) | 0.3313 (0.1251 to 0.5625) | 0.3473 (0.1474 to 0.5716) | 0.9162 (0.7915 to 0.9997) |
| msmarco | multi-positive | 819 | 80% | held-out transfer (1-1) | 0.3079 (0.1811 to 0.4394) | 0.3642 (0.2018 to 0.5427) | 0.3794 (0.2285 to 0.5494) | 0.8018 (0.6771 to 0.9164) |
| msmarco | multi-positive | 819 | 80% | L2S (in-sample) | 0.1523 (0.0555 to 0.2654) | 0.1294 (0.0496 to 0.2247) | 0.1371 (0.0615 to 0.2285) | 0.7081 (0.4997 to 0.9159) |
| msmarco | multi-positive | 921 | 90% | prefix-k | 0.1446 (0.0547 to 0.2487) | 0.1254 (0.0364 to 0.2350) | 0.1323 (0.0470 to 0.2397) | 0.6249 (0.3754 to 0.8331) |
| msmarco | multi-positive | 921 | 90% | all-positive oracle (privileged) | 0.2152 (0.0844 to 0.3487) | 0.2250 (0.0795 to 0.3961) | 0.2426 (0.1045 to 0.4128) | 0.8746 (0.7498 to 0.9991) |
| msmarco | multi-positive | 921 | 90% | held-out transfer (1-1) | 0.2355 (0.1138 to 0.3708) | 0.2633 (0.1198 to 0.4330) | 0.2802 (0.1429 to 0.4418) | 0.7810 (0.6354 to 0.9162) |
| msmarco | multi-positive | 921 | 90% | L2S (in-sample) | 0.1547 (0.0578 to 0.2659) | 0.1431 (0.0516 to 0.2524) | 0.1487 (0.0602 to 0.2542) | 0.6665 (0.4169 to 0.8747) |
| msmarco | multi-positive | 1024 | 100% | full query | 0.1463 (0.0541 to 0.2546) | 0.1258 (0.0467 to 0.2208) | 0.1315 (0.0543 to 0.2238) | 0.6249 (0.3756 to 0.8331) |
| msmarco | multi-positive | 1024 | 100% | prefix-k | 0.1463 (0.0541 to 0.2546) | 0.1258 (0.0467 to 0.2208) | 0.1315 (0.0543 to 0.2238) | 0.6249 (0.3756 to 0.8331) |
| msmarco | multi-positive | 1024 | 100% | all-positive oracle (privileged) | 0.1463 (0.0541 to 0.2546) | 0.1258 (0.0467 to 0.2208) | 0.1315 (0.0543 to 0.2238) | 0.6249 (0.3756 to 0.8331) |
| msmarco | multi-positive | 1024 | 100% | held-out transfer (1-1) | 0.1463 (0.0541 to 0.2546) | 0.1258 (0.0467 to 0.2208) | 0.1315 (0.0543 to 0.2238) | 0.6249 (0.3756 to 0.8331) |
| msmarco | multi-positive | 1024 | 100% | L2S (in-sample) | 0.1463 (0.0541 to 0.2546) | 0.1258 (0.0467 to 0.2208) | 0.1315 (0.0543 to 0.2238) | 0.6249 (0.3756 to 0.8331) |

## Paired differences, with queries better and worse

| environment | cohort | k | measure | comparison | queries | difference | 2.5% | 97.5% | share better | share worse |
|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | single-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.5846 | 0.4064 | 0.7442 | 0.812 | 0.000 |
| msmarco | single-positive | 102 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.1364 | -0.0010 | 0.2710 | 0.500 | 0.188 |
| msmarco | single-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.7210 | 0.5335 | 0.8751 | 0.875 | 0.000 |
| msmarco | single-positive | 102 | ndcg@10 | prefix-k minus full query | 16 | -0.1321 | -0.3054 | 0.0347 | 0.062 | 0.375 |
| msmarco | single-positive | 102 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.6853 | 0.4926 | 0.8456 | 0.812 | 0.000 |
| msmarco | single-positive | 102 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0871 | -0.0629 | 0.2415 | 0.500 | 0.188 |
| msmarco | single-positive | 102 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.7723 | 0.5938 | 0.9167 | 0.875 | 0.000 |
| msmarco | single-positive | 102 | mrr@10 | prefix-k minus full query | 16 | -0.1125 | -0.2890 | 0.0655 | 0.062 | 0.375 |
| msmarco | single-positive | 102 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.6820 | 0.4911 | 0.8409 | 0.812 | 0.000 |
| msmarco | single-positive | 102 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0751 | -0.0730 | 0.2283 | 0.562 | 0.188 |
| msmarco | single-positive | 102 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.7570 | 0.5832 | 0.8945 | 0.875 | 0.000 |
| msmarco | single-positive | 102 | mrr@100 | prefix-k minus full query | 16 | -0.1060 | -0.2690 | 0.0664 | 0.188 | 0.625 |
| msmarco | single-positive | 102 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 102 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | single-positive | 102 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 102 | recall@100 | prefix-k minus full query | 16 | -0.0625 | -0.1875 | 0.0000 | 0.000 | 0.062 |
| msmarco | single-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.6301 | 0.4507 | 0.7976 | 0.812 | 0.000 |
| msmarco | single-positive | 204 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0909 | -0.0246 | 0.2035 | 0.375 | 0.062 |
| msmarco | single-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.7210 | 0.5335 | 0.8751 | 0.875 | 0.000 |
| msmarco | single-positive | 204 | ndcg@10 | prefix-k minus full query | 16 | -0.0414 | -0.1322 | 0.0503 | 0.125 | 0.250 |
| msmarco | single-positive | 204 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.7083 | 0.5208 | 0.8730 | 0.812 | 0.000 |
| msmarco | single-positive | 204 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0640 | -0.0673 | 0.1962 | 0.375 | 0.062 |
| msmarco | single-positive | 204 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.7723 | 0.5938 | 0.9167 | 0.875 | 0.000 |
| msmarco | single-positive | 204 | mrr@10 | prefix-k minus full query | 16 | -0.0478 | -0.1264 | 0.0223 | 0.125 | 0.250 |
| msmarco | single-positive | 204 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.6961 | 0.5116 | 0.8577 | 0.812 | 0.000 |
| msmarco | single-positive | 204 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0610 | -0.0685 | 0.1934 | 0.688 | 0.062 |
| msmarco | single-positive | 204 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.7570 | 0.5832 | 0.8945 | 0.875 | 0.000 |
| msmarco | single-positive | 204 | mrr@100 | prefix-k minus full query | 16 | -0.0530 | -0.1279 | 0.0144 | 0.188 | 0.500 |
| msmarco | single-positive | 204 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.0625 | 0.0000 | 0.1875 | 0.062 | 0.000 |
| msmarco | single-positive | 204 | recall@100 | L2S (in-sample) minus full query | 16 | 0.1250 | 0.0000 | 0.3125 | 0.125 | 0.000 |
| msmarco | single-positive | 204 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 204 | recall@100 | prefix-k minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | single-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.5791 | 0.3854 | 0.7509 | 0.750 | 0.000 |
| msmarco | single-positive | 307 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.1188 | 0.0338 | 0.2103 | 0.375 | 0.125 |
| msmarco | single-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.6979 | 0.5186 | 0.8542 | 0.875 | 0.000 |
| msmarco | single-positive | 307 | ndcg@10 | prefix-k minus full query | 16 | -0.0668 | -0.1634 | 0.0146 | 0.125 | 0.312 |
| msmarco | single-positive | 307 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.6394 | 0.4332 | 0.8172 | 0.750 | 0.000 |
| msmarco | single-positive | 307 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.1017 | 0.0145 | 0.2136 | 0.375 | 0.125 |
| msmarco | single-positive | 307 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.7411 | 0.5640 | 0.8869 | 0.875 | 0.000 |
| msmarco | single-positive | 307 | mrr@10 | prefix-k minus full query | 16 | -0.0822 | -0.1719 | -0.0145 | 0.125 | 0.312 |
| msmarco | single-positive | 307 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.6315 | 0.4294 | 0.8086 | 0.750 | 0.000 |
| msmarco | single-positive | 307 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0943 | 0.0081 | 0.2037 | 0.562 | 0.188 |
| msmarco | single-positive | 307 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.7258 | 0.5548 | 0.8669 | 0.875 | 0.000 |
| msmarco | single-positive | 307 | mrr@100 | prefix-k minus full query | 16 | -0.0860 | -0.1698 | -0.0227 | 0.250 | 0.500 |
| msmarco | single-positive | 307 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1250 | 0.0000 | 0.3125 | 0.125 | 0.000 |
| msmarco | single-positive | 307 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0625 | 0.0000 | 0.1875 | 0.062 | 0.000 |
| msmarco | single-positive | 307 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 307 | recall@100 | prefix-k minus full query | 16 | -0.0625 | -0.1875 | 0.0000 | 0.000 | 0.062 |
| msmarco | single-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.5512 | 0.3730 | 0.7195 | 0.750 | 0.000 |
| msmarco | single-positive | 409 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0775 | 0.0018 | 0.1692 | 0.250 | 0.062 |
| msmarco | single-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.6287 | 0.4725 | 0.7745 | 0.875 | 0.000 |
| msmarco | single-positive | 409 | ndcg@10 | prefix-k minus full query | 16 | -0.0307 | -0.1332 | 0.0614 | 0.125 | 0.250 |
| msmarco | single-positive | 409 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.5635 | 0.3729 | 0.7427 | 0.750 | 0.000 |
| msmarco | single-positive | 409 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0838 | -0.0016 | 0.1958 | 0.250 | 0.062 |
| msmarco | single-positive | 409 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.6473 | 0.4910 | 0.7948 | 0.875 | 0.000 |
| msmarco | single-positive | 409 | mrr@10 | prefix-k minus full query | 16 | -0.0348 | -0.1384 | 0.0667 | 0.125 | 0.250 |
| msmarco | single-positive | 409 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.5487 | 0.3640 | 0.7259 | 0.750 | 0.000 |
| msmarco | single-positive | 409 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0834 | 0.0001 | 0.1937 | 0.500 | 0.125 |
| msmarco | single-positive | 409 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.6320 | 0.4771 | 0.7781 | 0.875 | 0.000 |
| msmarco | single-positive | 409 | mrr@100 | prefix-k minus full query | 16 | -0.0360 | -0.1331 | 0.0603 | 0.250 | 0.438 |
| msmarco | single-positive | 409 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 409 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | single-positive | 409 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 409 | recall@100 | prefix-k minus full query | 16 | -0.0625 | -0.1875 | 0.0000 | 0.000 | 0.062 |
| msmarco | single-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.4670 | 0.2850 | 0.6523 | 0.688 | 0.000 |
| msmarco | single-positive | 512 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.1152 | 0.0344 | 0.2105 | 0.375 | 0.000 |
| msmarco | single-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.5822 | 0.4330 | 0.7247 | 0.875 | 0.000 |
| msmarco | single-positive | 512 | ndcg@10 | prefix-k minus full query | 16 | 0.0029 | -0.0654 | 0.0721 | 0.188 | 0.188 |
| msmarco | single-positive | 512 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.4521 | 0.2729 | 0.6438 | 0.688 | 0.000 |
| msmarco | single-positive | 512 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.1348 | 0.0292 | 0.2598 | 0.375 | 0.000 |
| msmarco | single-positive | 512 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.5869 | 0.4292 | 0.7447 | 0.875 | 0.000 |
| msmarco | single-positive | 512 | mrr@10 | prefix-k minus full query | 16 | 0.0036 | -0.0885 | 0.0957 | 0.188 | 0.188 |
| msmarco | single-positive | 512 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.4355 | 0.2622 | 0.6212 | 0.688 | 0.000 |
| msmarco | single-positive | 512 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.1362 | 0.0301 | 0.2589 | 0.750 | 0.000 |
| msmarco | single-positive | 512 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.5716 | 0.4180 | 0.7242 | 0.875 | 0.000 |
| msmarco | single-positive | 512 | mrr@100 | prefix-k minus full query | 16 | 0.0030 | -0.0889 | 0.0959 | 0.500 | 0.312 |
| msmarco | single-positive | 512 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1250 | 0.0000 | 0.3125 | 0.125 | 0.000 |
| msmarco | single-positive | 512 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0625 | 0.0000 | 0.1875 | 0.062 | 0.000 |
| msmarco | single-positive | 512 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 512 | recall@100 | prefix-k minus full query | 16 | 0.0625 | 0.0000 | 0.1875 | 0.062 | 0.000 |
| msmarco | single-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.4335 | 0.2460 | 0.6269 | 0.625 | 0.000 |
| msmarco | single-positive | 614 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0889 | 0.0066 | 0.1827 | 0.250 | 0.000 |
| msmarco | single-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.5225 | 0.3619 | 0.6875 | 0.812 | 0.000 |
| msmarco | single-positive | 614 | ndcg@10 | prefix-k minus full query | 16 | 0.0025 | -0.0681 | 0.0707 | 0.188 | 0.125 |
| msmarco | single-positive | 614 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.4094 | 0.2250 | 0.6042 | 0.625 | 0.000 |
| msmarco | single-positive | 614 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.1182 | 0.0071 | 0.2432 | 0.250 | 0.000 |
| msmarco | single-positive | 614 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.5275 | 0.3623 | 0.7009 | 0.812 | 0.000 |
| msmarco | single-positive | 614 | mrr@10 | prefix-k minus full query | 16 | 0.0036 | -0.0923 | 0.0958 | 0.188 | 0.125 |
| msmarco | single-positive | 614 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.3936 | 0.2217 | 0.5814 | 0.688 | 0.000 |
| msmarco | single-positive | 614 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.1244 | 0.0142 | 0.2480 | 0.562 | 0.000 |
| msmarco | single-positive | 614 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.5179 | 0.3576 | 0.6827 | 0.875 | 0.000 |
| msmarco | single-positive | 614 | mrr@100 | prefix-k minus full query | 16 | 0.0015 | -0.0939 | 0.0938 | 0.250 | 0.375 |
| msmarco | single-positive | 614 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 614 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | single-positive | 614 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 614 | recall@100 | prefix-k minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | single-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.4084 | 0.2603 | 0.5625 | 0.750 | 0.000 |
| msmarco | single-positive | 716 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0608 | 0.0066 | 0.1364 | 0.250 | 0.000 |
| msmarco | single-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.4692 | 0.3282 | 0.6161 | 0.812 | 0.000 |
| msmarco | single-positive | 716 | ndcg@10 | prefix-k minus full query | 16 | 0.0275 | -0.0112 | 0.0841 | 0.188 | 0.062 |
| msmarco | single-positive | 716 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.3948 | 0.2469 | 0.5510 | 0.750 | 0.000 |
| msmarco | single-positive | 716 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0619 | 0.0062 | 0.1530 | 0.250 | 0.000 |
| msmarco | single-positive | 716 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.4567 | 0.3092 | 0.6078 | 0.812 | 0.000 |
| msmarco | single-positive | 716 | mrr@10 | prefix-k minus full query | 16 | 0.0369 | -0.0143 | 0.1120 | 0.188 | 0.062 |
| msmarco | single-positive | 716 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.3810 | 0.2405 | 0.5315 | 0.812 | 0.000 |
| msmarco | single-positive | 716 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0635 | 0.0093 | 0.1524 | 0.438 | 0.062 |
| msmarco | single-positive | 716 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.4445 | 0.2979 | 0.5917 | 0.875 | 0.000 |
| msmarco | single-positive | 716 | mrr@100 | prefix-k minus full query | 16 | 0.0340 | -0.0175 | 0.1101 | 0.312 | 0.312 |
| msmarco | single-positive | 716 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 716 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | single-positive | 716 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 716 | recall@100 | prefix-k minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | single-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2858 | 0.1656 | 0.4124 | 0.625 | 0.000 |
| msmarco | single-positive | 819 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0921 | 0.0264 | 0.1693 | 0.375 | 0.000 |
| msmarco | single-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.3779 | 0.2473 | 0.5181 | 0.750 | 0.000 |
| msmarco | single-positive | 819 | ndcg@10 | prefix-k minus full query | 16 | 0.0187 | -0.0130 | 0.0692 | 0.062 | 0.062 |
| msmarco | single-positive | 819 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2542 | 0.1344 | 0.3875 | 0.625 | 0.000 |
| msmarco | single-positive | 819 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.1036 | 0.0238 | 0.2051 | 0.375 | 0.000 |
| msmarco | single-positive | 819 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.3577 | 0.2171 | 0.5053 | 0.750 | 0.000 |
| msmarco | single-positive | 819 | mrr@10 | prefix-k minus full query | 16 | 0.0260 | -0.0156 | 0.0938 | 0.062 | 0.062 |
| msmarco | single-positive | 819 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2454 | 0.1312 | 0.3760 | 0.750 | 0.000 |
| msmarco | single-positive | 819 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.1033 | 0.0231 | 0.2036 | 0.625 | 0.062 |
| msmarco | single-positive | 819 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.3487 | 0.2161 | 0.4864 | 0.875 | 0.000 |
| msmarco | single-positive | 819 | mrr@100 | prefix-k minus full query | 16 | 0.0251 | -0.0150 | 0.0939 | 0.188 | 0.250 |
| msmarco | single-positive | 819 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 819 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | single-positive | 819 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.1875 | 0.0000 | 0.3750 | 0.188 | 0.000 |
| msmarco | single-positive | 819 | recall@100 | prefix-k minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | single-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2338 | 0.1399 | 0.3341 | 0.625 | 0.000 |
| msmarco | single-positive | 921 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0197 | 0.0000 | 0.0443 | 0.188 | 0.000 |
| msmarco | single-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.2536 | 0.1536 | 0.3586 | 0.625 | 0.000 |
| msmarco | single-positive | 921 | ndcg@10 | prefix-k minus full query | 16 | 0.0053 | -0.0654 | 0.0721 | 0.188 | 0.125 |
| msmarco | single-positive | 921 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2138 | 0.1150 | 0.3264 | 0.625 | 0.000 |
| msmarco | single-positive | 921 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0244 | 0.0000 | 0.0557 | 0.188 | 0.000 |
| msmarco | single-positive | 921 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.2382 | 0.1236 | 0.3681 | 0.625 | 0.000 |
| msmarco | single-positive | 921 | mrr@10 | prefix-k minus full query | 16 | 0.0067 | -0.0885 | 0.0967 | 0.188 | 0.125 |
| msmarco | single-positive | 921 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2058 | 0.1103 | 0.3139 | 0.812 | 0.000 |
| msmarco | single-positive | 921 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0267 | 0.0033 | 0.0577 | 0.500 | 0.062 |
| msmarco | single-positive | 921 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.2325 | 0.1219 | 0.3598 | 0.812 | 0.000 |
| msmarco | single-positive | 921 | mrr@100 | prefix-k minus full query | 16 | 0.0049 | -0.0888 | 0.0957 | 0.312 | 0.312 |
| msmarco | single-positive | 921 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1250 | 0.0000 | 0.3125 | 0.125 | 0.000 |
| msmarco | single-positive | 921 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | single-positive | 921 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.1250 | 0.0000 | 0.3125 | 0.125 | 0.000 |
| msmarco | single-positive | 921 | recall@100 | prefix-k minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | multi-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2746 | 0.1613 | 0.3882 | 0.916 | 0.000 |
| msmarco | multi-positive | 102 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0388 | -0.0023 | 0.0976 | 0.334 | 0.167 |
| msmarco | multi-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.3135 | 0.2044 | 0.4243 | 0.916 | 0.000 |
| msmarco | multi-positive | 102 | ndcg@10 | prefix-k minus full query | 16 | -0.0372 | -0.1495 | 0.0588 | 0.167 | 0.334 |
| msmarco | multi-positive | 102 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.3924 | 0.2018 | 0.6036 | 0.833 | 0.000 |
| msmarco | multi-positive | 102 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0560 | -0.0083 | 0.1481 | 0.334 | 0.167 |
| msmarco | multi-positive | 102 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.4484 | 0.2738 | 0.6526 | 0.916 | 0.000 |
| msmarco | multi-positive | 102 | mrr@10 | prefix-k minus full query | 16 | -0.0056 | -0.1116 | 0.1131 | 0.167 | 0.334 |
| msmarco | multi-positive | 102 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.3914 | 0.2052 | 0.5975 | 0.916 | 0.000 |
| msmarco | multi-positive | 102 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0577 | -0.0068 | 0.1494 | 0.583 | 0.250 |
| msmarco | multi-positive | 102 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.4491 | 0.2810 | 0.6453 | 0.999 | 0.000 |
| msmarco | multi-positive | 102 | mrr@100 | prefix-k minus full query | 16 | 0.0001 | -0.0967 | 0.1130 | 0.167 | 0.500 |
| msmarco | multi-positive | 102 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.3540 | 0.1668 | 0.5622 | 0.584 | 0.000 |
| msmarco | multi-positive | 102 | recall@100 | L2S (in-sample) minus full query | 16 | -0.0418 | -0.1669 | 0.0831 | 0.083 | 0.167 |
| msmarco | multi-positive | 102 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.3122 | 0.1250 | 0.5409 | 0.500 | 0.000 |
| msmarco | multi-positive | 102 | recall@100 | prefix-k minus full query | 16 | -0.1319 | -0.2498 | -0.0212 | 0.000 | 0.334 |
| msmarco | multi-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2693 | 0.1829 | 0.3615 | 0.916 | 0.000 |
| msmarco | multi-positive | 204 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0160 | -0.0104 | 0.0519 | 0.250 | 0.250 |
| msmarco | multi-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.2852 | 0.1977 | 0.3682 | 0.916 | 0.000 |
| msmarco | multi-positive | 204 | ndcg@10 | prefix-k minus full query | 16 | 0.0075 | -0.0411 | 0.0589 | 0.167 | 0.417 |
| msmarco | multi-positive | 204 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2399 | 0.1242 | 0.3676 | 0.833 | 0.000 |
| msmarco | multi-positive | 204 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0380 | -0.0284 | 0.1364 | 0.167 | 0.250 |
| msmarco | multi-positive | 204 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.2780 | 0.1739 | 0.3874 | 0.916 | 0.000 |
| msmarco | multi-positive | 204 | mrr@10 | prefix-k minus full query | 16 | 0.0261 | -0.0406 | 0.1237 | 0.167 | 0.417 |
| msmarco | multi-positive | 204 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2360 | 0.1243 | 0.3631 | 0.916 | 0.000 |
| msmarco | multi-positive | 204 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0402 | -0.0262 | 0.1376 | 0.417 | 0.334 |
| msmarco | multi-positive | 204 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.2763 | 0.1748 | 0.3850 | 1.000 | 0.000 |
| msmarco | multi-positive | 204 | mrr@100 | prefix-k minus full query | 16 | 0.0317 | -0.0302 | 0.1259 | 0.334 | 0.500 |
| msmarco | multi-positive | 204 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2707 | 0.1043 | 0.4579 | 0.500 | 0.000 |
| msmarco | multi-positive | 204 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0415 | -0.0002 | 0.1247 | 0.083 | 0.001 |
| msmarco | multi-positive | 204 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.3122 | 0.1249 | 0.5409 | 0.500 | 0.000 |
| msmarco | multi-positive | 204 | recall@100 | prefix-k minus full query | 16 | -0.0625 | -0.2081 | 0.0830 | 0.083 | 0.250 |
| msmarco | multi-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1943 | 0.1135 | 0.2716 | 0.750 | 0.000 |
| msmarco | multi-positive | 307 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0097 | -0.0005 | 0.0227 | 0.333 | 0.084 |
| msmarco | multi-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.2040 | 0.1215 | 0.2808 | 0.750 | 0.000 |
| msmarco | multi-positive | 307 | ndcg@10 | prefix-k minus full query | 16 | 0.0094 | -0.0610 | 0.0938 | 0.168 | 0.250 |
| msmarco | multi-positive | 307 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1989 | 0.1000 | 0.2987 | 0.667 | 0.000 |
| msmarco | multi-positive | 307 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0214 | -0.0022 | 0.0541 | 0.250 | 0.084 |
| msmarco | multi-positive | 307 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.2203 | 0.1207 | 0.3170 | 0.750 | 0.000 |
| msmarco | multi-positive | 307 | mrr@10 | prefix-k minus full query | 16 | 0.0011 | -0.0713 | 0.1042 | 0.084 | 0.333 |
| msmarco | multi-positive | 307 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2058 | 0.1155 | 0.2995 | 0.916 | 0.000 |
| msmarco | multi-positive | 307 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0266 | 0.0040 | 0.0580 | 0.666 | 0.084 |
| msmarco | multi-positive | 307 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.2324 | 0.1437 | 0.3202 | 1.000 | 0.000 |
| msmarco | multi-positive | 307 | mrr@100 | prefix-k minus full query | 16 | 0.0085 | -0.0538 | 0.1040 | 0.168 | 0.416 |
| msmarco | multi-positive | 307 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1876 | 0.0629 | 0.3331 | 0.417 | 0.000 |
| msmarco | multi-positive | 307 | recall@100 | L2S (in-sample) minus full query | 16 | 0.1246 | -0.0001 | 0.2494 | 0.250 | 0.001 |
| msmarco | multi-positive | 307 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.3122 | 0.1249 | 0.5409 | 0.500 | 0.000 |
| msmarco | multi-positive | 307 | recall@100 | prefix-k minus full query | 16 | -0.0209 | -0.0625 | 0.0000 | 0.000 | 0.084 |
| msmarco | multi-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1735 | 0.1032 | 0.2456 | 0.750 | 0.000 |
| msmarco | multi-positive | 409 | ndcg@10 | L2S (in-sample) minus full query | 16 | -0.0109 | -0.0507 | 0.0170 | 0.166 | 0.251 |
| msmarco | multi-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.1626 | 0.0989 | 0.2173 | 0.750 | 0.000 |
| msmarco | multi-positive | 409 | ndcg@10 | prefix-k minus full query | 16 | -0.0057 | -0.0509 | 0.0369 | 0.250 | 0.250 |
| msmarco | multi-positive | 409 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1728 | 0.0854 | 0.2606 | 0.667 | 0.000 |
| msmarco | multi-positive | 409 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0019 | -0.0334 | 0.0415 | 0.166 | 0.250 |
| msmarco | multi-positive | 409 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.1747 | 0.1003 | 0.2539 | 0.750 | 0.000 |
| msmarco | multi-positive | 409 | mrr@10 | prefix-k minus full query | 16 | 0.0172 | -0.0467 | 0.1148 | 0.167 | 0.250 |
| msmarco | multi-positive | 409 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1693 | 0.0872 | 0.2531 | 0.917 | 0.000 |
| msmarco | multi-positive | 409 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0119 | -0.0162 | 0.0473 | 0.582 | 0.251 |
| msmarco | multi-positive | 409 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.1811 | 0.1155 | 0.2565 | 1.000 | 0.000 |
| msmarco | multi-positive | 409 | mrr@100 | prefix-k minus full query | 16 | 0.0286 | -0.0319 | 0.1230 | 0.417 | 0.333 |
| msmarco | multi-positive | 409 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1876 | 0.0629 | 0.3331 | 0.417 | 0.000 |
| msmarco | multi-positive | 409 | recall@100 | L2S (in-sample) minus full query | 16 | 0.1246 | -0.0001 | 0.2494 | 0.250 | 0.001 |
| msmarco | multi-positive | 409 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.3122 | 0.1249 | 0.5409 | 0.500 | 0.000 |
| msmarco | multi-positive | 409 | recall@100 | prefix-k minus full query | 16 | -0.0209 | -0.1457 | 0.1038 | 0.083 | 0.167 |
| msmarco | multi-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1474 | 0.0739 | 0.2284 | 0.667 | 0.000 |
| msmarco | multi-positive | 512 | ndcg@10 | L2S (in-sample) minus full query | 16 | -0.0059 | -0.0460 | 0.0224 | 0.250 | 0.167 |
| msmarco | multi-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.1415 | 0.0757 | 0.2026 | 0.667 | 0.000 |
| msmarco | multi-positive | 512 | ndcg@10 | prefix-k minus full query | 16 | -0.0144 | -0.0598 | 0.0288 | 0.167 | 0.333 |
| msmarco | multi-positive | 512 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1484 | 0.0605 | 0.2411 | 0.584 | 0.000 |
| msmarco | multi-positive | 512 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0090 | -0.0264 | 0.0498 | 0.167 | 0.167 |
| msmarco | multi-positive | 512 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.1574 | 0.0815 | 0.2428 | 0.667 | 0.000 |
| msmarco | multi-positive | 512 | mrr@10 | prefix-k minus full query | 16 | 0.0076 | -0.0606 | 0.1067 | 0.167 | 0.333 |
| msmarco | multi-positive | 512 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1504 | 0.0723 | 0.2356 | 0.916 | 0.000 |
| msmarco | multi-positive | 512 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0203 | -0.0069 | 0.0557 | 0.583 | 0.167 |
| msmarco | multi-positive | 512 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.1707 | 0.1038 | 0.2480 | 1.000 | 0.000 |
| msmarco | multi-positive | 512 | mrr@100 | prefix-k minus full query | 16 | 0.0135 | -0.0529 | 0.1117 | 0.334 | 0.500 |
| msmarco | multi-positive | 512 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1875 | 0.0628 | 0.3330 | 0.417 | 0.000 |
| msmarco | multi-positive | 512 | recall@100 | L2S (in-sample) minus full query | 16 | 0.1246 | 0.0000 | 0.2494 | 0.250 | 0.000 |
| msmarco | multi-positive | 512 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.3122 | 0.1249 | 0.5409 | 0.500 | 0.000 |
| msmarco | multi-positive | 512 | recall@100 | prefix-k minus full query | 16 | -0.0209 | -0.1457 | 0.1038 | 0.083 | 0.167 |
| msmarco | multi-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1471 | 0.0733 | 0.2249 | 0.667 | 0.000 |
| msmarco | multi-positive | 614 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0035 | -0.0472 | 0.0584 | 0.166 | 0.168 |
| msmarco | multi-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.1506 | 0.0763 | 0.2217 | 0.667 | 0.000 |
| msmarco | multi-positive | 614 | ndcg@10 | prefix-k minus full query | 16 | -0.0216 | -0.0559 | 0.0000 | 0.001 | 0.250 |
| msmarco | multi-positive | 614 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1488 | 0.0662 | 0.2409 | 0.584 | 0.000 |
| msmarco | multi-positive | 614 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0422 | -0.0314 | 0.1637 | 0.083 | 0.167 |
| msmarco | multi-positive | 614 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.1910 | 0.0877 | 0.3124 | 0.667 | 0.000 |
| msmarco | multi-positive | 614 | mrr@10 | prefix-k minus full query | 16 | -0.0190 | -0.0432 | -0.0000 | 0.001 | 0.250 |
| msmarco | multi-positive | 614 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1465 | 0.0697 | 0.2335 | 0.916 | 0.000 |
| msmarco | multi-positive | 614 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0547 | -0.0081 | 0.1696 | 0.500 | 0.167 |
| msmarco | multi-positive | 614 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.2012 | 0.1048 | 0.3168 | 1.000 | 0.000 |
| msmarco | multi-positive | 614 | mrr@100 | prefix-k minus full query | 16 | -0.0130 | -0.0290 | 0.0007 | 0.250 | 0.334 |
| msmarco | multi-positive | 614 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1667 | 0.0419 | 0.2917 | 0.334 | 0.000 |
| msmarco | multi-positive | 614 | recall@100 | L2S (in-sample) minus full query | 16 | 0.1246 | -0.0001 | 0.2494 | 0.250 | 0.001 |
| msmarco | multi-positive | 614 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.2913 | 0.0833 | 0.5408 | 0.417 | 0.000 |
| msmarco | multi-positive | 614 | recall@100 | prefix-k minus full query | 16 | -0.0210 | -0.1458 | 0.1037 | 0.083 | 0.167 |
| msmarco | multi-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1354 | 0.0784 | 0.1921 | 0.666 | 0.001 |
| msmarco | multi-positive | 716 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0216 | -0.0018 | 0.0630 | 0.250 | 0.084 |
| msmarco | multi-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.1570 | 0.0865 | 0.2263 | 0.666 | 0.000 |
| msmarco | multi-positive | 716 | ndcg@10 | prefix-k minus full query | 16 | -0.0067 | -0.0147 | 0.0001 | 0.084 | 0.333 |
| msmarco | multi-positive | 716 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1708 | 0.0540 | 0.3176 | 0.583 | 0.001 |
| msmarco | multi-positive | 716 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0570 | -0.0042 | 0.1715 | 0.167 | 0.084 |
| msmarco | multi-positive | 716 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.2278 | 0.0892 | 0.3841 | 0.666 | 0.000 |
| msmarco | multi-positive | 716 | mrr@10 | prefix-k minus full query | 16 | -0.0158 | -0.0337 | -0.0001 | 0.084 | 0.333 |
| msmarco | multi-positive | 716 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1750 | 0.0610 | 0.3203 | 0.916 | 0.001 |
| msmarco | multi-positive | 716 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0614 | -0.0001 | 0.1753 | 0.583 | 0.084 |
| msmarco | multi-positive | 716 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.2364 | 0.1065 | 0.3901 | 1.000 | 0.000 |
| msmarco | multi-positive | 716 | mrr@100 | prefix-k minus full query | 16 | -0.0141 | -0.0325 | 0.0017 | 0.417 | 0.333 |
| msmarco | multi-positive | 716 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1667 | 0.0419 | 0.2917 | 0.334 | 0.000 |
| msmarco | multi-positive | 716 | recall@100 | L2S (in-sample) minus full query | 16 | 0.1246 | -0.0001 | 0.2494 | 0.250 | 0.001 |
| msmarco | multi-positive | 716 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.2913 | 0.0833 | 0.5408 | 0.417 | 0.000 |
| msmarco | multi-positive | 716 | recall@100 | prefix-k minus full query | 16 | -0.0002 | -0.1250 | 0.1245 | 0.083 | 0.084 |
| msmarco | multi-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.1143 | 0.0421 | 0.1912 | 0.583 | 0.001 |
| msmarco | multi-positive | 819 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0059 | -0.0025 | 0.0164 | 0.250 | 0.084 |
| msmarco | multi-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.1202 | 0.0491 | 0.1959 | 0.583 | 0.000 |
| msmarco | multi-positive | 819 | ndcg@10 | prefix-k minus full query | 16 | -0.0074 | -0.0187 | 0.0023 | 0.167 | 0.250 |
| msmarco | multi-positive | 819 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2019 | 0.0524 | 0.3776 | 0.583 | 0.001 |
| msmarco | multi-positive | 819 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0036 | -0.0102 | 0.0208 | 0.167 | 0.084 |
| msmarco | multi-positive | 819 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.2054 | 0.0643 | 0.3727 | 0.583 | 0.000 |
| msmarco | multi-positive | 819 | mrr@10 | prefix-k minus full query | 16 | -0.0185 | -0.0428 | 0.0029 | 0.167 | 0.250 |
| msmarco | multi-positive | 819 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2103 | 0.0639 | 0.3838 | 0.999 | 0.001 |
| msmarco | multi-positive | 819 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0055 | -0.0094 | 0.0228 | 0.500 | 0.084 |
| msmarco | multi-positive | 819 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.2158 | 0.0800 | 0.3792 | 1.000 | 0.000 |
| msmarco | multi-positive | 819 | mrr@100 | prefix-k minus full query | 16 | -0.0181 | -0.0425 | 0.0036 | 0.417 | 0.250 |
| msmarco | multi-positive | 819 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2081 | 0.0417 | 0.4161 | 0.334 | 0.000 |
| msmarco | multi-positive | 819 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0832 | 0.0000 | 0.2080 | 0.166 | 0.000 |
| msmarco | multi-positive | 819 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.2913 | 0.0833 | 0.5408 | 0.417 | 0.000 |
| msmarco | multi-positive | 819 | recall@100 | prefix-k minus full query | 16 | 0.0000 | -0.1248 | 0.1248 | 0.083 | 0.083 |
| msmarco | multi-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.0606 | 0.0085 | 0.1260 | 0.417 | 0.000 |
| msmarco | multi-positive | 921 | ndcg@10 | L2S (in-sample) minus full query | 16 | 0.0083 | -0.0000 | 0.0197 | 0.250 | 0.001 |
| msmarco | multi-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus full query | 16 | 0.0689 | 0.0118 | 0.1424 | 0.417 | 0.000 |
| msmarco | multi-positive | 921 | ndcg@10 | prefix-k minus full query | 16 | -0.0017 | -0.0145 | 0.0123 | 0.083 | 0.250 |
| msmarco | multi-positive | 921 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.0819 | 0.0085 | 0.1748 | 0.334 | 0.000 |
| msmarco | multi-positive | 921 | mrr@10 | L2S (in-sample) minus full query | 16 | 0.0173 | -0.0001 | 0.0451 | 0.166 | 0.000 |
| msmarco | multi-positive | 921 | mrr@10 | all-positive oracle (privileged) minus full query | 16 | 0.0992 | 0.0105 | 0.2247 | 0.334 | 0.000 |
| msmarco | multi-positive | 921 | mrr@10 | prefix-k minus full query | 16 | -0.0005 | -0.0282 | 0.0353 | 0.083 | 0.250 |
| msmarco | multi-positive | 921 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.0939 | 0.0258 | 0.1871 | 0.833 | 0.000 |
| msmarco | multi-positive | 921 | mrr@100 | L2S (in-sample) minus full query | 16 | 0.0172 | -0.0004 | 0.0452 | 0.250 | 0.167 |
| msmarco | multi-positive | 921 | mrr@100 | all-positive oracle (privileged) minus full query | 16 | 0.1111 | 0.0284 | 0.2338 | 0.833 | 0.000 |
| msmarco | multi-positive | 921 | mrr@100 | prefix-k minus full query | 16 | 0.0008 | -0.0275 | 0.0368 | 0.333 | 0.250 |
| msmarco | multi-positive | 921 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 16 | 0.2081 | 0.0417 | 0.4161 | 0.334 | 0.000 |
| msmarco | multi-positive | 921 | recall@100 | L2S (in-sample) minus full query | 16 | 0.0416 | 0.0000 | 0.1248 | 0.083 | 0.000 |
| msmarco | multi-positive | 921 | recall@100 | all-positive oracle (privileged) minus full query | 16 | 0.2497 | 0.0833 | 0.4577 | 0.417 | 0.000 |
| msmarco | multi-positive | 921 | recall@100 | prefix-k minus full query | 16 | 0.0000 | -0.1248 | 0.1248 | 0.083 | 0.083 |

## Held-out positives of multi-positive queries

Same partitions for every arm. Measures use only the held-out positives.

| environment | cohort | k | measure | comparison | queries | difference | 2.5% | 97.5% | share better | share worse |
|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | multi-positive | 102 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0183 | -0.0719 | 0.0184 | 0.500 | 0.500 |
| msmarco | multi-positive | 102 | held_rr | L2S (in-sample) minus full query | 16 | 0.0235 | -0.0075 | 0.0624 | 0.583 | 0.417 |
| msmarco | multi-positive | 102 | held_rr | held-out transfer (1-1) minus full query | 16 | 0.0052 | -0.0266 | 0.0404 | 0.250 | 0.750 |
| msmarco | multi-positive | 102 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2261 | 0.1419 | 0.3117 | 0.917 | 0.083 |
| msmarco | multi-positive | 102 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0645 | -0.7410 | 0.6031 | 0.500 | 0.500 |
| msmarco | multi-positive | 102 | held_logrank | L2S (in-sample) minus full query | 16 | 0.3192 | -0.0495 | 0.6699 | 0.750 | 0.250 |
| msmarco | multi-positive | 102 | held_logrank | held-out transfer (1-1) minus full query | 16 | 0.2546 | -0.3487 | 0.9015 | 0.667 | 0.333 |
| msmarco | multi-positive | 102 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | -2.0203 | -2.4926 | -1.5368 | 0.000 | 1.000 |
| msmarco | multi-positive | 102 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0695 | -0.2219 | 0.0830 | 0.083 | 0.250 |
| msmarco | multi-positive | 102 | held_recall10 | L2S (in-sample) minus full query | 16 | 0.0277 | -0.0004 | 0.0834 | 0.084 | 0.001 |
| msmarco | multi-positive | 102 | held_recall10 | held-out transfer (1-1) minus full query | 16 | -0.0418 | -0.1669 | 0.0832 | 0.084 | 0.167 |
| msmarco | multi-positive | 102 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.3471 | 0.1944 | 0.5272 | 0.667 | 0.000 |
| msmarco | multi-positive | 102 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0139 | -0.2635 | 0.1941 | 0.250 | 0.250 |
| msmarco | multi-positive | 102 | held_recall100 | L2S (in-sample) minus full query | 16 | -0.0417 | -0.1666 | 0.0831 | 0.083 | 0.167 |
| msmarco | multi-positive | 102 | held_recall100 | held-out transfer (1-1) minus full query | 16 | -0.0556 | -0.2913 | 0.1524 | 0.166 | 0.251 |
| msmarco | multi-positive | 102 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.3678 | 0.1461 | 0.6104 | 0.584 | 0.000 |
| msmarco | multi-positive | 204 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0129 | -0.0631 | 0.0296 | 0.416 | 0.584 |
| msmarco | multi-positive | 204 | held_rr | L2S (in-sample) minus full query | 16 | 0.0146 | -0.0094 | 0.0482 | 0.500 | 0.500 |
| msmarco | multi-positive | 204 | held_rr | held-out transfer (1-1) minus full query | 16 | 0.0017 | -0.0243 | 0.0336 | 0.416 | 0.584 |
| msmarco | multi-positive | 204 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.1590 | 0.1104 | 0.2051 | 1.000 | 0.000 |
| msmarco | multi-positive | 204 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 16 | 0.0608 | -0.4228 | 0.5235 | 0.584 | 0.416 |
| msmarco | multi-positive | 204 | held_logrank | L2S (in-sample) minus full query | 16 | 0.0178 | -0.1012 | 0.1261 | 0.500 | 0.500 |
| msmarco | multi-positive | 204 | held_logrank | held-out transfer (1-1) minus full query | 16 | 0.0786 | -0.4052 | 0.5515 | 0.750 | 0.250 |
| msmarco | multi-positive | 204 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | -1.8427 | -2.2780 | -1.3848 | 0.000 | 1.000 |
| msmarco | multi-positive | 204 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0939 | -0.2083 | -0.0004 | 0.000 | 0.251 |
| msmarco | multi-positive | 204 | held_recall10 | L2S (in-sample) minus full query | 16 | 0.0001 | -0.0002 | 0.0004 | 0.001 | 0.001 |
| msmarco | multi-positive | 204 | held_recall10 | held-out transfer (1-1) minus full query | 16 | -0.0938 | -0.2082 | -0.0003 | 0.000 | 0.250 |
| msmarco | multi-positive | 204 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.5100 | 0.3124 | 0.7072 | 0.833 | 0.000 |
| msmarco | multi-positive | 204 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 16 | 0.0415 | -0.0834 | 0.1664 | 0.167 | 0.084 |
| msmarco | multi-positive | 204 | held_recall100 | L2S (in-sample) minus full query | 16 | 0.0415 | -0.0001 | 0.1248 | 0.083 | 0.000 |
| msmarco | multi-positive | 204 | held_recall100 | held-out transfer (1-1) minus full query | 16 | 0.0830 | -0.0002 | 0.2078 | 0.166 | 0.001 |
| msmarco | multi-positive | 204 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2290 | 0.0419 | 0.4578 | 0.334 | 0.000 |
| msmarco | multi-positive | 307 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0134 | -0.0324 | 0.0087 | 0.167 | 0.833 |
| msmarco | multi-positive | 307 | held_rr | L2S (in-sample) minus full query | 16 | 0.0101 | 0.0021 | 0.0213 | 0.832 | 0.168 |
| msmarco | multi-positive | 307 | held_rr | held-out transfer (1-1) minus full query | 16 | -0.0033 | -0.0238 | 0.0203 | 0.333 | 0.667 |
| msmarco | multi-positive | 307 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.1356 | 0.0890 | 0.1840 | 1.000 | 0.000 |
| msmarco | multi-positive | 307 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 16 | 0.1910 | -0.2823 | 0.5677 | 0.667 | 0.333 |
| msmarco | multi-positive | 307 | held_logrank | L2S (in-sample) minus full query | 16 | -0.1185 | -0.2032 | -0.0367 | 0.251 | 0.749 |
| msmarco | multi-positive | 307 | held_logrank | held-out transfer (1-1) minus full query | 16 | 0.0724 | -0.3473 | 0.4612 | 0.667 | 0.333 |
| msmarco | multi-positive | 307 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | -1.6201 | -2.0205 | -1.2227 | 0.000 | 1.000 |
| msmarco | multi-positive | 307 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0935 | -0.2080 | 0.0000 | 0.000 | 0.250 |
| msmarco | multi-positive | 307 | held_recall10 | L2S (in-sample) minus full query | 16 | -0.0001 | -0.0002 | 0.0000 | 0.000 | 0.000 |
| msmarco | multi-positive | 307 | held_recall10 | held-out transfer (1-1) minus full query | 16 | -0.0936 | -0.2080 | 0.0000 | 0.000 | 0.250 |
| msmarco | multi-positive | 307 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.3571 | 0.1630 | 0.5584 | 0.666 | 0.000 |
| msmarco | multi-positive | 307 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0831 | -0.2491 | 0.0830 | 0.084 | 0.250 |
| msmarco | multi-positive | 307 | held_recall100 | L2S (in-sample) minus full query | 16 | 0.1246 | 0.0000 | 0.2494 | 0.250 | 0.000 |
| msmarco | multi-positive | 307 | held_recall100 | held-out transfer (1-1) minus full query | 16 | 0.0415 | -0.0002 | 0.1248 | 0.083 | 0.000 |
| msmarco | multi-positive | 307 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2706 | 0.0834 | 0.4993 | 0.417 | 0.000 |
| msmarco | multi-positive | 409 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0016 | -0.0201 | 0.0210 | 0.334 | 0.666 |
| msmarco | multi-positive | 409 | held_rr | L2S (in-sample) minus full query | 16 | 0.0038 | -0.0073 | 0.0170 | 0.666 | 0.334 |
| msmarco | multi-positive | 409 | held_rr | held-out transfer (1-1) minus full query | 16 | 0.0021 | -0.0157 | 0.0204 | 0.416 | 0.584 |
| msmarco | multi-positive | 409 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.0969 | 0.0693 | 0.1230 | 1.000 | 0.000 |
| msmarco | multi-positive | 409 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 16 | 0.1103 | -0.3331 | 0.4456 | 0.666 | 0.334 |
| msmarco | multi-positive | 409 | held_logrank | L2S (in-sample) minus full query | 16 | -0.0940 | -0.2060 | 0.0190 | 0.334 | 0.666 |
| msmarco | multi-positive | 409 | held_logrank | held-out transfer (1-1) minus full query | 16 | 0.0163 | -0.3728 | 0.3704 | 0.583 | 0.417 |
| msmarco | multi-positive | 409 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | -1.4072 | -1.8229 | -1.0436 | 0.000 | 1.000 |
| msmarco | multi-positive | 409 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0519 | -0.1351 | 0.0002 | 0.001 | 0.166 |
| msmarco | multi-positive | 409 | held_recall10 | L2S (in-sample) minus full query | 16 | -0.0417 | -0.1249 | -0.0001 | 0.000 | 0.084 |
| msmarco | multi-positive | 409 | held_recall10 | held-out transfer (1-1) minus full query | 16 | -0.0936 | -0.2080 | 0.0000 | 0.000 | 0.250 |
| msmarco | multi-positive | 409 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.3296 | 0.1285 | 0.5408 | 0.584 | 0.000 |
| msmarco | multi-positive | 409 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0832 | -0.2492 | 0.0829 | 0.084 | 0.250 |
| msmarco | multi-positive | 409 | held_recall100 | L2S (in-sample) minus full query | 16 | 0.1246 | 0.0000 | 0.2494 | 0.250 | 0.000 |
| msmarco | multi-positive | 409 | held_recall100 | held-out transfer (1-1) minus full query | 16 | 0.0414 | -0.0003 | 0.1247 | 0.083 | 0.001 |
| msmarco | multi-positive | 409 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2707 | 0.0835 | 0.4994 | 0.417 | 0.000 |
| msmarco | multi-positive | 512 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0158 | -0.0360 | 0.0047 | 0.250 | 0.750 |
| msmarco | multi-positive | 512 | held_rr | L2S (in-sample) minus full query | 16 | 0.0102 | -0.0012 | 0.0256 | 0.750 | 0.250 |
| msmarco | multi-positive | 512 | held_rr | held-out transfer (1-1) minus full query | 16 | -0.0055 | -0.0231 | 0.0134 | 0.250 | 0.750 |
| msmarco | multi-positive | 512 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.0975 | 0.0633 | 0.1344 | 1.000 | 0.000 |
| msmarco | multi-positive | 512 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 16 | 0.1574 | -0.2686 | 0.5051 | 0.667 | 0.333 |
| msmarco | multi-positive | 512 | held_logrank | L2S (in-sample) minus full query | 16 | -0.1360 | -0.2517 | -0.0327 | 0.167 | 0.833 |
| msmarco | multi-positive | 512 | held_logrank | held-out transfer (1-1) minus full query | 16 | 0.0214 | -0.3487 | 0.3862 | 0.666 | 0.334 |
| msmarco | multi-positive | 512 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | -1.3229 | -1.6940 | -0.9889 | 0.000 | 1.000 |
| msmarco | multi-positive | 512 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0520 | -0.1352 | 0.0000 | 0.000 | 0.166 |
| msmarco | multi-positive | 512 | held_recall10 | L2S (in-sample) minus full query | 16 | -0.0416 | -0.1249 | 0.0000 | 0.000 | 0.084 |
| msmarco | multi-positive | 512 | held_recall10 | held-out transfer (1-1) minus full query | 16 | -0.0936 | -0.2081 | -0.0001 | 0.000 | 0.250 |
| msmarco | multi-positive | 512 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2880 | 0.0938 | 0.5065 | 0.501 | 0.000 |
| msmarco | multi-positive | 512 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0831 | -0.2492 | 0.0830 | 0.084 | 0.250 |
| msmarco | multi-positive | 512 | held_recall100 | L2S (in-sample) minus full query | 16 | 0.1246 | 0.0000 | 0.2494 | 0.250 | 0.000 |
| msmarco | multi-positive | 512 | held_recall100 | held-out transfer (1-1) minus full query | 16 | 0.0415 | -0.0002 | 0.1248 | 0.083 | 0.000 |
| msmarco | multi-positive | 512 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2706 | 0.0833 | 0.4992 | 0.417 | 0.000 |
| msmarco | multi-positive | 614 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0217 | -0.0621 | 0.0068 | 0.333 | 0.667 |
| msmarco | multi-positive | 614 | held_rr | L2S (in-sample) minus full query | 16 | 0.0205 | -0.0033 | 0.0626 | 0.666 | 0.334 |
| msmarco | multi-positive | 614 | held_rr | held-out transfer (1-1) minus full query | 16 | -0.0013 | -0.0173 | 0.0168 | 0.417 | 0.583 |
| msmarco | multi-positive | 614 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.0997 | 0.0609 | 0.1407 | 1.000 | 0.000 |
| msmarco | multi-positive | 614 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 16 | 0.1410 | -0.1971 | 0.4507 | 0.583 | 0.417 |
| msmarco | multi-positive | 614 | held_logrank | L2S (in-sample) minus full query | 16 | -0.1566 | -0.3166 | -0.0227 | 0.250 | 0.750 |
| msmarco | multi-positive | 614 | held_logrank | held-out transfer (1-1) minus full query | 16 | -0.0157 | -0.3054 | 0.2884 | 0.666 | 0.334 |
| msmarco | multi-positive | 614 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | -1.2051 | -1.5293 | -0.9115 | 0.000 | 1.000 |
| msmarco | multi-positive | 614 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0519 | -0.1352 | 0.0001 | 0.000 | 0.166 |
| msmarco | multi-positive | 614 | held_recall10 | L2S (in-sample) minus full query | 16 | -0.0417 | -0.1249 | -0.0001 | 0.000 | 0.084 |
| msmarco | multi-positive | 614 | held_recall10 | held-out transfer (1-1) minus full query | 16 | -0.0936 | -0.2081 | -0.0001 | 0.000 | 0.250 |
| msmarco | multi-positive | 614 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2880 | 0.0937 | 0.5065 | 0.500 | 0.000 |
| msmarco | multi-positive | 614 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0832 | -0.2494 | 0.0830 | 0.084 | 0.250 |
| msmarco | multi-positive | 614 | held_recall100 | L2S (in-sample) minus full query | 16 | 0.1247 | 0.0000 | 0.2495 | 0.250 | 0.000 |
| msmarco | multi-positive | 614 | held_recall100 | held-out transfer (1-1) minus full query | 16 | 0.0415 | -0.0002 | 0.1248 | 0.083 | 0.000 |
| msmarco | multi-positive | 614 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2497 | 0.0419 | 0.4991 | 0.333 | 0.000 |
| msmarco | multi-positive | 716 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0249 | -0.0644 | 0.0040 | 0.250 | 0.750 |
| msmarco | multi-positive | 716 | held_rr | L2S (in-sample) minus full query | 16 | 0.0229 | 0.0008 | 0.0634 | 0.749 | 0.168 |
| msmarco | multi-positive | 716 | held_rr | held-out transfer (1-1) minus full query | 16 | -0.0020 | -0.0179 | 0.0156 | 0.417 | 0.583 |
| msmarco | multi-positive | 716 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.1036 | 0.0600 | 0.1507 | 1.000 | 0.000 |
| msmarco | multi-positive | 716 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 16 | 0.1529 | -0.1624 | 0.4316 | 0.750 | 0.250 |
| msmarco | multi-positive | 716 | held_logrank | L2S (in-sample) minus full query | 16 | -0.1602 | -0.2960 | -0.0528 | 0.250 | 0.666 |
| msmarco | multi-positive | 716 | held_logrank | held-out transfer (1-1) minus full query | 16 | -0.0073 | -0.2692 | 0.2482 | 0.666 | 0.334 |
| msmarco | multi-positive | 716 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | -1.1294 | -1.4418 | -0.8444 | 0.000 | 1.000 |
| msmarco | multi-positive | 716 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0936 | -0.2081 | -0.0002 | 0.000 | 0.250 |
| msmarco | multi-positive | 716 | held_recall10 | L2S (in-sample) minus full query | 16 | 0.0001 | -0.0002 | 0.0003 | 0.000 | 0.000 |
| msmarco | multi-positive | 716 | held_recall10 | held-out transfer (1-1) minus full query | 16 | -0.0936 | -0.2080 | 0.0000 | 0.000 | 0.250 |
| msmarco | multi-positive | 716 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2878 | 0.0935 | 0.5064 | 0.500 | 0.000 |
| msmarco | multi-positive | 716 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0831 | -0.2494 | 0.0832 | 0.084 | 0.250 |
| msmarco | multi-positive | 716 | held_recall100 | L2S (in-sample) minus full query | 16 | 0.1247 | 0.0000 | 0.2495 | 0.250 | 0.000 |
| msmarco | multi-positive | 716 | held_recall100 | held-out transfer (1-1) minus full query | 16 | 0.0416 | 0.0000 | 0.1248 | 0.083 | 0.000 |
| msmarco | multi-positive | 716 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2496 | 0.0417 | 0.4991 | 0.333 | 0.000 |
| msmarco | multi-positive | 819 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 16 | 0.0017 | -0.0195 | 0.0199 | 0.500 | 0.500 |
| msmarco | multi-positive | 819 | held_rr | L2S (in-sample) minus full query | 16 | 0.0059 | -0.0006 | 0.0141 | 0.749 | 0.251 |
| msmarco | multi-positive | 819 | held_rr | held-out transfer (1-1) minus full query | 16 | 0.0076 | -0.0105 | 0.0272 | 0.500 | 0.500 |
| msmarco | multi-positive | 819 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.0824 | 0.0427 | 0.1270 | 0.917 | 0.000 |
| msmarco | multi-positive | 819 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 16 | 0.0781 | -0.1825 | 0.3402 | 0.500 | 0.500 |
| msmarco | multi-positive | 819 | held_logrank | L2S (in-sample) minus full query | 16 | -0.1016 | -0.1553 | -0.0512 | 0.167 | 0.833 |
| msmarco | multi-positive | 819 | held_logrank | held-out transfer (1-1) minus full query | 16 | -0.0234 | -0.2598 | 0.2135 | 0.416 | 0.584 |
| msmarco | multi-positive | 819 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | -1.0061 | -1.3000 | -0.7214 | 0.000 | 0.917 |
| msmarco | multi-positive | 819 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0936 | -0.2081 | -0.0002 | 0.000 | 0.250 |
| msmarco | multi-positive | 819 | held_recall10 | L2S (in-sample) minus full query | 16 | 0.0001 | -0.0002 | 0.0003 | 0.000 | 0.000 |
| msmarco | multi-positive | 819 | held_recall10 | held-out transfer (1-1) minus full query | 16 | -0.0936 | -0.2080 | 0.0000 | 0.000 | 0.250 |
| msmarco | multi-positive | 819 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2045 | 0.0519 | 0.3848 | 0.416 | 0.000 |
| msmarco | multi-positive | 819 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0417 | -0.1666 | 0.0832 | 0.083 | 0.167 |
| msmarco | multi-positive | 819 | held_recall100 | L2S (in-sample) minus full query | 16 | 0.0832 | 0.0000 | 0.2080 | 0.166 | 0.000 |
| msmarco | multi-positive | 819 | held_recall100 | held-out transfer (1-1) minus full query | 16 | 0.0415 | -0.0002 | 0.1248 | 0.083 | 0.000 |
| msmarco | multi-positive | 819 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2497 | 0.0419 | 0.4991 | 0.333 | 0.000 |
| msmarco | multi-positive | 921 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0080 | -0.0190 | 0.0009 | 0.334 | 0.666 |
| msmarco | multi-positive | 921 | held_rr | L2S (in-sample) minus full query | 16 | 0.0069 | -0.0004 | 0.0176 | 0.583 | 0.417 |
| msmarco | multi-positive | 921 | held_rr | held-out transfer (1-1) minus full query | 16 | -0.0010 | -0.0134 | 0.0124 | 0.417 | 0.583 |
| msmarco | multi-positive | 921 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.0520 | 0.0211 | 0.0865 | 0.916 | 0.000 |
| msmarco | multi-positive | 921 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 16 | 0.0379 | -0.1369 | 0.2158 | 0.583 | 0.417 |
| msmarco | multi-positive | 921 | held_logrank | L2S (in-sample) minus full query | 16 | -0.0517 | -0.1024 | -0.0003 | 0.250 | 0.750 |
| msmarco | multi-positive | 921 | held_logrank | held-out transfer (1-1) minus full query | 16 | -0.0138 | -0.1888 | 0.1569 | 0.500 | 0.500 |
| msmarco | multi-positive | 921 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | -0.7293 | -0.9414 | -0.5142 | 0.000 | 0.917 |
| msmarco | multi-positive | 921 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0832 | -0.2080 | 0.0000 | 0.000 | 0.166 |
| msmarco | multi-positive | 921 | held_recall10 | L2S (in-sample) minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | multi-positive | 921 | held_recall10 | held-out transfer (1-1) minus full query | 16 | -0.0832 | -0.2080 | 0.0000 | 0.000 | 0.166 |
| msmarco | multi-positive | 921 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.1525 | 0.0000 | 0.3331 | 0.250 | 0.000 |
| msmarco | multi-positive | 921 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 16 | -0.0416 | -0.1248 | 0.0000 | 0.000 | 0.083 |
| msmarco | multi-positive | 921 | held_recall100 | L2S (in-sample) minus full query | 16 | 0.0416 | 0.0000 | 0.1248 | 0.083 | 0.000 |
| msmarco | multi-positive | 921 | held_recall100 | held-out transfer (1-1) minus full query | 16 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| msmarco | multi-positive | 921 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 16 | 0.2496 | 0.0832 | 0.4575 | 0.416 | 0.000 |

## Positive ranks, and how scores moved against the full query

| environment | cohort | k | arm | rr | logrank | recall10 | recall100 | positive score change | at rank 1 change | at rank 10 change | at rank 100 change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | single-positive | 102 | prefix-k | 0.1377 (0.0388 to 0.2804) | 3.6950 (2.5733 to 4.9185) | 0.2500 (0.0625 to 0.5000) | 0.7500 (0.5625 to 0.9375) | -0.5997 (-0.6429 to -0.5557) | -0.6630 (-0.6873 to -0.6398) | -0.5951 (-0.6238 to -0.5634) | -0.5027 (-0.5310 to -0.4719) |
| msmarco | single-positive | 102 | all-positive oracle (privileged) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.3223 (-0.3582 to -0.2832) | -0.4011 (-0.4201 to -0.3848) | -0.3798 (-0.4003 to -0.3567) | -0.3199 (-0.3384 to -0.2995) |
| msmarco | single-positive | 102 | L2S (in-sample) | 0.3191 (0.1611 to 0.5098) | 2.1263 (1.2672 to 3.0489) | 0.7500 (0.5000 to 0.9375) | 0.8125 (0.6250 to 1.0000) | -0.4852 (-0.5220 to -0.4441) | -0.5521 (-0.5732 to -0.5316) | -0.4927 (-0.5216 to -0.4597) | -0.4123 (-0.4422 to -0.3808) |
| msmarco | single-positive | 102 | pair-optimal, target only (secondary) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.3223 (-0.3582 to -0.2832) | -0.4011 (-0.4201 to -0.3848) | -0.3798 (-0.4003 to -0.3567) | -0.3199 (-0.3384 to -0.2995) |
| msmarco | single-positive | 204 | prefix-k | 0.1903 (0.0552 to 0.3676) | 3.2586 (2.2262 to 4.3475) | 0.4375 (0.1875 to 0.6875) | 0.8125 (0.6250 to 1.0000) | -0.5490 (-0.5898 to -0.5075) | -0.6073 (-0.6299 to -0.5852) | -0.5444 (-0.5718 to -0.5143) | -0.4600 (-0.4867 to -0.4318) |
| msmarco | single-positive | 204 | all-positive oracle (privileged) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1613 (-0.1907 to -0.1296) | -0.2402 (-0.2568 to -0.2255) | -0.2387 (-0.2546 to -0.2206) | -0.2005 (-0.2127 to -0.1861) |
| msmarco | single-positive | 204 | L2S (in-sample) | 0.3040 (0.1423 to 0.4885) | 2.1848 (1.3735 to 3.0956) | 0.6250 (0.3750 to 0.8750) | 0.9375 (0.8125 to 1.0000) | -0.3801 (-0.4123 to -0.3479) | -0.4393 (-0.4587 to -0.4216) | -0.3896 (-0.4160 to -0.3601) | -0.3239 (-0.3519 to -0.2968) |
| msmarco | single-positive | 204 | pair-optimal, target only (secondary) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1613 (-0.1907 to -0.1296) | -0.2402 (-0.2568 to -0.2255) | -0.2387 (-0.2546 to -0.2206) | -0.2005 (-0.2127 to -0.1861) |
| msmarco | single-positive | 307 | prefix-k | 0.1581 (0.0590 to 0.3003) | 3.1729 (2.2643 to 4.1087) | 0.4375 (0.1875 to 0.6875) | 0.7500 (0.5625 to 0.9375) | -0.4840 (-0.5203 to -0.4459) | -0.5361 (-0.5572 to -0.5141) | -0.4809 (-0.5078 to -0.4516) | -0.4059 (-0.4311 to -0.3789) |
| msmarco | single-positive | 307 | all-positive oracle (privileged) | 0.9688 (0.9062 to 1.0000) | 0.0433 (0.0000 to 0.1300) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0644 (-0.0886 to -0.0379) | -0.1431 (-0.1594 to -0.1273) | -0.1514 (-0.1639 to -0.1383) | -0.1263 (-0.1343 to -0.1178) |
| msmarco | single-positive | 307 | L2S (in-sample) | 0.3379 (0.1591 to 0.5415) | 2.2486 (1.3523 to 3.2211) | 0.6250 (0.3750 to 0.8750) | 0.8750 (0.6875 to 1.0000) | -0.3056 (-0.3312 to -0.2804) | -0.3561 (-0.3717 to -0.3419) | -0.3148 (-0.3378 to -0.2896) | -0.2616 (-0.2864 to -0.2380) |
| msmarco | single-positive | 307 | pair-optimal, target only (secondary) | 0.9688 (0.9062 to 1.0000) | 0.0433 (0.0000 to 0.1300) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0644 (-0.0886 to -0.0379) | -0.1431 (-0.1594 to -0.1273) | -0.1514 (-0.1639 to -0.1383) | -0.1263 (-0.1343 to -0.1178) |
| msmarco | single-positive | 409 | prefix-k | 0.2083 (0.0723 to 0.3831) | 2.9607 (2.0327 to 3.8869) | 0.4375 (0.1875 to 0.6875) | 0.7500 (0.5625 to 0.9375) | -0.4009 (-0.4317 to -0.3669) | -0.4466 (-0.4646 to -0.4282) | -0.4003 (-0.4244 to -0.3733) | -0.3376 (-0.3594 to -0.3129) |
| msmarco | single-positive | 409 | all-positive oracle (privileged) | 0.8750 (0.7812 to 0.9688) | 0.1733 (0.0433 to 0.3033) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0056 (-0.0263 to 0.0162) | -0.0831 (-0.0995 to -0.0662) | -0.0966 (-0.1054 to -0.0876) | -0.0788 (-0.0855 to -0.0723) |
| msmarco | single-positive | 409 | L2S (in-sample) | 0.3274 (0.1477 to 0.5354) | 2.4144 (1.4537 to 3.4254) | 0.5000 (0.2500 to 0.7500) | 0.8125 (0.6250 to 1.0000) | -0.2447 (-0.2639 to -0.2250) | -0.2859 (-0.2996 to -0.2734) | -0.2522 (-0.2710 to -0.2304) | -0.2092 (-0.2286 to -0.1893) |
| msmarco | single-positive | 409 | pair-optimal, target only (secondary) | 0.8750 (0.7812 to 0.9688) | 0.1733 (0.0433 to 0.3033) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0056 (-0.0263 to 0.0162) | -0.0831 (-0.0995 to -0.0662) | -0.0966 (-0.1054 to -0.0876) | -0.0788 (-0.0855 to -0.0723) |
| msmarco | single-positive | 512 | prefix-k | 0.2461 (0.1026 to 0.4183) | 2.8001 (1.7991 to 3.8131) | 0.4375 (0.1875 to 0.6875) | 0.8750 (0.6875 to 1.0000) | -0.3311 (-0.3563 to -0.3027) | -0.3698 (-0.3860 to -0.3526) | -0.3312 (-0.3521 to -0.3078) | -0.2797 (-0.2993 to -0.2580) |
| msmarco | single-positive | 512 | all-positive oracle (privileged) | 0.8146 (0.6729 to 0.9375) | 0.2992 (0.0866 to 0.5437) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0286 (0.0108 to 0.0472) | -0.0459 (-0.0616 to -0.0294) | -0.0595 (-0.0677 to -0.0509) | -0.0485 (-0.0545 to -0.0428) |
| msmarco | single-positive | 512 | L2S (in-sample) | 0.3795 (0.1807 to 0.6030) | 2.3325 (1.3221 to 3.3637) | 0.5000 (0.2500 to 0.7500) | 0.8750 (0.6875 to 1.0000) | -0.1894 (-0.2047 to -0.1728) | -0.2237 (-0.2335 to -0.2135) | -0.1959 (-0.2108 to -0.1778) | -0.1602 (-0.1752 to -0.1449) |
| msmarco | single-positive | 512 | pair-optimal, target only (secondary) | 0.8146 (0.6729 to 0.9375) | 0.2992 (0.0866 to 0.5437) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0286 (0.0108 to 0.0472) | -0.0459 (-0.0616 to -0.0294) | -0.0595 (-0.0677 to -0.0509) | -0.0485 (-0.0545 to -0.0428) |
| msmarco | single-positive | 614 | prefix-k | 0.2453 (0.1021 to 0.4196) | 2.8028 (1.8003 to 3.8222) | 0.4375 (0.1875 to 0.6875) | 0.8125 (0.6250 to 1.0000) | -0.2609 (-0.2821 to -0.2378) | -0.2919 (-0.3066 to -0.2780) | -0.2605 (-0.2798 to -0.2408) | -0.2198 (-0.2373 to -0.2026) |
| msmarco | single-positive | 614 | all-positive oracle (privileged) | 0.7609 (0.5989 to 0.9064) | 0.4351 (0.1300 to 0.7782) | 0.9375 (0.8125 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0456 (0.0303 to 0.0620) | -0.0264 (-0.0421 to -0.0100) | -0.0378 (-0.0456 to -0.0296) | -0.0304 (-0.0365 to -0.0245) |
| msmarco | single-positive | 614 | L2S (in-sample) | 0.3682 (0.1705 to 0.5921) | 2.4173 (1.3749 to 3.4869) | 0.4375 (0.1875 to 0.6875) | 0.8125 (0.6250 to 1.0000) | -0.1426 (-0.1562 to -0.1284) | -0.1687 (-0.1774 to -0.1599) | -0.1474 (-0.1590 to -0.1342) | -0.1202 (-0.1315 to -0.1081) |
| msmarco | single-positive | 614 | pair-optimal, target only (secondary) | 0.7609 (0.5989 to 0.9064) | 0.4351 (0.1300 to 0.7782) | 0.9375 (0.8125 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0456 (0.0303 to 0.0620) | -0.0264 (-0.0421 to -0.0100) | -0.0378 (-0.0456 to -0.0296) | -0.0304 (-0.0365 to -0.0245) |
| msmarco | single-positive | 716 | prefix-k | 0.2777 (0.1171 to 0.4705) | 2.8004 (1.7435 to 3.8713) | 0.4375 (0.1875 to 0.6875) | 0.8125 (0.6250 to 1.0000) | -0.1961 (-0.2129 to -0.1785) | -0.2196 (-0.2322 to -0.2073) | -0.1950 (-0.2110 to -0.1794) | -0.1641 (-0.1774 to -0.1510) |
| msmarco | single-positive | 716 | all-positive oracle (privileged) | 0.6875 (0.5249 to 0.8438) | 0.5911 (0.2306 to 0.9949) | 0.9375 (0.8125 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0520 (0.0384 to 0.0667) | -0.0171 (-0.0329 to -0.0005) | -0.0276 (-0.0356 to -0.0191) | -0.0217 (-0.0280 to -0.0155) |
| msmarco | single-positive | 716 | L2S (in-sample) | 0.3073 (0.1417 to 0.5001) | 2.5006 (1.4973 to 3.5596) | 0.5000 (0.2500 to 0.7500) | 0.8125 (0.6250 to 1.0000) | -0.0985 (-0.1089 to -0.0873) | -0.1204 (-0.1293 to -0.1122) | -0.1051 (-0.1144 to -0.0944) | -0.0853 (-0.0954 to -0.0742) |
| msmarco | single-positive | 716 | pair-optimal, target only (secondary) | 0.6875 (0.5249 to 0.8438) | 0.5911 (0.2306 to 0.9949) | 0.9375 (0.8125 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0520 (0.0384 to 0.0667) | -0.0171 (-0.0329 to -0.0005) | -0.0276 (-0.0356 to -0.0191) | -0.0217 (-0.0280 to -0.0155) |
| msmarco | single-positive | 819 | prefix-k | 0.2687 (0.1143 to 0.4614) | 2.7855 (1.7450 to 3.8263) | 0.4375 (0.1875 to 0.6875) | 0.8125 (0.6250 to 1.0000) | -0.1263 (-0.1388 to -0.1136) | -0.1422 (-0.1517 to -0.1330) | -0.1266 (-0.1387 to -0.1148) | -0.1053 (-0.1146 to -0.0965) |
| msmarco | single-positive | 819 | all-positive oracle (privileged) | 0.5916 (0.4141 to 0.7761) | 0.8868 (0.4152 to 1.3943) | 0.8750 (0.6875 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0514 (0.0393 to 0.0645) | -0.0129 (-0.0279 to 0.0024) | -0.0215 (-0.0285 to -0.0140) | -0.0166 (-0.0223 to -0.0110) |
| msmarco | single-positive | 819 | L2S (in-sample) | 0.3470 (0.1637 to 0.5543) | 2.4668 (1.4206 to 3.5308) | 0.5000 (0.2500 to 0.7500) | 0.8125 (0.6250 to 1.0000) | -0.0619 (-0.0702 to -0.0527) | -0.0783 (-0.0856 to -0.0709) | -0.0673 (-0.0749 to -0.0592) | -0.0546 (-0.0620 to -0.0469) |
| msmarco | single-positive | 819 | pair-optimal, target only (secondary) | 0.5916 (0.4141 to 0.7761) | 0.8868 (0.4152 to 1.3943) | 0.8750 (0.6875 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0514 (0.0393 to 0.0645) | -0.0129 (-0.0279 to 0.0024) | -0.0215 (-0.0285 to -0.0140) | -0.0166 (-0.0223 to -0.0110) |
| msmarco | single-positive | 921 | prefix-k | 0.2485 (0.1041 to 0.4201) | 2.7920 (1.8004 to 3.8158) | 0.4375 (0.1875 to 0.6875) | 0.8125 (0.6250 to 1.0000) | -0.0609 (-0.0675 to -0.0547) | -0.0685 (-0.0737 to -0.0631) | -0.0604 (-0.0668 to -0.0534) | -0.0504 (-0.0559 to -0.0443) |
| msmarco | single-positive | 921 | all-positive oracle (privileged) | 0.4759 (0.2827 to 0.6882) | 1.4538 (0.7672 to 2.1722) | 0.7500 (0.5625 to 0.9375) | 0.9375 (0.8125 to 1.0000) | 0.0419 (0.0328 to 0.0515) | -0.0077 (-0.0188 to 0.0036) | -0.0143 (-0.0187 to -0.0093) | -0.0104 (-0.0144 to -0.0064) |
| msmarco | single-positive | 921 | L2S (in-sample) | 0.2704 (0.1258 to 0.4491) | 2.6388 (1.6283 to 3.6824) | 0.4375 (0.1875 to 0.6875) | 0.8125 (0.6250 to 1.0000) | -0.0290 (-0.0350 to -0.0231) | -0.0383 (-0.0440 to -0.0326) | -0.0330 (-0.0375 to -0.0285) | -0.0252 (-0.0297 to -0.0208) |
| msmarco | single-positive | 921 | pair-optimal, target only (secondary) | 0.4759 (0.2827 to 0.6882) | 1.4538 (0.7672 to 2.1722) | 0.7500 (0.5625 to 0.9375) | 0.9375 (0.8125 to 1.0000) | 0.0419 (0.0328 to 0.0515) | -0.0077 (-0.0188 to 0.0036) | -0.0143 (-0.0187 to -0.0093) | -0.0104 (-0.0144 to -0.0064) |
| msmarco | multi-positive | 102 | prefix-k | 0.0693 (0.0161 to 0.1555) | 4.7500 (3.6665 to 5.7818) | 0.1597 (0.0418 to 0.3056) | 0.4930 (0.2641 to 0.7222) | -0.5583 (-0.6024 to -0.5175) | -0.6693 (-0.7070 to -0.6314) | -0.6002 (-0.6413 to -0.5653) | -0.5143 (-0.5610 to -0.4730) |
| msmarco | multi-positive | 102 | all-positive oracle (privileged) | 0.3071 (0.1996 to 0.4221) | 2.0487 (1.5941 to 2.4720) | 0.5759 (0.4159 to 0.7419) | 0.9371 (0.8334 to 0.9997) | -0.3111 (-0.3417 to -0.2842) | -0.4261 (-0.4547 to -0.3980) | -0.3739 (-0.3990 to -0.3500) | -0.3180 (-0.3446 to -0.2948) |
| msmarco | multi-positive | 102 | L2S (in-sample) | 0.0985 (0.0318 to 0.1845) | 4.1049 (3.0504 to 5.1118) | 0.2983 (0.1043 to 0.5201) | 0.5831 (0.3338 to 0.7912) | -0.4617 (-0.5013 to -0.4271) | -0.5587 (-0.5908 to -0.5262) | -0.5004 (-0.5364 to -0.4685) | -0.4268 (-0.4680 to -0.3895) |
| msmarco | multi-positive | 102 | pair-optimal, target only (secondary) | 0.8559 (0.7571 to 0.9477) | 0.2624 (0.0870 to 0.4498) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.2928 (-0.3248 to -0.2639) | n/a | n/a | n/a |
| msmarco | multi-positive | 204 | prefix-k | 0.0864 (0.0317 to 0.1638) | 4.1283 (3.0788 to 5.1153) | 0.2707 (0.1110 to 0.4581) | 0.5624 (0.3336 to 0.7912) | -0.5082 (-0.5485 to -0.4720) | -0.6131 (-0.6478 to -0.5771) | -0.5476 (-0.5841 to -0.5168) | -0.4690 (-0.5097 to -0.4329) |
| msmarco | multi-positive | 204 | all-positive oracle (privileged) | 0.2364 (0.1561 to 0.3218) | 2.0505 (1.5930 to 2.5159) | 0.6869 (0.4993 to 0.8740) | 0.9371 (0.8334 to 0.9997) | -0.1597 (-0.1811 to -0.1398) | -0.2675 (-0.2947 to -0.2417) | -0.2285 (-0.2463 to -0.2120) | -0.1907 (-0.2061 to -0.1771) |
| msmarco | multi-positive | 204 | L2S (in-sample) | 0.0896 (0.0319 to 0.1670) | 3.8054 (2.9775 to 4.6380) | 0.2706 (0.1042 to 0.4581) | 0.6664 (0.4582 to 0.8743) | -0.3711 (-0.4088 to -0.3388) | -0.4544 (-0.4851 to -0.4245) | -0.4064 (-0.4395 to -0.3770) | -0.3445 (-0.3814 to -0.3114) |
| msmarco | multi-positive | 204 | pair-optimal, target only (secondary) | 0.8348 (0.7375 to 0.9255) | 0.3005 (0.1177 to 0.5068) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1356 (-0.1592 to -0.1136) | n/a | n/a | n/a |
| msmarco | multi-positive | 307 | prefix-k | 0.0847 (0.0354 to 0.1442) | 3.9119 (3.0002 to 4.7753) | 0.2776 (0.0973 to 0.4856) | 0.6040 (0.3546 to 0.8329) | -0.4526 (-0.4918 to -0.4162) | -0.5486 (-0.5830 to -0.5149) | -0.4882 (-0.5253 to -0.4592) | -0.4175 (-0.4578 to -0.3841) |
| msmarco | multi-positive | 307 | all-positive oracle (privileged) | 0.2076 (0.1311 to 0.2938) | 2.2694 (1.7698 to 2.7607) | 0.5343 (0.3059 to 0.7493) | 0.9371 (0.8334 to 0.9997) | -0.0699 (-0.0866 to -0.0531) | -0.1656 (-0.1928 to -0.1426) | -0.1386 (-0.1537 to -0.1250) | -0.1106 (-0.1220 to -0.1004) |
| msmarco | multi-positive | 307 | L2S (in-sample) | 0.0850 (0.0387 to 0.1379) | 3.6846 (2.8792 to 4.5267) | 0.2707 (0.1042 to 0.4581) | 0.7495 (0.5818 to 0.9159) | -0.2940 (-0.3218 to -0.2691) | -0.3682 (-0.3908 to -0.3437) | -0.3255 (-0.3489 to -0.3037) | -0.2743 (-0.3025 to -0.2503) |
| msmarco | multi-positive | 307 | pair-optimal, target only (secondary) | 0.7634 (0.6583 to 0.8691) | 0.4610 (0.2126 to 0.7631) | 0.9583 (0.8751 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0427 (-0.0618 to -0.0243) | n/a | n/a | n/a |
| msmarco | multi-positive | 409 | prefix-k | 0.0880 (0.0313 to 0.1661) | 3.7957 (2.9338 to 4.6394) | 0.2291 (0.0628 to 0.4300) | 0.6040 (0.3960 to 0.8120) | -0.3741 (-0.4076 to -0.3420) | -0.4560 (-0.4871 to -0.4237) | -0.4049 (-0.4357 to -0.3792) | -0.3458 (-0.3782 to -0.3187) |
| msmarco | multi-positive | 409 | all-positive oracle (privileged) | 0.1743 (0.1075 to 0.2484) | 2.4227 (1.9037 to 2.9513) | 0.5066 (0.2919 to 0.7146) | 0.9371 (0.8334 to 0.9997) | -0.0154 (-0.0303 to -0.0007) | -0.1019 (-0.1262 to -0.0821) | -0.0828 (-0.0975 to -0.0707) | -0.0613 (-0.0708 to -0.0528) |
| msmarco | multi-positive | 409 | L2S (in-sample) | 0.0786 (0.0340 to 0.1306) | 3.7177 (2.9298 to 4.5459) | 0.2290 (0.0628 to 0.4299) | 0.7495 (0.5818 to 0.9159) | -0.2345 (-0.2576 to -0.2138) | -0.2958 (-0.3142 to -0.2770) | -0.2602 (-0.2781 to -0.2427) | -0.2187 (-0.2402 to -0.1992) |
| msmarco | multi-positive | 409 | pair-optimal, target only (secondary) | 0.6371 (0.4794 to 0.7887) | 0.7059 (0.3531 to 1.1209) | 0.9583 (0.8751 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0127 (-0.0040 to 0.0287) | n/a | n/a | n/a |
| msmarco | multi-positive | 512 | prefix-k | 0.0757 (0.0245 to 0.1509) | 3.9481 (3.1107 to 4.7295) | 0.2291 (0.0629 to 0.4300) | 0.6040 (0.3960 to 0.8120) | -0.3106 (-0.3409 to -0.2824) | -0.3742 (-0.4019 to -0.3445) | -0.3341 (-0.3630 to -0.3103) | -0.2850 (-0.3146 to -0.2606) |
| msmarco | multi-positive | 512 | all-positive oracle (privileged) | 0.1671 (0.0984 to 0.2451) | 2.5148 (1.9435 to 3.0853) | 0.4650 (0.2364 to 0.6939) | 0.9371 (0.8334 to 0.9997) | 0.0172 (0.0039 to 0.0301) | -0.0632 (-0.0835 to -0.0458) | -0.0468 (-0.0586 to -0.0361) | -0.0308 (-0.0388 to -0.0227) |
| msmarco | multi-positive | 512 | L2S (in-sample) | 0.0851 (0.0359 to 0.1389) | 3.6769 (2.8369 to 4.5459) | 0.2291 (0.0628 to 0.4300) | 0.7495 (0.5818 to 0.9159) | -0.1816 (-0.2008 to -0.1647) | -0.2321 (-0.2488 to -0.2162) | -0.2045 (-0.2205 to -0.1892) | -0.1720 (-0.1911 to -0.1546) |
| msmarco | multi-positive | 512 | pair-optimal, target only (secondary) | 0.5584 (0.4026 to 0.7102) | 0.9345 (0.5284 to 1.4200) | 0.8958 (0.7711 to 0.9997) | 1.0000 (1.0000 to 1.0000) | 0.0445 (0.0294 to 0.0587) | n/a | n/a | n/a |
| msmarco | multi-positive | 614 | prefix-k | 0.0663 (0.0265 to 0.1166) | 3.9150 (3.1500 to 4.6468) | 0.2292 (0.0630 to 0.4301) | 0.6039 (0.3959 to 0.8119) | -0.2476 (-0.2755 to -0.2211) | -0.2993 (-0.3214 to -0.2764) | -0.2666 (-0.2926 to -0.2454) | -0.2277 (-0.2537 to -0.2066) |
| msmarco | multi-positive | 614 | all-positive oracle (privileged) | 0.1736 (0.0954 to 0.2602) | 2.5978 (1.9799 to 3.1996) | 0.4650 (0.2363 to 0.6938) | 0.9162 (0.7915 to 0.9997) | 0.0341 (0.0220 to 0.0456) | -0.0384 (-0.0563 to -0.0227) | -0.0265 (-0.0375 to -0.0170) | -0.0143 (-0.0224 to -0.0063) |
| msmarco | multi-positive | 614 | L2S (in-sample) | 0.0954 (0.0363 to 0.1700) | 3.6502 (2.8281 to 4.4832) | 0.2290 (0.0627 to 0.4300) | 0.7495 (0.5818 to 0.9159) | -0.1339 (-0.1490 to -0.1201) | -0.1749 (-0.1889 to -0.1630) | -0.1540 (-0.1659 to -0.1437) | -0.1286 (-0.1421 to -0.1157) |
| msmarco | multi-positive | 614 | pair-optimal, target only (secondary) | 0.4857 (0.3300 to 0.6386) | 1.1864 (0.7158 to 1.7771) | 0.7918 (0.5839 to 0.9582) | 0.9584 (0.8752 to 1.0000) | 0.0596 (0.0458 to 0.0726) | n/a | n/a | n/a |
| msmarco | multi-positive | 716 | prefix-k | 0.0678 (0.0283 to 0.1184) | 3.8877 (3.0907 to 4.6532) | 0.2707 (0.1043 to 0.4583) | 0.6247 (0.4167 to 0.8326) | -0.1840 (-0.2070 to -0.1638) | -0.2214 (-0.2417 to -0.2023) | -0.1973 (-0.2209 to -0.1777) | -0.1684 (-0.1904 to -0.1502) |
| msmarco | multi-positive | 716 | all-positive oracle (privileged) | 0.1766 (0.0967 to 0.2640) | 2.6845 (2.0673 to 3.3035) | 0.4649 (0.2363 to 0.6938) | 0.9162 (0.7915 to 0.9997) | 0.0403 (0.0292 to 0.0508) | -0.0270 (-0.0442 to -0.0121) | -0.0159 (-0.0268 to -0.0072) | -0.0054 (-0.0132 to 0.0022) |
| msmarco | multi-positive | 716 | L2S (in-sample) | 0.0978 (0.0388 to 0.1712) | 3.6495 (2.8225 to 4.4917) | 0.2707 (0.1043 to 0.4581) | 0.7495 (0.5818 to 0.9159) | -0.0960 (-0.1082 to -0.0845) | -0.1273 (-0.1383 to -0.1172) | -0.1105 (-0.1202 to -0.1021) | -0.0911 (-0.1022 to -0.0814) |
| msmarco | multi-positive | 716 | pair-optimal, target only (secondary) | 0.4235 (0.2688 to 0.5864) | 1.3898 (0.8586 to 2.0283) | 0.7502 (0.5008 to 0.9580) | 0.9584 (0.8752 to 1.0000) | 0.0642 (0.0512 to 0.0763) | n/a | n/a | n/a |
| msmarco | multi-positive | 819 | prefix-k | 0.0683 (0.0286 to 0.1178) | 3.8303 (3.0436 to 4.5956) | 0.2707 (0.1043 to 0.4583) | 0.6249 (0.3754 to 0.8331) | -0.1178 (-0.1330 to -0.1034) | -0.1448 (-0.1568 to -0.1331) | -0.1271 (-0.1409 to -0.1160) | -0.1084 (-0.1218 to -0.0977) |
| msmarco | multi-positive | 819 | all-positive oracle (privileged) | 0.1651 (0.0869 to 0.2496) | 2.7885 (2.1562 to 3.4271) | 0.3817 (0.1737 to 0.5966) | 0.9162 (0.7915 to 0.9997) | 0.0404 (0.0301 to 0.0502) | -0.0215 (-0.0378 to -0.0072) | -0.0122 (-0.0220 to -0.0042) | -0.0018 (-0.0090 to 0.0050) |
| msmarco | multi-positive | 819 | L2S (in-sample) | 0.0808 (0.0348 to 0.1344) | 3.7094 (2.8811 to 4.5420) | 0.2707 (0.1043 to 0.4581) | 0.7081 (0.4997 to 0.9159) | -0.0600 (-0.0697 to -0.0509) | -0.0824 (-0.0912 to -0.0749) | -0.0699 (-0.0777 to -0.0629) | -0.0582 (-0.0664 to -0.0513) |
| msmarco | multi-positive | 819 | pair-optimal, target only (secondary) | 0.3423 (0.2120 to 0.4823) | 1.6859 (1.1504 to 2.3180) | 0.7084 (0.4998 to 0.9162) | 0.9584 (0.8752 to 1.0000) | 0.0624 (0.0503 to 0.0736) | n/a | n/a | n/a |
| msmarco | multi-positive | 921 | prefix-k | 0.0757 (0.0310 to 0.1288) | 3.7870 (2.9643 to 4.5938) | 0.2707 (0.1043 to 0.4583) | 0.6249 (0.3754 to 0.8331) | -0.0585 (-0.0702 to -0.0477) | -0.0760 (-0.0844 to -0.0685) | -0.0648 (-0.0745 to -0.0564) | -0.0544 (-0.0630 to -0.0473) |
| msmarco | multi-positive | 921 | all-positive oracle (privileged) | 0.1260 (0.0632 to 0.1981) | 3.0737 (2.3779 to 3.7472) | 0.3401 (0.1321 to 0.5620) | 0.8746 (0.7498 to 0.9991) | 0.0339 (0.0258 to 0.0416) | -0.0142 (-0.0271 to -0.0029) | -0.0081 (-0.0170 to -0.0008) | 0.0003 (-0.0058 to 0.0062) |
| msmarco | multi-positive | 921 | L2S (in-sample) | 0.0818 (0.0350 to 0.1347) | 3.7593 (2.9311 to 4.5726) | 0.2707 (0.1043 to 0.4583) | 0.6665 (0.4169 to 0.8747) | -0.0306 (-0.0371 to -0.0249) | -0.0426 (-0.0474 to -0.0383) | -0.0360 (-0.0402 to -0.0323) | -0.0293 (-0.0347 to -0.0242) |
| msmarco | multi-positive | 921 | pair-optimal, target only (secondary) | 0.2538 (0.1331 to 0.3944) | 2.2372 (1.5662 to 2.9806) | 0.5210 (0.2713 to 0.7701) | 0.9582 (0.8749 to 1.0000) | 0.0506 (0.0413 to 0.0593) | n/a | n/a | n/a |

## Agreement with the oracle mask

| environment | cohort | k | arm | Jaccard with oracle mask | score regret to ceiling / mass |
|---|---|---|---|---|---|
| msmarco | single-positive | 102 | prefix-k | 0.0824 (0.0756 to 0.0892) | 0.3448 (0.3349 to 0.3544) |
| msmarco | single-positive | 102 | L2S (in-sample) | 0.2327 (0.2196 to 0.2458) | 0.2029 (0.1937 to 0.2122) |
| msmarco | single-positive | 204 | prefix-k | 0.1074 (0.1001 to 0.1140) | 0.4815 (0.4704 to 0.4930) |
| msmarco | single-positive | 204 | L2S (in-sample) | 0.2725 (0.2568 to 0.2871) | 0.2726 (0.2583 to 0.2881) |
| msmarco | single-positive | 307 | prefix-k | 0.1629 (0.1563 to 0.1694) | 0.5213 (0.5067 to 0.5360) |
| msmarco | single-positive | 307 | L2S (in-sample) | 0.3131 (0.2969 to 0.3281) | 0.3010 (0.2833 to 0.3193) |
| msmarco | single-positive | 409 | prefix-k | 0.2408 (0.2340 to 0.2478) | 0.4912 (0.4747 to 0.5062) |
| msmarco | single-positive | 409 | L2S (in-sample) | 0.3636 (0.3542 to 0.3726) | 0.2984 (0.2818 to 0.3156) |
| msmarco | single-positive | 512 | prefix-k | 0.3311 (0.3241 to 0.3383) | 0.4471 (0.4322 to 0.4620) |
| msmarco | single-positive | 512 | L2S (in-sample) | 0.4254 (0.4178 to 0.4331) | 0.2720 (0.2586 to 0.2859) |
| msmarco | single-positive | 614 | prefix-k | 0.4290 (0.4223 to 0.4357) | 0.3812 (0.3690 to 0.3942) |
| msmarco | single-positive | 614 | L2S (in-sample) | 0.4902 (0.4818 to 0.4980) | 0.2351 (0.2213 to 0.2502) |
| msmarco | single-positive | 716 | prefix-k | 0.5393 (0.5345 to 0.5442) | 0.3092 (0.2936 to 0.3246) |
| msmarco | single-positive | 716 | L2S (in-sample) | 0.5742 (0.5647 to 0.5827) | 0.1884 (0.1737 to 0.2045) |
| msmarco | single-positive | 819 | prefix-k | 0.6706 (0.6666 to 0.6750) | 0.2220 (0.2056 to 0.2379) |
| msmarco | single-positive | 819 | L2S (in-sample) | 0.6818 (0.6756 to 0.6876) | 0.1421 (0.1281 to 0.1571) |
| msmarco | single-positive | 921 | prefix-k | 0.8196 (0.8168 to 0.8225) | 0.1290 (0.1157 to 0.1424) |
| msmarco | single-positive | 921 | L2S (in-sample) | 0.8226 (0.8191 to 0.8263) | 0.0893 (0.0778 to 0.1015) |
| msmarco | single-positive | 1024 | prefix-k | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| msmarco | single-positive | 1024 | L2S (in-sample) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| msmarco | multi-positive | 102 | prefix-k | 0.0905 (0.0768 to 0.1038) | 0.3150 (0.3033 to 0.3276) |
| msmarco | multi-positive | 102 | L2S (in-sample) | 0.2261 (0.2086 to 0.2448) | 0.1918 (0.1836 to 0.2009) |
| msmarco | multi-positive | 102 | held-out transfer (1-1) | 0.6755 (0.6352 to 0.7240) | n/a |
| msmarco | multi-positive | 204 | prefix-k | 0.1094 (0.1009 to 0.1179) | 0.4441 (0.4299 to 0.4603) |
| msmarco | multi-positive | 204 | L2S (in-sample) | 0.2622 (0.2473 to 0.2782) | 0.2690 (0.2538 to 0.2845) |
| msmarco | multi-positive | 204 | held-out transfer (1-1) | 0.7305 (0.6907 to 0.7750) | n/a |
| msmarco | multi-positive | 307 | prefix-k | 0.1622 (0.1534 to 0.1712) | 0.4876 (0.4678 to 0.5083) |
| msmarco | multi-positive | 307 | L2S (in-sample) | 0.3120 (0.3028 to 0.3210) | 0.2859 (0.2735 to 0.2980) |
| msmarco | multi-positive | 307 | held-out transfer (1-1) | 0.7463 (0.7155 to 0.7820) | n/a |
| msmarco | multi-positive | 409 | prefix-k | 0.2487 (0.2398 to 0.2576) | 0.4575 (0.4337 to 0.4813) |
| msmarco | multi-positive | 409 | L2S (in-sample) | 0.3560 (0.3441 to 0.3670) | 0.2799 (0.2649 to 0.2945) |
| msmarco | multi-positive | 409 | held-out transfer (1-1) | 0.7668 (0.7380 to 0.7998) | n/a |
| msmarco | multi-positive | 512 | prefix-k | 0.3304 (0.3209 to 0.3404) | 0.4181 (0.3970 to 0.4388) |
| msmarco | multi-positive | 512 | L2S (in-sample) | 0.4175 (0.4057 to 0.4290) | 0.2543 (0.2368 to 0.2720) |
| msmarco | multi-positive | 512 | held-out transfer (1-1) | 0.7863 (0.7599 to 0.8167) | n/a |
| msmarco | multi-positive | 614 | prefix-k | 0.4275 (0.4193 to 0.4365) | 0.3590 (0.3390 to 0.3790) |
| msmarco | multi-positive | 614 | L2S (in-sample) | 0.4936 (0.4836 to 0.5036) | 0.2151 (0.2009 to 0.2310) |
| msmarco | multi-positive | 614 | held-out transfer (1-1) | 0.8031 (0.7788 to 0.8307) | n/a |
| msmarco | multi-positive | 716 | prefix-k | 0.5414 (0.5332 to 0.5505) | 0.2864 (0.2675 to 0.3049) |
| msmarco | multi-positive | 716 | L2S (in-sample) | 0.5765 (0.5703 to 0.5826) | 0.1746 (0.1621 to 0.1885) |
| msmarco | multi-positive | 716 | held-out transfer (1-1) | 0.8141 (0.7892 to 0.8421) | n/a |
| msmarco | multi-positive | 819 | prefix-k | 0.6694 (0.6633 to 0.6755) | 0.2023 (0.1881 to 0.2172) |
| msmarco | multi-positive | 819 | L2S (in-sample) | 0.6839 (0.6800 to 0.6877) | 0.1288 (0.1175 to 0.1416) |
| msmarco | multi-positive | 819 | held-out transfer (1-1) | 0.8516 (0.8320 to 0.8748) | n/a |
| msmarco | multi-positive | 921 | prefix-k | 0.8197 (0.8158 to 0.8235) | 0.1182 (0.1057 to 0.1319) |
| msmarco | multi-positive | 921 | L2S (in-sample) | 0.8229 (0.8196 to 0.8260) | 0.0829 (0.0744 to 0.0913) |
| msmarco | multi-positive | 921 | held-out transfer (1-1) | 0.9156 (0.9042 to 0.9291) | n/a |
| msmarco | multi-positive | 1024 | prefix-k | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| msmarco | multi-positive | 1024 | L2S (in-sample) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| msmarco | multi-positive | 1024 | held-out transfer (1-1) | 1.0000 (1.0000 to 1.0000) | n/a |

## Relationship among a query's positives (multi-positive)

`pair_jaccard` is the mean overlap of pair-optimal masks. `sign_consensus` is the share of the oracle mask's coordinates on which all positives agree in sign, `positive_consensus` the share on which all are positive. `oracle_regret` is the regret of the all-positive mask against each positive's own optimum, `held_regret` of a mask built from a disjoint positive, each over mass.

| environment | k | queries | pair_jaccard | sign_consensus | positive_consensus | oracle_regret | held_regret | held_privileged_regret | held_oracle_jaccard |
|---|---|---|---|---|---|---|---|---|---|
| msmarco | 102 | 16 | 0.4659 (0.4077 to 0.5368) | 0.9851 (0.9729 to 0.9957) | 0.9851 (0.9729 to 0.9957) | 0.0238 (0.0178 to 0.0292) | 0.0823 (0.0627 to 0.1024) | 0.0234 (0.0177 to 0.0287) | 0.6755 (0.6352 to 0.7240) |
| msmarco | 204 | 16 | 0.5477 (0.4920 to 0.6133) | 0.9751 (0.9588 to 0.9894) | 0.9751 (0.9588 to 0.9894) | 0.0312 (0.0229 to 0.0394) | 0.1069 (0.0818 to 0.1325) | 0.0314 (0.0230 to 0.0397) | 0.7305 (0.6907 to 0.7750) |
| msmarco | 307 | 16 | 0.5691 (0.5268 to 0.6192) | 0.9463 (0.9173 to 0.9700) | 0.9463 (0.9173 to 0.9700) | 0.0352 (0.0260 to 0.0440) | 0.1208 (0.0928 to 0.1482) | 0.0355 (0.0260 to 0.0447) | 0.7463 (0.7155 to 0.7820) |
| msmarco | 409 | 16 | 0.6010 (0.5631 to 0.6468) | 0.9192 (0.8773 to 0.9523) | 0.9192 (0.8773 to 0.9523) | 0.0365 (0.0272 to 0.0456) | 0.1240 (0.0960 to 0.1514) | 0.0370 (0.0272 to 0.0471) | 0.7668 (0.7380 to 0.7998) |
| msmarco | 512 | 16 | 0.6304 (0.5948 to 0.6741) | 0.8822 (0.8320 to 0.9259) | 0.8822 (0.8320 to 0.9259) | 0.0354 (0.0261 to 0.0445) | 0.1215 (0.0936 to 0.1481) | 0.0359 (0.0262 to 0.0458) | 0.7863 (0.7599 to 0.8167) |
| msmarco | 614 | 16 | 0.6630 (0.6297 to 0.7027) | 0.8355 (0.7794 to 0.8883) | 0.8355 (0.7794 to 0.8883) | 0.0332 (0.0241 to 0.0418) | 0.1155 (0.0887 to 0.1414) | 0.0335 (0.0241 to 0.0426) | 0.8031 (0.7788 to 0.8307) |
| msmarco | 716 | 16 | 0.6832 (0.6504 to 0.7236) | 0.7810 (0.7213 to 0.8382) | 0.7810 (0.7213 to 0.8382) | 0.0310 (0.0223 to 0.0392) | 0.1091 (0.0830 to 0.1345) | 0.0312 (0.0223 to 0.0398) | 0.8141 (0.7892 to 0.8421) |
| msmarco | 819 | 16 | 0.7532 (0.7294 to 0.7846) | 0.7219 (0.6611 to 0.7810) | 0.7027 (0.6454 to 0.7585) | 0.0285 (0.0206 to 0.0360) | 0.0998 (0.0757 to 0.1227) | 0.0289 (0.0206 to 0.0370) | 0.8516 (0.8320 to 0.8748) |
| msmarco | 921 | 16 | 0.8607 (0.8460 to 0.8792) | 0.6892 (0.6253 to 0.7512) | 0.6249 (0.5739 to 0.6745) | 0.0218 (0.0160 to 0.0274) | 0.0753 (0.0574 to 0.0927) | 0.0222 (0.0160 to 0.0281) | 0.9156 (0.9042 to 0.9291) |
| msmarco | 1024 | 16 | 1.0000 (1.0000 to 1.0000) | 0.6893 (0.6243 to 0.7534) | 0.5620 (0.5162 to 0.6067) | 0.0000 (0.0000 to 0.0000) | 0.0000 (0.0000 to 0.0000) | 0.0000 (0.0000 to 0.0000) | 1.0000 (1.0000 to 1.0000) |

## Checks and cost

| environment | check | value |
|---|---|---|
| msmarco | boundary_note | "L2S ties at a mask boundary go to the lower dimension" |
| msmarco | documents_excluded_by_id | 0 |
| msmarco | float32_evaluation_gap | {"largest_ndcg10_gap": 0.0, "query_budget_pairs_above_1e-6": 0} |
| msmarco | ids_colliding_with_a_document | 32 |
| msmarco | k1024_list_mismatches | 0 |
| msmarco | k1024_rank_mismatches | 0 |
| msmarco | l2s_boundary_ties | 0 |
| msmarco | masks | 2222 |
| msmarco | masks_with_same_id_document | 2222 |
| msmarco | oracle_ceiling_violations | 0 |
| msmarco | ranks_out_of_bounds | 0 |
| msmarco | same_id_list_inconsistencies | 0 |
| msmarco | single_positive_masks_identical_to_oracle | 16 |
| msmarco | single_positive_queries | 16 |
| msmarco | stored_run_parity | {"compared": 1411, "mismatched": 0, "run": "20261109T163931Z-9e9f5a0a"} |
| msmarco | tied_tracked_documents | 0 |
| msmarco | tracked_ranks_charged_by_same_id_document | 0 |
| msmarco | total seconds | 74.6 |
| msmarco | device | cuda:0 |
