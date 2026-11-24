# P3: exact positive masks beside Learning-to-Select

Privileged, paired, in-domain diagnostic on training queries and training judgments. L2S numbers are **in-sample**: L2S was fitted on these queries. Nothing here measures performance on unseen queries.

- Run `20261109T235856Z-5d598106` (commit `cf2d71a2828b`, dirty `False`, device `cpu`), environments ['scifact'], query bound None.
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
| scifact | checkpoint fingerprint is the dataset's | True |
| scifact | dataset is the F2 dataset read here | True |
| scifact | embeddings are the F2 embeddings read here | True |
| scifact | embeddings fingerprint is the dataset's | True |
| scifact | environment matches | True |
| scifact | model revision matches | True |
| scifact | negatives are the latest F2 negatives | True |
| scifact | negatives fingerprint is the dataset's | True |
| scifact | official recipe | True |
| scifact | one published checkpoint | True |
| scifact | payload hash matches | True |
| scifact | per-domain, 1024 wide | True |
| scifact | published F2 dataset | True |
| scifact | split ids are disjoint | True |
| scifact | trained by train-selector | True |
| scifact | trained from a clean tree | True |

## Coverage

| environment | training queries | with a positive | evaluated | single-positive | multi-positive | positives | documents | capped | weights all 1 |
|---|---|---|---|---|---|---|---|---|---|
| scifact | 728 | 728 | 728 | 659 | 69 | 822 | 5183 | False | True |

## Official metrics, full training judgments

| environment | cohort | k | ratio | arm | ndcg@10 | mrr@10 | mrr@100 | recall@100 |
|---|---|---|---|---|---|---|---|---|
| scifact | single-positive | 102 | 10% | prefix-k | 0.5646 (0.5307 to 0.5977) | 0.5234 (0.4885 to 0.5579) | 0.5315 (0.4977 to 0.5655) | 0.8832 (0.8589 to 0.9059) |
| scifact | single-positive | 102 | 10% | all-positive oracle (privileged) | 0.9916 (0.9868 to 0.9957) | 0.9887 (0.9823 to 0.9943) | 0.9887 (0.9823 to 0.9943) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 102 | 10% | L2S (in-sample) | 0.9729 (0.9638 to 0.9807) | 0.9644 (0.9527 to 0.9745) | 0.9645 (0.9528 to 0.9746) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 204 | 20% | prefix-k | 0.6355 (0.6028 to 0.6685) | 0.5976 (0.5632 to 0.6319) | 0.6044 (0.5712 to 0.6379) | 0.9211 (0.8998 to 0.9408) |
| scifact | single-positive | 204 | 20% | all-positive oracle (privileged) | 0.9927 (0.9882 to 0.9966) | 0.9903 (0.9842 to 0.9954) | 0.9903 (0.9842 to 0.9954) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 204 | 20% | L2S (in-sample) | 0.9789 (0.9710 to 0.9861) | 0.9719 (0.9614 to 0.9814) | 0.9719 (0.9614 to 0.9814) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 307 | 30% | prefix-k | 0.6697 (0.6386 to 0.7018) | 0.6324 (0.5993 to 0.6657) | 0.6386 (0.6062 to 0.6713) | 0.9347 (0.9150 to 0.9530) |
| scifact | single-positive | 307 | 30% | all-positive oracle (privileged) | 0.9893 (0.9842 to 0.9940) | 0.9856 (0.9788 to 0.9920) | 0.9856 (0.9788 to 0.9920) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 307 | 30% | L2S (in-sample) | 0.9723 (0.9634 to 0.9802) | 0.9631 (0.9512 to 0.9736) | 0.9631 (0.9512 to 0.9736) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 409 | 40% | prefix-k | 0.6935 (0.6625 to 0.7244) | 0.6556 (0.6225 to 0.6886) | 0.6613 (0.6291 to 0.6937) | 0.9545 (0.9393 to 0.9697) |
| scifact | single-positive | 409 | 40% | all-positive oracle (privileged) | 0.9868 (0.9812 to 0.9922) | 0.9823 (0.9747 to 0.9894) | 0.9823 (0.9747 to 0.9894) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 409 | 40% | L2S (in-sample) | 0.9663 (0.9563 to 0.9749) | 0.9551 (0.9420 to 0.9667) | 0.9551 (0.9420 to 0.9667) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 512 | 50% | prefix-k | 0.7062 (0.6746 to 0.7365) | 0.6697 (0.6369 to 0.7021) | 0.6754 (0.6431 to 0.7073) | 0.9575 (0.9423 to 0.9727) |
| scifact | single-positive | 512 | 50% | all-positive oracle (privileged) | 0.9850 (0.9788 to 0.9908) | 0.9799 (0.9716 to 0.9877) | 0.9799 (0.9716 to 0.9877) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 512 | 50% | L2S (in-sample) | 0.9518 (0.9399 to 0.9627) | 0.9360 (0.9205 to 0.9504) | 0.9360 (0.9205 to 0.9504) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 614 | 60% | prefix-k | 0.7220 (0.6914 to 0.7518) | 0.6843 (0.6521 to 0.7165) | 0.6888 (0.6574 to 0.7206) | 0.9590 (0.9439 to 0.9742) |
| scifact | single-positive | 614 | 60% | all-positive oracle (privileged) | 0.9789 (0.9716 to 0.9860) | 0.9717 (0.9618 to 0.9812) | 0.9717 (0.9618 to 0.9812) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 614 | 60% | L2S (in-sample) | 0.9329 (0.9188 to 0.9467) | 0.9138 (0.8958 to 0.9308) | 0.9144 (0.8964 to 0.9312) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 716 | 70% | prefix-k | 0.7286 (0.6990 to 0.7576) | 0.6917 (0.6604 to 0.7228) | 0.6961 (0.6654 to 0.7268) | 0.9636 (0.9484 to 0.9772) |
| scifact | single-positive | 716 | 70% | all-positive oracle (privileged) | 0.9715 (0.9629 to 0.9796) | 0.9617 (0.9503 to 0.9726) | 0.9617 (0.9503 to 0.9726) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 716 | 70% | L2S (in-sample) | 0.9035 (0.8856 to 0.9200) | 0.8798 (0.8584 to 0.9001) | 0.8814 (0.8603 to 0.9014) | 0.9985 (0.9954 to 1.0000) |
| scifact | single-positive | 819 | 80% | prefix-k | 0.7287 (0.6988 to 0.7577) | 0.6925 (0.6610 to 0.7234) | 0.6971 (0.6662 to 0.7276) | 0.9621 (0.9469 to 0.9757) |
| scifact | single-positive | 819 | 80% | all-positive oracle (privileged) | 0.9608 (0.9507 to 0.9704) | 0.9487 (0.9357 to 0.9610) | 0.9490 (0.9359 to 0.9612) | 1.0000 (1.0000 to 1.0000) |
| scifact | single-positive | 819 | 80% | L2S (in-sample) | 0.8719 (0.8512 to 0.8918) | 0.8430 (0.8180 to 0.8664) | 0.8449 (0.8204 to 0.8679) | 0.9970 (0.9924 to 1.0000) |
| scifact | single-positive | 921 | 90% | prefix-k | 0.7380 (0.7089 to 0.7667) | 0.7024 (0.6712 to 0.7328) | 0.7066 (0.6758 to 0.7366) | 0.9575 (0.9423 to 0.9727) |
| scifact | single-positive | 921 | 90% | all-positive oracle (privileged) | 0.9121 (0.8961 to 0.9271) | 0.8879 (0.8680 to 0.9065) | 0.8885 (0.8683 to 0.9071) | 0.9985 (0.9954 to 1.0000) |
| scifact | single-positive | 921 | 90% | L2S (in-sample) | 0.8184 (0.7935 to 0.8430) | 0.7866 (0.7583 to 0.8144) | 0.7905 (0.7629 to 0.8180) | 0.9939 (0.9879 to 0.9985) |
| scifact | single-positive | 1024 | 100% | full query | 0.7399 (0.7115 to 0.7692) | 0.7061 (0.6751 to 0.7378) | 0.7104 (0.6797 to 0.7417) | 0.9530 (0.9363 to 0.9697) |
| scifact | single-positive | 1024 | 100% | prefix-k | 0.7399 (0.7115 to 0.7692) | 0.7061 (0.6751 to 0.7378) | 0.7104 (0.6797 to 0.7417) | 0.9530 (0.9363 to 0.9697) |
| scifact | single-positive | 1024 | 100% | all-positive oracle (privileged) | 0.7399 (0.7115 to 0.7692) | 0.7061 (0.6751 to 0.7378) | 0.7104 (0.6797 to 0.7417) | 0.9530 (0.9363 to 0.9697) |
| scifact | single-positive | 1024 | 100% | L2S (in-sample) | 0.7399 (0.7115 to 0.7692) | 0.7061 (0.6751 to 0.7378) | 0.7104 (0.6797 to 0.7417) | 0.9530 (0.9363 to 0.9697) |
| scifact | multi-positive | 102 | 10% | prefix-k | 0.5324 (0.4612 to 0.6081) | 0.5788 (0.4892 to 0.6687) | 0.5839 (0.4952 to 0.6729) | 0.9251 (0.8671 to 0.9758) |
| scifact | multi-positive | 102 | 10% | all-positive oracle (privileged) | 0.9410 (0.9120 to 0.9652) | 0.9498 (0.9068 to 0.9855) | 0.9498 (0.9068 to 0.9855) | 1.0000 (1.0000 to 1.0000) |
| scifact | multi-positive | 102 | 10% | held-out transfer (1-1) | 0.8870 (0.8603 to 0.9114) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.9875 (0.9734 to 0.9968) |
| scifact | multi-positive | 102 | 10% | L2S (in-sample) | 0.9026 (0.8657 to 0.9332) | 0.9287 (0.8816 to 0.9710) | 0.9287 (0.8816 to 0.9710) | 1.0000 (1.0000 to 1.0000) |
| scifact | multi-positive | 204 | 20% | prefix-k | 0.6125 (0.5410 to 0.6822) | 0.6603 (0.5750 to 0.7457) | 0.6626 (0.5779 to 0.7467) | 0.9565 (0.9058 to 0.9928) |
| scifact | multi-positive | 204 | 20% | all-positive oracle (privileged) | 0.9390 (0.9110 to 0.9621) | 0.9589 (0.9203 to 0.9903) | 0.9589 (0.9203 to 0.9903) | 1.0000 (1.0000 to 1.0000) |
| scifact | multi-positive | 204 | 20% | held-out transfer (1-1) | 0.8927 (0.8676 to 0.9163) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.9875 (0.9734 to 0.9968) |
| scifact | multi-positive | 204 | 20% | L2S (in-sample) | 0.9103 (0.8732 to 0.9414) | 0.9461 (0.9002 to 0.9831) | 0.9461 (0.9002 to 0.9831) | 1.0000 (1.0000 to 1.0000) |
| scifact | multi-positive | 307 | 30% | prefix-k | 0.6268 (0.5571 to 0.6975) | 0.6734 (0.5879 to 0.7590) | 0.6755 (0.5906 to 0.7606) | 0.9638 (0.9130 to 1.0000) |
| scifact | multi-positive | 307 | 30% | all-positive oracle (privileged) | 0.9242 (0.8941 to 0.9504) | 0.9541 (0.9130 to 0.9903) | 0.9541 (0.9130 to 0.9903) | 1.0000 (1.0000 to 1.0000) |
| scifact | multi-positive | 307 | 30% | held-out transfer (1-1) | 0.8913 (0.8665 to 0.9155) | 0.9949 (0.9870 to 1.0000) | 0.9949 (0.9870 to 1.0000) | 0.9891 (0.9746 to 1.0000) |
| scifact | multi-positive | 307 | 30% | L2S (in-sample) | 0.8940 (0.8538 to 0.9281) | 0.9401 (0.8942 to 0.9783) | 0.9401 (0.8942 to 0.9783) | 0.9928 (0.9783 to 1.0000) |
| scifact | multi-positive | 409 | 40% | prefix-k | 0.6412 (0.5691 to 0.7126) | 0.6977 (0.6091 to 0.7850) | 0.7010 (0.6131 to 0.7875) | 0.9855 (0.9638 to 1.0000) |
| scifact | multi-positive | 409 | 40% | all-positive oracle (privileged) | 0.9134 (0.8817 to 0.9414) | 0.9517 (0.9106 to 0.9855) | 0.9517 (0.9106 to 0.9855) | 1.0000 (1.0000 to 1.0000) |
| scifact | multi-positive | 409 | 40% | held-out transfer (1-1) | 0.8873 (0.8598 to 0.9128) | 0.9867 (0.9737 to 0.9967) | 0.9867 (0.9737 to 0.9967) | 0.9891 (0.9746 to 1.0000) |
| scifact | multi-positive | 409 | 40% | L2S (in-sample) | 0.8654 (0.8194 to 0.9048) | 0.9163 (0.8632 to 0.9622) | 0.9163 (0.8632 to 0.9622) | 0.9928 (0.9783 to 1.0000) |
| scifact | multi-positive | 512 | 50% | prefix-k | 0.6384 (0.5657 to 0.7105) | 0.6888 (0.6016 to 0.7739) | 0.6908 (0.6048 to 0.7753) | 0.9855 (0.9638 to 1.0000) |
| scifact | multi-positive | 512 | 50% | all-positive oracle (privileged) | 0.9052 (0.8727 to 0.9344) | 0.9493 (0.9082 to 0.9831) | 0.9493 (0.9082 to 0.9831) | 1.0000 (1.0000 to 1.0000) |
| scifact | multi-positive | 512 | 50% | held-out transfer (1-1) | 0.8811 (0.8491 to 0.9096) | 0.9710 (0.9466 to 0.9916) | 0.9710 (0.9466 to 0.9916) | 0.9891 (0.9746 to 1.0000) |
| scifact | multi-positive | 512 | 50% | L2S (in-sample) | 0.8480 (0.7980 to 0.8940) | 0.9023 (0.8446 to 0.9541) | 0.9023 (0.8446 to 0.9541) | 0.9928 (0.9783 to 1.0000) |
| scifact | multi-positive | 614 | 60% | prefix-k | 0.6566 (0.5849 to 0.7274) | 0.6787 (0.5932 to 0.7638) | 0.6810 (0.5969 to 0.7651) | 0.9807 (0.9565 to 1.0000) |
| scifact | multi-positive | 614 | 60% | all-positive oracle (privileged) | 0.8908 (0.8532 to 0.9249) | 0.9360 (0.8901 to 0.9747) | 0.9360 (0.8901 to 0.9747) | 1.0000 (1.0000 to 1.0000) |
| scifact | multi-positive | 614 | 60% | held-out transfer (1-1) | 0.8728 (0.8393 to 0.9031) | 0.9528 (0.9223 to 0.9795) | 0.9528 (0.9223 to 0.9795) | 0.9891 (0.9746 to 1.0000) |
| scifact | multi-positive | 614 | 60% | L2S (in-sample) | 0.8242 (0.7675 to 0.8748) | 0.8719 (0.8072 to 0.9306) | 0.8719 (0.8072 to 0.9306) | 0.9928 (0.9783 to 1.0000) |
| scifact | multi-positive | 716 | 70% | prefix-k | 0.6701 (0.5961 to 0.7414) | 0.6970 (0.6097 to 0.7830) | 0.7009 (0.6157 to 0.7855) | 0.9855 (0.9638 to 1.0000) |
| scifact | multi-positive | 716 | 70% | all-positive oracle (privileged) | 0.8771 (0.8366 to 0.9130) | 0.9251 (0.8744 to 0.9686) | 0.9251 (0.8744 to 0.9686) | 1.0000 (1.0000 to 1.0000) |
| scifact | multi-positive | 716 | 70% | held-out transfer (1-1) | 0.8629 (0.8275 to 0.8949) | 0.9456 (0.9123 to 0.9752) | 0.9456 (0.9123 to 0.9752) | 0.9891 (0.9746 to 1.0000) |
| scifact | multi-positive | 716 | 70% | L2S (in-sample) | 0.7999 (0.7383 to 0.8566) | 0.8388 (0.7633 to 0.9058) | 0.8401 (0.7660 to 0.9070) | 0.9928 (0.9783 to 1.0000) |
| scifact | multi-positive | 819 | 80% | prefix-k | 0.6658 (0.5918 to 0.7372) | 0.6986 (0.6091 to 0.7831) | 0.7042 (0.6186 to 0.7874) | 0.9807 (0.9565 to 1.0000) |
| scifact | multi-positive | 819 | 80% | all-positive oracle (privileged) | 0.8637 (0.8173 to 0.9033) | 0.9076 (0.8514 to 0.9565) | 0.9076 (0.8514 to 0.9565) | 1.0000 (1.0000 to 1.0000) |
| scifact | multi-positive | 819 | 80% | held-out transfer (1-1) | 0.8535 (0.8142 to 0.8899) | 0.9240 (0.8830 to 0.9607) | 0.9240 (0.8830 to 0.9607) | 0.9911 (0.9803 to 1.0000) |
| scifact | multi-positive | 819 | 80% | L2S (in-sample) | 0.7719 (0.7081 to 0.8309) | 0.8045 (0.7268 to 0.8770) | 0.8071 (0.7308 to 0.8787) | 0.9928 (0.9783 to 1.0000) |
| scifact | multi-positive | 921 | 90% | prefix-k | 0.6696 (0.5991 to 0.7385) | 0.6834 (0.5990 to 0.7633) | 0.6863 (0.6034 to 0.7656) | 0.9807 (0.9565 to 1.0000) |
| scifact | multi-positive | 921 | 90% | all-positive oracle (privileged) | 0.8210 (0.7682 to 0.8685) | 0.8505 (0.7804 to 0.9121) | 0.8505 (0.7804 to 0.9121) | 0.9928 (0.9783 to 1.0000) |
| scifact | multi-positive | 921 | 90% | held-out transfer (1-1) | 0.8124 (0.7595 to 0.8574) | 0.8612 (0.8039 to 0.9108) | 0.8627 (0.8066 to 0.9115) | 0.9948 (0.9859 to 1.0000) |
| scifact | multi-positive | 921 | 90% | L2S (in-sample) | 0.7382 (0.6715 to 0.7992) | 0.7531 (0.6720 to 0.8314) | 0.7562 (0.6770 to 0.8330) | 0.9928 (0.9783 to 1.0000) |
| scifact | multi-positive | 1024 | 100% | full query | 0.6786 (0.6069 to 0.7476) | 0.6960 (0.6101 to 0.7777) | 0.6992 (0.6152 to 0.7803) | 0.9807 (0.9565 to 1.0000) |
| scifact | multi-positive | 1024 | 100% | prefix-k | 0.6786 (0.6069 to 0.7476) | 0.6960 (0.6101 to 0.7777) | 0.6992 (0.6152 to 0.7803) | 0.9807 (0.9565 to 1.0000) |
| scifact | multi-positive | 1024 | 100% | all-positive oracle (privileged) | 0.6786 (0.6069 to 0.7476) | 0.6960 (0.6101 to 0.7777) | 0.6992 (0.6152 to 0.7803) | 0.9807 (0.9565 to 1.0000) |
| scifact | multi-positive | 1024 | 100% | held-out transfer (1-1) | 0.6786 (0.6069 to 0.7476) | 0.6960 (0.6101 to 0.7777) | 0.6992 (0.6152 to 0.7803) | 0.9807 (0.9565 to 1.0000) |
| scifact | multi-positive | 1024 | 100% | L2S (in-sample) | 0.6786 (0.6069 to 0.7476) | 0.6960 (0.6101 to 0.7777) | 0.6992 (0.6152 to 0.7803) | 0.9807 (0.9565 to 1.0000) |

## Paired differences, with queries better and worse

| environment | cohort | k | measure | comparison | queries | difference | 2.5% | 97.5% | share better | share worse |
|---|---|---|---|---|---|---|---|---|---|---|
| scifact | single-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0187 | 0.0114 | 0.0266 | 0.046 | 0.006 |
| scifact | single-positive | 102 | ndcg@10 | L2S (in-sample) minus full query | 659 | 0.2331 | 0.2054 | 0.2612 | 0.358 | 0.008 |
| scifact | single-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus full query | 659 | 0.2518 | 0.2237 | 0.2800 | 0.366 | 0.000 |
| scifact | single-positive | 102 | ndcg@10 | prefix-k minus full query | 659 | -0.1753 | -0.1997 | -0.1531 | 0.041 | 0.370 |
| scifact | single-positive | 102 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0244 | 0.0149 | 0.0348 | 0.046 | 0.006 |
| scifact | single-positive | 102 | mrr@10 | L2S (in-sample) minus full query | 659 | 0.2583 | 0.2286 | 0.2881 | 0.358 | 0.008 |
| scifact | single-positive | 102 | mrr@10 | all-positive oracle (privileged) minus full query | 659 | 0.2827 | 0.2520 | 0.3133 | 0.366 | 0.000 |
| scifact | single-positive | 102 | mrr@10 | prefix-k minus full query | 659 | -0.1827 | -0.2085 | -0.1580 | 0.041 | 0.370 |
| scifact | single-positive | 102 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0243 | 0.0149 | 0.0345 | 0.046 | 0.006 |
| scifact | single-positive | 102 | mrr@100 | L2S (in-sample) minus full query | 659 | 0.2541 | 0.2248 | 0.2833 | 0.360 | 0.008 |
| scifact | single-positive | 102 | mrr@100 | all-positive oracle (privileged) minus full query | 659 | 0.2783 | 0.2483 | 0.3085 | 0.366 | 0.000 |
| scifact | single-positive | 102 | mrr@100 | prefix-k minus full query | 659 | -0.1789 | -0.2036 | -0.1548 | 0.065 | 0.448 |
| scifact | single-positive | 102 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | single-positive | 102 | recall@100 | L2S (in-sample) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 102 | recall@100 | all-positive oracle (privileged) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 102 | recall@100 | prefix-k minus full query | 659 | -0.0698 | -0.0910 | -0.0486 | 0.008 | 0.077 |
| scifact | single-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0138 | 0.0076 | 0.0209 | 0.036 | 0.005 |
| scifact | single-positive | 204 | ndcg@10 | L2S (in-sample) minus full query | 659 | 0.2391 | 0.2114 | 0.2673 | 0.363 | 0.003 |
| scifact | single-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus full query | 659 | 0.2529 | 0.2249 | 0.2812 | 0.366 | 0.000 |
| scifact | single-positive | 204 | ndcg@10 | prefix-k minus full query | 659 | -0.1044 | -0.1243 | -0.0854 | 0.036 | 0.273 |
| scifact | single-positive | 204 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0183 | 0.0101 | 0.0277 | 0.036 | 0.005 |
| scifact | single-positive | 204 | mrr@10 | L2S (in-sample) minus full query | 659 | 0.2659 | 0.2359 | 0.2962 | 0.363 | 0.003 |
| scifact | single-positive | 204 | mrr@10 | all-positive oracle (privileged) minus full query | 659 | 0.2842 | 0.2530 | 0.3155 | 0.366 | 0.000 |
| scifact | single-positive | 204 | mrr@10 | prefix-k minus full query | 659 | -0.1085 | -0.1295 | -0.0881 | 0.036 | 0.273 |
| scifact | single-positive | 204 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0183 | 0.0101 | 0.0277 | 0.036 | 0.005 |
| scifact | single-positive | 204 | mrr@100 | L2S (in-sample) minus full query | 659 | 0.2615 | 0.2319 | 0.2912 | 0.363 | 0.003 |
| scifact | single-positive | 204 | mrr@100 | all-positive oracle (privileged) minus full query | 659 | 0.2798 | 0.2494 | 0.3100 | 0.366 | 0.000 |
| scifact | single-positive | 204 | mrr@100 | prefix-k minus full query | 659 | -0.1060 | -0.1265 | -0.0864 | 0.068 | 0.346 |
| scifact | single-positive | 204 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | single-positive | 204 | recall@100 | L2S (in-sample) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 204 | recall@100 | all-positive oracle (privileged) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 204 | recall@100 | prefix-k minus full query | 659 | -0.0319 | -0.0501 | -0.0137 | 0.012 | 0.044 |
| scifact | single-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0169 | 0.0107 | 0.0240 | 0.046 | 0.002 |
| scifact | single-positive | 307 | ndcg@10 | L2S (in-sample) minus full query | 659 | 0.2325 | 0.2053 | 0.2599 | 0.361 | 0.002 |
| scifact | single-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus full query | 659 | 0.2494 | 0.2214 | 0.2776 | 0.364 | 0.000 |
| scifact | single-positive | 307 | ndcg@10 | prefix-k minus full query | 659 | -0.0702 | -0.0864 | -0.0551 | 0.044 | 0.226 |
| scifact | single-positive | 307 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0225 | 0.0142 | 0.0319 | 0.046 | 0.002 |
| scifact | single-positive | 307 | mrr@10 | L2S (in-sample) minus full query | 659 | 0.2571 | 0.2274 | 0.2857 | 0.361 | 0.002 |
| scifact | single-positive | 307 | mrr@10 | all-positive oracle (privileged) minus full query | 659 | 0.2795 | 0.2491 | 0.3100 | 0.364 | 0.000 |
| scifact | single-positive | 307 | mrr@10 | prefix-k minus full query | 659 | -0.0737 | -0.0913 | -0.0567 | 0.044 | 0.226 |
| scifact | single-positive | 307 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0225 | 0.0142 | 0.0319 | 0.046 | 0.002 |
| scifact | single-positive | 307 | mrr@100 | L2S (in-sample) minus full query | 659 | 0.2527 | 0.2235 | 0.2807 | 0.361 | 0.002 |
| scifact | single-positive | 307 | mrr@100 | all-positive oracle (privileged) minus full query | 659 | 0.2752 | 0.2454 | 0.3050 | 0.364 | 0.000 |
| scifact | single-positive | 307 | mrr@100 | prefix-k minus full query | 659 | -0.0718 | -0.0892 | -0.0554 | 0.083 | 0.293 |
| scifact | single-positive | 307 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | single-positive | 307 | recall@100 | L2S (in-sample) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 307 | recall@100 | all-positive oracle (privileged) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 307 | recall@100 | prefix-k minus full query | 659 | -0.0182 | -0.0349 | -0.0030 | 0.012 | 0.030 |
| scifact | single-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0206 | 0.0141 | 0.0280 | 0.062 | 0.000 |
| scifact | single-positive | 409 | ndcg@10 | L2S (in-sample) minus full query | 659 | 0.2264 | 0.2005 | 0.2539 | 0.360 | 0.002 |
| scifact | single-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus full query | 659 | 0.2470 | 0.2188 | 0.2756 | 0.364 | 0.000 |
| scifact | single-positive | 409 | ndcg@10 | prefix-k minus full query | 659 | -0.0463 | -0.0588 | -0.0345 | 0.041 | 0.178 |
| scifact | single-positive | 409 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0272 | 0.0184 | 0.0370 | 0.062 | 0.000 |
| scifact | single-positive | 409 | mrr@10 | L2S (in-sample) minus full query | 659 | 0.2491 | 0.2211 | 0.2781 | 0.360 | 0.002 |
| scifact | single-positive | 409 | mrr@10 | all-positive oracle (privileged) minus full query | 659 | 0.2762 | 0.2459 | 0.3066 | 0.364 | 0.000 |
| scifact | single-positive | 409 | mrr@10 | prefix-k minus full query | 659 | -0.0505 | -0.0648 | -0.0368 | 0.041 | 0.178 |
| scifact | single-positive | 409 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0272 | 0.0184 | 0.0370 | 0.062 | 0.000 |
| scifact | single-positive | 409 | mrr@100 | L2S (in-sample) minus full query | 659 | 0.2447 | 0.2168 | 0.2731 | 0.360 | 0.002 |
| scifact | single-positive | 409 | mrr@100 | all-positive oracle (privileged) minus full query | 659 | 0.2719 | 0.2421 | 0.3017 | 0.364 | 0.000 |
| scifact | single-positive | 409 | mrr@100 | prefix-k minus full query | 659 | -0.0491 | -0.0632 | -0.0354 | 0.085 | 0.243 |
| scifact | single-positive | 409 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | single-positive | 409 | recall@100 | L2S (in-sample) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 409 | recall@100 | all-positive oracle (privileged) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 409 | recall@100 | prefix-k minus full query | 659 | 0.0015 | -0.0091 | 0.0121 | 0.011 | 0.009 |
| scifact | single-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0332 | 0.0243 | 0.0425 | 0.090 | 0.000 |
| scifact | single-positive | 512 | ndcg@10 | L2S (in-sample) minus full query | 659 | 0.2119 | 0.1873 | 0.2371 | 0.358 | 0.000 |
| scifact | single-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus full query | 659 | 0.2451 | 0.2173 | 0.2738 | 0.363 | 0.000 |
| scifact | single-positive | 512 | ndcg@10 | prefix-k minus full query | 659 | -0.0337 | -0.0444 | -0.0236 | 0.041 | 0.146 |
| scifact | single-positive | 512 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0438 | 0.0320 | 0.0562 | 0.090 | 0.000 |
| scifact | single-positive | 512 | mrr@10 | L2S (in-sample) minus full query | 659 | 0.2300 | 0.2038 | 0.2570 | 0.358 | 0.000 |
| scifact | single-positive | 512 | mrr@10 | all-positive oracle (privileged) minus full query | 659 | 0.2738 | 0.2439 | 0.3043 | 0.363 | 0.000 |
| scifact | single-positive | 512 | mrr@10 | prefix-k minus full query | 659 | -0.0363 | -0.0478 | -0.0250 | 0.041 | 0.146 |
| scifact | single-positive | 512 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0438 | 0.0320 | 0.0562 | 0.090 | 0.000 |
| scifact | single-positive | 512 | mrr@100 | L2S (in-sample) minus full query | 659 | 0.2256 | 0.1999 | 0.2520 | 0.358 | 0.000 |
| scifact | single-positive | 512 | mrr@100 | all-positive oracle (privileged) minus full query | 659 | 0.2694 | 0.2400 | 0.2994 | 0.363 | 0.000 |
| scifact | single-positive | 512 | mrr@100 | prefix-k minus full query | 659 | -0.0350 | -0.0461 | -0.0239 | 0.080 | 0.212 |
| scifact | single-positive | 512 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | single-positive | 512 | recall@100 | L2S (in-sample) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 512 | recall@100 | all-positive oracle (privileged) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 512 | recall@100 | prefix-k minus full query | 659 | 0.0046 | -0.0046 | 0.0137 | 0.009 | 0.005 |
| scifact | single-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0461 | 0.0351 | 0.0573 | 0.114 | 0.000 |
| scifact | single-positive | 614 | ndcg@10 | L2S (in-sample) minus full query | 659 | 0.1930 | 0.1700 | 0.2165 | 0.347 | 0.000 |
| scifact | single-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus full query | 659 | 0.2391 | 0.2117 | 0.2664 | 0.361 | 0.000 |
| scifact | single-positive | 614 | ndcg@10 | prefix-k minus full query | 659 | -0.0179 | -0.0260 | -0.0097 | 0.050 | 0.112 |
| scifact | single-positive | 614 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0579 | 0.0441 | 0.0718 | 0.114 | 0.000 |
| scifact | single-positive | 614 | mrr@10 | L2S (in-sample) minus full query | 659 | 0.2077 | 0.1832 | 0.2331 | 0.347 | 0.000 |
| scifact | single-positive | 614 | mrr@10 | all-positive oracle (privileged) minus full query | 659 | 0.2657 | 0.2358 | 0.2954 | 0.361 | 0.000 |
| scifact | single-positive | 614 | mrr@10 | prefix-k minus full query | 659 | -0.0218 | -0.0311 | -0.0121 | 0.050 | 0.112 |
| scifact | single-positive | 614 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0574 | 0.0438 | 0.0711 | 0.114 | 0.000 |
| scifact | single-positive | 614 | mrr@100 | L2S (in-sample) minus full query | 659 | 0.2040 | 0.1796 | 0.2290 | 0.357 | 0.000 |
| scifact | single-positive | 614 | mrr@100 | all-positive oracle (privileged) minus full query | 659 | 0.2613 | 0.2321 | 0.2907 | 0.361 | 0.000 |
| scifact | single-positive | 614 | mrr@100 | prefix-k minus full query | 659 | -0.0216 | -0.0308 | -0.0120 | 0.100 | 0.164 |
| scifact | single-positive | 614 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | single-positive | 614 | recall@100 | L2S (in-sample) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 614 | recall@100 | all-positive oracle (privileged) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 614 | recall@100 | prefix-k minus full query | 659 | 0.0061 | -0.0015 | 0.0137 | 0.008 | 0.002 |
| scifact | single-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0680 | 0.0549 | 0.0826 | 0.153 | 0.000 |
| scifact | single-positive | 716 | ndcg@10 | L2S (in-sample) minus full query | 659 | 0.1636 | 0.1435 | 0.1839 | 0.331 | 0.000 |
| scifact | single-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus full query | 659 | 0.2316 | 0.2051 | 0.2581 | 0.360 | 0.000 |
| scifact | single-positive | 716 | ndcg@10 | prefix-k minus full query | 659 | -0.0112 | -0.0189 | -0.0035 | 0.065 | 0.096 |
| scifact | single-positive | 716 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0819 | 0.0658 | 0.0994 | 0.153 | 0.000 |
| scifact | single-positive | 716 | mrr@10 | L2S (in-sample) minus full query | 659 | 0.1738 | 0.1518 | 0.1958 | 0.331 | 0.000 |
| scifact | single-positive | 716 | mrr@10 | all-positive oracle (privileged) minus full query | 659 | 0.2557 | 0.2266 | 0.2851 | 0.360 | 0.000 |
| scifact | single-positive | 716 | mrr@10 | prefix-k minus full query | 659 | -0.0144 | -0.0236 | -0.0052 | 0.065 | 0.096 |
| scifact | single-positive | 716 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0803 | 0.0645 | 0.0973 | 0.153 | 0.000 |
| scifact | single-positive | 716 | mrr@100 | L2S (in-sample) minus full query | 659 | 0.1710 | 0.1497 | 0.1930 | 0.354 | 0.000 |
| scifact | single-positive | 716 | mrr@100 | all-positive oracle (privileged) minus full query | 659 | 0.2513 | 0.2228 | 0.2806 | 0.360 | 0.000 |
| scifact | single-positive | 716 | mrr@100 | prefix-k minus full query | 659 | -0.0143 | -0.0235 | -0.0051 | 0.132 | 0.141 |
| scifact | single-positive | 716 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0015 | 0.0000 | 0.0046 | 0.002 | 0.000 |
| scifact | single-positive | 716 | recall@100 | L2S (in-sample) minus full query | 659 | 0.0455 | 0.0303 | 0.0622 | 0.046 | 0.000 |
| scifact | single-positive | 716 | recall@100 | all-positive oracle (privileged) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 716 | recall@100 | prefix-k minus full query | 659 | 0.0106 | 0.0015 | 0.0197 | 0.012 | 0.002 |
| scifact | single-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0889 | 0.0736 | 0.1046 | 0.196 | 0.000 |
| scifact | single-positive | 819 | ndcg@10 | L2S (in-sample) minus full query | 659 | 0.1320 | 0.1146 | 0.1502 | 0.307 | 0.000 |
| scifact | single-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus full query | 659 | 0.2210 | 0.1948 | 0.2472 | 0.357 | 0.000 |
| scifact | single-positive | 819 | ndcg@10 | prefix-k minus full query | 659 | -0.0111 | -0.0173 | -0.0051 | 0.042 | 0.083 |
| scifact | single-positive | 819 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.1058 | 0.0875 | 0.1247 | 0.196 | 0.000 |
| scifact | single-positive | 819 | mrr@10 | L2S (in-sample) minus full query | 659 | 0.1369 | 0.1180 | 0.1567 | 0.307 | 0.000 |
| scifact | single-positive | 819 | mrr@10 | all-positive oracle (privileged) minus full query | 659 | 0.2427 | 0.2145 | 0.2709 | 0.357 | 0.000 |
| scifact | single-positive | 819 | mrr@10 | prefix-k minus full query | 659 | -0.0136 | -0.0215 | -0.0059 | 0.042 | 0.083 |
| scifact | single-positive | 819 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.1041 | 0.0862 | 0.1227 | 0.199 | 0.000 |
| scifact | single-positive | 819 | mrr@100 | L2S (in-sample) minus full query | 659 | 0.1345 | 0.1159 | 0.1540 | 0.341 | 0.000 |
| scifact | single-positive | 819 | mrr@100 | all-positive oracle (privileged) minus full query | 659 | 0.2386 | 0.2112 | 0.2659 | 0.360 | 0.000 |
| scifact | single-positive | 819 | mrr@100 | prefix-k minus full query | 659 | -0.0133 | -0.0211 | -0.0057 | 0.097 | 0.137 |
| scifact | single-positive | 819 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0030 | 0.0000 | 0.0076 | 0.003 | 0.000 |
| scifact | single-positive | 819 | recall@100 | L2S (in-sample) minus full query | 659 | 0.0440 | 0.0288 | 0.0607 | 0.044 | 0.000 |
| scifact | single-positive | 819 | recall@100 | all-positive oracle (privileged) minus full query | 659 | 0.0470 | 0.0303 | 0.0637 | 0.047 | 0.000 |
| scifact | single-positive | 819 | recall@100 | prefix-k minus full query | 659 | 0.0091 | 0.0015 | 0.0182 | 0.011 | 0.002 |
| scifact | single-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0936 | 0.0788 | 0.1086 | 0.241 | 0.000 |
| scifact | single-positive | 921 | ndcg@10 | L2S (in-sample) minus full query | 659 | 0.0786 | 0.0660 | 0.0910 | 0.243 | 0.000 |
| scifact | single-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus full query | 659 | 0.1722 | 0.1512 | 0.1934 | 0.343 | 0.000 |
| scifact | single-positive | 921 | ndcg@10 | prefix-k minus full query | 659 | -0.0019 | -0.0072 | 0.0036 | 0.055 | 0.050 |
| scifact | single-positive | 921 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.1013 | 0.0848 | 0.1178 | 0.241 | 0.000 |
| scifact | single-positive | 921 | mrr@10 | L2S (in-sample) minus full query | 659 | 0.0805 | 0.0670 | 0.0944 | 0.243 | 0.000 |
| scifact | single-positive | 921 | mrr@10 | all-positive oracle (privileged) minus full query | 659 | 0.1818 | 0.1601 | 0.2040 | 0.343 | 0.000 |
| scifact | single-positive | 921 | mrr@10 | prefix-k minus full query | 659 | -0.0036 | -0.0104 | 0.0031 | 0.055 | 0.050 |
| scifact | single-positive | 921 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0981 | 0.0817 | 0.1140 | 0.253 | 0.000 |
| scifact | single-positive | 921 | mrr@100 | L2S (in-sample) minus full query | 659 | 0.0800 | 0.0667 | 0.0941 | 0.319 | 0.000 |
| scifact | single-positive | 921 | mrr@100 | all-positive oracle (privileged) minus full query | 659 | 0.1781 | 0.1569 | 0.1999 | 0.355 | 0.000 |
| scifact | single-positive | 921 | mrr@100 | prefix-k minus full query | 659 | -0.0038 | -0.0105 | 0.0029 | 0.108 | 0.094 |
| scifact | single-positive | 921 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 659 | 0.0046 | 0.0000 | 0.0106 | 0.005 | 0.000 |
| scifact | single-positive | 921 | recall@100 | L2S (in-sample) minus full query | 659 | 0.0410 | 0.0258 | 0.0561 | 0.041 | 0.000 |
| scifact | single-positive | 921 | recall@100 | all-positive oracle (privileged) minus full query | 659 | 0.0455 | 0.0303 | 0.0622 | 0.046 | 0.000 |
| scifact | single-positive | 921 | recall@100 | prefix-k minus full query | 659 | 0.0046 | -0.0015 | 0.0121 | 0.006 | 0.002 |
| scifact | multi-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0384 | 0.0121 | 0.0678 | 0.362 | 0.130 |
| scifact | multi-positive | 102 | ndcg@10 | L2S (in-sample) minus full query | 69 | 0.2240 | 0.1680 | 0.2806 | 0.739 | 0.058 |
| scifact | multi-positive | 102 | ndcg@10 | all-positive oracle (privileged) minus full query | 69 | 0.2624 | 0.2001 | 0.3301 | 0.754 | 0.000 |
| scifact | multi-positive | 102 | ndcg@10 | prefix-k minus full query | 69 | -0.1462 | -0.1990 | -0.0932 | 0.101 | 0.623 |
| scifact | multi-positive | 102 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0210 | -0.0171 | 0.0628 | 0.087 | 0.058 |
| scifact | multi-positive | 102 | mrr@10 | L2S (in-sample) minus full query | 69 | 0.2327 | 0.1541 | 0.3138 | 0.420 | 0.043 |
| scifact | multi-positive | 102 | mrr@10 | all-positive oracle (privileged) minus full query | 69 | 0.2537 | 0.1727 | 0.3379 | 0.406 | 0.000 |
| scifact | multi-positive | 102 | mrr@10 | prefix-k minus full query | 69 | -0.1172 | -0.1887 | -0.0480 | 0.058 | 0.377 |
| scifact | multi-positive | 102 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0210 | -0.0171 | 0.0628 | 0.087 | 0.058 |
| scifact | multi-positive | 102 | mrr@100 | L2S (in-sample) minus full query | 69 | 0.2295 | 0.1512 | 0.3102 | 0.420 | 0.043 |
| scifact | multi-positive | 102 | mrr@100 | all-positive oracle (privileged) minus full query | 69 | 0.2505 | 0.1702 | 0.3342 | 0.406 | 0.000 |
| scifact | multi-positive | 102 | mrr@100 | prefix-k minus full query | 69 | -0.1153 | -0.1854 | -0.0472 | 0.087 | 0.420 |
| scifact | multi-positive | 102 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | multi-positive | 102 | recall@100 | L2S (in-sample) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 102 | recall@100 | all-positive oracle (privileged) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 102 | recall@100 | prefix-k minus full query | 69 | -0.0556 | -0.1087 | -0.0121 | 0.014 | 0.087 |
| scifact | multi-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0287 | 0.0100 | 0.0494 | 0.348 | 0.159 |
| scifact | multi-positive | 204 | ndcg@10 | L2S (in-sample) minus full query | 69 | 0.2318 | 0.1759 | 0.2897 | 0.739 | 0.043 |
| scifact | multi-positive | 204 | ndcg@10 | all-positive oracle (privileged) minus full query | 69 | 0.2604 | 0.1991 | 0.3241 | 0.783 | 0.000 |
| scifact | multi-positive | 204 | ndcg@10 | prefix-k minus full query | 69 | -0.0661 | -0.1017 | -0.0322 | 0.203 | 0.464 |
| scifact | multi-positive | 204 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0128 | -0.0036 | 0.0341 | 0.058 | 0.029 |
| scifact | multi-positive | 204 | mrr@10 | L2S (in-sample) minus full query | 69 | 0.2501 | 0.1696 | 0.3309 | 0.420 | 0.014 |
| scifact | multi-positive | 204 | mrr@10 | all-positive oracle (privileged) minus full query | 69 | 0.2629 | 0.1826 | 0.3461 | 0.391 | 0.000 |
| scifact | multi-positive | 204 | mrr@10 | prefix-k minus full query | 69 | -0.0357 | -0.0906 | 0.0187 | 0.145 | 0.217 |
| scifact | multi-positive | 204 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0128 | -0.0036 | 0.0341 | 0.058 | 0.029 |
| scifact | multi-positive | 204 | mrr@100 | L2S (in-sample) minus full query | 69 | 0.2469 | 0.1670 | 0.3273 | 0.420 | 0.014 |
| scifact | multi-positive | 204 | mrr@100 | all-positive oracle (privileged) minus full query | 69 | 0.2597 | 0.1808 | 0.3412 | 0.391 | 0.000 |
| scifact | multi-positive | 204 | mrr@100 | prefix-k minus full query | 69 | -0.0366 | -0.0921 | 0.0187 | 0.159 | 0.261 |
| scifact | multi-positive | 204 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | multi-positive | 204 | recall@100 | L2S (in-sample) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 204 | recall@100 | all-positive oracle (privileged) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 204 | recall@100 | prefix-k minus full query | 69 | -0.0242 | -0.0652 | 0.0048 | 0.014 | 0.043 |
| scifact | multi-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0303 | 0.0095 | 0.0536 | 0.333 | 0.116 |
| scifact | multi-positive | 307 | ndcg@10 | L2S (in-sample) minus full query | 69 | 0.2154 | 0.1615 | 0.2707 | 0.696 | 0.000 |
| scifact | multi-positive | 307 | ndcg@10 | all-positive oracle (privileged) minus full query | 69 | 0.2457 | 0.1865 | 0.3076 | 0.783 | 0.000 |
| scifact | multi-positive | 307 | ndcg@10 | prefix-k minus full query | 69 | -0.0518 | -0.0810 | -0.0224 | 0.203 | 0.435 |
| scifact | multi-positive | 307 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0140 | -0.0048 | 0.0372 | 0.058 | 0.029 |
| scifact | multi-positive | 307 | mrr@10 | L2S (in-sample) minus full query | 69 | 0.2441 | 0.1657 | 0.3231 | 0.406 | 0.014 |
| scifact | multi-positive | 307 | mrr@10 | all-positive oracle (privileged) minus full query | 69 | 0.2581 | 0.1773 | 0.3404 | 0.391 | 0.000 |
| scifact | multi-positive | 307 | mrr@10 | prefix-k minus full query | 69 | -0.0226 | -0.0725 | 0.0278 | 0.130 | 0.217 |
| scifact | multi-positive | 307 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0140 | -0.0048 | 0.0372 | 0.058 | 0.029 |
| scifact | multi-positive | 307 | mrr@100 | L2S (in-sample) minus full query | 69 | 0.2409 | 0.1633 | 0.3200 | 0.406 | 0.014 |
| scifact | multi-positive | 307 | mrr@100 | all-positive oracle (privileged) minus full query | 69 | 0.2549 | 0.1753 | 0.3367 | 0.391 | 0.000 |
| scifact | multi-positive | 307 | mrr@100 | prefix-k minus full query | 69 | -0.0237 | -0.0738 | 0.0269 | 0.159 | 0.261 |
| scifact | multi-positive | 307 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0072 | 0.0000 | 0.0217 | 0.014 | 0.000 |
| scifact | multi-positive | 307 | recall@100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 307 | recall@100 | all-positive oracle (privileged) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 307 | recall@100 | prefix-k minus full query | 69 | -0.0169 | -0.0556 | 0.0097 | 0.014 | 0.029 |
| scifact | multi-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0480 | 0.0232 | 0.0769 | 0.319 | 0.101 |
| scifact | multi-positive | 409 | ndcg@10 | L2S (in-sample) minus full query | 69 | 0.1868 | 0.1389 | 0.2362 | 0.710 | 0.014 |
| scifact | multi-positive | 409 | ndcg@10 | all-positive oracle (privileged) minus full query | 69 | 0.2348 | 0.1767 | 0.2958 | 0.739 | 0.000 |
| scifact | multi-positive | 409 | ndcg@10 | prefix-k minus full query | 69 | -0.0374 | -0.0689 | -0.0085 | 0.203 | 0.464 |
| scifact | multi-positive | 409 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0354 | 0.0100 | 0.0680 | 0.101 | 0.014 |
| scifact | multi-positive | 409 | mrr@10 | L2S (in-sample) minus full query | 69 | 0.2203 | 0.1467 | 0.2930 | 0.377 | 0.014 |
| scifact | multi-positive | 409 | mrr@10 | all-positive oracle (privileged) minus full query | 69 | 0.2557 | 0.1734 | 0.3374 | 0.391 | 0.000 |
| scifact | multi-positive | 409 | mrr@10 | prefix-k minus full query | 69 | 0.0017 | -0.0448 | 0.0459 | 0.116 | 0.188 |
| scifact | multi-positive | 409 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0354 | 0.0100 | 0.0680 | 0.101 | 0.014 |
| scifact | multi-positive | 409 | mrr@100 | L2S (in-sample) minus full query | 69 | 0.2171 | 0.1439 | 0.2899 | 0.377 | 0.014 |
| scifact | multi-positive | 409 | mrr@100 | all-positive oracle (privileged) minus full query | 69 | 0.2525 | 0.1715 | 0.3332 | 0.391 | 0.000 |
| scifact | multi-positive | 409 | mrr@100 | prefix-k minus full query | 69 | 0.0018 | -0.0453 | 0.0462 | 0.145 | 0.232 |
| scifact | multi-positive | 409 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0072 | 0.0000 | 0.0217 | 0.014 | 0.000 |
| scifact | multi-positive | 409 | recall@100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 409 | recall@100 | all-positive oracle (privileged) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 409 | recall@100 | prefix-k minus full query | 69 | 0.0048 | 0.0000 | 0.0145 | 0.014 | 0.000 |
| scifact | multi-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0572 | 0.0250 | 0.0928 | 0.319 | 0.087 |
| scifact | multi-positive | 512 | ndcg@10 | L2S (in-sample) minus full query | 69 | 0.1694 | 0.1231 | 0.2170 | 0.667 | 0.014 |
| scifact | multi-positive | 512 | ndcg@10 | all-positive oracle (privileged) minus full query | 69 | 0.2266 | 0.1686 | 0.2893 | 0.725 | 0.000 |
| scifact | multi-positive | 512 | ndcg@10 | prefix-k minus full query | 69 | -0.0402 | -0.0737 | -0.0109 | 0.145 | 0.377 |
| scifact | multi-positive | 512 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0470 | 0.0109 | 0.0892 | 0.101 | 0.014 |
| scifact | multi-positive | 512 | mrr@10 | L2S (in-sample) minus full query | 69 | 0.2063 | 0.1365 | 0.2795 | 0.362 | 0.000 |
| scifact | multi-positive | 512 | mrr@10 | all-positive oracle (privileged) minus full query | 69 | 0.2533 | 0.1716 | 0.3355 | 0.391 | 0.000 |
| scifact | multi-positive | 512 | mrr@10 | prefix-k minus full query | 69 | -0.0072 | -0.0580 | 0.0403 | 0.116 | 0.159 |
| scifact | multi-positive | 512 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0470 | 0.0109 | 0.0892 | 0.101 | 0.014 |
| scifact | multi-positive | 512 | mrr@100 | L2S (in-sample) minus full query | 69 | 0.2031 | 0.1338 | 0.2740 | 0.362 | 0.000 |
| scifact | multi-positive | 512 | mrr@100 | all-positive oracle (privileged) minus full query | 69 | 0.2501 | 0.1695 | 0.3320 | 0.391 | 0.000 |
| scifact | multi-positive | 512 | mrr@100 | prefix-k minus full query | 69 | -0.0084 | -0.0597 | 0.0394 | 0.145 | 0.203 |
| scifact | multi-positive | 512 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0072 | 0.0000 | 0.0217 | 0.014 | 0.000 |
| scifact | multi-positive | 512 | recall@100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 512 | recall@100 | all-positive oracle (privileged) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 512 | recall@100 | prefix-k minus full query | 69 | 0.0048 | 0.0000 | 0.0145 | 0.014 | 0.000 |
| scifact | multi-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0666 | 0.0325 | 0.1051 | 0.333 | 0.043 |
| scifact | multi-positive | 614 | ndcg@10 | L2S (in-sample) minus full query | 69 | 0.1456 | 0.1063 | 0.1886 | 0.681 | 0.000 |
| scifact | multi-positive | 614 | ndcg@10 | all-positive oracle (privileged) minus full query | 69 | 0.2122 | 0.1567 | 0.2727 | 0.696 | 0.000 |
| scifact | multi-positive | 614 | ndcg@10 | prefix-k minus full query | 69 | -0.0220 | -0.0506 | 0.0045 | 0.130 | 0.362 |
| scifact | multi-positive | 614 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0641 | 0.0181 | 0.1138 | 0.159 | 0.014 |
| scifact | multi-positive | 614 | mrr@10 | L2S (in-sample) minus full query | 69 | 0.1758 | 0.1124 | 0.2419 | 0.333 | 0.000 |
| scifact | multi-positive | 614 | mrr@10 | all-positive oracle (privileged) minus full query | 69 | 0.2400 | 0.1626 | 0.3184 | 0.391 | 0.000 |
| scifact | multi-positive | 614 | mrr@10 | prefix-k minus full query | 69 | -0.0173 | -0.0633 | 0.0269 | 0.087 | 0.174 |
| scifact | multi-positive | 614 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0641 | 0.0181 | 0.1138 | 0.159 | 0.014 |
| scifact | multi-positive | 614 | mrr@100 | L2S (in-sample) minus full query | 69 | 0.1726 | 0.1085 | 0.2389 | 0.333 | 0.000 |
| scifact | multi-positive | 614 | mrr@100 | all-positive oracle (privileged) minus full query | 69 | 0.2368 | 0.1602 | 0.3137 | 0.391 | 0.000 |
| scifact | multi-positive | 614 | mrr@100 | prefix-k minus full query | 69 | -0.0182 | -0.0643 | 0.0260 | 0.101 | 0.217 |
| scifact | multi-positive | 614 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0072 | 0.0000 | 0.0217 | 0.014 | 0.000 |
| scifact | multi-positive | 614 | recall@100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 614 | recall@100 | all-positive oracle (privileged) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 614 | recall@100 | prefix-k minus full query | 69 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | multi-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0773 | 0.0420 | 0.1189 | 0.362 | 0.000 |
| scifact | multi-positive | 716 | ndcg@10 | L2S (in-sample) minus full query | 69 | 0.1213 | 0.0869 | 0.1584 | 0.638 | 0.000 |
| scifact | multi-positive | 716 | ndcg@10 | all-positive oracle (privileged) minus full query | 69 | 0.1985 | 0.1451 | 0.2575 | 0.696 | 0.000 |
| scifact | multi-positive | 716 | ndcg@10 | prefix-k minus full query | 69 | -0.0084 | -0.0318 | 0.0133 | 0.145 | 0.290 |
| scifact | multi-positive | 716 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0864 | 0.0386 | 0.1431 | 0.174 | 0.000 |
| scifact | multi-positive | 716 | mrr@10 | L2S (in-sample) minus full query | 69 | 0.1428 | 0.0853 | 0.2009 | 0.319 | 0.000 |
| scifact | multi-positive | 716 | mrr@10 | all-positive oracle (privileged) minus full query | 69 | 0.2291 | 0.1553 | 0.3069 | 0.391 | 0.000 |
| scifact | multi-positive | 716 | mrr@10 | prefix-k minus full query | 69 | 0.0010 | -0.0357 | 0.0410 | 0.072 | 0.116 |
| scifact | multi-positive | 716 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0850 | 0.0372 | 0.1412 | 0.174 | 0.000 |
| scifact | multi-positive | 716 | mrr@100 | L2S (in-sample) minus full query | 69 | 0.1409 | 0.0848 | 0.1994 | 0.333 | 0.000 |
| scifact | multi-positive | 716 | mrr@100 | all-positive oracle (privileged) minus full query | 69 | 0.2259 | 0.1530 | 0.3033 | 0.391 | 0.000 |
| scifact | multi-positive | 716 | mrr@100 | prefix-k minus full query | 69 | 0.0017 | -0.0349 | 0.0413 | 0.087 | 0.159 |
| scifact | multi-positive | 716 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0072 | 0.0000 | 0.0217 | 0.014 | 0.000 |
| scifact | multi-positive | 716 | recall@100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 716 | recall@100 | all-positive oracle (privileged) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 716 | recall@100 | prefix-k minus full query | 69 | 0.0048 | 0.0000 | 0.0145 | 0.014 | 0.000 |
| scifact | multi-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0919 | 0.0546 | 0.1355 | 0.449 | 0.029 |
| scifact | multi-positive | 819 | ndcg@10 | L2S (in-sample) minus full query | 69 | 0.0933 | 0.0636 | 0.1238 | 0.594 | 0.000 |
| scifact | multi-positive | 819 | ndcg@10 | all-positive oracle (privileged) minus full query | 69 | 0.1852 | 0.1343 | 0.2419 | 0.696 | 0.000 |
| scifact | multi-positive | 819 | ndcg@10 | prefix-k minus full query | 69 | -0.0127 | -0.0390 | 0.0116 | 0.130 | 0.246 |
| scifact | multi-positive | 819 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.1031 | 0.0513 | 0.1620 | 0.203 | 0.000 |
| scifact | multi-positive | 819 | mrr@10 | L2S (in-sample) minus full query | 69 | 0.1085 | 0.0627 | 0.1586 | 0.304 | 0.000 |
| scifact | multi-positive | 819 | mrr@10 | all-positive oracle (privileged) minus full query | 69 | 0.2116 | 0.1415 | 0.2880 | 0.377 | 0.000 |
| scifact | multi-positive | 819 | mrr@10 | prefix-k minus full query | 69 | 0.0025 | -0.0302 | 0.0336 | 0.072 | 0.087 |
| scifact | multi-positive | 819 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.1005 | 0.0499 | 0.1585 | 0.203 | 0.000 |
| scifact | multi-positive | 819 | mrr@100 | L2S (in-sample) minus full query | 69 | 0.1079 | 0.0629 | 0.1566 | 0.333 | 0.000 |
| scifact | multi-positive | 819 | mrr@100 | all-positive oracle (privileged) minus full query | 69 | 0.2084 | 0.1391 | 0.2841 | 0.377 | 0.000 |
| scifact | multi-positive | 819 | mrr@100 | prefix-k minus full query | 69 | 0.0050 | -0.0272 | 0.0354 | 0.116 | 0.116 |
| scifact | multi-positive | 819 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0072 | 0.0000 | 0.0217 | 0.014 | 0.000 |
| scifact | multi-positive | 819 | recall@100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 819 | recall@100 | all-positive oracle (privileged) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 819 | recall@100 | prefix-k minus full query | 69 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | multi-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0828 | 0.0541 | 0.1142 | 0.493 | 0.014 |
| scifact | multi-positive | 921 | ndcg@10 | L2S (in-sample) minus full query | 69 | 0.0596 | 0.0370 | 0.0843 | 0.493 | 0.029 |
| scifact | multi-positive | 921 | ndcg@10 | all-positive oracle (privileged) minus full query | 69 | 0.1424 | 0.1044 | 0.1849 | 0.681 | 0.000 |
| scifact | multi-positive | 921 | ndcg@10 | prefix-k minus full query | 69 | -0.0090 | -0.0234 | 0.0033 | 0.101 | 0.145 |
| scifact | multi-positive | 921 | mrr@10 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0974 | 0.0532 | 0.1469 | 0.246 | 0.014 |
| scifact | multi-positive | 921 | mrr@10 | L2S (in-sample) minus full query | 69 | 0.0571 | 0.0283 | 0.0906 | 0.246 | 0.000 |
| scifact | multi-positive | 921 | mrr@10 | all-positive oracle (privileged) minus full query | 69 | 0.1545 | 0.0991 | 0.2138 | 0.377 | 0.000 |
| scifact | multi-positive | 921 | mrr@10 | prefix-k minus full query | 69 | -0.0126 | -0.0368 | 0.0045 | 0.043 | 0.058 |
| scifact | multi-positive | 921 | mrr@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0943 | 0.0495 | 0.1440 | 0.246 | 0.014 |
| scifact | multi-positive | 921 | mrr@100 | L2S (in-sample) minus full query | 69 | 0.0570 | 0.0282 | 0.0900 | 0.290 | 0.000 |
| scifact | multi-positive | 921 | mrr@100 | all-positive oracle (privileged) minus full query | 69 | 0.1513 | 0.0960 | 0.2105 | 0.377 | 0.000 |
| scifact | multi-positive | 921 | mrr@100 | prefix-k minus full query | 69 | -0.0129 | -0.0368 | 0.0045 | 0.072 | 0.087 |
| scifact | multi-positive | 921 | recall@100 | all-positive oracle (privileged) minus L2S (in-sample) | 69 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| scifact | multi-positive | 921 | recall@100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 921 | recall@100 | all-positive oracle (privileged) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 921 | recall@100 | prefix-k minus full query | 69 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |

## Held-out positives of multi-positive queries

Same partitions for every arm. Measures use only the held-out positives.

| environment | cohort | k | measure | comparison | queries | difference | 2.5% | 97.5% | share better | share worse |
|---|---|---|---|---|---|---|---|---|---|---|
| scifact | multi-positive | 102 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.2815 | -0.3223 | -0.2398 | 0.029 | 0.913 |
| scifact | multi-positive | 102 | held_rr | L2S (in-sample) minus full query | 69 | 0.1803 | 0.1325 | 0.2286 | 0.710 | 0.087 |
| scifact | multi-positive | 102 | held_rr | held-out transfer (1-1) minus full query | 69 | -0.1013 | -0.1388 | -0.0616 | 0.217 | 0.710 |
| scifact | multi-positive | 102 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.3157 | 0.2692 | 0.3609 | 0.928 | 0.000 |
| scifact | multi-positive | 102 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 69 | 0.7289 | 0.5762 | 0.8935 | 0.913 | 0.029 |
| scifact | multi-positive | 102 | held_logrank | L2S (in-sample) minus full query | 69 | -0.6180 | -0.7987 | -0.4560 | 0.072 | 0.725 |
| scifact | multi-positive | 102 | held_logrank | held-out transfer (1-1) minus full query | 69 | 0.1109 | -0.0222 | 0.2299 | 0.667 | 0.261 |
| scifact | multi-positive | 102 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | -0.8520 | -1.0492 | -0.6748 | 0.000 | 0.928 |
| scifact | multi-positive | 102 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0990 | -0.1522 | -0.0531 | 0.000 | 0.217 |
| scifact | multi-positive | 102 | held_recall10 | L2S (in-sample) minus full query | 69 | 0.1498 | 0.0894 | 0.2150 | 0.275 | 0.000 |
| scifact | multi-positive | 102 | held_recall10 | held-out transfer (1-1) minus full query | 69 | 0.0507 | 0.0048 | 0.0990 | 0.130 | 0.029 |
| scifact | multi-positive | 102 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.1135 | 0.0604 | 0.1787 | 0.217 | 0.000 |
| scifact | multi-positive | 102 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0242 | -0.0508 | -0.0048 | 0.000 | 0.058 |
| scifact | multi-positive | 102 | held_recall100 | L2S (in-sample) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 102 | held_recall100 | held-out transfer (1-1) minus full query | 69 | -0.0048 | -0.0290 | 0.0217 | 0.029 | 0.029 |
| scifact | multi-positive | 102 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0242 | 0.0048 | 0.0508 | 0.058 | 0.000 |
| scifact | multi-positive | 204 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.2854 | -0.3246 | -0.2452 | 0.014 | 0.928 |
| scifact | multi-positive | 204 | held_rr | L2S (in-sample) minus full query | 69 | 0.1895 | 0.1425 | 0.2375 | 0.739 | 0.072 |
| scifact | multi-positive | 204 | held_rr | held-out transfer (1-1) minus full query | 69 | -0.0959 | -0.1313 | -0.0581 | 0.217 | 0.710 |
| scifact | multi-positive | 204 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.3077 | 0.2668 | 0.3461 | 0.942 | 0.000 |
| scifact | multi-positive | 204 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 69 | 0.7074 | 0.5639 | 0.8636 | 0.942 | 0.000 |
| scifact | multi-positive | 204 | held_logrank | L2S (in-sample) minus full query | 69 | -0.6402 | -0.8298 | -0.4717 | 0.072 | 0.739 |
| scifact | multi-positive | 204 | held_logrank | held-out transfer (1-1) minus full query | 69 | 0.0672 | -0.0657 | 0.1789 | 0.652 | 0.275 |
| scifact | multi-positive | 204 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | -0.7933 | -0.9713 | -0.6367 | 0.000 | 0.942 |
| scifact | multi-positive | 204 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0918 | -0.1401 | -0.0507 | 0.000 | 0.188 |
| scifact | multi-positive | 204 | held_recall10 | L2S (in-sample) minus full query | 69 | 0.1425 | 0.0870 | 0.2041 | 0.275 | 0.000 |
| scifact | multi-positive | 204 | held_recall10 | held-out transfer (1-1) minus full query | 69 | 0.0507 | 0.0060 | 0.1014 | 0.145 | 0.029 |
| scifact | multi-positive | 204 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.1111 | 0.0604 | 0.1715 | 0.203 | 0.000 |
| scifact | multi-positive | 204 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0242 | -0.0508 | -0.0048 | 0.000 | 0.058 |
| scifact | multi-positive | 204 | held_recall100 | L2S (in-sample) minus full query | 69 | 0.0193 | 0.0000 | 0.0435 | 0.043 | 0.000 |
| scifact | multi-positive | 204 | held_recall100 | held-out transfer (1-1) minus full query | 69 | -0.0048 | -0.0290 | 0.0217 | 0.029 | 0.029 |
| scifact | multi-positive | 204 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0242 | 0.0048 | 0.0508 | 0.058 | 0.000 |
| scifact | multi-positive | 307 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.2725 | -0.3077 | -0.2356 | 0.000 | 0.913 |
| scifact | multi-positive | 307 | held_rr | L2S (in-sample) minus full query | 69 | 0.1793 | 0.1344 | 0.2266 | 0.725 | 0.014 |
| scifact | multi-positive | 307 | held_rr | held-out transfer (1-1) minus full query | 69 | -0.0932 | -0.1297 | -0.0556 | 0.232 | 0.681 |
| scifact | multi-positive | 307 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.2908 | 0.2508 | 0.3279 | 0.928 | 0.000 |
| scifact | multi-positive | 307 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 69 | 0.6361 | 0.5204 | 0.7702 | 0.913 | 0.014 |
| scifact | multi-positive | 307 | held_logrank | L2S (in-sample) minus full query | 69 | -0.5920 | -0.7589 | -0.4441 | 0.000 | 0.739 |
| scifact | multi-positive | 307 | held_logrank | held-out transfer (1-1) minus full query | 69 | 0.0440 | -0.0873 | 0.1592 | 0.594 | 0.333 |
| scifact | multi-positive | 307 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | -0.7310 | -0.8949 | -0.5878 | 0.000 | 0.928 |
| scifact | multi-positive | 307 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0791 | -0.1232 | -0.0399 | 0.000 | 0.174 |
| scifact | multi-positive | 307 | held_recall10 | L2S (in-sample) minus full query | 69 | 0.1280 | 0.0749 | 0.1884 | 0.246 | 0.000 |
| scifact | multi-positive | 307 | held_recall10 | held-out transfer (1-1) minus full query | 69 | 0.0489 | -0.0006 | 0.1069 | 0.145 | 0.043 |
| scifact | multi-positive | 307 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.1057 | 0.0567 | 0.1636 | 0.203 | 0.000 |
| scifact | multi-positive | 307 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0145 | -0.0362 | 0.0000 | 0.000 | 0.029 |
| scifact | multi-positive | 307 | held_recall100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 307 | held_recall100 | held-out transfer (1-1) minus full query | 69 | -0.0024 | -0.0290 | 0.0242 | 0.029 | 0.029 |
| scifact | multi-positive | 307 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0217 | 0.0000 | 0.0507 | 0.043 | 0.000 |
| scifact | multi-positive | 409 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.2290 | -0.2657 | -0.1918 | 0.000 | 0.884 |
| scifact | multi-positive | 409 | held_rr | L2S (in-sample) minus full query | 69 | 0.1580 | 0.1168 | 0.2004 | 0.739 | 0.029 |
| scifact | multi-positive | 409 | held_rr | held-out transfer (1-1) minus full query | 69 | -0.0710 | -0.1061 | -0.0361 | 0.304 | 0.580 |
| scifact | multi-positive | 409 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.2634 | 0.2235 | 0.3037 | 0.884 | 0.000 |
| scifact | multi-positive | 409 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 69 | 0.5426 | 0.4344 | 0.6660 | 0.870 | 0.014 |
| scifact | multi-positive | 409 | held_logrank | L2S (in-sample) minus full query | 69 | -0.5339 | -0.6886 | -0.3968 | 0.014 | 0.754 |
| scifact | multi-positive | 409 | held_logrank | held-out transfer (1-1) minus full query | 69 | 0.0087 | -0.1097 | 0.1161 | 0.580 | 0.319 |
| scifact | multi-positive | 409 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | -0.6743 | -0.8414 | -0.5316 | 0.014 | 0.884 |
| scifact | multi-positive | 409 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0574 | -0.1014 | -0.0181 | 0.029 | 0.145 |
| scifact | multi-positive | 409 | held_recall10 | L2S (in-sample) minus full query | 69 | 0.1014 | 0.0531 | 0.1582 | 0.203 | 0.000 |
| scifact | multi-positive | 409 | held_recall10 | held-out transfer (1-1) minus full query | 69 | 0.0441 | 0.0012 | 0.0930 | 0.130 | 0.029 |
| scifact | multi-positive | 409 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0984 | 0.0531 | 0.1510 | 0.203 | 0.000 |
| scifact | multi-positive | 409 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0145 | -0.0362 | 0.0000 | 0.000 | 0.029 |
| scifact | multi-positive | 409 | held_recall100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 409 | held_recall100 | held-out transfer (1-1) minus full query | 69 | -0.0024 | -0.0290 | 0.0242 | 0.029 | 0.029 |
| scifact | multi-positive | 409 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0217 | 0.0000 | 0.0507 | 0.043 | 0.000 |
| scifact | multi-positive | 512 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.2024 | -0.2404 | -0.1649 | 0.014 | 0.855 |
| scifact | multi-positive | 512 | held_rr | L2S (in-sample) minus full query | 69 | 0.1431 | 0.1024 | 0.1847 | 0.696 | 0.029 |
| scifact | multi-positive | 512 | held_rr | held-out transfer (1-1) minus full query | 69 | -0.0593 | -0.0922 | -0.0273 | 0.333 | 0.522 |
| scifact | multi-positive | 512 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.2459 | 0.2063 | 0.2870 | 0.884 | 0.000 |
| scifact | multi-positive | 512 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 69 | 0.4739 | 0.3731 | 0.5875 | 0.855 | 0.029 |
| scifact | multi-positive | 512 | held_logrank | L2S (in-sample) minus full query | 69 | -0.4919 | -0.6292 | -0.3675 | 0.000 | 0.725 |
| scifact | multi-positive | 512 | held_logrank | held-out transfer (1-1) minus full query | 69 | -0.0180 | -0.1314 | 0.0809 | 0.536 | 0.348 |
| scifact | multi-positive | 512 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | -0.6293 | -0.7842 | -0.4926 | 0.014 | 0.884 |
| scifact | multi-positive | 512 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0537 | -0.0990 | -0.0127 | 0.043 | 0.130 |
| scifact | multi-positive | 512 | held_recall10 | L2S (in-sample) minus full query | 69 | 0.0906 | 0.0459 | 0.1413 | 0.188 | 0.000 |
| scifact | multi-positive | 512 | held_recall10 | held-out transfer (1-1) minus full query | 69 | 0.0368 | -0.0042 | 0.0821 | 0.116 | 0.029 |
| scifact | multi-positive | 512 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0984 | 0.0501 | 0.1558 | 0.188 | 0.000 |
| scifact | multi-positive | 512 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0145 | -0.0362 | 0.0000 | 0.000 | 0.029 |
| scifact | multi-positive | 512 | held_recall100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 512 | held_recall100 | held-out transfer (1-1) minus full query | 69 | -0.0024 | -0.0290 | 0.0242 | 0.029 | 0.029 |
| scifact | multi-positive | 512 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0217 | 0.0000 | 0.0507 | 0.043 | 0.000 |
| scifact | multi-positive | 614 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.1842 | -0.2224 | -0.1479 | 0.043 | 0.841 |
| scifact | multi-positive | 614 | held_rr | L2S (in-sample) minus full query | 69 | 0.1256 | 0.0898 | 0.1632 | 0.725 | 0.000 |
| scifact | multi-positive | 614 | held_rr | held-out transfer (1-1) minus full query | 69 | -0.0586 | -0.0903 | -0.0291 | 0.319 | 0.536 |
| scifact | multi-positive | 614 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.2353 | 0.1959 | 0.2759 | 0.884 | 0.000 |
| scifact | multi-positive | 614 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 69 | 0.4259 | 0.3273 | 0.5347 | 0.855 | 0.043 |
| scifact | multi-positive | 614 | held_logrank | L2S (in-sample) minus full query | 69 | -0.4473 | -0.5779 | -0.3303 | 0.000 | 0.725 |
| scifact | multi-positive | 614 | held_logrank | held-out transfer (1-1) minus full query | 69 | -0.0215 | -0.1266 | 0.0711 | 0.536 | 0.348 |
| scifact | multi-positive | 614 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | -0.5996 | -0.7383 | -0.4782 | 0.014 | 0.884 |
| scifact | multi-positive | 614 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0429 | -0.0833 | -0.0048 | 0.043 | 0.116 |
| scifact | multi-positive | 614 | held_recall10 | L2S (in-sample) minus full query | 69 | 0.0761 | 0.0386 | 0.1184 | 0.174 | 0.000 |
| scifact | multi-positive | 614 | held_recall10 | held-out transfer (1-1) minus full query | 69 | 0.0332 | -0.0072 | 0.0797 | 0.101 | 0.029 |
| scifact | multi-positive | 614 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0876 | 0.0435 | 0.1395 | 0.188 | 0.000 |
| scifact | multi-positive | 614 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0145 | -0.0362 | 0.0000 | 0.000 | 0.029 |
| scifact | multi-positive | 614 | held_recall100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 614 | held_recall100 | held-out transfer (1-1) minus full query | 69 | -0.0024 | -0.0290 | 0.0242 | 0.029 | 0.029 |
| scifact | multi-positive | 614 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0217 | 0.0000 | 0.0507 | 0.043 | 0.000 |
| scifact | multi-positive | 716 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.1629 | -0.2004 | -0.1258 | 0.043 | 0.826 |
| scifact | multi-positive | 716 | held_rr | L2S (in-sample) minus full query | 69 | 0.1042 | 0.0728 | 0.1382 | 0.710 | 0.000 |
| scifact | multi-positive | 716 | held_rr | held-out transfer (1-1) minus full query | 69 | -0.0587 | -0.0877 | -0.0293 | 0.290 | 0.594 |
| scifact | multi-positive | 716 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.2243 | 0.1861 | 0.2655 | 0.884 | 0.000 |
| scifact | multi-positive | 716 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 69 | 0.3848 | 0.2931 | 0.4850 | 0.826 | 0.058 |
| scifact | multi-positive | 716 | held_logrank | L2S (in-sample) minus full query | 69 | -0.3845 | -0.4957 | -0.2867 | 0.000 | 0.710 |
| scifact | multi-positive | 716 | held_logrank | held-out transfer (1-1) minus full query | 69 | 0.0003 | -0.0926 | 0.0819 | 0.565 | 0.333 |
| scifact | multi-positive | 716 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | -0.5841 | -0.7185 | -0.4635 | 0.014 | 0.884 |
| scifact | multi-positive | 716 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0405 | -0.0815 | -0.0024 | 0.043 | 0.116 |
| scifact | multi-positive | 716 | held_recall10 | L2S (in-sample) minus full query | 69 | 0.0652 | 0.0290 | 0.1063 | 0.145 | 0.000 |
| scifact | multi-positive | 716 | held_recall10 | held-out transfer (1-1) minus full query | 69 | 0.0248 | -0.0133 | 0.0676 | 0.101 | 0.029 |
| scifact | multi-positive | 716 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0888 | 0.0459 | 0.1401 | 0.188 | 0.000 |
| scifact | multi-positive | 716 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0145 | -0.0362 | 0.0000 | 0.000 | 0.029 |
| scifact | multi-positive | 716 | held_recall100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 716 | held_recall100 | held-out transfer (1-1) minus full query | 69 | -0.0024 | -0.0290 | 0.0242 | 0.029 | 0.029 |
| scifact | multi-positive | 716 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0217 | 0.0000 | 0.0507 | 0.043 | 0.000 |
| scifact | multi-positive | 819 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.1095 | -0.1435 | -0.0784 | 0.072 | 0.797 |
| scifact | multi-positive | 819 | held_rr | L2S (in-sample) minus full query | 69 | 0.0777 | 0.0519 | 0.1051 | 0.681 | 0.000 |
| scifact | multi-positive | 819 | held_rr | held-out transfer (1-1) minus full query | 69 | -0.0318 | -0.0610 | -0.0038 | 0.304 | 0.536 |
| scifact | multi-positive | 819 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.1833 | 0.1460 | 0.2252 | 0.870 | 0.000 |
| scifact | multi-positive | 819 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 69 | 0.2667 | 0.1912 | 0.3485 | 0.797 | 0.087 |
| scifact | multi-positive | 819 | held_logrank | L2S (in-sample) minus full query | 69 | -0.3048 | -0.3986 | -0.2242 | 0.000 | 0.681 |
| scifact | multi-positive | 819 | held_logrank | held-out transfer (1-1) minus full query | 69 | -0.0381 | -0.1214 | 0.0360 | 0.478 | 0.391 |
| scifact | multi-positive | 819 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | -0.5024 | -0.6244 | -0.3919 | 0.014 | 0.870 |
| scifact | multi-positive | 819 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0236 | -0.0580 | 0.0091 | 0.043 | 0.072 |
| scifact | multi-positive | 819 | held_recall10 | L2S (in-sample) minus full query | 69 | 0.0580 | 0.0242 | 0.0966 | 0.130 | 0.000 |
| scifact | multi-positive | 819 | held_recall10 | held-out transfer (1-1) minus full query | 69 | 0.0344 | -0.0018 | 0.0755 | 0.101 | 0.014 |
| scifact | multi-positive | 819 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0791 | 0.0380 | 0.1280 | 0.159 | 0.000 |
| scifact | multi-positive | 819 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0097 | -0.0266 | 0.0000 | 0.000 | 0.029 |
| scifact | multi-positive | 819 | held_recall100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 819 | held_recall100 | held-out transfer (1-1) minus full query | 69 | 0.0024 | -0.0193 | 0.0242 | 0.029 | 0.014 |
| scifact | multi-positive | 819 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0169 | 0.0000 | 0.0386 | 0.043 | 0.000 |
| scifact | multi-positive | 921 | held_rr | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0677 | -0.0930 | -0.0444 | 0.116 | 0.710 |
| scifact | multi-positive | 921 | held_rr | L2S (in-sample) minus full query | 69 | 0.0441 | 0.0267 | 0.0636 | 0.580 | 0.029 |
| scifact | multi-positive | 921 | held_rr | held-out transfer (1-1) minus full query | 69 | -0.0237 | -0.0490 | 0.0007 | 0.348 | 0.406 |
| scifact | multi-positive | 921 | held_rr | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.1321 | 0.1008 | 0.1657 | 0.841 | 0.000 |
| scifact | multi-positive | 921 | held_logrank | held-out transfer (1-1) minus L2S (in-sample) | 69 | 0.1654 | 0.0967 | 0.2301 | 0.710 | 0.130 |
| scifact | multi-positive | 921 | held_logrank | L2S (in-sample) minus full query | 69 | -0.1979 | -0.2697 | -0.1378 | 0.029 | 0.580 |
| scifact | multi-positive | 921 | held_logrank | held-out transfer (1-1) minus full query | 69 | -0.0326 | -0.1076 | 0.0304 | 0.391 | 0.406 |
| scifact | multi-positive | 921 | held_logrank | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | -0.3838 | -0.4754 | -0.2969 | 0.014 | 0.841 |
| scifact | multi-positive | 921 | held_recall10 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0278 | -0.0580 | -0.0024 | 0.014 | 0.072 |
| scifact | multi-positive | 921 | held_recall10 | L2S (in-sample) minus full query | 69 | 0.0507 | 0.0193 | 0.0870 | 0.116 | 0.000 |
| scifact | multi-positive | 921 | held_recall10 | held-out transfer (1-1) minus full query | 69 | 0.0229 | -0.0072 | 0.0519 | 0.087 | 0.014 |
| scifact | multi-positive | 921 | held_recall10 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0857 | 0.0435 | 0.1377 | 0.174 | 0.000 |
| scifact | multi-positive | 921 | held_recall100 | held-out transfer (1-1) minus L2S (in-sample) | 69 | -0.0024 | -0.0072 | 0.0000 | 0.000 | 0.014 |
| scifact | multi-positive | 921 | held_recall100 | L2S (in-sample) minus full query | 69 | 0.0121 | 0.0000 | 0.0314 | 0.029 | 0.000 |
| scifact | multi-positive | 921 | held_recall100 | held-out transfer (1-1) minus full query | 69 | 0.0097 | 0.0000 | 0.0290 | 0.029 | 0.000 |
| scifact | multi-positive | 921 | held_recall100 | all-positive oracle (privileged) minus held-out transfer (1-1) | 69 | 0.0024 | 0.0000 | 0.0072 | 0.014 | 0.000 |

## Positive ranks, and how scores moved against the full query

| environment | cohort | k | arm | rr | logrank | recall10 | recall100 | positive score change | at rank 1 change | at rank 10 change | at rank 100 change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact | single-positive | 102 | prefix-k | 0.5320 (0.4983 to 0.5659) | 1.6202 (1.4679 to 1.7724) | 0.6935 (0.6571 to 0.7299) | 0.8832 (0.8589 to 0.9059) | -0.5557 (-0.5638 to -0.5475) | -0.5826 (-0.5888 to -0.5764) | -0.4360 (-0.4398 to -0.4324) | -0.3547 (-0.3578 to -0.3513) |
| scifact | single-positive | 102 | all-positive oracle (privileged) | 0.9887 (0.9823 to 0.9943) | 0.0174 (0.0086 to 0.0280) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.2907 (-0.2975 to -0.2838) | -0.3255 (-0.3303 to -0.3206) | -0.2744 (-0.2772 to -0.2717) | -0.2220 (-0.2243 to -0.2195) |
| scifact | single-positive | 102 | L2S (in-sample) | 0.9645 (0.9528 to 0.9746) | 0.0593 (0.0416 to 0.0796) | 0.9985 (0.9954 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.3856 (-0.3927 to -0.3787) | -0.4200 (-0.4252 to -0.4148) | -0.3438 (-0.3470 to -0.3406) | -0.2873 (-0.2903 to -0.2842) |
| scifact | single-positive | 102 | pair-optimal, target only (secondary) | 0.9887 (0.9823 to 0.9943) | 0.0174 (0.0086 to 0.0280) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.2907 (-0.2975 to -0.2838) | -0.3255 (-0.3303 to -0.3206) | -0.2744 (-0.2772 to -0.2717) | -0.2220 (-0.2243 to -0.2195) |
| scifact | single-positive | 204 | prefix-k | 0.6047 (0.5715 to 0.6382) | 1.2990 (1.1617 to 1.4383) | 0.7557 (0.7223 to 0.7876) | 0.9211 (0.8998 to 0.9408) | -0.5055 (-0.5129 to -0.4980) | -0.5304 (-0.5361 to -0.5247) | -0.3954 (-0.3990 to -0.3919) | -0.3204 (-0.3233 to -0.3173) |
| scifact | single-positive | 204 | all-positive oracle (privileged) | 0.9903 (0.9842 to 0.9954) | 0.0153 (0.0069 to 0.0254) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1349 (-0.1404 to -0.1293) | -0.1694 (-0.1733 to -0.1653) | -0.1694 (-0.1714 to -0.1673) | -0.1362 (-0.1380 to -0.1344) |
| scifact | single-positive | 204 | L2S (in-sample) | 0.9719 (0.9614 to 0.9814) | 0.0465 (0.0304 to 0.0646) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.2712 (-0.2772 to -0.2653) | -0.3053 (-0.3101 to -0.3005) | -0.2663 (-0.2691 to -0.2634) | -0.2257 (-0.2284 to -0.2230) |
| scifact | single-positive | 204 | pair-optimal, target only (secondary) | 0.9903 (0.9842 to 0.9954) | 0.0153 (0.0069 to 0.0254) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1349 (-0.1404 to -0.1293) | -0.1694 (-0.1733 to -0.1653) | -0.1694 (-0.1714 to -0.1673) | -0.1362 (-0.1380 to -0.1344) |
| scifact | single-positive | 307 | prefix-k | 0.6390 (0.6065 to 0.6715) | 1.1339 (1.0099 to 1.2658) | 0.7860 (0.7541 to 0.8164) | 0.9347 (0.9150 to 0.9530) | -0.4454 (-0.4522 to -0.4386) | -0.4679 (-0.4730 to -0.4627) | -0.3464 (-0.3496 to -0.3433) | -0.2795 (-0.2822 to -0.2769) |
| scifact | single-positive | 307 | all-positive oracle (privileged) | 0.9856 (0.9788 to 0.9920) | 0.0221 (0.0116 to 0.0326) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0424 (-0.0469 to -0.0378) | -0.0767 (-0.0802 to -0.0730) | -0.1048 (-0.1064 to -0.1030) | -0.0837 (-0.0851 to -0.0823) |
| scifact | single-positive | 307 | L2S (in-sample) | 0.9631 (0.9512 to 0.9736) | 0.0609 (0.0428 to 0.0813) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1949 (-0.2002 to -0.1898) | -0.2285 (-0.2328 to -0.2242) | -0.2118 (-0.2143 to -0.2093) | -0.1818 (-0.1841 to -0.1795) |
| scifact | single-positive | 307 | pair-optimal, target only (secondary) | 0.9856 (0.9788 to 0.9920) | 0.0221 (0.0116 to 0.0326) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0424 (-0.0469 to -0.0378) | -0.0767 (-0.0802 to -0.0730) | -0.1048 (-0.1064 to -0.1030) | -0.0837 (-0.0851 to -0.0823) |
| scifact | single-positive | 409 | prefix-k | 0.6615 (0.6293 to 0.6938) | 1.0306 (0.9098 to 1.1547) | 0.8134 (0.7830 to 0.8422) | 0.9545 (0.9393 to 0.9697) | -0.3734 (-0.3793 to -0.3675) | -0.3927 (-0.3971 to -0.3883) | -0.2890 (-0.2920 to -0.2862) | -0.2324 (-0.2348 to -0.2299) |
| scifact | single-positive | 409 | all-positive oracle (privileged) | 0.9823 (0.9747 to 0.9894) | 0.0269 (0.0158 to 0.0385) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0125 (0.0087 to 0.0164) | -0.0215 (-0.0246 to -0.0181) | -0.0643 (-0.0657 to -0.0628) | -0.0508 (-0.0519 to -0.0496) |
| scifact | single-positive | 409 | L2S (in-sample) | 0.9551 (0.9420 to 0.9667) | 0.0761 (0.0550 to 0.0992) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1399 (-0.1445 to -0.1355) | -0.1727 (-0.1766 to -0.1688) | -0.1702 (-0.1723 to -0.1679) | -0.1482 (-0.1502 to -0.1462) |
| scifact | single-positive | 409 | pair-optimal, target only (secondary) | 0.9823 (0.9747 to 0.9894) | 0.0269 (0.0158 to 0.0385) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0125 (0.0087 to 0.0164) | -0.0215 (-0.0246 to -0.0181) | -0.0643 (-0.0657 to -0.0628) | -0.0508 (-0.0519 to -0.0496) |
| scifact | single-positive | 512 | prefix-k | 0.6756 (0.6434 to 0.7074) | 0.9806 (0.8632 to 1.1040) | 0.8194 (0.7891 to 0.8483) | 0.9575 (0.9423 to 0.9727) | -0.3024 (-0.3073 to -0.2976) | -0.3188 (-0.3226 to -0.3150) | -0.2317 (-0.2341 to -0.2295) | -0.1848 (-0.1866 to -0.1829) |
| scifact | single-positive | 512 | all-positive oracle (privileged) | 0.9799 (0.9716 to 0.9877) | 0.0315 (0.0190 to 0.0449) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0441 (0.0408 to 0.0474) | 0.0106 (0.0075 to 0.0139) | -0.0388 (-0.0402 to -0.0374) | -0.0302 (-0.0312 to -0.0291) |
| scifact | single-positive | 512 | L2S (in-sample) | 0.9360 (0.9205 to 0.9504) | 0.1116 (0.0851 to 0.1400) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0975 (-0.1014 to -0.0938) | -0.1290 (-0.1324 to -0.1255) | -0.1353 (-0.1372 to -0.1333) | -0.1198 (-0.1215 to -0.1181) |
| scifact | single-positive | 512 | pair-optimal, target only (secondary) | 0.9799 (0.9716 to 0.9877) | 0.0315 (0.0190 to 0.0449) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0441 (0.0408 to 0.0474) | 0.0106 (0.0075 to 0.0139) | -0.0388 (-0.0402 to -0.0374) | -0.0302 (-0.0312 to -0.0291) |
| scifact | single-positive | 614 | prefix-k | 0.6890 (0.6575 to 0.7207) | 0.9207 (0.8056 to 1.0395) | 0.8392 (0.8103 to 0.8665) | 0.9590 (0.9439 to 0.9742) | -0.2390 (-0.2431 to -0.2351) | -0.2525 (-0.2556 to -0.2494) | -0.1830 (-0.1850 to -0.1811) | -0.1457 (-0.1471 to -0.1442) |
| scifact | single-positive | 614 | all-positive oracle (privileged) | 0.9717 (0.9618 to 0.9812) | 0.0436 (0.0285 to 0.0591) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0596 (0.0567 to 0.0625) | 0.0264 (0.0234 to 0.0295) | -0.0242 (-0.0255 to -0.0229) | -0.0184 (-0.0194 to -0.0174) |
| scifact | single-positive | 614 | L2S (in-sample) | 0.9144 (0.8964 to 0.9312) | 0.1595 (0.1237 to 0.1962) | 0.9909 (0.9833 to 0.9970) | 1.0000 (1.0000 to 1.0000) | -0.0649 (-0.0681 to -0.0618) | -0.0947 (-0.0977 to -0.0916) | -0.1060 (-0.1076 to -0.1042) | -0.0951 (-0.0966 to -0.0937) |
| scifact | single-positive | 614 | pair-optimal, target only (secondary) | 0.9717 (0.9618 to 0.9812) | 0.0436 (0.0285 to 0.0591) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0596 (0.0567 to 0.0625) | 0.0264 (0.0234 to 0.0295) | -0.0242 (-0.0255 to -0.0229) | -0.0184 (-0.0194 to -0.0174) |
| scifact | single-positive | 716 | prefix-k | 0.6963 (0.6656 to 0.7270) | 0.8945 (0.7834 to 1.0104) | 0.8437 (0.8149 to 0.8710) | 0.9636 (0.9484 to 0.9772) | -0.1797 (-0.1829 to -0.1765) | -0.1899 (-0.1924 to -0.1874) | -0.1374 (-0.1389 to -0.1360) | -0.1088 (-0.1100 to -0.1077) |
| scifact | single-positive | 716 | all-positive oracle (privileged) | 0.9617 (0.9503 to 0.9726) | 0.0598 (0.0420 to 0.0786) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0647 (0.0621 to 0.0673) | 0.0320 (0.0290 to 0.0350) | -0.0169 (-0.0182 to -0.0156) | -0.0125 (-0.0134 to -0.0115) |
| scifact | single-positive | 716 | L2S (in-sample) | 0.8815 (0.8603 to 0.9014) | 0.2354 (0.1893 to 0.2818) | 0.9757 (0.9651 to 0.9863) | 0.9985 (0.9954 to 1.0000) | -0.0396 (-0.0422 to -0.0371) | -0.0666 (-0.0692 to -0.0639) | -0.0799 (-0.0813 to -0.0784) | -0.0729 (-0.0741 to -0.0716) |
| scifact | single-positive | 716 | pair-optimal, target only (secondary) | 0.9617 (0.9503 to 0.9726) | 0.0598 (0.0420 to 0.0786) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0647 (0.0621 to 0.0673) | 0.0320 (0.0290 to 0.0350) | -0.0169 (-0.0182 to -0.0156) | -0.0125 (-0.0134 to -0.0115) |
| scifact | single-positive | 819 | prefix-k | 0.6973 (0.6664 to 0.7277) | 0.8932 (0.7809 to 1.0093) | 0.8407 (0.8133 to 0.8680) | 0.9621 (0.9469 to 0.9757) | -0.1146 (-0.1168 to -0.1123) | -0.1212 (-0.1230 to -0.1194) | -0.0865 (-0.0876 to -0.0854) | -0.0673 (-0.0681 to -0.0665) |
| scifact | single-positive | 819 | all-positive oracle (privileged) | 0.9490 (0.9359 to 0.9612) | 0.0864 (0.0635 to 0.1103) | 0.9970 (0.9924 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0628 (0.0605 to 0.0652) | 0.0314 (0.0286 to 0.0342) | -0.0123 (-0.0135 to -0.0111) | -0.0088 (-0.0096 to -0.0079) |
| scifact | single-positive | 819 | L2S (in-sample) | 0.8449 (0.8204 to 0.8679) | 0.3366 (0.2774 to 0.3970) | 0.9621 (0.9484 to 0.9757) | 0.9970 (0.9924 to 1.0000) | -0.0207 (-0.0225 to -0.0188) | -0.0432 (-0.0454 to -0.0411) | -0.0557 (-0.0568 to -0.0545) | -0.0519 (-0.0529 to -0.0509) |
| scifact | single-positive | 819 | pair-optimal, target only (secondary) | 0.9490 (0.9359 to 0.9612) | 0.0864 (0.0635 to 0.1103) | 0.9970 (0.9924 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0628 (0.0605 to 0.0652) | 0.0314 (0.0286 to 0.0342) | -0.0123 (-0.0135 to -0.0111) | -0.0088 (-0.0096 to -0.0079) |
| scifact | single-positive | 921 | prefix-k | 0.7069 (0.6761 to 0.7368) | 0.8673 (0.7566 to 0.9838) | 0.8483 (0.8209 to 0.8741) | 0.9575 (0.9423 to 0.9727) | -0.0523 (-0.0535 to -0.0510) | -0.0558 (-0.0568 to -0.0547) | -0.0381 (-0.0388 to -0.0375) | -0.0284 (-0.0289 to -0.0279) |
| scifact | single-positive | 921 | all-positive oracle (privileged) | 0.8885 (0.8683 to 0.9071) | 0.2180 (0.1765 to 0.2627) | 0.9863 (0.9772 to 0.9939) | 0.9985 (0.9954 to 1.0000) | 0.0506 (0.0489 to 0.0523) | 0.0240 (0.0216 to 0.0263) | -0.0072 (-0.0081 to -0.0064) | -0.0048 (-0.0055 to -0.0042) |
| scifact | single-positive | 921 | L2S (in-sample) | 0.7905 (0.7629 to 0.8180) | 0.5140 (0.4388 to 0.5930) | 0.9181 (0.8968 to 0.9378) | 0.9939 (0.9879 to 0.9985) | -0.0070 (-0.0082 to -0.0059) | -0.0226 (-0.0240 to -0.0212) | -0.0319 (-0.0326 to -0.0311) | -0.0306 (-0.0313 to -0.0299) |
| scifact | single-positive | 921 | pair-optimal, target only (secondary) | 0.8885 (0.8683 to 0.9071) | 0.2180 (0.1765 to 0.2627) | 0.9863 (0.9772 to 0.9939) | 0.9985 (0.9954 to 1.0000) | 0.0506 (0.0489 to 0.0523) | 0.0240 (0.0216 to 0.0263) | -0.0072 (-0.0081 to -0.0064) | -0.0048 (-0.0055 to -0.0042) |
| scifact | multi-positive | 102 | prefix-k | 0.3427 (0.2915 to 0.3963) | 1.8903 (1.6001 to 2.2059) | 0.6693 (0.5886 to 0.7505) | 0.9251 (0.8671 to 0.9758) | -0.5627 (-0.5788 to -0.5463) | -0.6191 (-0.6318 to -0.6055) | -0.4833 (-0.4941 to -0.4726) | -0.3726 (-0.3825 to -0.3625) |
| scifact | multi-positive | 102 | all-positive oracle (privileged) | 0.6508 (0.6154 to 0.6836) | 0.5754 (0.5014 to 0.6572) | 0.9952 (0.9855 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.3094 (-0.3218 to -0.2967) | -0.3641 (-0.3740 to -0.3535) | -0.2922 (-0.2996 to -0.2844) | -0.2245 (-0.2312 to -0.2176) |
| scifact | multi-positive | 102 | L2S (in-sample) | 0.6163 (0.5776 to 0.6497) | 0.6986 (0.6063 to 0.8057) | 0.9807 (0.9565 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.4025 (-0.4149 to -0.3895) | -0.4585 (-0.4697 to -0.4473) | -0.3701 (-0.3791 to -0.3607) | -0.2951 (-0.3033 to -0.2864) |
| scifact | multi-positive | 102 | pair-optimal, target only (secondary) | 0.9660 (0.9436 to 0.9849) | 0.0480 (0.0209 to 0.0798) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.2949 (-0.3082 to -0.2814) | n/a | n/a | n/a |
| scifact | multi-positive | 204 | prefix-k | 0.3967 (0.3431 to 0.4494) | 1.6277 (1.3674 to 1.9263) | 0.7490 (0.6696 to 0.8237) | 0.9565 (0.9058 to 0.9928) | -0.5122 (-0.5269 to -0.4970) | -0.5641 (-0.5755 to -0.5515) | -0.4393 (-0.4495 to -0.4292) | -0.3367 (-0.3463 to -0.3272) |
| scifact | multi-positive | 204 | all-positive oracle (privileged) | 0.6491 (0.6153 to 0.6792) | 0.5892 (0.5154 to 0.6692) | 0.9928 (0.9783 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1590 (-0.1687 to -0.1492) | -0.2069 (-0.2151 to -0.1984) | -0.1729 (-0.1788 to -0.1672) | -0.1319 (-0.1368 to -0.1272) |
| scifact | multi-positive | 204 | L2S (in-sample) | 0.6271 (0.5890 to 0.6613) | 0.6747 (0.5791 to 0.7849) | 0.9734 (0.9444 to 0.9952) | 1.0000 (1.0000 to 1.0000) | -0.2921 (-0.3032 to -0.2812) | -0.3414 (-0.3512 to -0.3314) | -0.2833 (-0.2915 to -0.2747) | -0.2292 (-0.2360 to -0.2221) |
| scifact | multi-positive | 204 | pair-optimal, target only (secondary) | 0.9696 (0.9489 to 0.9871) | 0.0430 (0.0184 to 0.0729) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.1391 (-0.1496 to -0.1284) | n/a | n/a | n/a |
| scifact | multi-positive | 307 | prefix-k | 0.4109 (0.3596 to 0.4635) | 1.5210 (1.2874 to 1.7943) | 0.7640 (0.6884 to 0.8321) | 0.9638 (0.9130 to 1.0000) | -0.4514 (-0.4656 to -0.4377) | -0.4975 (-0.5087 to -0.4856) | -0.3858 (-0.3952 to -0.3763) | -0.2933 (-0.3015 to -0.2852) |
| scifact | multi-positive | 307 | all-positive oracle (privileged) | 0.6367 (0.6025 to 0.6672) | 0.6261 (0.5517 to 0.7088) | 0.9855 (0.9638 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0681 (-0.0759 to -0.0602) | -0.1112 (-0.1185 to -0.1038) | -0.0994 (-0.1039 to -0.0948) | -0.0764 (-0.0800 to -0.0730) |
| scifact | multi-positive | 307 | L2S (in-sample) | 0.6154 (0.5784 to 0.6492) | 0.7247 (0.6252 to 0.8373) | 0.9589 (0.9251 to 0.9855) | 0.9928 (0.9783 to 1.0000) | -0.2169 (-0.2264 to -0.2076) | -0.2617 (-0.2711 to -0.2524) | -0.2212 (-0.2288 to -0.2137) | -0.1826 (-0.1885 to -0.1767) |
| scifact | multi-positive | 307 | pair-optimal, target only (secondary) | 0.9630 (0.9412 to 0.9819) | 0.0522 (0.0251 to 0.0829) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | -0.0465 (-0.0551 to -0.0376) | n/a | n/a | n/a |
| scifact | multi-positive | 409 | prefix-k | 0.4232 (0.3713 to 0.4750) | 1.4327 (1.2098 to 1.6826) | 0.7664 (0.6843 to 0.8384) | 0.9855 (0.9638 to 1.0000) | -0.3809 (-0.3933 to -0.3685) | -0.4210 (-0.4308 to -0.4104) | -0.3244 (-0.3332 to -0.3157) | -0.2452 (-0.2529 to -0.2376) |
| scifact | multi-positive | 409 | all-positive oracle (privileged) | 0.6302 (0.5963 to 0.6608) | 0.6477 (0.5715 to 0.7327) | 0.9734 (0.9444 to 0.9952) | 1.0000 (1.0000 to 1.0000) | -0.0135 (-0.0199 to -0.0067) | -0.0519 (-0.0585 to -0.0454) | -0.0535 (-0.0574 to -0.0497) | -0.0413 (-0.0439 to -0.0387) |
| scifact | multi-positive | 409 | L2S (in-sample) | 0.5950 (0.5543 to 0.6315) | 0.7819 (0.6736 to 0.9038) | 0.9324 (0.8889 to 0.9662) | 0.9928 (0.9783 to 1.0000) | -0.1631 (-0.1715 to -0.1547) | -0.2033 (-0.2118 to -0.1946) | -0.1764 (-0.1828 to -0.1699) | -0.1479 (-0.1528 to -0.1428) |
| scifact | multi-positive | 409 | pair-optimal, target only (secondary) | 0.9235 (0.8925 to 0.9511) | 0.1090 (0.0691 to 0.1542) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0087 (0.0015 to 0.0164) | n/a | n/a | n/a |
| scifact | multi-positive | 512 | prefix-k | 0.4240 (0.3697 to 0.4786) | 1.4156 (1.1922 to 1.6614) | 0.7563 (0.6790 to 0.8261) | 0.9855 (0.9638 to 1.0000) | -0.3048 (-0.3150 to -0.2947) | -0.3373 (-0.3457 to -0.3287) | -0.2584 (-0.2659 to -0.2512) | -0.1935 (-0.1995 to -0.1878) |
| scifact | multi-positive | 512 | all-positive oracle (privileged) | 0.6245 (0.5907 to 0.6560) | 0.6660 (0.5891 to 0.7534) | 0.9662 (0.9348 to 0.9928) | 1.0000 (1.0000 to 1.0000) | 0.0187 (0.0132 to 0.0246) | -0.0168 (-0.0231 to -0.0108) | -0.0239 (-0.0273 to -0.0207) | -0.0181 (-0.0204 to -0.0159) |
| scifact | multi-positive | 512 | L2S (in-sample) | 0.5827 (0.5398 to 0.6217) | 0.8205 (0.7040 to 0.9520) | 0.9186 (0.8746 to 0.9565) | 0.9928 (0.9783 to 1.0000) | -0.1204 (-0.1272 to -0.1134) | -0.1553 (-0.1627 to -0.1478) | -0.1385 (-0.1436 to -0.1330) | -0.1184 (-0.1225 to -0.1143) |
| scifact | multi-positive | 512 | pair-optimal, target only (secondary) | 0.8952 (0.8598 to 0.9306) | 0.1547 (0.1013 to 0.2097) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0406 (0.0341 to 0.0473) | n/a | n/a | n/a |
| scifact | multi-positive | 614 | prefix-k | 0.4246 (0.3716 to 0.4797) | 1.3758 (1.1609 to 1.6091) | 0.8058 (0.7348 to 0.8722) | 0.9807 (0.9565 to 1.0000) | -0.2434 (-0.2520 to -0.2348) | -0.2702 (-0.2775 to -0.2626) | -0.2066 (-0.2129 to -0.2004) | -0.1542 (-0.1591 to -0.1494) |
| scifact | multi-positive | 614 | all-positive oracle (privileged) | 0.6145 (0.5783 to 0.6483) | 0.6926 (0.6078 to 0.7877) | 0.9517 (0.9155 to 0.9807) | 1.0000 (1.0000 to 1.0000) | 0.0353 (0.0306 to 0.0404) | 0.0024 (-0.0034 to 0.0081) | -0.0080 (-0.0111 to -0.0050) | -0.0058 (-0.0077 to -0.0039) |
| scifact | multi-positive | 614 | L2S (in-sample) | 0.5632 (0.5160 to 0.6075) | 0.8680 (0.7431 to 1.0069) | 0.9041 (0.8558 to 0.9457) | 0.9928 (0.9783 to 1.0000) | -0.0858 (-0.0911 to -0.0804) | -0.1162 (-0.1227 to -0.1097) | -0.1070 (-0.1113 to -0.1026) | -0.0931 (-0.0963 to -0.0897) |
| scifact | multi-positive | 614 | pair-optimal, target only (secondary) | 0.8742 (0.8357 to 0.9107) | 0.1897 (0.1337 to 0.2514) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0561 (0.0504 to 0.0621) | n/a | n/a | n/a |
| scifact | multi-positive | 716 | prefix-k | 0.4360 (0.3811 to 0.4907) | 1.3533 (1.1381 to 1.5914) | 0.8155 (0.7382 to 0.8870) | 0.9855 (0.9638 to 1.0000) | -0.1825 (-0.1894 to -0.1758) | -0.2030 (-0.2090 to -0.1969) | -0.1543 (-0.1595 to -0.1489) | -0.1149 (-0.1187 to -0.1113) |
| scifact | multi-positive | 716 | all-positive oracle (privileged) | 0.6034 (0.5653 to 0.6391) | 0.7298 (0.6351 to 0.8318) | 0.9444 (0.9057 to 0.9783) | 1.0000 (1.0000 to 1.0000) | 0.0417 (0.0375 to 0.0463) | 0.0107 (0.0049 to 0.0163) | -0.0009 (-0.0040 to 0.0020) | 0.0002 (-0.0016 to 0.0020) |
| scifact | multi-positive | 716 | L2S (in-sample) | 0.5420 (0.4918 to 0.5886) | 0.9299 (0.7909 to 1.0860) | 0.8940 (0.8406 to 0.9408) | 0.9928 (0.9783 to 1.0000) | -0.0576 (-0.0619 to -0.0533) | -0.0824 (-0.0874 to -0.0774) | -0.0790 (-0.0826 to -0.0755) | -0.0709 (-0.0736 to -0.0684) |
| scifact | multi-positive | 716 | pair-optimal, target only (secondary) | 0.8555 (0.8161 to 0.8956) | 0.2157 (0.1552 to 0.2776) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0613 (0.0561 to 0.0668) | n/a | n/a | n/a |
| scifact | multi-positive | 819 | prefix-k | 0.4357 (0.3829 to 0.4883) | 1.3372 (1.1253 to 1.5735) | 0.8022 (0.7213 to 0.8761) | 0.9807 (0.9565 to 1.0000) | -0.1167 (-0.1216 to -0.1117) | -0.1315 (-0.1356 to -0.1272) | -0.0974 (-0.1011 to -0.0938) | -0.0713 (-0.0737 to -0.0689) |
| scifact | multi-positive | 819 | all-positive oracle (privileged) | 0.5895 (0.5494 to 0.6274) | 0.7730 (0.6667 to 0.8885) | 0.9444 (0.9057 to 0.9783) | 1.0000 (1.0000 to 1.0000) | 0.0418 (0.0381 to 0.0460) | 0.0126 (0.0071 to 0.0180) | 0.0019 (-0.0008 to 0.0046) | 0.0023 (0.0007 to 0.0040) |
| scifact | multi-positive | 819 | L2S (in-sample) | 0.5155 (0.4653 to 0.5639) | 1.0095 (0.8591 to 1.1766) | 0.8867 (0.8275 to 0.9384) | 0.9928 (0.9783 to 1.0000) | -0.0348 (-0.0378 to -0.0316) | -0.0541 (-0.0578 to -0.0505) | -0.0532 (-0.0561 to -0.0503) | -0.0494 (-0.0517 to -0.0471) |
| scifact | multi-positive | 819 | pair-optimal, target only (secondary) | 0.8003 (0.7566 to 0.8419) | 0.3152 (0.2448 to 0.3911) | 1.0000 (1.0000 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0598 (0.0551 to 0.0647) | n/a | n/a | n/a |
| scifact | multi-positive | 921 | prefix-k | 0.4321 (0.3806 to 0.4832) | 1.3264 (1.1138 to 1.5611) | 0.8203 (0.7471 to 0.8877) | 0.9807 (0.9565 to 1.0000) | -0.0535 (-0.0559 to -0.0510) | -0.0611 (-0.0636 to -0.0586) | -0.0440 (-0.0463 to -0.0417) | -0.0304 (-0.0319 to -0.0290) |
| scifact | multi-positive | 921 | all-positive oracle (privileged) | 0.5463 (0.4991 to 0.5896) | 0.8974 (0.7689 to 1.0418) | 0.9396 (0.9009 to 0.9734) | 0.9928 (0.9783 to 1.0000) | 0.0349 (0.0321 to 0.0379) | 0.0109 (0.0062 to 0.0155) | 0.0027 (0.0005 to 0.0049) | 0.0033 (0.0020 to 0.0046) |
| scifact | multi-positive | 921 | L2S (in-sample) | 0.4821 (0.4316 to 0.5312) | 1.1161 (0.9510 to 1.3026) | 0.8795 (0.8142 to 0.9338) | 0.9928 (0.9783 to 1.0000) | -0.0169 (-0.0189 to -0.0148) | -0.0288 (-0.0314 to -0.0263) | -0.0298 (-0.0319 to -0.0278) | -0.0285 (-0.0301 to -0.0270) |
| scifact | multi-positive | 921 | pair-optimal, target only (secondary) | 0.7083 (0.6575 to 0.7545) | 0.5092 (0.4140 to 0.6207) | 0.9783 (0.9493 to 1.0000) | 1.0000 (1.0000 to 1.0000) | 0.0485 (0.0451 to 0.0520) | n/a | n/a | n/a |

## Agreement with the oracle mask

| environment | cohort | k | arm | Jaccard with oracle mask | score regret to ceiling / mass |
|---|---|---|---|---|---|
| scifact | single-positive | 102 | prefix-k | 0.0849 (0.0837 to 0.0861) | 0.3371 (0.3357 to 0.3385) |
| scifact | single-positive | 102 | L2S (in-sample) | 0.3833 (0.3777 to 0.3891) | 0.1214 (0.1188 to 0.1238) |
| scifact | single-positive | 204 | prefix-k | 0.1121 (0.1110 to 0.1132) | 0.4714 (0.4698 to 0.4729) |
| scifact | single-positive | 204 | L2S (in-sample) | 0.4052 (0.4001 to 0.4104) | 0.1742 (0.1709 to 0.1773) |
| scifact | single-positive | 307 | prefix-k | 0.1676 (0.1664 to 0.1688) | 0.5126 (0.5108 to 0.5144) |
| scifact | single-positive | 307 | L2S (in-sample) | 0.4289 (0.4247 to 0.4333) | 0.1948 (0.1916 to 0.1979) |
| scifact | single-positive | 409 | prefix-k | 0.2449 (0.2437 to 0.2462) | 0.4911 (0.4893 to 0.4929) |
| scifact | single-positive | 409 | L2S (in-sample) | 0.4605 (0.4571 to 0.4639) | 0.1947 (0.1917 to 0.1976) |
| scifact | single-positive | 512 | prefix-k | 0.3329 (0.3318 to 0.3339) | 0.4412 (0.4396 to 0.4429) |
| scifact | single-positive | 512 | L2S (in-sample) | 0.5012 (0.4985 to 0.5041) | 0.1810 (0.1784 to 0.1836) |
| scifact | single-positive | 614 | prefix-k | 0.4315 (0.4305 to 0.4325) | 0.3805 (0.3788 to 0.3823) |
| scifact | single-positive | 614 | L2S (in-sample) | 0.5542 (0.5520 to 0.5563) | 0.1593 (0.1569 to 0.1617) |
| scifact | single-positive | 716 | prefix-k | 0.5424 (0.5416 to 0.5432) | 0.3120 (0.3100 to 0.3139) |
| scifact | single-positive | 716 | L2S (in-sample) | 0.6228 (0.6212 to 0.6245) | 0.1339 (0.1317 to 0.1361) |
| scifact | single-positive | 819 | prefix-k | 0.6717 (0.6709 to 0.6723) | 0.2272 (0.2250 to 0.2293) |
| scifact | single-positive | 819 | L2S (in-sample) | 0.7167 (0.7154 to 0.7181) | 0.1076 (0.1057 to 0.1096) |
| scifact | single-positive | 921 | prefix-k | 0.8202 (0.8197 to 0.8206) | 0.1321 (0.1304 to 0.1339) |
| scifact | single-positive | 921 | L2S (in-sample) | 0.8402 (0.8393 to 0.8413) | 0.0744 (0.0729 to 0.0761) |
| scifact | single-positive | 1024 | prefix-k | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| scifact | single-positive | 1024 | L2S (in-sample) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| scifact | multi-positive | 102 | prefix-k | 0.0856 (0.0822 to 0.0893) | 0.3215 (0.3161 to 0.3268) |
| scifact | multi-positive | 102 | L2S (in-sample) | 0.3705 (0.3558 to 0.3872) | 0.1187 (0.1118 to 0.1251) |
| scifact | multi-positive | 102 | held-out transfer (1-1) | 0.7050 (0.6911 to 0.7184) | n/a |
| scifact | multi-positive | 204 | prefix-k | 0.1130 (0.1092 to 0.1166) | 0.4482 (0.4420 to 0.4540) |
| scifact | multi-positive | 204 | L2S (in-sample) | 0.3936 (0.3822 to 0.4060) | 0.1695 (0.1618 to 0.1764) |
| scifact | multi-positive | 204 | held-out transfer (1-1) | 0.7404 (0.7288 to 0.7515) | n/a |
| scifact | multi-positive | 307 | prefix-k | 0.1676 (0.1642 to 0.1710) | 0.4863 (0.4799 to 0.4924) |
| scifact | multi-positive | 307 | L2S (in-sample) | 0.4210 (0.4100 to 0.4320) | 0.1894 (0.1822 to 0.1966) |
| scifact | multi-positive | 307 | held-out transfer (1-1) | 0.7652 (0.7544 to 0.7758) | n/a |
| scifact | multi-positive | 409 | prefix-k | 0.2428 (0.2393 to 0.2462) | 0.4663 (0.4596 to 0.4728) |
| scifact | multi-positive | 409 | L2S (in-sample) | 0.4541 (0.4452 to 0.4629) | 0.1904 (0.1836 to 0.1972) |
| scifact | multi-positive | 409 | held-out transfer (1-1) | 0.7879 (0.7776 to 0.7980) | n/a |
| scifact | multi-positive | 512 | prefix-k | 0.3338 (0.3304 to 0.3371) | 0.4107 (0.4039 to 0.4175) |
| scifact | multi-positive | 512 | L2S (in-sample) | 0.4928 (0.4867 to 0.4994) | 0.1771 (0.1710 to 0.1827) |
| scifact | multi-positive | 512 | held-out transfer (1-1) | 0.8023 (0.7926 to 0.8111) | n/a |
| scifact | multi-positive | 614 | prefix-k | 0.4296 (0.4270 to 0.4323) | 0.3539 (0.3478 to 0.3600) |
| scifact | multi-positive | 614 | L2S (in-sample) | 0.5462 (0.5415 to 0.5514) | 0.1542 (0.1491 to 0.1593) |
| scifact | multi-positive | 614 | held-out transfer (1-1) | 0.8168 (0.8080 to 0.8254) | n/a |
| scifact | multi-positive | 716 | prefix-k | 0.5434 (0.5411 to 0.5457) | 0.2850 (0.2790 to 0.2914) |
| scifact | multi-positive | 716 | L2S (in-sample) | 0.6143 (0.6111 to 0.6177) | 0.1266 (0.1221 to 0.1311) |
| scifact | multi-positive | 716 | held-out transfer (1-1) | 0.8280 (0.8200 to 0.8361) | n/a |
| scifact | multi-positive | 819 | prefix-k | 0.6709 (0.6690 to 0.6727) | 0.2017 (0.1965 to 0.2072) |
| scifact | multi-positive | 819 | L2S (in-sample) | 0.7088 (0.7060 to 0.7115) | 0.0979 (0.0942 to 0.1017) |
| scifact | multi-positive | 819 | held-out transfer (1-1) | 0.8609 (0.8549 to 0.8665) | n/a |
| scifact | multi-positive | 921 | prefix-k | 0.8190 (0.8180 to 0.8200) | 0.1126 (0.1090 to 0.1167) |
| scifact | multi-positive | 921 | L2S (in-sample) | 0.8344 (0.8324 to 0.8363) | 0.0662 (0.0635 to 0.0690) |
| scifact | multi-positive | 921 | held-out transfer (1-1) | 0.9227 (0.9192 to 0.9261) | n/a |
| scifact | multi-positive | 1024 | prefix-k | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| scifact | multi-positive | 1024 | L2S (in-sample) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) |
| scifact | multi-positive | 1024 | held-out transfer (1-1) | 1.0000 (1.0000 to 1.0000) | n/a |

## Relationship among a query's positives (multi-positive)

`pair_jaccard` is the mean overlap of pair-optimal masks. `sign_consensus` is the share of the oracle mask's coordinates on which all positives agree in sign, `positive_consensus` the share on which all are positive. `oracle_regret` is the regret of the all-positive mask against each positive's own optimum, `held_regret` of a mask built from a disjoint positive, each over mass.

| environment | k | queries | pair_jaccard | sign_consensus | positive_consensus | oracle_regret | held_regret | held_privileged_regret | held_oracle_jaccard |
|---|---|---|---|---|---|---|---|---|---|
| scifact | 102 | 69 | 0.5122 (0.4914 to 0.5335) | 0.9943 (0.9916 to 0.9967) | 0.9943 (0.9916 to 0.9967) | 0.0187 (0.0167 to 0.0209) | 0.0655 (0.0586 to 0.0726) | 0.0187 (0.0166 to 0.0209) | 0.7050 (0.6911 to 0.7184) |
| scifact | 204 | 69 | 0.5585 (0.5402 to 0.5765) | 0.9829 (0.9773 to 0.9884) | 0.9829 (0.9773 to 0.9884) | 0.0257 (0.0230 to 0.0285) | 0.0897 (0.0804 to 0.0994) | 0.0257 (0.0230 to 0.0285) | 0.7404 (0.7288 to 0.7515) |
| scifact | 307 | 69 | 0.5969 (0.5793 to 0.6142) | 0.9674 (0.9578 to 0.9758) | 0.9674 (0.9578 to 0.9758) | 0.0279 (0.0249 to 0.0310) | 0.0999 (0.0891 to 0.1108) | 0.0279 (0.0249 to 0.0310) | 0.7652 (0.7544 to 0.7758) |
| scifact | 409 | 69 | 0.6320 (0.6157 to 0.6484) | 0.9441 (0.9319 to 0.9558) | 0.9441 (0.9319 to 0.9558) | 0.0286 (0.0255 to 0.0319) | 0.1015 (0.0903 to 0.1127) | 0.0287 (0.0256 to 0.0319) | 0.7879 (0.7776 to 0.7980) |
| scifact | 512 | 69 | 0.6554 (0.6405 to 0.6702) | 0.9096 (0.8945 to 0.9241) | 0.9096 (0.8945 to 0.9241) | 0.0281 (0.0251 to 0.0313) | 0.0999 (0.0887 to 0.1114) | 0.0281 (0.0251 to 0.0312) | 0.8023 (0.7926 to 0.8111) |
| scifact | 614 | 69 | 0.6787 (0.6652 to 0.6929) | 0.8657 (0.8482 to 0.8817) | 0.8657 (0.8482 to 0.8817) | 0.0268 (0.0239 to 0.0298) | 0.0956 (0.0847 to 0.1067) | 0.0269 (0.0239 to 0.0298) | 0.8168 (0.8080 to 0.8254) |
| scifact | 716 | 69 | 0.7011 (0.6885 to 0.7141) | 0.8110 (0.7927 to 0.8285) | 0.8105 (0.7922 to 0.8283) | 0.0253 (0.0225 to 0.0282) | 0.0911 (0.0803 to 0.1017) | 0.0254 (0.0225 to 0.0282) | 0.8280 (0.8200 to 0.8361) |
| scifact | 819 | 69 | 0.7649 (0.7566 to 0.7731) | 0.7480 (0.7302 to 0.7649) | 0.7299 (0.7125 to 0.7477) | 0.0232 (0.0206 to 0.0258) | 0.0837 (0.0741 to 0.0933) | 0.0232 (0.0207 to 0.0258) | 0.8609 (0.8549 to 0.8665) |
| scifact | 921 | 69 | 0.8676 (0.8628 to 0.8724) | 0.7176 (0.6989 to 0.7354) | 0.6494 (0.6338 to 0.6655) | 0.0176 (0.0157 to 0.0195) | 0.0632 (0.0563 to 0.0701) | 0.0176 (0.0157 to 0.0196) | 0.9227 (0.9192 to 0.9261) |
| scifact | 1024 | 69 | 1.0000 (1.0000 to 1.0000) | 0.7170 (0.6976 to 0.7356) | 0.5841 (0.5701 to 0.5985) | 0.0000 (0.0000 to 0.0000) | 0.0000 (0.0000 to 0.0000) | 0.0000 (0.0000 to 0.0000) | 1.0000 (1.0000 to 1.0000) |

## Checks and cost

| environment | check | value |
|---|---|---|
| scifact | boundary_note | "L2S ties at a mask boundary go to the lower dimension" |
| scifact | float32_evaluation_gap | {"largest_ndcg10_gap": 0.0, "query_budget_pairs_above_1e-6": 0} |
| scifact | ids_colliding_with_a_document | 0 |
| scifact | k1024_rank_mismatches | 0 |
| scifact | l2s_boundary_ties | 18 |
| scifact | masks | 26418 |
| scifact | oracle_ceiling_violations | 0 |
| scifact | ranks_out_of_bounds | 0 |
| scifact | single_positive_masks_identical_to_oracle | 659 |
| scifact | single_positive_queries | 659 |
| scifact | stored_run_parity | {"compared": 3309, "mismatched": 0, "run": "20261109T163931Z-9e9f5a0a"} |
| scifact | tied_tracked_documents | 0 |
| scifact | total seconds | 4.6 |
| scifact | device | cpu |
