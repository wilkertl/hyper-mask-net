# P1: what the corpus's geometry alone does

Protocol: in-domain. Scoring: exact inner product over the full corpus, hard top-k mask.
`variance` keeps the k dimensions with the largest `q_d² · Var(x_d)`, with the variance
from document embeddings only (no judgment); it is paired per query with the S1 run's
selectors (seeds averaged) and baselines.

- probe-geometry `20261015T150050Z-8401cd55` (commit `a820eddffc50`), S1 compare-selectors `20261015T135520Z-3d0fd03e`

## Readings

| reading | value | rule | holds |
|---|---|---|---|
| SC-001 variance recovers >= half of L2S's gain over global importance | 2 of 3 environments | >= 2 of 3 | yes |
| SC-002 participation ratio orders L2S's best budget | nfcorpus 90.9 (819) < scifact 98.8 (409) < msmarco 223.9 (1024) | same order (descriptive, three environments) | no |

## Variance against the selectors and global importance (mean nDCG@10 over 10%-90%)

Holm over the environments, separately for each reference.

| test | against | variance | reference | diff | ci_low | ci_high | p | p_holm | note |
|---|---|---|---|---|---|---|---|---|---|
| msmarco | L2S | 0.3760 | 0.3607 | +0.0154 | +0.0131 | +0.0178 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | L2S | 0.3587 | 0.3622 | -0.0034 | -0.0137 | +0.0067 | 0.5238 | 0.5238 | see note: nfcorpus |
| scifact | L2S | 0.6964 | 0.8166 | -0.1203 | -0.1517 | -0.0902 | 0.0001 | 0.0003 | — |
| msmarco | S1 | 0.3760 | 0.3658 | +0.0102 | +0.0083 | +0.0122 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | S1 | 0.3587 | 0.3622 | -0.0034 | -0.0137 | +0.0067 | 0.5238 | 0.5238 | see note: nfcorpus |
| scifact | S1 | 0.6964 | 0.8282 | -0.1319 | -0.1657 | -0.1001 | 0.0001 | 0.0003 | — |
| msmarco | global | 0.3760 | 0.3516 | +0.0244 | +0.0222 | +0.0266 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | global | 0.3587 | 0.3336 | +0.0252 | +0.0170 | +0.0337 | 0.0001 | 0.0003 | see note: nfcorpus |
| scifact | global | 0.6964 | 0.6821 | +0.0142 | +0.0056 | +0.0228 | 0.0012 | 0.0012 | — |

| test | variance | L2S | global importance | share of L2S's gain | recovers half |
|---|---|---|---|---|---|
| msmarco | 0.3760 | 0.3607 | 0.3516 | +2.7088 | yes |
| nfcorpus | 0.3587 | 0.3622 | 0.3336 | +0.8800 | yes |
| scifact | 0.6964 | 0.8166 | 0.6821 | +0.1058 | no |

## Effective dimensionality and the best budget

Participation ratio `(Σλ)² / Σλ²` of the document covariance; energy in the mean
`‖μ‖² / E‖x‖²`. Three environments describe; they do not test a relation.

| environment | documents | participation ratio | share of 1024 | energy in mean | best budget L2S | best budget S1 | best budget variance |
|---|---|---|---|---|---|---|---|
| msmarco | 8841823 | 223.9 | 0.2187 | 0.1635 | 1024 | 1024 | 614 |
| nfcorpus | 3633 | 90.9 | 0.0887 | 0.3081 | 819 | 819 | 716 |
| scifact | 5183 | 98.8 | 0.0965 | 0.2847 | 409 | 409 | 819 |

## nDCG@10 by budget

| test | query set | method | 10% (k=102) | 20% (k=204) | 30% (k=307) | 40% (k=409) | 50% (k=512) | 60% (k=614) | 70% (k=716) | 80% (k=819) | 90% (k=921) | 100% (k=1024) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | test | full | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 |
| msmarco | test | prefix | 0.2598 | 0.3180 | 0.3385 | 0.3528 | 0.3626 | 0.3702 | 0.3746 | 0.3777 | 0.3792 | 0.3800 |
| msmarco | test | random | 0.2397 | 0.3186 | 0.3425 | 0.3587 | 0.3643 | 0.3695 | 0.3746 | 0.3758 | 0.3778 | 0.3800 |
| msmarco | test | global-importance | 0.2435 | 0.3230 | 0.3524 | 0.3639 | 0.3714 | 0.3737 | 0.3774 | 0.3797 | 0.3798 | 0.3800 |
| msmarco | test | variance | 0.3584 | 0.3717 | 0.3765 | 0.3783 | 0.3791 | 0.3806 | 0.3792 | 0.3802 | 0.3804 | 0.3800 |
| msmarco | test | l2s:msmarco | 0.3047 | 0.3444 | 0.3572 | 0.3640 | 0.3702 | 0.3718 | 0.3752 | 0.3785 | 0.3799 | 0.3800 |
| msmarco | test | retrieval:msmarco | 0.3185 | 0.3516 | 0.3645 | 0.3704 | 0.3745 | 0.3762 | 0.3784 | 0.3790 | 0.3796 | 0.3800 |
| msmarco | trec-dl-2019 | full | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 |
| msmarco | trec-dl-2019 | prefix | 0.5240 | 0.5803 | 0.5990 | 0.6270 | 0.6327 | 0.6466 | 0.6609 | 0.6620 | 0.6671 | 0.6654 |
| msmarco | trec-dl-2019 | random | 0.4374 | 0.5820 | 0.6367 | 0.6472 | 0.6611 | 0.6559 | 0.6553 | 0.6563 | 0.6600 | 0.6654 |
| msmarco | trec-dl-2019 | global-importance | 0.4592 | 0.5850 | 0.6165 | 0.6387 | 0.6522 | 0.6664 | 0.6583 | 0.6606 | 0.6589 | 0.6654 |
| msmarco | trec-dl-2019 | variance | 0.6273 | 0.6451 | 0.6540 | 0.6605 | 0.6628 | 0.6671 | 0.6687 | 0.6680 | 0.6659 | 0.6654 |
| msmarco | trec-dl-2019 | l2s:msmarco | 0.5829 | 0.6342 | 0.6348 | 0.6482 | 0.6574 | 0.6636 | 0.6638 | 0.6696 | 0.6685 | 0.6654 |
| msmarco | trec-dl-2019 | retrieval:msmarco | 0.5736 | 0.6359 | 0.6349 | 0.6458 | 0.6690 | 0.6636 | 0.6649 | 0.6611 | 0.6627 | 0.6654 |
| msmarco | trec-dl-2020 | full | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 |
| msmarco | trec-dl-2020 | prefix | 0.4900 | 0.5717 | 0.6234 | 0.6497 | 0.6525 | 0.6595 | 0.6616 | 0.6591 | 0.6678 | 0.6767 |
| msmarco | trec-dl-2020 | random | 0.4936 | 0.6049 | 0.6478 | 0.6658 | 0.6668 | 0.6785 | 0.6732 | 0.6736 | 0.6687 | 0.6767 |
| msmarco | trec-dl-2020 | global-importance | 0.4598 | 0.5977 | 0.6450 | 0.6580 | 0.6697 | 0.6714 | 0.6742 | 0.6747 | 0.6767 | 0.6767 |
| msmarco | trec-dl-2020 | variance | 0.6488 | 0.6673 | 0.6738 | 0.6723 | 0.6702 | 0.6732 | 0.6751 | 0.6733 | 0.6756 | 0.6767 |
| msmarco | trec-dl-2020 | l2s:msmarco | 0.5939 | 0.6347 | 0.6534 | 0.6550 | 0.6590 | 0.6680 | 0.6721 | 0.6778 | 0.6803 | 0.6767 |
| msmarco | trec-dl-2020 | retrieval:msmarco | 0.6058 | 0.6497 | 0.6522 | 0.6604 | 0.6650 | 0.6712 | 0.6733 | 0.6728 | 0.6791 | 0.6767 |
| msmarco | trec-dl-hard | full | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 |
| msmarco | trec-dl-hard | prefix | 0.2813 | 0.2999 | 0.3363 | 0.3404 | 0.3435 | 0.3508 | 0.3586 | 0.3600 | 0.3707 | 0.3681 |
| msmarco | trec-dl-hard | random | 0.2088 | 0.2979 | 0.3385 | 0.3441 | 0.3543 | 0.3655 | 0.3734 | 0.3726 | 0.3678 | 0.3681 |
| msmarco | trec-dl-hard | global-importance | 0.2490 | 0.3157 | 0.3420 | 0.3469 | 0.3563 | 0.3610 | 0.3689 | 0.3704 | 0.3681 | 0.3681 |
| msmarco | trec-dl-hard | variance | 0.3558 | 0.3724 | 0.3697 | 0.3700 | 0.3709 | 0.3698 | 0.3690 | 0.3692 | 0.3683 | 0.3681 |
| msmarco | trec-dl-hard | l2s:msmarco | 0.3252 | 0.3484 | 0.3509 | 0.3594 | 0.3772 | 0.3803 | 0.3781 | 0.3835 | 0.3729 | 0.3681 |
| msmarco | trec-dl-hard | retrieval:msmarco | 0.3009 | 0.3412 | 0.3478 | 0.3615 | 0.3663 | 0.3677 | 0.3750 | 0.3706 | 0.3724 | 0.3681 |
| nfcorpus | test | full | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 |
| nfcorpus | test | prefix | 0.2716 | 0.3165 | 0.3263 | 0.3394 | 0.3459 | 0.3492 | 0.3555 | 0.3609 | 0.3617 | 0.3619 |
| nfcorpus | test | random | 0.2400 | 0.2987 | 0.3183 | 0.3345 | 0.3488 | 0.3546 | 0.3582 | 0.3614 | 0.3633 | 0.3619 |
| nfcorpus | test | global-importance | 0.2471 | 0.3091 | 0.3358 | 0.3371 | 0.3501 | 0.3511 | 0.3540 | 0.3559 | 0.3620 | 0.3619 |
| nfcorpus | test | variance | 0.3443 | 0.3560 | 0.3596 | 0.3590 | 0.3603 | 0.3629 | 0.3630 | 0.3621 | 0.3615 | 0.3619 |
| nfcorpus | test | l2s:nfcorpus | 0.3132 | 0.3480 | 0.3607 | 0.3682 | 0.3736 | 0.3747 | 0.3751 | 0.3751 | 0.3709 | 0.3619 |
| nfcorpus | test | retrieval:nfcorpus | 0.3132 | 0.3480 | 0.3607 | 0.3682 | 0.3736 | 0.3747 | 0.3751 | 0.3751 | 0.3709 | 0.3619 |
| scifact | test | full | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 |
| scifact | test | prefix | 0.5520 | 0.6387 | 0.6640 | 0.6672 | 0.6895 | 0.6915 | 0.6921 | 0.6937 | 0.6974 | 0.7000 |
| scifact | test | random | 0.5285 | 0.6379 | 0.6556 | 0.6708 | 0.6838 | 0.6914 | 0.6871 | 0.6926 | 0.7012 | 0.7000 |
| scifact | test | global-importance | 0.5559 | 0.6671 | 0.6922 | 0.6988 | 0.7049 | 0.7055 | 0.7090 | 0.7037 | 0.7019 | 0.7000 |
| scifact | test | variance | 0.6721 | 0.7011 | 0.6973 | 0.6975 | 0.6971 | 0.6978 | 0.7008 | 0.7027 | 0.7006 | 0.7000 |
| scifact | test | l2s:scifact | 0.8021 | 0.8270 | 0.8353 | 0.8369 | 0.8312 | 0.8247 | 0.8142 | 0.8033 | 0.7749 | 0.7000 |
| scifact | test | retrieval:scifact | 0.8060 | 0.8410 | 0.8458 | 0.8473 | 0.8453 | 0.8374 | 0.8301 | 0.8171 | 0.7842 | 0.7000 |
