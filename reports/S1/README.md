# S1: a selector trained for retrieval against Learning-to-Select

Protocol: in-domain. Scoring: exact inner product over the full corpus, hard top-k mask.
Budgets: 102, 204, 307, 409, 512, 614, 716, 819, 921 (10%-90%); 100% is the full query.
Selectors: the retrieval grid's selected cell per environment (msmarco `fixed-307-scratch-lr0.001-ts0.05-b0.1`, nfcorpus `fixed-307-l2s-lr0.0001-ts0.02-b0.1`, scifact `multi-budget-l2s-lr0.0001-ts0.02-b0.02`) and E1's
`wide-grid` L2S selectors, seeds 0, 1, 2; seed scores
are averaged per query before every paired test (F5 research D7).

- compare-selectors `20261015T135520Z-3d0fd03e` (commit `2768d95ab22a`)

## Criteria

| criterion | value | threshold | met |
|---|---|---|---|
| SC-001 better than L2S (mean nDCG@10 over 10%-90%, Holm, alpha 0.05) | 2 of 3 environments | >= 2 of 3 | yes |
| SC-002 worse than L2S in no environment (same test) | 0 of 3 environments | 0 | yes |

## Against L2S on the test sets

Holm over the environments, separately for each summary.

| test | query set | summary | L2S | retrieval | diff | ci_low | ci_high | p | p_holm | note |
|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | test | mean 10%-90% | 0.3607 | 0.3658 | +0.0052 | +0.0037 | +0.0067 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | mean 10%-90% | 0.3622 | 0.3622 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | 1.0000 | see note: nfcorpus |
| scifact | test | mean 10%-90% | 0.8166 | 0.8282 | +0.0116 | +0.0053 | +0.0182 | 0.0006 | 0.0012 | — |
| msmarco | test | k = 307 | 0.3572 | 0.3645 | +0.0073 | +0.0042 | +0.0105 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | k = 307 | 0.3607 | 0.3607 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | 1.0000 | see note: nfcorpus |
| scifact | test | k = 307 | 0.8353 | 0.8458 | +0.0104 | +0.0012 | +0.0196 | 0.0285 | 0.0570 | — |

## nDCG@10 by budget

| test | query set | method | 10% (k=102) | 20% (k=204) | 30% (k=307) | 40% (k=409) | 50% (k=512) | 60% (k=614) | 70% (k=716) | 80% (k=819) | 90% (k=921) | 100% (k=1024) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | test | full | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | — |
| msmarco | test | prefix | 0.2598 | 0.3180 | 0.3385 | 0.3528 | 0.3626 | 0.3702 | 0.3746 | 0.3777 | 0.3792 | 0.3800 | see note: msmarco |
| msmarco | test | random | 0.2397 | 0.3186 | 0.3425 | 0.3587 | 0.3643 | 0.3695 | 0.3746 | 0.3758 | 0.3778 | 0.3800 | see note: msmarco |
| msmarco | test | global-importance | 0.2435 | 0.3230 | 0.3524 | 0.3639 | 0.3714 | 0.3737 | 0.3774 | 0.3797 | 0.3798 | 0.3800 | see note: msmarco |
| msmarco | test | global-selector | 0.3064 | 0.3460 | 0.3591 | 0.3680 | 0.3742 | 0.3757 | 0.3767 | 0.3786 | 0.3797 | 0.3800 | see note: msmarco |
| msmarco | test | l2s:msmarco | 0.3047 | 0.3444 | 0.3572 | 0.3640 | 0.3702 | 0.3718 | 0.3752 | 0.3785 | 0.3799 | 0.3800 | see note: msmarco |
| msmarco | test | retrieval:msmarco | 0.3185 | 0.3516 | 0.3645 | 0.3704 | 0.3745 | 0.3762 | 0.3784 | 0.3790 | 0.3796 | 0.3800 | see note: msmarco |
| msmarco | test | family:fixed-307-l2s:msmarco | 0.3203 | 0.3522 | 0.3642 | 0.3703 | 0.3756 | 0.3769 | 0.3786 | 0.3793 | 0.3797 | 0.3800 | see note: msmarco |
| msmarco | test | family:fixed-307-scratch:msmarco | 0.3189 | 0.3526 | 0.3653 | 0.3709 | 0.3745 | 0.3758 | 0.3781 | 0.3794 | 0.3789 | 0.3800 | see note: msmarco |
| msmarco | test | family:multi-budget-l2s:msmarco | 0.3112 | 0.3484 | 0.3592 | 0.3667 | 0.3719 | 0.3758 | 0.3778 | 0.3798 | 0.3791 | 0.3800 | see note: msmarco |
| msmarco | test | family:multi-budget-scratch:msmarco | 0.3114 | 0.3465 | 0.3604 | 0.3687 | 0.3725 | 0.3752 | 0.3781 | 0.3787 | 0.3797 | 0.3800 | see note: msmarco |
| msmarco | trec-dl-2019 | full | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | 0.6654 | — |
| msmarco | trec-dl-2019 | prefix | 0.5240 | 0.5803 | 0.5990 | 0.6270 | 0.6327 | 0.6466 | 0.6609 | 0.6620 | 0.6671 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | random | 0.4374 | 0.5820 | 0.6367 | 0.6472 | 0.6611 | 0.6559 | 0.6553 | 0.6563 | 0.6600 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | global-importance | 0.4592 | 0.5850 | 0.6165 | 0.6387 | 0.6522 | 0.6664 | 0.6583 | 0.6606 | 0.6589 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | global-selector | 0.5762 | 0.6410 | 0.6495 | 0.6472 | 0.6511 | 0.6648 | 0.6645 | 0.6651 | 0.6632 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | l2s:msmarco | 0.5829 | 0.6342 | 0.6348 | 0.6482 | 0.6574 | 0.6636 | 0.6638 | 0.6696 | 0.6685 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | retrieval:msmarco | 0.5736 | 0.6359 | 0.6349 | 0.6458 | 0.6690 | 0.6636 | 0.6649 | 0.6611 | 0.6627 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | family:fixed-307-l2s:msmarco | 0.5602 | 0.6344 | 0.6345 | 0.6537 | 0.6596 | 0.6652 | 0.6639 | 0.6642 | 0.6651 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | family:fixed-307-scratch:msmarco | 0.5755 | 0.6371 | 0.6340 | 0.6467 | 0.6651 | 0.6610 | 0.6636 | 0.6615 | 0.6663 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | family:multi-budget-l2s:msmarco | 0.5564 | 0.6273 | 0.6413 | 0.6563 | 0.6752 | 0.6664 | 0.6754 | 0.6706 | 0.6630 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | family:multi-budget-scratch:msmarco | 0.5516 | 0.6372 | 0.6425 | 0.6566 | 0.6728 | 0.6731 | 0.6731 | 0.6730 | 0.6647 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2020 | full | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | 0.6767 | — |
| msmarco | trec-dl-2020 | prefix | 0.4900 | 0.5717 | 0.6234 | 0.6497 | 0.6525 | 0.6595 | 0.6616 | 0.6591 | 0.6678 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | random | 0.4936 | 0.6049 | 0.6478 | 0.6658 | 0.6668 | 0.6785 | 0.6732 | 0.6736 | 0.6687 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | global-importance | 0.4598 | 0.5977 | 0.6450 | 0.6580 | 0.6697 | 0.6714 | 0.6742 | 0.6747 | 0.6767 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | global-selector | 0.5698 | 0.6370 | 0.6507 | 0.6566 | 0.6661 | 0.6734 | 0.6757 | 0.6715 | 0.6769 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | l2s:msmarco | 0.5939 | 0.6347 | 0.6534 | 0.6550 | 0.6590 | 0.6680 | 0.6721 | 0.6778 | 0.6803 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | retrieval:msmarco | 0.6058 | 0.6497 | 0.6522 | 0.6604 | 0.6650 | 0.6712 | 0.6733 | 0.6728 | 0.6791 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | family:fixed-307-l2s:msmarco | 0.6079 | 0.6510 | 0.6589 | 0.6624 | 0.6709 | 0.6767 | 0.6695 | 0.6728 | 0.6790 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | family:fixed-307-scratch:msmarco | 0.6046 | 0.6525 | 0.6508 | 0.6648 | 0.6715 | 0.6751 | 0.6708 | 0.6742 | 0.6789 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | family:multi-budget-l2s:msmarco | 0.5986 | 0.6501 | 0.6540 | 0.6626 | 0.6597 | 0.6679 | 0.6772 | 0.6783 | 0.6813 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | family:multi-budget-scratch:msmarco | 0.6036 | 0.6530 | 0.6447 | 0.6551 | 0.6671 | 0.6713 | 0.6771 | 0.6755 | 0.6780 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-hard | full | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | 0.3681 | — |
| msmarco | trec-dl-hard | prefix | 0.2813 | 0.2999 | 0.3363 | 0.3404 | 0.3435 | 0.3508 | 0.3586 | 0.3600 | 0.3707 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | random | 0.2088 | 0.2979 | 0.3385 | 0.3441 | 0.3543 | 0.3655 | 0.3734 | 0.3726 | 0.3678 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | global-importance | 0.2490 | 0.3157 | 0.3420 | 0.3469 | 0.3563 | 0.3610 | 0.3689 | 0.3704 | 0.3681 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | global-selector | 0.3146 | 0.3368 | 0.3462 | 0.3477 | 0.3557 | 0.3611 | 0.3577 | 0.3686 | 0.3700 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | l2s:msmarco | 0.3252 | 0.3484 | 0.3509 | 0.3594 | 0.3772 | 0.3803 | 0.3781 | 0.3835 | 0.3729 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | retrieval:msmarco | 0.3009 | 0.3412 | 0.3478 | 0.3615 | 0.3663 | 0.3677 | 0.3750 | 0.3706 | 0.3724 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | family:fixed-307-l2s:msmarco | 0.3129 | 0.3498 | 0.3497 | 0.3649 | 0.3594 | 0.3613 | 0.3728 | 0.3675 | 0.3761 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | family:fixed-307-scratch:msmarco | 0.3129 | 0.3423 | 0.3504 | 0.3640 | 0.3611 | 0.3692 | 0.3691 | 0.3666 | 0.3767 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | family:multi-budget-l2s:msmarco | 0.3083 | 0.3408 | 0.3489 | 0.3538 | 0.3680 | 0.3706 | 0.3676 | 0.3719 | 0.3710 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | family:multi-budget-scratch:msmarco | 0.3097 | 0.3362 | 0.3369 | 0.3537 | 0.3687 | 0.3687 | 0.3701 | 0.3705 | 0.3753 | 0.3681 | see note: msmarco |
| nfcorpus | test | full | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | 0.3619 | — |
| nfcorpus | test | prefix | 0.2716 | 0.3165 | 0.3263 | 0.3394 | 0.3459 | 0.3492 | 0.3555 | 0.3609 | 0.3617 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | random | 0.2400 | 0.2987 | 0.3183 | 0.3345 | 0.3488 | 0.3546 | 0.3582 | 0.3614 | 0.3633 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | global-importance | 0.2471 | 0.3091 | 0.3358 | 0.3371 | 0.3501 | 0.3511 | 0.3540 | 0.3559 | 0.3620 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | global-selector | 0.3075 | 0.3304 | 0.3436 | 0.3495 | 0.3550 | 0.3555 | 0.3581 | 0.3572 | 0.3590 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | l2s:nfcorpus | 0.3132 | 0.3480 | 0.3607 | 0.3682 | 0.3736 | 0.3747 | 0.3751 | 0.3751 | 0.3709 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | retrieval:nfcorpus | 0.3132 | 0.3480 | 0.3607 | 0.3682 | 0.3736 | 0.3747 | 0.3751 | 0.3751 | 0.3709 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | family:fixed-307-l2s:nfcorpus | 0.3067 | 0.3503 | 0.3599 | 0.3684 | 0.3749 | 0.3772 | 0.3747 | 0.3759 | 0.3684 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | family:fixed-307-scratch:nfcorpus | 0.3159 | 0.3574 | 0.3657 | 0.3719 | 0.3758 | 0.3739 | 0.3756 | 0.3715 | 0.3701 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | family:multi-budget-l2s:nfcorpus | 0.3067 | 0.3503 | 0.3599 | 0.3684 | 0.3749 | 0.3772 | 0.3747 | 0.3759 | 0.3684 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | family:multi-budget-scratch:nfcorpus | 0.3109 | 0.3551 | 0.3669 | 0.3728 | 0.3793 | 0.3813 | 0.3824 | 0.3720 | 0.3710 | 0.3619 | see note: nfcorpus |
| scifact | test | full | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | — |
| scifact | test | prefix | 0.5520 | 0.6387 | 0.6640 | 0.6672 | 0.6895 | 0.6915 | 0.6921 | 0.6937 | 0.6974 | 0.7000 | — |
| scifact | test | random | 0.5285 | 0.6379 | 0.6556 | 0.6708 | 0.6838 | 0.6914 | 0.6871 | 0.6926 | 0.7012 | 0.7000 | — |
| scifact | test | global-importance | 0.5559 | 0.6671 | 0.6922 | 0.6988 | 0.7049 | 0.7055 | 0.7090 | 0.7037 | 0.7019 | 0.7000 | — |
| scifact | test | global-selector | 0.6735 | 0.7099 | 0.7238 | 0.7256 | 0.7252 | 0.7146 | 0.7089 | 0.7070 | 0.7061 | 0.7000 | — |
| scifact | test | l2s:scifact | 0.8021 | 0.8270 | 0.8353 | 0.8369 | 0.8312 | 0.8247 | 0.8142 | 0.8033 | 0.7749 | 0.7000 | — |
| scifact | test | retrieval:scifact | 0.8060 | 0.8410 | 0.8458 | 0.8473 | 0.8453 | 0.8374 | 0.8301 | 0.8171 | 0.7842 | 0.7000 | — |
| scifact | test | family:fixed-307-l2s:scifact | 0.8165 | 0.8428 | 0.8428 | 0.8515 | 0.8467 | 0.8388 | 0.8280 | 0.8133 | 0.7688 | 0.7000 | — |
| scifact | test | family:fixed-307-scratch:scifact | 0.8069 | 0.8388 | 0.8576 | 0.8555 | 0.8532 | 0.8485 | 0.8375 | 0.8168 | 0.7789 | 0.7000 | — |
| scifact | test | family:multi-budget-l2s:scifact | 0.7976 | 0.8349 | 0.8478 | 0.8497 | 0.8461 | 0.8369 | 0.8298 | 0.8219 | 0.7818 | 0.7000 | — |
| scifact | test | family:multi-budget-scratch:scifact | 0.8094 | 0.8412 | 0.8471 | 0.8583 | 0.8576 | 0.8473 | 0.8387 | 0.8269 | 0.7852 | 0.7000 | — |

## Against full 1024

| test | query set | budget | full | l2s - full | retrieval - full |
|---|---|---|---|---|---|
| msmarco | test | 102 | 0.3800 | -0.0754 | -0.0616 |
| msmarco | test | 204 | 0.3800 | -0.0356 | -0.0284 |
| msmarco | test | 307 | 0.3800 | -0.0228 | -0.0155 |
| msmarco | test | 409 | 0.3800 | -0.0160 | -0.0097 |
| msmarco | test | 512 | 0.3800 | -0.0099 | -0.0056 |
| msmarco | test | 614 | 0.3800 | -0.0082 | -0.0038 |
| msmarco | test | 716 | 0.3800 | -0.0049 | -0.0017 |
| msmarco | test | 819 | 0.3800 | -0.0016 | -0.0010 |
| msmarco | test | 921 | 0.3800 | -0.0001 | -0.0004 |
| msmarco | trec-dl-2019 | 102 | 0.6654 | -0.0825 | -0.0918 |
| msmarco | trec-dl-2019 | 204 | 0.6654 | -0.0312 | -0.0295 |
| msmarco | trec-dl-2019 | 307 | 0.6654 | -0.0306 | -0.0305 |
| msmarco | trec-dl-2019 | 409 | 0.6654 | -0.0172 | -0.0196 |
| msmarco | trec-dl-2019 | 512 | 0.6654 | -0.0080 | +0.0036 |
| msmarco | trec-dl-2019 | 614 | 0.6654 | -0.0018 | -0.0018 |
| msmarco | trec-dl-2019 | 716 | 0.6654 | -0.0016 | -0.0005 |
| msmarco | trec-dl-2019 | 819 | 0.6654 | +0.0042 | -0.0043 |
| msmarco | trec-dl-2019 | 921 | 0.6654 | +0.0031 | -0.0027 |
| msmarco | trec-dl-2020 | 102 | 0.6767 | -0.0828 | -0.0708 |
| msmarco | trec-dl-2020 | 204 | 0.6767 | -0.0420 | -0.0269 |
| msmarco | trec-dl-2020 | 307 | 0.6767 | -0.0233 | -0.0245 |
| msmarco | trec-dl-2020 | 409 | 0.6767 | -0.0217 | -0.0162 |
| msmarco | trec-dl-2020 | 512 | 0.6767 | -0.0176 | -0.0117 |
| msmarco | trec-dl-2020 | 614 | 0.6767 | -0.0087 | -0.0054 |
| msmarco | trec-dl-2020 | 716 | 0.6767 | -0.0046 | -0.0034 |
| msmarco | trec-dl-2020 | 819 | 0.6767 | +0.0011 | -0.0039 |
| msmarco | trec-dl-2020 | 921 | 0.6767 | +0.0036 | +0.0024 |
| msmarco | trec-dl-hard | 102 | 0.3681 | -0.0429 | -0.0672 |
| msmarco | trec-dl-hard | 204 | 0.3681 | -0.0197 | -0.0269 |
| msmarco | trec-dl-hard | 307 | 0.3681 | -0.0172 | -0.0203 |
| msmarco | trec-dl-hard | 409 | 0.3681 | -0.0087 | -0.0067 |
| msmarco | trec-dl-hard | 512 | 0.3681 | +0.0091 | -0.0018 |
| msmarco | trec-dl-hard | 614 | 0.3681 | +0.0122 | -0.0004 |
| msmarco | trec-dl-hard | 716 | 0.3681 | +0.0100 | +0.0069 |
| msmarco | trec-dl-hard | 819 | 0.3681 | +0.0154 | +0.0024 |
| msmarco | trec-dl-hard | 921 | 0.3681 | +0.0048 | +0.0043 |
| nfcorpus | test | 102 | 0.3619 | -0.0488 | -0.0488 |
| nfcorpus | test | 204 | 0.3619 | -0.0140 | -0.0140 |
| nfcorpus | test | 307 | 0.3619 | -0.0012 | -0.0012 |
| nfcorpus | test | 409 | 0.3619 | +0.0063 | +0.0063 |
| nfcorpus | test | 512 | 0.3619 | +0.0117 | +0.0117 |
| nfcorpus | test | 614 | 0.3619 | +0.0127 | +0.0127 |
| nfcorpus | test | 716 | 0.3619 | +0.0132 | +0.0132 |
| nfcorpus | test | 819 | 0.3619 | +0.0132 | +0.0132 |
| nfcorpus | test | 921 | 0.3619 | +0.0090 | +0.0090 |
| scifact | test | 102 | 0.7000 | +0.1022 | +0.1061 |
| scifact | test | 204 | 0.7000 | +0.1271 | +0.1411 |
| scifact | test | 307 | 0.7000 | +0.1354 | +0.1458 |
| scifact | test | 409 | 0.7000 | +0.1370 | +0.1473 |
| scifact | test | 512 | 0.7000 | +0.1312 | +0.1454 |
| scifact | test | 614 | 0.7000 | +0.1247 | +0.1375 |
| scifact | test | 716 | 0.7000 | +0.1142 | +0.1301 |
| scifact | test | 819 | 0.7000 | +0.1034 | +0.1171 |
| scifact | test | 921 | 0.7000 | +0.0749 | +0.0842 |

## TREC DL sets (context, not in the criteria)

| test | query set | summary | L2S | retrieval | diff | ci_low | ci_high | p | note |
|---|---|---|---|---|---|---|---|---|---|
| msmarco | trec-dl-2019 | mean 10%-90%, ndcg@10 | 0.6470 | 0.6457 | -0.0013 | -0.0117 | +0.0084 | 0.8168 | see note: msmarco |
| msmarco | trec-dl-2019 | mean 10%-90%, ap | 0.4162 | 0.4241 | +0.0079 | -0.0002 | +0.0163 | 0.0666 | see note: msmarco |
| msmarco | trec-dl-2019 | k = 307, ndcg@10 | 0.6348 | 0.6349 | +0.0001 | -0.0195 | +0.0193 | 0.9950 | see note: msmarco |
| msmarco | trec-dl-2019 | k = 307, ap | 0.4051 | 0.4207 | +0.0156 | +0.0004 | +0.0322 | 0.0599 | see note: msmarco |
| msmarco | trec-dl-2020 | mean 10%-90%, ndcg@10 | 0.6549 | 0.6588 | +0.0039 | -0.0052 | +0.0131 | 0.4097 | see note: msmarco |
| msmarco | trec-dl-2020 | mean 10%-90%, ap | 0.4202 | 0.4265 | +0.0062 | -0.0006 | +0.0132 | 0.0946 | see note: msmarco |
| msmarco | trec-dl-2020 | k = 307, ndcg@10 | 0.6534 | 0.6522 | -0.0012 | -0.0189 | +0.0155 | 0.8904 | see note: msmarco |
| msmarco | trec-dl-2020 | k = 307, ap | 0.4098 | 0.4186 | +0.0089 | -0.0062 | +0.0240 | 0.2578 | see note: msmarco |
| msmarco | trec-dl-hard | mean 10%-90%, ndcg@10 | 0.3640 | 0.3559 | -0.0080 | -0.0190 | +0.0028 | 0.1583 | see note: msmarco |
| msmarco | trec-dl-hard | mean 10%-90%, ap | 0.2286 | 0.2246 | -0.0041 | -0.0109 | +0.0028 | 0.2533 | see note: msmarco |
| msmarco | trec-dl-hard | k = 307, ndcg@10 | 0.3509 | 0.3478 | -0.0031 | -0.0219 | +0.0154 | 0.7491 | see note: msmarco |
| msmarco | trec-dl-hard | k = 307, ap | 0.2198 | 0.2181 | -0.0018 | -0.0152 | +0.0129 | 0.8157 | see note: msmarco |

## Families (validation-best cell of each variant and start, seed 0; context)

| test | family | cell | mean 10%-90% | k = 307 |
|---|---|---|---|---|
| msmarco | l2s (3 seeds) | — | 0.3607 | 0.3572 |
| msmarco | fixed-307-l2s | fixed-307-l2s-lr0.0001-ts0.05-b0.1 | 0.3663 | 0.3642 |
| msmarco | fixed-307-scratch | fixed-307-scratch-lr0.001-ts0.05-b0.1 | 0.3660 | 0.3653 |
| msmarco | multi-budget-l2s | multi-budget-l2s-lr0.001-ts0.05-b0.1 | 0.3633 | 0.3592 |
| msmarco | multi-budget-scratch | multi-budget-scratch-lr0.001-ts0.05-b0.1 | 0.3635 | 0.3604 |
| nfcorpus | l2s (3 seeds) | — | 0.3622 | 0.3607 |
| nfcorpus | fixed-307-l2s | fixed-307-l2s-lr0.0001-ts0.02-b0.1 | 0.3618 | 0.3599 |
| nfcorpus | fixed-307-scratch | fixed-307-scratch-lr0.001-ts0.05-b0.1 | 0.3642 | 0.3657 |
| nfcorpus | multi-budget-l2s | multi-budget-l2s-lr0.0001-ts0.02-b0.1 | 0.3618 | 0.3599 |
| nfcorpus | multi-budget-scratch | multi-budget-scratch-lr0.001-ts0.05-b0.02 | 0.3658 | 0.3669 |
| scifact | l2s (3 seeds) | — | 0.8166 | 0.8353 |
| scifact | fixed-307-l2s | fixed-307-l2s-lr0.001-ts0.02-b0.1 | 0.8277 | 0.8428 |
| scifact | fixed-307-scratch | fixed-307-scratch-lr0.001-ts0.02-b0.1 | 0.8326 | 0.8576 |
| scifact | multi-budget-l2s | multi-budget-l2s-lr0.0001-ts0.02-b0.02 | 0.8274 | 0.8478 |
| scifact | multi-budget-scratch | multi-budget-scratch-lr0.001-ts0.02-b0.02 | 0.8346 | 0.8471 |

## Where the difference comes from

Top-k overlap on test queries between the two selectors (seed-matched) and between
seeds of each (noise floor):

| environment | budget | retrieval vs L2S | L2S seeds | retrieval seeds |
|---|---|---|---|---|
| msmarco | 102 | 0.5276 | 0.9389 | 0.8925 |
| msmarco | 204 | 0.6220 | 0.9473 | 0.9197 |
| msmarco | 307 | 0.6760 | 0.9544 | 0.9369 |
| msmarco | 409 | 0.7178 | 0.9606 | 0.9502 |
| msmarco | 512 | 0.7563 | 0.9662 | 0.9607 |
| msmarco | 614 | 0.7948 | 0.9717 | 0.9694 |
| msmarco | 716 | 0.8346 | 0.9774 | 0.9768 |
| msmarco | 819 | 0.8784 | 0.9837 | 0.9838 |
| msmarco | 921 | 0.9301 | 0.9910 | 0.9897 |
| nfcorpus | 102 | 1.0000 | 0.6714 | 0.6714 |
| nfcorpus | 204 | 1.0000 | 0.7135 | 0.7135 |
| nfcorpus | 307 | 1.0000 | 0.7517 | 0.7517 |
| nfcorpus | 409 | 1.0000 | 0.7861 | 0.7861 |
| nfcorpus | 512 | 1.0000 | 0.8201 | 0.8201 |
| nfcorpus | 614 | 1.0000 | 0.8546 | 0.8546 |
| nfcorpus | 716 | 1.0000 | 0.8903 | 0.8903 |
| nfcorpus | 819 | 1.0000 | 0.9259 | 0.9259 |
| nfcorpus | 921 | 1.0000 | 0.9621 | 0.9621 |
| scifact | 102 | 0.6497 | 0.5754 | 0.5622 |
| scifact | 204 | 0.6933 | 0.6119 | 0.6123 |
| scifact | 307 | 0.7286 | 0.6492 | 0.6528 |
| scifact | 409 | 0.7640 | 0.6881 | 0.6957 |
| scifact | 512 | 0.7997 | 0.7324 | 0.7427 |
| scifact | 614 | 0.8355 | 0.7793 | 0.7902 |
| scifact | 716 | 0.8722 | 0.8291 | 0.8392 |
| scifact | 819 | 0.9103 | 0.8827 | 0.8904 |
| scifact | 921 | 0.9503 | 0.9395 | 0.9420 |

Ranking fit on validation queries (top-k agreement with the oracle mask, gain over
global importance, F4 research D2), Holm over environments per selector:

| environment | selector | mean gain over global importance | p | p_holm | fits |
|---|---|---|---|---|---|
| msmarco | l2s | +0.0320 | 0.0001 | 0.0003 | yes |
| msmarco | retrieval | +0.0423 | 0.0001 | 0.0003 | yes |
| nfcorpus | l2s | +0.0278 | 0.0001 | 0.0003 | yes |
| nfcorpus | retrieval | +0.0278 | 0.0001 | 0.0003 | yes |
| scifact | l2s | +0.1401 | 0.0001 | 0.0003 | yes |
| scifact | retrieval | +0.1032 | 0.0001 | 0.0003 | yes |

Training of the seeded retrieval selectors:

| environment | seed | steps | best step | best validation rerank nDCG@10 | final soft mass |
|---|---|---|---|---|---|
| msmarco | 0 | 20000 | 19000 | 0.7032 | 0.9327 |
| msmarco | 1 | 20000 | 15000 | 0.7034 | 0.9327 |
| msmarco | 2 | 20000 | 18000 | 0.7031 | 0.9327 |
| nfcorpus | 0 | 20000 | 0 | 0.6139 | 0.9377 |
| nfcorpus | 1 | 20000 | 0 | 0.6154 | 0.9379 |
| nfcorpus | 2 | 20000 | 0 | 0.6113 | 0.9380 |
| scifact | 0 | 20000 | 1000 | 0.9259 | 0.9868 |
| scifact | 1 | 20000 | 9000 | 0.9263 | 0.9869 |
| scifact | 2 | 20000 | 16000 | 0.9264 | 0.9867 |

## Notes

- **msmarco**: E0 does not reproduce Learning-to-Select on MS MARCO (ADR-0006). E1 scores MS MARCO dev against the full 8.8M-passage corpus; the paper evaluates the `test` split of an unpublished `msmarco_subset_50k` folder, with an ideal DCG taken from the retrieved list (F4 research D9).
- **nfcorpus**: NFCorpus's selector does not fit its oracle distribution as a calibrated distribution (ADR-0006); read its shift with its ranking-fit result (F4 research D2).
- **nfcorpus**: the selected cell starts from the L2S selector and every seed keeps its step-0 checkpoint (no training step improved validation), so `retrieval:nfcorpus` is the L2S selector itself and its difference from L2S is zero by construction.

Every grid cell is in `tables/grid.csv`.
