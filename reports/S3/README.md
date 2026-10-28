# S3: a selector that learns how much to trust the geometric prior

Protocol: in-domain. Scoring: exact inner product over the full corpus, hard top-k mask.
The weighted selector scores `a · log(q_d² · Var(x_d) + ε) + (W q + b)_d`: S2's selector
with one learned scalar `a` starting at 1 (at step 0 it is the `variance` method); it is
trained with S1's retrieval objective and S2's grid rules.
Selected cells: msmarco `fixed-307-weighted-lr0.0001-ts0.02-b0.1`, nfcorpus `multi-budget-weighted-lr0.001-ts0.05-b0.1`, scifact `multi-budget-weighted-lr0.001-ts0.02-b0.1`; seeds 0, 1, 2, averaged
per query before every paired test.

- compare-selectors `20261016T062557Z-8ba7f520` (commit `96fab686c815`)

## Criteria

| criterion | value | threshold | met |
|---|---|---|---|
| SC-001 worse than none of the S1 selector, S2 selector, variance (Holm, 0.05) | none | none | yes |
| SC-002 better than the S1 selector (mean nDCG@10, 10%-90%, Holm, 0.05) | 3 of 3 environments | >= 2 of 3 | yes |

## Against the S1 and S2 selectors, the variance method, and L2S on the test sets

Holm over the environments, separately for each reference and summary.

| test | query set | against | summary | reference | weighted | diff | ci_low | ci_high | p | p_holm | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | test | S1 selector | mean 10%-90% | 0.3658 | 0.3760 | +0.0102 | +0.0083 | +0.0122 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | S1 selector | mean 10%-90% | 0.3622 | 0.3674 | +0.0052 | +0.0002 | +0.0104 | 0.0470 | 0.0470 | see note: nfcorpus |
| scifact | test | S1 selector | mean 10%-90% | 0.8282 | 0.8340 | +0.0057 | +0.0011 | +0.0102 | 0.0157 | 0.0314 | — |
| msmarco | test | S1 selector | k = 307 | 0.3645 | 0.3765 | +0.0120 | +0.0085 | +0.0154 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | S1 selector | k = 307 | 0.3607 | 0.3681 | +0.0074 | -0.0011 | +0.0159 | 0.0875 | 0.1750 | see note: nfcorpus |
| scifact | test | S1 selector | k = 307 | 0.8458 | 0.8517 | +0.0060 | -0.0030 | +0.0153 | 0.2017 | 0.2017 | — |
| msmarco | test | S2 selector | mean 10%-90% | 0.3760 | 0.3760 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | 1.0000 | see note: msmarco |
| nfcorpus | test | S2 selector | mean 10%-90% | 0.3725 | 0.3674 | -0.0051 | -0.0118 | +0.0016 | 0.1337 | 0.2674 | see note: nfcorpus |
| scifact | test | S2 selector | mean 10%-90% | 0.7970 | 0.8340 | +0.0370 | +0.0250 | +0.0493 | 0.0001 | 0.0003 | — |
| msmarco | test | S2 selector | k = 307 | 0.3765 | 0.3765 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | 1.0000 | see note: msmarco |
| nfcorpus | test | S2 selector | k = 307 | 0.3752 | 0.3681 | -0.0071 | -0.0172 | +0.0031 | 0.1675 | 0.3350 | see note: nfcorpus |
| scifact | test | S2 selector | k = 307 | 0.8390 | 0.8517 | +0.0127 | -0.0013 | +0.0270 | 0.0818 | 0.2454 | — |
| msmarco | test | variance | mean 10%-90% | 0.3760 | 0.3760 | +0.0000 | -0.0000 | +0.0000 | 1.0000 | 1.0000 | see note: msmarco |
| nfcorpus | test | variance | mean 10%-90% | 0.3587 | 0.3674 | +0.0086 | -0.0014 | +0.0189 | 0.0934 | 0.1868 | see note: nfcorpus |
| scifact | test | variance | mean 10%-90% | 0.6964 | 0.8340 | +0.1376 | +0.1060 | +0.1712 | 0.0001 | 0.0003 | — |
| msmarco | test | variance | k = 307 | 0.3765 | 0.3765 | +0.0000 | -0.0000 | +0.0000 | 1.0000 | 1.0000 | see note: msmarco |
| nfcorpus | test | variance | k = 307 | 0.3596 | 0.3681 | +0.0085 | -0.0050 | +0.0223 | 0.2278 | 0.4556 | see note: nfcorpus |
| scifact | test | variance | k = 307 | 0.6973 | 0.8517 | +0.1544 | +0.1179 | +0.1930 | 0.0001 | 0.0003 | — |
| msmarco | test | L2S | mean 10%-90% | 0.3607 | 0.3760 | +0.0154 | +0.0131 | +0.0178 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | L2S | mean 10%-90% | 0.3622 | 0.3674 | +0.0052 | +0.0002 | +0.0104 | 0.0470 | 0.0470 | see note: nfcorpus |
| scifact | test | L2S | mean 10%-90% | 0.8166 | 0.8340 | +0.0173 | +0.0097 | +0.0253 | 0.0001 | 0.0003 | — |
| msmarco | test | L2S | k = 307 | 0.3572 | 0.3765 | +0.0193 | +0.0153 | +0.0235 | 0.0001 | 0.0003 | see note: msmarco |
| nfcorpus | test | L2S | k = 307 | 0.3607 | 0.3681 | +0.0074 | -0.0011 | +0.0159 | 0.0875 | 0.0875 | see note: nfcorpus |
| scifact | test | L2S | k = 307 | 0.8353 | 0.8517 | +0.0164 | +0.0039 | +0.0294 | 0.0116 | 0.0232 | — |

## The learned prior weight

| environment | seed | best step | prior weight at best step | final weight |
|---|---|---|---|---|
| msmarco | 0 | 0 | 1.0000 | 0.0162 |
| msmarco | 1 | 0 | 1.0000 | 0.0162 |
| msmarco | 2 | 0 | 1.0000 | 0.0161 |
| nfcorpus | 0 | 7000 | 0.0026 | 0.0008 |
| nfcorpus | 1 | 5000 | 0.0012 | 0.0005 |
| nfcorpus | 2 | 1000 | 0.0106 | 0.0009 |
| scifact | 0 | 17000 | 0.0186 | 0.0183 |
| scifact | 1 | 5000 | 0.0202 | 0.0179 |
| scifact | 2 | 12000 | 0.0182 | 0.0184 |

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
| msmarco | test | family:fixed-307-weighted:msmarco | 0.3584 | 0.3717 | 0.3765 | 0.3783 | 0.3791 | 0.3806 | 0.3792 | 0.3802 | 0.3804 | 0.3800 | see note: msmarco |
| msmarco | test | family:multi-budget-weighted:msmarco | 0.3584 | 0.3717 | 0.3765 | 0.3783 | 0.3791 | 0.3806 | 0.3792 | 0.3802 | 0.3804 | 0.3800 | see note: msmarco |
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
| msmarco | trec-dl-2019 | family:fixed-307-weighted:msmarco | 0.6273 | 0.6451 | 0.6540 | 0.6605 | 0.6628 | 0.6671 | 0.6687 | 0.6680 | 0.6659 | 0.6654 | see note: msmarco |
| msmarco | trec-dl-2019 | family:multi-budget-weighted:msmarco | 0.6273 | 0.6451 | 0.6540 | 0.6605 | 0.6628 | 0.6671 | 0.6687 | 0.6680 | 0.6659 | 0.6654 | see note: msmarco |
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
| msmarco | trec-dl-2020 | family:fixed-307-weighted:msmarco | 0.6488 | 0.6673 | 0.6738 | 0.6723 | 0.6702 | 0.6732 | 0.6751 | 0.6733 | 0.6756 | 0.6767 | see note: msmarco |
| msmarco | trec-dl-2020 | family:multi-budget-weighted:msmarco | 0.6488 | 0.6673 | 0.6738 | 0.6723 | 0.6702 | 0.6732 | 0.6751 | 0.6733 | 0.6756 | 0.6767 | see note: msmarco |
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
| msmarco | trec-dl-hard | family:fixed-307-weighted:msmarco | 0.3558 | 0.3724 | 0.3697 | 0.3700 | 0.3709 | 0.3698 | 0.3690 | 0.3692 | 0.3683 | 0.3681 | see note: msmarco |
| msmarco | trec-dl-hard | family:multi-budget-weighted:msmarco | 0.3558 | 0.3724 | 0.3697 | 0.3700 | 0.3709 | 0.3698 | 0.3690 | 0.3692 | 0.3683 | 0.3681 | see note: msmarco |
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
| nfcorpus | test | family:fixed-307-weighted:nfcorpus | 0.3348 | 0.3619 | 0.3747 | 0.3784 | 0.3830 | 0.3822 | 0.3805 | 0.3789 | 0.3699 | 0.3619 | see note: nfcorpus |
| nfcorpus | test | family:multi-budget-weighted:nfcorpus | 0.3266 | 0.3525 | 0.3650 | 0.3735 | 0.3784 | 0.3807 | 0.3789 | 0.3776 | 0.3682 | 0.3619 | see note: nfcorpus |
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
| scifact | test | family:fixed-307-weighted:scifact | 0.8444 | 0.8518 | 0.8558 | 0.8516 | 0.8501 | 0.8378 | 0.8276 | 0.8114 | 0.7592 | 0.7000 | — |
| scifact | test | family:multi-budget-weighted:scifact | 0.8276 | 0.8457 | 0.8473 | 0.8463 | 0.8438 | 0.8423 | 0.8308 | 0.8246 | 0.7805 | 0.7000 | — |

## Against full 1024

| test | query set | budget | full | variance - full | retrieval - full | geometric - full | weighted - full |
|---|---|---|---|---|---|---|---|
| msmarco | test | 102 | 0.3800 | -0.0216 | -0.0616 | -0.0216 | -0.0216 |
| msmarco | test | 204 | 0.3800 | -0.0083 | -0.0284 | -0.0083 | -0.0083 |
| msmarco | test | 307 | 0.3800 | -0.0036 | -0.0155 | -0.0036 | -0.0036 |
| msmarco | test | 409 | 0.3800 | -0.0017 | -0.0097 | -0.0017 | -0.0017 |
| msmarco | test | 512 | 0.3800 | -0.0009 | -0.0056 | -0.0009 | -0.0009 |
| msmarco | test | 614 | 0.3800 | +0.0005 | -0.0038 | +0.0005 | +0.0005 |
| msmarco | test | 716 | 0.3800 | -0.0008 | -0.0017 | -0.0008 | -0.0008 |
| msmarco | test | 819 | 0.3800 | +0.0002 | -0.0010 | +0.0002 | +0.0002 |
| msmarco | test | 921 | 0.3800 | +0.0003 | -0.0004 | +0.0003 | +0.0003 |
| msmarco | trec-dl-2019 | 102 | 0.6654 | -0.0381 | -0.0918 | -0.0381 | -0.0381 |
| msmarco | trec-dl-2019 | 204 | 0.6654 | -0.0203 | -0.0295 | -0.0203 | -0.0203 |
| msmarco | trec-dl-2019 | 307 | 0.6654 | -0.0114 | -0.0305 | -0.0114 | -0.0114 |
| msmarco | trec-dl-2019 | 409 | 0.6654 | -0.0049 | -0.0196 | -0.0049 | -0.0049 |
| msmarco | trec-dl-2019 | 512 | 0.6654 | -0.0026 | +0.0036 | -0.0026 | -0.0026 |
| msmarco | trec-dl-2019 | 614 | 0.6654 | +0.0017 | -0.0018 | +0.0017 | +0.0017 |
| msmarco | trec-dl-2019 | 716 | 0.6654 | +0.0033 | -0.0005 | +0.0033 | +0.0033 |
| msmarco | trec-dl-2019 | 819 | 0.6654 | +0.0026 | -0.0043 | +0.0026 | +0.0026 |
| msmarco | trec-dl-2019 | 921 | 0.6654 | +0.0005 | -0.0027 | +0.0005 | +0.0005 |
| msmarco | trec-dl-2020 | 102 | 0.6767 | -0.0279 | -0.0708 | -0.0279 | -0.0279 |
| msmarco | trec-dl-2020 | 204 | 0.6767 | -0.0093 | -0.0269 | -0.0093 | -0.0093 |
| msmarco | trec-dl-2020 | 307 | 0.6767 | -0.0029 | -0.0245 | -0.0029 | -0.0029 |
| msmarco | trec-dl-2020 | 409 | 0.6767 | -0.0044 | -0.0162 | -0.0044 | -0.0044 |
| msmarco | trec-dl-2020 | 512 | 0.6767 | -0.0064 | -0.0117 | -0.0064 | -0.0064 |
| msmarco | trec-dl-2020 | 614 | 0.6767 | -0.0034 | -0.0054 | -0.0034 | -0.0034 |
| msmarco | trec-dl-2020 | 716 | 0.6767 | -0.0016 | -0.0034 | -0.0016 | -0.0016 |
| msmarco | trec-dl-2020 | 819 | 0.6767 | -0.0033 | -0.0039 | -0.0033 | -0.0033 |
| msmarco | trec-dl-2020 | 921 | 0.6767 | -0.0011 | +0.0024 | -0.0011 | -0.0011 |
| msmarco | trec-dl-hard | 102 | 0.3681 | -0.0124 | -0.0672 | -0.0124 | -0.0124 |
| msmarco | trec-dl-hard | 204 | 0.3681 | +0.0043 | -0.0269 | +0.0043 | +0.0043 |
| msmarco | trec-dl-hard | 307 | 0.3681 | +0.0016 | -0.0203 | +0.0016 | +0.0016 |
| msmarco | trec-dl-hard | 409 | 0.3681 | +0.0019 | -0.0067 | +0.0019 | +0.0019 |
| msmarco | trec-dl-hard | 512 | 0.3681 | +0.0028 | -0.0018 | +0.0028 | +0.0028 |
| msmarco | trec-dl-hard | 614 | 0.3681 | +0.0017 | -0.0004 | +0.0017 | +0.0017 |
| msmarco | trec-dl-hard | 716 | 0.3681 | +0.0009 | +0.0069 | +0.0009 | +0.0009 |
| msmarco | trec-dl-hard | 819 | 0.3681 | +0.0011 | +0.0024 | +0.0011 | +0.0011 |
| msmarco | trec-dl-hard | 921 | 0.3681 | +0.0002 | +0.0043 | +0.0002 | +0.0002 |
| nfcorpus | test | 102 | 0.3619 | -0.0176 | -0.0487 | -0.0079 | -0.0453 |
| nfcorpus | test | 204 | 0.3619 | -0.0060 | -0.0140 | +0.0165 | -0.0084 |
| nfcorpus | test | 307 | 0.3619 | -0.0024 | -0.0012 | +0.0132 | +0.0062 |
| nfcorpus | test | 409 | 0.3619 | -0.0029 | +0.0063 | +0.0177 | +0.0149 |
| nfcorpus | test | 512 | 0.3619 | -0.0016 | +0.0117 | +0.0167 | +0.0187 |
| nfcorpus | test | 614 | 0.3619 | +0.0009 | +0.0127 | +0.0149 | +0.0204 |
| nfcorpus | test | 716 | 0.3619 | +0.0011 | +0.0132 | +0.0130 | +0.0179 |
| nfcorpus | test | 819 | 0.3619 | +0.0001 | +0.0132 | +0.0083 | +0.0160 |
| nfcorpus | test | 921 | 0.3619 | -0.0004 | +0.0090 | +0.0024 | +0.0086 |
| scifact | test | 102 | 0.7000 | -0.0278 | +0.1061 | +0.1359 | +0.1289 |
| scifact | test | 204 | 0.7000 | +0.0012 | +0.1411 | +0.1401 | +0.1475 |
| scifact | test | 307 | 0.7000 | -0.0026 | +0.1458 | +0.1390 | +0.1518 |
| scifact | test | 409 | 0.7000 | -0.0024 | +0.1473 | +0.1251 | +0.1494 |
| scifact | test | 512 | 0.7000 | -0.0028 | +0.1454 | +0.1150 | +0.1458 |
| scifact | test | 614 | 0.7000 | -0.0021 | +0.1375 | +0.0961 | +0.1424 |
| scifact | test | 716 | 0.7000 | +0.0009 | +0.1301 | +0.0684 | +0.1351 |
| scifact | test | 819 | 0.7000 | +0.0027 | +0.1171 | +0.0443 | +0.1244 |
| scifact | test | 921 | 0.7000 | +0.0006 | +0.0842 | +0.0093 | +0.0808 |

## TREC DL sets (context, not in the criteria)

| test | query set | against | summary | reference | weighted | diff | ci_low | ci_high | p | note |
|---|---|---|---|---|---|---|---|---|---|---|
| msmarco | trec-dl-2019 | S1 selector | mean 10%-90% | 0.6457 | 0.6577 | +0.0120 | -0.0006 | +0.0250 | 0.0758 | see note: msmarco |
| msmarco | trec-dl-2019 | S1 selector | k = 307 | 0.6349 | 0.6540 | +0.0191 | -0.0004 | +0.0391 | 0.0648 | see note: msmarco |
| msmarco | trec-dl-2019 | S2 selector | mean 10%-90% | 0.6577 | 0.6577 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-2019 | S2 selector | k = 307 | 0.6540 | 0.6540 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-2019 | variance | mean 10%-90% | 0.6577 | 0.6577 | +0.0000 | -0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-2019 | variance | k = 307 | 0.6540 | 0.6540 | +0.0000 | -0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-2020 | S1 selector | mean 10%-90% | 0.6588 | 0.6700 | +0.0111 | -0.0007 | +0.0235 | 0.0797 | see note: msmarco |
| msmarco | trec-dl-2020 | S1 selector | k = 307 | 0.6522 | 0.6738 | +0.0216 | +0.0028 | +0.0422 | 0.0410 | see note: msmarco |
| msmarco | trec-dl-2020 | S2 selector | mean 10%-90% | 0.6700 | 0.6700 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-2020 | S2 selector | k = 307 | 0.6738 | 0.6738 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-2020 | variance | mean 10%-90% | 0.6700 | 0.6700 | +0.0000 | -0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-2020 | variance | k = 307 | 0.6738 | 0.6738 | +0.0000 | -0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-hard | S1 selector | mean 10%-90% | 0.3559 | 0.3683 | +0.0124 | -0.0007 | +0.0255 | 0.0699 | see note: msmarco |
| msmarco | trec-dl-hard | S1 selector | k = 307 | 0.3478 | 0.3697 | +0.0219 | +0.0022 | +0.0421 | 0.0407 | see note: msmarco |
| msmarco | trec-dl-hard | S2 selector | mean 10%-90% | 0.3683 | 0.3683 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-hard | S2 selector | k = 307 | 0.3697 | 0.3697 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-hard | variance | mean 10%-90% | 0.3683 | 0.3683 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | see note: msmarco |
| msmarco | trec-dl-hard | variance | k = 307 | 0.3697 | 0.3697 | +0.0000 | -0.0000 | +0.0000 | 1.0000 | see note: msmarco |

## Families (validation-best cell of each variant, seed 0; context)

| test | family | cell | mean 10%-90% | k = 307 |
|---|---|---|---|---|
| msmarco | l2s (3 seeds) | — | 0.3607 | 0.3572 |
| msmarco | fixed-307-weighted | fixed-307-weighted-lr0.0001-ts0.02-b0.1 | 0.3760 | 0.3765 |
| msmarco | multi-budget-weighted | multi-budget-weighted-lr0.0001-ts0.02-b0.1 | 0.3760 | 0.3765 |
| nfcorpus | l2s (3 seeds) | — | 0.3622 | 0.3607 |
| nfcorpus | fixed-307-weighted | fixed-307-weighted-lr0.001-ts0.05-b0.1 | 0.3716 | 0.3747 |
| nfcorpus | multi-budget-weighted | multi-budget-weighted-lr0.001-ts0.05-b0.1 | 0.3668 | 0.3650 |
| scifact | l2s (3 seeds) | — | 0.8166 | 0.8353 |
| scifact | fixed-307-weighted | fixed-307-weighted-lr0.001-ts0.02-b0.1 | 0.8322 | 0.8558 |
| scifact | multi-budget-weighted | multi-budget-weighted-lr0.001-ts0.02-b0.1 | 0.8321 | 0.8473 |

## Where the difference comes from

| environment | budget | weighted vs L2S | L2S seeds | weighted seeds |
|---|---|---|---|---|
| msmarco | 102 | 0.3334 | 0.9389 | 1.0000 |
| msmarco | 204 | 0.3947 | 0.9473 | 1.0000 |
| msmarco | 307 | 0.4576 | 0.9544 | 1.0000 |
| msmarco | 409 | 0.5222 | 0.9606 | 1.0000 |
| msmarco | 512 | 0.5898 | 0.9662 | 1.0000 |
| msmarco | 614 | 0.6593 | 0.9717 | 1.0000 |
| msmarco | 716 | 0.7332 | 0.9774 | 1.0000 |
| msmarco | 819 | 0.8141 | 0.9837 | 1.0000 |
| msmarco | 921 | 0.9025 | 0.9910 | 1.0000 |
| nfcorpus | 102 | 0.4250 | 0.6714 | 0.5769 |
| nfcorpus | 204 | 0.5159 | 0.7135 | 0.6516 |
| nfcorpus | 307 | 0.5875 | 0.7517 | 0.7059 |
| nfcorpus | 409 | 0.6502 | 0.7861 | 0.7503 |
| nfcorpus | 512 | 0.7085 | 0.8201 | 0.7925 |
| nfcorpus | 614 | 0.7655 | 0.8546 | 0.8331 |
| nfcorpus | 716 | 0.8215 | 0.8903 | 0.8723 |
| nfcorpus | 819 | 0.8776 | 0.9259 | 0.9113 |
| nfcorpus | 921 | 0.9348 | 0.9621 | 0.9519 |
| scifact | 102 | 0.4690 | 0.5754 | 0.8035 |
| scifact | 204 | 0.5215 | 0.6119 | 0.8346 |
| scifact | 307 | 0.5706 | 0.6492 | 0.8513 |
| scifact | 409 | 0.6215 | 0.6881 | 0.8700 |
| scifact | 512 | 0.6769 | 0.7324 | 0.8912 |
| scifact | 614 | 0.7338 | 0.7793 | 0.9107 |
| scifact | 716 | 0.7935 | 0.8291 | 0.9307 |
| scifact | 819 | 0.8570 | 0.8827 | 0.9514 |
| scifact | 921 | 0.9238 | 0.9395 | 0.9738 |

| environment | selector | mean gain over global importance | p | p_holm | fits |
|---|---|---|---|---|---|
| msmarco | l2s | +0.0320 | 0.0001 | 0.0003 | yes |
| msmarco | retrieval | +0.0423 | 0.0001 | 0.0003 | yes |
| msmarco | geometric | +0.2070 | 0.0001 | 0.0003 | yes |
| msmarco | weighted | +0.2070 | 0.0001 | 0.0003 | yes |
| nfcorpus | l2s | +0.0278 | 0.0001 | 0.0003 | yes |
| nfcorpus | retrieval | +0.0278 | 0.0001 | 0.0003 | yes |
| nfcorpus | geometric | +0.1068 | 0.0001 | 0.0003 | yes |
| nfcorpus | weighted | +0.0370 | 0.0001 | 0.0003 | yes |
| scifact | l2s | +0.1401 | 0.0001 | 0.0003 | yes |
| scifact | retrieval | +0.1032 | 0.0001 | 0.0003 | yes |
| scifact | geometric | +0.2564 | 0.0001 | 0.0003 | yes |
| scifact | weighted | +0.1468 | 0.0001 | 0.0003 | yes |

| environment | seed | steps | best step | best validation rerank nDCG@10 | final soft mass |
|---|---|---|---|---|---|
| msmarco | 0 | 20000 | 0 | 0.7159 | 0.9329 |
| msmarco | 1 | 20000 | 0 | 0.7159 | 0.9328 |
| msmarco | 2 | 20000 | 0 | 0.7159 | 0.9329 |
| nfcorpus | 0 | 20000 | 7000 | 0.6143 | 0.9376 |
| nfcorpus | 1 | 20000 | 5000 | 0.6145 | 0.9377 |
| nfcorpus | 2 | 20000 | 1000 | 0.6170 | 0.9379 |
| scifact | 0 | 20000 | 17000 | 0.9262 | 0.9372 |
| scifact | 1 | 20000 | 5000 | 0.9276 | 0.9372 |
| scifact | 2 | 20000 | 12000 | 0.9284 | 0.9375 |

## Notes

- **msmarco**: E0 does not reproduce Learning-to-Select on MS MARCO (ADR-0006). E1 scores MS MARCO dev against the full 8.8M-passage corpus; the paper evaluates the `test` split of an unpublished `msmarco_subset_50k` folder, with an ideal DCG taken from the retrieved list (F4 research D9).
- **nfcorpus**: NFCorpus's selector does not fit its oracle distribution as a calibrated distribution (ADR-0006); read its shift with its ranking-fit result (F4 research D2).
- **msmarco**: the selected cell starts from the geometric prior and every seed keeps its step-0 checkpoint (no training step improved validation), so `weighted:msmarco` is the `variance` method itself.

Every grid cell is in `tables/grid.csv`.
