# P3: coverage pilot of a score-matched reference control

Exploratory and descriptive, training judgments only. This run measures whether each
relevant document can be paired with an unjudged corpus document of a similar score. It
computes no mask, no ordering, and no all-k curve, and it ranks no corpus: scores exist
only for a bounded seeded sample. Reference documents are unjudged, not negatives, and
high-scoring ones may be relevant. Published P3 results are not changed by this pilot.

- Run `20261107T001827Z-099fb32c` (commit `8500602612f5`, device `cuda:0`)
- At most 500 multi-positive queries per environment, 3 seeded samples, caliper +/- 0.02, sample of min(200000, half the corpus) documents, fixed before seeing results.

## Coverage

A query is complete in a repeat when all its positives match distinct documents. A
weighted rate is a stratified estimate over all eligible multi-positive queries, with
weights in the strata table; the unweighted rates describe the selection only and are
not population coverage. The standard error covers selecting queries and not sampling
documents; the by-repeat columns show the second.

- **scifact: limited coverage.** About 0.242 of eligible multi-positive queries match completely. Matched queries are then a selected subset, and a comparison on them would not describe the others.
- **nfcorpus: limited coverage.** About 0.409 of eligible multi-positive queries match completely. Matched queries are then a selected subset, and a comparison on them would not describe the others.
- **msmarco: limited coverage.** About 0.279 of eligible multi-positive queries match completely. Matched queries are then a selected subset, and a comparison on them would not describe the others.

| environment | selected queries | pairs | unweighted complete rate | unweighted pair rate | complete rate by repeat | complete in every repeat | complete in no repeat | weighted complete rate (SE) | weighted pair rate |
|---|---|---|---|---|---|---|---|---|---|
| scifact | 69 | 163 | 0.242 | 0.450 | 0.217 / 0.217 / 0.290 | 12 | 47 | 0.242 (0.000) | 0.450 |
| nfcorpus | 500 | 19646 | 0.424 | 0.922 | 0.438 / 0.418 / 0.416 | 188 | 265 | 0.409 (0.017) | 0.922 |
| msmarco | 500 | 1106 | 0.278 | 0.481 | 0.260 / 0.274 / 0.300 | 84 | 298 | 0.279 (0.015) | 0.478 |

| environment | positives per query | eligible queries | selected | sampling weight | complete rate | complete by repeat | pair rate |
|---|---|---|---|---|---|---|---|
| scifact | 2-5 | 69 | 69 | 1.000 | 0.242 | 0.217 / 0.217 / 0.290 | 0.450 |
| scifact | 6-20 | 0 | 0 | 0.000 | n/a | n/a | n/a |
| scifact | 21+ | 0 | 0 | 0.000 | n/a | n/a | n/a |
| nfcorpus | 2-5 | 399 | 111 | 3.595 | 0.631 | 0.649 / 0.622 / 0.622 | 0.852 |
| nfcorpus | 6-20 | 688 | 161 | 4.273 | 0.518 | 0.534 / 0.516 / 0.503 | 0.919 |
| nfcorpus | 21+ | 1075 | 228 | 4.715 | 0.257 | 0.268 / 0.250 / 0.254 | 0.923 |
| msmarco | 2-5 | 2317 | 496 | 4.671 | 0.279 | 0.260 / 0.276 / 0.300 | 0.477 |
| msmarco | 6-20 | 4 | 4 | 1.000 | 0.167 | 0.250 / 0.000 / 0.250 | 0.653 |
| msmarco | 21+ | 0 | 0 | 0.000 | n/a | n/a | n/a |

## Why groups are incomplete

The matching is exact, so an incomplete group is never an algorithm's failure. The
two exact methods are compared on every group. Greedy nearest-unused is shown only to
count the complete groups it would have missed.

| environment | query-repeat outcome | count |
|---|---|---|
| scifact | a positive has no eligible sample document within the caliper | 150 |
| scifact | complete | 50 |
| scifact | every positive has a document, but positives compete for too few | 7 |
| scifact | fewer eligible sample documents than positives | 0 |
| nfcorpus | a positive has no eligible sample document within the caliper | 812 |
| nfcorpus | complete | 636 |
| nfcorpus | every positive has a document, but positives compete for too few | 52 |
| nfcorpus | fewer eligible sample documents than positives | 0 |
| msmarco | a positive has no eligible sample document within the caliper | 1036 |
| msmarco | complete | 417 |
| msmarco | every positive has a document, but positives compete for too few | 47 |
| msmarco | fewer eligible sample documents than positives | 0 |

| environment | exact checks disagreeing | complete query-repeats, exact | complete, greedy nearest-unused | greedy false failures |
|---|---|---|---|---|
| scifact | 0 | 50 | 50 | 0 |
| nfcorpus | 0 | 636 | 629 | 7 |
| msmarco | 0 | 417 | 409 | 8 |

## Scores and residuals

| environment | matched positive score | matched reference score | unmatched positive score | reference minus positive | absolute residual | largest absolute residual | sample mean score per query |
|---|---|---|---|---|---|---|---|
| scifact | 0.611 (0.501 to 0.722), n=220 | 0.609 (0.498 to 0.725), n=220 | 0.718 (0.615 to 0.805), n=269 | -0.002 (-0.014 to 0.009), n=220 | 0.007 (0.001 to 0.017), n=220 | 0.019801 | 0.267 (0.221 to 0.305), n=207 |
| nfcorpus | 0.291 (0.170 to 0.435), n=54323 | 0.291 (0.170 to 0.434), n=54323 | 0.542 (0.409 to 0.669), n=4615 | -0.001 (-0.002 to 0.000), n=54323 | 0.001 (0.000 to 0.003), n=54323 | 0.019995 | 0.238 (0.174 to 0.303), n=1500 |
| msmarco | 0.608 (0.490 to 0.727), n=1595 | 0.606 (0.489 to 0.722), n=1595 | 0.711 (0.618 to 0.796), n=1723 | -0.002 (-0.013 to 0.007), n=1595 | 0.005 (0.000 to 0.014), n=1595 | 0.020000 | 0.117 (0.093 to 0.140), n=1500 |

Pair rate by the positive's score, over pairs of the selection:

| environment | positive score decile | score range | pairs | pair rate |
|---|---|---|---|---|
| scifact | 1 | 0.400 to 0.555 | 48 | 0.896 |
| scifact | 2 | 0.555 to 0.590 | 48 | 0.812 |
| scifact | 3 | 0.590 to 0.616 | 51 | 0.706 |
| scifact | 4 | 0.616 to 0.652 | 48 | 0.771 |
| scifact | 5 | 0.652 to 0.683 | 48 | 0.417 |
| scifact | 6 | 0.683 to 0.704 | 48 | 0.104 |
| scifact | 7 | 0.704 to 0.727 | 51 | 0.353 |
| scifact | 8 | 0.727 to 0.756 | 48 | 0.250 |
| scifact | 9 | 0.756 to 0.781 | 48 | 0.125 |
| scifact | 10 | 0.781 to 0.851 | 51 | 0.078 |
| nfcorpus | 1 | 0.004 to 0.173 | 5892 | 0.999 |
| nfcorpus | 2 | 0.173 to 0.207 | 5895 | 1.000 |
| nfcorpus | 3 | 0.207 to 0.234 | 5895 | 1.000 |
| nfcorpus | 4 | 0.234 to 0.261 | 5892 | 1.000 |
| nfcorpus | 5 | 0.261 to 0.289 | 5895 | 1.000 |
| nfcorpus | 6 | 0.289 to 0.320 | 5892 | 0.999 |
| nfcorpus | 7 | 0.320 to 0.358 | 5895 | 0.988 |
| nfcorpus | 8 | 0.358 to 0.408 | 5892 | 0.938 |
| nfcorpus | 9 | 0.408 to 0.485 | 5895 | 0.836 |
| nfcorpus | 10 | 0.485 to 0.837 | 5895 | 0.457 |
| msmarco | 1 | 0.140 to 0.529 | 330 | 0.952 |
| msmarco | 2 | 0.529 to 0.581 | 333 | 0.817 |
| msmarco | 3 | 0.581 to 0.615 | 333 | 0.781 |
| msmarco | 4 | 0.615 to 0.644 | 330 | 0.579 |
| msmarco | 5 | 0.644 to 0.670 | 333 | 0.456 |
| msmarco | 6 | 0.670 to 0.697 | 330 | 0.376 |
| msmarco | 7 | 0.697 to 0.722 | 333 | 0.336 |
| msmarco | 8 | 0.722 to 0.747 | 330 | 0.188 |
| msmarco | 9 | 0.747 to 0.779 | 333 | 0.192 |
| msmarco | 10 | 0.779 to 0.905 | 333 | 0.132 |

## Selection

Queries that match are not a random part of the eligible ones. Anything later computed
on matched queries describes this group only.

| environment | group | queries | weighted share | positives per query | cosine among positives |
|---|---|---|---|---|---|
| scifact | complete in every repeat | 12 | 0.174 | 2.000 (2.000 to 2.000), n=12 | 0.680 (0.582 to 0.839), n=12 |
| scifact | complete in some repeats | 10 | 0.145 | 2.200 (2.000 to 3.000), n=10 | 0.710 (0.627 to 0.808), n=10 |
| scifact | complete in no repeat | 47 | 0.681 | 2.489 (2.000 to 4.000), n=47 | 0.728 (0.525 to 0.832), n=47 |
| nfcorpus | complete in every repeat | 188 | 0.361 | 17.074 (2.000 to 40.000), n=188 | 0.461 (0.353 to 0.601), n=188 |
| nfcorpus | complete in some repeats | 47 | 0.095 | 37.830 (3.600 to 65.200), n=47 | 0.428 (0.344 to 0.550), n=47 |
| nfcorpus | complete in no repeat | 265 | 0.545 | 55.313 (4.000 to 113.600), n=265 | 0.425 (0.334 to 0.553), n=265 |
| msmarco | complete in every repeat | 84 | 0.169 | 2.143 (2.000 to 2.700), n=84 | 0.603 (0.477 to 0.770), n=84 |
| msmarco | complete in some repeats | 118 | 0.236 | 2.254 (2.000 to 3.000), n=118 | 0.632 (0.500 to 0.769), n=118 |
| msmarco | complete in no repeat | 298 | 0.595 | 2.215 (2.000 to 3.000), n=298 | 0.684 (0.523 to 0.839), n=298 |

## Document-document cosine on complete groups

Score is matched, and cosine among a group's documents is not. Random references are the
same number of eligible documents from the same sample. The positives' cosine is for all
queries complete at least once; differences are per query, and they are descriptive.

| environment | queries complete at least once | among positives | among matched references | among random references, same queries | positives minus matched | positives minus random | matched by repeat |
|---|---|---|---|---|---|---|---|
| scifact | 22 | 0.694 (0.589 to 0.817), n=22 | 0.586 (0.477 to 0.726), n=22 | 0.284 (0.242 to 0.358), n=22 | 0.108 (0.030 to 0.211), n=22 | 0.410 (0.308 to 0.569), n=22 | 0.564 / 0.578 / 0.569 |
| nfcorpus | 235 | 0.455 (0.350 to 0.598), n=235 | 0.340 (0.298 to 0.397), n=235 | 0.306 (0.286 to 0.325), n=235 | 0.115 (0.017 to 0.245), n=235 | 0.148 (0.040 to 0.303), n=235 | 0.337 / 0.339 / 0.340 |
| msmarco | 202 | 0.620 (0.480 to 0.772), n=202 | 0.571 (0.395 to 0.735), n=202 | 0.162 (0.102 to 0.222), n=202 | 0.049 (-0.082 to 0.215), n=202 | 0.458 (0.301 to 0.634), n=202 | 0.557 / 0.558 / 0.552 |

## Cost

One run on the device named. The last column scales this run's seconds per query-repeat
to every eligible multi-positive query; it is an extrapolation, covers matching only, and
includes no all-k curves. An earlier estimate, made before any measurement, gave this
pilot under 5 minutes and under 10 MB and a full curve run of 1 to 3 GB; treat the full
figure as unverified. Peak RSS counts pages of memory-mapped files.

| environment | device | seconds | seconds per query-repeat | read | transfer | compute | match | write | peak device MiB | peak RSS MiB (process so far) | artifact KiB | matching alone, all eligible queries, 3 repeats, seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact | NVIDIA GeForce RTX 4090 | 0.3 | 0.0014 | 0.0 | 0.0 | 0.1 | 0.1 | 0.0 | 54 | 1042 | 11 | 0 |
| nfcorpus | NVIDIA GeForce RTX 4090 | 1.7 | 0.0011 | 0.0 | 0.0 | 0.3 | 1.3 | 0.1 | 99 | 1166 | 955 | 7 |
| msmarco | NVIDIA GeForce RTX 4090 | 28.1 | 0.0187 | 1.7 | 0.7 | 1.3 | 23.8 | 0.0 | 1988 | 21342 | 64 | 130 |
