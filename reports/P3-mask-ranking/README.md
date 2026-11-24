# P3: rankings under masks that maximize positive similarity

Privileged offline diagnostic, training judgments only, in-domain. Nothing here is a retrieval method, a ranking upper bound, or evidence about unseen queries.

- Run `20261109T163931Z-9e9f5a0a` (commit `bd0fd382202d`, dirty `False`, device `cuda:0`)
- Budgets [102, 204, 307, 409, 512, 614, 716, 819, 921], a grid: representation curves cover every k, these rankings do not. The full query is k = 1024.
- Exact float64 scoring of every corpus document for every mask; ties go to the lower row; ids equal to the query id are excluded, as in `rank_documents`.

## Reading these tables

The shared mask saw the positives it is scored on. Construction-only masks see only their construction positives, and are scored on held-out positives of the same training query. Each positive's own mask maximizes that positive's similarity: a privileged diagnostic, and never an upper bound on its rank. Construction positives remain relevant and remain in every primary ranking. Judgments are incomplete, so competitors are other documents, not confirmed irrelevant ones. Masks maximize mean similarity; they are neither minimax nor ranking optimal.

Every arm here, the full query and prefix-k included, breaks exact ties the same way: the lower corpus row wins, and scores are exact. The repository's float32 `exact_search` leaves such ties to `topk`, so its full-query numbers can differ on queries with duplicate documents. Earlier results stand as published and are not recomputed here.

Intervals are pointwise bootstrap intervals over queries and are exploratory. Only the two families below carry a multiplicity adjustment, each within itself.

## Pre-declared families

Primary: does a construction-only mask move held-out positives? Secondary: all-label metrics, in which a mask can gain by lifting the very documents that built it. A gain in the secondary family without a gain in the primary one is memorization of construction documents, not better retrieval of relevant documents.

| family | environment | arm | k | measure | queries | difference from the full query | 95% interval (pointwise, exploratory) | p (sign-flip) | p (Holm, within family) |
|---|---|---|---|---|---|---|---|---|---|
| primary: held-out positives | scifact | 1-1 | 102 | held-out reciprocal rank | 69 | -0.1013 | -0.1378 to -0.0641 | 0.0001 | 0.0108 |
| primary: held-out positives | scifact | 1-1 | 102 | held-out log rank | 69 | 0.1109 | -0.0223 to 0.2312 | 0.0998 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 102 | held-out in top 10 | 69 | 0.0507 | 0.0024 to 0.0991 | 0.0685 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 102 | held-out in top 100 | 69 | -0.0048 | -0.0290 to 0.0193 | 1.0000 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 204 | held-out reciprocal rank | 69 | -0.0959 | -0.1330 to -0.0606 | 0.0001 | 0.0108 |
| primary: held-out positives | scifact | 1-1 | 204 | held-out log rank | 69 | 0.0672 | -0.0619 to 0.1814 | 0.3014 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 204 | held-out in top 10 | 69 | 0.0507 | 0.0060 to 0.0979 | 0.0427 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 204 | held-out in top 100 | 69 | -0.0048 | -0.0290 to 0.0193 | 1.0000 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 307 | held-out reciprocal rank | 69 | -0.0932 | -0.1303 to -0.0568 | 0.0002 | 0.0122 |
| primary: held-out positives | scifact | 1-1 | 307 | held-out log rank | 69 | 0.0440 | -0.0850 to 0.1609 | 0.4962 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 307 | held-out in top 10 | 69 | 0.0489 | -0.0024 to 0.1063 | 0.0953 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 307 | held-out in top 100 | 69 | -0.0024 | -0.0290 to 0.0242 | 1.0000 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 409 | held-out reciprocal rank | 69 | -0.0710 | -0.1051 to -0.0369 | 0.0003 | 0.0171 |
| primary: held-out positives | scifact | 1-1 | 409 | held-out log rank | 69 | 0.0087 | -0.1118 to 0.1126 | 0.8838 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 409 | held-out in top 10 | 69 | 0.0441 | -0.0000 to 0.0888 | 0.0770 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 409 | held-out in top 100 | 69 | -0.0024 | -0.0290 to 0.0242 | 1.0000 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 512 | held-out reciprocal rank | 69 | -0.0593 | -0.0929 to -0.0286 | 0.0004 | 0.0224 |
| primary: held-out positives | scifact | 1-1 | 512 | held-out log rank | 69 | -0.0180 | -0.1273 to 0.0778 | 0.7404 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 512 | held-out in top 10 | 69 | 0.0368 | -0.0042 to 0.0803 | 0.1293 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 512 | held-out in top 100 | 69 | -0.0024 | -0.0290 to 0.0242 | 1.0000 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 614 | held-out reciprocal rank | 69 | -0.0586 | -0.0902 to -0.0298 | 0.0005 | 0.0275 |
| primary: held-out positives | scifact | 1-1 | 614 | held-out log rank | 69 | -0.0215 | -0.1299 to 0.0700 | 0.6854 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 614 | held-out in top 10 | 69 | 0.0332 | -0.0079 to 0.0767 | 0.1915 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 614 | held-out in top 100 | 69 | -0.0024 | -0.0290 to 0.0242 | 1.0000 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 716 | held-out reciprocal rank | 69 | -0.0587 | -0.0883 to -0.0316 | 0.0002 | 0.0122 |
| primary: held-out positives | scifact | 1-1 | 716 | held-out log rank | 69 | 0.0003 | -0.0900 to 0.0819 | 0.9942 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 716 | held-out in top 10 | 69 | 0.0248 | -0.0127 to 0.0664 | 0.2810 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 716 | held-out in top 100 | 69 | -0.0024 | -0.0290 to 0.0242 | 1.0000 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 819 | held-out reciprocal rank | 69 | -0.0318 | -0.0628 to -0.0041 | 0.0368 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 819 | held-out log rank | 69 | -0.0381 | -0.1174 to 0.0407 | 0.3613 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 819 | held-out in top 10 | 69 | 0.0344 | -0.0000 to 0.0749 | 0.0942 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 819 | held-out in top 100 | 69 | 0.0024 | -0.0170 to 0.0242 | 1.0000 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 921 | held-out reciprocal rank | 69 | -0.0237 | -0.0495 to -0.0010 | 0.0635 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 921 | held-out log rank | 69 | -0.0326 | -0.1020 to 0.0338 | 0.3705 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 921 | held-out in top 10 | 69 | 0.0229 | -0.0048 to 0.0519 | 0.1678 | 1.0000 |
| primary: held-out positives | scifact | 1-1 | 921 | held-out in top 100 | 69 | 0.0097 | 0.0000 to 0.0266 | 0.4951 | 1.0000 |
| primary: held-out positives | nfcorpus | 1-1 | 102 | held-out reciprocal rank | 2162 | -0.0070 | -0.0105 to -0.0037 | 0.0002 | 0.0122 |
| primary: held-out positives | nfcorpus | 1-1 | 102 | held-out log rank | 2162 | -0.4191 | -0.4582 to -0.3790 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 102 | held-out in top 10 | 2162 | 0.0358 | 0.0289 to 0.0425 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 102 | held-out in top 100 | 2162 | 0.0866 | 0.0772 to 0.0955 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 204 | held-out reciprocal rank | 2162 | -0.0047 | -0.0080 to -0.0017 | 0.0056 | 0.2744 |
| primary: held-out positives | nfcorpus | 1-1 | 204 | held-out log rank | 2162 | -0.4912 | -0.5283 to -0.4525 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 204 | held-out in top 10 | 2162 | 0.0461 | 0.0396 to 0.0526 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 204 | held-out in top 100 | 2162 | 0.1029 | 0.0938 to 0.1117 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 307 | held-out reciprocal rank | 2162 | -0.0021 | -0.0053 to 0.0009 | 0.1936 | 1.0000 |
| primary: held-out positives | nfcorpus | 1-1 | 307 | held-out log rank | 2162 | -0.5107 | -0.5459 to -0.4738 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 307 | held-out in top 10 | 2162 | 0.0469 | 0.0406 to 0.0533 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 307 | held-out in top 100 | 2162 | 0.1065 | 0.0975 to 0.1154 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 409 | held-out reciprocal rank | 2162 | 0.0003 | -0.0028 to 0.0032 | 0.8522 | 1.0000 |
| primary: held-out positives | nfcorpus | 1-1 | 409 | held-out log rank | 2162 | -0.5148 | -0.5492 to -0.4790 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 409 | held-out in top 10 | 2162 | 0.0461 | 0.0400 to 0.0518 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 409 | held-out in top 100 | 2162 | 0.1048 | 0.0960 to 0.1136 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 512 | held-out reciprocal rank | 2162 | 0.0021 | -0.0008 to 0.0049 | 0.1682 | 1.0000 |
| primary: held-out positives | nfcorpus | 1-1 | 512 | held-out log rank | 2162 | -0.5107 | -0.5443 to -0.4758 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 512 | held-out in top 10 | 2162 | 0.0437 | 0.0375 to 0.0496 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 512 | held-out in top 100 | 2162 | 0.1035 | 0.0947 to 0.1120 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 614 | held-out reciprocal rank | 2162 | 0.0035 | 0.0006 to 0.0060 | 0.0130 | 0.6239 |
| primary: held-out positives | nfcorpus | 1-1 | 614 | held-out log rank | 2162 | -0.5036 | -0.5365 to -0.4695 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 614 | held-out in top 10 | 2162 | 0.0422 | 0.0364 to 0.0479 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 614 | held-out in top 100 | 2162 | 0.1014 | 0.0930 to 0.1098 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 716 | held-out reciprocal rank | 2162 | 0.0045 | 0.0019 to 0.0070 | 0.0013 | 0.0689 |
| primary: held-out positives | nfcorpus | 1-1 | 716 | held-out log rank | 2162 | -0.4851 | -0.5162 to -0.4530 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 716 | held-out in top 10 | 2162 | 0.0395 | 0.0339 to 0.0448 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 716 | held-out in top 100 | 2162 | 0.0974 | 0.0889 to 0.1054 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 819 | held-out reciprocal rank | 2162 | 0.0057 | 0.0033 to 0.0080 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 819 | held-out log rank | 2162 | -0.4377 | -0.4656 to -0.4095 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 819 | held-out in top 10 | 2162 | 0.0328 | 0.0279 to 0.0376 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 819 | held-out in top 100 | 2162 | 0.0857 | 0.0781 to 0.0930 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 921 | held-out reciprocal rank | 2162 | 0.0058 | 0.0041 to 0.0077 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 921 | held-out log rank | 2162 | -0.3169 | -0.3359 to -0.2981 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 921 | held-out in top 10 | 2162 | 0.0190 | 0.0151 to 0.0227 | 0.0001 | 0.0108 |
| primary: held-out positives | nfcorpus | 1-1 | 921 | held-out in top 100 | 2162 | 0.0579 | 0.0516 to 0.0641 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 102 | held-out reciprocal rank | 2321 | -0.0578 | -0.0640 to -0.0517 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 102 | held-out log rank | 2321 | 0.1970 | 0.1557 to 0.2407 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 102 | held-out in top 10 | 2321 | -0.0295 | -0.0411 to -0.0177 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 102 | held-out in top 100 | 2321 | -0.0183 | -0.0291 to -0.0076 | 0.0013 | 0.0689 |
| primary: held-out positives | msmarco | 1-1 | 204 | held-out reciprocal rank | 2321 | -0.0527 | -0.0588 to -0.0468 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 204 | held-out log rank | 2321 | 0.0959 | 0.0576 to 0.1355 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 204 | held-out in top 10 | 2321 | -0.0106 | -0.0221 to 0.0003 | 0.0564 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 204 | held-out in top 100 | 2321 | 0.0031 | -0.0071 to 0.0127 | 0.5537 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 307 | held-out reciprocal rank | 2321 | -0.0479 | -0.0537 to -0.0424 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 307 | held-out log rank | 2321 | 0.0518 | 0.0169 to 0.0879 | 0.0038 | 0.1900 |
| primary: held-out positives | msmarco | 1-1 | 307 | held-out in top 10 | 2321 | -0.0061 | -0.0166 to 0.0047 | 0.2612 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 307 | held-out in top 100 | 2321 | 0.0090 | -0.0009 to 0.0186 | 0.0662 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 409 | held-out reciprocal rank | 2321 | -0.0431 | -0.0488 to -0.0375 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 409 | held-out log rank | 2321 | 0.0223 | -0.0109 to 0.0567 | 0.1857 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 409 | held-out in top 10 | 2321 | 0.0003 | -0.0095 to 0.0102 | 0.9512 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 409 | held-out in top 100 | 2321 | 0.0143 | 0.0048 to 0.0232 | 0.0018 | 0.0918 |
| primary: held-out positives | msmarco | 1-1 | 512 | held-out reciprocal rank | 2321 | -0.0388 | -0.0443 to -0.0336 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 512 | held-out log rank | 2321 | 0.0031 | -0.0283 to 0.0359 | 0.8479 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 512 | held-out in top 10 | 2321 | 0.0048 | -0.0046 to 0.0141 | 0.3280 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 512 | held-out in top 100 | 2321 | 0.0154 | 0.0064 to 0.0236 | 0.0009 | 0.0486 |
| primary: held-out positives | msmarco | 1-1 | 614 | held-out reciprocal rank | 2321 | -0.0340 | -0.0393 to -0.0289 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 614 | held-out log rank | 2321 | -0.0123 | -0.0425 to 0.0186 | 0.4164 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 614 | held-out in top 10 | 2321 | 0.0029 | -0.0064 to 0.0119 | 0.5514 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 614 | held-out in top 100 | 2321 | 0.0201 | 0.0118 to 0.0282 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 716 | held-out reciprocal rank | 2321 | -0.0298 | -0.0350 to -0.0251 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 716 | held-out log rank | 2321 | -0.0221 | -0.0503 to 0.0066 | 0.1187 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 716 | held-out in top 10 | 2321 | 0.0054 | -0.0035 to 0.0140 | 0.2296 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 716 | held-out in top 100 | 2321 | 0.0210 | 0.0130 to 0.0286 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 819 | held-out reciprocal rank | 2321 | -0.0272 | -0.0320 to -0.0229 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 819 | held-out log rank | 2321 | -0.0301 | -0.0563 to -0.0040 | 0.0232 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 819 | held-out in top 10 | 2321 | 0.0046 | -0.0039 to 0.0132 | 0.2928 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 819 | held-out in top 100 | 2321 | 0.0220 | 0.0145 to 0.0296 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 921 | held-out reciprocal rank | 2321 | -0.0174 | -0.0217 to -0.0135 | 0.0001 | 0.0108 |
| primary: held-out positives | msmarco | 1-1 | 921 | held-out log rank | 2321 | -0.0465 | -0.0667 to -0.0267 | 0.0002 | 0.0122 |
| primary: held-out positives | msmarco | 1-1 | 921 | held-out in top 10 | 2321 | 0.0067 | -0.0004 to 0.0140 | 0.0607 | 1.0000 |
| primary: held-out positives | msmarco | 1-1 | 921 | held-out in top 100 | 2321 | 0.0224 | 0.0158 to 0.0288 | 0.0001 | 0.0108 |
| secondary: all-label, includes construction positives | scifact | shared | 102 | ndcg@10 | 69 | 0.2624 | 0.1980 to 0.3289 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 102 | mrr@10 | 69 | 0.2537 | 0.1740 to 0.3414 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 102 | recall@100 | 69 | 0.0193 | 0.0000 to 0.0435 | 0.2504 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 102 | ndcg@10 | 69 | 0.2084 | 0.1564 to 0.2623 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 102 | mrr@10 | 69 | 0.3040 | 0.2182 to 0.3902 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 102 | recall@100 | 69 | 0.0068 | -0.0109 to 0.0282 | 0.7471 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | shared | 204 | ndcg@10 | 69 | 0.2604 | 0.1973 to 0.3257 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 204 | mrr@10 | 69 | 0.2629 | 0.1787 to 0.3487 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 204 | recall@100 | 69 | 0.0193 | 0.0000 to 0.0435 | 0.2504 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 204 | ndcg@10 | 69 | 0.2141 | 0.1618 to 0.2690 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 204 | mrr@10 | 69 | 0.3040 | 0.2182 to 0.3902 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 204 | recall@100 | 69 | 0.0068 | -0.0109 to 0.0282 | 0.7471 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | shared | 307 | ndcg@10 | 69 | 0.2457 | 0.1869 to 0.3085 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 307 | mrr@10 | 69 | 0.2581 | 0.1758 to 0.3440 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 307 | recall@100 | 69 | 0.0193 | 0.0000 to 0.0435 | 0.2504 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 307 | ndcg@10 | 69 | 0.2127 | 0.1587 to 0.2678 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 307 | mrr@10 | 69 | 0.2989 | 0.2130 to 0.3833 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 307 | recall@100 | 69 | 0.0085 | -0.0109 to 0.0302 | 0.5581 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | shared | 409 | ndcg@10 | 69 | 0.2348 | 0.1756 to 0.2968 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 409 | mrr@10 | 69 | 0.2557 | 0.1749 to 0.3396 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 409 | recall@100 | 69 | 0.0193 | 0.0000 to 0.0435 | 0.2504 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 409 | ndcg@10 | 69 | 0.2087 | 0.1570 to 0.2613 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 409 | mrr@10 | 69 | 0.2907 | 0.2072 to 0.3739 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 409 | recall@100 | 69 | 0.0085 | -0.0109 to 0.0302 | 0.5581 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | shared | 512 | ndcg@10 | 69 | 0.2266 | 0.1677 to 0.2901 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 512 | mrr@10 | 69 | 0.2533 | 0.1722 to 0.3372 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 512 | recall@100 | 69 | 0.0193 | 0.0000 to 0.0435 | 0.2504 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 512 | ndcg@10 | 69 | 0.2025 | 0.1541 to 0.2524 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 512 | mrr@10 | 69 | 0.2750 | 0.1932 to 0.3589 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 512 | recall@100 | 69 | 0.0085 | -0.0109 to 0.0302 | 0.5581 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | shared | 614 | ndcg@10 | 69 | 0.2122 | 0.1550 to 0.2731 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 614 | mrr@10 | 69 | 0.2400 | 0.1615 to 0.3202 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 614 | recall@100 | 69 | 0.0193 | 0.0000 to 0.0435 | 0.2504 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 614 | ndcg@10 | 69 | 0.1942 | 0.1476 to 0.2428 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 614 | mrr@10 | 69 | 0.2568 | 0.1786 to 0.3396 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 614 | recall@100 | 69 | 0.0085 | -0.0109 to 0.0302 | 0.5581 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | shared | 716 | ndcg@10 | 69 | 0.1985 | 0.1446 to 0.2575 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 716 | mrr@10 | 69 | 0.2291 | 0.1517 to 0.3088 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 716 | recall@100 | 69 | 0.0193 | 0.0000 to 0.0435 | 0.2504 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 716 | ndcg@10 | 69 | 0.1843 | 0.1404 to 0.2321 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 716 | mrr@10 | 69 | 0.2496 | 0.1734 to 0.3314 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 716 | recall@100 | 69 | 0.0085 | -0.0109 to 0.0302 | 0.5581 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | shared | 819 | ndcg@10 | 69 | 0.1852 | 0.1331 to 0.2415 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 819 | mrr@10 | 69 | 0.2116 | 0.1380 to 0.2877 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 819 | recall@100 | 69 | 0.0193 | 0.0000 to 0.0435 | 0.2504 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 819 | ndcg@10 | 69 | 0.1749 | 0.1314 to 0.2228 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 819 | mrr@10 | 69 | 0.2280 | 0.1543 to 0.3088 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 819 | recall@100 | 69 | 0.0105 | -0.0040 to 0.0290 | 0.4936 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | shared | 921 | ndcg@10 | 69 | 0.1424 | 0.1040 to 0.1842 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 921 | mrr@10 | 69 | 0.1545 | 0.0981 to 0.2143 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | shared | 921 | recall@100 | 69 | 0.0121 | 0.0000 to 0.0314 | 0.4951 | 1.0000 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 921 | ndcg@10 | 69 | 0.1338 | 0.0984 to 0.1703 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 921 | mrr@10 | 69 | 0.1651 | 0.1076 to 0.2286 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | scifact | 1-1 | 921 | recall@100 | 69 | 0.0141 | 0.0000 to 0.0322 | 0.2504 | 1.0000 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 102 | ndcg@10 | 2162 | 0.1943 | 0.1851 to 0.2042 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 102 | mrr@10 | 2162 | 0.2135 | 0.1982 to 0.2293 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 102 | recall@100 | 2162 | 0.2476 | 0.2381 to 0.2570 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 102 | ndcg@10 | 2162 | 0.1977 | 0.1901 to 0.2056 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 102 | mrr@10 | 2162 | 0.4079 | 0.3907 to 0.4253 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 102 | recall@100 | 2162 | 0.1372 | 0.1297 to 0.1445 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 204 | ndcg@10 | 2162 | 0.2131 | 0.2038 to 0.2223 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 204 | mrr@10 | 2162 | 0.2280 | 0.2123 to 0.2431 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 204 | recall@100 | 2162 | 0.2584 | 0.2488 to 0.2681 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 204 | ndcg@10 | 2162 | 0.2111 | 0.2040 to 0.2185 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 204 | mrr@10 | 2162 | 0.4056 | 0.3885 to 0.4227 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 204 | recall@100 | 2162 | 0.1494 | 0.1421 to 0.1566 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 307 | ndcg@10 | 2162 | 0.2075 | 0.1983 to 0.2163 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 307 | mrr@10 | 2162 | 0.2169 | 0.2022 to 0.2311 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 307 | recall@100 | 2162 | 0.2550 | 0.2450 to 0.2647 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 307 | ndcg@10 | 2162 | 0.2132 | 0.2063 to 0.2204 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 307 | mrr@10 | 2162 | 0.3969 | 0.3797 to 0.4136 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 307 | recall@100 | 2162 | 0.1530 | 0.1457 to 0.1602 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 409 | ndcg@10 | 2162 | 0.1960 | 0.1873 to 0.2047 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 409 | mrr@10 | 2162 | 0.2102 | 0.1953 to 0.2243 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 409 | recall@100 | 2162 | 0.2501 | 0.2405 to 0.2600 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 409 | ndcg@10 | 2162 | 0.2113 | 0.2044 to 0.2184 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 409 | mrr@10 | 2162 | 0.3851 | 0.3680 to 0.4014 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 409 | recall@100 | 2162 | 0.1528 | 0.1458 to 0.1599 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 512 | ndcg@10 | 2162 | 0.1842 | 0.1756 to 0.1927 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 512 | mrr@10 | 2162 | 0.1990 | 0.1851 to 0.2127 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 512 | recall@100 | 2162 | 0.2412 | 0.2315 to 0.2508 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 512 | ndcg@10 | 2162 | 0.2062 | 0.1996 to 0.2130 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 512 | mrr@10 | 2162 | 0.3693 | 0.3528 to 0.3851 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 512 | recall@100 | 2162 | 0.1512 | 0.1444 to 0.1580 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 614 | ndcg@10 | 2162 | 0.1732 | 0.1652 to 0.1814 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 614 | mrr@10 | 2162 | 0.1928 | 0.1794 to 0.2062 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 614 | recall@100 | 2162 | 0.2322 | 0.2227 to 0.2418 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 614 | ndcg@10 | 2162 | 0.2010 | 0.1945 to 0.2076 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 614 | mrr@10 | 2162 | 0.3546 | 0.3383 to 0.3705 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 614 | recall@100 | 2162 | 0.1497 | 0.1431 to 0.1564 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 716 | ndcg@10 | 2162 | 0.1605 | 0.1527 to 0.1685 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 716 | mrr@10 | 2162 | 0.1826 | 0.1692 to 0.1956 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 716 | recall@100 | 2162 | 0.2218 | 0.2121 to 0.2313 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 716 | ndcg@10 | 2162 | 0.1890 | 0.1828 to 0.1953 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 716 | mrr@10 | 2162 | 0.3257 | 0.3097 to 0.3416 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 716 | recall@100 | 2162 | 0.1460 | 0.1394 to 0.1525 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 819 | ndcg@10 | 2162 | 0.1303 | 0.1237 to 0.1374 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 819 | mrr@10 | 2162 | 0.1523 | 0.1405 to 0.1644 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 819 | recall@100 | 2162 | 0.1982 | 0.1896 to 0.2076 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 819 | ndcg@10 | 2162 | 0.1562 | 0.1505 to 0.1620 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 819 | mrr@10 | 2162 | 0.2551 | 0.2413 to 0.2690 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 819 | recall@100 | 2162 | 0.1359 | 0.1297 to 0.1421 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 921 | ndcg@10 | 2162 | 0.0788 | 0.0738 to 0.0838 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 921 | mrr@10 | 2162 | 0.0957 | 0.0860 to 0.1049 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | shared | 921 | recall@100 | 2162 | 0.1360 | 0.1288 to 0.1434 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 921 | ndcg@10 | 2162 | 0.0826 | 0.0788 to 0.0869 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 921 | mrr@10 | 2162 | 0.1198 | 0.1110 to 0.1286 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | nfcorpus | 1-1 | 921 | recall@100 | 2162 | 0.1050 | 0.0998 to 0.1102 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 102 | ndcg@10 | 2321 | 0.3267 | 0.3152 to 0.3372 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 102 | mrr@10 | 2321 | 0.3566 | 0.3418 to 0.3705 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 102 | recall@100 | 2321 | 0.2097 | 0.1971 to 0.2220 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 102 | ndcg@10 | 2321 | 0.3546 | 0.3456 to 0.3634 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 102 | mrr@10 | 2321 | 0.5609 | 0.5463 to 0.5757 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 102 | recall@100 | 2321 | 0.1089 | 0.0982 to 0.1188 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 204 | ndcg@10 | 2321 | 0.3181 | 0.3077 to 0.3284 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 204 | mrr@10 | 2321 | 0.3474 | 0.3335 to 0.3615 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 204 | recall@100 | 2321 | 0.2084 | 0.1957 to 0.2211 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 204 | ndcg@10 | 2321 | 0.3520 | 0.3433 to 0.3604 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 204 | mrr@10 | 2321 | 0.5478 | 0.5331 to 0.5629 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 204 | recall@100 | 2321 | 0.1202 | 0.1098 to 0.1296 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 307 | ndcg@10 | 2321 | 0.2944 | 0.2842 to 0.3037 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 307 | mrr@10 | 2321 | 0.3215 | 0.3082 to 0.3341 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 307 | recall@100 | 2321 | 0.2040 | 0.1914 to 0.2161 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 307 | ndcg@10 | 2321 | 0.3371 | 0.3287 to 0.3454 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 307 | mrr@10 | 2321 | 0.5146 | 0.5005 to 0.5290 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 307 | recall@100 | 2321 | 0.1233 | 0.1132 to 0.1327 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 409 | ndcg@10 | 2321 | 0.2676 | 0.2578 to 0.2766 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 409 | mrr@10 | 2321 | 0.2931 | 0.2798 to 0.3057 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 409 | recall@100 | 2321 | 0.1961 | 0.1833 to 0.2082 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 409 | ndcg@10 | 2321 | 0.3175 | 0.3090 to 0.3255 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 409 | mrr@10 | 2321 | 0.4735 | 0.4595 to 0.4876 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 409 | recall@100 | 2321 | 0.1258 | 0.1162 to 0.1350 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 512 | ndcg@10 | 2321 | 0.2407 | 0.2315 to 0.2493 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 512 | mrr@10 | 2321 | 0.2632 | 0.2512 to 0.2751 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 512 | recall@100 | 2321 | 0.1868 | 0.1743 to 0.1987 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 512 | ndcg@10 | 2321 | 0.2910 | 0.2829 to 0.2989 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 512 | mrr@10 | 2321 | 0.4218 | 0.4083 to 0.4352 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 512 | recall@100 | 2321 | 0.1256 | 0.1162 to 0.1343 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 614 | ndcg@10 | 2321 | 0.2149 | 0.2060 to 0.2228 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 614 | mrr@10 | 2321 | 0.2369 | 0.2251 to 0.2482 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 614 | recall@100 | 2321 | 0.1776 | 0.1657 to 0.1893 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 614 | ndcg@10 | 2321 | 0.2643 | 0.2563 to 0.2721 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 614 | mrr@10 | 2321 | 0.3744 | 0.3611 to 0.3877 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 614 | recall@100 | 2321 | 0.1273 | 0.1180 to 0.1359 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 716 | ndcg@10 | 2321 | 0.1897 | 0.1815 to 0.1973 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 716 | mrr@10 | 2321 | 0.2095 | 0.1983 to 0.2203 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 716 | recall@100 | 2321 | 0.1678 | 0.1559 to 0.1794 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 716 | ndcg@10 | 2321 | 0.2393 | 0.2317 to 0.2468 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 716 | mrr@10 | 2321 | 0.3304 | 0.3176 to 0.3431 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 716 | recall@100 | 2321 | 0.1262 | 0.1172 to 0.1344 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 819 | ndcg@10 | 2321 | 0.1677 | 0.1597 to 0.1747 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 819 | mrr@10 | 2321 | 0.1842 | 0.1737 to 0.1943 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 819 | recall@100 | 2321 | 0.1570 | 0.1456 to 0.1683 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 819 | ndcg@10 | 2321 | 0.2055 | 0.1983 to 0.2126 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 819 | mrr@10 | 2321 | 0.2750 | 0.2626 to 0.2863 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 819 | recall@100 | 2321 | 0.1228 | 0.1141 to 0.1310 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 921 | ndcg@10 | 2321 | 0.1214 | 0.1150 to 0.1274 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 921 | mrr@10 | 2321 | 0.1340 | 0.1252 to 0.1429 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | shared | 921 | recall@100 | 2321 | 0.1232 | 0.1134 to 0.1331 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 921 | ndcg@10 | 2321 | 0.1385 | 0.1328 to 0.1442 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 921 | mrr@10 | 2321 | 0.1739 | 0.1647 to 0.1828 | 0.0001 | 0.0162 |
| secondary: all-label, includes construction positives | msmarco | 1-1 | 921 | recall@100 | 2321 | 0.1072 | 0.0994 to 0.1146 | 0.0001 | 0.0162 |

## All-label gains against held-out positives

Same partitions, query-balanced. The construction columns show how far the documents that built the mask moved; the held-out columns show what happened to the others.

| environment | cell | k | queries | all-label nDCG@10 change | held-out-label nDCG@10 change | construction positive reciprocal rank change | held-out positive reciprocal rank change | construction positive in top 10 change | held-out positive in top 10 change |
|---|---|---|---|---|---|---|---|---|---|
| scifact | 1-1 | 102 | 69 | 0.2084 | -0.0628 | 0.5259 | -0.1013 | 0.1727 | 0.0507 |
| scifact | 1-1 | 204 | 69 | 0.2141 | -0.0576 | 0.5295 | -0.0959 | 0.1727 | 0.0507 |
| scifact | 1-1 | 307 | 69 | 0.2127 | -0.0559 | 0.5219 | -0.0932 | 0.1727 | 0.0489 |
| scifact | 1-1 | 409 | 69 | 0.2087 | -0.0408 | 0.4852 | -0.0710 | 0.1727 | 0.0441 |
| scifact | 1-1 | 512 | 69 | 0.2025 | -0.0339 | 0.4577 | -0.0593 | 0.1727 | 0.0368 |
| scifact | 1-1 | 614 | 69 | 0.1942 | -0.0345 | 0.4368 | -0.0586 | 0.1727 | 0.0332 |
| scifact | 1-1 | 716 | 69 | 0.1843 | -0.0378 | 0.4154 | -0.0587 | 0.1727 | 0.0248 |
| scifact | 1-1 | 819 | 69 | 0.1749 | -0.0143 | 0.3602 | -0.0318 | 0.1727 | 0.0344 |
| scifact | 1-1 | 921 | 69 | 0.1338 | -0.0114 | 0.2684 | -0.0237 | 0.1510 | 0.0229 |
| scifact | 2-2 | 102 | 6 | 0.0914 | -0.0893 | 0.2109 | -0.1128 | 0.0938 | 0.0521 |
| scifact | 2-2 | 204 | 6 | 0.0781 | -0.0858 | 0.1904 | -0.1068 | 0.0938 | 0.0451 |
| scifact | 2-2 | 307 | 6 | 0.0822 | -0.0630 | 0.1671 | -0.0925 | 0.0938 | 0.0729 |
| scifact | 2-2 | 409 | 6 | 0.0824 | -0.0539 | 0.1573 | -0.0830 | 0.0938 | 0.0729 |
| scifact | 2-2 | 512 | 6 | 0.0759 | -0.0381 | 0.1344 | -0.0704 | 0.0938 | 0.0833 |
| scifact | 2-2 | 614 | 6 | 0.0720 | -0.0231 | 0.1121 | -0.0543 | 0.0938 | 0.0833 |
| scifact | 2-2 | 716 | 6 | 0.0717 | -0.0194 | 0.1078 | -0.0506 | 0.0938 | 0.0833 |
| scifact | 2-2 | 819 | 6 | 0.0721 | -0.0110 | 0.0984 | -0.0423 | 0.0938 | 0.0833 |
| scifact | 2-2 | 921 | 6 | 0.0602 | -0.0208 | 0.0908 | -0.0357 | 0.0938 | 0.0347 |
| nfcorpus | 1-1 | 102 | 2162 | 0.1977 | 0.0016 | 0.8619 | -0.0070 | 0.8629 | 0.0358 |
| nfcorpus | 1-1 | 204 | 2162 | 0.2111 | 0.0057 | 0.8459 | -0.0047 | 0.8627 | 0.0461 |
| nfcorpus | 1-1 | 307 | 2162 | 0.2132 | 0.0079 | 0.8115 | -0.0021 | 0.8612 | 0.0469 |
| nfcorpus | 1-1 | 409 | 2162 | 0.2113 | 0.0094 | 0.7686 | 0.0003 | 0.8577 | 0.0461 |
| nfcorpus | 1-1 | 512 | 2162 | 0.2062 | 0.0102 | 0.7181 | 0.0021 | 0.8508 | 0.0437 |
| nfcorpus | 1-1 | 614 | 2162 | 0.2010 | 0.0109 | 0.6743 | 0.0035 | 0.8410 | 0.0422 |
| nfcorpus | 1-1 | 716 | 2162 | 0.1890 | 0.0111 | 0.6015 | 0.0045 | 0.8127 | 0.0395 |
| nfcorpus | 1-1 | 819 | 2162 | 0.1562 | 0.0106 | 0.4470 | 0.0057 | 0.6954 | 0.0328 |
| nfcorpus | 1-1 | 921 | 2162 | 0.0826 | 0.0079 | 0.1995 | 0.0058 | 0.3779 | 0.0190 |
| nfcorpus | 2-2 | 102 | 1922 | 0.1909 | 0.0103 | 0.3626 | 0.0027 | 0.6193 | 0.0369 |
| nfcorpus | 2-2 | 204 | 1922 | 0.2051 | 0.0156 | 0.3532 | 0.0060 | 0.6166 | 0.0455 |
| nfcorpus | 2-2 | 307 | 1922 | 0.1998 | 0.0171 | 0.3273 | 0.0076 | 0.5868 | 0.0454 |
| nfcorpus | 2-2 | 409 | 1922 | 0.1902 | 0.0185 | 0.2991 | 0.0086 | 0.5492 | 0.0463 |
| nfcorpus | 2-2 | 512 | 1922 | 0.1775 | 0.0185 | 0.2691 | 0.0093 | 0.5074 | 0.0440 |
| nfcorpus | 2-2 | 614 | 1922 | 0.1662 | 0.0184 | 0.2451 | 0.0098 | 0.4709 | 0.0420 |
| nfcorpus | 2-2 | 716 | 1922 | 0.1501 | 0.0174 | 0.2161 | 0.0097 | 0.4234 | 0.0383 |
| nfcorpus | 2-2 | 819 | 1922 | 0.1161 | 0.0147 | 0.1595 | 0.0086 | 0.3213 | 0.0311 |
| nfcorpus | 2-2 | 921 | 1922 | 0.0637 | 0.0105 | 0.0804 | 0.0067 | 0.1623 | 0.0198 |
| nfcorpus | 5-5 | 102 | 1517 | 0.1213 | 0.0111 | 0.0895 | 0.0048 | 0.1982 | 0.0219 |
| nfcorpus | 5-5 | 204 | 1517 | 0.1391 | 0.0163 | 0.0901 | 0.0071 | 0.2013 | 0.0284 |
| nfcorpus | 5-5 | 307 | 1517 | 0.1364 | 0.0181 | 0.0841 | 0.0081 | 0.1895 | 0.0296 |
| nfcorpus | 5-5 | 409 | 1517 | 0.1301 | 0.0180 | 0.0778 | 0.0081 | 0.1759 | 0.0289 |
| nfcorpus | 5-5 | 512 | 1517 | 0.1220 | 0.0181 | 0.0710 | 0.0082 | 0.1601 | 0.0282 |
| nfcorpus | 5-5 | 614 | 1517 | 0.1138 | 0.0177 | 0.0648 | 0.0082 | 0.1463 | 0.0269 |
| nfcorpus | 5-5 | 716 | 1517 | 0.1045 | 0.0167 | 0.0592 | 0.0078 | 0.1329 | 0.0251 |
| nfcorpus | 5-5 | 819 | 1517 | 0.0847 | 0.0139 | 0.0471 | 0.0068 | 0.1057 | 0.0204 |
| nfcorpus | 5-5 | 921 | 1517 | 0.0521 | 0.0094 | 0.0277 | 0.0046 | 0.0608 | 0.0135 |
| nfcorpus | 10-10 | 102 | 1105 | 0.0743 | 0.0014 | 0.0310 | 0.0012 | 0.0714 | 0.0053 |
| nfcorpus | 10-10 | 204 | 1105 | 0.0921 | 0.0055 | 0.0317 | 0.0027 | 0.0749 | 0.0097 |
| nfcorpus | 10-10 | 307 | 1105 | 0.0921 | 0.0072 | 0.0305 | 0.0033 | 0.0713 | 0.0112 |
| nfcorpus | 10-10 | 409 | 1105 | 0.0885 | 0.0073 | 0.0285 | 0.0032 | 0.0657 | 0.0114 |
| nfcorpus | 10-10 | 512 | 1105 | 0.0831 | 0.0073 | 0.0263 | 0.0033 | 0.0606 | 0.0108 |
| nfcorpus | 10-10 | 614 | 1105 | 0.0778 | 0.0073 | 0.0243 | 0.0033 | 0.0552 | 0.0105 |
| nfcorpus | 10-10 | 716 | 1105 | 0.0724 | 0.0072 | 0.0223 | 0.0032 | 0.0509 | 0.0102 |
| nfcorpus | 10-10 | 819 | 1105 | 0.0613 | 0.0065 | 0.0185 | 0.0029 | 0.0418 | 0.0087 |
| nfcorpus | 10-10 | 921 | 1105 | 0.0402 | 0.0046 | 0.0118 | 0.0021 | 0.0263 | 0.0061 |
| nfcorpus | 20-20 | 102 | 611 | 0.0411 | -0.0019 | 0.0111 | 0.0005 | 0.0237 | -0.0000 |
| nfcorpus | 20-20 | 204 | 611 | 0.0627 | 0.0043 | 0.0122 | 0.0017 | 0.0273 | 0.0033 |
| nfcorpus | 20-20 | 307 | 611 | 0.0635 | 0.0054 | 0.0118 | 0.0020 | 0.0261 | 0.0035 |
| nfcorpus | 20-20 | 409 | 611 | 0.0619 | 0.0057 | 0.0111 | 0.0020 | 0.0244 | 0.0036 |
| nfcorpus | 20-20 | 512 | 611 | 0.0583 | 0.0061 | 0.0102 | 0.0021 | 0.0226 | 0.0036 |
| nfcorpus | 20-20 | 614 | 611 | 0.0548 | 0.0057 | 0.0095 | 0.0020 | 0.0211 | 0.0033 |
| nfcorpus | 20-20 | 716 | 611 | 0.0512 | 0.0060 | 0.0088 | 0.0020 | 0.0192 | 0.0034 |
| nfcorpus | 20-20 | 819 | 611 | 0.0442 | 0.0051 | 0.0075 | 0.0017 | 0.0160 | 0.0028 |
| nfcorpus | 20-20 | 921 | 611 | 0.0298 | 0.0043 | 0.0050 | 0.0013 | 0.0102 | 0.0024 |
| msmarco | 1-1 | 102 | 2321 | 0.3546 | -0.0519 | 0.6623 | -0.0578 | 0.5953 | -0.0295 |
| msmarco | 1-1 | 204 | 2321 | 0.3520 | -0.0435 | 0.6467 | -0.0527 | 0.5887 | -0.0106 |
| msmarco | 1-1 | 307 | 2321 | 0.3371 | -0.0387 | 0.6094 | -0.0479 | 0.5798 | -0.0061 |
| msmarco | 1-1 | 409 | 2321 | 0.3175 | -0.0336 | 0.5639 | -0.0431 | 0.5629 | 0.0003 |
| msmarco | 1-1 | 512 | 2321 | 0.2910 | -0.0291 | 0.5073 | -0.0388 | 0.5376 | 0.0048 |
| msmarco | 1-1 | 614 | 2321 | 0.2643 | -0.0260 | 0.4535 | -0.0340 | 0.5092 | 0.0029 |
| msmarco | 1-1 | 716 | 2321 | 0.2393 | -0.0221 | 0.4036 | -0.0298 | 0.4772 | 0.0054 |
| msmarco | 1-1 | 819 | 2321 | 0.2055 | -0.0204 | 0.3427 | -0.0272 | 0.4286 | 0.0046 |
| msmarco | 1-1 | 921 | 2321 | 0.1385 | -0.0123 | 0.2227 | -0.0174 | 0.3111 | 0.0067 |
| msmarco | 2-2 | 102 | 81 | 0.2129 | -0.0258 | 0.2620 | -0.0179 | 0.4408 | -0.0213 |
| msmarco | 2-2 | 204 | 81 | 0.2086 | -0.0158 | 0.2455 | -0.0125 | 0.4326 | -0.0041 |
| msmarco | 2-2 | 307 | 81 | 0.1909 | -0.0136 | 0.2218 | -0.0119 | 0.4077 | 0.0010 |
| msmarco | 2-2 | 409 | 81 | 0.1706 | -0.0081 | 0.1937 | -0.0082 | 0.3650 | 0.0067 |
| msmarco | 2-2 | 512 | 81 | 0.1513 | -0.0077 | 0.1715 | -0.0074 | 0.3266 | 0.0044 |
| msmarco | 2-2 | 614 | 81 | 0.1286 | -0.0067 | 0.1492 | -0.0066 | 0.2729 | 0.0046 |
| msmarco | 2-2 | 716 | 81 | 0.1141 | -0.0078 | 0.1360 | -0.0066 | 0.2397 | 0.0010 |
| msmarco | 2-2 | 819 | 81 | 0.1008 | -0.0054 | 0.1179 | -0.0036 | 0.2145 | -0.0005 |
| msmarco | 2-2 | 921 | 81 | 0.0734 | -0.0041 | 0.0837 | -0.0029 | 0.1633 | -0.0005 |

## Ranking metrics, full training judgments

These count construction positives as relevant. Read them with the table above.

| environment | arm | k | metric | queries | mean | full query, same queries | difference | 95% interval (pointwise) |
|---|---|---|---|---|---|---|---|---|
| scifact | full query | 1024 | nDCG@10 | 69 | 0.6786 |  |  |  |
| scifact | prefix | 102 | nDCG@10 | 69 | 0.5324 | 0.6786 | -0.1462 | -0.2031 to -0.0939 |
| scifact | prefix | 204 | nDCG@10 | 69 | 0.6125 | 0.6786 | -0.0661 | -0.1029 to -0.0318 |
| scifact | prefix | 307 | nDCG@10 | 69 | 0.6268 | 0.6786 | -0.0518 | -0.0813 to -0.0243 |
| scifact | prefix | 409 | nDCG@10 | 69 | 0.6412 | 0.6786 | -0.0374 | -0.0686 to -0.0116 |
| scifact | prefix | 512 | nDCG@10 | 69 | 0.6384 | 0.6786 | -0.0402 | -0.0734 to -0.0108 |
| scifact | prefix | 614 | nDCG@10 | 69 | 0.6566 | 0.6786 | -0.0220 | -0.0507 to 0.0038 |
| scifact | prefix | 716 | nDCG@10 | 69 | 0.6701 | 0.6786 | -0.0084 | -0.0309 to 0.0123 |
| scifact | prefix | 819 | nDCG@10 | 69 | 0.6658 | 0.6786 | -0.0127 | -0.0392 to 0.0103 |
| scifact | prefix | 921 | nDCG@10 | 69 | 0.6696 | 0.6786 | -0.0090 | -0.0228 to 0.0029 |
| scifact | shared | 102 | nDCG@10 | 69 | 0.9410 | 0.6786 | 0.2624 | 0.1980 to 0.3289 |
| scifact | shared | 204 | nDCG@10 | 69 | 0.9390 | 0.6786 | 0.2604 | 0.1973 to 0.3257 |
| scifact | shared | 307 | nDCG@10 | 69 | 0.9242 | 0.6786 | 0.2457 | 0.1869 to 0.3085 |
| scifact | shared | 409 | nDCG@10 | 69 | 0.9134 | 0.6786 | 0.2348 | 0.1756 to 0.2968 |
| scifact | shared | 512 | nDCG@10 | 69 | 0.9052 | 0.6786 | 0.2266 | 0.1677 to 0.2901 |
| scifact | shared | 614 | nDCG@10 | 69 | 0.8908 | 0.6786 | 0.2122 | 0.1550 to 0.2731 |
| scifact | shared | 716 | nDCG@10 | 69 | 0.8771 | 0.6786 | 0.1985 | 0.1446 to 0.2575 |
| scifact | shared | 819 | nDCG@10 | 69 | 0.8637 | 0.6786 | 0.1852 | 0.1331 to 0.2415 |
| scifact | shared | 921 | nDCG@10 | 69 | 0.8210 | 0.6786 | 0.1424 | 0.1040 to 0.1842 |
| scifact | 1-1 | 102 | nDCG@10 | 69 | 0.8870 | 0.6786 | 0.2084 | 0.1564 to 0.2623 |
| scifact | 1-1 | 204 | nDCG@10 | 69 | 0.8927 | 0.6786 | 0.2141 | 0.1618 to 0.2690 |
| scifact | 1-1 | 307 | nDCG@10 | 69 | 0.8913 | 0.6786 | 0.2127 | 0.1587 to 0.2678 |
| scifact | 1-1 | 409 | nDCG@10 | 69 | 0.8873 | 0.6786 | 0.2087 | 0.1570 to 0.2613 |
| scifact | 1-1 | 512 | nDCG@10 | 69 | 0.8811 | 0.6786 | 0.2025 | 0.1541 to 0.2524 |
| scifact | 1-1 | 614 | nDCG@10 | 69 | 0.8728 | 0.6786 | 0.1942 | 0.1476 to 0.2428 |
| scifact | 1-1 | 716 | nDCG@10 | 69 | 0.8629 | 0.6786 | 0.1843 | 0.1404 to 0.2321 |
| scifact | 1-1 | 819 | nDCG@10 | 69 | 0.8535 | 0.6786 | 0.1749 | 0.1314 to 0.2228 |
| scifact | 1-1 | 921 | nDCG@10 | 69 | 0.8124 | 0.6786 | 0.1338 | 0.0984 to 0.1703 |
| scifact | 2-2 | 102 | nDCG@10 | 6 | 0.8845 | 0.7931 | 0.0914 | 0.0277 to 0.1653 |
| scifact | 2-2 | 204 | nDCG@10 | 6 | 0.8712 | 0.7931 | 0.0781 | 0.0254 to 0.1313 |
| scifact | 2-2 | 307 | nDCG@10 | 6 | 0.8753 | 0.7931 | 0.0822 | 0.0242 to 0.1420 |
| scifact | 2-2 | 409 | nDCG@10 | 6 | 0.8755 | 0.7931 | 0.0824 | 0.0242 to 0.1426 |
| scifact | 2-2 | 512 | nDCG@10 | 6 | 0.8690 | 0.7931 | 0.0759 | 0.0218 to 0.1327 |
| scifact | 2-2 | 614 | nDCG@10 | 6 | 0.8651 | 0.7931 | 0.0720 | 0.0228 to 0.1243 |
| scifact | 2-2 | 716 | nDCG@10 | 6 | 0.8648 | 0.7931 | 0.0717 | 0.0208 to 0.1235 |
| scifact | 2-2 | 819 | nDCG@10 | 6 | 0.8652 | 0.7931 | 0.0721 | 0.0226 to 0.1227 |
| scifact | 2-2 | 921 | nDCG@10 | 6 | 0.8534 | 0.7931 | 0.0602 | 0.0225 to 0.0991 |
| scifact | full query | 1024 | MRR@10 | 69 | 0.6960 |  |  |  |
| scifact | prefix | 102 | MRR@10 | 69 | 0.5788 | 0.6960 | -0.1172 | -0.1870 to -0.0465 |
| scifact | prefix | 204 | MRR@10 | 69 | 0.6603 | 0.6960 | -0.0357 | -0.0964 to 0.0234 |
| scifact | prefix | 307 | MRR@10 | 69 | 0.6734 | 0.6960 | -0.0226 | -0.0757 to 0.0268 |
| scifact | prefix | 409 | MRR@10 | 69 | 0.6977 | 0.6960 | 0.0017 | -0.0437 to 0.0458 |
| scifact | prefix | 512 | MRR@10 | 69 | 0.6888 | 0.6960 | -0.0072 | -0.0575 to 0.0394 |
| scifact | prefix | 614 | MRR@10 | 69 | 0.6787 | 0.6960 | -0.0173 | -0.0632 to 0.0286 |
| scifact | prefix | 716 | MRR@10 | 69 | 0.6970 | 0.6960 | 0.0010 | -0.0376 to 0.0382 |
| scifact | prefix | 819 | MRR@10 | 69 | 0.6986 | 0.6960 | 0.0025 | -0.0295 to 0.0327 |
| scifact | prefix | 921 | MRR@10 | 69 | 0.6834 | 0.6960 | -0.0126 | -0.0374 to 0.0043 |
| scifact | shared | 102 | MRR@10 | 69 | 0.9498 | 0.6960 | 0.2537 | 0.1740 to 0.3414 |
| scifact | shared | 204 | MRR@10 | 69 | 0.9589 | 0.6960 | 0.2629 | 0.1787 to 0.3487 |
| scifact | shared | 307 | MRR@10 | 69 | 0.9541 | 0.6960 | 0.2581 | 0.1758 to 0.3440 |
| scifact | shared | 409 | MRR@10 | 69 | 0.9517 | 0.6960 | 0.2557 | 0.1749 to 0.3396 |
| scifact | shared | 512 | MRR@10 | 69 | 0.9493 | 0.6960 | 0.2533 | 0.1722 to 0.3372 |
| scifact | shared | 614 | MRR@10 | 69 | 0.9360 | 0.6960 | 0.2400 | 0.1615 to 0.3202 |
| scifact | shared | 716 | MRR@10 | 69 | 0.9251 | 0.6960 | 0.2291 | 0.1517 to 0.3088 |
| scifact | shared | 819 | MRR@10 | 69 | 0.9076 | 0.6960 | 0.2116 | 0.1380 to 0.2877 |
| scifact | shared | 921 | MRR@10 | 69 | 0.8505 | 0.6960 | 0.1545 | 0.0981 to 0.2143 |
| scifact | 1-1 | 102 | MRR@10 | 69 | 1.0000 | 0.6960 | 0.3040 | 0.2182 to 0.3902 |
| scifact | 1-1 | 204 | MRR@10 | 69 | 1.0000 | 0.6960 | 0.3040 | 0.2182 to 0.3902 |
| scifact | 1-1 | 307 | MRR@10 | 69 | 0.9949 | 0.6960 | 0.2989 | 0.2130 to 0.3833 |
| scifact | 1-1 | 409 | MRR@10 | 69 | 0.9867 | 0.6960 | 0.2907 | 0.2072 to 0.3739 |
| scifact | 1-1 | 512 | MRR@10 | 69 | 0.9710 | 0.6960 | 0.2750 | 0.1932 to 0.3589 |
| scifact | 1-1 | 614 | MRR@10 | 69 | 0.9528 | 0.6960 | 0.2568 | 0.1786 to 0.3396 |
| scifact | 1-1 | 716 | MRR@10 | 69 | 0.9456 | 0.6960 | 0.2496 | 0.1734 to 0.3314 |
| scifact | 1-1 | 819 | MRR@10 | 69 | 0.9240 | 0.6960 | 0.2280 | 0.1543 to 0.3088 |
| scifact | 1-1 | 921 | MRR@10 | 69 | 0.8612 | 0.6960 | 0.1651 | 0.1076 to 0.2286 |
| scifact | 2-2 | 102 | MRR@10 | 6 | 0.8843 | 0.8056 | 0.0787 | 0.0000 to 0.2299 |
| scifact | 2-2 | 204 | MRR@10 | 6 | 0.8611 | 0.8056 | 0.0556 | 0.0000 to 0.1250 |
| scifact | 2-2 | 307 | MRR@10 | 6 | 0.8426 | 0.8056 | 0.0370 | 0.0000 to 0.1111 |
| scifact | 2-2 | 409 | MRR@10 | 6 | 0.8333 | 0.8056 | 0.0278 | 0.0000 to 0.0833 |
| scifact | 2-2 | 512 | MRR@10 | 6 | 0.8287 | 0.8056 | 0.0231 | -0.0139 to 0.0833 |
| scifact | 2-2 | 614 | MRR@10 | 6 | 0.8241 | 0.8056 | 0.0185 | -0.0139 to 0.0694 |
| scifact | 2-2 | 716 | MRR@10 | 6 | 0.8241 | 0.8056 | 0.0185 | -0.0139 to 0.0694 |
| scifact | 2-2 | 819 | MRR@10 | 6 | 0.8241 | 0.8056 | 0.0185 | -0.0139 to 0.0694 |
| scifact | 2-2 | 921 | MRR@10 | 6 | 0.8287 | 0.8056 | 0.0231 | 0.0000 to 0.0694 |
| scifact | full query | 1024 | MRR@100 | 69 | 0.6992 |  |  |  |
| scifact | prefix | 102 | MRR@100 | 69 | 0.5839 | 0.6992 | -0.1153 | -0.1840 to -0.0448 |
| scifact | prefix | 204 | MRR@100 | 69 | 0.6626 | 0.6992 | -0.0366 | -0.0977 to 0.0223 |
| scifact | prefix | 307 | MRR@100 | 69 | 0.6755 | 0.6992 | -0.0237 | -0.0764 to 0.0256 |
| scifact | prefix | 409 | MRR@100 | 69 | 0.7010 | 0.6992 | 0.0018 | -0.0433 to 0.0455 |
| scifact | prefix | 512 | MRR@100 | 69 | 0.6908 | 0.6992 | -0.0084 | -0.0584 to 0.0385 |
| scifact | prefix | 614 | MRR@100 | 69 | 0.6810 | 0.6992 | -0.0182 | -0.0639 to 0.0275 |
| scifact | prefix | 716 | MRR@100 | 69 | 0.7009 | 0.6992 | 0.0017 | -0.0365 to 0.0387 |
| scifact | prefix | 819 | MRR@100 | 69 | 0.7042 | 0.6992 | 0.0050 | -0.0264 to 0.0353 |
| scifact | prefix | 921 | MRR@100 | 69 | 0.6863 | 0.6992 | -0.0129 | -0.0376 to 0.0040 |
| scifact | shared | 102 | MRR@100 | 69 | 0.9498 | 0.6992 | 0.2505 | 0.1727 to 0.3368 |
| scifact | shared | 204 | MRR@100 | 69 | 0.9589 | 0.6992 | 0.2597 | 0.1773 to 0.3444 |
| scifact | shared | 307 | MRR@100 | 69 | 0.9541 | 0.6992 | 0.2549 | 0.1743 to 0.3395 |
| scifact | shared | 409 | MRR@100 | 69 | 0.9517 | 0.6992 | 0.2525 | 0.1734 to 0.3349 |
| scifact | shared | 512 | MRR@100 | 69 | 0.9493 | 0.6992 | 0.2501 | 0.1706 to 0.3326 |
| scifact | shared | 614 | MRR@100 | 69 | 0.9360 | 0.6992 | 0.2368 | 0.1591 to 0.3164 |
| scifact | shared | 716 | MRR@100 | 69 | 0.9251 | 0.6992 | 0.2259 | 0.1498 to 0.3048 |
| scifact | shared | 819 | MRR@100 | 69 | 0.9076 | 0.6992 | 0.2084 | 0.1349 to 0.2828 |
| scifact | shared | 921 | MRR@100 | 69 | 0.8505 | 0.6992 | 0.1513 | 0.0950 to 0.2111 |
| scifact | 1-1 | 102 | MRR@100 | 69 | 1.0000 | 0.6992 | 0.3008 | 0.2163 to 0.3849 |
| scifact | 1-1 | 204 | MRR@100 | 69 | 1.0000 | 0.6992 | 0.3008 | 0.2163 to 0.3849 |
| scifact | 1-1 | 307 | MRR@100 | 69 | 0.9949 | 0.6992 | 0.2957 | 0.2107 to 0.3788 |
| scifact | 1-1 | 409 | MRR@100 | 69 | 0.9867 | 0.6992 | 0.2875 | 0.2058 to 0.3695 |
| scifact | 1-1 | 512 | MRR@100 | 69 | 0.9710 | 0.6992 | 0.2718 | 0.1910 to 0.3539 |
| scifact | 1-1 | 614 | MRR@100 | 69 | 0.9528 | 0.6992 | 0.2536 | 0.1769 to 0.3354 |
| scifact | 1-1 | 716 | MRR@100 | 69 | 0.9456 | 0.6992 | 0.2464 | 0.1721 to 0.3274 |
| scifact | 1-1 | 819 | MRR@100 | 69 | 0.9240 | 0.6992 | 0.2248 | 0.1519 to 0.3042 |
| scifact | 1-1 | 921 | MRR@100 | 69 | 0.8627 | 0.6992 | 0.1635 | 0.1063 to 0.2261 |
| scifact | 2-2 | 102 | MRR@100 | 6 | 0.8843 | 0.8056 | 0.0787 | 0.0000 to 0.2299 |
| scifact | 2-2 | 204 | MRR@100 | 6 | 0.8611 | 0.8056 | 0.0556 | 0.0000 to 0.1250 |
| scifact | 2-2 | 307 | MRR@100 | 6 | 0.8426 | 0.8056 | 0.0370 | 0.0000 to 0.1111 |
| scifact | 2-2 | 409 | MRR@100 | 6 | 0.8333 | 0.8056 | 0.0278 | 0.0000 to 0.0833 |
| scifact | 2-2 | 512 | MRR@100 | 6 | 0.8287 | 0.8056 | 0.0231 | -0.0139 to 0.0833 |
| scifact | 2-2 | 614 | MRR@100 | 6 | 0.8241 | 0.8056 | 0.0185 | -0.0139 to 0.0694 |
| scifact | 2-2 | 716 | MRR@100 | 6 | 0.8241 | 0.8056 | 0.0185 | -0.0139 to 0.0694 |
| scifact | 2-2 | 819 | MRR@100 | 6 | 0.8241 | 0.8056 | 0.0185 | -0.0139 to 0.0694 |
| scifact | 2-2 | 921 | MRR@100 | 6 | 0.8287 | 0.8056 | 0.0231 | 0.0000 to 0.0694 |
| scifact | full query | 1024 | Recall@100 | 69 | 0.9807 |  |  |  |
| scifact | prefix | 102 | Recall@100 | 69 | 0.9251 | 0.9807 | -0.0556 | -0.1111 to -0.0121 |
| scifact | prefix | 204 | Recall@100 | 69 | 0.9565 | 0.9807 | -0.0242 | -0.0652 to 0.0072 |
| scifact | prefix | 307 | Recall@100 | 69 | 0.9638 | 0.9807 | -0.0169 | -0.0531 to 0.0097 |
| scifact | prefix | 409 | Recall@100 | 69 | 0.9855 | 0.9807 | 0.0048 | 0.0000 to 0.0145 |
| scifact | prefix | 512 | Recall@100 | 69 | 0.9855 | 0.9807 | 0.0048 | 0.0000 to 0.0145 |
| scifact | prefix | 614 | Recall@100 | 69 | 0.9807 | 0.9807 | 0.0000 | 0.0000 to 0.0000 |
| scifact | prefix | 716 | Recall@100 | 69 | 0.9855 | 0.9807 | 0.0048 | 0.0000 to 0.0145 |
| scifact | prefix | 819 | Recall@100 | 69 | 0.9807 | 0.9807 | 0.0000 | 0.0000 to 0.0000 |
| scifact | prefix | 921 | Recall@100 | 69 | 0.9807 | 0.9807 | 0.0000 | 0.0000 to 0.0000 |
| scifact | shared | 102 | Recall@100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 204 | Recall@100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 307 | Recall@100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 409 | Recall@100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 512 | Recall@100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 614 | Recall@100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 716 | Recall@100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 819 | Recall@100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 921 | Recall@100 | 69 | 0.9928 | 0.9807 | 0.0121 | 0.0000 to 0.0314 |
| scifact | 1-1 | 102 | Recall@100 | 69 | 0.9875 | 0.9807 | 0.0068 | -0.0109 to 0.0282 |
| scifact | 1-1 | 204 | Recall@100 | 69 | 0.9875 | 0.9807 | 0.0068 | -0.0109 to 0.0282 |
| scifact | 1-1 | 307 | Recall@100 | 69 | 0.9891 | 0.9807 | 0.0085 | -0.0109 to 0.0302 |
| scifact | 1-1 | 409 | Recall@100 | 69 | 0.9891 | 0.9807 | 0.0085 | -0.0109 to 0.0302 |
| scifact | 1-1 | 512 | Recall@100 | 69 | 0.9891 | 0.9807 | 0.0085 | -0.0109 to 0.0302 |
| scifact | 1-1 | 614 | Recall@100 | 69 | 0.9891 | 0.9807 | 0.0085 | -0.0109 to 0.0302 |
| scifact | 1-1 | 716 | Recall@100 | 69 | 0.9891 | 0.9807 | 0.0085 | -0.0109 to 0.0302 |
| scifact | 1-1 | 819 | Recall@100 | 69 | 0.9911 | 0.9807 | 0.0105 | -0.0040 to 0.0290 |
| scifact | 1-1 | 921 | Recall@100 | 69 | 0.9948 | 0.9807 | 0.0141 | 0.0000 to 0.0322 |
| scifact | 2-2 | 102 | Recall@100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 204 | Recall@100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 307 | Recall@100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 409 | Recall@100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 512 | Recall@100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 614 | Recall@100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 716 | Recall@100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 819 | Recall@100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 921 | Recall@100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | full query | 1024 | nDCG@10 | 2162 | 0.3479 |  |  |  |
| nfcorpus | prefix | 102 | nDCG@10 | 2162 | 0.2599 | 0.3479 | -0.0880 | -0.0947 to -0.0815 |
| nfcorpus | prefix | 204 | nDCG@10 | 2162 | 0.2949 | 0.3479 | -0.0530 | -0.0583 to -0.0481 |
| nfcorpus | prefix | 307 | nDCG@10 | 2162 | 0.3121 | 0.3479 | -0.0358 | -0.0402 to -0.0318 |
| nfcorpus | prefix | 409 | nDCG@10 | 2162 | 0.3250 | 0.3479 | -0.0229 | -0.0268 to -0.0192 |
| nfcorpus | prefix | 512 | nDCG@10 | 2162 | 0.3318 | 0.3479 | -0.0161 | -0.0191 to -0.0129 |
| nfcorpus | prefix | 614 | nDCG@10 | 2162 | 0.3375 | 0.3479 | -0.0104 | -0.0131 to -0.0076 |
| nfcorpus | prefix | 716 | nDCG@10 | 2162 | 0.3410 | 0.3479 | -0.0070 | -0.0092 to -0.0047 |
| nfcorpus | prefix | 819 | nDCG@10 | 2162 | 0.3437 | 0.3479 | -0.0042 | -0.0063 to -0.0022 |
| nfcorpus | prefix | 921 | nDCG@10 | 2162 | 0.3450 | 0.3479 | -0.0029 | -0.0044 to -0.0015 |
| nfcorpus | shared | 102 | nDCG@10 | 2162 | 0.5423 | 0.3479 | 0.1943 | 0.1851 to 0.2042 |
| nfcorpus | shared | 204 | nDCG@10 | 2162 | 0.5611 | 0.3479 | 0.2131 | 0.2038 to 0.2223 |
| nfcorpus | shared | 307 | nDCG@10 | 2162 | 0.5554 | 0.3479 | 0.2075 | 0.1983 to 0.2163 |
| nfcorpus | shared | 409 | nDCG@10 | 2162 | 0.5440 | 0.3479 | 0.1960 | 0.1873 to 0.2047 |
| nfcorpus | shared | 512 | nDCG@10 | 2162 | 0.5321 | 0.3479 | 0.1842 | 0.1756 to 0.1927 |
| nfcorpus | shared | 614 | nDCG@10 | 2162 | 0.5212 | 0.3479 | 0.1732 | 0.1652 to 0.1814 |
| nfcorpus | shared | 716 | nDCG@10 | 2162 | 0.5084 | 0.3479 | 0.1605 | 0.1527 to 0.1685 |
| nfcorpus | shared | 819 | nDCG@10 | 2162 | 0.4782 | 0.3479 | 0.1303 | 0.1237 to 0.1374 |
| nfcorpus | shared | 921 | nDCG@10 | 2162 | 0.4268 | 0.3479 | 0.0788 | 0.0738 to 0.0838 |
| nfcorpus | 1-1 | 102 | nDCG@10 | 2162 | 0.5456 | 0.3479 | 0.1977 | 0.1901 to 0.2056 |
| nfcorpus | 1-1 | 204 | nDCG@10 | 2162 | 0.5590 | 0.3479 | 0.2111 | 0.2040 to 0.2185 |
| nfcorpus | 1-1 | 307 | nDCG@10 | 2162 | 0.5612 | 0.3479 | 0.2132 | 0.2063 to 0.2204 |
| nfcorpus | 1-1 | 409 | nDCG@10 | 2162 | 0.5593 | 0.3479 | 0.2113 | 0.2044 to 0.2184 |
| nfcorpus | 1-1 | 512 | nDCG@10 | 2162 | 0.5541 | 0.3479 | 0.2062 | 0.1996 to 0.2130 |
| nfcorpus | 1-1 | 614 | nDCG@10 | 2162 | 0.5490 | 0.3479 | 0.2010 | 0.1945 to 0.2076 |
| nfcorpus | 1-1 | 716 | nDCG@10 | 2162 | 0.5370 | 0.3479 | 0.1890 | 0.1828 to 0.1953 |
| nfcorpus | 1-1 | 819 | nDCG@10 | 2162 | 0.5041 | 0.3479 | 0.1562 | 0.1505 to 0.1620 |
| nfcorpus | 1-1 | 921 | nDCG@10 | 2162 | 0.4306 | 0.3479 | 0.0826 | 0.0788 to 0.0869 |
| nfcorpus | 2-2 | 102 | nDCG@10 | 1922 | 0.5487 | 0.3578 | 0.1909 | 0.1828 to 0.1993 |
| nfcorpus | 2-2 | 204 | nDCG@10 | 1922 | 0.5629 | 0.3578 | 0.2051 | 0.1974 to 0.2132 |
| nfcorpus | 2-2 | 307 | nDCG@10 | 1922 | 0.5575 | 0.3578 | 0.1998 | 0.1923 to 0.2079 |
| nfcorpus | 2-2 | 409 | nDCG@10 | 1922 | 0.5480 | 0.3578 | 0.1902 | 0.1829 to 0.1980 |
| nfcorpus | 2-2 | 512 | nDCG@10 | 1922 | 0.5352 | 0.3578 | 0.1775 | 0.1703 to 0.1851 |
| nfcorpus | 2-2 | 614 | nDCG@10 | 1922 | 0.5240 | 0.3578 | 0.1662 | 0.1594 to 0.1735 |
| nfcorpus | 2-2 | 716 | nDCG@10 | 1922 | 0.5079 | 0.3578 | 0.1501 | 0.1437 to 0.1571 |
| nfcorpus | 2-2 | 819 | nDCG@10 | 1922 | 0.4738 | 0.3578 | 0.1161 | 0.1107 to 0.1221 |
| nfcorpus | 2-2 | 921 | nDCG@10 | 1922 | 0.4215 | 0.3578 | 0.0637 | 0.0598 to 0.0676 |
| nfcorpus | 5-5 | 102 | nDCG@10 | 1517 | 0.5171 | 0.3958 | 0.1213 | 0.1125 to 0.1303 |
| nfcorpus | 5-5 | 204 | nDCG@10 | 1517 | 0.5349 | 0.3958 | 0.1391 | 0.1305 to 0.1482 |
| nfcorpus | 5-5 | 307 | nDCG@10 | 1517 | 0.5322 | 0.3958 | 0.1364 | 0.1282 to 0.1450 |
| nfcorpus | 5-5 | 409 | nDCG@10 | 1517 | 0.5258 | 0.3958 | 0.1301 | 0.1220 to 0.1382 |
| nfcorpus | 5-5 | 512 | nDCG@10 | 1517 | 0.5178 | 0.3958 | 0.1220 | 0.1144 to 0.1302 |
| nfcorpus | 5-5 | 614 | nDCG@10 | 1517 | 0.5096 | 0.3958 | 0.1138 | 0.1065 to 0.1216 |
| nfcorpus | 5-5 | 716 | nDCG@10 | 1517 | 0.5003 | 0.3958 | 0.1045 | 0.0976 to 0.1119 |
| nfcorpus | 5-5 | 819 | nDCG@10 | 1517 | 0.4805 | 0.3958 | 0.0847 | 0.0788 to 0.0911 |
| nfcorpus | 5-5 | 921 | nDCG@10 | 1517 | 0.4478 | 0.3958 | 0.0521 | 0.0479 to 0.0565 |
| nfcorpus | 10-10 | 102 | nDCG@10 | 1105 | 0.5290 | 0.4547 | 0.0743 | 0.0650 to 0.0837 |
| nfcorpus | 10-10 | 204 | nDCG@10 | 1105 | 0.5469 | 0.4547 | 0.0921 | 0.0836 to 0.1012 |
| nfcorpus | 10-10 | 307 | nDCG@10 | 1105 | 0.5469 | 0.4547 | 0.0921 | 0.0838 to 0.1011 |
| nfcorpus | 10-10 | 409 | nDCG@10 | 1105 | 0.5432 | 0.4547 | 0.0885 | 0.0806 to 0.0970 |
| nfcorpus | 10-10 | 512 | nDCG@10 | 1105 | 0.5379 | 0.4547 | 0.0831 | 0.0757 to 0.0909 |
| nfcorpus | 10-10 | 614 | nDCG@10 | 1105 | 0.5326 | 0.4547 | 0.0778 | 0.0709 to 0.0851 |
| nfcorpus | 10-10 | 716 | nDCG@10 | 1105 | 0.5271 | 0.4547 | 0.0724 | 0.0657 to 0.0793 |
| nfcorpus | 10-10 | 819 | nDCG@10 | 1105 | 0.5160 | 0.4547 | 0.0613 | 0.0554 to 0.0672 |
| nfcorpus | 10-10 | 921 | nDCG@10 | 1105 | 0.4950 | 0.4547 | 0.0402 | 0.0358 to 0.0445 |
| nfcorpus | 20-20 | 102 | nDCG@10 | 611 | 0.5777 | 0.5365 | 0.0411 | 0.0302 to 0.0522 |
| nfcorpus | 20-20 | 204 | nDCG@10 | 611 | 0.5993 | 0.5365 | 0.0627 | 0.0524 to 0.0734 |
| nfcorpus | 20-20 | 307 | nDCG@10 | 611 | 0.6000 | 0.5365 | 0.0635 | 0.0539 to 0.0736 |
| nfcorpus | 20-20 | 409 | nDCG@10 | 611 | 0.5984 | 0.5365 | 0.0619 | 0.0529 to 0.0720 |
| nfcorpus | 20-20 | 512 | nDCG@10 | 611 | 0.5948 | 0.5365 | 0.0583 | 0.0500 to 0.0675 |
| nfcorpus | 20-20 | 614 | nDCG@10 | 611 | 0.5914 | 0.5365 | 0.0548 | 0.0471 to 0.0636 |
| nfcorpus | 20-20 | 716 | nDCG@10 | 611 | 0.5878 | 0.5365 | 0.0512 | 0.0439 to 0.0596 |
| nfcorpus | 20-20 | 819 | nDCG@10 | 611 | 0.5807 | 0.5365 | 0.0442 | 0.0378 to 0.0511 |
| nfcorpus | 20-20 | 921 | nDCG@10 | 611 | 0.5663 | 0.5365 | 0.0298 | 0.0249 to 0.0349 |
| nfcorpus | full query | 1024 | MRR@10 | 2162 | 0.5710 |  |  |  |
| nfcorpus | prefix | 102 | MRR@10 | 2162 | 0.4621 | 0.5710 | -0.1089 | -0.1224 to -0.0959 |
| nfcorpus | prefix | 204 | MRR@10 | 2162 | 0.5056 | 0.5710 | -0.0654 | -0.0765 to -0.0557 |
| nfcorpus | prefix | 307 | MRR@10 | 2162 | 0.5299 | 0.5710 | -0.0411 | -0.0505 to -0.0323 |
| nfcorpus | prefix | 409 | MRR@10 | 2162 | 0.5492 | 0.5710 | -0.0218 | -0.0301 to -0.0135 |
| nfcorpus | prefix | 512 | MRR@10 | 2162 | 0.5532 | 0.5710 | -0.0178 | -0.0250 to -0.0103 |
| nfcorpus | prefix | 614 | MRR@10 | 2162 | 0.5609 | 0.5710 | -0.0101 | -0.0168 to -0.0037 |
| nfcorpus | prefix | 716 | MRR@10 | 2162 | 0.5664 | 0.5710 | -0.0046 | -0.0102 to 0.0010 |
| nfcorpus | prefix | 819 | MRR@10 | 2162 | 0.5646 | 0.5710 | -0.0064 | -0.0116 to -0.0015 |
| nfcorpus | prefix | 921 | MRR@10 | 2162 | 0.5694 | 0.5710 | -0.0016 | -0.0058 to 0.0023 |
| nfcorpus | shared | 102 | MRR@10 | 2162 | 0.7845 | 0.5710 | 0.2135 | 0.1982 to 0.2293 |
| nfcorpus | shared | 204 | MRR@10 | 2162 | 0.7990 | 0.5710 | 0.2280 | 0.2123 to 0.2431 |
| nfcorpus | shared | 307 | MRR@10 | 2162 | 0.7879 | 0.5710 | 0.2169 | 0.2022 to 0.2311 |
| nfcorpus | shared | 409 | MRR@10 | 2162 | 0.7812 | 0.5710 | 0.2102 | 0.1953 to 0.2243 |
| nfcorpus | shared | 512 | MRR@10 | 2162 | 0.7700 | 0.5710 | 0.1990 | 0.1851 to 0.2127 |
| nfcorpus | shared | 614 | MRR@10 | 2162 | 0.7638 | 0.5710 | 0.1928 | 0.1794 to 0.2062 |
| nfcorpus | shared | 716 | MRR@10 | 2162 | 0.7536 | 0.5710 | 0.1826 | 0.1692 to 0.1956 |
| nfcorpus | shared | 819 | MRR@10 | 2162 | 0.7233 | 0.5710 | 0.1523 | 0.1405 to 0.1644 |
| nfcorpus | shared | 921 | MRR@10 | 2162 | 0.6668 | 0.5710 | 0.0957 | 0.0860 to 0.1049 |
| nfcorpus | 1-1 | 102 | MRR@10 | 2162 | 0.9789 | 0.5710 | 0.4079 | 0.3907 to 0.4253 |
| nfcorpus | 1-1 | 204 | MRR@10 | 2162 | 0.9767 | 0.5710 | 0.4056 | 0.3885 to 0.4227 |
| nfcorpus | 1-1 | 307 | MRR@10 | 2162 | 0.9680 | 0.5710 | 0.3969 | 0.3797 to 0.4136 |
| nfcorpus | 1-1 | 409 | MRR@10 | 2162 | 0.9561 | 0.5710 | 0.3851 | 0.3680 to 0.4014 |
| nfcorpus | 1-1 | 512 | MRR@10 | 2162 | 0.9403 | 0.5710 | 0.3693 | 0.3528 to 0.3851 |
| nfcorpus | 1-1 | 614 | MRR@10 | 2162 | 0.9256 | 0.5710 | 0.3546 | 0.3383 to 0.3705 |
| nfcorpus | 1-1 | 716 | MRR@10 | 2162 | 0.8967 | 0.5710 | 0.3257 | 0.3097 to 0.3416 |
| nfcorpus | 1-1 | 819 | MRR@10 | 2162 | 0.8261 | 0.5710 | 0.2551 | 0.2413 to 0.2690 |
| nfcorpus | 1-1 | 921 | MRR@10 | 2162 | 0.6908 | 0.5710 | 0.1198 | 0.1110 to 0.1286 |
| nfcorpus | 2-2 | 102 | MRR@10 | 1922 | 0.8787 | 0.6001 | 0.2787 | 0.2619 to 0.2949 |
| nfcorpus | 2-2 | 204 | MRR@10 | 1922 | 0.8849 | 0.6001 | 0.2848 | 0.2686 to 0.3007 |
| nfcorpus | 2-2 | 307 | MRR@10 | 1922 | 0.8732 | 0.6001 | 0.2731 | 0.2574 to 0.2884 |
| nfcorpus | 2-2 | 409 | MRR@10 | 1922 | 0.8580 | 0.6001 | 0.2580 | 0.2428 to 0.2729 |
| nfcorpus | 2-2 | 512 | MRR@10 | 1922 | 0.8381 | 0.6001 | 0.2380 | 0.2237 to 0.2528 |
| nfcorpus | 2-2 | 614 | MRR@10 | 1922 | 0.8218 | 0.6001 | 0.2218 | 0.2078 to 0.2357 |
| nfcorpus | 2-2 | 716 | MRR@10 | 1922 | 0.7996 | 0.6001 | 0.1995 | 0.1861 to 0.2130 |
| nfcorpus | 2-2 | 819 | MRR@10 | 1922 | 0.7526 | 0.6001 | 0.1526 | 0.1409 to 0.1646 |
| nfcorpus | 2-2 | 921 | MRR@10 | 1922 | 0.6830 | 0.6001 | 0.0829 | 0.0746 to 0.0915 |
| nfcorpus | 5-5 | 102 | MRR@10 | 1517 | 0.7891 | 0.6458 | 0.1433 | 0.1281 to 0.1592 |
| nfcorpus | 5-5 | 204 | MRR@10 | 1517 | 0.8001 | 0.6458 | 0.1543 | 0.1393 to 0.1693 |
| nfcorpus | 5-5 | 307 | MRR@10 | 1517 | 0.7940 | 0.6458 | 0.1481 | 0.1338 to 0.1629 |
| nfcorpus | 5-5 | 409 | MRR@10 | 1517 | 0.7881 | 0.6458 | 0.1423 | 0.1280 to 0.1565 |
| nfcorpus | 5-5 | 512 | MRR@10 | 1517 | 0.7811 | 0.6458 | 0.1352 | 0.1215 to 0.1492 |
| nfcorpus | 5-5 | 614 | MRR@10 | 1517 | 0.7728 | 0.6458 | 0.1269 | 0.1138 to 0.1406 |
| nfcorpus | 5-5 | 716 | MRR@10 | 1517 | 0.7626 | 0.6458 | 0.1168 | 0.1040 to 0.1299 |
| nfcorpus | 5-5 | 819 | MRR@10 | 1517 | 0.7405 | 0.6458 | 0.0947 | 0.0833 to 0.1063 |
| nfcorpus | 5-5 | 921 | MRR@10 | 1517 | 0.7057 | 0.6458 | 0.0599 | 0.0508 to 0.0691 |
| nfcorpus | 10-10 | 102 | MRR@10 | 1105 | 0.7729 | 0.7005 | 0.0725 | 0.0569 to 0.0886 |
| nfcorpus | 10-10 | 204 | MRR@10 | 1105 | 0.7831 | 0.7005 | 0.0827 | 0.0679 to 0.0981 |
| nfcorpus | 10-10 | 307 | MRR@10 | 1105 | 0.7818 | 0.7005 | 0.0813 | 0.0666 to 0.0962 |
| nfcorpus | 10-10 | 409 | MRR@10 | 1105 | 0.7797 | 0.7005 | 0.0792 | 0.0654 to 0.0932 |
| nfcorpus | 10-10 | 512 | MRR@10 | 1105 | 0.7761 | 0.7005 | 0.0756 | 0.0624 to 0.0892 |
| nfcorpus | 10-10 | 614 | MRR@10 | 1105 | 0.7728 | 0.7005 | 0.0723 | 0.0597 to 0.0853 |
| nfcorpus | 10-10 | 716 | MRR@10 | 1105 | 0.7678 | 0.7005 | 0.0673 | 0.0549 to 0.0803 |
| nfcorpus | 10-10 | 819 | MRR@10 | 1105 | 0.7581 | 0.7005 | 0.0576 | 0.0460 to 0.0694 |
| nfcorpus | 10-10 | 921 | MRR@10 | 1105 | 0.7392 | 0.7005 | 0.0387 | 0.0294 to 0.0482 |
| nfcorpus | 20-20 | 102 | MRR@10 | 611 | 0.8013 | 0.7578 | 0.0436 | 0.0234 to 0.0647 |
| nfcorpus | 20-20 | 204 | MRR@10 | 611 | 0.8143 | 0.7578 | 0.0565 | 0.0383 to 0.0757 |
| nfcorpus | 20-20 | 307 | MRR@10 | 611 | 0.8150 | 0.7578 | 0.0573 | 0.0402 to 0.0750 |
| nfcorpus | 20-20 | 409 | MRR@10 | 611 | 0.8130 | 0.7578 | 0.0552 | 0.0386 to 0.0726 |
| nfcorpus | 20-20 | 512 | MRR@10 | 611 | 0.8117 | 0.7578 | 0.0539 | 0.0379 to 0.0707 |
| nfcorpus | 20-20 | 614 | MRR@10 | 611 | 0.8086 | 0.7578 | 0.0509 | 0.0359 to 0.0671 |
| nfcorpus | 20-20 | 716 | MRR@10 | 611 | 0.8076 | 0.7578 | 0.0498 | 0.0350 to 0.0657 |
| nfcorpus | 20-20 | 819 | MRR@10 | 611 | 0.8032 | 0.7578 | 0.0454 | 0.0318 to 0.0602 |
| nfcorpus | 20-20 | 921 | MRR@10 | 611 | 0.7841 | 0.7578 | 0.0263 | 0.0156 to 0.0373 |
| nfcorpus | full query | 1024 | MRR@100 | 2162 | 0.5764 |  |  |  |
| nfcorpus | prefix | 102 | MRR@100 | 2162 | 0.4698 | 0.5764 | -0.1066 | -0.1197 to -0.0937 |
| nfcorpus | prefix | 204 | MRR@100 | 2162 | 0.5127 | 0.5764 | -0.0638 | -0.0745 to -0.0541 |
| nfcorpus | prefix | 307 | MRR@100 | 2162 | 0.5361 | 0.5764 | -0.0403 | -0.0495 to -0.0316 |
| nfcorpus | prefix | 409 | MRR@100 | 2162 | 0.5552 | 0.5764 | -0.0212 | -0.0295 to -0.0131 |
| nfcorpus | prefix | 512 | MRR@100 | 2162 | 0.5588 | 0.5764 | -0.0177 | -0.0247 to -0.0104 |
| nfcorpus | prefix | 614 | MRR@100 | 2162 | 0.5669 | 0.5764 | -0.0095 | -0.0161 to -0.0033 |
| nfcorpus | prefix | 716 | MRR@100 | 2162 | 0.5724 | 0.5764 | -0.0040 | -0.0095 to 0.0017 |
| nfcorpus | prefix | 819 | MRR@100 | 2162 | 0.5705 | 0.5764 | -0.0059 | -0.0110 to -0.0011 |
| nfcorpus | prefix | 921 | MRR@100 | 2162 | 0.5752 | 0.5764 | -0.0012 | -0.0054 to 0.0028 |
| nfcorpus | shared | 102 | MRR@100 | 2162 | 0.7866 | 0.5764 | 0.2102 | 0.1950 to 0.2259 |
| nfcorpus | shared | 204 | MRR@100 | 2162 | 0.8012 | 0.5764 | 0.2247 | 0.2093 to 0.2394 |
| nfcorpus | shared | 307 | MRR@100 | 2162 | 0.7899 | 0.5764 | 0.2135 | 0.1992 to 0.2275 |
| nfcorpus | shared | 409 | MRR@100 | 2162 | 0.7833 | 0.5764 | 0.2069 | 0.1923 to 0.2210 |
| nfcorpus | shared | 512 | MRR@100 | 2162 | 0.7724 | 0.5764 | 0.1960 | 0.1824 to 0.2096 |
| nfcorpus | shared | 614 | MRR@100 | 2162 | 0.7666 | 0.5764 | 0.1902 | 0.1769 to 0.2031 |
| nfcorpus | shared | 716 | MRR@100 | 2162 | 0.7568 | 0.5764 | 0.1804 | 0.1674 to 0.1932 |
| nfcorpus | shared | 819 | MRR@100 | 2162 | 0.7276 | 0.5764 | 0.1511 | 0.1395 to 0.1630 |
| nfcorpus | shared | 921 | MRR@100 | 2162 | 0.6722 | 0.5764 | 0.0957 | 0.0862 to 0.1049 |
| nfcorpus | 1-1 | 102 | MRR@100 | 2162 | 0.9789 | 0.5764 | 0.4025 | 0.3856 to 0.4195 |
| nfcorpus | 1-1 | 204 | MRR@100 | 2162 | 0.9767 | 0.5764 | 0.4002 | 0.3832 to 0.4172 |
| nfcorpus | 1-1 | 307 | MRR@100 | 2162 | 0.9680 | 0.5764 | 0.3915 | 0.3748 to 0.4079 |
| nfcorpus | 1-1 | 409 | MRR@100 | 2162 | 0.9561 | 0.5764 | 0.3797 | 0.3630 to 0.3958 |
| nfcorpus | 1-1 | 512 | MRR@100 | 2162 | 0.9404 | 0.5764 | 0.3639 | 0.3475 to 0.3797 |
| nfcorpus | 1-1 | 614 | MRR@100 | 2162 | 0.9257 | 0.5764 | 0.3493 | 0.3334 to 0.3651 |
| nfcorpus | 1-1 | 716 | MRR@100 | 2162 | 0.8971 | 0.5764 | 0.3207 | 0.3048 to 0.3363 |
| nfcorpus | 1-1 | 819 | MRR@100 | 2162 | 0.8274 | 0.5764 | 0.2510 | 0.2373 to 0.2646 |
| nfcorpus | 1-1 | 921 | MRR@100 | 2162 | 0.6951 | 0.5764 | 0.1187 | 0.1101 to 0.1273 |
| nfcorpus | 2-2 | 102 | MRR@100 | 1922 | 0.8792 | 0.6054 | 0.2738 | 0.2575 to 0.2898 |
| nfcorpus | 2-2 | 204 | MRR@100 | 1922 | 0.8853 | 0.6054 | 0.2799 | 0.2639 to 0.2954 |
| nfcorpus | 2-2 | 307 | MRR@100 | 1922 | 0.8737 | 0.6054 | 0.2683 | 0.2527 to 0.2832 |
| nfcorpus | 2-2 | 409 | MRR@100 | 1922 | 0.8589 | 0.6054 | 0.2535 | 0.2385 to 0.2681 |
| nfcorpus | 2-2 | 512 | MRR@100 | 1922 | 0.8392 | 0.6054 | 0.2338 | 0.2197 to 0.2480 |
| nfcorpus | 2-2 | 614 | MRR@100 | 1922 | 0.8232 | 0.6054 | 0.2178 | 0.2040 to 0.2315 |
| nfcorpus | 2-2 | 716 | MRR@100 | 1922 | 0.8013 | 0.6054 | 0.1960 | 0.1827 to 0.2091 |
| nfcorpus | 2-2 | 819 | MRR@100 | 1922 | 0.7557 | 0.6054 | 0.1503 | 0.1387 to 0.1619 |
| nfcorpus | 2-2 | 921 | MRR@100 | 1922 | 0.6879 | 0.6054 | 0.0825 | 0.0743 to 0.0910 |
| nfcorpus | 5-5 | 102 | MRR@100 | 1517 | 0.7911 | 0.6507 | 0.1403 | 0.1254 to 0.1559 |
| nfcorpus | 5-5 | 204 | MRR@100 | 1517 | 0.8018 | 0.6507 | 0.1511 | 0.1365 to 0.1659 |
| nfcorpus | 5-5 | 307 | MRR@100 | 1517 | 0.7958 | 0.6507 | 0.1451 | 0.1307 to 0.1597 |
| nfcorpus | 5-5 | 409 | MRR@100 | 1517 | 0.7901 | 0.6507 | 0.1394 | 0.1255 to 0.1533 |
| nfcorpus | 5-5 | 512 | MRR@100 | 1517 | 0.7832 | 0.6507 | 0.1324 | 0.1192 to 0.1460 |
| nfcorpus | 5-5 | 614 | MRR@100 | 1517 | 0.7751 | 0.6507 | 0.1243 | 0.1114 to 0.1378 |
| nfcorpus | 5-5 | 716 | MRR@100 | 1517 | 0.7651 | 0.6507 | 0.1143 | 0.1017 to 0.1272 |
| nfcorpus | 5-5 | 819 | MRR@100 | 1517 | 0.7436 | 0.6507 | 0.0929 | 0.0818 to 0.1043 |
| nfcorpus | 5-5 | 921 | MRR@100 | 1517 | 0.7100 | 0.6507 | 0.0593 | 0.0503 to 0.0680 |
| nfcorpus | 10-10 | 102 | MRR@100 | 1105 | 0.7749 | 0.7043 | 0.0706 | 0.0551 to 0.0864 |
| nfcorpus | 10-10 | 204 | MRR@100 | 1105 | 0.7850 | 0.7043 | 0.0807 | 0.0658 to 0.0960 |
| nfcorpus | 10-10 | 307 | MRR@100 | 1105 | 0.7837 | 0.7043 | 0.0794 | 0.0650 to 0.0938 |
| nfcorpus | 10-10 | 409 | MRR@100 | 1105 | 0.7818 | 0.7043 | 0.0775 | 0.0637 to 0.0913 |
| nfcorpus | 10-10 | 512 | MRR@100 | 1105 | 0.7784 | 0.7043 | 0.0740 | 0.0609 to 0.0873 |
| nfcorpus | 10-10 | 614 | MRR@100 | 1105 | 0.7751 | 0.7043 | 0.0708 | 0.0581 to 0.0837 |
| nfcorpus | 10-10 | 716 | MRR@100 | 1105 | 0.7702 | 0.7043 | 0.0659 | 0.0534 to 0.0787 |
| nfcorpus | 10-10 | 819 | MRR@100 | 1105 | 0.7608 | 0.7043 | 0.0565 | 0.0450 to 0.0681 |
| nfcorpus | 10-10 | 921 | MRR@100 | 1105 | 0.7425 | 0.7043 | 0.0382 | 0.0290 to 0.0476 |
| nfcorpus | 20-20 | 102 | MRR@100 | 611 | 0.8025 | 0.7599 | 0.0426 | 0.0224 to 0.0631 |
| nfcorpus | 20-20 | 204 | MRR@100 | 611 | 0.8154 | 0.7599 | 0.0554 | 0.0372 to 0.0744 |
| nfcorpus | 20-20 | 307 | MRR@100 | 611 | 0.8161 | 0.7599 | 0.0561 | 0.0393 to 0.0737 |
| nfcorpus | 20-20 | 409 | MRR@100 | 611 | 0.8142 | 0.7599 | 0.0543 | 0.0377 to 0.0715 |
| nfcorpus | 20-20 | 512 | MRR@100 | 611 | 0.8129 | 0.7599 | 0.0530 | 0.0370 to 0.0694 |
| nfcorpus | 20-20 | 614 | MRR@100 | 611 | 0.8100 | 0.7599 | 0.0501 | 0.0351 to 0.0660 |
| nfcorpus | 20-20 | 716 | MRR@100 | 611 | 0.8091 | 0.7599 | 0.0491 | 0.0344 to 0.0648 |
| nfcorpus | 20-20 | 819 | MRR@100 | 611 | 0.8049 | 0.7599 | 0.0450 | 0.0314 to 0.0593 |
| nfcorpus | 20-20 | 921 | MRR@100 | 611 | 0.7861 | 0.7599 | 0.0262 | 0.0155 to 0.0371 |
| nfcorpus | full query | 1024 | Recall@100 | 2162 | 0.3104 |  |  |  |
| nfcorpus | prefix | 102 | Recall@100 | 2162 | 0.2505 | 0.3104 | -0.0599 | -0.0655 to -0.0541 |
| nfcorpus | prefix | 204 | Recall@100 | 2162 | 0.2717 | 0.3104 | -0.0387 | -0.0433 to -0.0342 |
| nfcorpus | prefix | 307 | Recall@100 | 2162 | 0.2848 | 0.3104 | -0.0255 | -0.0296 to -0.0217 |
| nfcorpus | prefix | 409 | Recall@100 | 2162 | 0.2956 | 0.3104 | -0.0148 | -0.0189 to -0.0109 |
| nfcorpus | prefix | 512 | Recall@100 | 2162 | 0.3008 | 0.3104 | -0.0096 | -0.0133 to -0.0059 |
| nfcorpus | prefix | 614 | Recall@100 | 2162 | 0.3044 | 0.3104 | -0.0059 | -0.0091 to -0.0029 |
| nfcorpus | prefix | 716 | Recall@100 | 2162 | 0.3052 | 0.3104 | -0.0052 | -0.0079 to -0.0026 |
| nfcorpus | prefix | 819 | Recall@100 | 2162 | 0.3079 | 0.3104 | -0.0025 | -0.0048 to -0.0001 |
| nfcorpus | prefix | 921 | Recall@100 | 2162 | 0.3094 | 0.3104 | -0.0010 | -0.0029 to 0.0008 |
| nfcorpus | shared | 102 | Recall@100 | 2162 | 0.5580 | 0.3104 | 0.2476 | 0.2381 to 0.2570 |
| nfcorpus | shared | 204 | Recall@100 | 2162 | 0.5687 | 0.3104 | 0.2584 | 0.2488 to 0.2681 |
| nfcorpus | shared | 307 | Recall@100 | 2162 | 0.5654 | 0.3104 | 0.2550 | 0.2450 to 0.2647 |
| nfcorpus | shared | 409 | Recall@100 | 2162 | 0.5605 | 0.3104 | 0.2501 | 0.2405 to 0.2600 |
| nfcorpus | shared | 512 | Recall@100 | 2162 | 0.5515 | 0.3104 | 0.2412 | 0.2315 to 0.2508 |
| nfcorpus | shared | 614 | Recall@100 | 2162 | 0.5426 | 0.3104 | 0.2322 | 0.2227 to 0.2418 |
| nfcorpus | shared | 716 | Recall@100 | 2162 | 0.5322 | 0.3104 | 0.2218 | 0.2121 to 0.2313 |
| nfcorpus | shared | 819 | Recall@100 | 2162 | 0.5085 | 0.3104 | 0.1982 | 0.1896 to 0.2076 |
| nfcorpus | shared | 921 | Recall@100 | 2162 | 0.4464 | 0.3104 | 0.1360 | 0.1288 to 0.1434 |
| nfcorpus | 1-1 | 102 | Recall@100 | 2162 | 0.4476 | 0.3104 | 0.1372 | 0.1297 to 0.1445 |
| nfcorpus | 1-1 | 204 | Recall@100 | 2162 | 0.4598 | 0.3104 | 0.1494 | 0.1421 to 0.1566 |
| nfcorpus | 1-1 | 307 | Recall@100 | 2162 | 0.4634 | 0.3104 | 0.1530 | 0.1457 to 0.1602 |
| nfcorpus | 1-1 | 409 | Recall@100 | 2162 | 0.4632 | 0.3104 | 0.1528 | 0.1458 to 0.1599 |
| nfcorpus | 1-1 | 512 | Recall@100 | 2162 | 0.4616 | 0.3104 | 0.1512 | 0.1444 to 0.1580 |
| nfcorpus | 1-1 | 614 | Recall@100 | 2162 | 0.4601 | 0.3104 | 0.1497 | 0.1431 to 0.1564 |
| nfcorpus | 1-1 | 716 | Recall@100 | 2162 | 0.4564 | 0.3104 | 0.1460 | 0.1394 to 0.1525 |
| nfcorpus | 1-1 | 819 | Recall@100 | 2162 | 0.4462 | 0.3104 | 0.1359 | 0.1297 to 0.1421 |
| nfcorpus | 1-1 | 921 | Recall@100 | 2162 | 0.4153 | 0.3104 | 0.1050 | 0.0998 to 0.1102 |
| nfcorpus | 2-2 | 102 | Recall@100 | 1922 | 0.4577 | 0.2917 | 0.1660 | 0.1582 to 0.1737 |
| nfcorpus | 2-2 | 204 | Recall@100 | 1922 | 0.4702 | 0.2917 | 0.1785 | 0.1711 to 0.1865 |
| nfcorpus | 2-2 | 307 | Recall@100 | 1922 | 0.4713 | 0.2917 | 0.1796 | 0.1722 to 0.1874 |
| nfcorpus | 2-2 | 409 | Recall@100 | 1922 | 0.4686 | 0.2917 | 0.1769 | 0.1694 to 0.1843 |
| nfcorpus | 2-2 | 512 | Recall@100 | 1922 | 0.4647 | 0.2917 | 0.1730 | 0.1659 to 0.1803 |
| nfcorpus | 2-2 | 614 | Recall@100 | 1922 | 0.4602 | 0.2917 | 0.1685 | 0.1613 to 0.1757 |
| nfcorpus | 2-2 | 716 | Recall@100 | 1922 | 0.4533 | 0.2917 | 0.1616 | 0.1547 to 0.1686 |
| nfcorpus | 2-2 | 819 | Recall@100 | 1922 | 0.4348 | 0.2917 | 0.1431 | 0.1366 to 0.1496 |
| nfcorpus | 2-2 | 921 | Recall@100 | 1922 | 0.3891 | 0.2917 | 0.0974 | 0.0925 to 0.1028 |
| nfcorpus | 5-5 | 102 | Recall@100 | 1517 | 0.4146 | 0.2764 | 0.1381 | 0.1308 to 0.1457 |
| nfcorpus | 5-5 | 204 | Recall@100 | 1517 | 0.4290 | 0.2764 | 0.1525 | 0.1453 to 0.1603 |
| nfcorpus | 5-5 | 307 | Recall@100 | 1517 | 0.4274 | 0.2764 | 0.1509 | 0.1439 to 0.1586 |
| nfcorpus | 5-5 | 409 | Recall@100 | 1517 | 0.4234 | 0.2764 | 0.1470 | 0.1400 to 0.1545 |
| nfcorpus | 5-5 | 512 | Recall@100 | 1517 | 0.4166 | 0.2764 | 0.1402 | 0.1335 to 0.1476 |
| nfcorpus | 5-5 | 614 | Recall@100 | 1517 | 0.4106 | 0.2764 | 0.1342 | 0.1277 to 0.1415 |
| nfcorpus | 5-5 | 716 | Recall@100 | 1517 | 0.4029 | 0.2764 | 0.1265 | 0.1203 to 0.1334 |
| nfcorpus | 5-5 | 819 | Recall@100 | 1517 | 0.3848 | 0.2764 | 0.1083 | 0.1025 to 0.1144 |
| nfcorpus | 5-5 | 921 | Recall@100 | 1517 | 0.3482 | 0.2764 | 0.0717 | 0.0674 to 0.0764 |
| nfcorpus | 10-10 | 102 | Recall@100 | 1105 | 0.3471 | 0.2561 | 0.0910 | 0.0842 to 0.0978 |
| nfcorpus | 10-10 | 204 | Recall@100 | 1105 | 0.3596 | 0.2561 | 0.1035 | 0.0965 to 0.1104 |
| nfcorpus | 10-10 | 307 | Recall@100 | 1105 | 0.3593 | 0.2561 | 0.1032 | 0.0961 to 0.1100 |
| nfcorpus | 10-10 | 409 | Recall@100 | 1105 | 0.3557 | 0.2561 | 0.0996 | 0.0928 to 0.1063 |
| nfcorpus | 10-10 | 512 | Recall@100 | 1105 | 0.3514 | 0.2561 | 0.0953 | 0.0886 to 0.1019 |
| nfcorpus | 10-10 | 614 | Recall@100 | 1105 | 0.3466 | 0.2561 | 0.0906 | 0.0842 to 0.0970 |
| nfcorpus | 10-10 | 716 | Recall@100 | 1105 | 0.3412 | 0.2561 | 0.0851 | 0.0789 to 0.0912 |
| nfcorpus | 10-10 | 819 | Recall@100 | 1105 | 0.3299 | 0.2561 | 0.0738 | 0.0683 to 0.0795 |
| nfcorpus | 10-10 | 921 | Recall@100 | 1105 | 0.3062 | 0.2561 | 0.0501 | 0.0462 to 0.0541 |
| nfcorpus | 20-20 | 102 | Recall@100 | 611 | 0.2805 | 0.2330 | 0.0475 | 0.0421 to 0.0532 |
| nfcorpus | 20-20 | 204 | Recall@100 | 611 | 0.2922 | 0.2330 | 0.0592 | 0.0537 to 0.0650 |
| nfcorpus | 20-20 | 307 | Recall@100 | 611 | 0.2932 | 0.2330 | 0.0602 | 0.0549 to 0.0656 |
| nfcorpus | 20-20 | 409 | Recall@100 | 611 | 0.2914 | 0.2330 | 0.0584 | 0.0534 to 0.0635 |
| nfcorpus | 20-20 | 512 | Recall@100 | 611 | 0.2880 | 0.2330 | 0.0550 | 0.0504 to 0.0598 |
| nfcorpus | 20-20 | 614 | Recall@100 | 611 | 0.2848 | 0.2330 | 0.0518 | 0.0473 to 0.0564 |
| nfcorpus | 20-20 | 716 | Recall@100 | 611 | 0.2817 | 0.2330 | 0.0487 | 0.0445 to 0.0530 |
| nfcorpus | 20-20 | 819 | Recall@100 | 611 | 0.2749 | 0.2330 | 0.0419 | 0.0382 to 0.0458 |
| nfcorpus | 20-20 | 921 | Recall@100 | 611 | 0.2607 | 0.2330 | 0.0277 | 0.0251 to 0.0306 |
| msmarco | full query | 1024 | nDCG@10 | 2321 | 0.2775 |  |  |  |
| msmarco | prefix | 102 | nDCG@10 | 2321 | 0.1828 | 0.2775 | -0.0947 | -0.1036 to -0.0858 |
| msmarco | prefix | 204 | nDCG@10 | 2321 | 0.2252 | 0.2775 | -0.0523 | -0.0597 to -0.0448 |
| msmarco | prefix | 307 | nDCG@10 | 2321 | 0.2416 | 0.2775 | -0.0359 | -0.0418 to -0.0299 |
| msmarco | prefix | 409 | nDCG@10 | 2321 | 0.2529 | 0.2775 | -0.0246 | -0.0299 to -0.0191 |
| msmarco | prefix | 512 | nDCG@10 | 2321 | 0.2607 | 0.2775 | -0.0168 | -0.0218 to -0.0119 |
| msmarco | prefix | 614 | nDCG@10 | 2321 | 0.2664 | 0.2775 | -0.0111 | -0.0153 to -0.0068 |
| msmarco | prefix | 716 | nDCG@10 | 2321 | 0.2721 | 0.2775 | -0.0054 | -0.0090 to -0.0017 |
| msmarco | prefix | 819 | nDCG@10 | 2321 | 0.2753 | 0.2775 | -0.0022 | -0.0053 to 0.0009 |
| msmarco | prefix | 921 | nDCG@10 | 2321 | 0.2762 | 0.2775 | -0.0013 | -0.0037 to 0.0012 |
| msmarco | shared | 102 | nDCG@10 | 2321 | 0.6042 | 0.2775 | 0.3267 | 0.3152 to 0.3372 |
| msmarco | shared | 204 | nDCG@10 | 2321 | 0.5956 | 0.2775 | 0.3181 | 0.3077 to 0.3284 |
| msmarco | shared | 307 | nDCG@10 | 2321 | 0.5719 | 0.2775 | 0.2944 | 0.2842 to 0.3037 |
| msmarco | shared | 409 | nDCG@10 | 2321 | 0.5451 | 0.2775 | 0.2676 | 0.2578 to 0.2766 |
| msmarco | shared | 512 | nDCG@10 | 2321 | 0.5182 | 0.2775 | 0.2407 | 0.2315 to 0.2493 |
| msmarco | shared | 614 | nDCG@10 | 2321 | 0.4924 | 0.2775 | 0.2149 | 0.2060 to 0.2228 |
| msmarco | shared | 716 | nDCG@10 | 2321 | 0.4672 | 0.2775 | 0.1897 | 0.1815 to 0.1973 |
| msmarco | shared | 819 | nDCG@10 | 2321 | 0.4452 | 0.2775 | 0.1677 | 0.1597 to 0.1747 |
| msmarco | shared | 921 | nDCG@10 | 2321 | 0.3989 | 0.2775 | 0.1214 | 0.1150 to 0.1274 |
| msmarco | 1-1 | 102 | nDCG@10 | 2321 | 0.6321 | 0.2775 | 0.3546 | 0.3456 to 0.3634 |
| msmarco | 1-1 | 204 | nDCG@10 | 2321 | 0.6295 | 0.2775 | 0.3520 | 0.3433 to 0.3604 |
| msmarco | 1-1 | 307 | nDCG@10 | 2321 | 0.6146 | 0.2775 | 0.3371 | 0.3287 to 0.3454 |
| msmarco | 1-1 | 409 | nDCG@10 | 2321 | 0.5950 | 0.2775 | 0.3175 | 0.3090 to 0.3255 |
| msmarco | 1-1 | 512 | nDCG@10 | 2321 | 0.5685 | 0.2775 | 0.2910 | 0.2829 to 0.2989 |
| msmarco | 1-1 | 614 | nDCG@10 | 2321 | 0.5418 | 0.2775 | 0.2643 | 0.2563 to 0.2721 |
| msmarco | 1-1 | 716 | nDCG@10 | 2321 | 0.5168 | 0.2775 | 0.2393 | 0.2317 to 0.2468 |
| msmarco | 1-1 | 819 | nDCG@10 | 2321 | 0.4830 | 0.2775 | 0.2055 | 0.1983 to 0.2126 |
| msmarco | 1-1 | 921 | nDCG@10 | 2321 | 0.4160 | 0.2775 | 0.1385 | 0.1328 to 0.1442 |
| msmarco | 2-2 | 102 | nDCG@10 | 81 | 0.4033 | 0.1904 | 0.2129 | 0.1808 to 0.2415 |
| msmarco | 2-2 | 204 | nDCG@10 | 81 | 0.3990 | 0.1904 | 0.2086 | 0.1781 to 0.2356 |
| msmarco | 2-2 | 307 | nDCG@10 | 81 | 0.3813 | 0.1904 | 0.1909 | 0.1627 to 0.2165 |
| msmarco | 2-2 | 409 | nDCG@10 | 81 | 0.3609 | 0.1904 | 0.1706 | 0.1440 to 0.1961 |
| msmarco | 2-2 | 512 | nDCG@10 | 81 | 0.3417 | 0.1904 | 0.1513 | 0.1266 to 0.1745 |
| msmarco | 2-2 | 614 | nDCG@10 | 81 | 0.3190 | 0.1904 | 0.1286 | 0.1055 to 0.1514 |
| msmarco | 2-2 | 716 | nDCG@10 | 81 | 0.3045 | 0.1904 | 0.1141 | 0.0921 to 0.1368 |
| msmarco | 2-2 | 819 | nDCG@10 | 81 | 0.2912 | 0.1904 | 0.1008 | 0.0806 to 0.1209 |
| msmarco | 2-2 | 921 | nDCG@10 | 81 | 0.2637 | 0.1904 | 0.0734 | 0.0577 to 0.0890 |
| msmarco | full query | 1024 | MRR@10 | 2321 | 0.2967 |  |  |  |
| msmarco | prefix | 102 | MRR@10 | 2321 | 0.2039 | 0.2967 | -0.0928 | -0.1055 to -0.0806 |
| msmarco | prefix | 204 | MRR@10 | 2321 | 0.2465 | 0.2967 | -0.0502 | -0.0608 to -0.0401 |
| msmarco | prefix | 307 | MRR@10 | 2321 | 0.2630 | 0.2967 | -0.0337 | -0.0426 to -0.0247 |
| msmarco | prefix | 409 | MRR@10 | 2321 | 0.2722 | 0.2967 | -0.0245 | -0.0328 to -0.0153 |
| msmarco | prefix | 512 | MRR@10 | 2321 | 0.2815 | 0.2967 | -0.0152 | -0.0234 to -0.0072 |
| msmarco | prefix | 614 | MRR@10 | 2321 | 0.2871 | 0.2967 | -0.0097 | -0.0166 to -0.0028 |
| msmarco | prefix | 716 | MRR@10 | 2321 | 0.2922 | 0.2967 | -0.0046 | -0.0104 to 0.0015 |
| msmarco | prefix | 819 | MRR@10 | 2321 | 0.2955 | 0.2967 | -0.0012 | -0.0060 to 0.0036 |
| msmarco | prefix | 921 | MRR@10 | 2321 | 0.2942 | 0.2967 | -0.0025 | -0.0066 to 0.0014 |
| msmarco | shared | 102 | MRR@10 | 2321 | 0.6533 | 0.2967 | 0.3566 | 0.3418 to 0.3705 |
| msmarco | shared | 204 | MRR@10 | 2321 | 0.6442 | 0.2967 | 0.3474 | 0.3335 to 0.3615 |
| msmarco | shared | 307 | MRR@10 | 2321 | 0.6182 | 0.2967 | 0.3215 | 0.3082 to 0.3341 |
| msmarco | shared | 409 | MRR@10 | 2321 | 0.5898 | 0.2967 | 0.2931 | 0.2798 to 0.3057 |
| msmarco | shared | 512 | MRR@10 | 2321 | 0.5599 | 0.2967 | 0.2632 | 0.2512 to 0.2751 |
| msmarco | shared | 614 | MRR@10 | 2321 | 0.5336 | 0.2967 | 0.2369 | 0.2251 to 0.2482 |
| msmarco | shared | 716 | MRR@10 | 2321 | 0.5062 | 0.2967 | 0.2095 | 0.1983 to 0.2203 |
| msmarco | shared | 819 | MRR@10 | 2321 | 0.4809 | 0.2967 | 0.1842 | 0.1737 to 0.1943 |
| msmarco | shared | 921 | MRR@10 | 2321 | 0.4307 | 0.2967 | 0.1340 | 0.1252 to 0.1429 |
| msmarco | 1-1 | 102 | MRR@10 | 2321 | 0.8576 | 0.2967 | 0.5609 | 0.5463 to 0.5757 |
| msmarco | 1-1 | 204 | MRR@10 | 2321 | 0.8446 | 0.2967 | 0.5478 | 0.5331 to 0.5629 |
| msmarco | 1-1 | 307 | MRR@10 | 2321 | 0.8113 | 0.2967 | 0.5146 | 0.5005 to 0.5290 |
| msmarco | 1-1 | 409 | MRR@10 | 2321 | 0.7702 | 0.2967 | 0.4735 | 0.4595 to 0.4876 |
| msmarco | 1-1 | 512 | MRR@10 | 2321 | 0.7185 | 0.2967 | 0.4218 | 0.4083 to 0.4352 |
| msmarco | 1-1 | 614 | MRR@10 | 2321 | 0.6711 | 0.2967 | 0.3744 | 0.3611 to 0.3877 |
| msmarco | 1-1 | 716 | MRR@10 | 2321 | 0.6272 | 0.2967 | 0.3304 | 0.3176 to 0.3431 |
| msmarco | 1-1 | 819 | MRR@10 | 2321 | 0.5718 | 0.2967 | 0.2750 | 0.2626 to 0.2863 |
| msmarco | 1-1 | 921 | MRR@10 | 2321 | 0.4707 | 0.2967 | 0.1739 | 0.1647 to 0.1828 |
| msmarco | 2-2 | 102 | MRR@10 | 81 | 0.6025 | 0.2740 | 0.3285 | 0.2604 to 0.3917 |
| msmarco | 2-2 | 204 | MRR@10 | 81 | 0.5850 | 0.2740 | 0.3111 | 0.2457 to 0.3731 |
| msmarco | 2-2 | 307 | MRR@10 | 81 | 0.5503 | 0.2740 | 0.2763 | 0.2156 to 0.3345 |
| msmarco | 2-2 | 409 | MRR@10 | 81 | 0.5212 | 0.2740 | 0.2472 | 0.1909 to 0.3001 |
| msmarco | 2-2 | 512 | MRR@10 | 81 | 0.4956 | 0.2740 | 0.2216 | 0.1690 to 0.2745 |
| msmarco | 2-2 | 614 | MRR@10 | 81 | 0.4647 | 0.2740 | 0.1907 | 0.1417 to 0.2398 |
| msmarco | 2-2 | 716 | MRR@10 | 81 | 0.4472 | 0.2740 | 0.1732 | 0.1263 to 0.2206 |
| msmarco | 2-2 | 819 | MRR@10 | 81 | 0.4279 | 0.2740 | 0.1539 | 0.1097 to 0.1974 |
| msmarco | 2-2 | 921 | MRR@10 | 81 | 0.3819 | 0.2740 | 0.1080 | 0.0745 to 0.1429 |
| msmarco | full query | 1024 | MRR@100 | 2321 | 0.3095 |  |  |  |
| msmarco | prefix | 102 | MRR@100 | 2321 | 0.2175 | 0.3095 | -0.0920 | -0.1041 to -0.0803 |
| msmarco | prefix | 204 | MRR@100 | 2321 | 0.2600 | 0.3095 | -0.0495 | -0.0597 to -0.0398 |
| msmarco | prefix | 307 | MRR@100 | 2321 | 0.2770 | 0.3095 | -0.0325 | -0.0414 to -0.0235 |
| msmarco | prefix | 409 | MRR@100 | 2321 | 0.2858 | 0.3095 | -0.0237 | -0.0319 to -0.0149 |
| msmarco | prefix | 512 | MRR@100 | 2321 | 0.2949 | 0.3095 | -0.0146 | -0.0227 to -0.0066 |
| msmarco | prefix | 614 | MRR@100 | 2321 | 0.3002 | 0.3095 | -0.0092 | -0.0160 to -0.0026 |
| msmarco | prefix | 716 | MRR@100 | 2321 | 0.3050 | 0.3095 | -0.0045 | -0.0104 to 0.0014 |
| msmarco | prefix | 819 | MRR@100 | 2321 | 0.3082 | 0.3095 | -0.0013 | -0.0061 to 0.0034 |
| msmarco | prefix | 921 | MRR@100 | 2321 | 0.3074 | 0.3095 | -0.0021 | -0.0061 to 0.0017 |
| msmarco | shared | 102 | MRR@100 | 2321 | 0.6582 | 0.3095 | 0.3487 | 0.3342 to 0.3622 |
| msmarco | shared | 204 | MRR@100 | 2321 | 0.6491 | 0.3095 | 0.3396 | 0.3259 to 0.3529 |
| msmarco | shared | 307 | MRR@100 | 2321 | 0.6238 | 0.3095 | 0.3143 | 0.3011 to 0.3266 |
| msmarco | shared | 409 | MRR@100 | 2321 | 0.5960 | 0.3095 | 0.2866 | 0.2736 to 0.2987 |
| msmarco | shared | 512 | MRR@100 | 2321 | 0.5668 | 0.3095 | 0.2574 | 0.2456 to 0.2689 |
| msmarco | shared | 614 | MRR@100 | 2321 | 0.5410 | 0.3095 | 0.2315 | 0.2204 to 0.2425 |
| msmarco | shared | 716 | MRR@100 | 2321 | 0.5147 | 0.3095 | 0.2052 | 0.1944 to 0.2156 |
| msmarco | shared | 819 | MRR@100 | 2321 | 0.4905 | 0.3095 | 0.1810 | 0.1712 to 0.1909 |
| msmarco | shared | 921 | MRR@100 | 2321 | 0.4416 | 0.3095 | 0.1322 | 0.1236 to 0.1408 |
| msmarco | 1-1 | 102 | MRR@100 | 2321 | 0.8583 | 0.3095 | 0.5488 | 0.5344 to 0.5635 |
| msmarco | 1-1 | 204 | MRR@100 | 2321 | 0.8456 | 0.3095 | 0.5361 | 0.5220 to 0.5506 |
| msmarco | 1-1 | 307 | MRR@100 | 2321 | 0.8127 | 0.3095 | 0.5033 | 0.4896 to 0.5173 |
| msmarco | 1-1 | 409 | MRR@100 | 2321 | 0.7722 | 0.3095 | 0.4627 | 0.4491 to 0.4766 |
| msmarco | 1-1 | 512 | MRR@100 | 2321 | 0.7215 | 0.3095 | 0.4121 | 0.3992 to 0.4251 |
| msmarco | 1-1 | 614 | MRR@100 | 2321 | 0.6752 | 0.3095 | 0.3658 | 0.3529 to 0.3787 |
| msmarco | 1-1 | 716 | MRR@100 | 2321 | 0.6324 | 0.3095 | 0.3230 | 0.3103 to 0.3351 |
| msmarco | 1-1 | 819 | MRR@100 | 2321 | 0.5786 | 0.3095 | 0.2691 | 0.2572 to 0.2801 |
| msmarco | 1-1 | 921 | MRR@100 | 2321 | 0.4800 | 0.3095 | 0.1705 | 0.1614 to 0.1789 |
| msmarco | 2-2 | 102 | MRR@100 | 81 | 0.6098 | 0.2894 | 0.3204 | 0.2546 to 0.3816 |
| msmarco | 2-2 | 204 | MRR@100 | 81 | 0.5915 | 0.2894 | 0.3020 | 0.2376 to 0.3617 |
| msmarco | 2-2 | 307 | MRR@100 | 81 | 0.5575 | 0.2894 | 0.2680 | 0.2093 to 0.3241 |
| msmarco | 2-2 | 409 | MRR@100 | 81 | 0.5288 | 0.2894 | 0.2393 | 0.1841 to 0.2906 |
| msmarco | 2-2 | 512 | MRR@100 | 81 | 0.5040 | 0.2894 | 0.2145 | 0.1628 to 0.2657 |
| msmarco | 2-2 | 614 | MRR@100 | 81 | 0.4753 | 0.2894 | 0.1858 | 0.1375 to 0.2342 |
| msmarco | 2-2 | 716 | MRR@100 | 81 | 0.4590 | 0.2894 | 0.1696 | 0.1238 to 0.2160 |
| msmarco | 2-2 | 819 | MRR@100 | 81 | 0.4394 | 0.2894 | 0.1500 | 0.1069 to 0.1938 |
| msmarco | 2-2 | 921 | MRR@100 | 81 | 0.3940 | 0.2894 | 0.1046 | 0.0709 to 0.1391 |
| msmarco | full query | 1024 | Recall@100 | 2321 | 0.7462 |  |  |  |
| msmarco | prefix | 102 | Recall@100 | 2321 | 0.5831 | 0.7462 | -0.1631 | -0.1754 to -0.1507 |
| msmarco | prefix | 204 | Recall@100 | 2321 | 0.6666 | 0.7462 | -0.0797 | -0.0892 to -0.0697 |
| msmarco | prefix | 307 | Recall@100 | 2321 | 0.6943 | 0.7462 | -0.0520 | -0.0610 to -0.0427 |
| msmarco | prefix | 409 | Recall@100 | 2321 | 0.7092 | 0.7462 | -0.0371 | -0.0451 to -0.0292 |
| msmarco | prefix | 512 | Recall@100 | 2321 | 0.7265 | 0.7462 | -0.0198 | -0.0267 to -0.0127 |
| msmarco | prefix | 614 | Recall@100 | 2321 | 0.7328 | 0.7462 | -0.0134 | -0.0201 to -0.0068 |
| msmarco | prefix | 716 | Recall@100 | 2321 | 0.7377 | 0.7462 | -0.0086 | -0.0141 to -0.0028 |
| msmarco | prefix | 819 | Recall@100 | 2321 | 0.7422 | 0.7462 | -0.0040 | -0.0086 to 0.0003 |
| msmarco | prefix | 921 | Recall@100 | 2321 | 0.7440 | 0.7462 | -0.0023 | -0.0059 to 0.0015 |
| msmarco | shared | 102 | Recall@100 | 2321 | 0.9560 | 0.7462 | 0.2097 | 0.1971 to 0.2220 |
| msmarco | shared | 204 | Recall@100 | 2321 | 0.9547 | 0.7462 | 0.2084 | 0.1957 to 0.2211 |
| msmarco | shared | 307 | Recall@100 | 2321 | 0.9502 | 0.7462 | 0.2040 | 0.1914 to 0.2161 |
| msmarco | shared | 409 | Recall@100 | 2321 | 0.9423 | 0.7462 | 0.1961 | 0.1833 to 0.2082 |
| msmarco | shared | 512 | Recall@100 | 2321 | 0.9330 | 0.7462 | 0.1868 | 0.1743 to 0.1987 |
| msmarco | shared | 614 | Recall@100 | 2321 | 0.9239 | 0.7462 | 0.1776 | 0.1657 to 0.1893 |
| msmarco | shared | 716 | Recall@100 | 2321 | 0.9140 | 0.7462 | 0.1678 | 0.1559 to 0.1794 |
| msmarco | shared | 819 | Recall@100 | 2321 | 0.9032 | 0.7462 | 0.1570 | 0.1456 to 0.1683 |
| msmarco | shared | 921 | Recall@100 | 2321 | 0.8694 | 0.7462 | 0.1232 | 0.1134 to 0.1331 |
| msmarco | 1-1 | 102 | Recall@100 | 2321 | 0.8552 | 0.7462 | 0.1089 | 0.0982 to 0.1188 |
| msmarco | 1-1 | 204 | Recall@100 | 2321 | 0.8665 | 0.7462 | 0.1202 | 0.1098 to 0.1296 |
| msmarco | 1-1 | 307 | Recall@100 | 2321 | 0.8695 | 0.7462 | 0.1233 | 0.1132 to 0.1327 |
| msmarco | 1-1 | 409 | Recall@100 | 2321 | 0.8721 | 0.7462 | 0.1258 | 0.1162 to 0.1350 |
| msmarco | 1-1 | 512 | Recall@100 | 2321 | 0.8718 | 0.7462 | 0.1256 | 0.1162 to 0.1343 |
| msmarco | 1-1 | 614 | Recall@100 | 2321 | 0.8735 | 0.7462 | 0.1273 | 0.1180 to 0.1359 |
| msmarco | 1-1 | 716 | Recall@100 | 2321 | 0.8725 | 0.7462 | 0.1262 | 0.1172 to 0.1344 |
| msmarco | 1-1 | 819 | Recall@100 | 2321 | 0.8690 | 0.7462 | 0.1228 | 0.1141 to 0.1310 |
| msmarco | 1-1 | 921 | Recall@100 | 2321 | 0.8535 | 0.7462 | 0.1072 | 0.0994 to 0.1146 |
| msmarco | 2-2 | 102 | Recall@100 | 81 | 0.7822 | 0.5930 | 0.1892 | 0.1406 to 0.2416 |
| msmarco | 2-2 | 204 | Recall@100 | 81 | 0.7916 | 0.5930 | 0.1986 | 0.1518 to 0.2499 |
| msmarco | 2-2 | 307 | Recall@100 | 81 | 0.7856 | 0.5930 | 0.1925 | 0.1470 to 0.2434 |
| msmarco | 2-2 | 409 | Recall@100 | 81 | 0.7818 | 0.5930 | 0.1887 | 0.1445 to 0.2356 |
| msmarco | 2-2 | 512 | Recall@100 | 81 | 0.7768 | 0.5930 | 0.1838 | 0.1389 to 0.2311 |
| msmarco | 2-2 | 614 | Recall@100 | 81 | 0.7657 | 0.5930 | 0.1727 | 0.1295 to 0.2184 |
| msmarco | 2-2 | 716 | Recall@100 | 81 | 0.7556 | 0.5930 | 0.1626 | 0.1205 to 0.2071 |
| msmarco | 2-2 | 819 | Recall@100 | 81 | 0.7423 | 0.5930 | 0.1493 | 0.1074 to 0.1929 |
| msmarco | 2-2 | 921 | Recall@100 | 81 | 0.7177 | 0.5930 | 0.1246 | 0.0868 to 0.1640 |

## Where relevant documents went

Primary ranks, query-balanced. The held-out nDCG is a restricted-label diagnostic, and the secondary rank skips construction positives only to separate crowding from displacement.

| environment | arm | k | positives evaluated | measure | queries | arm value | full query, same documents | difference | 95% interval (pointwise) |
|---|---|---|---|---|---|---|---|---|---|
| scifact | prefix | 102 | all known positives | reciprocal rank | 69 | 0.3427 | 0.4383 | -0.0955 | -0.1384 to -0.0545 |
| scifact | prefix | 102 | all known positives | in top 10 | 69 | 0.6693 | 0.8287 | -0.1594 | -0.2295 to -0.0966 |
| scifact | prefix | 102 | all known positives | in top 100 | 69 | 0.9251 | 0.9807 | -0.0556 | -0.1111 to -0.0121 |
| scifact | prefix | 102 | all known positives | log rank (lower is better) | 69 | 1.8903 | 1.3133 | 0.5769 | 0.4069 to 0.7661 |
| scifact | prefix | 102 | all known positives | worst rank in the set | 69 | 72.7826 | 19.0725 | 53.7101 | 16.6949 to 106.1891 |
| scifact | prefix | 204 | all known positives | reciprocal rank | 69 | 0.3967 | 0.4383 | -0.0416 | -0.0770 to -0.0112 |
| scifact | prefix | 204 | all known positives | in top 10 | 69 | 0.7490 | 0.8287 | -0.0797 | -0.1280 to -0.0338 |
| scifact | prefix | 204 | all known positives | in top 100 | 69 | 0.9565 | 0.9807 | -0.0242 | -0.0652 to 0.0072 |
| scifact | prefix | 204 | all known positives | log rank (lower is better) | 69 | 1.6277 | 1.3133 | 0.3144 | 0.1887 to 0.4523 |
| scifact | prefix | 204 | all known positives | worst rank in the set | 69 | 45.7681 | 19.0725 | 26.6957 | 4.8225 to 56.2960 |
| scifact | prefix | 307 | all known positives | reciprocal rank | 69 | 0.4109 | 0.4383 | -0.0274 | -0.0509 to -0.0051 |
| scifact | prefix | 307 | all known positives | in top 10 | 69 | 0.7640 | 0.8287 | -0.0647 | -0.1063 to -0.0261 |
| scifact | prefix | 307 | all known positives | in top 100 | 69 | 0.9638 | 0.9807 | -0.0169 | -0.0531 to 0.0097 |
| scifact | prefix | 307 | all known positives | log rank (lower is better) | 69 | 1.5210 | 1.3133 | 0.2077 | 0.1099 to 0.3212 |
| scifact | prefix | 307 | all known positives | worst rank in the set | 69 | 33.1594 | 19.0725 | 14.0870 | 2.7678 to 29.5518 |
| scifact | prefix | 409 | all known positives | reciprocal rank | 69 | 0.4232 | 0.4383 | -0.0151 | -0.0386 to 0.0072 |
| scifact | prefix | 409 | all known positives | in top 10 | 69 | 0.7664 | 0.8287 | -0.0623 | -0.1135 to -0.0164 |
| scifact | prefix | 409 | all known positives | in top 100 | 69 | 0.9855 | 0.9807 | 0.0048 | 0.0000 to 0.0145 |
| scifact | prefix | 409 | all known positives | log rank (lower is better) | 69 | 1.4327 | 1.3133 | 0.1194 | 0.0482 to 0.1999 |
| scifact | prefix | 409 | all known positives | worst rank in the set | 69 | 23.2609 | 19.0725 | 4.1884 | -1.2757 to 11.3482 |
| scifact | prefix | 512 | all known positives | reciprocal rank | 69 | 0.4240 | 0.4383 | -0.0142 | -0.0423 to 0.0114 |
| scifact | prefix | 512 | all known positives | in top 10 | 69 | 0.7563 | 0.8287 | -0.0725 | -0.1159 to -0.0314 |
| scifact | prefix | 512 | all known positives | in top 100 | 69 | 0.9855 | 0.9807 | 0.0048 | 0.0000 to 0.0145 |
| scifact | prefix | 512 | all known positives | log rank (lower is better) | 69 | 1.4156 | 1.3133 | 0.1023 | 0.0304 to 0.1804 |
| scifact | prefix | 512 | all known positives | worst rank in the set | 69 | 19.5072 | 19.0725 | 0.4348 | -3.1594 to 3.6957 |
| scifact | prefix | 614 | all known positives | reciprocal rank | 69 | 0.4246 | 0.4383 | -0.0136 | -0.0377 to 0.0104 |
| scifact | prefix | 614 | all known positives | in top 10 | 69 | 0.8058 | 0.8287 | -0.0229 | -0.0556 to 0.0048 |
| scifact | prefix | 614 | all known positives | in top 100 | 69 | 0.9807 | 0.9807 | 0.0000 | 0.0000 to 0.0000 |
| scifact | prefix | 614 | all known positives | log rank (lower is better) | 69 | 1.3758 | 1.3133 | 0.0625 | 0.0083 to 0.1194 |
| scifact | prefix | 614 | all known positives | worst rank in the set | 69 | 20.7681 | 19.0725 | 1.6957 | -1.3482 to 5.2612 |
| scifact | prefix | 716 | all known positives | reciprocal rank | 69 | 0.4360 | 0.4383 | -0.0023 | -0.0219 to 0.0181 |
| scifact | prefix | 716 | all known positives | in top 10 | 69 | 0.8155 | 0.8287 | -0.0133 | -0.0386 to 0.0072 |
| scifact | prefix | 716 | all known positives | in top 100 | 69 | 0.9855 | 0.9807 | 0.0048 | 0.0000 to 0.0145 |
| scifact | prefix | 716 | all known positives | log rank (lower is better) | 69 | 1.3533 | 1.3133 | 0.0400 | -0.0052 to 0.0868 |
| scifact | prefix | 716 | all known positives | worst rank in the set | 69 | 21.0290 | 19.0725 | 1.9565 | -1.1457 to 6.5801 |
| scifact | prefix | 819 | all known positives | reciprocal rank | 69 | 0.4357 | 0.4383 | -0.0025 | -0.0213 to 0.0159 |
| scifact | prefix | 819 | all known positives | in top 10 | 69 | 0.8022 | 0.8287 | -0.0266 | -0.0725 to 0.0072 |
| scifact | prefix | 819 | all known positives | in top 100 | 69 | 0.9807 | 0.9807 | 0.0000 | 0.0000 to 0.0000 |
| scifact | prefix | 819 | all known positives | log rank (lower is better) | 69 | 1.3372 | 1.3133 | 0.0239 | -0.0207 to 0.0715 |
| scifact | prefix | 819 | all known positives | worst rank in the set | 69 | 21.8841 | 19.0725 | 2.8116 | -0.8264 to 9.1739 |
| scifact | prefix | 921 | all known positives | reciprocal rank | 69 | 0.4321 | 0.4383 | -0.0061 | -0.0169 to 0.0020 |
| scifact | prefix | 921 | all known positives | in top 10 | 69 | 0.8203 | 0.8287 | -0.0085 | -0.0362 to 0.0145 |
| scifact | prefix | 921 | all known positives | in top 100 | 69 | 0.9807 | 0.9807 | 0.0000 | 0.0000 to 0.0000 |
| scifact | prefix | 921 | all known positives | log rank (lower is better) | 69 | 1.3264 | 1.3133 | 0.0131 | -0.0139 to 0.0436 |
| scifact | prefix | 921 | all known positives | worst rank in the set | 69 | 20.7246 | 19.0725 | 1.6522 | -1.2612 to 6.6377 |
| scifact | shared | 102 | all known positives | reciprocal rank | 69 | 0.6508 | 0.4383 | 0.2125 | 0.1602 to 0.2668 |
| scifact | shared | 102 | all known positives | in top 10 | 69 | 0.9952 | 0.8287 | 0.1664 | 0.0990 to 0.2408 |
| scifact | shared | 102 | all known positives | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 102 | all known positives | log rank (lower is better) | 69 | 0.5754 | 1.3133 | -0.7379 | -0.9596 to -0.5394 |
| scifact | shared | 102 | all known positives | worst rank in the set | 69 | 3.2319 | 19.0725 | -15.8406 | -29.2630 to -6.5942 |
| scifact | shared | 204 | all known positives | reciprocal rank | 69 | 0.6491 | 0.4383 | 0.2108 | 0.1593 to 0.2652 |
| scifact | shared | 204 | all known positives | in top 10 | 69 | 0.9928 | 0.8287 | 0.1640 | 0.1024 to 0.2362 |
| scifact | shared | 204 | all known positives | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 204 | all known positives | log rank (lower is better) | 69 | 0.5892 | 1.3133 | -0.7241 | -0.9413 to -0.5328 |
| scifact | shared | 204 | all known positives | worst rank in the set | 69 | 3.5362 | 19.0725 | -15.5362 | -28.5663 to -6.4783 |
| scifact | shared | 307 | all known positives | reciprocal rank | 69 | 0.6367 | 0.4383 | 0.1984 | 0.1482 to 0.2523 |
| scifact | shared | 307 | all known positives | in top 10 | 69 | 0.9855 | 0.8287 | 0.1568 | 0.0942 to 0.2275 |
| scifact | shared | 307 | all known positives | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 307 | all known positives | log rank (lower is better) | 69 | 0.6261 | 1.3133 | -0.6872 | -0.8981 to -0.5032 |
| scifact | shared | 307 | all known positives | worst rank in the set | 69 | 3.7246 | 19.0725 | -15.3478 | -28.3377 to -6.3181 |
| scifact | shared | 409 | all known positives | reciprocal rank | 69 | 0.6302 | 0.4383 | 0.1920 | 0.1437 to 0.2447 |
| scifact | shared | 409 | all known positives | in top 10 | 69 | 0.9734 | 0.8287 | 0.1447 | 0.0845 to 0.2126 |
| scifact | shared | 409 | all known positives | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 409 | all known positives | log rank (lower is better) | 69 | 0.6477 | 1.3133 | -0.6656 | -0.8706 to -0.4858 |
| scifact | shared | 409 | all known positives | worst rank in the set | 69 | 3.9420 | 19.0725 | -15.1304 | -28.1475 to -6.1297 |
| scifact | shared | 512 | all known positives | reciprocal rank | 69 | 0.6245 | 0.4383 | 0.1863 | 0.1377 to 0.2396 |
| scifact | shared | 512 | all known positives | in top 10 | 69 | 0.9662 | 0.8287 | 0.1374 | 0.0785 to 0.2027 |
| scifact | shared | 512 | all known positives | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 512 | all known positives | log rank (lower is better) | 69 | 0.6660 | 1.3133 | -0.6473 | -0.8531 to -0.4683 |
| scifact | shared | 512 | all known positives | worst rank in the set | 69 | 4.1159 | 19.0725 | -14.9565 | -27.9007 to -6.0286 |
| scifact | shared | 614 | all known positives | reciprocal rank | 69 | 0.6145 | 0.4383 | 0.1763 | 0.1284 to 0.2260 |
| scifact | shared | 614 | all known positives | in top 10 | 69 | 0.9517 | 0.8287 | 0.1229 | 0.0676 to 0.1879 |
| scifact | shared | 614 | all known positives | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 614 | all known positives | log rank (lower is better) | 69 | 0.6926 | 1.3133 | -0.6207 | -0.8192 to -0.4509 |
| scifact | shared | 614 | all known positives | worst rank in the set | 69 | 4.4058 | 19.0725 | -14.6667 | -27.5370 to -5.8978 |
| scifact | shared | 716 | all known positives | reciprocal rank | 69 | 0.6034 | 0.4383 | 0.1652 | 0.1196 to 0.2134 |
| scifact | shared | 716 | all known positives | in top 10 | 69 | 0.9444 | 0.8287 | 0.1157 | 0.0640 to 0.1761 |
| scifact | shared | 716 | all known positives | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 716 | all known positives | log rank (lower is better) | 69 | 0.7298 | 1.3133 | -0.5835 | -0.7739 to -0.4191 |
| scifact | shared | 716 | all known positives | worst rank in the set | 69 | 4.8696 | 19.0725 | -14.2029 | -26.6678 to -5.6518 |
| scifact | shared | 819 | all known positives | reciprocal rank | 69 | 0.5895 | 0.4383 | 0.1512 | 0.1074 to 0.1973 |
| scifact | shared | 819 | all known positives | in top 10 | 69 | 0.9444 | 0.8287 | 0.1157 | 0.0640 to 0.1761 |
| scifact | shared | 819 | all known positives | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | shared | 819 | all known positives | log rank (lower is better) | 69 | 0.7730 | 1.3133 | -0.5403 | -0.7131 to -0.3885 |
| scifact | shared | 819 | all known positives | worst rank in the set | 69 | 5.4638 | 19.0725 | -13.6087 | -25.6087 to -5.4051 |
| scifact | shared | 921 | all known positives | reciprocal rank | 69 | 0.5463 | 0.4383 | 0.1080 | 0.0755 to 0.1431 |
| scifact | shared | 921 | all known positives | in top 10 | 69 | 0.9396 | 0.8287 | 0.1109 | 0.0614 to 0.1698 |
| scifact | shared | 921 | all known positives | in top 100 | 69 | 0.9928 | 0.9807 | 0.0121 | 0.0000 to 0.0314 |
| scifact | shared | 921 | all known positives | log rank (lower is better) | 69 | 0.8974 | 1.3133 | -0.4159 | -0.5408 to -0.3067 |
| scifact | shared | 921 | all known positives | worst rank in the set | 69 | 7.7101 | 19.0725 | -11.3623 | -20.4348 to -4.7101 |
| scifact | own | 102 | the mask's own positive | reciprocal rank | 69 | 0.9660 | 0.4383 | 0.5277 | 0.4619 to 0.5882 |
| scifact | own | 102 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 102 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 102 | the mask's own positive | log rank (lower is better) | 69 | 0.0480 | 1.3133 | -1.2653 | -1.5042 to -1.0501 |
| scifact | own | 102 | the mask's own positive | worst rank in the set | 69 | 1.0713 | 11.4116 | -10.3403 | -18.7467 to -4.9031 |
| scifact | own | 204 | the mask's own positive | reciprocal rank | 69 | 0.9696 | 0.4383 | 0.5313 | 0.4658 to 0.5925 |
| scifact | own | 204 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 204 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 204 | the mask's own positive | log rank (lower is better) | 69 | 0.0430 | 1.3133 | -1.2703 | -1.5092 to -1.0546 |
| scifact | own | 204 | the mask's own positive | worst rank in the set | 69 | 1.0640 | 11.4116 | -10.3476 | -18.7752 to -4.9147 |
| scifact | own | 307 | the mask's own positive | reciprocal rank | 69 | 0.9630 | 0.4383 | 0.5247 | 0.4594 to 0.5852 |
| scifact | own | 307 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 307 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 307 | the mask's own positive | log rank (lower is better) | 69 | 0.0522 | 1.3133 | -1.2611 | -1.5003 to -1.0468 |
| scifact | own | 307 | the mask's own positive | worst rank in the set | 69 | 1.0773 | 11.4116 | -10.3343 | -18.7713 to -4.9026 |
| scifact | own | 409 | the mask's own positive | reciprocal rank | 69 | 0.9235 | 0.4383 | 0.4853 | 0.4154 to 0.5506 |
| scifact | own | 409 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 409 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 409 | the mask's own positive | log rank (lower is better) | 69 | 0.1090 | 1.3133 | -1.2043 | -1.4434 to -0.9811 |
| scifact | own | 409 | the mask's own positive | worst rank in the set | 69 | 1.1643 | 11.4116 | -10.2473 | -18.6847 to -4.8297 |
| scifact | own | 512 | the mask's own positive | reciprocal rank | 69 | 0.8952 | 0.4383 | 0.4570 | 0.3883 to 0.5231 |
| scifact | own | 512 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 512 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 512 | the mask's own positive | log rank (lower is better) | 69 | 0.1547 | 1.3133 | -1.1586 | -1.3995 to -0.9391 |
| scifact | own | 512 | the mask's own positive | worst rank in the set | 69 | 1.2476 | 11.4116 | -10.1640 | -18.6506 to -4.7270 |
| scifact | own | 614 | the mask's own positive | reciprocal rank | 69 | 0.8742 | 0.4383 | 0.4360 | 0.3720 to 0.5014 |
| scifact | own | 614 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 614 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 614 | the mask's own positive | log rank (lower is better) | 69 | 0.1897 | 1.3133 | -1.1236 | -1.3618 to -0.9094 |
| scifact | own | 614 | the mask's own positive | worst rank in the set | 69 | 1.3164 | 11.4116 | -10.0952 | -18.5417 to -4.6541 |
| scifact | own | 716 | the mask's own positive | reciprocal rank | 69 | 0.8555 | 0.4383 | 0.4173 | 0.3527 to 0.4839 |
| scifact | own | 716 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 716 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 716 | the mask's own positive | log rank (lower is better) | 69 | 0.2157 | 1.3133 | -1.0976 | -1.3384 to -0.8825 |
| scifact | own | 716 | the mask's own positive | worst rank in the set | 69 | 1.3539 | 11.4116 | -10.0577 | -18.5146 to -4.6111 |
| scifact | own | 819 | the mask's own positive | reciprocal rank | 69 | 0.8003 | 0.4383 | 0.3620 | 0.2976 to 0.4280 |
| scifact | own | 819 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 819 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 819 | the mask's own positive | log rank (lower is better) | 69 | 0.3152 | 1.3133 | -0.9981 | -1.2193 to -0.8037 |
| scifact | own | 819 | the mask's own positive | worst rank in the set | 69 | 1.5800 | 11.4116 | -9.8316 | -18.2486 to -4.4248 |
| scifact | own | 921 | the mask's own positive | reciprocal rank | 69 | 0.7083 | 0.4383 | 0.2700 | 0.2171 to 0.3257 |
| scifact | own | 921 | the mask's own positive | in top 10 | 69 | 0.9783 | 0.8287 | 0.1495 | 0.0942 to 0.2126 |
| scifact | own | 921 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 921 | the mask's own positive | log rank (lower is better) | 69 | 0.5092 | 1.3133 | -0.8041 | -0.9811 to -0.6498 |
| scifact | own | 921 | the mask's own positive | worst rank in the set | 69 | 2.2106 | 11.4116 | -9.2010 | -17.2844 to -3.9750 |
| scifact | 1-1 | 102 | held-out | reciprocal rank | 69 | 0.3333 | 0.4346 | -0.1013 | -0.1378 to -0.0641 |
| scifact | 1-1 | 102 | held-out | in top 10 | 69 | 0.8816 | 0.8309 | 0.0507 | 0.0024 to 0.0991 |
| scifact | 1-1 | 102 | held-out | in top 100 | 69 | 0.9758 | 0.9807 | -0.0048 | -0.0290 to 0.0193 |
| scifact | 1-1 | 102 | held-out | log rank (lower is better) | 69 | 1.4306 | 1.3197 | 0.1109 | -0.0223 to 0.2312 |
| scifact | 1-1 | 102 | held-out | worst rank in the set | 69 | 11.6987 | 11.4058 | 0.2929 | -2.1880 to 3.2870 |
| scifact | 1-1 | 102 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4615 | 0.5243 | -0.0628 | -0.0969 to -0.0253 |
| scifact | 1-1 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5327 | 0.5066 | 0.0261 | -0.0236 to 0.0785 |
| scifact | 1-1 | 204 | held-out | reciprocal rank | 69 | 0.3387 | 0.4346 | -0.0959 | -0.1330 to -0.0606 |
| scifact | 1-1 | 204 | held-out | in top 10 | 69 | 0.8816 | 0.8309 | 0.0507 | 0.0060 to 0.0979 |
| scifact | 1-1 | 204 | held-out | in top 100 | 69 | 0.9758 | 0.9807 | -0.0048 | -0.0290 to 0.0193 |
| scifact | 1-1 | 204 | held-out | log rank (lower is better) | 69 | 1.3870 | 1.3197 | 0.0672 | -0.0619 to 0.1814 |
| scifact | 1-1 | 204 | held-out | worst rank in the set | 69 | 11.2802 | 11.4058 | -0.1256 | -2.5600 to 2.7377 |
| scifact | 1-1 | 204 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4667 | 0.5243 | -0.0576 | -0.0919 to -0.0217 |
| scifact | 1-1 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5512 | 0.5066 | 0.0446 | -0.0053 to 0.0948 |
| scifact | 1-1 | 307 | held-out | reciprocal rank | 69 | 0.3414 | 0.4346 | -0.0932 | -0.1303 to -0.0568 |
| scifact | 1-1 | 307 | held-out | in top 10 | 69 | 0.8798 | 0.8309 | 0.0489 | -0.0024 to 0.1063 |
| scifact | 1-1 | 307 | held-out | in top 100 | 69 | 0.9783 | 0.9807 | -0.0024 | -0.0290 to 0.0242 |
| scifact | 1-1 | 307 | held-out | log rank (lower is better) | 69 | 1.3638 | 1.3197 | 0.0440 | -0.0850 to 0.1609 |
| scifact | 1-1 | 307 | held-out | worst rank in the set | 69 | 11.0586 | 11.4058 | -0.3472 | -2.9580 to 2.6385 |
| scifact | 1-1 | 307 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4684 | 0.5243 | -0.0559 | -0.0914 to -0.0188 |
| scifact | 1-1 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5476 | 0.5066 | 0.0409 | -0.0071 to 0.0906 |
| scifact | 1-1 | 409 | held-out | reciprocal rank | 69 | 0.3636 | 0.4346 | -0.0710 | -0.1051 to -0.0369 |
| scifact | 1-1 | 409 | held-out | in top 10 | 69 | 0.8750 | 0.8309 | 0.0441 | -0.0000 to 0.0888 |
| scifact | 1-1 | 409 | held-out | in top 100 | 69 | 0.9783 | 0.9807 | -0.0024 | -0.0290 to 0.0242 |
| scifact | 1-1 | 409 | held-out | log rank (lower is better) | 69 | 1.3285 | 1.3197 | 0.0087 | -0.1118 to 0.1126 |
| scifact | 1-1 | 409 | held-out | worst rank in the set | 69 | 10.6268 | 11.4058 | -0.7790 | -3.2181 to 1.8220 |
| scifact | 1-1 | 409 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4834 | 0.5243 | -0.0408 | -0.0721 to -0.0076 |
| scifact | 1-1 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5520 | 0.5066 | 0.0453 | 0.0037 to 0.0883 |
| scifact | 1-1 | 512 | held-out | reciprocal rank | 69 | 0.3753 | 0.4346 | -0.0593 | -0.0929 to -0.0286 |
| scifact | 1-1 | 512 | held-out | in top 10 | 69 | 0.8678 | 0.8309 | 0.0368 | -0.0042 to 0.0803 |
| scifact | 1-1 | 512 | held-out | in top 100 | 69 | 0.9783 | 0.9807 | -0.0024 | -0.0290 to 0.0242 |
| scifact | 1-1 | 512 | held-out | log rank (lower is better) | 69 | 1.3018 | 1.3197 | -0.0180 | -0.1273 to 0.0778 |
| scifact | 1-1 | 512 | held-out | worst rank in the set | 69 | 11.1413 | 11.4058 | -0.2645 | -3.3479 to 3.5985 |
| scifact | 1-1 | 512 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4904 | 0.5243 | -0.0339 | -0.0634 to -0.0044 |
| scifact | 1-1 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5652 | 0.5066 | 0.0585 | 0.0213 to 0.0957 |
| scifact | 1-1 | 614 | held-out | reciprocal rank | 69 | 0.3760 | 0.4346 | -0.0586 | -0.0902 to -0.0298 |
| scifact | 1-1 | 614 | held-out | in top 10 | 69 | 0.8641 | 0.8309 | 0.0332 | -0.0079 to 0.0767 |
| scifact | 1-1 | 614 | held-out | in top 100 | 69 | 0.9783 | 0.9807 | -0.0024 | -0.0290 to 0.0242 |
| scifact | 1-1 | 614 | held-out | log rank (lower is better) | 69 | 1.2983 | 1.3197 | -0.0215 | -0.1299 to 0.0700 |
| scifact | 1-1 | 614 | held-out | worst rank in the set | 69 | 10.4746 | 11.4058 | -0.9312 | -3.6143 to 1.9288 |
| scifact | 1-1 | 614 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4898 | 0.5243 | -0.0345 | -0.0618 to -0.0056 |
| scifact | 1-1 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5672 | 0.5066 | 0.0605 | 0.0258 to 0.0964 |
| scifact | 1-1 | 716 | held-out | reciprocal rank | 69 | 0.3759 | 0.4346 | -0.0587 | -0.0883 to -0.0316 |
| scifact | 1-1 | 716 | held-out | in top 10 | 69 | 0.8557 | 0.8309 | 0.0248 | -0.0127 to 0.0664 |
| scifact | 1-1 | 716 | held-out | in top 100 | 69 | 0.9783 | 0.9807 | -0.0024 | -0.0290 to 0.0242 |
| scifact | 1-1 | 716 | held-out | log rank (lower is better) | 69 | 1.3201 | 1.3197 | 0.0003 | -0.0900 to 0.0819 |
| scifact | 1-1 | 716 | held-out | worst rank in the set | 69 | 10.5386 | 11.4058 | -0.8671 | -3.2391 to 1.6450 |
| scifact | 1-1 | 716 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4865 | 0.5243 | -0.0378 | -0.0631 to -0.0121 |
| scifact | 1-1 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5489 | 0.5066 | 0.0423 | 0.0168 to 0.0681 |
| scifact | 1-1 | 819 | held-out | reciprocal rank | 69 | 0.4028 | 0.4346 | -0.0318 | -0.0628 to -0.0041 |
| scifact | 1-1 | 819 | held-out | in top 10 | 69 | 0.8653 | 0.8309 | 0.0344 | -0.0000 to 0.0749 |
| scifact | 1-1 | 819 | held-out | in top 100 | 69 | 0.9831 | 0.9807 | 0.0024 | -0.0170 to 0.0242 |
| scifact | 1-1 | 819 | held-out | log rank (lower is better) | 69 | 1.2817 | 1.3197 | -0.0381 | -0.1174 to 0.0407 |
| scifact | 1-1 | 819 | held-out | worst rank in the set | 69 | 10.1395 | 11.4058 | -1.2663 | -3.2808 to 0.4470 |
| scifact | 1-1 | 819 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.5100 | 0.5243 | -0.0143 | -0.0416 to 0.0113 |
| scifact | 1-1 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5540 | 0.5066 | 0.0474 | 0.0207 to 0.0753 |
| scifact | 1-1 | 921 | held-out | reciprocal rank | 69 | 0.4109 | 0.4346 | -0.0237 | -0.0495 to -0.0010 |
| scifact | 1-1 | 921 | held-out | in top 10 | 69 | 0.8539 | 0.8309 | 0.0229 | -0.0048 to 0.0519 |
| scifact | 1-1 | 921 | held-out | in top 100 | 69 | 0.9903 | 0.9807 | 0.0097 | 0.0000 to 0.0266 |
| scifact | 1-1 | 921 | held-out | log rank (lower is better) | 69 | 1.2872 | 1.3197 | -0.0326 | -0.1020 to 0.0338 |
| scifact | 1-1 | 921 | held-out | worst rank in the set | 69 | 10.3593 | 11.4058 | -1.0465 | -2.3599 to -0.0071 |
| scifact | 1-1 | 921 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.5129 | 0.5243 | -0.0114 | -0.0334 to 0.0092 |
| scifact | 1-1 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5419 | 0.5066 | 0.0353 | 0.0139 to 0.0575 |
| scifact | 2-2 | 102 | held-out | reciprocal rank | 6 | 0.2633 | 0.3761 | -0.1128 | -0.1633 to -0.0633 |
| scifact | 2-2 | 102 | held-out | in top 10 | 6 | 0.9375 | 0.8854 | 0.0521 | 0.0000 to 0.1562 |
| scifact | 2-2 | 102 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 102 | held-out | log rank (lower is better) | 6 | 1.4698 | 1.2652 | 0.2046 | 0.0622 to 0.3324 |
| scifact | 2-2 | 102 | held-out | worst rank in the set | 6 | 6.1458 | 6.6389 | -0.4931 | -1.6837 to 0.4583 |
| scifact | 2-2 | 102 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5138 | 0.6031 | -0.0893 | -0.1523 to -0.0174 |
| scifact | 2-2 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5151 | 0.4889 | 0.0262 | -0.0147 to 0.0623 |
| scifact | 2-2 | 204 | held-out | reciprocal rank | 6 | 0.2693 | 0.3761 | -0.1068 | -0.1597 to -0.0514 |
| scifact | 2-2 | 204 | held-out | in top 10 | 6 | 0.9306 | 0.8854 | 0.0451 | 0.0000 to 0.1193 |
| scifact | 2-2 | 204 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 204 | held-out | log rank (lower is better) | 6 | 1.4565 | 1.2652 | 0.1913 | 0.0439 to 0.3084 |
| scifact | 2-2 | 204 | held-out | worst rank in the set | 6 | 6.1319 | 6.6389 | -0.5069 | -1.6944 to 0.3889 |
| scifact | 2-2 | 204 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5172 | 0.6031 | -0.0858 | -0.1477 to -0.0220 |
| scifact | 2-2 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5141 | 0.4889 | 0.0252 | -0.0195 to 0.0652 |
| scifact | 2-2 | 307 | held-out | reciprocal rank | 6 | 0.2836 | 0.3761 | -0.0925 | -0.1456 to -0.0432 |
| scifact | 2-2 | 307 | held-out | in top 10 | 6 | 0.9583 | 0.8854 | 0.0729 | 0.0000 to 0.1750 |
| scifact | 2-2 | 307 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 307 | held-out | log rank (lower is better) | 6 | 1.4269 | 1.2652 | 0.1617 | 0.0121 to 0.2736 |
| scifact | 2-2 | 307 | held-out | worst rank in the set | 6 | 6.0417 | 6.6389 | -0.5972 | -1.6683 to 0.2500 |
| scifact | 2-2 | 307 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5401 | 0.6031 | -0.0630 | -0.1317 to 0.0041 |
| scifact | 2-2 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5256 | 0.4889 | 0.0367 | -0.0103 to 0.0871 |
| scifact | 2-2 | 409 | held-out | reciprocal rank | 6 | 0.2931 | 0.3761 | -0.0830 | -0.1300 to -0.0388 |
| scifact | 2-2 | 409 | held-out | in top 10 | 6 | 0.9583 | 0.8854 | 0.0729 | 0.0000 to 0.1750 |
| scifact | 2-2 | 409 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 409 | held-out | log rank (lower is better) | 6 | 1.4002 | 1.2652 | 0.1350 | 0.0117 to 0.2431 |
| scifact | 2-2 | 409 | held-out | worst rank in the set | 6 | 5.8819 | 6.6389 | -0.7569 | -1.8958 to 0.0833 |
| scifact | 2-2 | 409 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5492 | 0.6031 | -0.0539 | -0.1124 to 0.0095 |
| scifact | 2-2 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5283 | 0.4889 | 0.0394 | -0.0046 to 0.0868 |
| scifact | 2-2 | 512 | held-out | reciprocal rank | 6 | 0.3057 | 0.3761 | -0.0704 | -0.1253 to -0.0279 |
| scifact | 2-2 | 512 | held-out | in top 10 | 6 | 0.9688 | 0.8854 | 0.0833 | 0.0000 to 0.2000 |
| scifact | 2-2 | 512 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 512 | held-out | log rank (lower is better) | 6 | 1.3750 | 1.2652 | 0.1098 | -0.0044 to 0.2117 |
| scifact | 2-2 | 512 | held-out | worst rank in the set | 6 | 5.9236 | 6.6389 | -0.7153 | -1.8200 to 0.1667 |
| scifact | 2-2 | 512 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5649 | 0.6031 | -0.0381 | -0.1072 to 0.0299 |
| scifact | 2-2 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5227 | 0.4889 | 0.0337 | -0.0108 to 0.0792 |
| scifact | 2-2 | 614 | held-out | reciprocal rank | 6 | 0.3218 | 0.3761 | -0.0543 | -0.0833 to -0.0278 |
| scifact | 2-2 | 614 | held-out | in top 10 | 6 | 0.9688 | 0.8854 | 0.0833 | 0.0000 to 0.2000 |
| scifact | 2-2 | 614 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 614 | held-out | log rank (lower is better) | 6 | 1.3427 | 1.2652 | 0.0775 | -0.0034 to 0.1477 |
| scifact | 2-2 | 614 | held-out | worst rank in the set | 6 | 5.9167 | 6.6389 | -0.7222 | -1.8056 to 0.1667 |
| scifact | 2-2 | 614 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5800 | 0.6031 | -0.0231 | -0.0684 to 0.0273 |
| scifact | 2-2 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5213 | 0.4889 | 0.0324 | -0.0120 to 0.0788 |
| scifact | 2-2 | 716 | held-out | reciprocal rank | 6 | 0.3256 | 0.3761 | -0.0506 | -0.0822 to -0.0208 |
| scifact | 2-2 | 716 | held-out | in top 10 | 6 | 0.9688 | 0.8854 | 0.0833 | 0.0000 to 0.2000 |
| scifact | 2-2 | 716 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 716 | held-out | log rank (lower is better) | 6 | 1.3298 | 1.2652 | 0.0646 | -0.0140 to 0.1321 |
| scifact | 2-2 | 716 | held-out | worst rank in the set | 6 | 5.9097 | 6.6389 | -0.7292 | -1.7100 to 0.0861 |
| scifact | 2-2 | 716 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5836 | 0.6031 | -0.0194 | -0.0671 to 0.0290 |
| scifact | 2-2 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5207 | 0.4889 | 0.0318 | -0.0038 to 0.0691 |
| scifact | 2-2 | 819 | held-out | reciprocal rank | 6 | 0.3338 | 0.3761 | -0.0423 | -0.0656 to -0.0180 |
| scifact | 2-2 | 819 | held-out | in top 10 | 6 | 0.9688 | 0.8854 | 0.0833 | 0.0000 to 0.2000 |
| scifact | 2-2 | 819 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 819 | held-out | log rank (lower is better) | 6 | 1.3030 | 1.2652 | 0.0378 | -0.0273 to 0.0916 |
| scifact | 2-2 | 819 | held-out | worst rank in the set | 6 | 5.7708 | 6.6389 | -0.8681 | -1.8448 to 0.0000 |
| scifact | 2-2 | 819 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5920 | 0.6031 | -0.0110 | -0.0494 to 0.0289 |
| scifact | 2-2 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5186 | 0.4889 | 0.0297 | -0.0013 to 0.0592 |
| scifact | 2-2 | 921 | held-out | reciprocal rank | 6 | 0.3404 | 0.3761 | -0.0357 | -0.0630 to -0.0084 |
| scifact | 2-2 | 921 | held-out | in top 10 | 6 | 0.9201 | 0.8854 | 0.0347 | 0.0000 to 0.0833 |
| scifact | 2-2 | 921 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 921 | held-out | log rank (lower is better) | 6 | 1.2862 | 1.2652 | 0.0210 | -0.0373 to 0.0627 |
| scifact | 2-2 | 921 | held-out | worst rank in the set | 6 | 5.8889 | 6.6389 | -0.7500 | -1.5833 to -0.0235 |
| scifact | 2-2 | 921 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5822 | 0.6031 | -0.0208 | -0.0489 to 0.0048 |
| scifact | 2-2 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5199 | 0.4889 | 0.0310 | 0.0047 to 0.0559 |
| nfcorpus | prefix | 102 | all known positives | reciprocal rank | 2162 | 0.0530 | 0.0717 | -0.0187 | -0.0212 to -0.0164 |
| nfcorpus | prefix | 102 | all known positives | in top 10 | 2162 | 0.1028 | 0.1400 | -0.0372 | -0.0417 to -0.0328 |
| nfcorpus | prefix | 102 | all known positives | in top 100 | 2162 | 0.2505 | 0.3104 | -0.0599 | -0.0655 to -0.0541 |
| nfcorpus | prefix | 102 | all known positives | log rank (lower is better) | 2162 | 5.7483 | 5.4229 | 0.3254 | 0.3031 to 0.3488 |
| nfcorpus | prefix | 102 | all known positives | worst rank in the set | 2162 | 2688.3876 | 2569.7812 | 118.6064 | 95.5373 to 141.5358 |
| nfcorpus | prefix | 204 | all known positives | reciprocal rank | 2162 | 0.0612 | 0.0717 | -0.0105 | -0.0122 to -0.0088 |
| nfcorpus | prefix | 204 | all known positives | in top 10 | 2162 | 0.1174 | 0.1400 | -0.0226 | -0.0262 to -0.0192 |
| nfcorpus | prefix | 204 | all known positives | in top 100 | 2162 | 0.2717 | 0.3104 | -0.0387 | -0.0433 to -0.0342 |
| nfcorpus | prefix | 204 | all known positives | log rank (lower is better) | 2162 | 5.6241 | 5.4229 | 0.2012 | 0.1839 to 0.2185 |
| nfcorpus | prefix | 204 | all known positives | worst rank in the set | 2162 | 2640.7576 | 2569.7812 | 70.9764 | 51.1435 to 90.4125 |
| nfcorpus | prefix | 307 | all known positives | reciprocal rank | 2162 | 0.0648 | 0.0717 | -0.0069 | -0.0084 to -0.0054 |
| nfcorpus | prefix | 307 | all known positives | in top 10 | 2162 | 0.1234 | 0.1400 | -0.0166 | -0.0197 to -0.0135 |
| nfcorpus | prefix | 307 | all known positives | in top 100 | 2162 | 0.2848 | 0.3104 | -0.0255 | -0.0296 to -0.0217 |
| nfcorpus | prefix | 307 | all known positives | log rank (lower is better) | 2162 | 5.5673 | 5.4229 | 0.1444 | 0.1310 to 0.1582 |
| nfcorpus | prefix | 307 | all known positives | worst rank in the set | 2162 | 2624.4662 | 2569.7812 | 54.6850 | 38.3701 to 71.2069 |
| nfcorpus | prefix | 409 | all known positives | reciprocal rank | 2162 | 0.0680 | 0.0717 | -0.0037 | -0.0054 to -0.0021 |
| nfcorpus | prefix | 409 | all known positives | in top 10 | 2162 | 0.1279 | 0.1400 | -0.0120 | -0.0152 to -0.0092 |
| nfcorpus | prefix | 409 | all known positives | in top 100 | 2162 | 0.2956 | 0.3104 | -0.0148 | -0.0189 to -0.0109 |
| nfcorpus | prefix | 409 | all known positives | log rank (lower is better) | 2162 | 5.5151 | 5.4229 | 0.0922 | 0.0796 to 0.1048 |
| nfcorpus | prefix | 409 | all known positives | worst rank in the set | 2162 | 2616.9204 | 2569.7812 | 47.1392 | 33.5969 to 60.9232 |
| nfcorpus | prefix | 512 | all known positives | reciprocal rank | 2162 | 0.0689 | 0.0717 | -0.0028 | -0.0042 to -0.0016 |
| nfcorpus | prefix | 512 | all known positives | in top 10 | 2162 | 0.1323 | 0.1400 | -0.0076 | -0.0100 to -0.0053 |
| nfcorpus | prefix | 512 | all known positives | in top 100 | 2162 | 0.3008 | 0.3104 | -0.0096 | -0.0133 to -0.0059 |
| nfcorpus | prefix | 512 | all known positives | log rank (lower is better) | 2162 | 5.4769 | 5.4229 | 0.0540 | 0.0438 to 0.0650 |
| nfcorpus | prefix | 512 | all known positives | worst rank in the set | 2162 | 2599.2206 | 2569.7812 | 29.4394 | 17.5934 to 41.2642 |
| nfcorpus | prefix | 614 | all known positives | reciprocal rank | 2162 | 0.0698 | 0.0717 | -0.0019 | -0.0031 to -0.0007 |
| nfcorpus | prefix | 614 | all known positives | in top 10 | 2162 | 0.1350 | 0.1400 | -0.0050 | -0.0072 to -0.0027 |
| nfcorpus | prefix | 614 | all known positives | in top 100 | 2162 | 0.3044 | 0.3104 | -0.0059 | -0.0091 to -0.0029 |
| nfcorpus | prefix | 614 | all known positives | log rank (lower is better) | 2162 | 5.4567 | 5.4229 | 0.0338 | 0.0259 to 0.0425 |
| nfcorpus | prefix | 614 | all known positives | worst rank in the set | 2162 | 2585.8150 | 2569.7812 | 16.0338 | 6.3953 to 25.9346 |
| nfcorpus | prefix | 716 | all known positives | reciprocal rank | 2162 | 0.0709 | 0.0717 | -0.0008 | -0.0018 to 0.0000 |
| nfcorpus | prefix | 716 | all known positives | in top 10 | 2162 | 0.1363 | 0.1400 | -0.0036 | -0.0054 to -0.0018 |
| nfcorpus | prefix | 716 | all known positives | in top 100 | 2162 | 0.3052 | 0.3104 | -0.0052 | -0.0079 to -0.0026 |
| nfcorpus | prefix | 716 | all known positives | log rank (lower is better) | 2162 | 5.4504 | 5.4229 | 0.0275 | 0.0212 to 0.0343 |
| nfcorpus | prefix | 716 | all known positives | worst rank in the set | 2162 | 2582.6078 | 2569.7812 | 12.8265 | 3.9789 to 21.4219 |
| nfcorpus | prefix | 819 | all known positives | reciprocal rank | 2162 | 0.0712 | 0.0717 | -0.0005 | -0.0013 to 0.0003 |
| nfcorpus | prefix | 819 | all known positives | in top 10 | 2162 | 0.1379 | 0.1400 | -0.0021 | -0.0038 to -0.0005 |
| nfcorpus | prefix | 819 | all known positives | in top 100 | 2162 | 0.3079 | 0.3104 | -0.0025 | -0.0048 to -0.0001 |
| nfcorpus | prefix | 819 | all known positives | log rank (lower is better) | 2162 | 5.4379 | 5.4229 | 0.0150 | 0.0100 to 0.0202 |
| nfcorpus | prefix | 819 | all known positives | worst rank in the set | 2162 | 2577.9514 | 2569.7812 | 8.1702 | 1.1616 to 15.1985 |
| nfcorpus | prefix | 921 | all known positives | reciprocal rank | 2162 | 0.0714 | 0.0717 | -0.0003 | -0.0007 to 0.0002 |
| nfcorpus | prefix | 921 | all known positives | in top 10 | 2162 | 0.1376 | 0.1400 | -0.0023 | -0.0036 to -0.0011 |
| nfcorpus | prefix | 921 | all known positives | in top 100 | 2162 | 0.3094 | 0.3104 | -0.0010 | -0.0029 to 0.0008 |
| nfcorpus | prefix | 921 | all known positives | log rank (lower is better) | 2162 | 5.4334 | 5.4229 | 0.0105 | 0.0072 to 0.0137 |
| nfcorpus | prefix | 921 | all known positives | worst rank in the set | 2162 | 2580.7618 | 2569.7812 | 10.9806 | 6.2030 to 15.9981 |
| nfcorpus | shared | 102 | all known positives | reciprocal rank | 2162 | 0.1535 | 0.0717 | 0.0818 | 0.0773 to 0.0864 |
| nfcorpus | shared | 102 | all known positives | in top 10 | 2162 | 0.3000 | 0.1400 | 0.1601 | 0.1522 to 0.1687 |
| nfcorpus | shared | 102 | all known positives | in top 100 | 2162 | 0.5580 | 0.3104 | 0.2476 | 0.2381 to 0.2570 |
| nfcorpus | shared | 102 | all known positives | log rank (lower is better) | 2162 | 4.0215 | 5.4229 | -1.4014 | -1.4455 to -1.3547 |
| nfcorpus | shared | 102 | all known positives | worst rank in the set | 2162 | 1783.2604 | 2569.7812 | -786.5208 | -817.9935 to -753.6591 |
| nfcorpus | shared | 204 | all known positives | reciprocal rank | 2162 | 0.1574 | 0.0717 | 0.0857 | 0.0812 to 0.0902 |
| nfcorpus | shared | 204 | all known positives | in top 10 | 2162 | 0.3110 | 0.1400 | 0.1711 | 0.1631 to 0.1792 |
| nfcorpus | shared | 204 | all known positives | in top 100 | 2162 | 0.5687 | 0.3104 | 0.2584 | 0.2488 to 0.2681 |
| nfcorpus | shared | 204 | all known positives | log rank (lower is better) | 2162 | 3.9610 | 5.4229 | -1.4619 | -1.5083 to -1.4140 |
| nfcorpus | shared | 204 | all known positives | worst rank in the set | 2162 | 1742.6776 | 2569.7812 | -827.1036 | -857.6654 to -795.8324 |
| nfcorpus | shared | 307 | all known positives | reciprocal rank | 2162 | 0.1545 | 0.0717 | 0.0828 | 0.0784 to 0.0873 |
| nfcorpus | shared | 307 | all known positives | in top 10 | 2162 | 0.3064 | 0.1400 | 0.1664 | 0.1586 to 0.1745 |
| nfcorpus | shared | 307 | all known positives | in top 100 | 2162 | 0.5654 | 0.3104 | 0.2550 | 0.2450 to 0.2647 |
| nfcorpus | shared | 307 | all known positives | log rank (lower is better) | 2162 | 3.9780 | 5.4229 | -1.4449 | -1.4906 to -1.3979 |
| nfcorpus | shared | 307 | all known positives | worst rank in the set | 2162 | 1748.7965 | 2569.7812 | -820.9847 | -851.7491 to -788.9453 |
| nfcorpus | shared | 409 | all known positives | reciprocal rank | 2162 | 0.1499 | 0.0717 | 0.0782 | 0.0738 to 0.0826 |
| nfcorpus | shared | 409 | all known positives | in top 10 | 2162 | 0.2983 | 0.1400 | 0.1583 | 0.1506 to 0.1663 |
| nfcorpus | shared | 409 | all known positives | in top 100 | 2162 | 0.5605 | 0.3104 | 0.2501 | 0.2405 to 0.2600 |
| nfcorpus | shared | 409 | all known positives | log rank (lower is better) | 2162 | 4.0223 | 5.4229 | -1.4006 | -1.4460 to -1.3549 |
| nfcorpus | shared | 409 | all known positives | worst rank in the set | 2162 | 1761.2072 | 2569.7812 | -808.5740 | -838.3423 to -777.5375 |
| nfcorpus | shared | 512 | all known positives | reciprocal rank | 2162 | 0.1451 | 0.0717 | 0.0734 | 0.0693 to 0.0777 |
| nfcorpus | shared | 512 | all known positives | in top 10 | 2162 | 0.2888 | 0.1400 | 0.1488 | 0.1411 to 0.1568 |
| nfcorpus | shared | 512 | all known positives | in top 100 | 2162 | 0.5515 | 0.3104 | 0.2412 | 0.2315 to 0.2508 |
| nfcorpus | shared | 512 | all known positives | log rank (lower is better) | 2162 | 4.0738 | 5.4229 | -1.3491 | -1.3937 to -1.3043 |
| nfcorpus | shared | 512 | all known positives | worst rank in the set | 2162 | 1784.9514 | 2569.7812 | -784.8298 | -813.6963 to -754.6985 |
| nfcorpus | shared | 614 | all known positives | reciprocal rank | 2162 | 0.1412 | 0.0717 | 0.0695 | 0.0655 to 0.0737 |
| nfcorpus | shared | 614 | all known positives | in top 10 | 2162 | 0.2799 | 0.1400 | 0.1399 | 0.1320 to 0.1480 |
| nfcorpus | shared | 614 | all known positives | in top 100 | 2162 | 0.5426 | 0.3104 | 0.2322 | 0.2227 to 0.2418 |
| nfcorpus | shared | 614 | all known positives | log rank (lower is better) | 2162 | 4.1249 | 5.4229 | -1.2980 | -1.3417 to -1.2541 |
| nfcorpus | shared | 614 | all known positives | worst rank in the set | 2162 | 1808.8992 | 2569.7812 | -760.8821 | -789.4536 to -730.9008 |
| nfcorpus | shared | 716 | all known positives | reciprocal rank | 2162 | 0.1363 | 0.0717 | 0.0646 | 0.0606 to 0.0687 |
| nfcorpus | shared | 716 | all known positives | in top 10 | 2162 | 0.2688 | 0.1400 | 0.1289 | 0.1213 to 0.1367 |
| nfcorpus | shared | 716 | all known positives | in top 100 | 2162 | 0.5322 | 0.3104 | 0.2218 | 0.2121 to 0.2313 |
| nfcorpus | shared | 716 | all known positives | log rank (lower is better) | 2162 | 4.1855 | 5.4229 | -1.2374 | -1.2801 to -1.1952 |
| nfcorpus | shared | 716 | all known positives | worst rank in the set | 2162 | 1834.3164 | 2569.7812 | -735.4648 | -763.0602 to -706.6785 |
| nfcorpus | shared | 819 | all known positives | reciprocal rank | 2162 | 0.1237 | 0.0717 | 0.0520 | 0.0486 to 0.0555 |
| nfcorpus | shared | 819 | all known positives | in top 10 | 2162 | 0.2431 | 0.1400 | 0.1032 | 0.0966 to 0.1103 |
| nfcorpus | shared | 819 | all known positives | in top 100 | 2162 | 0.5085 | 0.3104 | 0.1982 | 0.1896 to 0.2076 |
| nfcorpus | shared | 819 | all known positives | log rank (lower is better) | 2162 | 4.3323 | 5.4229 | -1.0906 | -1.1275 to -1.0540 |
| nfcorpus | shared | 819 | all known positives | worst rank in the set | 2162 | 1895.3034 | 2569.7812 | -674.4778 | -700.9594 to -647.2940 |
| nfcorpus | shared | 921 | all known positives | reciprocal rank | 2162 | 0.1021 | 0.0717 | 0.0304 | 0.0280 to 0.0329 |
| nfcorpus | shared | 921 | all known positives | in top 10 | 2162 | 0.2000 | 0.1400 | 0.0601 | 0.0551 to 0.0652 |
| nfcorpus | shared | 921 | all known positives | in top 100 | 2162 | 0.4464 | 0.3104 | 0.1360 | 0.1288 to 0.1434 |
| nfcorpus | shared | 921 | all known positives | log rank (lower is better) | 2162 | 4.6574 | 5.4229 | -0.7655 | -0.7910 to -0.7408 |
| nfcorpus | shared | 921 | all known positives | worst rank in the set | 2162 | 2050.0920 | 2569.7812 | -519.6892 | -541.0685 to -498.3255 |
| nfcorpus | own | 102 | the mask's own positive | reciprocal rank | 2162 | 0.9310 | 0.0717 | 0.8593 | 0.8533 to 0.8652 |
| nfcorpus | own | 102 | the mask's own positive | in top 10 | 2162 | 0.9994 | 0.1400 | 0.8595 | 0.8520 to 0.8671 |
| nfcorpus | own | 102 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7005 |
| nfcorpus | own | 102 | the mask's own positive | log rank (lower is better) | 2162 | 0.1082 | 5.4229 | -5.3147 | -5.3703 to -5.2598 |
| nfcorpus | own | 102 | the mask's own positive | worst rank in the set | 2162 | 1.1999 | 902.8887 | -901.6888 | -924.4317 to -880.5645 |
| nfcorpus | own | 204 | the mask's own positive | reciprocal rank | 2162 | 0.9148 | 0.0717 | 0.8431 | 0.8364 to 0.8499 |
| nfcorpus | own | 204 | the mask's own positive | in top 10 | 2162 | 0.9993 | 0.1400 | 0.8593 | 0.8519 to 0.8670 |
| nfcorpus | own | 204 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7005 |
| nfcorpus | own | 204 | the mask's own positive | log rank (lower is better) | 2162 | 0.1351 | 5.4229 | -5.2878 | -5.3452 to -5.2323 |
| nfcorpus | own | 204 | the mask's own positive | worst rank in the set | 2162 | 1.2536 | 902.8887 | -901.6351 | -924.3833 to -880.5129 |
| nfcorpus | own | 307 | the mask's own positive | reciprocal rank | 2162 | 0.8808 | 0.0717 | 0.8091 | 0.8014 to 0.8169 |
| nfcorpus | own | 307 | the mask's own positive | in top 10 | 2162 | 0.9978 | 0.1400 | 0.8578 | 0.8503 to 0.8654 |
| nfcorpus | own | 307 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7005 |
| nfcorpus | own | 307 | the mask's own positive | log rank (lower is better) | 2162 | 0.1951 | 5.4229 | -5.2278 | -5.2856 to -5.1720 |
| nfcorpus | own | 307 | the mask's own positive | worst rank in the set | 2162 | 1.3933 | 902.8887 | -901.4955 | -924.2455 to -880.3795 |
| nfcorpus | own | 409 | the mask's own positive | reciprocal rank | 2162 | 0.8386 | 0.0717 | 0.7669 | 0.7580 to 0.7757 |
| nfcorpus | own | 409 | the mask's own positive | in top 10 | 2162 | 0.9943 | 0.1400 | 0.8544 | 0.8469 to 0.8619 |
| nfcorpus | own | 409 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7005 |
| nfcorpus | own | 409 | the mask's own positive | log rank (lower is better) | 2162 | 0.2750 | 5.4229 | -5.1479 | -5.2067 to -5.0913 |
| nfcorpus | own | 409 | the mask's own positive | worst rank in the set | 2162 | 1.6089 | 902.8887 | -901.2799 | -924.0270 to -880.1696 |
| nfcorpus | own | 512 | the mask's own positive | reciprocal rank | 2162 | 0.7898 | 0.0717 | 0.7181 | 0.7083 to 0.7277 |
| nfcorpus | own | 512 | the mask's own positive | in top 10 | 2162 | 0.9872 | 0.1400 | 0.8473 | 0.8395 to 0.8550 |
| nfcorpus | own | 512 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7004 |
| nfcorpus | own | 512 | the mask's own positive | log rank (lower is better) | 2162 | 0.3740 | 5.4229 | -5.0489 | -5.1088 to -4.9916 |
| nfcorpus | own | 512 | the mask's own positive | worst rank in the set | 2162 | 1.9154 | 902.8887 | -900.9733 | -923.7205 to -879.8652 |
| nfcorpus | own | 614 | the mask's own positive | reciprocal rank | 2162 | 0.7461 | 0.0717 | 0.6744 | 0.6640 to 0.6847 |
| nfcorpus | own | 614 | the mask's own positive | in top 10 | 2162 | 0.9771 | 0.1400 | 0.8371 | 0.8291 to 0.8452 |
| nfcorpus | own | 614 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7004 |
| nfcorpus | own | 614 | the mask's own positive | log rank (lower is better) | 2162 | 0.4693 | 5.4229 | -4.9536 | -5.0138 to -4.8963 |
| nfcorpus | own | 614 | the mask's own positive | worst rank in the set | 2162 | 2.2617 | 902.8887 | -900.6270 | -923.3780 to -879.5234 |
| nfcorpus | own | 716 | the mask's own positive | reciprocal rank | 2162 | 0.6726 | 0.0717 | 0.6009 | 0.5903 to 0.6114 |
| nfcorpus | own | 716 | the mask's own positive | in top 10 | 2162 | 0.9472 | 0.1400 | 0.8073 | 0.7987 to 0.8160 |
| nfcorpus | own | 716 | the mask's own positive | in top 100 | 2162 | 0.9998 | 0.3104 | 0.6895 | 0.6787 to 0.7003 |
| nfcorpus | own | 716 | the mask's own positive | log rank (lower is better) | 2162 | 0.6513 | 5.4229 | -4.7716 | -4.8320 to -4.7137 |
| nfcorpus | own | 716 | the mask's own positive | worst rank in the set | 2162 | 3.1552 | 902.8887 | -899.7335 | -922.4990 to -878.6409 |
| nfcorpus | own | 819 | the mask's own positive | reciprocal rank | 2162 | 0.5192 | 0.0717 | 0.4475 | 0.4373 to 0.4578 |
| nfcorpus | own | 819 | the mask's own positive | in top 10 | 2162 | 0.8309 | 0.1400 | 0.6910 | 0.6810 to 0.7017 |
| nfcorpus | own | 819 | the mask's own positive | in top 100 | 2162 | 0.9935 | 0.3104 | 0.6831 | 0.6725 to 0.6939 |
| nfcorpus | own | 819 | the mask's own positive | log rank (lower is better) | 2162 | 1.1394 | 5.4229 | -4.2835 | -4.3424 to -4.2264 |
| nfcorpus | own | 819 | the mask's own positive | worst rank in the set | 2162 | 7.6777 | 902.8887 | -895.2110 | -917.9554 to -874.2050 |
| nfcorpus | own | 921 | the mask's own positive | reciprocal rank | 2162 | 0.2744 | 0.0717 | 0.2027 | 0.1963 to 0.2093 |
| nfcorpus | own | 921 | the mask's own positive | in top 10 | 2162 | 0.5164 | 0.1400 | 0.3765 | 0.3662 to 0.3871 |
| nfcorpus | own | 921 | the mask's own positive | in top 100 | 2162 | 0.8757 | 0.3104 | 0.5653 | 0.5543 to 0.5760 |
| nfcorpus | own | 921 | the mask's own positive | log rank (lower is better) | 2162 | 2.3990 | 5.4229 | -3.0239 | -3.0711 to -2.9787 |
| nfcorpus | own | 921 | the mask's own positive | worst rank in the set | 2162 | 48.9378 | 902.8887 | -853.9509 | -875.7315 to -833.7007 |
| nfcorpus | 1-1 | 102 | held-out | reciprocal rank | 2162 | 0.0664 | 0.0734 | -0.0070 | -0.0105 to -0.0037 |
| nfcorpus | 1-1 | 102 | held-out | in top 10 | 2162 | 0.1798 | 0.1440 | 0.0358 | 0.0289 to 0.0425 |
| nfcorpus | 1-1 | 102 | held-out | in top 100 | 2162 | 0.3998 | 0.3132 | 0.0866 | 0.0772 to 0.0955 |
| nfcorpus | 1-1 | 102 | held-out | log rank (lower is better) | 2162 | 4.9802 | 5.3993 | -0.4191 | -0.4582 to -0.3790 |
| nfcorpus | 1-1 | 102 | held-out | worst rank in the set | 2162 | 701.8894 | 896.8044 | -194.9150 | -212.6802 to -176.8685 |
| nfcorpus | 1-1 | 102 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0859 | 0.0843 | 0.0016 | -0.0024 to 0.0053 |
| nfcorpus | 1-1 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.0988 | 0.0765 | 0.0222 | 0.0182 to 0.0263 |
| nfcorpus | 1-1 | 204 | held-out | reciprocal rank | 2162 | 0.0687 | 0.0734 | -0.0047 | -0.0080 to -0.0017 |
| nfcorpus | 1-1 | 204 | held-out | in top 10 | 2162 | 0.1902 | 0.1440 | 0.0461 | 0.0396 to 0.0526 |
| nfcorpus | 1-1 | 204 | held-out | in top 100 | 2162 | 0.4161 | 0.3132 | 0.1029 | 0.0938 to 0.1117 |
| nfcorpus | 1-1 | 204 | held-out | log rank (lower is better) | 2162 | 4.9081 | 5.3993 | -0.4912 | -0.5283 to -0.4525 |
| nfcorpus | 1-1 | 204 | held-out | worst rank in the set | 2162 | 675.5328 | 896.8044 | -221.2717 | -238.3804 to -204.2119 |
| nfcorpus | 1-1 | 204 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0900 | 0.0843 | 0.0057 | 0.0020 to 0.0093 |
| nfcorpus | 1-1 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1011 | 0.0765 | 0.0246 | 0.0210 to 0.0283 |
| nfcorpus | 1-1 | 307 | held-out | reciprocal rank | 2162 | 0.0713 | 0.0734 | -0.0021 | -0.0053 to 0.0009 |
| nfcorpus | 1-1 | 307 | held-out | in top 10 | 2162 | 0.1910 | 0.1440 | 0.0469 | 0.0406 to 0.0533 |
| nfcorpus | 1-1 | 307 | held-out | in top 100 | 2162 | 0.4198 | 0.3132 | 0.1065 | 0.0975 to 0.1154 |
| nfcorpus | 1-1 | 307 | held-out | log rank (lower is better) | 2162 | 4.8886 | 5.3993 | -0.5107 | -0.5459 to -0.4738 |
| nfcorpus | 1-1 | 307 | held-out | worst rank in the set | 2162 | 668.7993 | 896.8044 | -228.0051 | -244.3527 to -211.4747 |
| nfcorpus | 1-1 | 307 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0922 | 0.0843 | 0.0079 | 0.0044 to 0.0112 |
| nfcorpus | 1-1 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1030 | 0.0765 | 0.0264 | 0.0230 to 0.0301 |
| nfcorpus | 1-1 | 409 | held-out | reciprocal rank | 2162 | 0.0737 | 0.0734 | 0.0003 | -0.0028 to 0.0032 |
| nfcorpus | 1-1 | 409 | held-out | in top 10 | 2162 | 0.1901 | 0.1440 | 0.0461 | 0.0400 to 0.0518 |
| nfcorpus | 1-1 | 409 | held-out | in top 100 | 2162 | 0.4181 | 0.3132 | 0.1048 | 0.0960 to 0.1136 |
| nfcorpus | 1-1 | 409 | held-out | log rank (lower is better) | 2162 | 4.8845 | 5.3993 | -0.5148 | -0.5492 to -0.4790 |
| nfcorpus | 1-1 | 409 | held-out | worst rank in the set | 2162 | 668.1379 | 896.8044 | -228.6665 | -244.9298 to -212.2639 |
| nfcorpus | 1-1 | 409 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0938 | 0.0843 | 0.0094 | 0.0062 to 0.0126 |
| nfcorpus | 1-1 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1037 | 0.0765 | 0.0272 | 0.0239 to 0.0307 |
| nfcorpus | 1-1 | 512 | held-out | reciprocal rank | 2162 | 0.0754 | 0.0734 | 0.0021 | -0.0008 to 0.0049 |
| nfcorpus | 1-1 | 512 | held-out | in top 10 | 2162 | 0.1877 | 0.1440 | 0.0437 | 0.0375 to 0.0496 |
| nfcorpus | 1-1 | 512 | held-out | in top 100 | 2162 | 0.4168 | 0.3132 | 0.1035 | 0.0947 to 0.1120 |
| nfcorpus | 1-1 | 512 | held-out | log rank (lower is better) | 2162 | 4.8886 | 5.3993 | -0.5107 | -0.5443 to -0.4758 |
| nfcorpus | 1-1 | 512 | held-out | worst rank in the set | 2162 | 668.1020 | 896.8044 | -228.7025 | -244.5456 to -212.5868 |
| nfcorpus | 1-1 | 512 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0945 | 0.0843 | 0.0102 | 0.0069 to 0.0133 |
| nfcorpus | 1-1 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1029 | 0.0765 | 0.0264 | 0.0232 to 0.0297 |
| nfcorpus | 1-1 | 614 | held-out | reciprocal rank | 2162 | 0.0768 | 0.0734 | 0.0035 | 0.0006 to 0.0060 |
| nfcorpus | 1-1 | 614 | held-out | in top 10 | 2162 | 0.1862 | 0.1440 | 0.0422 | 0.0364 to 0.0479 |
| nfcorpus | 1-1 | 614 | held-out | in top 100 | 2162 | 0.4147 | 0.3132 | 0.1014 | 0.0930 to 0.1098 |
| nfcorpus | 1-1 | 614 | held-out | log rank (lower is better) | 2162 | 4.8957 | 5.3993 | -0.5036 | -0.5365 to -0.4695 |
| nfcorpus | 1-1 | 614 | held-out | worst rank in the set | 2162 | 669.3998 | 896.8044 | -227.4046 | -243.0426 to -211.6963 |
| nfcorpus | 1-1 | 614 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0952 | 0.0843 | 0.0109 | 0.0078 to 0.0139 |
| nfcorpus | 1-1 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1021 | 0.0765 | 0.0256 | 0.0224 to 0.0289 |
| nfcorpus | 1-1 | 716 | held-out | reciprocal rank | 2162 | 0.0778 | 0.0734 | 0.0045 | 0.0019 to 0.0070 |
| nfcorpus | 1-1 | 716 | held-out | in top 10 | 2162 | 0.1835 | 0.1440 | 0.0395 | 0.0339 to 0.0448 |
| nfcorpus | 1-1 | 716 | held-out | in top 100 | 2162 | 0.4107 | 0.3132 | 0.0974 | 0.0889 to 0.1054 |
| nfcorpus | 1-1 | 716 | held-out | log rank (lower is better) | 2162 | 4.9141 | 5.3993 | -0.4851 | -0.5162 to -0.4530 |
| nfcorpus | 1-1 | 716 | held-out | worst rank in the set | 2162 | 674.2873 | 896.8044 | -222.5171 | -237.4436 to -207.2567 |
| nfcorpus | 1-1 | 716 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0954 | 0.0843 | 0.0111 | 0.0082 to 0.0140 |
| nfcorpus | 1-1 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1001 | 0.0765 | 0.0236 | 0.0207 to 0.0267 |
| nfcorpus | 1-1 | 819 | held-out | reciprocal rank | 2162 | 0.0790 | 0.0734 | 0.0057 | 0.0033 to 0.0080 |
| nfcorpus | 1-1 | 819 | held-out | in top 10 | 2162 | 0.1768 | 0.1440 | 0.0328 | 0.0279 to 0.0376 |
| nfcorpus | 1-1 | 819 | held-out | in top 100 | 2162 | 0.3989 | 0.3132 | 0.0857 | 0.0781 to 0.0930 |
| nfcorpus | 1-1 | 819 | held-out | log rank (lower is better) | 2162 | 4.9615 | 5.3993 | -0.4377 | -0.4656 to -0.4095 |
| nfcorpus | 1-1 | 819 | held-out | worst rank in the set | 2162 | 688.3534 | 896.8044 | -208.4511 | -222.0334 to -194.8095 |
| nfcorpus | 1-1 | 819 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0949 | 0.0843 | 0.0106 | 0.0081 to 0.0133 |
| nfcorpus | 1-1 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.0959 | 0.0765 | 0.0194 | 0.0167 to 0.0222 |
| nfcorpus | 1-1 | 921 | held-out | reciprocal rank | 2162 | 0.0792 | 0.0734 | 0.0058 | 0.0041 to 0.0077 |
| nfcorpus | 1-1 | 921 | held-out | in top 10 | 2162 | 0.1630 | 0.1440 | 0.0190 | 0.0151 to 0.0227 |
| nfcorpus | 1-1 | 921 | held-out | in top 100 | 2162 | 0.3711 | 0.3132 | 0.0579 | 0.0516 to 0.0641 |
| nfcorpus | 1-1 | 921 | held-out | log rank (lower is better) | 2162 | 5.0824 | 5.3993 | -0.3169 | -0.3359 to -0.2981 |
| nfcorpus | 1-1 | 921 | held-out | worst rank in the set | 2162 | 729.4013 | 896.8044 | -167.4031 | -178.2234 to -156.5633 |
| nfcorpus | 1-1 | 921 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0922 | 0.0843 | 0.0079 | 0.0060 to 0.0099 |
| nfcorpus | 1-1 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.0890 | 0.0765 | 0.0125 | 0.0106 to 0.0145 |
| nfcorpus | 2-2 | 102 | held-out | reciprocal rank | 1922 | 0.0588 | 0.0560 | 0.0027 | 0.0004 to 0.0050 |
| nfcorpus | 2-2 | 102 | held-out | in top 10 | 1922 | 0.1523 | 0.1154 | 0.0369 | 0.0313 to 0.0426 |
| nfcorpus | 2-2 | 102 | held-out | in top 100 | 1922 | 0.3955 | 0.2938 | 0.1017 | 0.0929 to 0.1099 |
| nfcorpus | 2-2 | 102 | held-out | log rank (lower is better) | 1922 | 5.0398 | 5.5554 | -0.5157 | -0.5521 to -0.4773 |
| nfcorpus | 2-2 | 102 | held-out | worst rank in the set | 1922 | 1070.8496 | 1394.9434 | -324.0938 | -346.7292 to -300.0942 |
| nfcorpus | 2-2 | 102 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0888 | 0.0786 | 0.0103 | 0.0068 to 0.0137 |
| nfcorpus | 2-2 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0803 | 0.0590 | 0.0213 | 0.0185 to 0.0242 |
| nfcorpus | 2-2 | 204 | held-out | reciprocal rank | 1922 | 0.0620 | 0.0560 | 0.0060 | 0.0038 to 0.0081 |
| nfcorpus | 2-2 | 204 | held-out | in top 10 | 1922 | 0.1609 | 0.1154 | 0.0455 | 0.0401 to 0.0510 |
| nfcorpus | 2-2 | 204 | held-out | in top 100 | 1922 | 0.4091 | 0.2938 | 0.1153 | 0.1070 to 0.1235 |
| nfcorpus | 2-2 | 204 | held-out | log rank (lower is better) | 1922 | 4.9699 | 5.5554 | -0.5856 | -0.6211 to -0.5483 |
| nfcorpus | 2-2 | 204 | held-out | worst rank in the set | 1922 | 1038.0774 | 1394.9434 | -356.8660 | -378.8057 to -334.0155 |
| nfcorpus | 2-2 | 204 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0942 | 0.0786 | 0.0156 | 0.0122 to 0.0189 |
| nfcorpus | 2-2 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0839 | 0.0590 | 0.0249 | 0.0222 to 0.0276 |
| nfcorpus | 2-2 | 307 | held-out | reciprocal rank | 1922 | 0.0637 | 0.0560 | 0.0076 | 0.0056 to 0.0097 |
| nfcorpus | 2-2 | 307 | held-out | in top 10 | 1922 | 0.1607 | 0.1154 | 0.0454 | 0.0400 to 0.0507 |
| nfcorpus | 2-2 | 307 | held-out | in top 100 | 1922 | 0.4108 | 0.2938 | 0.1170 | 0.1088 to 0.1249 |
| nfcorpus | 2-2 | 307 | held-out | log rank (lower is better) | 1922 | 4.9562 | 5.5554 | -0.5992 | -0.6333 to -0.5631 |
| nfcorpus | 2-2 | 307 | held-out | worst rank in the set | 1922 | 1028.7894 | 1394.9434 | -366.1540 | -387.2588 to -344.2130 |
| nfcorpus | 2-2 | 307 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0957 | 0.0786 | 0.0171 | 0.0140 to 0.0203 |
| nfcorpus | 2-2 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0843 | 0.0590 | 0.0253 | 0.0227 to 0.0279 |
| nfcorpus | 2-2 | 409 | held-out | reciprocal rank | 1922 | 0.0646 | 0.0560 | 0.0086 | 0.0066 to 0.0106 |
| nfcorpus | 2-2 | 409 | held-out | in top 10 | 1922 | 0.1617 | 0.1154 | 0.0463 | 0.0411 to 0.0514 |
| nfcorpus | 2-2 | 409 | held-out | in top 100 | 1922 | 0.4102 | 0.2938 | 0.1163 | 0.1084 to 0.1241 |
| nfcorpus | 2-2 | 409 | held-out | log rank (lower is better) | 1922 | 4.9601 | 5.5554 | -0.5954 | -0.6284 to -0.5608 |
| nfcorpus | 2-2 | 409 | held-out | worst rank in the set | 1922 | 1031.2851 | 1394.9434 | -363.6583 | -385.1339 to -341.8709 |
| nfcorpus | 2-2 | 409 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0970 | 0.0786 | 0.0185 | 0.0155 to 0.0215 |
| nfcorpus | 2-2 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0838 | 0.0590 | 0.0248 | 0.0224 to 0.0274 |
| nfcorpus | 2-2 | 512 | held-out | reciprocal rank | 1922 | 0.0654 | 0.0560 | 0.0093 | 0.0074 to 0.0113 |
| nfcorpus | 2-2 | 512 | held-out | in top 10 | 1922 | 0.1594 | 0.1154 | 0.0440 | 0.0390 to 0.0489 |
| nfcorpus | 2-2 | 512 | held-out | in top 100 | 1922 | 0.4070 | 0.2938 | 0.1131 | 0.1053 to 0.1209 |
| nfcorpus | 2-2 | 512 | held-out | log rank (lower is better) | 1922 | 4.9713 | 5.5554 | -0.5842 | -0.6154 to -0.5506 |
| nfcorpus | 2-2 | 512 | held-out | worst rank in the set | 1922 | 1035.6610 | 1394.9434 | -359.2824 | -380.4060 to -337.8939 |
| nfcorpus | 2-2 | 512 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0971 | 0.0786 | 0.0185 | 0.0156 to 0.0214 |
| nfcorpus | 2-2 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0827 | 0.0590 | 0.0237 | 0.0213 to 0.0261 |
| nfcorpus | 2-2 | 614 | held-out | reciprocal rank | 1922 | 0.0658 | 0.0560 | 0.0098 | 0.0080 to 0.0117 |
| nfcorpus | 2-2 | 614 | held-out | in top 10 | 1922 | 0.1574 | 0.1154 | 0.0420 | 0.0373 to 0.0466 |
| nfcorpus | 2-2 | 614 | held-out | in top 100 | 1922 | 0.4025 | 0.2938 | 0.1086 | 0.1009 to 0.1162 |
| nfcorpus | 2-2 | 614 | held-out | log rank (lower is better) | 1922 | 4.9858 | 5.5554 | -0.5697 | -0.6003 to -0.5372 |
| nfcorpus | 2-2 | 614 | held-out | worst rank in the set | 1922 | 1041.7813 | 1394.9434 | -353.1621 | -373.8173 to -332.6064 |
| nfcorpus | 2-2 | 614 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0970 | 0.0786 | 0.0184 | 0.0157 to 0.0212 |
| nfcorpus | 2-2 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0816 | 0.0590 | 0.0226 | 0.0203 to 0.0251 |
| nfcorpus | 2-2 | 716 | held-out | reciprocal rank | 1922 | 0.0658 | 0.0560 | 0.0097 | 0.0080 to 0.0115 |
| nfcorpus | 2-2 | 716 | held-out | in top 10 | 1922 | 0.1536 | 0.1154 | 0.0383 | 0.0339 to 0.0427 |
| nfcorpus | 2-2 | 716 | held-out | in top 100 | 1922 | 0.3964 | 0.2938 | 0.1026 | 0.0952 to 0.1098 |
| nfcorpus | 2-2 | 716 | held-out | log rank (lower is better) | 1922 | 5.0106 | 5.5554 | -0.5448 | -0.5736 to -0.5142 |
| nfcorpus | 2-2 | 716 | held-out | worst rank in the set | 1922 | 1051.3907 | 1394.9434 | -343.5527 | -363.7278 to -323.5866 |
| nfcorpus | 2-2 | 716 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0960 | 0.0786 | 0.0174 | 0.0148 to 0.0201 |
| nfcorpus | 2-2 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0800 | 0.0590 | 0.0210 | 0.0189 to 0.0232 |
| nfcorpus | 2-2 | 819 | held-out | reciprocal rank | 1922 | 0.0647 | 0.0560 | 0.0086 | 0.0071 to 0.0104 |
| nfcorpus | 2-2 | 819 | held-out | in top 10 | 1922 | 0.1464 | 0.1154 | 0.0311 | 0.0273 to 0.0350 |
| nfcorpus | 2-2 | 819 | held-out | in top 100 | 1922 | 0.3825 | 0.2938 | 0.0887 | 0.0822 to 0.0953 |
| nfcorpus | 2-2 | 819 | held-out | log rank (lower is better) | 1922 | 5.0733 | 5.5554 | -0.4822 | -0.5068 to -0.4553 |
| nfcorpus | 2-2 | 819 | held-out | worst rank in the set | 1922 | 1077.1667 | 1394.9434 | -317.7767 | -336.3629 to -299.6646 |
| nfcorpus | 2-2 | 819 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0932 | 0.0786 | 0.0147 | 0.0123 to 0.0171 |
| nfcorpus | 2-2 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0759 | 0.0590 | 0.0168 | 0.0150 to 0.0188 |
| nfcorpus | 2-2 | 921 | held-out | reciprocal rank | 1922 | 0.0628 | 0.0560 | 0.0067 | 0.0055 to 0.0080 |
| nfcorpus | 2-2 | 921 | held-out | in top 10 | 1922 | 0.1352 | 0.1154 | 0.0198 | 0.0169 to 0.0229 |
| nfcorpus | 2-2 | 921 | held-out | in top 100 | 1922 | 0.3539 | 0.2938 | 0.0601 | 0.0549 to 0.0655 |
| nfcorpus | 2-2 | 921 | held-out | log rank (lower is better) | 1922 | 5.2067 | 5.5554 | -0.3488 | -0.3662 to -0.3304 |
| nfcorpus | 2-2 | 921 | held-out | worst rank in the set | 1922 | 1142.7696 | 1394.9434 | -252.1738 | -266.7550 to -237.6841 |
| nfcorpus | 2-2 | 921 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0890 | 0.0786 | 0.0105 | 0.0087 to 0.0123 |
| nfcorpus | 2-2 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0701 | 0.0590 | 0.0111 | 0.0098 to 0.0125 |
| nfcorpus | 5-5 | 102 | held-out | reciprocal rank | 1517 | 0.0505 | 0.0458 | 0.0048 | 0.0032 to 0.0064 |
| nfcorpus | 5-5 | 102 | held-out | in top 10 | 1517 | 0.1229 | 0.1010 | 0.0219 | 0.0180 to 0.0261 |
| nfcorpus | 5-5 | 102 | held-out | in top 100 | 1517 | 0.3635 | 0.2762 | 0.0873 | 0.0802 to 0.0945 |
| nfcorpus | 5-5 | 102 | held-out | log rank (lower is better) | 1517 | 5.2090 | 5.6726 | -0.4637 | -0.4989 to -0.4309 |
| nfcorpus | 5-5 | 102 | held-out | worst rank in the set | 1517 | 1732.0631 | 2119.1761 | -387.1130 | -413.2856 to -360.7543 |
| nfcorpus | 5-5 | 102 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1007 | 0.0895 | 0.0111 | 0.0079 to 0.0145 |
| nfcorpus | 5-5 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0632 | 0.0495 | 0.0137 | 0.0118 to 0.0157 |
| nfcorpus | 5-5 | 204 | held-out | reciprocal rank | 1517 | 0.0529 | 0.0458 | 0.0071 | 0.0056 to 0.0086 |
| nfcorpus | 5-5 | 204 | held-out | in top 10 | 1517 | 0.1294 | 0.1010 | 0.0284 | 0.0244 to 0.0327 |
| nfcorpus | 5-5 | 204 | held-out | in top 100 | 1517 | 0.3781 | 0.2762 | 0.1019 | 0.0947 to 0.1095 |
| nfcorpus | 5-5 | 204 | held-out | log rank (lower is better) | 1517 | 5.1363 | 5.6726 | -0.5363 | -0.5707 to -0.5041 |
| nfcorpus | 5-5 | 204 | held-out | worst rank in the set | 1517 | 1684.1957 | 2119.1761 | -434.9804 | -460.1005 to -409.6080 |
| nfcorpus | 5-5 | 204 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1058 | 0.0895 | 0.0163 | 0.0130 to 0.0197 |
| nfcorpus | 5-5 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0662 | 0.0495 | 0.0166 | 0.0147 to 0.0187 |
| nfcorpus | 5-5 | 307 | held-out | reciprocal rank | 1517 | 0.0538 | 0.0458 | 0.0081 | 0.0067 to 0.0095 |
| nfcorpus | 5-5 | 307 | held-out | in top 10 | 1517 | 0.1306 | 0.1010 | 0.0296 | 0.0256 to 0.0337 |
| nfcorpus | 5-5 | 307 | held-out | in top 100 | 1517 | 0.3790 | 0.2762 | 0.1028 | 0.0958 to 0.1105 |
| nfcorpus | 5-5 | 307 | held-out | log rank (lower is better) | 1517 | 5.1261 | 5.6726 | -0.5465 | -0.5798 to -0.5147 |
| nfcorpus | 5-5 | 307 | held-out | worst rank in the set | 1517 | 1674.1786 | 2119.1761 | -444.9974 | -469.7388 to -420.5814 |
| nfcorpus | 5-5 | 307 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1076 | 0.0895 | 0.0181 | 0.0149 to 0.0214 |
| nfcorpus | 5-5 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0666 | 0.0495 | 0.0171 | 0.0152 to 0.0191 |
| nfcorpus | 5-5 | 409 | held-out | reciprocal rank | 1517 | 0.0539 | 0.0458 | 0.0081 | 0.0068 to 0.0096 |
| nfcorpus | 5-5 | 409 | held-out | in top 10 | 1517 | 0.1299 | 0.1010 | 0.0289 | 0.0250 to 0.0327 |
| nfcorpus | 5-5 | 409 | held-out | in top 100 | 1517 | 0.3779 | 0.2762 | 0.1016 | 0.0948 to 0.1092 |
| nfcorpus | 5-5 | 409 | held-out | log rank (lower is better) | 1517 | 5.1340 | 5.6726 | -0.5386 | -0.5711 to -0.5076 |
| nfcorpus | 5-5 | 409 | held-out | worst rank in the set | 1517 | 1678.5127 | 2119.1761 | -440.6634 | -464.7442 to -416.7647 |
| nfcorpus | 5-5 | 409 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1076 | 0.0895 | 0.0180 | 0.0150 to 0.0211 |
| nfcorpus | 5-5 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0661 | 0.0495 | 0.0166 | 0.0148 to 0.0185 |
| nfcorpus | 5-5 | 512 | held-out | reciprocal rank | 1517 | 0.0540 | 0.0458 | 0.0082 | 0.0069 to 0.0095 |
| nfcorpus | 5-5 | 512 | held-out | in top 10 | 1517 | 0.1292 | 0.1010 | 0.0282 | 0.0246 to 0.0320 |
| nfcorpus | 5-5 | 512 | held-out | in top 100 | 1517 | 0.3721 | 0.2762 | 0.0959 | 0.0893 to 0.1032 |
| nfcorpus | 5-5 | 512 | held-out | log rank (lower is better) | 1517 | 5.1488 | 5.6726 | -0.5238 | -0.5548 to -0.4946 |
| nfcorpus | 5-5 | 512 | held-out | worst rank in the set | 1517 | 1687.6811 | 2119.1761 | -431.4950 | -455.3637 to -408.2626 |
| nfcorpus | 5-5 | 512 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1076 | 0.0895 | 0.0181 | 0.0153 to 0.0211 |
| nfcorpus | 5-5 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0653 | 0.0495 | 0.0158 | 0.0140 to 0.0176 |
| nfcorpus | 5-5 | 614 | held-out | reciprocal rank | 1517 | 0.0540 | 0.0458 | 0.0082 | 0.0069 to 0.0095 |
| nfcorpus | 5-5 | 614 | held-out | in top 10 | 1517 | 0.1279 | 0.1010 | 0.0269 | 0.0232 to 0.0304 |
| nfcorpus | 5-5 | 614 | held-out | in top 100 | 1517 | 0.3688 | 0.2762 | 0.0925 | 0.0862 to 0.0998 |
| nfcorpus | 5-5 | 614 | held-out | log rank (lower is better) | 1517 | 5.1656 | 5.6726 | -0.5070 | -0.5374 to -0.4785 |
| nfcorpus | 5-5 | 614 | held-out | worst rank in the set | 1517 | 1698.5177 | 2119.1761 | -420.6584 | -443.8769 to -398.3174 |
| nfcorpus | 5-5 | 614 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1072 | 0.0895 | 0.0177 | 0.0149 to 0.0206 |
| nfcorpus | 5-5 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0647 | 0.0495 | 0.0151 | 0.0135 to 0.0169 |
| nfcorpus | 5-5 | 716 | held-out | reciprocal rank | 1517 | 0.0535 | 0.0458 | 0.0078 | 0.0066 to 0.0090 |
| nfcorpus | 5-5 | 716 | held-out | in top 10 | 1517 | 0.1261 | 0.1010 | 0.0251 | 0.0216 to 0.0285 |
| nfcorpus | 5-5 | 716 | held-out | in top 100 | 1517 | 0.3647 | 0.2762 | 0.0884 | 0.0823 to 0.0954 |
| nfcorpus | 5-5 | 716 | held-out | log rank (lower is better) | 1517 | 5.1889 | 5.6726 | -0.4837 | -0.5121 to -0.4568 |
| nfcorpus | 5-5 | 716 | held-out | worst rank in the set | 1517 | 1714.4782 | 2119.1761 | -404.6978 | -426.9317 to -383.1534 |
| nfcorpus | 5-5 | 716 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1062 | 0.0895 | 0.0167 | 0.0140 to 0.0194 |
| nfcorpus | 5-5 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0635 | 0.0495 | 0.0140 | 0.0125 to 0.0157 |
| nfcorpus | 5-5 | 819 | held-out | reciprocal rank | 1517 | 0.0525 | 0.0458 | 0.0068 | 0.0057 to 0.0078 |
| nfcorpus | 5-5 | 819 | held-out | in top 10 | 1517 | 0.1214 | 0.1010 | 0.0204 | 0.0175 to 0.0232 |
| nfcorpus | 5-5 | 819 | held-out | in top 100 | 1517 | 0.3525 | 0.2762 | 0.0762 | 0.0706 to 0.0824 |
| nfcorpus | 5-5 | 819 | held-out | log rank (lower is better) | 1517 | 5.2454 | 5.6726 | -0.4273 | -0.4514 to -0.4038 |
| nfcorpus | 5-5 | 819 | held-out | worst rank in the set | 1517 | 1751.2251 | 2119.1761 | -367.9510 | -388.2186 to -348.9217 |
| nfcorpus | 5-5 | 819 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1034 | 0.0895 | 0.0139 | 0.0117 to 0.0162 |
| nfcorpus | 5-5 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0610 | 0.0495 | 0.0115 | 0.0101 to 0.0129 |
| nfcorpus | 5-5 | 921 | held-out | reciprocal rank | 1517 | 0.0503 | 0.0458 | 0.0046 | 0.0038 to 0.0054 |
| nfcorpus | 5-5 | 921 | held-out | in top 10 | 1517 | 0.1145 | 0.1010 | 0.0135 | 0.0114 to 0.0157 |
| nfcorpus | 5-5 | 921 | held-out | in top 100 | 1517 | 0.3273 | 0.2762 | 0.0510 | 0.0467 to 0.0555 |
| nfcorpus | 5-5 | 921 | held-out | log rank (lower is better) | 1517 | 5.3678 | 5.6726 | -0.3048 | -0.3218 to -0.2886 |
| nfcorpus | 5-5 | 921 | held-out | worst rank in the set | 1517 | 1838.9617 | 2119.1761 | -280.2144 | -295.6766 to -266.4304 |
| nfcorpus | 5-5 | 921 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.0989 | 0.0895 | 0.0094 | 0.0077 to 0.0111 |
| nfcorpus | 5-5 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0568 | 0.0495 | 0.0072 | 0.0063 to 0.0082 |
| nfcorpus | 10-10 | 102 | held-out | reciprocal rank | 1105 | 0.0387 | 0.0375 | 0.0012 | 0.0002 to 0.0023 |
| nfcorpus | 10-10 | 102 | held-out | in top 10 | 1105 | 0.0896 | 0.0844 | 0.0053 | 0.0028 to 0.0080 |
| nfcorpus | 10-10 | 102 | held-out | in top 100 | 1105 | 0.3108 | 0.2544 | 0.0564 | 0.0502 to 0.0627 |
| nfcorpus | 10-10 | 102 | held-out | log rank (lower is better) | 1105 | 5.4751 | 5.7993 | -0.3242 | -0.3526 to -0.2965 |
| nfcorpus | 10-10 | 102 | held-out | worst rank in the set | 1105 | 2350.2149 | 2646.2550 | -296.0400 | -319.8164 to -272.2182 |
| nfcorpus | 10-10 | 102 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.0954 | 0.0940 | 0.0014 | -0.0014 to 0.0043 |
| nfcorpus | 10-10 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0476 | 0.0422 | 0.0055 | 0.0041 to 0.0069 |
| nfcorpus | 10-10 | 204 | held-out | reciprocal rank | 1105 | 0.0402 | 0.0375 | 0.0027 | 0.0018 to 0.0037 |
| nfcorpus | 10-10 | 204 | held-out | in top 10 | 1105 | 0.0941 | 0.0844 | 0.0097 | 0.0072 to 0.0122 |
| nfcorpus | 10-10 | 204 | held-out | in top 100 | 1105 | 0.3243 | 0.2544 | 0.0699 | 0.0634 to 0.0764 |
| nfcorpus | 10-10 | 204 | held-out | log rank (lower is better) | 1105 | 5.4119 | 5.7993 | -0.3874 | -0.4157 to -0.3601 |
| nfcorpus | 10-10 | 204 | held-out | worst rank in the set | 1105 | 2308.3803 | 2646.2550 | -337.8747 | -360.7227 to -314.6835 |
| nfcorpus | 10-10 | 204 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.0995 | 0.0940 | 0.0055 | 0.0030 to 0.0081 |
| nfcorpus | 10-10 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0496 | 0.0422 | 0.0074 | 0.0061 to 0.0087 |
| nfcorpus | 10-10 | 307 | held-out | reciprocal rank | 1105 | 0.0408 | 0.0375 | 0.0033 | 0.0024 to 0.0042 |
| nfcorpus | 10-10 | 307 | held-out | in top 10 | 1105 | 0.0955 | 0.0844 | 0.0112 | 0.0088 to 0.0136 |
| nfcorpus | 10-10 | 307 | held-out | in top 100 | 1105 | 0.3250 | 0.2544 | 0.0706 | 0.0641 to 0.0769 |
| nfcorpus | 10-10 | 307 | held-out | log rank (lower is better) | 1105 | 5.3999 | 5.7993 | -0.3994 | -0.4268 to -0.3735 |
| nfcorpus | 10-10 | 307 | held-out | worst rank in the set | 1105 | 2304.0484 | 2646.2550 | -342.2066 | -364.5720 to -319.9498 |
| nfcorpus | 10-10 | 307 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1012 | 0.0940 | 0.0072 | 0.0048 to 0.0097 |
| nfcorpus | 10-10 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0500 | 0.0422 | 0.0079 | 0.0067 to 0.0090 |
| nfcorpus | 10-10 | 409 | held-out | reciprocal rank | 1105 | 0.0407 | 0.0375 | 0.0032 | 0.0024 to 0.0041 |
| nfcorpus | 10-10 | 409 | held-out | in top 10 | 1105 | 0.0957 | 0.0844 | 0.0114 | 0.0092 to 0.0136 |
| nfcorpus | 10-10 | 409 | held-out | in top 100 | 1105 | 0.3234 | 0.2544 | 0.0690 | 0.0627 to 0.0752 |
| nfcorpus | 10-10 | 409 | held-out | log rank (lower is better) | 1105 | 5.4056 | 5.7993 | -0.3937 | -0.4202 to -0.3685 |
| nfcorpus | 10-10 | 409 | held-out | worst rank in the set | 1105 | 2307.8843 | 2646.2550 | -338.3707 | -359.9289 to -317.7007 |
| nfcorpus | 10-10 | 409 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1013 | 0.0940 | 0.0073 | 0.0051 to 0.0096 |
| nfcorpus | 10-10 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0497 | 0.0422 | 0.0076 | 0.0065 to 0.0087 |
| nfcorpus | 10-10 | 512 | held-out | reciprocal rank | 1105 | 0.0408 | 0.0375 | 0.0033 | 0.0025 to 0.0042 |
| nfcorpus | 10-10 | 512 | held-out | in top 10 | 1105 | 0.0952 | 0.0844 | 0.0108 | 0.0087 to 0.0131 |
| nfcorpus | 10-10 | 512 | held-out | in top 100 | 1105 | 0.3211 | 0.2544 | 0.0667 | 0.0606 to 0.0728 |
| nfcorpus | 10-10 | 512 | held-out | log rank (lower is better) | 1105 | 5.4186 | 5.7993 | -0.3807 | -0.4061 to -0.3565 |
| nfcorpus | 10-10 | 512 | held-out | worst rank in the set | 1105 | 2316.9550 | 2646.2550 | -329.3000 | -349.7602 to -309.5462 |
| nfcorpus | 10-10 | 512 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1013 | 0.0940 | 0.0073 | 0.0052 to 0.0095 |
| nfcorpus | 10-10 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0494 | 0.0422 | 0.0073 | 0.0061 to 0.0084 |
| nfcorpus | 10-10 | 614 | held-out | reciprocal rank | 1105 | 0.0407 | 0.0375 | 0.0033 | 0.0025 to 0.0041 |
| nfcorpus | 10-10 | 614 | held-out | in top 10 | 1105 | 0.0949 | 0.0844 | 0.0105 | 0.0085 to 0.0126 |
| nfcorpus | 10-10 | 614 | held-out | in top 100 | 1105 | 0.3181 | 0.2544 | 0.0637 | 0.0579 to 0.0696 |
| nfcorpus | 10-10 | 614 | held-out | log rank (lower is better) | 1105 | 5.4335 | 5.7993 | -0.3658 | -0.3900 to -0.3422 |
| nfcorpus | 10-10 | 614 | held-out | worst rank in the set | 1105 | 2327.3859 | 2646.2550 | -318.8691 | -338.5646 to -299.3166 |
| nfcorpus | 10-10 | 614 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1013 | 0.0940 | 0.0073 | 0.0053 to 0.0094 |
| nfcorpus | 10-10 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0491 | 0.0422 | 0.0069 | 0.0059 to 0.0080 |
| nfcorpus | 10-10 | 716 | held-out | reciprocal rank | 1105 | 0.0406 | 0.0375 | 0.0032 | 0.0024 to 0.0039 |
| nfcorpus | 10-10 | 716 | held-out | in top 10 | 1105 | 0.0945 | 0.0844 | 0.0102 | 0.0082 to 0.0122 |
| nfcorpus | 10-10 | 716 | held-out | in top 100 | 1105 | 0.3149 | 0.2544 | 0.0605 | 0.0550 to 0.0662 |
| nfcorpus | 10-10 | 716 | held-out | log rank (lower is better) | 1105 | 5.4500 | 5.7993 | -0.3493 | -0.3722 to -0.3269 |
| nfcorpus | 10-10 | 716 | held-out | worst rank in the set | 1105 | 2339.7557 | 2646.2550 | -306.4993 | -325.3902 to -287.8695 |
| nfcorpus | 10-10 | 716 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1012 | 0.0940 | 0.0072 | 0.0053 to 0.0092 |
| nfcorpus | 10-10 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0487 | 0.0422 | 0.0065 | 0.0055 to 0.0075 |
| nfcorpus | 10-10 | 819 | held-out | reciprocal rank | 1105 | 0.0404 | 0.0375 | 0.0029 | 0.0023 to 0.0036 |
| nfcorpus | 10-10 | 819 | held-out | in top 10 | 1105 | 0.0931 | 0.0844 | 0.0087 | 0.0069 to 0.0105 |
| nfcorpus | 10-10 | 819 | held-out | in top 100 | 1105 | 0.3072 | 0.2544 | 0.0528 | 0.0480 to 0.0579 |
| nfcorpus | 10-10 | 819 | held-out | log rank (lower is better) | 1105 | 5.4878 | 5.7993 | -0.3115 | -0.3315 to -0.2919 |
| nfcorpus | 10-10 | 819 | held-out | worst rank in the set | 1105 | 2368.8078 | 2646.2550 | -277.4472 | -294.4460 to -260.6505 |
| nfcorpus | 10-10 | 819 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1005 | 0.0940 | 0.0065 | 0.0047 to 0.0083 |
| nfcorpus | 10-10 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0478 | 0.0422 | 0.0057 | 0.0048 to 0.0065 |
| nfcorpus | 10-10 | 921 | held-out | reciprocal rank | 1105 | 0.0396 | 0.0375 | 0.0021 | 0.0016 to 0.0026 |
| nfcorpus | 10-10 | 921 | held-out | in top 10 | 1105 | 0.0905 | 0.0844 | 0.0061 | 0.0046 to 0.0075 |
| nfcorpus | 10-10 | 921 | held-out | in top 100 | 1105 | 0.2911 | 0.2544 | 0.0367 | 0.0329 to 0.0406 |
| nfcorpus | 10-10 | 921 | held-out | log rank (lower is better) | 1105 | 5.5733 | 5.7993 | -0.2260 | -0.2399 to -0.2118 |
| nfcorpus | 10-10 | 921 | held-out | worst rank in the set | 1105 | 2438.4052 | 2646.2550 | -207.8498 | -220.9125 to -194.5683 |
| nfcorpus | 10-10 | 921 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.0987 | 0.0940 | 0.0046 | 0.0033 to 0.0061 |
| nfcorpus | 10-10 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0460 | 0.0422 | 0.0039 | 0.0033 to 0.0045 |
| nfcorpus | 20-20 | 102 | held-out | reciprocal rank | 611 | 0.0297 | 0.0292 | 0.0005 | -0.0003 to 0.0012 |
| nfcorpus | 20-20 | 102 | held-out | in top 10 | 611 | 0.0655 | 0.0656 | -0.0000 | -0.0021 to 0.0020 |
| nfcorpus | 20-20 | 102 | held-out | in top 100 | 611 | 0.2595 | 0.2317 | 0.0278 | 0.0225 to 0.0333 |
| nfcorpus | 20-20 | 102 | held-out | log rank (lower is better) | 611 | 5.7292 | 5.9328 | -0.2036 | -0.2296 to -0.1793 |
| nfcorpus | 20-20 | 102 | held-out | worst rank in the set | 611 | 2874.4106 | 3056.7717 | -182.3611 | -205.0844 to -159.5518 |
| nfcorpus | 20-20 | 102 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1401 | 0.1419 | -0.0019 | -0.0058 to 0.0023 |
| nfcorpus | 20-20 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0368 | 0.0345 | 0.0023 | 0.0014 to 0.0032 |
| nfcorpus | 20-20 | 204 | held-out | reciprocal rank | 611 | 0.0309 | 0.0292 | 0.0017 | 0.0010 to 0.0023 |
| nfcorpus | 20-20 | 204 | held-out | in top 10 | 611 | 0.0689 | 0.0656 | 0.0033 | 0.0015 to 0.0052 |
| nfcorpus | 20-20 | 204 | held-out | in top 100 | 611 | 0.2725 | 0.2317 | 0.0408 | 0.0357 to 0.0461 |
| nfcorpus | 20-20 | 204 | held-out | log rank (lower is better) | 611 | 5.6702 | 5.9328 | -0.2626 | -0.2872 to -0.2391 |
| nfcorpus | 20-20 | 204 | held-out | worst rank in the set | 611 | 2838.5681 | 3056.7717 | -218.2036 | -241.4368 to -196.3435 |
| nfcorpus | 20-20 | 204 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1462 | 0.1419 | 0.0043 | 0.0007 to 0.0081 |
| nfcorpus | 20-20 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0384 | 0.0345 | 0.0039 | 0.0031 to 0.0047 |
| nfcorpus | 20-20 | 307 | held-out | reciprocal rank | 611 | 0.0312 | 0.0292 | 0.0020 | 0.0014 to 0.0027 |
| nfcorpus | 20-20 | 307 | held-out | in top 10 | 611 | 0.0691 | 0.0656 | 0.0035 | 0.0019 to 0.0053 |
| nfcorpus | 20-20 | 307 | held-out | in top 100 | 611 | 0.2738 | 0.2317 | 0.0421 | 0.0371 to 0.0471 |
| nfcorpus | 20-20 | 307 | held-out | log rank (lower is better) | 611 | 5.6610 | 5.9328 | -0.2719 | -0.2958 to -0.2490 |
| nfcorpus | 20-20 | 307 | held-out | worst rank in the set | 611 | 2838.0908 | 3056.7717 | -218.6809 | -241.0726 to -197.8286 |
| nfcorpus | 20-20 | 307 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1474 | 0.1419 | 0.0054 | 0.0021 to 0.0089 |
| nfcorpus | 20-20 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0387 | 0.0345 | 0.0042 | 0.0034 to 0.0050 |
| nfcorpus | 20-20 | 409 | held-out | reciprocal rank | 611 | 0.0313 | 0.0292 | 0.0020 | 0.0015 to 0.0027 |
| nfcorpus | 20-20 | 409 | held-out | in top 10 | 611 | 0.0692 | 0.0656 | 0.0036 | 0.0020 to 0.0053 |
| nfcorpus | 20-20 | 409 | held-out | in top 100 | 611 | 0.2730 | 0.2317 | 0.0413 | 0.0367 to 0.0462 |
| nfcorpus | 20-20 | 409 | held-out | log rank (lower is better) | 611 | 5.6621 | 5.9328 | -0.2707 | -0.2931 to -0.2491 |
| nfcorpus | 20-20 | 409 | held-out | worst rank in the set | 611 | 2836.5260 | 3056.7717 | -220.2457 | -241.8675 to -200.0391 |
| nfcorpus | 20-20 | 409 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1477 | 0.1419 | 0.0057 | 0.0027 to 0.0090 |
| nfcorpus | 20-20 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0387 | 0.0345 | 0.0042 | 0.0035 to 0.0050 |
| nfcorpus | 20-20 | 512 | held-out | reciprocal rank | 611 | 0.0313 | 0.0292 | 0.0021 | 0.0016 to 0.0027 |
| nfcorpus | 20-20 | 512 | held-out | in top 10 | 611 | 0.0692 | 0.0656 | 0.0036 | 0.0021 to 0.0051 |
| nfcorpus | 20-20 | 512 | held-out | in top 100 | 611 | 0.2706 | 0.2317 | 0.0389 | 0.0346 to 0.0434 |
| nfcorpus | 20-20 | 512 | held-out | log rank (lower is better) | 611 | 5.6707 | 5.9328 | -0.2622 | -0.2833 to -0.2418 |
| nfcorpus | 20-20 | 512 | held-out | worst rank in the set | 611 | 2840.9239 | 3056.7717 | -215.8478 | -236.5524 to -195.9241 |
| nfcorpus | 20-20 | 512 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1481 | 0.1419 | 0.0061 | 0.0032 to 0.0092 |
| nfcorpus | 20-20 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0386 | 0.0345 | 0.0041 | 0.0035 to 0.0048 |
| nfcorpus | 20-20 | 614 | held-out | reciprocal rank | 611 | 0.0312 | 0.0292 | 0.0020 | 0.0015 to 0.0025 |
| nfcorpus | 20-20 | 614 | held-out | in top 10 | 611 | 0.0689 | 0.0656 | 0.0033 | 0.0018 to 0.0047 |
| nfcorpus | 20-20 | 614 | held-out | in top 100 | 611 | 0.2686 | 0.2317 | 0.0369 | 0.0329 to 0.0412 |
| nfcorpus | 20-20 | 614 | held-out | log rank (lower is better) | 611 | 5.6818 | 5.9328 | -0.2510 | -0.2715 to -0.2318 |
| nfcorpus | 20-20 | 614 | held-out | worst rank in the set | 611 | 2849.8343 | 3056.7717 | -206.9374 | -226.4124 to -188.3536 |
| nfcorpus | 20-20 | 614 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1476 | 0.1419 | 0.0057 | 0.0030 to 0.0086 |
| nfcorpus | 20-20 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0384 | 0.0345 | 0.0039 | 0.0033 to 0.0046 |
| nfcorpus | 20-20 | 716 | held-out | reciprocal rank | 611 | 0.0312 | 0.0292 | 0.0020 | 0.0015 to 0.0025 |
| nfcorpus | 20-20 | 716 | held-out | in top 10 | 611 | 0.0690 | 0.0656 | 0.0034 | 0.0020 to 0.0048 |
| nfcorpus | 20-20 | 716 | held-out | in top 100 | 611 | 0.2668 | 0.2317 | 0.0351 | 0.0312 to 0.0392 |
| nfcorpus | 20-20 | 716 | held-out | log rank (lower is better) | 611 | 5.6938 | 5.9328 | -0.2390 | -0.2581 to -0.2209 |
| nfcorpus | 20-20 | 716 | held-out | worst rank in the set | 611 | 2858.5818 | 3056.7717 | -198.1899 | -217.4809 to -180.3735 |
| nfcorpus | 20-20 | 716 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1479 | 0.1419 | 0.0060 | 0.0035 to 0.0088 |
| nfcorpus | 20-20 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0382 | 0.0345 | 0.0037 | 0.0031 to 0.0043 |
| nfcorpus | 20-20 | 819 | held-out | reciprocal rank | 611 | 0.0310 | 0.0292 | 0.0017 | 0.0013 to 0.0022 |
| nfcorpus | 20-20 | 819 | held-out | in top 10 | 611 | 0.0684 | 0.0656 | 0.0028 | 0.0016 to 0.0040 |
| nfcorpus | 20-20 | 819 | held-out | in top 100 | 611 | 0.2618 | 0.2317 | 0.0301 | 0.0266 to 0.0337 |
| nfcorpus | 20-20 | 819 | held-out | log rank (lower is better) | 611 | 5.7199 | 5.9328 | -0.2130 | -0.2294 to -0.1973 |
| nfcorpus | 20-20 | 819 | held-out | worst rank in the set | 611 | 2879.5910 | 3056.7717 | -177.1806 | -194.3353 to -160.7865 |
| nfcorpus | 20-20 | 819 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1470 | 0.1419 | 0.0051 | 0.0028 to 0.0076 |
| nfcorpus | 20-20 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0377 | 0.0345 | 0.0032 | 0.0027 to 0.0038 |
| nfcorpus | 20-20 | 921 | held-out | reciprocal rank | 611 | 0.0305 | 0.0292 | 0.0013 | 0.0010 to 0.0016 |
| nfcorpus | 20-20 | 921 | held-out | in top 10 | 611 | 0.0680 | 0.0656 | 0.0024 | 0.0014 to 0.0035 |
| nfcorpus | 20-20 | 921 | held-out | in top 100 | 611 | 0.2525 | 0.2317 | 0.0208 | 0.0182 to 0.0236 |
| nfcorpus | 20-20 | 921 | held-out | log rank (lower is better) | 611 | 5.7781 | 5.9328 | -0.1548 | -0.1662 to -0.1438 |
| nfcorpus | 20-20 | 921 | held-out | worst rank in the set | 611 | 2926.6964 | 3056.7717 | -130.0753 | -142.5747 to -118.3253 |
| nfcorpus | 20-20 | 921 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1462 | 0.1419 | 0.0043 | 0.0026 to 0.0063 |
| nfcorpus | 20-20 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0368 | 0.0345 | 0.0023 | 0.0020 to 0.0027 |
| msmarco | prefix | 102 | all known positives | reciprocal rank | 2321 | 0.1248 | 0.1850 | -0.0603 | -0.0667 to -0.0540 |
| msmarco | prefix | 102 | all known positives | in top 10 | 2321 | 0.2608 | 0.3916 | -0.1309 | -0.1432 to -0.1186 |
| msmarco | prefix | 102 | all known positives | in top 100 | 2321 | 0.5831 | 0.7462 | -0.1631 | -0.1754 to -0.1507 |
| msmarco | prefix | 102 | all known positives | log rank (lower is better) | 2321 | 4.2212 | 3.2084 | 1.0128 | 0.9665 to 1.0616 |
| msmarco | prefix | 102 | all known positives | worst rank in the set | 2321 | 9419.5894 | 2545.2809 | 6874.3085 | 3646.9294 to 10750.2565 |
| msmarco | prefix | 204 | all known positives | reciprocal rank | 2321 | 0.1520 | 0.1850 | -0.0331 | -0.0387 to -0.0278 |
| msmarco | prefix | 204 | all known positives | in top 10 | 2321 | 0.3203 | 0.3916 | -0.0713 | -0.0818 to -0.0609 |
| msmarco | prefix | 204 | all known positives | in top 100 | 2321 | 0.6666 | 0.7462 | -0.0797 | -0.0892 to -0.0697 |
| msmarco | prefix | 204 | all known positives | log rank (lower is better) | 2321 | 3.7116 | 3.2084 | 0.5032 | 0.4706 to 0.5376 |
| msmarco | prefix | 204 | all known positives | worst rank in the set | 2321 | 5368.7152 | 2545.2809 | 2823.4343 | 545.0251 to 5230.9644 |
| msmarco | prefix | 307 | all known positives | reciprocal rank | 2321 | 0.1631 | 0.1850 | -0.0220 | -0.0266 to -0.0174 |
| msmarco | prefix | 307 | all known positives | in top 10 | 2321 | 0.3415 | 0.3916 | -0.0501 | -0.0596 to -0.0408 |
| msmarco | prefix | 307 | all known positives | in top 100 | 2321 | 0.6943 | 0.7462 | -0.0520 | -0.0610 to -0.0427 |
| msmarco | prefix | 307 | all known positives | log rank (lower is better) | 2321 | 3.5188 | 3.2084 | 0.3104 | 0.2846 to 0.3359 |
| msmarco | prefix | 307 | all known positives | worst rank in the set | 2321 | 3350.8923 | 2545.2809 | 805.6114 | -898.8890 to 2297.5732 |
| msmarco | prefix | 409 | all known positives | reciprocal rank | 2321 | 0.1699 | 0.1850 | -0.0151 | -0.0194 to -0.0107 |
| msmarco | prefix | 409 | all known positives | in top 10 | 2321 | 0.3583 | 0.3916 | -0.0333 | -0.0416 to -0.0249 |
| msmarco | prefix | 409 | all known positives | in top 100 | 2321 | 0.7092 | 0.7462 | -0.0371 | -0.0451 to -0.0292 |
| msmarco | prefix | 409 | all known positives | log rank (lower is better) | 2321 | 3.4082 | 3.2084 | 0.1998 | 0.1789 to 0.2203 |
| msmarco | prefix | 409 | all known positives | worst rank in the set | 2321 | 2777.6907 | 2545.2809 | 232.4097 | -930.6906 to 1110.9854 |
| msmarco | prefix | 512 | all known positives | reciprocal rank | 2321 | 0.1750 | 0.1850 | -0.0100 | -0.0142 to -0.0060 |
| msmarco | prefix | 512 | all known positives | in top 10 | 2321 | 0.3681 | 0.3916 | -0.0235 | -0.0313 to -0.0161 |
| msmarco | prefix | 512 | all known positives | in top 100 | 2321 | 0.7265 | 0.7462 | -0.0198 | -0.0267 to -0.0127 |
| msmarco | prefix | 512 | all known positives | log rank (lower is better) | 2321 | 3.3383 | 3.2084 | 0.1299 | 0.1129 to 0.1472 |
| msmarco | prefix | 512 | all known positives | worst rank in the set | 2321 | 2310.7454 | 2545.2809 | -234.5355 | -1401.4388 to 628.1047 |
| msmarco | prefix | 614 | all known positives | reciprocal rank | 2321 | 0.1785 | 0.1850 | -0.0066 | -0.0101 to -0.0031 |
| msmarco | prefix | 614 | all known positives | in top 10 | 2321 | 0.3763 | 0.3916 | -0.0154 | -0.0219 to -0.0086 |
| msmarco | prefix | 614 | all known positives | in top 100 | 2321 | 0.7328 | 0.7462 | -0.0134 | -0.0201 to -0.0068 |
| msmarco | prefix | 614 | all known positives | log rank (lower is better) | 2321 | 3.2928 | 3.2084 | 0.0844 | 0.0707 to 0.0987 |
| msmarco | prefix | 614 | all known positives | worst rank in the set | 2321 | 2116.4123 | 2545.2809 | -428.8686 | -1512.5863 to 241.6502 |
| msmarco | prefix | 716 | all known positives | reciprocal rank | 2321 | 0.1818 | 0.1850 | -0.0032 | -0.0061 to -0.0003 |
| msmarco | prefix | 716 | all known positives | in top 10 | 2321 | 0.3847 | 0.3916 | -0.0069 | -0.0127 to -0.0010 |
| msmarco | prefix | 716 | all known positives | in top 100 | 2321 | 0.7377 | 0.7462 | -0.0086 | -0.0141 to -0.0028 |
| msmarco | prefix | 716 | all known positives | log rank (lower is better) | 2321 | 3.2503 | 3.2084 | 0.0419 | 0.0311 to 0.0533 |
| msmarco | prefix | 716 | all known positives | worst rank in the set | 2321 | 2141.3783 | 2545.2809 | -403.9026 | -1217.0317 to 205.3371 |
| msmarco | prefix | 819 | all known positives | reciprocal rank | 2321 | 0.1841 | 0.1850 | -0.0010 | -0.0034 to 0.0014 |
| msmarco | prefix | 819 | all known positives | in top 10 | 2321 | 0.3878 | 0.3916 | -0.0038 | -0.0088 to 0.0014 |
| msmarco | prefix | 819 | all known positives | in top 100 | 2321 | 0.7422 | 0.7462 | -0.0040 | -0.0086 to 0.0003 |
| msmarco | prefix | 819 | all known positives | log rank (lower is better) | 2321 | 3.2290 | 3.2084 | 0.0206 | 0.0123 to 0.0289 |
| msmarco | prefix | 819 | all known positives | worst rank in the set | 2321 | 2335.6937 | 2545.2809 | -209.5872 | -785.0153 to 147.4837 |
| msmarco | prefix | 921 | all known positives | reciprocal rank | 2321 | 0.1841 | 0.1850 | -0.0010 | -0.0030 to 0.0009 |
| msmarco | prefix | 921 | all known positives | in top 10 | 2321 | 0.3905 | 0.3916 | -0.0011 | -0.0053 to 0.0030 |
| msmarco | prefix | 921 | all known positives | in top 100 | 2321 | 0.7440 | 0.7462 | -0.0023 | -0.0059 to 0.0015 |
| msmarco | prefix | 921 | all known positives | log rank (lower is better) | 2321 | 3.2161 | 3.2084 | 0.0077 | 0.0020 to 0.0136 |
| msmarco | prefix | 921 | all known positives | worst rank in the set | 2321 | 2641.2115 | 2545.2809 | 95.9306 | -121.2382 to 392.9066 |
| msmarco | shared | 102 | all known positives | reciprocal rank | 2321 | 0.4156 | 0.1850 | 0.2306 | 0.2220 to 0.2385 |
| msmarco | shared | 102 | all known positives | in top 10 | 2321 | 0.7389 | 0.3916 | 0.3473 | 0.3330 to 0.3607 |
| msmarco | shared | 102 | all known positives | in top 100 | 2321 | 0.9560 | 0.7462 | 0.2097 | 0.1971 to 0.2220 |
| msmarco | shared | 102 | all known positives | log rank (lower is better) | 2321 | 1.5678 | 3.2084 | -1.6406 | -1.6880 to -1.5891 |
| msmarco | shared | 102 | all known positives | worst rank in the set | 2321 | 66.7161 | 2545.2809 | -2478.5648 | -4585.2467 to -1035.8177 |
| msmarco | shared | 204 | all known positives | reciprocal rank | 2321 | 0.4090 | 0.1850 | 0.2239 | 0.2159 to 0.2315 |
| msmarco | shared | 204 | all known positives | in top 10 | 2321 | 0.7326 | 0.3916 | 0.3409 | 0.3269 to 0.3547 |
| msmarco | shared | 204 | all known positives | in top 100 | 2321 | 0.9547 | 0.7462 | 0.2084 | 0.1957 to 0.2211 |
| msmarco | shared | 204 | all known positives | log rank (lower is better) | 2321 | 1.5864 | 3.2084 | -1.6220 | -1.6688 to -1.5742 |
| msmarco | shared | 204 | all known positives | worst rank in the set | 2321 | 66.0500 | 2545.2809 | -2479.2309 | -4587.1807 to -1038.7052 |
| msmarco | shared | 307 | all known positives | reciprocal rank | 2321 | 0.3905 | 0.1850 | 0.2055 | 0.1978 to 0.2125 |
| msmarco | shared | 307 | all known positives | in top 10 | 2321 | 0.7139 | 0.3916 | 0.3222 | 0.3086 to 0.3361 |
| msmarco | shared | 307 | all known positives | in top 100 | 2321 | 0.9502 | 0.7462 | 0.2040 | 0.1914 to 0.2161 |
| msmarco | shared | 307 | all known positives | log rank (lower is better) | 2321 | 1.6711 | 3.2084 | -1.5373 | -1.5824 to -1.4909 |
| msmarco | shared | 307 | all known positives | worst rank in the set | 2321 | 84.1271 | 2545.2809 | -2461.1538 | -4562.0179 to -1031.0686 |
| msmarco | shared | 409 | all known positives | reciprocal rank | 2321 | 0.3711 | 0.1850 | 0.1861 | 0.1786 to 0.1930 |
| msmarco | shared | 409 | all known positives | in top 10 | 2321 | 0.6880 | 0.3916 | 0.2964 | 0.2830 to 0.3091 |
| msmarco | shared | 409 | all known positives | in top 100 | 2321 | 0.9423 | 0.7462 | 0.1961 | 0.1833 to 0.2082 |
| msmarco | shared | 409 | all known positives | log rank (lower is better) | 2321 | 1.7726 | 3.2084 | -1.4358 | -1.4793 to -1.3924 |
| msmarco | shared | 409 | all known positives | worst rank in the set | 2321 | 93.8983 | 2545.2809 | -2451.3826 | -4550.5121 to -1021.1351 |
| msmarco | shared | 512 | all known positives | reciprocal rank | 2321 | 0.3512 | 0.1850 | 0.1661 | 0.1592 to 0.1727 |
| msmarco | shared | 512 | all known positives | in top 10 | 2321 | 0.6641 | 0.3916 | 0.2724 | 0.2595 to 0.2849 |
| msmarco | shared | 512 | all known positives | in top 100 | 2321 | 0.9330 | 0.7462 | 0.1868 | 0.1743 to 0.1987 |
| msmarco | shared | 512 | all known positives | log rank (lower is better) | 2321 | 1.8797 | 3.2084 | -1.3287 | -1.3695 to -1.2881 |
| msmarco | shared | 512 | all known positives | worst rank in the set | 2321 | 113.2378 | 2545.2809 | -2432.0431 | -4536.9862 to -1004.2580 |
| msmarco | shared | 614 | all known positives | reciprocal rank | 2321 | 0.3332 | 0.1850 | 0.1482 | 0.1416 to 0.1543 |
| msmarco | shared | 614 | all known positives | in top 10 | 2321 | 0.6370 | 0.3916 | 0.2454 | 0.2328 to 0.2579 |
| msmarco | shared | 614 | all known positives | in top 100 | 2321 | 0.9239 | 0.7462 | 0.1776 | 0.1657 to 0.1893 |
| msmarco | shared | 614 | all known positives | log rank (lower is better) | 2321 | 1.9818 | 3.2084 | -1.2266 | -1.2652 to -1.1884 |
| msmarco | shared | 614 | all known positives | worst rank in the set | 2321 | 132.2688 | 2545.2809 | -2413.0121 | -4511.2813 to -985.5777 |
| msmarco | shared | 716 | all known positives | reciprocal rank | 2321 | 0.3158 | 0.1850 | 0.1308 | 0.1246 to 0.1367 |
| msmarco | shared | 716 | all known positives | in top 10 | 2321 | 0.6102 | 0.3916 | 0.2185 | 0.2066 to 0.2305 |
| msmarco | shared | 716 | all known positives | in top 100 | 2321 | 0.9140 | 0.7462 | 0.1678 | 0.1559 to 0.1794 |
| msmarco | shared | 716 | all known positives | log rank (lower is better) | 2321 | 2.0816 | 3.2084 | -1.1268 | -1.1627 to -1.0907 |
| msmarco | shared | 716 | all known positives | worst rank in the set | 2321 | 158.7635 | 2545.2809 | -2386.5174 | -4462.3453 to -965.4691 |
| msmarco | shared | 819 | all known positives | reciprocal rank | 2321 | 0.3005 | 0.1850 | 0.1155 | 0.1099 to 0.1210 |
| msmarco | shared | 819 | all known positives | in top 10 | 2321 | 0.5862 | 0.3916 | 0.1946 | 0.1828 to 0.2060 |
| msmarco | shared | 819 | all known positives | in top 100 | 2321 | 0.9032 | 0.7462 | 0.1570 | 0.1456 to 0.1683 |
| msmarco | shared | 819 | all known positives | log rank (lower is better) | 2321 | 2.1778 | 3.2084 | -1.0306 | -1.0637 to -0.9969 |
| msmarco | shared | 819 | all known positives | worst rank in the set | 2321 | 190.3688 | 2545.2809 | -2354.9121 | -4416.6820 to -941.4595 |
| msmarco | shared | 921 | all known positives | reciprocal rank | 2321 | 0.2685 | 0.1850 | 0.0835 | 0.0789 to 0.0880 |
| msmarco | shared | 921 | all known positives | in top 10 | 2321 | 0.5344 | 0.3916 | 0.1427 | 0.1327 to 0.1531 |
| msmarco | shared | 921 | all known positives | in top 100 | 2321 | 0.8694 | 0.7462 | 0.1232 | 0.1134 to 0.1331 |
| msmarco | shared | 921 | all known positives | log rank (lower is better) | 2321 | 2.4145 | 3.2084 | -0.7939 | -0.8192 to -0.7691 |
| msmarco | shared | 921 | all known positives | worst rank in the set | 2321 | 309.0539 | 2545.2809 | -2236.2271 | -4251.0463 to -874.9885 |
| msmarco | own | 102 | the mask's own positive | reciprocal rank | 2321 | 0.8479 | 0.1850 | 0.6629 | 0.6526 to 0.6731 |
| msmarco | own | 102 | the mask's own positive | in top 10 | 2321 | 0.9874 | 0.3916 | 0.5957 | 0.5794 to 0.6110 |
| msmarco | own | 102 | the mask's own positive | in top 100 | 2321 | 0.9999 | 0.7462 | 0.2536 | 0.2390 to 0.2672 |
| msmarco | own | 102 | the mask's own positive | log rank (lower is better) | 2321 | 0.2745 | 3.2084 | -2.9339 | -2.9986 to -2.8656 |
| msmarco | own | 102 | the mask's own positive | worst rank in the set | 2321 | 1.7881 | 1250.4414 | -1248.6533 | -2279.9679 to -548.7145 |
| msmarco | own | 204 | the mask's own positive | reciprocal rank | 2321 | 0.8321 | 0.1850 | 0.6471 | 0.6367 to 0.6578 |
| msmarco | own | 204 | the mask's own positive | in top 10 | 2321 | 0.9808 | 0.3916 | 0.5892 | 0.5727 to 0.6047 |
| msmarco | own | 204 | the mask's own positive | in top 100 | 2321 | 0.9999 | 0.7462 | 0.2536 | 0.2390 to 0.2672 |
| msmarco | own | 204 | the mask's own positive | log rank (lower is better) | 2321 | 0.3102 | 3.2084 | -2.8982 | -2.9615 to -2.8303 |
| msmarco | own | 204 | the mask's own positive | worst rank in the set | 2321 | 1.9541 | 1250.4414 | -1248.4873 | -2279.7578 to -548.5159 |
| msmarco | own | 307 | the mask's own positive | reciprocal rank | 2321 | 0.7949 | 0.1850 | 0.6099 | 0.5989 to 0.6207 |
| msmarco | own | 307 | the mask's own positive | in top 10 | 2321 | 0.9719 | 0.3916 | 0.5803 | 0.5641 to 0.5955 |
| msmarco | own | 307 | the mask's own positive | in top 100 | 2321 | 0.9997 | 0.7462 | 0.2535 | 0.2389 to 0.2670 |
| msmarco | own | 307 | the mask's own positive | log rank (lower is better) | 2321 | 0.3927 | 3.2084 | -2.8157 | -2.8775 to -2.7498 |
| msmarco | own | 307 | the mask's own positive | worst rank in the set | 2321 | 2.3890 | 1250.4414 | -1248.0524 | -2279.3071 to -548.0854 |
| msmarco | own | 409 | the mask's own positive | reciprocal rank | 2321 | 0.7495 | 0.1850 | 0.5644 | 0.5536 to 0.5756 |
| msmarco | own | 409 | the mask's own positive | in top 10 | 2321 | 0.9551 | 0.3916 | 0.5634 | 0.5478 to 0.5787 |
| msmarco | own | 409 | the mask's own positive | in top 100 | 2321 | 0.9992 | 0.7462 | 0.2530 | 0.2386 to 0.2665 |
| msmarco | own | 409 | the mask's own positive | log rank (lower is better) | 2321 | 0.5010 | 3.2084 | -2.7074 | -2.7680 to -2.6417 |
| msmarco | own | 409 | the mask's own positive | worst rank in the set | 2321 | 3.0212 | 1250.4414 | -1247.4202 | -2278.6537 to -547.3989 |
| msmarco | own | 512 | the mask's own positive | reciprocal rank | 2321 | 0.6929 | 0.1850 | 0.5078 | 0.4971 to 0.5186 |
| msmarco | own | 512 | the mask's own positive | in top 10 | 2321 | 0.9301 | 0.3916 | 0.5384 | 0.5227 to 0.5543 |
| msmarco | own | 512 | the mask's own positive | in top 100 | 2321 | 0.9977 | 0.7462 | 0.2515 | 0.2374 to 0.2650 |
| msmarco | own | 512 | the mask's own positive | log rank (lower is better) | 2321 | 0.6451 | 3.2084 | -2.5633 | -2.6204 to -2.5002 |
| msmarco | own | 512 | the mask's own positive | worst rank in the set | 2321 | 4.0423 | 1250.4414 | -1246.3990 | -2277.5780 to -546.3539 |
| msmarco | own | 614 | the mask's own positive | reciprocal rank | 2321 | 0.6388 | 0.1850 | 0.4538 | 0.4428 to 0.4646 |
| msmarco | own | 614 | the mask's own positive | in top 10 | 2321 | 0.9017 | 0.3916 | 0.5101 | 0.4945 to 0.5260 |
| msmarco | own | 614 | the mask's own positive | in top 100 | 2321 | 0.9958 | 0.7462 | 0.2496 | 0.2357 to 0.2630 |
| msmarco | own | 614 | the mask's own positive | log rank (lower is better) | 2321 | 0.7975 | 3.2084 | -2.4109 | -2.4670 to -2.3514 |
| msmarco | own | 614 | the mask's own positive | worst rank in the set | 2321 | 5.3370 | 1250.4414 | -1245.1044 | -2276.2048 to -545.0872 |
| msmarco | own | 716 | the mask's own positive | reciprocal rank | 2321 | 0.5891 | 0.1850 | 0.4040 | 0.3925 to 0.4148 |
| msmarco | own | 716 | the mask's own positive | in top 10 | 2321 | 0.8698 | 0.3916 | 0.4781 | 0.4631 to 0.4935 |
| msmarco | own | 716 | the mask's own positive | in top 100 | 2321 | 0.9927 | 0.7462 | 0.2465 | 0.2327 to 0.2599 |
| msmarco | own | 716 | the mask's own positive | log rank (lower is better) | 2321 | 0.9413 | 3.2084 | -2.2671 | -2.3219 to -2.2111 |
| msmarco | own | 716 | the mask's own positive | worst rank in the set | 2321 | 6.8117 | 1250.4414 | -1243.6297 | -2274.6659 to -543.6340 |
| msmarco | own | 819 | the mask's own positive | reciprocal rank | 2321 | 0.5283 | 0.1850 | 0.3433 | 0.3324 to 0.3533 |
| msmarco | own | 819 | the mask's own positive | in top 10 | 2321 | 0.8215 | 0.3916 | 0.4298 | 0.4152 to 0.4447 |
| msmarco | own | 819 | the mask's own positive | in top 100 | 2321 | 0.9837 | 0.7462 | 0.2375 | 0.2240 to 0.2506 |
| msmarco | own | 819 | the mask's own positive | log rank (lower is better) | 2321 | 1.1500 | 3.2084 | -2.0584 | -2.1063 to -2.0089 |
| msmarco | own | 819 | the mask's own positive | worst rank in the set | 2321 | 10.4532 | 1250.4414 | -1239.9882 | -2270.7397 to -540.0581 |
| msmarco | own | 921 | the mask's own positive | reciprocal rank | 2321 | 0.4082 | 0.1850 | 0.2231 | 0.2144 to 0.2311 |
| msmarco | own | 921 | the mask's own positive | in top 10 | 2321 | 0.7038 | 0.3916 | 0.3122 | 0.2987 to 0.3257 |
| msmarco | own | 921 | the mask's own positive | in top 100 | 2321 | 0.9493 | 0.7462 | 0.2031 | 0.1906 to 0.2150 |
| msmarco | own | 921 | the mask's own positive | log rank (lower is better) | 2321 | 1.6712 | 3.2084 | -1.5372 | -1.5750 to -1.5004 |
| msmarco | own | 921 | the mask's own positive | worst rank in the set | 2321 | 32.2790 | 1250.4414 | -1218.1624 | -2244.5610 to -520.8552 |
| msmarco | 1-1 | 102 | held-out | reciprocal rank | 2321 | 0.1271 | 0.1848 | -0.0578 | -0.0640 to -0.0517 |
| msmarco | 1-1 | 102 | held-out | in top 10 | 2321 | 0.3618 | 0.3913 | -0.0295 | -0.0411 to -0.0177 |
| msmarco | 1-1 | 102 | held-out | in top 100 | 2321 | 0.7278 | 0.7461 | -0.0183 | -0.0291 to -0.0076 |
| msmarco | 1-1 | 102 | held-out | log rank (lower is better) | 2321 | 3.4064 | 3.2094 | 0.1970 | 0.1557 to 0.2407 |
| msmarco | 1-1 | 102 | held-out | worst rank in the set | 2321 | 1208.4141 | 1249.5958 | -41.1817 | -967.7516 to 621.0914 |
| msmarco | 1-1 | 102 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1706 | 0.2225 | -0.0519 | -0.0586 to -0.0452 |
| msmarco | 1-1 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1850 | 0.2002 | -0.0153 | -0.0226 to -0.0080 |
| msmarco | 1-1 | 204 | held-out | reciprocal rank | 2321 | 0.1322 | 0.1848 | -0.0527 | -0.0588 to -0.0468 |
| msmarco | 1-1 | 204 | held-out | in top 10 | 2321 | 0.3807 | 0.3913 | -0.0106 | -0.0221 to 0.0003 |
| msmarco | 1-1 | 204 | held-out | in top 100 | 2321 | 0.7492 | 0.7461 | 0.0031 | -0.0071 to 0.0127 |
| msmarco | 1-1 | 204 | held-out | log rank (lower is better) | 2321 | 3.3054 | 3.2094 | 0.0959 | 0.0576 to 0.1355 |
| msmarco | 1-1 | 204 | held-out | worst rank in the set | 2321 | 696.0518 | 1249.5958 | -553.5441 | -1470.1167 to 29.7309 |
| msmarco | 1-1 | 204 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1790 | 0.2225 | -0.0435 | -0.0498 to -0.0373 |
| msmarco | 1-1 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1900 | 0.2002 | -0.0102 | -0.0171 to -0.0035 |
| msmarco | 1-1 | 307 | held-out | reciprocal rank | 2321 | 0.1370 | 0.1848 | -0.0479 | -0.0537 to -0.0424 |
| msmarco | 1-1 | 307 | held-out | in top 10 | 2321 | 0.3853 | 0.3913 | -0.0061 | -0.0166 to 0.0047 |
| msmarco | 1-1 | 307 | held-out | in top 100 | 2321 | 0.7552 | 0.7461 | 0.0090 | -0.0009 to 0.0186 |
| msmarco | 1-1 | 307 | held-out | log rank (lower is better) | 2321 | 3.2612 | 3.2094 | 0.0518 | 0.0169 to 0.0879 |
| msmarco | 1-1 | 307 | held-out | worst rank in the set | 2321 | 658.4948 | 1249.5958 | -591.1010 | -1484.2426 to -20.4163 |
| msmarco | 1-1 | 307 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1838 | 0.2225 | -0.0387 | -0.0444 to -0.0329 |
| msmarco | 1-1 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1936 | 0.2002 | -0.0066 | -0.0132 to -0.0003 |
| msmarco | 1-1 | 409 | held-out | reciprocal rank | 2321 | 0.1417 | 0.1848 | -0.0431 | -0.0488 to -0.0375 |
| msmarco | 1-1 | 409 | held-out | in top 10 | 2321 | 0.3917 | 0.3913 | 0.0003 | -0.0095 to 0.0102 |
| msmarco | 1-1 | 409 | held-out | in top 100 | 2321 | 0.7604 | 0.7461 | 0.0143 | 0.0048 to 0.0232 |
| msmarco | 1-1 | 409 | held-out | log rank (lower is better) | 2321 | 3.2317 | 3.2094 | 0.0223 | -0.0109 to 0.0567 |
| msmarco | 1-1 | 409 | held-out | worst rank in the set | 2321 | 672.1563 | 1249.5958 | -577.4395 | -1440.6037 to -49.6912 |
| msmarco | 1-1 | 409 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1889 | 0.2225 | -0.0336 | -0.0393 to -0.0280 |
| msmarco | 1-1 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1974 | 0.2002 | -0.0028 | -0.0091 to 0.0031 |
| msmarco | 1-1 | 512 | held-out | reciprocal rank | 2321 | 0.1460 | 0.1848 | -0.0388 | -0.0443 to -0.0336 |
| msmarco | 1-1 | 512 | held-out | in top 10 | 2321 | 0.3961 | 0.3913 | 0.0048 | -0.0046 to 0.0141 |
| msmarco | 1-1 | 512 | held-out | in top 100 | 2321 | 0.7615 | 0.7461 | 0.0154 | 0.0064 to 0.0236 |
| msmarco | 1-1 | 512 | held-out | log rank (lower is better) | 2321 | 3.2125 | 3.2094 | 0.0031 | -0.0283 to 0.0359 |
| msmarco | 1-1 | 512 | held-out | worst rank in the set | 2321 | 698.4171 | 1249.5958 | -551.1787 | -1395.4973 to -54.7193 |
| msmarco | 1-1 | 512 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1934 | 0.2225 | -0.0291 | -0.0344 to -0.0238 |
| msmarco | 1-1 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1981 | 0.2002 | -0.0021 | -0.0081 to 0.0037 |
| msmarco | 1-1 | 614 | held-out | reciprocal rank | 2321 | 0.1509 | 0.1848 | -0.0340 | -0.0393 to -0.0289 |
| msmarco | 1-1 | 614 | held-out | in top 10 | 2321 | 0.3942 | 0.3913 | 0.0029 | -0.0064 to 0.0119 |
| msmarco | 1-1 | 614 | held-out | in top 100 | 2321 | 0.7663 | 0.7461 | 0.0201 | 0.0118 to 0.0282 |
| msmarco | 1-1 | 614 | held-out | log rank (lower is better) | 2321 | 3.1972 | 3.2094 | -0.0123 | -0.0425 to 0.0186 |
| msmarco | 1-1 | 614 | held-out | worst rank in the set | 2321 | 657.2932 | 1249.5958 | -592.3026 | -1439.9340 to -78.5581 |
| msmarco | 1-1 | 614 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1965 | 0.2225 | -0.0260 | -0.0313 to -0.0207 |
| msmarco | 1-1 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1989 | 0.2002 | -0.0014 | -0.0069 to 0.0040 |
| msmarco | 1-1 | 716 | held-out | reciprocal rank | 2321 | 0.1550 | 0.1848 | -0.0298 | -0.0350 to -0.0251 |
| msmarco | 1-1 | 716 | held-out | in top 10 | 2321 | 0.3967 | 0.3913 | 0.0054 | -0.0035 to 0.0140 |
| msmarco | 1-1 | 716 | held-out | in top 100 | 2321 | 0.7671 | 0.7461 | 0.0210 | 0.0130 to 0.0286 |
| msmarco | 1-1 | 716 | held-out | log rank (lower is better) | 2321 | 3.1874 | 3.2094 | -0.0221 | -0.0503 to 0.0066 |
| msmarco | 1-1 | 716 | held-out | worst rank in the set | 2321 | 653.9015 | 1249.5958 | -595.6943 | -1433.8428 to -90.5098 |
| msmarco | 1-1 | 716 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.2004 | 0.2225 | -0.0221 | -0.0272 to -0.0173 |
| msmarco | 1-1 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1997 | 0.2002 | -0.0005 | -0.0057 to 0.0046 |
| msmarco | 1-1 | 819 | held-out | reciprocal rank | 2321 | 0.1576 | 0.1848 | -0.0272 | -0.0320 to -0.0229 |
| msmarco | 1-1 | 819 | held-out | in top 10 | 2321 | 0.3959 | 0.3913 | 0.0046 | -0.0039 to 0.0132 |
| msmarco | 1-1 | 819 | held-out | in top 100 | 2321 | 0.7681 | 0.7461 | 0.0220 | 0.0145 to 0.0296 |
| msmarco | 1-1 | 819 | held-out | log rank (lower is better) | 2321 | 3.1793 | 3.2094 | -0.0301 | -0.0563 to -0.0040 |
| msmarco | 1-1 | 819 | held-out | worst rank in the set | 2321 | 670.3369 | 1249.5958 | -579.2589 | -1396.5575 to -99.3513 |
| msmarco | 1-1 | 819 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.2021 | 0.2225 | -0.0204 | -0.0251 to -0.0158 |
| msmarco | 1-1 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1990 | 0.2002 | -0.0012 | -0.0061 to 0.0035 |
| msmarco | 1-1 | 921 | held-out | reciprocal rank | 2321 | 0.1674 | 0.1848 | -0.0174 | -0.0217 to -0.0135 |
| msmarco | 1-1 | 921 | held-out | in top 10 | 2321 | 0.3981 | 0.3913 | 0.0067 | -0.0004 to 0.0140 |
| msmarco | 1-1 | 921 | held-out | in top 100 | 2321 | 0.7685 | 0.7461 | 0.0224 | 0.0158 to 0.0288 |
| msmarco | 1-1 | 921 | held-out | log rank (lower is better) | 2321 | 3.1629 | 3.2094 | -0.0465 | -0.0667 to -0.0267 |
| msmarco | 1-1 | 921 | held-out | worst rank in the set | 2321 | 711.3463 | 1249.5958 | -538.2496 | -1253.9266 to -111.8249 |
| msmarco | 1-1 | 921 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.2102 | 0.2225 | -0.0123 | -0.0164 to -0.0084 |
| msmarco | 1-1 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.2000 | 0.2002 | -0.0003 | -0.0043 to 0.0038 |
| msmarco | 2-2 | 102 | held-out | reciprocal rank | 81 | 0.0833 | 0.1012 | -0.0179 | -0.0332 to -0.0028 |
| msmarco | 2-2 | 102 | held-out | in top 10 | 81 | 0.2001 | 0.2215 | -0.0213 | -0.0479 to 0.0035 |
| msmarco | 2-2 | 102 | held-out | in top 100 | 81 | 0.6402 | 0.5934 | 0.0468 | -0.0022 to 0.0996 |
| msmarco | 2-2 | 102 | held-out | log rank (lower is better) | 81 | 3.9734 | 4.0342 | -0.0608 | -0.2206 to 0.0798 |
| msmarco | 2-2 | 102 | held-out | worst rank in the set | 81 | 1450.7356 | 1595.1883 | -144.4527 | -911.6489 to 504.9452 |
| msmarco | 2-2 | 102 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1183 | 0.1441 | -0.0258 | -0.0440 to -0.0080 |
| msmarco | 2-2 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1159 | 0.1095 | 0.0064 | -0.0099 to 0.0233 |
| msmarco | 2-2 | 204 | held-out | reciprocal rank | 81 | 0.0887 | 0.1012 | -0.0125 | -0.0277 to 0.0023 |
| msmarco | 2-2 | 204 | held-out | in top 10 | 81 | 0.2173 | 0.2215 | -0.0041 | -0.0307 to 0.0205 |
| msmarco | 2-2 | 204 | held-out | in top 100 | 81 | 0.6602 | 0.5934 | 0.0669 | 0.0211 to 0.1174 |
| msmarco | 2-2 | 204 | held-out | log rank (lower is better) | 81 | 3.8795 | 4.0342 | -0.1547 | -0.3101 to -0.0248 |
| msmarco | 2-2 | 204 | held-out | worst rank in the set | 81 | 1143.8601 | 1595.1883 | -451.3282 | -1431.4914 to 132.1880 |
| msmarco | 2-2 | 204 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1283 | 0.1441 | -0.0158 | -0.0331 to 0.0019 |
| msmarco | 2-2 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1198 | 0.1095 | 0.0103 | -0.0045 to 0.0252 |
| msmarco | 2-2 | 307 | held-out | reciprocal rank | 81 | 0.0892 | 0.1012 | -0.0119 | -0.0258 to 0.0017 |
| msmarco | 2-2 | 307 | held-out | in top 10 | 81 | 0.2225 | 0.2215 | 0.0010 | -0.0248 to 0.0254 |
| msmarco | 2-2 | 307 | held-out | in top 100 | 81 | 0.6597 | 0.5934 | 0.0664 | 0.0239 to 0.1141 |
| msmarco | 2-2 | 307 | held-out | log rank (lower is better) | 81 | 3.8557 | 4.0342 | -0.1785 | -0.3224 to -0.0568 |
| msmarco | 2-2 | 307 | held-out | worst rank in the set | 81 | 1056.6543 | 1595.1883 | -538.5339 | -1639.3839 to 92.5523 |
| msmarco | 2-2 | 307 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1305 | 0.1441 | -0.0136 | -0.0294 to 0.0028 |
| msmarco | 2-2 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1199 | 0.1095 | 0.0104 | -0.0026 to 0.0240 |
| msmarco | 2-2 | 409 | held-out | reciprocal rank | 81 | 0.0930 | 0.1012 | -0.0082 | -0.0209 to 0.0047 |
| msmarco | 2-2 | 409 | held-out | in top 10 | 81 | 0.2281 | 0.2215 | 0.0067 | -0.0184 to 0.0304 |
| msmarco | 2-2 | 409 | held-out | in top 100 | 81 | 0.6590 | 0.5934 | 0.0656 | 0.0256 to 0.1092 |
| msmarco | 2-2 | 409 | held-out | log rank (lower is better) | 81 | 3.8583 | 4.0342 | -0.1759 | -0.3088 to -0.0615 |
| msmarco | 2-2 | 409 | held-out | worst rank in the set | 81 | 1102.4218 | 1595.1883 | -492.7665 | -1508.5731 to 102.1527 |
| msmarco | 2-2 | 409 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1360 | 0.1441 | -0.0081 | -0.0234 to 0.0078 |
| msmarco | 2-2 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1203 | 0.1095 | 0.0108 | -0.0022 to 0.0243 |
| msmarco | 2-2 | 512 | held-out | reciprocal rank | 81 | 0.0938 | 0.1012 | -0.0074 | -0.0188 to 0.0049 |
| msmarco | 2-2 | 512 | held-out | in top 10 | 81 | 0.2258 | 0.2215 | 0.0044 | -0.0206 to 0.0275 |
| msmarco | 2-2 | 512 | held-out | in top 100 | 81 | 0.6590 | 0.5934 | 0.0656 | 0.0266 to 0.1072 |
| msmarco | 2-2 | 512 | held-out | log rank (lower is better) | 81 | 3.8712 | 4.0342 | -0.1630 | -0.2889 to -0.0542 |
| msmarco | 2-2 | 512 | held-out | worst rank in the set | 81 | 1146.7454 | 1595.1883 | -448.4429 | -1400.6514 to 122.4809 |
| msmarco | 2-2 | 512 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1364 | 0.1441 | -0.0077 | -0.0216 to 0.0070 |
| msmarco | 2-2 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1199 | 0.1095 | 0.0104 | -0.0012 to 0.0225 |
| msmarco | 2-2 | 614 | held-out | reciprocal rank | 81 | 0.0946 | 0.1012 | -0.0066 | -0.0178 to 0.0048 |
| msmarco | 2-2 | 614 | held-out | in top 10 | 81 | 0.2261 | 0.2215 | 0.0046 | -0.0197 to 0.0276 |
| msmarco | 2-2 | 614 | held-out | in top 100 | 81 | 0.6515 | 0.5934 | 0.0581 | 0.0211 to 0.0976 |
| msmarco | 2-2 | 614 | held-out | log rank (lower is better) | 81 | 3.8744 | 4.0342 | -0.1598 | -0.2793 to -0.0578 |
| msmarco | 2-2 | 614 | held-out | worst rank in the set | 81 | 1156.5288 | 1595.1883 | -438.6595 | -1369.5680 to 122.2087 |
| msmarco | 2-2 | 614 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1374 | 0.1441 | -0.0067 | -0.0207 to 0.0074 |
| msmarco | 2-2 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1189 | 0.1095 | 0.0094 | -0.0015 to 0.0208 |
| msmarco | 2-2 | 716 | held-out | reciprocal rank | 81 | 0.0946 | 0.1012 | -0.0066 | -0.0174 to 0.0041 |
| msmarco | 2-2 | 716 | held-out | in top 10 | 81 | 0.2225 | 0.2215 | 0.0010 | -0.0211 to 0.0216 |
| msmarco | 2-2 | 716 | held-out | in top 100 | 81 | 0.6438 | 0.5934 | 0.0504 | 0.0171 to 0.0872 |
| msmarco | 2-2 | 716 | held-out | log rank (lower is better) | 81 | 3.8796 | 4.0342 | -0.1546 | -0.2698 to -0.0571 |
| msmarco | 2-2 | 716 | held-out | worst rank in the set | 81 | 1131.4485 | 1595.1883 | -463.7397 | -1422.3527 to 95.8645 |
| msmarco | 2-2 | 716 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1363 | 0.1441 | -0.0078 | -0.0212 to 0.0057 |
| msmarco | 2-2 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1174 | 0.1095 | 0.0079 | -0.0027 to 0.0192 |
| msmarco | 2-2 | 819 | held-out | reciprocal rank | 81 | 0.0975 | 0.1012 | -0.0036 | -0.0139 to 0.0067 |
| msmarco | 2-2 | 819 | held-out | in top 10 | 81 | 0.2209 | 0.2215 | -0.0005 | -0.0222 to 0.0189 |
| msmarco | 2-2 | 819 | held-out | in top 100 | 81 | 0.6394 | 0.5934 | 0.0460 | 0.0130 to 0.0833 |
| msmarco | 2-2 | 819 | held-out | log rank (lower is better) | 81 | 3.8863 | 4.0342 | -0.1479 | -0.2512 to -0.0585 |
| msmarco | 2-2 | 819 | held-out | worst rank in the set | 81 | 1151.1919 | 1595.1883 | -443.9964 | -1370.1628 to 102.2465 |
| msmarco | 2-2 | 819 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1387 | 0.1441 | -0.0054 | -0.0185 to 0.0073 |
| msmarco | 2-2 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1178 | 0.1095 | 0.0083 | -0.0018 to 0.0190 |
| msmarco | 2-2 | 921 | held-out | reciprocal rank | 81 | 0.0983 | 0.1012 | -0.0029 | -0.0115 to 0.0055 |
| msmarco | 2-2 | 921 | held-out | in top 10 | 81 | 0.2209 | 0.2215 | -0.0005 | -0.0234 to 0.0199 |
| msmarco | 2-2 | 921 | held-out | in top 100 | 81 | 0.6381 | 0.5934 | 0.0448 | 0.0141 to 0.0791 |
| msmarco | 2-2 | 921 | held-out | log rank (lower is better) | 81 | 3.9156 | 4.0342 | -0.1186 | -0.1989 to -0.0471 |
| msmarco | 2-2 | 921 | held-out | worst rank in the set | 81 | 1174.1800 | 1595.1883 | -421.0082 | -1258.6538 to 55.7784 |
| msmarco | 2-2 | 921 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1400 | 0.1441 | -0.0041 | -0.0163 to 0.0075 |
| msmarco | 2-2 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1146 | 0.1095 | 0.0051 | -0.0031 to 0.0140 |

Document-weighted: every evaluated positive counts once. Larger queries dominate.

| environment | arm | k | positives evaluated | measure | queries | arm value | full query, same documents | difference | 95% interval (pointwise) |
|---|---|---|---|---|---|---|---|---|---|
| scifact | prefix | 102 | all known positives | reciprocal rank | 69 | 0.3460 | 0.4358 | -0.0898 | -0.1290 to -0.0517 |
| scifact | prefix | 102 | all known positives | in top 10 | 69 | 0.6871 | 0.8344 | -0.1472 | -0.2125 to -0.0859 |
| scifact | prefix | 102 | all known positives | in top 100 | 69 | 0.9325 | 0.9816 | -0.0491 | -0.1000 to -0.0062 |
| scifact | prefix | 102 | all known positives | log rank (lower is better) | 69 | 1.8410 | 1.2983 | 0.5427 | 0.3816 to 0.7220 |
| scifact | prefix | 102 | all known positives | worst rank in the set | 69 | 66.4356 | 18.1227 | 48.3129 | 16.5351 to 94.1305 |
| scifact | prefix | 204 | all known positives | reciprocal rank | 69 | 0.3925 | 0.4358 | -0.0433 | -0.0744 to -0.0152 |
| scifact | prefix | 204 | all known positives | in top 10 | 69 | 0.7669 | 0.8344 | -0.0675 | -0.1118 to -0.0266 |
| scifact | prefix | 204 | all known positives | in top 100 | 69 | 0.9632 | 0.9816 | -0.0184 | -0.0524 to 0.0113 |
| scifact | prefix | 204 | all known positives | log rank (lower is better) | 69 | 1.5926 | 1.2983 | 0.2943 | 0.1839 to 0.4210 |
| scifact | prefix | 204 | all known positives | worst rank in the set | 69 | 41.4110 | 18.1227 | 23.2883 | 4.4731 to 49.3798 |
| scifact | prefix | 307 | all known positives | reciprocal rank | 69 | 0.4013 | 0.4358 | -0.0345 | -0.0580 to -0.0127 |
| scifact | prefix | 307 | all known positives | in top 10 | 69 | 0.7730 | 0.8344 | -0.0613 | -0.1056 to -0.0186 |
| scifact | prefix | 307 | all known positives | in top 100 | 69 | 0.9693 | 0.9816 | -0.0123 | -0.0452 to 0.0124 |
| scifact | prefix | 307 | all known positives | log rank (lower is better) | 69 | 1.5072 | 1.2983 | 0.2089 | 0.1192 to 0.3108 |
| scifact | prefix | 307 | all known positives | worst rank in the set | 69 | 30.3865 | 18.1227 | 12.2638 | 2.4809 to 25.5876 |
| scifact | prefix | 409 | all known positives | reciprocal rank | 69 | 0.4198 | 0.4358 | -0.0159 | -0.0365 to 0.0034 |
| scifact | prefix | 409 | all known positives | in top 10 | 69 | 0.7791 | 0.8344 | -0.0552 | -0.1090 to -0.0061 |
| scifact | prefix | 409 | all known positives | in top 100 | 69 | 0.9877 | 0.9816 | 0.0061 | 0.0000 to 0.0191 |
| scifact | prefix | 409 | all known positives | log rank (lower is better) | 69 | 1.4106 | 1.2983 | 0.1124 | 0.0462 to 0.1894 |
| scifact | prefix | 409 | all known positives | worst rank in the set | 69 | 21.7055 | 18.1227 | 3.5828 | -1.3431 to 9.8535 |
| scifact | prefix | 512 | all known positives | reciprocal rank | 69 | 0.4215 | 0.4358 | -0.0142 | -0.0388 to 0.0080 |
| scifact | prefix | 512 | all known positives | in top 10 | 69 | 0.7669 | 0.8344 | -0.0675 | -0.1118 to -0.0253 |
| scifact | prefix | 512 | all known positives | in top 100 | 69 | 0.9877 | 0.9816 | 0.0061 | 0.0000 to 0.0191 |
| scifact | prefix | 512 | all known positives | log rank (lower is better) | 69 | 1.3933 | 1.2983 | 0.0950 | 0.0323 to 0.1649 |
| scifact | prefix | 512 | all known positives | worst rank in the set | 69 | 18.4417 | 18.1227 | 0.3190 | -2.8414 to 3.1430 |
| scifact | prefix | 614 | all known positives | reciprocal rank | 69 | 0.4209 | 0.4358 | -0.0149 | -0.0367 to 0.0066 |
| scifact | prefix | 614 | all known positives | in top 10 | 69 | 0.8160 | 0.8344 | -0.0184 | -0.0500 to 0.0119 |
| scifact | prefix | 614 | all known positives | in top 100 | 69 | 0.9816 | 0.9816 | 0.0000 | 0.0000 to 0.0000 |
| scifact | prefix | 614 | all known positives | log rank (lower is better) | 69 | 1.3585 | 1.2983 | 0.0602 | 0.0111 to 0.1112 |
| scifact | prefix | 614 | all known positives | worst rank in the set | 69 | 19.7117 | 18.1227 | 1.5890 | -1.0703 to 4.7294 |
| scifact | prefix | 716 | all known positives | reciprocal rank | 69 | 0.4314 | 0.4358 | -0.0043 | -0.0215 to 0.0132 |
| scifact | prefix | 716 | all known positives | in top 10 | 69 | 0.8221 | 0.8344 | -0.0123 | -0.0377 to 0.0116 |
| scifact | prefix | 716 | all known positives | in top 100 | 69 | 0.9877 | 0.9816 | 0.0061 | 0.0000 to 0.0191 |
| scifact | prefix | 716 | all known positives | log rank (lower is better) | 69 | 1.3383 | 1.2983 | 0.0401 | -0.0003 to 0.0851 |
| scifact | prefix | 716 | all known positives | worst rank in the set | 69 | 19.6933 | 18.1227 | 1.5706 | -1.2130 to 5.4656 |
| scifact | prefix | 819 | all known positives | reciprocal rank | 69 | 0.4323 | 0.4358 | -0.0035 | -0.0200 to 0.0124 |
| scifact | prefix | 819 | all known positives | in top 10 | 69 | 0.8098 | 0.8344 | -0.0245 | -0.0729 to 0.0123 |
| scifact | prefix | 819 | all known positives | in top 100 | 69 | 0.9816 | 0.9816 | 0.0000 | 0.0000 to 0.0000 |
| scifact | prefix | 819 | all known positives | log rank (lower is better) | 69 | 1.3243 | 1.2983 | 0.0261 | -0.0141 to 0.0712 |
| scifact | prefix | 819 | all known positives | worst rank in the set | 69 | 20.5644 | 18.1227 | 2.4417 | -0.7593 to 8.0320 |
| scifact | prefix | 921 | all known positives | reciprocal rank | 69 | 0.4293 | 0.4358 | -0.0064 | -0.0170 to 0.0016 |
| scifact | prefix | 921 | all known positives | in top 10 | 69 | 0.8282 | 0.8344 | -0.0061 | -0.0364 to 0.0189 |
| scifact | prefix | 921 | all known positives | in top 100 | 69 | 0.9816 | 0.9816 | 0.0000 | 0.0000 to 0.0000 |
| scifact | prefix | 921 | all known positives | log rank (lower is better) | 69 | 1.3114 | 1.2983 | 0.0131 | -0.0124 to 0.0427 |
| scifact | prefix | 921 | all known positives | worst rank in the set | 69 | 19.5337 | 18.1227 | 1.4110 | -1.1447 to 5.7279 |
| scifact | shared | 102 | all known positives | reciprocal rank | 69 | 0.6229 | 0.4358 | 0.1872 | 0.1383 to 0.2396 |
| scifact | shared | 102 | all known positives | in top 10 | 69 | 0.9939 | 0.8344 | 0.1595 | 0.0949 to 0.2284 |
| scifact | shared | 102 | all known positives | in top 100 | 69 | 1.0000 | 0.9816 | 0.0184 | 0.0000 to 0.0422 |
| scifact | shared | 102 | all known positives | log rank (lower is better) | 69 | 0.6345 | 1.2983 | -0.6638 | -0.8726 to -0.4854 |
| scifact | shared | 102 | all known positives | worst rank in the set | 69 | 3.5583 | 18.1227 | -14.5644 | -26.2573 to -6.4061 |
| scifact | shared | 204 | all known positives | reciprocal rank | 69 | 0.6229 | 0.4358 | 0.1872 | 0.1390 to 0.2377 |
| scifact | shared | 204 | all known positives | in top 10 | 69 | 0.9939 | 0.8344 | 0.1595 | 0.0994 to 0.2281 |
| scifact | shared | 204 | all known positives | in top 100 | 69 | 1.0000 | 0.9816 | 0.0184 | 0.0000 to 0.0422 |
| scifact | shared | 204 | all known positives | log rank (lower is better) | 69 | 0.6431 | 1.2983 | -0.6552 | -0.8534 to -0.4839 |
| scifact | shared | 204 | all known positives | worst rank in the set | 69 | 3.8098 | 18.1227 | -14.3129 | -25.6433 to -6.2505 |
| scifact | shared | 307 | all known positives | reciprocal rank | 69 | 0.6125 | 0.4358 | 0.1768 | 0.1305 to 0.2274 |
| scifact | shared | 307 | all known positives | in top 10 | 69 | 0.9877 | 0.8344 | 0.1534 | 0.0937 to 0.2204 |
| scifact | shared | 307 | all known positives | in top 100 | 69 | 1.0000 | 0.9816 | 0.0184 | 0.0000 to 0.0422 |
| scifact | shared | 307 | all known positives | log rank (lower is better) | 69 | 0.6736 | 1.2983 | -0.6247 | -0.8226 to -0.4606 |
| scifact | shared | 307 | all known positives | worst rank in the set | 69 | 3.9325 | 18.1227 | -14.1902 | -25.5726 to -6.1360 |
| scifact | shared | 409 | all known positives | reciprocal rank | 69 | 0.6070 | 0.4358 | 0.1713 | 0.1265 to 0.2204 |
| scifact | shared | 409 | all known positives | in top 10 | 69 | 0.9755 | 0.8344 | 0.1411 | 0.0833 to 0.2036 |
| scifact | shared | 409 | all known positives | in top 100 | 69 | 1.0000 | 0.9816 | 0.0184 | 0.0000 to 0.0422 |
| scifact | shared | 409 | all known positives | log rank (lower is better) | 69 | 0.6926 | 1.2983 | -0.6056 | -0.8012 to -0.4424 |
| scifact | shared | 409 | all known positives | worst rank in the set | 69 | 4.1411 | 18.1227 | -13.9816 | -25.4233 to -5.9742 |
| scifact | shared | 512 | all known positives | reciprocal rank | 69 | 0.6020 | 0.4358 | 0.1662 | 0.1217 to 0.2154 |
| scifact | shared | 512 | all known positives | in top 10 | 69 | 0.9693 | 0.8344 | 0.1350 | 0.0774 to 0.1976 |
| scifact | shared | 512 | all known positives | in top 100 | 69 | 1.0000 | 0.9816 | 0.0184 | 0.0000 to 0.0422 |
| scifact | shared | 512 | all known positives | log rank (lower is better) | 69 | 0.7086 | 1.2983 | -0.5897 | -0.7849 to -0.4292 |
| scifact | shared | 512 | all known positives | worst rank in the set | 69 | 4.2945 | 18.1227 | -13.8282 | -25.1980 to -5.8538 |
| scifact | shared | 614 | all known positives | reciprocal rank | 69 | 0.5931 | 0.4358 | 0.1574 | 0.1149 to 0.2053 |
| scifact | shared | 614 | all known positives | in top 10 | 69 | 0.9571 | 0.8344 | 0.1227 | 0.0674 to 0.1856 |
| scifact | shared | 614 | all known positives | in top 100 | 69 | 1.0000 | 0.9816 | 0.0184 | 0.0000 to 0.0422 |
| scifact | shared | 614 | all known positives | log rank (lower is better) | 69 | 0.7332 | 1.2983 | -0.5651 | -0.7517 to -0.4132 |
| scifact | shared | 614 | all known positives | worst rank in the set | 69 | 4.5767 | 18.1227 | -13.5460 | -24.7527 to -5.6535 |
| scifact | shared | 716 | all known positives | reciprocal rank | 69 | 0.5819 | 0.4358 | 0.1461 | 0.1059 to 0.1916 |
| scifact | shared | 716 | all known positives | in top 10 | 69 | 0.9509 | 0.8344 | 0.1166 | 0.0636 to 0.1745 |
| scifact | shared | 716 | all known positives | in top 100 | 69 | 1.0000 | 0.9816 | 0.0184 | 0.0000 to 0.0422 |
| scifact | shared | 716 | all known positives | log rank (lower is better) | 69 | 0.7701 | 1.2983 | -0.5282 | -0.7014 to -0.3860 |
| scifact | shared | 716 | all known positives | worst rank in the set | 69 | 5.0245 | 18.1227 | -13.0982 | -23.9377 to -5.3998 |
| scifact | shared | 819 | all known positives | reciprocal rank | 69 | 0.5694 | 0.4358 | 0.1336 | 0.0946 to 0.1769 |
| scifact | shared | 819 | all known positives | in top 10 | 69 | 0.9509 | 0.8344 | 0.1166 | 0.0636 to 0.1745 |
| scifact | shared | 819 | all known positives | in top 100 | 69 | 1.0000 | 0.9816 | 0.0184 | 0.0000 to 0.0422 |
| scifact | shared | 819 | all known positives | log rank (lower is better) | 69 | 0.8102 | 1.2983 | -0.4881 | -0.6496 to -0.3536 |
| scifact | shared | 819 | all known positives | worst rank in the set | 69 | 5.6012 | 18.1227 | -12.5215 | -22.9631 to -5.1469 |
| scifact | shared | 921 | all known positives | reciprocal rank | 69 | 0.5326 | 0.4358 | 0.0969 | 0.0677 to 0.1295 |
| scifact | shared | 921 | all known positives | in top 10 | 69 | 0.9448 | 0.8344 | 0.1104 | 0.0621 to 0.1656 |
| scifact | shared | 921 | all known positives | in top 100 | 69 | 0.9939 | 0.9816 | 0.0123 | 0.0000 to 0.0316 |
| scifact | shared | 921 | all known positives | log rank (lower is better) | 69 | 0.9176 | 1.2983 | -0.3807 | -0.4966 to -0.2824 |
| scifact | shared | 921 | all known positives | worst rank in the set | 69 | 7.6074 | 18.1227 | -10.5153 | -18.5751 to -4.5291 |
| scifact | own | 102 | the mask's own positive | reciprocal rank | 69 | 0.9660 | 0.4383 | 0.5277 | 0.4619 to 0.5882 |
| scifact | own | 102 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 102 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 102 | the mask's own positive | log rank (lower is better) | 69 | 0.0480 | 1.3133 | -1.2653 | -1.5042 to -1.0501 |
| scifact | own | 102 | the mask's own positive | worst rank in the set | 69 | 1.0713 | 11.4116 | -10.3403 | -18.7467 to -4.9031 |
| scifact | own | 204 | the mask's own positive | reciprocal rank | 69 | 0.9696 | 0.4383 | 0.5313 | 0.4658 to 0.5925 |
| scifact | own | 204 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 204 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 204 | the mask's own positive | log rank (lower is better) | 69 | 0.0430 | 1.3133 | -1.2703 | -1.5092 to -1.0546 |
| scifact | own | 204 | the mask's own positive | worst rank in the set | 69 | 1.0640 | 11.4116 | -10.3476 | -18.7752 to -4.9147 |
| scifact | own | 307 | the mask's own positive | reciprocal rank | 69 | 0.9630 | 0.4383 | 0.5247 | 0.4594 to 0.5852 |
| scifact | own | 307 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 307 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 307 | the mask's own positive | log rank (lower is better) | 69 | 0.0522 | 1.3133 | -1.2611 | -1.5003 to -1.0468 |
| scifact | own | 307 | the mask's own positive | worst rank in the set | 69 | 1.0773 | 11.4116 | -10.3343 | -18.7713 to -4.9026 |
| scifact | own | 409 | the mask's own positive | reciprocal rank | 69 | 0.9235 | 0.4383 | 0.4853 | 0.4154 to 0.5506 |
| scifact | own | 409 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 409 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 409 | the mask's own positive | log rank (lower is better) | 69 | 0.1090 | 1.3133 | -1.2043 | -1.4434 to -0.9811 |
| scifact | own | 409 | the mask's own positive | worst rank in the set | 69 | 1.1643 | 11.4116 | -10.2473 | -18.6847 to -4.8297 |
| scifact | own | 512 | the mask's own positive | reciprocal rank | 69 | 0.8952 | 0.4383 | 0.4570 | 0.3883 to 0.5231 |
| scifact | own | 512 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 512 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 512 | the mask's own positive | log rank (lower is better) | 69 | 0.1547 | 1.3133 | -1.1586 | -1.3995 to -0.9391 |
| scifact | own | 512 | the mask's own positive | worst rank in the set | 69 | 1.2476 | 11.4116 | -10.1640 | -18.6506 to -4.7270 |
| scifact | own | 614 | the mask's own positive | reciprocal rank | 69 | 0.8742 | 0.4383 | 0.4360 | 0.3720 to 0.5014 |
| scifact | own | 614 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 614 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 614 | the mask's own positive | log rank (lower is better) | 69 | 0.1897 | 1.3133 | -1.1236 | -1.3618 to -0.9094 |
| scifact | own | 614 | the mask's own positive | worst rank in the set | 69 | 1.3164 | 11.4116 | -10.0952 | -18.5417 to -4.6541 |
| scifact | own | 716 | the mask's own positive | reciprocal rank | 69 | 0.8555 | 0.4383 | 0.4173 | 0.3527 to 0.4839 |
| scifact | own | 716 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 716 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 716 | the mask's own positive | log rank (lower is better) | 69 | 0.2157 | 1.3133 | -1.0976 | -1.3384 to -0.8825 |
| scifact | own | 716 | the mask's own positive | worst rank in the set | 69 | 1.3539 | 11.4116 | -10.0577 | -18.5146 to -4.6111 |
| scifact | own | 819 | the mask's own positive | reciprocal rank | 69 | 0.8003 | 0.4383 | 0.3620 | 0.2976 to 0.4280 |
| scifact | own | 819 | the mask's own positive | in top 10 | 69 | 1.0000 | 0.8287 | 0.1713 | 0.1051 to 0.2461 |
| scifact | own | 819 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 819 | the mask's own positive | log rank (lower is better) | 69 | 0.3152 | 1.3133 | -0.9981 | -1.2193 to -0.8037 |
| scifact | own | 819 | the mask's own positive | worst rank in the set | 69 | 1.5800 | 11.4116 | -9.8316 | -18.2486 to -4.4248 |
| scifact | own | 921 | the mask's own positive | reciprocal rank | 69 | 0.7083 | 0.4383 | 0.2700 | 0.2171 to 0.3257 |
| scifact | own | 921 | the mask's own positive | in top 10 | 69 | 0.9783 | 0.8287 | 0.1495 | 0.0942 to 0.2126 |
| scifact | own | 921 | the mask's own positive | in top 100 | 69 | 1.0000 | 0.9807 | 0.0193 | 0.0000 to 0.0435 |
| scifact | own | 921 | the mask's own positive | log rank (lower is better) | 69 | 0.5092 | 1.3133 | -0.8041 | -0.9811 to -0.6498 |
| scifact | own | 921 | the mask's own positive | worst rank in the set | 69 | 2.2106 | 11.4116 | -9.2010 | -17.2844 to -3.9750 |
| scifact | 1-1 | 102 | held-out | reciprocal rank | 69 | 0.3333 | 0.4346 | -0.1013 | -0.1378 to -0.0641 |
| scifact | 1-1 | 102 | held-out | in top 10 | 69 | 0.8816 | 0.8309 | 0.0507 | 0.0024 to 0.0991 |
| scifact | 1-1 | 102 | held-out | in top 100 | 69 | 0.9758 | 0.9807 | -0.0048 | -0.0290 to 0.0193 |
| scifact | 1-1 | 102 | held-out | log rank (lower is better) | 69 | 1.4306 | 1.3197 | 0.1109 | -0.0223 to 0.2312 |
| scifact | 1-1 | 102 | held-out | worst rank in the set | 69 | 11.6987 | 11.4058 | 0.2929 | -2.1880 to 3.2870 |
| scifact | 1-1 | 102 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4615 | 0.5243 | -0.0628 | -0.0969 to -0.0253 |
| scifact | 1-1 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5327 | 0.5066 | 0.0261 | -0.0236 to 0.0785 |
| scifact | 1-1 | 204 | held-out | reciprocal rank | 69 | 0.3387 | 0.4346 | -0.0959 | -0.1330 to -0.0606 |
| scifact | 1-1 | 204 | held-out | in top 10 | 69 | 0.8816 | 0.8309 | 0.0507 | 0.0060 to 0.0979 |
| scifact | 1-1 | 204 | held-out | in top 100 | 69 | 0.9758 | 0.9807 | -0.0048 | -0.0290 to 0.0193 |
| scifact | 1-1 | 204 | held-out | log rank (lower is better) | 69 | 1.3870 | 1.3197 | 0.0672 | -0.0619 to 0.1814 |
| scifact | 1-1 | 204 | held-out | worst rank in the set | 69 | 11.2802 | 11.4058 | -0.1256 | -2.5600 to 2.7377 |
| scifact | 1-1 | 204 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4667 | 0.5243 | -0.0576 | -0.0919 to -0.0217 |
| scifact | 1-1 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5512 | 0.5066 | 0.0446 | -0.0053 to 0.0948 |
| scifact | 1-1 | 307 | held-out | reciprocal rank | 69 | 0.3414 | 0.4346 | -0.0932 | -0.1303 to -0.0568 |
| scifact | 1-1 | 307 | held-out | in top 10 | 69 | 0.8798 | 0.8309 | 0.0489 | -0.0024 to 0.1063 |
| scifact | 1-1 | 307 | held-out | in top 100 | 69 | 0.9783 | 0.9807 | -0.0024 | -0.0290 to 0.0242 |
| scifact | 1-1 | 307 | held-out | log rank (lower is better) | 69 | 1.3638 | 1.3197 | 0.0440 | -0.0850 to 0.1609 |
| scifact | 1-1 | 307 | held-out | worst rank in the set | 69 | 11.0586 | 11.4058 | -0.3472 | -2.9580 to 2.6385 |
| scifact | 1-1 | 307 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4684 | 0.5243 | -0.0559 | -0.0914 to -0.0188 |
| scifact | 1-1 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5476 | 0.5066 | 0.0409 | -0.0071 to 0.0906 |
| scifact | 1-1 | 409 | held-out | reciprocal rank | 69 | 0.3636 | 0.4346 | -0.0710 | -0.1051 to -0.0369 |
| scifact | 1-1 | 409 | held-out | in top 10 | 69 | 0.8750 | 0.8309 | 0.0441 | -0.0000 to 0.0888 |
| scifact | 1-1 | 409 | held-out | in top 100 | 69 | 0.9783 | 0.9807 | -0.0024 | -0.0290 to 0.0242 |
| scifact | 1-1 | 409 | held-out | log rank (lower is better) | 69 | 1.3285 | 1.3197 | 0.0087 | -0.1118 to 0.1126 |
| scifact | 1-1 | 409 | held-out | worst rank in the set | 69 | 10.6268 | 11.4058 | -0.7790 | -3.2181 to 1.8220 |
| scifact | 1-1 | 409 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4834 | 0.5243 | -0.0408 | -0.0721 to -0.0076 |
| scifact | 1-1 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5520 | 0.5066 | 0.0453 | 0.0037 to 0.0883 |
| scifact | 1-1 | 512 | held-out | reciprocal rank | 69 | 0.3753 | 0.4346 | -0.0593 | -0.0929 to -0.0286 |
| scifact | 1-1 | 512 | held-out | in top 10 | 69 | 0.8678 | 0.8309 | 0.0368 | -0.0042 to 0.0803 |
| scifact | 1-1 | 512 | held-out | in top 100 | 69 | 0.9783 | 0.9807 | -0.0024 | -0.0290 to 0.0242 |
| scifact | 1-1 | 512 | held-out | log rank (lower is better) | 69 | 1.3018 | 1.3197 | -0.0180 | -0.1273 to 0.0778 |
| scifact | 1-1 | 512 | held-out | worst rank in the set | 69 | 11.1413 | 11.4058 | -0.2645 | -3.3479 to 3.5985 |
| scifact | 1-1 | 512 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4904 | 0.5243 | -0.0339 | -0.0634 to -0.0044 |
| scifact | 1-1 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5652 | 0.5066 | 0.0585 | 0.0213 to 0.0957 |
| scifact | 1-1 | 614 | held-out | reciprocal rank | 69 | 0.3760 | 0.4346 | -0.0586 | -0.0902 to -0.0298 |
| scifact | 1-1 | 614 | held-out | in top 10 | 69 | 0.8641 | 0.8309 | 0.0332 | -0.0079 to 0.0767 |
| scifact | 1-1 | 614 | held-out | in top 100 | 69 | 0.9783 | 0.9807 | -0.0024 | -0.0290 to 0.0242 |
| scifact | 1-1 | 614 | held-out | log rank (lower is better) | 69 | 1.2983 | 1.3197 | -0.0215 | -0.1299 to 0.0700 |
| scifact | 1-1 | 614 | held-out | worst rank in the set | 69 | 10.4746 | 11.4058 | -0.9312 | -3.6143 to 1.9288 |
| scifact | 1-1 | 614 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4898 | 0.5243 | -0.0345 | -0.0618 to -0.0056 |
| scifact | 1-1 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5672 | 0.5066 | 0.0605 | 0.0258 to 0.0964 |
| scifact | 1-1 | 716 | held-out | reciprocal rank | 69 | 0.3759 | 0.4346 | -0.0587 | -0.0883 to -0.0316 |
| scifact | 1-1 | 716 | held-out | in top 10 | 69 | 0.8557 | 0.8309 | 0.0248 | -0.0127 to 0.0664 |
| scifact | 1-1 | 716 | held-out | in top 100 | 69 | 0.9783 | 0.9807 | -0.0024 | -0.0290 to 0.0242 |
| scifact | 1-1 | 716 | held-out | log rank (lower is better) | 69 | 1.3201 | 1.3197 | 0.0003 | -0.0900 to 0.0819 |
| scifact | 1-1 | 716 | held-out | worst rank in the set | 69 | 10.5386 | 11.4058 | -0.8671 | -3.2391 to 1.6450 |
| scifact | 1-1 | 716 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.4865 | 0.5243 | -0.0378 | -0.0631 to -0.0121 |
| scifact | 1-1 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5489 | 0.5066 | 0.0423 | 0.0168 to 0.0681 |
| scifact | 1-1 | 819 | held-out | reciprocal rank | 69 | 0.4028 | 0.4346 | -0.0318 | -0.0628 to -0.0041 |
| scifact | 1-1 | 819 | held-out | in top 10 | 69 | 0.8653 | 0.8309 | 0.0344 | -0.0000 to 0.0749 |
| scifact | 1-1 | 819 | held-out | in top 100 | 69 | 0.9831 | 0.9807 | 0.0024 | -0.0170 to 0.0242 |
| scifact | 1-1 | 819 | held-out | log rank (lower is better) | 69 | 1.2817 | 1.3197 | -0.0381 | -0.1174 to 0.0407 |
| scifact | 1-1 | 819 | held-out | worst rank in the set | 69 | 10.1395 | 11.4058 | -1.2663 | -3.2808 to 0.4470 |
| scifact | 1-1 | 819 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.5100 | 0.5243 | -0.0143 | -0.0416 to 0.0113 |
| scifact | 1-1 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5540 | 0.5066 | 0.0474 | 0.0207 to 0.0753 |
| scifact | 1-1 | 921 | held-out | reciprocal rank | 69 | 0.4109 | 0.4346 | -0.0237 | -0.0495 to -0.0010 |
| scifact | 1-1 | 921 | held-out | in top 10 | 69 | 0.8539 | 0.8309 | 0.0229 | -0.0048 to 0.0519 |
| scifact | 1-1 | 921 | held-out | in top 100 | 69 | 0.9903 | 0.9807 | 0.0097 | 0.0000 to 0.0266 |
| scifact | 1-1 | 921 | held-out | log rank (lower is better) | 69 | 1.2872 | 1.3197 | -0.0326 | -0.1020 to 0.0338 |
| scifact | 1-1 | 921 | held-out | worst rank in the set | 69 | 10.3593 | 11.4058 | -1.0465 | -2.3599 to -0.0071 |
| scifact | 1-1 | 921 | held-out | nDCG@10, restricted to held-out labels | 69 | 0.5129 | 0.5243 | -0.0114 | -0.0334 to 0.0092 |
| scifact | 1-1 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 69 | 0.5419 | 0.5066 | 0.0353 | 0.0139 to 0.0575 |
| scifact | 2-2 | 102 | held-out | reciprocal rank | 6 | 0.2633 | 0.3761 | -0.1128 | -0.1633 to -0.0633 |
| scifact | 2-2 | 102 | held-out | in top 10 | 6 | 0.9375 | 0.8854 | 0.0521 | 0.0000 to 0.1562 |
| scifact | 2-2 | 102 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 102 | held-out | log rank (lower is better) | 6 | 1.4698 | 1.2652 | 0.2046 | 0.0622 to 0.3324 |
| scifact | 2-2 | 102 | held-out | worst rank in the set | 6 | 6.1458 | 6.6389 | -0.4931 | -1.6837 to 0.4583 |
| scifact | 2-2 | 102 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5138 | 0.6031 | -0.0893 | -0.1523 to -0.0174 |
| scifact | 2-2 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5151 | 0.4889 | 0.0262 | -0.0147 to 0.0623 |
| scifact | 2-2 | 204 | held-out | reciprocal rank | 6 | 0.2693 | 0.3761 | -0.1068 | -0.1597 to -0.0514 |
| scifact | 2-2 | 204 | held-out | in top 10 | 6 | 0.9306 | 0.8854 | 0.0451 | 0.0000 to 0.1193 |
| scifact | 2-2 | 204 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 204 | held-out | log rank (lower is better) | 6 | 1.4565 | 1.2652 | 0.1913 | 0.0439 to 0.3084 |
| scifact | 2-2 | 204 | held-out | worst rank in the set | 6 | 6.1319 | 6.6389 | -0.5069 | -1.6944 to 0.3889 |
| scifact | 2-2 | 204 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5172 | 0.6031 | -0.0858 | -0.1477 to -0.0220 |
| scifact | 2-2 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5141 | 0.4889 | 0.0252 | -0.0195 to 0.0652 |
| scifact | 2-2 | 307 | held-out | reciprocal rank | 6 | 0.2836 | 0.3761 | -0.0925 | -0.1456 to -0.0432 |
| scifact | 2-2 | 307 | held-out | in top 10 | 6 | 0.9583 | 0.8854 | 0.0729 | 0.0000 to 0.1750 |
| scifact | 2-2 | 307 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 307 | held-out | log rank (lower is better) | 6 | 1.4269 | 1.2652 | 0.1617 | 0.0121 to 0.2736 |
| scifact | 2-2 | 307 | held-out | worst rank in the set | 6 | 6.0417 | 6.6389 | -0.5972 | -1.6683 to 0.2500 |
| scifact | 2-2 | 307 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5401 | 0.6031 | -0.0630 | -0.1317 to 0.0041 |
| scifact | 2-2 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5256 | 0.4889 | 0.0367 | -0.0103 to 0.0871 |
| scifact | 2-2 | 409 | held-out | reciprocal rank | 6 | 0.2931 | 0.3761 | -0.0830 | -0.1300 to -0.0388 |
| scifact | 2-2 | 409 | held-out | in top 10 | 6 | 0.9583 | 0.8854 | 0.0729 | 0.0000 to 0.1750 |
| scifact | 2-2 | 409 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 409 | held-out | log rank (lower is better) | 6 | 1.4002 | 1.2652 | 0.1350 | 0.0117 to 0.2431 |
| scifact | 2-2 | 409 | held-out | worst rank in the set | 6 | 5.8819 | 6.6389 | -0.7569 | -1.8958 to 0.0833 |
| scifact | 2-2 | 409 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5492 | 0.6031 | -0.0539 | -0.1124 to 0.0095 |
| scifact | 2-2 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5283 | 0.4889 | 0.0394 | -0.0046 to 0.0868 |
| scifact | 2-2 | 512 | held-out | reciprocal rank | 6 | 0.3057 | 0.3761 | -0.0704 | -0.1253 to -0.0279 |
| scifact | 2-2 | 512 | held-out | in top 10 | 6 | 0.9688 | 0.8854 | 0.0833 | 0.0000 to 0.2000 |
| scifact | 2-2 | 512 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 512 | held-out | log rank (lower is better) | 6 | 1.3750 | 1.2652 | 0.1098 | -0.0044 to 0.2117 |
| scifact | 2-2 | 512 | held-out | worst rank in the set | 6 | 5.9236 | 6.6389 | -0.7153 | -1.8200 to 0.1667 |
| scifact | 2-2 | 512 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5649 | 0.6031 | -0.0381 | -0.1072 to 0.0299 |
| scifact | 2-2 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5227 | 0.4889 | 0.0337 | -0.0108 to 0.0792 |
| scifact | 2-2 | 614 | held-out | reciprocal rank | 6 | 0.3218 | 0.3761 | -0.0543 | -0.0833 to -0.0278 |
| scifact | 2-2 | 614 | held-out | in top 10 | 6 | 0.9688 | 0.8854 | 0.0833 | 0.0000 to 0.2000 |
| scifact | 2-2 | 614 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 614 | held-out | log rank (lower is better) | 6 | 1.3427 | 1.2652 | 0.0775 | -0.0034 to 0.1477 |
| scifact | 2-2 | 614 | held-out | worst rank in the set | 6 | 5.9167 | 6.6389 | -0.7222 | -1.8056 to 0.1667 |
| scifact | 2-2 | 614 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5800 | 0.6031 | -0.0231 | -0.0684 to 0.0273 |
| scifact | 2-2 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5213 | 0.4889 | 0.0324 | -0.0120 to 0.0788 |
| scifact | 2-2 | 716 | held-out | reciprocal rank | 6 | 0.3256 | 0.3761 | -0.0506 | -0.0822 to -0.0208 |
| scifact | 2-2 | 716 | held-out | in top 10 | 6 | 0.9688 | 0.8854 | 0.0833 | 0.0000 to 0.2000 |
| scifact | 2-2 | 716 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 716 | held-out | log rank (lower is better) | 6 | 1.3298 | 1.2652 | 0.0646 | -0.0140 to 0.1321 |
| scifact | 2-2 | 716 | held-out | worst rank in the set | 6 | 5.9097 | 6.6389 | -0.7292 | -1.7100 to 0.0861 |
| scifact | 2-2 | 716 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5836 | 0.6031 | -0.0194 | -0.0671 to 0.0290 |
| scifact | 2-2 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5207 | 0.4889 | 0.0318 | -0.0038 to 0.0691 |
| scifact | 2-2 | 819 | held-out | reciprocal rank | 6 | 0.3338 | 0.3761 | -0.0423 | -0.0656 to -0.0180 |
| scifact | 2-2 | 819 | held-out | in top 10 | 6 | 0.9688 | 0.8854 | 0.0833 | 0.0000 to 0.2000 |
| scifact | 2-2 | 819 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 819 | held-out | log rank (lower is better) | 6 | 1.3030 | 1.2652 | 0.0378 | -0.0273 to 0.0916 |
| scifact | 2-2 | 819 | held-out | worst rank in the set | 6 | 5.7708 | 6.6389 | -0.8681 | -1.8448 to 0.0000 |
| scifact | 2-2 | 819 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5920 | 0.6031 | -0.0110 | -0.0494 to 0.0289 |
| scifact | 2-2 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5186 | 0.4889 | 0.0297 | -0.0013 to 0.0592 |
| scifact | 2-2 | 921 | held-out | reciprocal rank | 6 | 0.3404 | 0.3761 | -0.0357 | -0.0630 to -0.0084 |
| scifact | 2-2 | 921 | held-out | in top 10 | 6 | 0.9201 | 0.8854 | 0.0347 | 0.0000 to 0.0833 |
| scifact | 2-2 | 921 | held-out | in top 100 | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 2-2 | 921 | held-out | log rank (lower is better) | 6 | 1.2862 | 1.2652 | 0.0210 | -0.0373 to 0.0627 |
| scifact | 2-2 | 921 | held-out | worst rank in the set | 6 | 5.8889 | 6.6389 | -0.7500 | -1.5833 to -0.0235 |
| scifact | 2-2 | 921 | held-out | nDCG@10, restricted to held-out labels | 6 | 0.5822 | 0.6031 | -0.0208 | -0.0489 to 0.0048 |
| scifact | 2-2 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 6 | 0.5199 | 0.4889 | 0.0310 | 0.0047 to 0.0559 |
| nfcorpus | prefix | 102 | all known positives | reciprocal rank | 2162 | 0.0218 | 0.0285 | -0.0067 | -0.0074 to -0.0060 |
| nfcorpus | prefix | 102 | all known positives | in top 10 | 2162 | 0.0450 | 0.0603 | -0.0153 | -0.0172 to -0.0137 |
| nfcorpus | prefix | 102 | all known positives | in top 100 | 2162 | 0.1576 | 0.2023 | -0.0447 | -0.0488 to -0.0405 |
| nfcorpus | prefix | 102 | all known positives | log rank (lower is better) | 2162 | 6.3064 | 6.0734 | 0.2329 | 0.2149 to 0.2512 |
| nfcorpus | prefix | 102 | all known positives | worst rank in the set | 2162 | 3363.0166 | 3317.4008 | 45.6158 | 34.7354 to 57.0948 |
| nfcorpus | prefix | 204 | all known positives | reciprocal rank | 2162 | 0.0244 | 0.0285 | -0.0041 | -0.0046 to -0.0037 |
| nfcorpus | prefix | 204 | all known positives | in top 10 | 2162 | 0.0506 | 0.0603 | -0.0097 | -0.0110 to -0.0085 |
| nfcorpus | prefix | 204 | all known positives | in top 100 | 2162 | 0.1736 | 0.2023 | -0.0287 | -0.0317 to -0.0258 |
| nfcorpus | prefix | 204 | all known positives | log rank (lower is better) | 2162 | 6.2287 | 6.0734 | 0.1552 | 0.1436 to 0.1676 |
| nfcorpus | prefix | 204 | all known positives | worst rank in the set | 2162 | 3348.1410 | 3317.4008 | 30.7402 | 21.2146 to 40.5153 |
| nfcorpus | prefix | 307 | all known positives | reciprocal rank | 2162 | 0.0258 | 0.0285 | -0.0027 | -0.0031 to -0.0024 |
| nfcorpus | prefix | 307 | all known positives | in top 10 | 2162 | 0.0536 | 0.0603 | -0.0067 | -0.0077 to -0.0058 |
| nfcorpus | prefix | 307 | all known positives | in top 100 | 2162 | 0.1833 | 0.2023 | -0.0190 | -0.0213 to -0.0168 |
| nfcorpus | prefix | 307 | all known positives | log rank (lower is better) | 2162 | 6.1799 | 6.0734 | 0.1064 | 0.0977 to 0.1158 |
| nfcorpus | prefix | 307 | all known positives | worst rank in the set | 2162 | 3343.0678 | 3317.4008 | 25.6670 | 17.8100 to 34.4445 |
| nfcorpus | prefix | 409 | all known positives | reciprocal rank | 2162 | 0.0268 | 0.0285 | -0.0017 | -0.0020 to -0.0014 |
| nfcorpus | prefix | 409 | all known positives | in top 10 | 2162 | 0.0558 | 0.0603 | -0.0045 | -0.0054 to -0.0037 |
| nfcorpus | prefix | 409 | all known positives | in top 100 | 2162 | 0.1902 | 0.2023 | -0.0121 | -0.0138 to -0.0103 |
| nfcorpus | prefix | 409 | all known positives | log rank (lower is better) | 2162 | 6.1488 | 6.0734 | 0.0753 | 0.0689 to 0.0819 |
| nfcorpus | prefix | 409 | all known positives | worst rank in the set | 2162 | 3343.4707 | 3317.4008 | 26.0699 | 18.6435 to 34.3292 |
| nfcorpus | prefix | 512 | all known positives | reciprocal rank | 2162 | 0.0273 | 0.0285 | -0.0011 | -0.0014 to -0.0009 |
| nfcorpus | prefix | 512 | all known positives | in top 10 | 2162 | 0.0573 | 0.0603 | -0.0030 | -0.0037 to -0.0024 |
| nfcorpus | prefix | 512 | all known positives | in top 100 | 2162 | 0.1946 | 0.2023 | -0.0077 | -0.0091 to -0.0063 |
| nfcorpus | prefix | 512 | all known positives | log rank (lower is better) | 2162 | 6.1186 | 6.0734 | 0.0452 | 0.0404 to 0.0501 |
| nfcorpus | prefix | 512 | all known positives | worst rank in the set | 2162 | 3329.4726 | 3317.4008 | 12.0718 | 5.7202 to 18.8323 |
| nfcorpus | prefix | 614 | all known positives | reciprocal rank | 2162 | 0.0278 | 0.0285 | -0.0007 | -0.0009 to -0.0005 |
| nfcorpus | prefix | 614 | all known positives | in top 10 | 2162 | 0.0582 | 0.0603 | -0.0021 | -0.0027 to -0.0015 |
| nfcorpus | prefix | 614 | all known positives | in top 100 | 2162 | 0.1974 | 0.2023 | -0.0049 | -0.0060 to -0.0036 |
| nfcorpus | prefix | 614 | all known positives | log rank (lower is better) | 2162 | 6.1027 | 6.0734 | 0.0292 | 0.0253 to 0.0334 |
| nfcorpus | prefix | 614 | all known positives | worst rank in the set | 2162 | 3323.9060 | 3317.4008 | 6.5052 | 0.6970 to 12.1659 |
| nfcorpus | prefix | 716 | all known positives | reciprocal rank | 2162 | 0.0280 | 0.0285 | -0.0004 | -0.0006 to -0.0003 |
| nfcorpus | prefix | 716 | all known positives | in top 10 | 2162 | 0.0588 | 0.0603 | -0.0015 | -0.0021 to -0.0010 |
| nfcorpus | prefix | 716 | all known positives | in top 100 | 2162 | 0.1983 | 0.2023 | -0.0040 | -0.0051 to -0.0030 |
| nfcorpus | prefix | 716 | all known positives | log rank (lower is better) | 2162 | 6.0976 | 6.0734 | 0.0242 | 0.0209 to 0.0276 |
| nfcorpus | prefix | 716 | all known positives | worst rank in the set | 2162 | 3324.4522 | 3317.4008 | 7.0514 | 2.2258 to 11.8404 |
| nfcorpus | prefix | 819 | all known positives | reciprocal rank | 2162 | 0.0282 | 0.0285 | -0.0003 | -0.0004 to -0.0002 |
| nfcorpus | prefix | 819 | all known positives | in top 10 | 2162 | 0.0594 | 0.0603 | -0.0009 | -0.0013 to -0.0004 |
| nfcorpus | prefix | 819 | all known positives | in top 100 | 2162 | 0.2001 | 0.2023 | -0.0022 | -0.0031 to -0.0013 |
| nfcorpus | prefix | 819 | all known positives | log rank (lower is better) | 2162 | 6.0881 | 6.0734 | 0.0146 | 0.0120 to 0.0171 |
| nfcorpus | prefix | 819 | all known positives | worst rank in the set | 2162 | 3323.0367 | 3317.4008 | 5.6359 | 1.5526 to 9.6895 |
| nfcorpus | prefix | 921 | all known positives | reciprocal rank | 2162 | 0.0284 | 0.0285 | -0.0001 | -0.0002 to -0.0000 |
| nfcorpus | prefix | 921 | all known positives | in top 10 | 2162 | 0.0596 | 0.0603 | -0.0007 | -0.0011 to -0.0004 |
| nfcorpus | prefix | 921 | all known positives | in top 100 | 2162 | 0.2014 | 0.2023 | -0.0009 | -0.0016 to -0.0002 |
| nfcorpus | prefix | 921 | all known positives | log rank (lower is better) | 2162 | 6.0821 | 6.0734 | 0.0087 | 0.0068 to 0.0106 |
| nfcorpus | prefix | 921 | all known positives | worst rank in the set | 2162 | 3323.3604 | 3317.4008 | 5.9596 | 2.9188 to 9.1619 |
| nfcorpus | shared | 102 | all known positives | reciprocal rank | 2162 | 0.0403 | 0.0285 | 0.0118 | 0.0106 to 0.0132 |
| nfcorpus | shared | 102 | all known positives | in top 10 | 2162 | 0.0870 | 0.0603 | 0.0267 | 0.0238 to 0.0300 |
| nfcorpus | shared | 102 | all known positives | in top 100 | 2162 | 0.2693 | 0.2023 | 0.0670 | 0.0597 to 0.0751 |
| nfcorpus | shared | 102 | all known positives | log rank (lower is better) | 2162 | 5.6656 | 6.0734 | -0.4078 | -0.4516 to -0.3665 |
| nfcorpus | shared | 102 | all known positives | worst rank in the set | 2162 | 3071.7906 | 3317.4008 | -245.6102 | -274.4568 to -218.9543 |
| nfcorpus | shared | 204 | all known positives | reciprocal rank | 2162 | 0.0415 | 0.0285 | 0.0130 | 0.0117 to 0.0145 |
| nfcorpus | shared | 204 | all known positives | in top 10 | 2162 | 0.0905 | 0.0603 | 0.0302 | 0.0272 to 0.0338 |
| nfcorpus | shared | 204 | all known positives | in top 100 | 2162 | 0.2796 | 0.2023 | 0.0773 | 0.0694 to 0.0860 |
| nfcorpus | shared | 204 | all known positives | log rank (lower is better) | 2162 | 5.6153 | 6.0734 | -0.4581 | -0.5040 to -0.4149 |
| nfcorpus | shared | 204 | all known positives | worst rank in the set | 2162 | 3043.9517 | 3317.4008 | -273.4491 | -304.2653 to -245.1040 |
| nfcorpus | shared | 307 | all known positives | reciprocal rank | 2162 | 0.0412 | 0.0285 | 0.0127 | 0.0114 to 0.0141 |
| nfcorpus | shared | 307 | all known positives | in top 10 | 2162 | 0.0900 | 0.0603 | 0.0297 | 0.0266 to 0.0331 |
| nfcorpus | shared | 307 | all known positives | in top 100 | 2162 | 0.2796 | 0.2023 | 0.0773 | 0.0697 to 0.0857 |
| nfcorpus | shared | 307 | all known positives | log rank (lower is better) | 2162 | 5.6113 | 6.0734 | -0.4621 | -0.5065 to -0.4201 |
| nfcorpus | shared | 307 | all known positives | worst rank in the set | 2162 | 3048.1566 | 3317.4008 | -269.2442 | -299.1964 to -241.2139 |
| nfcorpus | shared | 409 | all known positives | reciprocal rank | 2162 | 0.0406 | 0.0285 | 0.0121 | 0.0109 to 0.0134 |
| nfcorpus | shared | 409 | all known positives | in top 10 | 2162 | 0.0883 | 0.0603 | 0.0280 | 0.0252 to 0.0312 |
| nfcorpus | shared | 409 | all known positives | in top 100 | 2162 | 0.2774 | 0.2023 | 0.0752 | 0.0676 to 0.0833 |
| nfcorpus | shared | 409 | all known positives | log rank (lower is better) | 2162 | 5.6221 | 6.0734 | -0.4514 | -0.4943 to -0.4106 |
| nfcorpus | shared | 409 | all known positives | worst rank in the set | 2162 | 3050.7097 | 3317.4008 | -266.6911 | -295.9031 to -239.3765 |
| nfcorpus | shared | 512 | all known positives | reciprocal rank | 2162 | 0.0399 | 0.0285 | 0.0114 | 0.0103 to 0.0127 |
| nfcorpus | shared | 512 | all known positives | in top 10 | 2162 | 0.0866 | 0.0603 | 0.0263 | 0.0236 to 0.0294 |
| nfcorpus | shared | 512 | all known positives | in top 100 | 2162 | 0.2740 | 0.2023 | 0.0717 | 0.0645 to 0.0797 |
| nfcorpus | shared | 512 | all known positives | log rank (lower is better) | 2162 | 5.6402 | 6.0734 | -0.4332 | -0.4743 to -0.3945 |
| nfcorpus | shared | 512 | all known positives | worst rank in the set | 2162 | 3059.8317 | 3317.4008 | -257.5691 | -286.0957 to -230.5907 |
| nfcorpus | shared | 614 | all known positives | reciprocal rank | 2162 | 0.0393 | 0.0285 | 0.0108 | 0.0097 to 0.0120 |
| nfcorpus | shared | 614 | all known positives | in top 10 | 2162 | 0.0849 | 0.0603 | 0.0246 | 0.0221 to 0.0275 |
| nfcorpus | shared | 614 | all known positives | in top 100 | 2162 | 0.2708 | 0.2023 | 0.0685 | 0.0615 to 0.0759 |
| nfcorpus | shared | 614 | all known positives | log rank (lower is better) | 2162 | 5.6596 | 6.0734 | -0.4138 | -0.4529 to -0.3767 |
| nfcorpus | shared | 614 | all known positives | worst rank in the set | 2162 | 3068.6876 | 3317.4008 | -248.7132 | -276.4058 to -222.5673 |
| nfcorpus | shared | 716 | all known positives | reciprocal rank | 2162 | 0.0386 | 0.0285 | 0.0101 | 0.0091 to 0.0112 |
| nfcorpus | shared | 716 | all known positives | in top 10 | 2162 | 0.0830 | 0.0603 | 0.0227 | 0.0203 to 0.0254 |
| nfcorpus | shared | 716 | all known positives | in top 100 | 2162 | 0.2671 | 0.2023 | 0.0648 | 0.0584 to 0.0719 |
| nfcorpus | shared | 716 | all known positives | log rank (lower is better) | 2162 | 5.6795 | 6.0734 | -0.3940 | -0.4312 to -0.3585 |
| nfcorpus | shared | 716 | all known positives | worst rank in the set | 2162 | 3079.3422 | 3317.4008 | -238.0586 | -264.6044 to -213.1889 |
| nfcorpus | shared | 819 | all known positives | reciprocal rank | 2162 | 0.0369 | 0.0285 | 0.0084 | 0.0076 to 0.0093 |
| nfcorpus | shared | 819 | all known positives | in top 10 | 2162 | 0.0789 | 0.0603 | 0.0186 | 0.0166 to 0.0209 |
| nfcorpus | shared | 819 | all known positives | in top 100 | 2162 | 0.2592 | 0.2023 | 0.0569 | 0.0512 to 0.0632 |
| nfcorpus | shared | 819 | all known positives | log rank (lower is better) | 2162 | 5.7245 | 6.0734 | -0.3490 | -0.3821 to -0.3181 |
| nfcorpus | shared | 819 | all known positives | worst rank in the set | 2162 | 3102.6301 | 3317.4008 | -214.7707 | -238.6112 to -192.2563 |
| nfcorpus | shared | 921 | all known positives | reciprocal rank | 2162 | 0.0338 | 0.0285 | 0.0053 | 0.0047 to 0.0059 |
| nfcorpus | shared | 921 | all known positives | in top 10 | 2162 | 0.0720 | 0.0603 | 0.0117 | 0.0104 to 0.0132 |
| nfcorpus | shared | 921 | all known positives | in top 100 | 2162 | 0.2404 | 0.2023 | 0.0381 | 0.0343 to 0.0425 |
| nfcorpus | shared | 921 | all known positives | log rank (lower is better) | 2162 | 5.8251 | 6.0734 | -0.2484 | -0.2720 to -0.2264 |
| nfcorpus | shared | 921 | all known positives | worst rank in the set | 2162 | 3160.7865 | 3317.4008 | -156.6143 | -174.4626 to -139.9026 |
| nfcorpus | own | 102 | the mask's own positive | reciprocal rank | 2162 | 0.9310 | 0.0717 | 0.8593 | 0.8533 to 0.8652 |
| nfcorpus | own | 102 | the mask's own positive | in top 10 | 2162 | 0.9994 | 0.1400 | 0.8595 | 0.8520 to 0.8671 |
| nfcorpus | own | 102 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7005 |
| nfcorpus | own | 102 | the mask's own positive | log rank (lower is better) | 2162 | 0.1082 | 5.4229 | -5.3147 | -5.3703 to -5.2598 |
| nfcorpus | own | 102 | the mask's own positive | worst rank in the set | 2162 | 1.1999 | 902.8887 | -901.6888 | -924.4317 to -880.5645 |
| nfcorpus | own | 204 | the mask's own positive | reciprocal rank | 2162 | 0.9148 | 0.0717 | 0.8431 | 0.8364 to 0.8499 |
| nfcorpus | own | 204 | the mask's own positive | in top 10 | 2162 | 0.9993 | 0.1400 | 0.8593 | 0.8519 to 0.8670 |
| nfcorpus | own | 204 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7005 |
| nfcorpus | own | 204 | the mask's own positive | log rank (lower is better) | 2162 | 0.1351 | 5.4229 | -5.2878 | -5.3452 to -5.2323 |
| nfcorpus | own | 204 | the mask's own positive | worst rank in the set | 2162 | 1.2536 | 902.8887 | -901.6351 | -924.3833 to -880.5129 |
| nfcorpus | own | 307 | the mask's own positive | reciprocal rank | 2162 | 0.8808 | 0.0717 | 0.8091 | 0.8014 to 0.8169 |
| nfcorpus | own | 307 | the mask's own positive | in top 10 | 2162 | 0.9978 | 0.1400 | 0.8578 | 0.8503 to 0.8654 |
| nfcorpus | own | 307 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7005 |
| nfcorpus | own | 307 | the mask's own positive | log rank (lower is better) | 2162 | 0.1951 | 5.4229 | -5.2278 | -5.2856 to -5.1720 |
| nfcorpus | own | 307 | the mask's own positive | worst rank in the set | 2162 | 1.3933 | 902.8887 | -901.4955 | -924.2455 to -880.3795 |
| nfcorpus | own | 409 | the mask's own positive | reciprocal rank | 2162 | 0.8386 | 0.0717 | 0.7669 | 0.7580 to 0.7757 |
| nfcorpus | own | 409 | the mask's own positive | in top 10 | 2162 | 0.9943 | 0.1400 | 0.8544 | 0.8469 to 0.8619 |
| nfcorpus | own | 409 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7005 |
| nfcorpus | own | 409 | the mask's own positive | log rank (lower is better) | 2162 | 0.2750 | 5.4229 | -5.1479 | -5.2067 to -5.0913 |
| nfcorpus | own | 409 | the mask's own positive | worst rank in the set | 2162 | 1.6089 | 902.8887 | -901.2799 | -924.0270 to -880.1696 |
| nfcorpus | own | 512 | the mask's own positive | reciprocal rank | 2162 | 0.7898 | 0.0717 | 0.7181 | 0.7083 to 0.7277 |
| nfcorpus | own | 512 | the mask's own positive | in top 10 | 2162 | 0.9872 | 0.1400 | 0.8473 | 0.8395 to 0.8550 |
| nfcorpus | own | 512 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7004 |
| nfcorpus | own | 512 | the mask's own positive | log rank (lower is better) | 2162 | 0.3740 | 5.4229 | -5.0489 | -5.1088 to -4.9916 |
| nfcorpus | own | 512 | the mask's own positive | worst rank in the set | 2162 | 1.9154 | 902.8887 | -900.9733 | -923.7205 to -879.8652 |
| nfcorpus | own | 614 | the mask's own positive | reciprocal rank | 2162 | 0.7461 | 0.0717 | 0.6744 | 0.6640 to 0.6847 |
| nfcorpus | own | 614 | the mask's own positive | in top 10 | 2162 | 0.9771 | 0.1400 | 0.8371 | 0.8291 to 0.8452 |
| nfcorpus | own | 614 | the mask's own positive | in top 100 | 2162 | 1.0000 | 0.3104 | 0.6896 | 0.6789 to 0.7004 |
| nfcorpus | own | 614 | the mask's own positive | log rank (lower is better) | 2162 | 0.4693 | 5.4229 | -4.9536 | -5.0138 to -4.8963 |
| nfcorpus | own | 614 | the mask's own positive | worst rank in the set | 2162 | 2.2617 | 902.8887 | -900.6270 | -923.3780 to -879.5234 |
| nfcorpus | own | 716 | the mask's own positive | reciprocal rank | 2162 | 0.6726 | 0.0717 | 0.6009 | 0.5903 to 0.6114 |
| nfcorpus | own | 716 | the mask's own positive | in top 10 | 2162 | 0.9472 | 0.1400 | 0.8073 | 0.7987 to 0.8160 |
| nfcorpus | own | 716 | the mask's own positive | in top 100 | 2162 | 0.9998 | 0.3104 | 0.6895 | 0.6787 to 0.7003 |
| nfcorpus | own | 716 | the mask's own positive | log rank (lower is better) | 2162 | 0.6513 | 5.4229 | -4.7716 | -4.8320 to -4.7137 |
| nfcorpus | own | 716 | the mask's own positive | worst rank in the set | 2162 | 3.1552 | 902.8887 | -899.7335 | -922.4990 to -878.6409 |
| nfcorpus | own | 819 | the mask's own positive | reciprocal rank | 2162 | 0.5192 | 0.0717 | 0.4475 | 0.4373 to 0.4578 |
| nfcorpus | own | 819 | the mask's own positive | in top 10 | 2162 | 0.8309 | 0.1400 | 0.6910 | 0.6810 to 0.7017 |
| nfcorpus | own | 819 | the mask's own positive | in top 100 | 2162 | 0.9935 | 0.3104 | 0.6831 | 0.6725 to 0.6939 |
| nfcorpus | own | 819 | the mask's own positive | log rank (lower is better) | 2162 | 1.1394 | 5.4229 | -4.2835 | -4.3424 to -4.2264 |
| nfcorpus | own | 819 | the mask's own positive | worst rank in the set | 2162 | 7.6777 | 902.8887 | -895.2110 | -917.9554 to -874.2050 |
| nfcorpus | own | 921 | the mask's own positive | reciprocal rank | 2162 | 0.2744 | 0.0717 | 0.2027 | 0.1963 to 0.2093 |
| nfcorpus | own | 921 | the mask's own positive | in top 10 | 2162 | 0.5164 | 0.1400 | 0.3765 | 0.3662 to 0.3871 |
| nfcorpus | own | 921 | the mask's own positive | in top 100 | 2162 | 0.8757 | 0.3104 | 0.5653 | 0.5543 to 0.5760 |
| nfcorpus | own | 921 | the mask's own positive | log rank (lower is better) | 2162 | 2.3990 | 5.4229 | -3.0239 | -3.0711 to -2.9787 |
| nfcorpus | own | 921 | the mask's own positive | worst rank in the set | 2162 | 48.9378 | 902.8887 | -853.9509 | -875.7315 to -833.7007 |
| nfcorpus | 1-1 | 102 | held-out | reciprocal rank | 2162 | 0.0664 | 0.0734 | -0.0070 | -0.0105 to -0.0037 |
| nfcorpus | 1-1 | 102 | held-out | in top 10 | 2162 | 0.1798 | 0.1440 | 0.0358 | 0.0289 to 0.0425 |
| nfcorpus | 1-1 | 102 | held-out | in top 100 | 2162 | 0.3998 | 0.3132 | 0.0866 | 0.0772 to 0.0955 |
| nfcorpus | 1-1 | 102 | held-out | log rank (lower is better) | 2162 | 4.9802 | 5.3993 | -0.4191 | -0.4582 to -0.3790 |
| nfcorpus | 1-1 | 102 | held-out | worst rank in the set | 2162 | 701.8894 | 896.8044 | -194.9150 | -212.6802 to -176.8685 |
| nfcorpus | 1-1 | 102 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0859 | 0.0843 | 0.0016 | -0.0024 to 0.0053 |
| nfcorpus | 1-1 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.0988 | 0.0765 | 0.0222 | 0.0182 to 0.0263 |
| nfcorpus | 1-1 | 204 | held-out | reciprocal rank | 2162 | 0.0687 | 0.0734 | -0.0047 | -0.0080 to -0.0017 |
| nfcorpus | 1-1 | 204 | held-out | in top 10 | 2162 | 0.1902 | 0.1440 | 0.0461 | 0.0396 to 0.0526 |
| nfcorpus | 1-1 | 204 | held-out | in top 100 | 2162 | 0.4161 | 0.3132 | 0.1029 | 0.0938 to 0.1117 |
| nfcorpus | 1-1 | 204 | held-out | log rank (lower is better) | 2162 | 4.9081 | 5.3993 | -0.4912 | -0.5283 to -0.4525 |
| nfcorpus | 1-1 | 204 | held-out | worst rank in the set | 2162 | 675.5328 | 896.8044 | -221.2717 | -238.3804 to -204.2119 |
| nfcorpus | 1-1 | 204 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0900 | 0.0843 | 0.0057 | 0.0020 to 0.0093 |
| nfcorpus | 1-1 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1011 | 0.0765 | 0.0246 | 0.0210 to 0.0283 |
| nfcorpus | 1-1 | 307 | held-out | reciprocal rank | 2162 | 0.0713 | 0.0734 | -0.0021 | -0.0053 to 0.0009 |
| nfcorpus | 1-1 | 307 | held-out | in top 10 | 2162 | 0.1910 | 0.1440 | 0.0469 | 0.0406 to 0.0533 |
| nfcorpus | 1-1 | 307 | held-out | in top 100 | 2162 | 0.4198 | 0.3132 | 0.1065 | 0.0975 to 0.1154 |
| nfcorpus | 1-1 | 307 | held-out | log rank (lower is better) | 2162 | 4.8886 | 5.3993 | -0.5107 | -0.5459 to -0.4738 |
| nfcorpus | 1-1 | 307 | held-out | worst rank in the set | 2162 | 668.7993 | 896.8044 | -228.0051 | -244.3527 to -211.4747 |
| nfcorpus | 1-1 | 307 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0922 | 0.0843 | 0.0079 | 0.0044 to 0.0112 |
| nfcorpus | 1-1 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1030 | 0.0765 | 0.0264 | 0.0230 to 0.0301 |
| nfcorpus | 1-1 | 409 | held-out | reciprocal rank | 2162 | 0.0737 | 0.0734 | 0.0003 | -0.0028 to 0.0032 |
| nfcorpus | 1-1 | 409 | held-out | in top 10 | 2162 | 0.1901 | 0.1440 | 0.0461 | 0.0400 to 0.0518 |
| nfcorpus | 1-1 | 409 | held-out | in top 100 | 2162 | 0.4181 | 0.3132 | 0.1048 | 0.0960 to 0.1136 |
| nfcorpus | 1-1 | 409 | held-out | log rank (lower is better) | 2162 | 4.8845 | 5.3993 | -0.5148 | -0.5492 to -0.4790 |
| nfcorpus | 1-1 | 409 | held-out | worst rank in the set | 2162 | 668.1379 | 896.8044 | -228.6665 | -244.9298 to -212.2639 |
| nfcorpus | 1-1 | 409 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0938 | 0.0843 | 0.0094 | 0.0062 to 0.0126 |
| nfcorpus | 1-1 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1037 | 0.0765 | 0.0272 | 0.0239 to 0.0307 |
| nfcorpus | 1-1 | 512 | held-out | reciprocal rank | 2162 | 0.0754 | 0.0734 | 0.0021 | -0.0008 to 0.0049 |
| nfcorpus | 1-1 | 512 | held-out | in top 10 | 2162 | 0.1877 | 0.1440 | 0.0437 | 0.0375 to 0.0496 |
| nfcorpus | 1-1 | 512 | held-out | in top 100 | 2162 | 0.4168 | 0.3132 | 0.1035 | 0.0947 to 0.1120 |
| nfcorpus | 1-1 | 512 | held-out | log rank (lower is better) | 2162 | 4.8886 | 5.3993 | -0.5107 | -0.5443 to -0.4758 |
| nfcorpus | 1-1 | 512 | held-out | worst rank in the set | 2162 | 668.1020 | 896.8044 | -228.7025 | -244.5456 to -212.5868 |
| nfcorpus | 1-1 | 512 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0945 | 0.0843 | 0.0102 | 0.0069 to 0.0133 |
| nfcorpus | 1-1 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1029 | 0.0765 | 0.0264 | 0.0232 to 0.0297 |
| nfcorpus | 1-1 | 614 | held-out | reciprocal rank | 2162 | 0.0768 | 0.0734 | 0.0035 | 0.0006 to 0.0060 |
| nfcorpus | 1-1 | 614 | held-out | in top 10 | 2162 | 0.1862 | 0.1440 | 0.0422 | 0.0364 to 0.0479 |
| nfcorpus | 1-1 | 614 | held-out | in top 100 | 2162 | 0.4147 | 0.3132 | 0.1014 | 0.0930 to 0.1098 |
| nfcorpus | 1-1 | 614 | held-out | log rank (lower is better) | 2162 | 4.8957 | 5.3993 | -0.5036 | -0.5365 to -0.4695 |
| nfcorpus | 1-1 | 614 | held-out | worst rank in the set | 2162 | 669.3998 | 896.8044 | -227.4046 | -243.0426 to -211.6963 |
| nfcorpus | 1-1 | 614 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0952 | 0.0843 | 0.0109 | 0.0078 to 0.0139 |
| nfcorpus | 1-1 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1021 | 0.0765 | 0.0256 | 0.0224 to 0.0289 |
| nfcorpus | 1-1 | 716 | held-out | reciprocal rank | 2162 | 0.0778 | 0.0734 | 0.0045 | 0.0019 to 0.0070 |
| nfcorpus | 1-1 | 716 | held-out | in top 10 | 2162 | 0.1835 | 0.1440 | 0.0395 | 0.0339 to 0.0448 |
| nfcorpus | 1-1 | 716 | held-out | in top 100 | 2162 | 0.4107 | 0.3132 | 0.0974 | 0.0889 to 0.1054 |
| nfcorpus | 1-1 | 716 | held-out | log rank (lower is better) | 2162 | 4.9141 | 5.3993 | -0.4851 | -0.5162 to -0.4530 |
| nfcorpus | 1-1 | 716 | held-out | worst rank in the set | 2162 | 674.2873 | 896.8044 | -222.5171 | -237.4436 to -207.2567 |
| nfcorpus | 1-1 | 716 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0954 | 0.0843 | 0.0111 | 0.0082 to 0.0140 |
| nfcorpus | 1-1 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.1001 | 0.0765 | 0.0236 | 0.0207 to 0.0267 |
| nfcorpus | 1-1 | 819 | held-out | reciprocal rank | 2162 | 0.0790 | 0.0734 | 0.0057 | 0.0033 to 0.0080 |
| nfcorpus | 1-1 | 819 | held-out | in top 10 | 2162 | 0.1768 | 0.1440 | 0.0328 | 0.0279 to 0.0376 |
| nfcorpus | 1-1 | 819 | held-out | in top 100 | 2162 | 0.3989 | 0.3132 | 0.0857 | 0.0781 to 0.0930 |
| nfcorpus | 1-1 | 819 | held-out | log rank (lower is better) | 2162 | 4.9615 | 5.3993 | -0.4377 | -0.4656 to -0.4095 |
| nfcorpus | 1-1 | 819 | held-out | worst rank in the set | 2162 | 688.3534 | 896.8044 | -208.4511 | -222.0334 to -194.8095 |
| nfcorpus | 1-1 | 819 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0949 | 0.0843 | 0.0106 | 0.0081 to 0.0133 |
| nfcorpus | 1-1 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.0959 | 0.0765 | 0.0194 | 0.0167 to 0.0222 |
| nfcorpus | 1-1 | 921 | held-out | reciprocal rank | 2162 | 0.0792 | 0.0734 | 0.0058 | 0.0041 to 0.0077 |
| nfcorpus | 1-1 | 921 | held-out | in top 10 | 2162 | 0.1630 | 0.1440 | 0.0190 | 0.0151 to 0.0227 |
| nfcorpus | 1-1 | 921 | held-out | in top 100 | 2162 | 0.3711 | 0.3132 | 0.0579 | 0.0516 to 0.0641 |
| nfcorpus | 1-1 | 921 | held-out | log rank (lower is better) | 2162 | 5.0824 | 5.3993 | -0.3169 | -0.3359 to -0.2981 |
| nfcorpus | 1-1 | 921 | held-out | worst rank in the set | 2162 | 729.4013 | 896.8044 | -167.4031 | -178.2234 to -156.5633 |
| nfcorpus | 1-1 | 921 | held-out | nDCG@10, restricted to held-out labels | 2162 | 0.0922 | 0.0843 | 0.0079 | 0.0060 to 0.0099 |
| nfcorpus | 1-1 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 2162 | 0.0890 | 0.0765 | 0.0125 | 0.0106 to 0.0145 |
| nfcorpus | 2-2 | 102 | held-out | reciprocal rank | 1922 | 0.0588 | 0.0560 | 0.0027 | 0.0004 to 0.0050 |
| nfcorpus | 2-2 | 102 | held-out | in top 10 | 1922 | 0.1523 | 0.1154 | 0.0369 | 0.0313 to 0.0426 |
| nfcorpus | 2-2 | 102 | held-out | in top 100 | 1922 | 0.3955 | 0.2938 | 0.1017 | 0.0929 to 0.1099 |
| nfcorpus | 2-2 | 102 | held-out | log rank (lower is better) | 1922 | 5.0398 | 5.5554 | -0.5157 | -0.5521 to -0.4773 |
| nfcorpus | 2-2 | 102 | held-out | worst rank in the set | 1922 | 1070.8496 | 1394.9434 | -324.0938 | -346.7292 to -300.0942 |
| nfcorpus | 2-2 | 102 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0888 | 0.0786 | 0.0103 | 0.0068 to 0.0137 |
| nfcorpus | 2-2 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0803 | 0.0590 | 0.0213 | 0.0185 to 0.0242 |
| nfcorpus | 2-2 | 204 | held-out | reciprocal rank | 1922 | 0.0620 | 0.0560 | 0.0060 | 0.0038 to 0.0081 |
| nfcorpus | 2-2 | 204 | held-out | in top 10 | 1922 | 0.1609 | 0.1154 | 0.0455 | 0.0401 to 0.0510 |
| nfcorpus | 2-2 | 204 | held-out | in top 100 | 1922 | 0.4091 | 0.2938 | 0.1153 | 0.1070 to 0.1235 |
| nfcorpus | 2-2 | 204 | held-out | log rank (lower is better) | 1922 | 4.9699 | 5.5554 | -0.5856 | -0.6211 to -0.5483 |
| nfcorpus | 2-2 | 204 | held-out | worst rank in the set | 1922 | 1038.0774 | 1394.9434 | -356.8660 | -378.8057 to -334.0155 |
| nfcorpus | 2-2 | 204 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0942 | 0.0786 | 0.0156 | 0.0122 to 0.0189 |
| nfcorpus | 2-2 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0839 | 0.0590 | 0.0249 | 0.0222 to 0.0276 |
| nfcorpus | 2-2 | 307 | held-out | reciprocal rank | 1922 | 0.0637 | 0.0560 | 0.0076 | 0.0056 to 0.0097 |
| nfcorpus | 2-2 | 307 | held-out | in top 10 | 1922 | 0.1607 | 0.1154 | 0.0454 | 0.0400 to 0.0507 |
| nfcorpus | 2-2 | 307 | held-out | in top 100 | 1922 | 0.4108 | 0.2938 | 0.1170 | 0.1088 to 0.1249 |
| nfcorpus | 2-2 | 307 | held-out | log rank (lower is better) | 1922 | 4.9562 | 5.5554 | -0.5992 | -0.6333 to -0.5631 |
| nfcorpus | 2-2 | 307 | held-out | worst rank in the set | 1922 | 1028.7894 | 1394.9434 | -366.1540 | -387.2588 to -344.2130 |
| nfcorpus | 2-2 | 307 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0957 | 0.0786 | 0.0171 | 0.0140 to 0.0203 |
| nfcorpus | 2-2 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0843 | 0.0590 | 0.0253 | 0.0227 to 0.0279 |
| nfcorpus | 2-2 | 409 | held-out | reciprocal rank | 1922 | 0.0646 | 0.0560 | 0.0086 | 0.0066 to 0.0106 |
| nfcorpus | 2-2 | 409 | held-out | in top 10 | 1922 | 0.1617 | 0.1154 | 0.0463 | 0.0411 to 0.0514 |
| nfcorpus | 2-2 | 409 | held-out | in top 100 | 1922 | 0.4102 | 0.2938 | 0.1163 | 0.1084 to 0.1241 |
| nfcorpus | 2-2 | 409 | held-out | log rank (lower is better) | 1922 | 4.9601 | 5.5554 | -0.5954 | -0.6284 to -0.5608 |
| nfcorpus | 2-2 | 409 | held-out | worst rank in the set | 1922 | 1031.2851 | 1394.9434 | -363.6583 | -385.1339 to -341.8709 |
| nfcorpus | 2-2 | 409 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0970 | 0.0786 | 0.0185 | 0.0155 to 0.0215 |
| nfcorpus | 2-2 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0838 | 0.0590 | 0.0248 | 0.0224 to 0.0274 |
| nfcorpus | 2-2 | 512 | held-out | reciprocal rank | 1922 | 0.0654 | 0.0560 | 0.0093 | 0.0074 to 0.0113 |
| nfcorpus | 2-2 | 512 | held-out | in top 10 | 1922 | 0.1594 | 0.1154 | 0.0440 | 0.0390 to 0.0489 |
| nfcorpus | 2-2 | 512 | held-out | in top 100 | 1922 | 0.4070 | 0.2938 | 0.1131 | 0.1053 to 0.1209 |
| nfcorpus | 2-2 | 512 | held-out | log rank (lower is better) | 1922 | 4.9713 | 5.5554 | -0.5842 | -0.6154 to -0.5506 |
| nfcorpus | 2-2 | 512 | held-out | worst rank in the set | 1922 | 1035.6610 | 1394.9434 | -359.2824 | -380.4060 to -337.8939 |
| nfcorpus | 2-2 | 512 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0971 | 0.0786 | 0.0185 | 0.0156 to 0.0214 |
| nfcorpus | 2-2 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0827 | 0.0590 | 0.0237 | 0.0213 to 0.0261 |
| nfcorpus | 2-2 | 614 | held-out | reciprocal rank | 1922 | 0.0658 | 0.0560 | 0.0098 | 0.0080 to 0.0117 |
| nfcorpus | 2-2 | 614 | held-out | in top 10 | 1922 | 0.1574 | 0.1154 | 0.0420 | 0.0373 to 0.0466 |
| nfcorpus | 2-2 | 614 | held-out | in top 100 | 1922 | 0.4025 | 0.2938 | 0.1086 | 0.1009 to 0.1162 |
| nfcorpus | 2-2 | 614 | held-out | log rank (lower is better) | 1922 | 4.9858 | 5.5554 | -0.5697 | -0.6003 to -0.5372 |
| nfcorpus | 2-2 | 614 | held-out | worst rank in the set | 1922 | 1041.7813 | 1394.9434 | -353.1621 | -373.8173 to -332.6064 |
| nfcorpus | 2-2 | 614 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0970 | 0.0786 | 0.0184 | 0.0157 to 0.0212 |
| nfcorpus | 2-2 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0816 | 0.0590 | 0.0226 | 0.0203 to 0.0251 |
| nfcorpus | 2-2 | 716 | held-out | reciprocal rank | 1922 | 0.0658 | 0.0560 | 0.0097 | 0.0080 to 0.0115 |
| nfcorpus | 2-2 | 716 | held-out | in top 10 | 1922 | 0.1536 | 0.1154 | 0.0383 | 0.0339 to 0.0427 |
| nfcorpus | 2-2 | 716 | held-out | in top 100 | 1922 | 0.3964 | 0.2938 | 0.1026 | 0.0952 to 0.1098 |
| nfcorpus | 2-2 | 716 | held-out | log rank (lower is better) | 1922 | 5.0106 | 5.5554 | -0.5448 | -0.5736 to -0.5142 |
| nfcorpus | 2-2 | 716 | held-out | worst rank in the set | 1922 | 1051.3907 | 1394.9434 | -343.5527 | -363.7278 to -323.5866 |
| nfcorpus | 2-2 | 716 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0960 | 0.0786 | 0.0174 | 0.0148 to 0.0201 |
| nfcorpus | 2-2 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0800 | 0.0590 | 0.0210 | 0.0189 to 0.0232 |
| nfcorpus | 2-2 | 819 | held-out | reciprocal rank | 1922 | 0.0647 | 0.0560 | 0.0086 | 0.0071 to 0.0104 |
| nfcorpus | 2-2 | 819 | held-out | in top 10 | 1922 | 0.1464 | 0.1154 | 0.0311 | 0.0273 to 0.0350 |
| nfcorpus | 2-2 | 819 | held-out | in top 100 | 1922 | 0.3825 | 0.2938 | 0.0887 | 0.0822 to 0.0953 |
| nfcorpus | 2-2 | 819 | held-out | log rank (lower is better) | 1922 | 5.0733 | 5.5554 | -0.4822 | -0.5068 to -0.4553 |
| nfcorpus | 2-2 | 819 | held-out | worst rank in the set | 1922 | 1077.1667 | 1394.9434 | -317.7767 | -336.3629 to -299.6646 |
| nfcorpus | 2-2 | 819 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0932 | 0.0786 | 0.0147 | 0.0123 to 0.0171 |
| nfcorpus | 2-2 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0759 | 0.0590 | 0.0168 | 0.0150 to 0.0188 |
| nfcorpus | 2-2 | 921 | held-out | reciprocal rank | 1922 | 0.0628 | 0.0560 | 0.0067 | 0.0055 to 0.0080 |
| nfcorpus | 2-2 | 921 | held-out | in top 10 | 1922 | 0.1352 | 0.1154 | 0.0198 | 0.0169 to 0.0229 |
| nfcorpus | 2-2 | 921 | held-out | in top 100 | 1922 | 0.3539 | 0.2938 | 0.0601 | 0.0549 to 0.0655 |
| nfcorpus | 2-2 | 921 | held-out | log rank (lower is better) | 1922 | 5.2067 | 5.5554 | -0.3488 | -0.3662 to -0.3304 |
| nfcorpus | 2-2 | 921 | held-out | worst rank in the set | 1922 | 1142.7696 | 1394.9434 | -252.1738 | -266.7550 to -237.6841 |
| nfcorpus | 2-2 | 921 | held-out | nDCG@10, restricted to held-out labels | 1922 | 0.0890 | 0.0786 | 0.0105 | 0.0087 to 0.0123 |
| nfcorpus | 2-2 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 1922 | 0.0701 | 0.0590 | 0.0111 | 0.0098 to 0.0125 |
| nfcorpus | 5-5 | 102 | held-out | reciprocal rank | 1517 | 0.0505 | 0.0458 | 0.0048 | 0.0032 to 0.0064 |
| nfcorpus | 5-5 | 102 | held-out | in top 10 | 1517 | 0.1229 | 0.1010 | 0.0219 | 0.0180 to 0.0261 |
| nfcorpus | 5-5 | 102 | held-out | in top 100 | 1517 | 0.3635 | 0.2762 | 0.0873 | 0.0802 to 0.0945 |
| nfcorpus | 5-5 | 102 | held-out | log rank (lower is better) | 1517 | 5.2090 | 5.6726 | -0.4637 | -0.4989 to -0.4309 |
| nfcorpus | 5-5 | 102 | held-out | worst rank in the set | 1517 | 1732.0631 | 2119.1761 | -387.1130 | -413.2856 to -360.7543 |
| nfcorpus | 5-5 | 102 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1007 | 0.0895 | 0.0111 | 0.0079 to 0.0145 |
| nfcorpus | 5-5 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0632 | 0.0495 | 0.0137 | 0.0118 to 0.0157 |
| nfcorpus | 5-5 | 204 | held-out | reciprocal rank | 1517 | 0.0529 | 0.0458 | 0.0071 | 0.0056 to 0.0086 |
| nfcorpus | 5-5 | 204 | held-out | in top 10 | 1517 | 0.1294 | 0.1010 | 0.0284 | 0.0244 to 0.0327 |
| nfcorpus | 5-5 | 204 | held-out | in top 100 | 1517 | 0.3781 | 0.2762 | 0.1019 | 0.0947 to 0.1095 |
| nfcorpus | 5-5 | 204 | held-out | log rank (lower is better) | 1517 | 5.1363 | 5.6726 | -0.5363 | -0.5707 to -0.5041 |
| nfcorpus | 5-5 | 204 | held-out | worst rank in the set | 1517 | 1684.1957 | 2119.1761 | -434.9804 | -460.1005 to -409.6080 |
| nfcorpus | 5-5 | 204 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1058 | 0.0895 | 0.0163 | 0.0130 to 0.0197 |
| nfcorpus | 5-5 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0662 | 0.0495 | 0.0166 | 0.0147 to 0.0187 |
| nfcorpus | 5-5 | 307 | held-out | reciprocal rank | 1517 | 0.0538 | 0.0458 | 0.0081 | 0.0067 to 0.0095 |
| nfcorpus | 5-5 | 307 | held-out | in top 10 | 1517 | 0.1306 | 0.1010 | 0.0296 | 0.0256 to 0.0337 |
| nfcorpus | 5-5 | 307 | held-out | in top 100 | 1517 | 0.3790 | 0.2762 | 0.1028 | 0.0958 to 0.1105 |
| nfcorpus | 5-5 | 307 | held-out | log rank (lower is better) | 1517 | 5.1261 | 5.6726 | -0.5465 | -0.5798 to -0.5147 |
| nfcorpus | 5-5 | 307 | held-out | worst rank in the set | 1517 | 1674.1786 | 2119.1761 | -444.9974 | -469.7388 to -420.5814 |
| nfcorpus | 5-5 | 307 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1076 | 0.0895 | 0.0181 | 0.0149 to 0.0214 |
| nfcorpus | 5-5 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0666 | 0.0495 | 0.0171 | 0.0152 to 0.0191 |
| nfcorpus | 5-5 | 409 | held-out | reciprocal rank | 1517 | 0.0539 | 0.0458 | 0.0081 | 0.0068 to 0.0096 |
| nfcorpus | 5-5 | 409 | held-out | in top 10 | 1517 | 0.1299 | 0.1010 | 0.0289 | 0.0250 to 0.0327 |
| nfcorpus | 5-5 | 409 | held-out | in top 100 | 1517 | 0.3779 | 0.2762 | 0.1016 | 0.0948 to 0.1092 |
| nfcorpus | 5-5 | 409 | held-out | log rank (lower is better) | 1517 | 5.1340 | 5.6726 | -0.5386 | -0.5711 to -0.5076 |
| nfcorpus | 5-5 | 409 | held-out | worst rank in the set | 1517 | 1678.5127 | 2119.1761 | -440.6634 | -464.7442 to -416.7647 |
| nfcorpus | 5-5 | 409 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1076 | 0.0895 | 0.0180 | 0.0150 to 0.0211 |
| nfcorpus | 5-5 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0661 | 0.0495 | 0.0166 | 0.0148 to 0.0185 |
| nfcorpus | 5-5 | 512 | held-out | reciprocal rank | 1517 | 0.0540 | 0.0458 | 0.0082 | 0.0069 to 0.0095 |
| nfcorpus | 5-5 | 512 | held-out | in top 10 | 1517 | 0.1292 | 0.1010 | 0.0282 | 0.0246 to 0.0320 |
| nfcorpus | 5-5 | 512 | held-out | in top 100 | 1517 | 0.3721 | 0.2762 | 0.0959 | 0.0893 to 0.1032 |
| nfcorpus | 5-5 | 512 | held-out | log rank (lower is better) | 1517 | 5.1488 | 5.6726 | -0.5238 | -0.5548 to -0.4946 |
| nfcorpus | 5-5 | 512 | held-out | worst rank in the set | 1517 | 1687.6811 | 2119.1761 | -431.4950 | -455.3637 to -408.2626 |
| nfcorpus | 5-5 | 512 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1076 | 0.0895 | 0.0181 | 0.0153 to 0.0211 |
| nfcorpus | 5-5 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0653 | 0.0495 | 0.0158 | 0.0140 to 0.0176 |
| nfcorpus | 5-5 | 614 | held-out | reciprocal rank | 1517 | 0.0540 | 0.0458 | 0.0082 | 0.0069 to 0.0095 |
| nfcorpus | 5-5 | 614 | held-out | in top 10 | 1517 | 0.1279 | 0.1010 | 0.0269 | 0.0232 to 0.0304 |
| nfcorpus | 5-5 | 614 | held-out | in top 100 | 1517 | 0.3688 | 0.2762 | 0.0925 | 0.0862 to 0.0998 |
| nfcorpus | 5-5 | 614 | held-out | log rank (lower is better) | 1517 | 5.1656 | 5.6726 | -0.5070 | -0.5374 to -0.4785 |
| nfcorpus | 5-5 | 614 | held-out | worst rank in the set | 1517 | 1698.5177 | 2119.1761 | -420.6584 | -443.8769 to -398.3174 |
| nfcorpus | 5-5 | 614 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1072 | 0.0895 | 0.0177 | 0.0149 to 0.0206 |
| nfcorpus | 5-5 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0647 | 0.0495 | 0.0151 | 0.0135 to 0.0169 |
| nfcorpus | 5-5 | 716 | held-out | reciprocal rank | 1517 | 0.0535 | 0.0458 | 0.0078 | 0.0066 to 0.0090 |
| nfcorpus | 5-5 | 716 | held-out | in top 10 | 1517 | 0.1261 | 0.1010 | 0.0251 | 0.0216 to 0.0285 |
| nfcorpus | 5-5 | 716 | held-out | in top 100 | 1517 | 0.3647 | 0.2762 | 0.0884 | 0.0823 to 0.0954 |
| nfcorpus | 5-5 | 716 | held-out | log rank (lower is better) | 1517 | 5.1889 | 5.6726 | -0.4837 | -0.5121 to -0.4568 |
| nfcorpus | 5-5 | 716 | held-out | worst rank in the set | 1517 | 1714.4782 | 2119.1761 | -404.6978 | -426.9317 to -383.1534 |
| nfcorpus | 5-5 | 716 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1062 | 0.0895 | 0.0167 | 0.0140 to 0.0194 |
| nfcorpus | 5-5 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0635 | 0.0495 | 0.0140 | 0.0125 to 0.0157 |
| nfcorpus | 5-5 | 819 | held-out | reciprocal rank | 1517 | 0.0525 | 0.0458 | 0.0068 | 0.0057 to 0.0078 |
| nfcorpus | 5-5 | 819 | held-out | in top 10 | 1517 | 0.1214 | 0.1010 | 0.0204 | 0.0175 to 0.0232 |
| nfcorpus | 5-5 | 819 | held-out | in top 100 | 1517 | 0.3525 | 0.2762 | 0.0762 | 0.0706 to 0.0824 |
| nfcorpus | 5-5 | 819 | held-out | log rank (lower is better) | 1517 | 5.2454 | 5.6726 | -0.4273 | -0.4514 to -0.4038 |
| nfcorpus | 5-5 | 819 | held-out | worst rank in the set | 1517 | 1751.2251 | 2119.1761 | -367.9510 | -388.2186 to -348.9217 |
| nfcorpus | 5-5 | 819 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.1034 | 0.0895 | 0.0139 | 0.0117 to 0.0162 |
| nfcorpus | 5-5 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0610 | 0.0495 | 0.0115 | 0.0101 to 0.0129 |
| nfcorpus | 5-5 | 921 | held-out | reciprocal rank | 1517 | 0.0503 | 0.0458 | 0.0046 | 0.0038 to 0.0054 |
| nfcorpus | 5-5 | 921 | held-out | in top 10 | 1517 | 0.1145 | 0.1010 | 0.0135 | 0.0114 to 0.0157 |
| nfcorpus | 5-5 | 921 | held-out | in top 100 | 1517 | 0.3273 | 0.2762 | 0.0510 | 0.0467 to 0.0555 |
| nfcorpus | 5-5 | 921 | held-out | log rank (lower is better) | 1517 | 5.3678 | 5.6726 | -0.3048 | -0.3218 to -0.2886 |
| nfcorpus | 5-5 | 921 | held-out | worst rank in the set | 1517 | 1838.9617 | 2119.1761 | -280.2144 | -295.6766 to -266.4304 |
| nfcorpus | 5-5 | 921 | held-out | nDCG@10, restricted to held-out labels | 1517 | 0.0989 | 0.0895 | 0.0094 | 0.0077 to 0.0111 |
| nfcorpus | 5-5 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 1517 | 0.0568 | 0.0495 | 0.0072 | 0.0063 to 0.0082 |
| nfcorpus | 10-10 | 102 | held-out | reciprocal rank | 1105 | 0.0387 | 0.0375 | 0.0012 | 0.0002 to 0.0023 |
| nfcorpus | 10-10 | 102 | held-out | in top 10 | 1105 | 0.0896 | 0.0844 | 0.0053 | 0.0028 to 0.0080 |
| nfcorpus | 10-10 | 102 | held-out | in top 100 | 1105 | 0.3108 | 0.2544 | 0.0564 | 0.0502 to 0.0627 |
| nfcorpus | 10-10 | 102 | held-out | log rank (lower is better) | 1105 | 5.4751 | 5.7993 | -0.3242 | -0.3526 to -0.2965 |
| nfcorpus | 10-10 | 102 | held-out | worst rank in the set | 1105 | 2350.2149 | 2646.2550 | -296.0400 | -319.8164 to -272.2182 |
| nfcorpus | 10-10 | 102 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.0954 | 0.0940 | 0.0014 | -0.0014 to 0.0043 |
| nfcorpus | 10-10 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0476 | 0.0422 | 0.0055 | 0.0041 to 0.0069 |
| nfcorpus | 10-10 | 204 | held-out | reciprocal rank | 1105 | 0.0402 | 0.0375 | 0.0027 | 0.0018 to 0.0037 |
| nfcorpus | 10-10 | 204 | held-out | in top 10 | 1105 | 0.0941 | 0.0844 | 0.0097 | 0.0072 to 0.0122 |
| nfcorpus | 10-10 | 204 | held-out | in top 100 | 1105 | 0.3243 | 0.2544 | 0.0699 | 0.0634 to 0.0764 |
| nfcorpus | 10-10 | 204 | held-out | log rank (lower is better) | 1105 | 5.4119 | 5.7993 | -0.3874 | -0.4157 to -0.3601 |
| nfcorpus | 10-10 | 204 | held-out | worst rank in the set | 1105 | 2308.3803 | 2646.2550 | -337.8747 | -360.7227 to -314.6835 |
| nfcorpus | 10-10 | 204 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.0995 | 0.0940 | 0.0055 | 0.0030 to 0.0081 |
| nfcorpus | 10-10 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0496 | 0.0422 | 0.0074 | 0.0061 to 0.0087 |
| nfcorpus | 10-10 | 307 | held-out | reciprocal rank | 1105 | 0.0408 | 0.0375 | 0.0033 | 0.0024 to 0.0042 |
| nfcorpus | 10-10 | 307 | held-out | in top 10 | 1105 | 0.0955 | 0.0844 | 0.0112 | 0.0088 to 0.0136 |
| nfcorpus | 10-10 | 307 | held-out | in top 100 | 1105 | 0.3250 | 0.2544 | 0.0706 | 0.0641 to 0.0769 |
| nfcorpus | 10-10 | 307 | held-out | log rank (lower is better) | 1105 | 5.3999 | 5.7993 | -0.3994 | -0.4268 to -0.3735 |
| nfcorpus | 10-10 | 307 | held-out | worst rank in the set | 1105 | 2304.0484 | 2646.2550 | -342.2066 | -364.5720 to -319.9498 |
| nfcorpus | 10-10 | 307 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1012 | 0.0940 | 0.0072 | 0.0048 to 0.0097 |
| nfcorpus | 10-10 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0500 | 0.0422 | 0.0079 | 0.0067 to 0.0090 |
| nfcorpus | 10-10 | 409 | held-out | reciprocal rank | 1105 | 0.0407 | 0.0375 | 0.0032 | 0.0024 to 0.0041 |
| nfcorpus | 10-10 | 409 | held-out | in top 10 | 1105 | 0.0957 | 0.0844 | 0.0114 | 0.0092 to 0.0136 |
| nfcorpus | 10-10 | 409 | held-out | in top 100 | 1105 | 0.3234 | 0.2544 | 0.0690 | 0.0627 to 0.0752 |
| nfcorpus | 10-10 | 409 | held-out | log rank (lower is better) | 1105 | 5.4056 | 5.7993 | -0.3937 | -0.4202 to -0.3685 |
| nfcorpus | 10-10 | 409 | held-out | worst rank in the set | 1105 | 2307.8843 | 2646.2550 | -338.3707 | -359.9289 to -317.7007 |
| nfcorpus | 10-10 | 409 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1013 | 0.0940 | 0.0073 | 0.0051 to 0.0096 |
| nfcorpus | 10-10 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0497 | 0.0422 | 0.0076 | 0.0065 to 0.0087 |
| nfcorpus | 10-10 | 512 | held-out | reciprocal rank | 1105 | 0.0408 | 0.0375 | 0.0033 | 0.0025 to 0.0042 |
| nfcorpus | 10-10 | 512 | held-out | in top 10 | 1105 | 0.0952 | 0.0844 | 0.0108 | 0.0087 to 0.0131 |
| nfcorpus | 10-10 | 512 | held-out | in top 100 | 1105 | 0.3211 | 0.2544 | 0.0667 | 0.0606 to 0.0728 |
| nfcorpus | 10-10 | 512 | held-out | log rank (lower is better) | 1105 | 5.4186 | 5.7993 | -0.3807 | -0.4061 to -0.3565 |
| nfcorpus | 10-10 | 512 | held-out | worst rank in the set | 1105 | 2316.9550 | 2646.2550 | -329.3000 | -349.7602 to -309.5462 |
| nfcorpus | 10-10 | 512 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1013 | 0.0940 | 0.0073 | 0.0052 to 0.0095 |
| nfcorpus | 10-10 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0494 | 0.0422 | 0.0073 | 0.0061 to 0.0084 |
| nfcorpus | 10-10 | 614 | held-out | reciprocal rank | 1105 | 0.0407 | 0.0375 | 0.0033 | 0.0025 to 0.0041 |
| nfcorpus | 10-10 | 614 | held-out | in top 10 | 1105 | 0.0949 | 0.0844 | 0.0105 | 0.0085 to 0.0126 |
| nfcorpus | 10-10 | 614 | held-out | in top 100 | 1105 | 0.3181 | 0.2544 | 0.0637 | 0.0579 to 0.0696 |
| nfcorpus | 10-10 | 614 | held-out | log rank (lower is better) | 1105 | 5.4335 | 5.7993 | -0.3658 | -0.3900 to -0.3422 |
| nfcorpus | 10-10 | 614 | held-out | worst rank in the set | 1105 | 2327.3859 | 2646.2550 | -318.8691 | -338.5646 to -299.3166 |
| nfcorpus | 10-10 | 614 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1013 | 0.0940 | 0.0073 | 0.0053 to 0.0094 |
| nfcorpus | 10-10 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0491 | 0.0422 | 0.0069 | 0.0059 to 0.0080 |
| nfcorpus | 10-10 | 716 | held-out | reciprocal rank | 1105 | 0.0406 | 0.0375 | 0.0032 | 0.0024 to 0.0039 |
| nfcorpus | 10-10 | 716 | held-out | in top 10 | 1105 | 0.0945 | 0.0844 | 0.0102 | 0.0082 to 0.0122 |
| nfcorpus | 10-10 | 716 | held-out | in top 100 | 1105 | 0.3149 | 0.2544 | 0.0605 | 0.0550 to 0.0662 |
| nfcorpus | 10-10 | 716 | held-out | log rank (lower is better) | 1105 | 5.4500 | 5.7993 | -0.3493 | -0.3722 to -0.3269 |
| nfcorpus | 10-10 | 716 | held-out | worst rank in the set | 1105 | 2339.7557 | 2646.2550 | -306.4993 | -325.3902 to -287.8695 |
| nfcorpus | 10-10 | 716 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1012 | 0.0940 | 0.0072 | 0.0053 to 0.0092 |
| nfcorpus | 10-10 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0487 | 0.0422 | 0.0065 | 0.0055 to 0.0075 |
| nfcorpus | 10-10 | 819 | held-out | reciprocal rank | 1105 | 0.0404 | 0.0375 | 0.0029 | 0.0023 to 0.0036 |
| nfcorpus | 10-10 | 819 | held-out | in top 10 | 1105 | 0.0931 | 0.0844 | 0.0087 | 0.0069 to 0.0105 |
| nfcorpus | 10-10 | 819 | held-out | in top 100 | 1105 | 0.3072 | 0.2544 | 0.0528 | 0.0480 to 0.0579 |
| nfcorpus | 10-10 | 819 | held-out | log rank (lower is better) | 1105 | 5.4878 | 5.7993 | -0.3115 | -0.3315 to -0.2919 |
| nfcorpus | 10-10 | 819 | held-out | worst rank in the set | 1105 | 2368.8078 | 2646.2550 | -277.4472 | -294.4460 to -260.6505 |
| nfcorpus | 10-10 | 819 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.1005 | 0.0940 | 0.0065 | 0.0047 to 0.0083 |
| nfcorpus | 10-10 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0478 | 0.0422 | 0.0057 | 0.0048 to 0.0065 |
| nfcorpus | 10-10 | 921 | held-out | reciprocal rank | 1105 | 0.0396 | 0.0375 | 0.0021 | 0.0016 to 0.0026 |
| nfcorpus | 10-10 | 921 | held-out | in top 10 | 1105 | 0.0905 | 0.0844 | 0.0061 | 0.0046 to 0.0075 |
| nfcorpus | 10-10 | 921 | held-out | in top 100 | 1105 | 0.2911 | 0.2544 | 0.0367 | 0.0329 to 0.0406 |
| nfcorpus | 10-10 | 921 | held-out | log rank (lower is better) | 1105 | 5.5733 | 5.7993 | -0.2260 | -0.2399 to -0.2118 |
| nfcorpus | 10-10 | 921 | held-out | worst rank in the set | 1105 | 2438.4052 | 2646.2550 | -207.8498 | -220.9125 to -194.5683 |
| nfcorpus | 10-10 | 921 | held-out | nDCG@10, restricted to held-out labels | 1105 | 0.0987 | 0.0940 | 0.0046 | 0.0033 to 0.0061 |
| nfcorpus | 10-10 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 1105 | 0.0460 | 0.0422 | 0.0039 | 0.0033 to 0.0045 |
| nfcorpus | 20-20 | 102 | held-out | reciprocal rank | 611 | 0.0297 | 0.0292 | 0.0005 | -0.0003 to 0.0012 |
| nfcorpus | 20-20 | 102 | held-out | in top 10 | 611 | 0.0655 | 0.0656 | -0.0000 | -0.0021 to 0.0020 |
| nfcorpus | 20-20 | 102 | held-out | in top 100 | 611 | 0.2595 | 0.2317 | 0.0278 | 0.0225 to 0.0333 |
| nfcorpus | 20-20 | 102 | held-out | log rank (lower is better) | 611 | 5.7292 | 5.9328 | -0.2036 | -0.2296 to -0.1793 |
| nfcorpus | 20-20 | 102 | held-out | worst rank in the set | 611 | 2874.4106 | 3056.7717 | -182.3611 | -205.0844 to -159.5518 |
| nfcorpus | 20-20 | 102 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1401 | 0.1419 | -0.0019 | -0.0058 to 0.0023 |
| nfcorpus | 20-20 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0368 | 0.0345 | 0.0023 | 0.0014 to 0.0032 |
| nfcorpus | 20-20 | 204 | held-out | reciprocal rank | 611 | 0.0309 | 0.0292 | 0.0017 | 0.0010 to 0.0023 |
| nfcorpus | 20-20 | 204 | held-out | in top 10 | 611 | 0.0689 | 0.0656 | 0.0033 | 0.0015 to 0.0052 |
| nfcorpus | 20-20 | 204 | held-out | in top 100 | 611 | 0.2725 | 0.2317 | 0.0408 | 0.0357 to 0.0461 |
| nfcorpus | 20-20 | 204 | held-out | log rank (lower is better) | 611 | 5.6702 | 5.9328 | -0.2626 | -0.2872 to -0.2391 |
| nfcorpus | 20-20 | 204 | held-out | worst rank in the set | 611 | 2838.5681 | 3056.7717 | -218.2036 | -241.4368 to -196.3435 |
| nfcorpus | 20-20 | 204 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1462 | 0.1419 | 0.0043 | 0.0007 to 0.0081 |
| nfcorpus | 20-20 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0384 | 0.0345 | 0.0039 | 0.0031 to 0.0047 |
| nfcorpus | 20-20 | 307 | held-out | reciprocal rank | 611 | 0.0312 | 0.0292 | 0.0020 | 0.0014 to 0.0027 |
| nfcorpus | 20-20 | 307 | held-out | in top 10 | 611 | 0.0691 | 0.0656 | 0.0035 | 0.0019 to 0.0053 |
| nfcorpus | 20-20 | 307 | held-out | in top 100 | 611 | 0.2738 | 0.2317 | 0.0421 | 0.0371 to 0.0471 |
| nfcorpus | 20-20 | 307 | held-out | log rank (lower is better) | 611 | 5.6610 | 5.9328 | -0.2719 | -0.2958 to -0.2490 |
| nfcorpus | 20-20 | 307 | held-out | worst rank in the set | 611 | 2838.0908 | 3056.7717 | -218.6809 | -241.0726 to -197.8286 |
| nfcorpus | 20-20 | 307 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1474 | 0.1419 | 0.0054 | 0.0021 to 0.0089 |
| nfcorpus | 20-20 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0387 | 0.0345 | 0.0042 | 0.0034 to 0.0050 |
| nfcorpus | 20-20 | 409 | held-out | reciprocal rank | 611 | 0.0313 | 0.0292 | 0.0020 | 0.0015 to 0.0027 |
| nfcorpus | 20-20 | 409 | held-out | in top 10 | 611 | 0.0692 | 0.0656 | 0.0036 | 0.0020 to 0.0053 |
| nfcorpus | 20-20 | 409 | held-out | in top 100 | 611 | 0.2730 | 0.2317 | 0.0413 | 0.0367 to 0.0462 |
| nfcorpus | 20-20 | 409 | held-out | log rank (lower is better) | 611 | 5.6621 | 5.9328 | -0.2707 | -0.2931 to -0.2491 |
| nfcorpus | 20-20 | 409 | held-out | worst rank in the set | 611 | 2836.5260 | 3056.7717 | -220.2457 | -241.8675 to -200.0391 |
| nfcorpus | 20-20 | 409 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1477 | 0.1419 | 0.0057 | 0.0027 to 0.0090 |
| nfcorpus | 20-20 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0387 | 0.0345 | 0.0042 | 0.0035 to 0.0050 |
| nfcorpus | 20-20 | 512 | held-out | reciprocal rank | 611 | 0.0313 | 0.0292 | 0.0021 | 0.0016 to 0.0027 |
| nfcorpus | 20-20 | 512 | held-out | in top 10 | 611 | 0.0692 | 0.0656 | 0.0036 | 0.0021 to 0.0051 |
| nfcorpus | 20-20 | 512 | held-out | in top 100 | 611 | 0.2706 | 0.2317 | 0.0389 | 0.0346 to 0.0434 |
| nfcorpus | 20-20 | 512 | held-out | log rank (lower is better) | 611 | 5.6707 | 5.9328 | -0.2622 | -0.2833 to -0.2418 |
| nfcorpus | 20-20 | 512 | held-out | worst rank in the set | 611 | 2840.9239 | 3056.7717 | -215.8478 | -236.5524 to -195.9241 |
| nfcorpus | 20-20 | 512 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1481 | 0.1419 | 0.0061 | 0.0032 to 0.0092 |
| nfcorpus | 20-20 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0386 | 0.0345 | 0.0041 | 0.0035 to 0.0048 |
| nfcorpus | 20-20 | 614 | held-out | reciprocal rank | 611 | 0.0312 | 0.0292 | 0.0020 | 0.0015 to 0.0025 |
| nfcorpus | 20-20 | 614 | held-out | in top 10 | 611 | 0.0689 | 0.0656 | 0.0033 | 0.0018 to 0.0047 |
| nfcorpus | 20-20 | 614 | held-out | in top 100 | 611 | 0.2686 | 0.2317 | 0.0369 | 0.0329 to 0.0412 |
| nfcorpus | 20-20 | 614 | held-out | log rank (lower is better) | 611 | 5.6818 | 5.9328 | -0.2510 | -0.2715 to -0.2318 |
| nfcorpus | 20-20 | 614 | held-out | worst rank in the set | 611 | 2849.8343 | 3056.7717 | -206.9374 | -226.4124 to -188.3536 |
| nfcorpus | 20-20 | 614 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1476 | 0.1419 | 0.0057 | 0.0030 to 0.0086 |
| nfcorpus | 20-20 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0384 | 0.0345 | 0.0039 | 0.0033 to 0.0046 |
| nfcorpus | 20-20 | 716 | held-out | reciprocal rank | 611 | 0.0312 | 0.0292 | 0.0020 | 0.0015 to 0.0025 |
| nfcorpus | 20-20 | 716 | held-out | in top 10 | 611 | 0.0690 | 0.0656 | 0.0034 | 0.0020 to 0.0048 |
| nfcorpus | 20-20 | 716 | held-out | in top 100 | 611 | 0.2668 | 0.2317 | 0.0351 | 0.0312 to 0.0392 |
| nfcorpus | 20-20 | 716 | held-out | log rank (lower is better) | 611 | 5.6938 | 5.9328 | -0.2390 | -0.2581 to -0.2209 |
| nfcorpus | 20-20 | 716 | held-out | worst rank in the set | 611 | 2858.5818 | 3056.7717 | -198.1899 | -217.4809 to -180.3735 |
| nfcorpus | 20-20 | 716 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1479 | 0.1419 | 0.0060 | 0.0035 to 0.0088 |
| nfcorpus | 20-20 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0382 | 0.0345 | 0.0037 | 0.0031 to 0.0043 |
| nfcorpus | 20-20 | 819 | held-out | reciprocal rank | 611 | 0.0310 | 0.0292 | 0.0017 | 0.0013 to 0.0022 |
| nfcorpus | 20-20 | 819 | held-out | in top 10 | 611 | 0.0684 | 0.0656 | 0.0028 | 0.0016 to 0.0040 |
| nfcorpus | 20-20 | 819 | held-out | in top 100 | 611 | 0.2618 | 0.2317 | 0.0301 | 0.0266 to 0.0337 |
| nfcorpus | 20-20 | 819 | held-out | log rank (lower is better) | 611 | 5.7199 | 5.9328 | -0.2130 | -0.2294 to -0.1973 |
| nfcorpus | 20-20 | 819 | held-out | worst rank in the set | 611 | 2879.5910 | 3056.7717 | -177.1806 | -194.3353 to -160.7865 |
| nfcorpus | 20-20 | 819 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1470 | 0.1419 | 0.0051 | 0.0028 to 0.0076 |
| nfcorpus | 20-20 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0377 | 0.0345 | 0.0032 | 0.0027 to 0.0038 |
| nfcorpus | 20-20 | 921 | held-out | reciprocal rank | 611 | 0.0305 | 0.0292 | 0.0013 | 0.0010 to 0.0016 |
| nfcorpus | 20-20 | 921 | held-out | in top 10 | 611 | 0.0680 | 0.0656 | 0.0024 | 0.0014 to 0.0035 |
| nfcorpus | 20-20 | 921 | held-out | in top 100 | 611 | 0.2525 | 0.2317 | 0.0208 | 0.0182 to 0.0236 |
| nfcorpus | 20-20 | 921 | held-out | log rank (lower is better) | 611 | 5.7781 | 5.9328 | -0.1548 | -0.1662 to -0.1438 |
| nfcorpus | 20-20 | 921 | held-out | worst rank in the set | 611 | 2926.6964 | 3056.7717 | -130.0753 | -142.5747 to -118.3253 |
| nfcorpus | 20-20 | 921 | held-out | nDCG@10, restricted to held-out labels | 611 | 0.1462 | 0.1419 | 0.0043 | 0.0026 to 0.0063 |
| nfcorpus | 20-20 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 611 | 0.0368 | 0.0345 | 0.0023 | 0.0020 to 0.0027 |
| msmarco | prefix | 102 | all known positives | reciprocal rank | 2321 | 0.1213 | 0.1796 | -0.0582 | -0.0645 to -0.0523 |
| msmarco | prefix | 102 | all known positives | in top 10 | 2321 | 0.2546 | 0.3811 | -0.1264 | -0.1385 to -0.1143 |
| msmarco | prefix | 102 | all known positives | in top 100 | 2321 | 0.5744 | 0.7382 | -0.1638 | -0.1763 to -0.1514 |
| msmarco | prefix | 102 | all known positives | log rank (lower is better) | 2321 | 4.2745 | 3.2600 | 1.0146 | 0.9673 to 1.0637 |
| msmarco | prefix | 102 | all known positives | worst rank in the set | 2321 | 9484.8112 | 2644.1045 | 6840.7068 | 3767.1288 to 10381.7955 |
| msmarco | prefix | 204 | all known positives | reciprocal rank | 2321 | 0.1477 | 0.1796 | -0.0319 | -0.0372 to -0.0267 |
| msmarco | prefix | 204 | all known positives | in top 10 | 2321 | 0.3120 | 0.3811 | -0.0691 | -0.0797 to -0.0591 |
| msmarco | prefix | 204 | all known positives | in top 100 | 2321 | 0.6575 | 0.7382 | -0.0807 | -0.0903 to -0.0708 |
| msmarco | prefix | 204 | all known positives | log rank (lower is better) | 2321 | 3.7670 | 3.2600 | 0.5070 | 0.4746 to 0.5431 |
| msmarco | prefix | 204 | all known positives | worst rank in the set | 2321 | 5358.9765 | 2644.1045 | 2714.8720 | 551.3653 to 4966.0507 |
| msmarco | prefix | 307 | all known positives | reciprocal rank | 2321 | 0.1585 | 0.1796 | -0.0211 | -0.0254 to -0.0167 |
| msmarco | prefix | 307 | all known positives | in top 10 | 2321 | 0.3326 | 0.3811 | -0.0485 | -0.0578 to -0.0393 |
| msmarco | prefix | 307 | all known positives | in top 100 | 2321 | 0.6844 | 0.7382 | -0.0538 | -0.0627 to -0.0444 |
| msmarco | prefix | 307 | all known positives | log rank (lower is better) | 2321 | 3.5731 | 3.2600 | 0.3131 | 0.2872 to 0.3389 |
| msmarco | prefix | 307 | all known positives | worst rank in the set | 2321 | 3442.1217 | 2644.1045 | 798.0172 | -832.9048 to 2210.0476 |
| msmarco | prefix | 409 | all known positives | reciprocal rank | 2321 | 0.1648 | 0.1796 | -0.0147 | -0.0188 to -0.0105 |
| msmarco | prefix | 409 | all known positives | in top 10 | 2321 | 0.3484 | 0.3811 | -0.0326 | -0.0407 to -0.0246 |
| msmarco | prefix | 409 | all known positives | in top 100 | 2321 | 0.7006 | 0.7382 | -0.0376 | -0.0459 to -0.0296 |
| msmarco | prefix | 409 | all known positives | log rank (lower is better) | 2321 | 3.4613 | 3.2600 | 0.2013 | 0.1800 to 0.2225 |
| msmarco | prefix | 409 | all known positives | worst rank in the set | 2321 | 2891.1419 | 2644.1045 | 247.0374 | -849.2418 to 1070.3096 |
| msmarco | prefix | 512 | all known positives | reciprocal rank | 2321 | 0.1696 | 0.1796 | -0.0099 | -0.0138 to -0.0061 |
| msmarco | prefix | 512 | all known positives | in top 10 | 2321 | 0.3579 | 0.3811 | -0.0231 | -0.0307 to -0.0159 |
| msmarco | prefix | 512 | all known positives | in top 100 | 2321 | 0.7171 | 0.7382 | -0.0212 | -0.0284 to -0.0143 |
| msmarco | prefix | 512 | all known positives | log rank (lower is better) | 2321 | 3.3937 | 3.2600 | 0.1337 | 0.1164 to 0.1508 |
| msmarco | prefix | 512 | all known positives | worst rank in the set | 2321 | 2436.9911 | 2644.1045 | -207.1134 | -1334.7807 to 634.7093 |
| msmarco | prefix | 614 | all known positives | reciprocal rank | 2321 | 0.1730 | 0.1796 | -0.0066 | -0.0099 to -0.0032 |
| msmarco | prefix | 614 | all known positives | in top 10 | 2321 | 0.3660 | 0.3811 | -0.0150 | -0.0216 to -0.0085 |
| msmarco | prefix | 614 | all known positives | in top 100 | 2321 | 0.7238 | 0.7382 | -0.0144 | -0.0209 to -0.0079 |
| msmarco | prefix | 614 | all known positives | log rank (lower is better) | 2321 | 3.3472 | 3.2600 | 0.0872 | 0.0732 to 0.1015 |
| msmarco | prefix | 614 | all known positives | worst rank in the set | 2321 | 2226.2491 | 2644.1045 | -417.8554 | -1443.9120 to 237.2262 |
| msmarco | prefix | 716 | all known positives | reciprocal rank | 2321 | 0.1763 | 0.1796 | -0.0033 | -0.0060 to -0.0005 |
| msmarco | prefix | 716 | all known positives | in top 10 | 2321 | 0.3738 | 0.3811 | -0.0073 | -0.0131 to -0.0016 |
| msmarco | prefix | 716 | all known positives | in top 100 | 2321 | 0.7287 | 0.7382 | -0.0095 | -0.0149 to -0.0037 |
| msmarco | prefix | 716 | all known positives | log rank (lower is better) | 2321 | 3.3046 | 3.2600 | 0.0447 | 0.0338 to 0.0559 |
| msmarco | prefix | 716 | all known positives | worst rank in the set | 2321 | 2246.2748 | 2644.1045 | -397.8296 | -1175.2422 to 179.9738 |
| msmarco | prefix | 819 | all known positives | reciprocal rank | 2321 | 0.1784 | 0.1796 | -0.0012 | -0.0035 to 0.0011 |
| msmarco | prefix | 819 | all known positives | in top 10 | 2321 | 0.3769 | 0.3811 | -0.0042 | -0.0091 to 0.0010 |
| msmarco | prefix | 819 | all known positives | in top 100 | 2321 | 0.7341 | 0.7382 | -0.0042 | -0.0088 to 0.0002 |
| msmarco | prefix | 819 | all known positives | log rank (lower is better) | 2321 | 3.2821 | 3.2600 | 0.0221 | 0.0140 to 0.0305 |
| msmarco | prefix | 819 | all known positives | worst rank in the set | 2321 | 2463.9098 | 2644.1045 | -180.1947 | -725.3450 to 174.1537 |
| msmarco | prefix | 921 | all known positives | reciprocal rank | 2321 | 0.1785 | 0.1796 | -0.0010 | -0.0029 to 0.0007 |
| msmarco | prefix | 921 | all known positives | in top 10 | 2321 | 0.3797 | 0.3811 | -0.0014 | -0.0056 to 0.0028 |
| msmarco | prefix | 921 | all known positives | in top 100 | 2321 | 0.7357 | 0.7382 | -0.0026 | -0.0061 to 0.0012 |
| msmarco | prefix | 921 | all known positives | log rank (lower is better) | 2321 | 3.2680 | 3.2600 | 0.0080 | 0.0026 to 0.0137 |
| msmarco | prefix | 921 | all known positives | worst rank in the set | 2321 | 2758.1812 | 2644.1045 | 114.0768 | -92.1995 to 402.8312 |
| msmarco | shared | 102 | all known positives | reciprocal rank | 2321 | 0.4007 | 0.1796 | 0.2211 | 0.2124 to 0.2289 |
| msmarco | shared | 102 | all known positives | in top 10 | 2321 | 0.7196 | 0.3811 | 0.3385 | 0.3248 to 0.3521 |
| msmarco | shared | 102 | all known positives | in top 100 | 2321 | 0.9486 | 0.7382 | 0.2103 | 0.1977 to 0.2226 |
| msmarco | shared | 102 | all known positives | log rank (lower is better) | 2321 | 1.6477 | 3.2600 | -1.6123 | -1.6603 to -1.5605 |
| msmarco | shared | 102 | all known positives | worst rank in the set | 2321 | 89.4090 | 2644.1045 | -2554.6955 | -4529.3182 to -1104.8821 |
| msmarco | shared | 204 | all known positives | reciprocal rank | 2321 | 0.3941 | 0.1796 | 0.2146 | 0.2064 to 0.2222 |
| msmarco | shared | 204 | all known positives | in top 10 | 2321 | 0.7131 | 0.3811 | 0.3320 | 0.3182 to 0.3455 |
| msmarco | shared | 204 | all known positives | in top 100 | 2321 | 0.9472 | 0.7382 | 0.2089 | 0.1963 to 0.2213 |
| msmarco | shared | 204 | all known positives | log rank (lower is better) | 2321 | 1.6656 | 3.2600 | -1.5944 | -1.6413 to -1.5466 |
| msmarco | shared | 204 | all known positives | worst rank in the set | 2321 | 88.9769 | 2644.1045 | -2555.1276 | -4530.8181 to -1104.1013 |
| msmarco | shared | 307 | all known positives | reciprocal rank | 2321 | 0.3766 | 0.1796 | 0.1970 | 0.1893 to 0.2038 |
| msmarco | shared | 307 | all known positives | in top 10 | 2321 | 0.6945 | 0.3811 | 0.3134 | 0.2999 to 0.3272 |
| msmarco | shared | 307 | all known positives | in top 100 | 2321 | 0.9424 | 0.7382 | 0.2042 | 0.1918 to 0.2162 |
| msmarco | shared | 307 | all known positives | log rank (lower is better) | 2321 | 1.7492 | 3.2600 | -1.5108 | -1.5548 to -1.4648 |
| msmarco | shared | 307 | all known positives | worst rank in the set | 2321 | 113.3287 | 2644.1045 | -2530.7758 | -4507.1493 to -1093.7544 |
| msmarco | shared | 409 | all known positives | reciprocal rank | 2321 | 0.3582 | 0.1796 | 0.1786 | 0.1715 to 0.1854 |
| msmarco | shared | 409 | all known positives | in top 10 | 2321 | 0.6694 | 0.3811 | 0.2883 | 0.2750 to 0.3011 |
| msmarco | shared | 409 | all known positives | in top 100 | 2321 | 0.9341 | 0.7382 | 0.1959 | 0.1834 to 0.2080 |
| msmarco | shared | 409 | all known positives | log rank (lower is better) | 2321 | 1.8485 | 3.2600 | -1.4115 | -1.4541 to -1.3689 |
| msmarco | shared | 409 | all known positives | worst rank in the set | 2321 | 123.9800 | 2644.1045 | -2520.1245 | -4490.7922 to -1086.0754 |
| msmarco | shared | 512 | all known positives | reciprocal rank | 2321 | 0.3390 | 0.1796 | 0.1594 | 0.1525 to 0.1657 |
| msmarco | shared | 512 | all known positives | in top 10 | 2321 | 0.6456 | 0.3811 | 0.2645 | 0.2512 to 0.2768 |
| msmarco | shared | 512 | all known positives | in top 100 | 2321 | 0.9244 | 0.7382 | 0.1862 | 0.1739 to 0.1982 |
| msmarco | shared | 512 | all known positives | log rank (lower is better) | 2321 | 1.9542 | 3.2600 | -1.3058 | -1.3453 to -1.2648 |
| msmarco | shared | 512 | all known positives | worst rank in the set | 2321 | 148.4759 | 2644.1045 | -2495.6286 | -4450.0048 to -1070.6887 |
| msmarco | shared | 614 | all known positives | reciprocal rank | 2321 | 0.3217 | 0.1796 | 0.1421 | 0.1357 to 0.1480 |
| msmarco | shared | 614 | all known positives | in top 10 | 2321 | 0.6195 | 0.3811 | 0.2384 | 0.2259 to 0.2506 |
| msmarco | shared | 614 | all known positives | in top 100 | 2321 | 0.9151 | 0.7382 | 0.1769 | 0.1647 to 0.1883 |
| msmarco | shared | 614 | all known positives | log rank (lower is better) | 2321 | 2.0550 | 3.2600 | -1.2050 | -1.2424 to -1.1671 |
| msmarco | shared | 614 | all known positives | worst rank in the set | 2321 | 172.0441 | 2644.1045 | -2472.0603 | -4425.4196 to -1057.0073 |
| msmarco | shared | 716 | all known positives | reciprocal rank | 2321 | 0.3050 | 0.1796 | 0.1255 | 0.1195 to 0.1312 |
| msmarco | shared | 716 | all known positives | in top 10 | 2321 | 0.5932 | 0.3811 | 0.2121 | 0.2008 to 0.2238 |
| msmarco | shared | 716 | all known positives | in top 100 | 2321 | 0.9052 | 0.7382 | 0.1670 | 0.1558 to 0.1783 |
| msmarco | shared | 716 | all known positives | log rank (lower is better) | 2321 | 2.1530 | 3.2600 | -1.1070 | -1.1427 to -1.0708 |
| msmarco | shared | 716 | all known positives | worst rank in the set | 2321 | 206.3199 | 2644.1045 | -2437.7845 | -4382.5992 to -1037.3034 |
| msmarco | shared | 819 | all known positives | reciprocal rank | 2321 | 0.2904 | 0.1796 | 0.1108 | 0.1053 to 0.1161 |
| msmarco | shared | 819 | all known positives | in top 10 | 2321 | 0.5696 | 0.3811 | 0.1886 | 0.1770 to 0.1997 |
| msmarco | shared | 819 | all known positives | in top 100 | 2321 | 0.8943 | 0.7382 | 0.1561 | 0.1450 to 0.1674 |
| msmarco | shared | 819 | all known positives | log rank (lower is better) | 2321 | 2.2478 | 3.2600 | -1.0122 | -1.0449 to -0.9789 |
| msmarco | shared | 819 | all known positives | worst rank in the set | 2321 | 244.4422 | 2644.1045 | -2399.6622 | -4331.4981 to -1009.7510 |
| msmarco | shared | 921 | all known positives | reciprocal rank | 2321 | 0.2598 | 0.1796 | 0.0802 | 0.0757 to 0.0846 |
| msmarco | shared | 921 | all known positives | in top 10 | 2321 | 0.5192 | 0.3811 | 0.1381 | 0.1283 to 0.1479 |
| msmarco | shared | 921 | all known positives | in top 100 | 2321 | 0.8611 | 0.7382 | 0.1229 | 0.1127 to 0.1324 |
| msmarco | shared | 921 | all known positives | log rank (lower is better) | 2321 | 2.4801 | 3.2600 | -0.7799 | -0.8050 to -0.7543 |
| msmarco | shared | 921 | all known positives | worst rank in the set | 2321 | 374.0590 | 2644.1045 | -2270.0455 | -4159.5722 to -925.5079 |
| msmarco | own | 102 | the mask's own positive | reciprocal rank | 2321 | 0.8479 | 0.1850 | 0.6629 | 0.6526 to 0.6731 |
| msmarco | own | 102 | the mask's own positive | in top 10 | 2321 | 0.9874 | 0.3916 | 0.5957 | 0.5794 to 0.6110 |
| msmarco | own | 102 | the mask's own positive | in top 100 | 2321 | 0.9999 | 0.7462 | 0.2536 | 0.2390 to 0.2672 |
| msmarco | own | 102 | the mask's own positive | log rank (lower is better) | 2321 | 0.2745 | 3.2084 | -2.9339 | -2.9986 to -2.8656 |
| msmarco | own | 102 | the mask's own positive | worst rank in the set | 2321 | 1.7881 | 1250.4414 | -1248.6533 | -2279.9679 to -548.7145 |
| msmarco | own | 204 | the mask's own positive | reciprocal rank | 2321 | 0.8321 | 0.1850 | 0.6471 | 0.6367 to 0.6578 |
| msmarco | own | 204 | the mask's own positive | in top 10 | 2321 | 0.9808 | 0.3916 | 0.5892 | 0.5727 to 0.6047 |
| msmarco | own | 204 | the mask's own positive | in top 100 | 2321 | 0.9999 | 0.7462 | 0.2536 | 0.2390 to 0.2672 |
| msmarco | own | 204 | the mask's own positive | log rank (lower is better) | 2321 | 0.3102 | 3.2084 | -2.8982 | -2.9615 to -2.8303 |
| msmarco | own | 204 | the mask's own positive | worst rank in the set | 2321 | 1.9541 | 1250.4414 | -1248.4873 | -2279.7578 to -548.5159 |
| msmarco | own | 307 | the mask's own positive | reciprocal rank | 2321 | 0.7949 | 0.1850 | 0.6099 | 0.5989 to 0.6207 |
| msmarco | own | 307 | the mask's own positive | in top 10 | 2321 | 0.9719 | 0.3916 | 0.5803 | 0.5641 to 0.5955 |
| msmarco | own | 307 | the mask's own positive | in top 100 | 2321 | 0.9997 | 0.7462 | 0.2535 | 0.2389 to 0.2670 |
| msmarco | own | 307 | the mask's own positive | log rank (lower is better) | 2321 | 0.3927 | 3.2084 | -2.8157 | -2.8775 to -2.7498 |
| msmarco | own | 307 | the mask's own positive | worst rank in the set | 2321 | 2.3890 | 1250.4414 | -1248.0524 | -2279.3071 to -548.0854 |
| msmarco | own | 409 | the mask's own positive | reciprocal rank | 2321 | 0.7495 | 0.1850 | 0.5644 | 0.5536 to 0.5756 |
| msmarco | own | 409 | the mask's own positive | in top 10 | 2321 | 0.9551 | 0.3916 | 0.5634 | 0.5478 to 0.5787 |
| msmarco | own | 409 | the mask's own positive | in top 100 | 2321 | 0.9992 | 0.7462 | 0.2530 | 0.2386 to 0.2665 |
| msmarco | own | 409 | the mask's own positive | log rank (lower is better) | 2321 | 0.5010 | 3.2084 | -2.7074 | -2.7680 to -2.6417 |
| msmarco | own | 409 | the mask's own positive | worst rank in the set | 2321 | 3.0212 | 1250.4414 | -1247.4202 | -2278.6537 to -547.3989 |
| msmarco | own | 512 | the mask's own positive | reciprocal rank | 2321 | 0.6929 | 0.1850 | 0.5078 | 0.4971 to 0.5186 |
| msmarco | own | 512 | the mask's own positive | in top 10 | 2321 | 0.9301 | 0.3916 | 0.5384 | 0.5227 to 0.5543 |
| msmarco | own | 512 | the mask's own positive | in top 100 | 2321 | 0.9977 | 0.7462 | 0.2515 | 0.2374 to 0.2650 |
| msmarco | own | 512 | the mask's own positive | log rank (lower is better) | 2321 | 0.6451 | 3.2084 | -2.5633 | -2.6204 to -2.5002 |
| msmarco | own | 512 | the mask's own positive | worst rank in the set | 2321 | 4.0423 | 1250.4414 | -1246.3990 | -2277.5780 to -546.3539 |
| msmarco | own | 614 | the mask's own positive | reciprocal rank | 2321 | 0.6388 | 0.1850 | 0.4538 | 0.4428 to 0.4646 |
| msmarco | own | 614 | the mask's own positive | in top 10 | 2321 | 0.9017 | 0.3916 | 0.5101 | 0.4945 to 0.5260 |
| msmarco | own | 614 | the mask's own positive | in top 100 | 2321 | 0.9958 | 0.7462 | 0.2496 | 0.2357 to 0.2630 |
| msmarco | own | 614 | the mask's own positive | log rank (lower is better) | 2321 | 0.7975 | 3.2084 | -2.4109 | -2.4670 to -2.3514 |
| msmarco | own | 614 | the mask's own positive | worst rank in the set | 2321 | 5.3370 | 1250.4414 | -1245.1044 | -2276.2048 to -545.0872 |
| msmarco | own | 716 | the mask's own positive | reciprocal rank | 2321 | 0.5891 | 0.1850 | 0.4040 | 0.3925 to 0.4148 |
| msmarco | own | 716 | the mask's own positive | in top 10 | 2321 | 0.8698 | 0.3916 | 0.4781 | 0.4631 to 0.4935 |
| msmarco | own | 716 | the mask's own positive | in top 100 | 2321 | 0.9927 | 0.7462 | 0.2465 | 0.2327 to 0.2599 |
| msmarco | own | 716 | the mask's own positive | log rank (lower is better) | 2321 | 0.9413 | 3.2084 | -2.2671 | -2.3219 to -2.2111 |
| msmarco | own | 716 | the mask's own positive | worst rank in the set | 2321 | 6.8117 | 1250.4414 | -1243.6297 | -2274.6659 to -543.6340 |
| msmarco | own | 819 | the mask's own positive | reciprocal rank | 2321 | 0.5283 | 0.1850 | 0.3433 | 0.3324 to 0.3533 |
| msmarco | own | 819 | the mask's own positive | in top 10 | 2321 | 0.8215 | 0.3916 | 0.4298 | 0.4152 to 0.4447 |
| msmarco | own | 819 | the mask's own positive | in top 100 | 2321 | 0.9837 | 0.7462 | 0.2375 | 0.2240 to 0.2506 |
| msmarco | own | 819 | the mask's own positive | log rank (lower is better) | 2321 | 1.1500 | 3.2084 | -2.0584 | -2.1063 to -2.0089 |
| msmarco | own | 819 | the mask's own positive | worst rank in the set | 2321 | 10.4532 | 1250.4414 | -1239.9882 | -2270.7397 to -540.0581 |
| msmarco | own | 921 | the mask's own positive | reciprocal rank | 2321 | 0.4082 | 0.1850 | 0.2231 | 0.2144 to 0.2311 |
| msmarco | own | 921 | the mask's own positive | in top 10 | 2321 | 0.7038 | 0.3916 | 0.3122 | 0.2987 to 0.3257 |
| msmarco | own | 921 | the mask's own positive | in top 100 | 2321 | 0.9493 | 0.7462 | 0.2031 | 0.1906 to 0.2150 |
| msmarco | own | 921 | the mask's own positive | log rank (lower is better) | 2321 | 1.6712 | 3.2084 | -1.5372 | -1.5750 to -1.5004 |
| msmarco | own | 921 | the mask's own positive | worst rank in the set | 2321 | 32.2790 | 1250.4414 | -1218.1624 | -2244.5610 to -520.8552 |
| msmarco | 1-1 | 102 | held-out | reciprocal rank | 2321 | 0.1271 | 0.1848 | -0.0578 | -0.0640 to -0.0517 |
| msmarco | 1-1 | 102 | held-out | in top 10 | 2321 | 0.3618 | 0.3913 | -0.0295 | -0.0411 to -0.0177 |
| msmarco | 1-1 | 102 | held-out | in top 100 | 2321 | 0.7278 | 0.7461 | -0.0183 | -0.0291 to -0.0076 |
| msmarco | 1-1 | 102 | held-out | log rank (lower is better) | 2321 | 3.4064 | 3.2094 | 0.1970 | 0.1557 to 0.2407 |
| msmarco | 1-1 | 102 | held-out | worst rank in the set | 2321 | 1208.4141 | 1249.5958 | -41.1817 | -967.7516 to 621.0914 |
| msmarco | 1-1 | 102 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1706 | 0.2225 | -0.0519 | -0.0586 to -0.0452 |
| msmarco | 1-1 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1850 | 0.2002 | -0.0153 | -0.0226 to -0.0080 |
| msmarco | 1-1 | 204 | held-out | reciprocal rank | 2321 | 0.1322 | 0.1848 | -0.0527 | -0.0588 to -0.0468 |
| msmarco | 1-1 | 204 | held-out | in top 10 | 2321 | 0.3807 | 0.3913 | -0.0106 | -0.0221 to 0.0003 |
| msmarco | 1-1 | 204 | held-out | in top 100 | 2321 | 0.7492 | 0.7461 | 0.0031 | -0.0071 to 0.0127 |
| msmarco | 1-1 | 204 | held-out | log rank (lower is better) | 2321 | 3.3054 | 3.2094 | 0.0959 | 0.0576 to 0.1355 |
| msmarco | 1-1 | 204 | held-out | worst rank in the set | 2321 | 696.0518 | 1249.5958 | -553.5441 | -1470.1167 to 29.7309 |
| msmarco | 1-1 | 204 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1790 | 0.2225 | -0.0435 | -0.0498 to -0.0373 |
| msmarco | 1-1 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1900 | 0.2002 | -0.0102 | -0.0171 to -0.0035 |
| msmarco | 1-1 | 307 | held-out | reciprocal rank | 2321 | 0.1370 | 0.1848 | -0.0479 | -0.0537 to -0.0424 |
| msmarco | 1-1 | 307 | held-out | in top 10 | 2321 | 0.3853 | 0.3913 | -0.0061 | -0.0166 to 0.0047 |
| msmarco | 1-1 | 307 | held-out | in top 100 | 2321 | 0.7552 | 0.7461 | 0.0090 | -0.0009 to 0.0186 |
| msmarco | 1-1 | 307 | held-out | log rank (lower is better) | 2321 | 3.2612 | 3.2094 | 0.0518 | 0.0169 to 0.0879 |
| msmarco | 1-1 | 307 | held-out | worst rank in the set | 2321 | 658.4948 | 1249.5958 | -591.1010 | -1484.2426 to -20.4163 |
| msmarco | 1-1 | 307 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1838 | 0.2225 | -0.0387 | -0.0444 to -0.0329 |
| msmarco | 1-1 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1936 | 0.2002 | -0.0066 | -0.0132 to -0.0003 |
| msmarco | 1-1 | 409 | held-out | reciprocal rank | 2321 | 0.1417 | 0.1848 | -0.0431 | -0.0488 to -0.0375 |
| msmarco | 1-1 | 409 | held-out | in top 10 | 2321 | 0.3917 | 0.3913 | 0.0003 | -0.0095 to 0.0102 |
| msmarco | 1-1 | 409 | held-out | in top 100 | 2321 | 0.7604 | 0.7461 | 0.0143 | 0.0048 to 0.0232 |
| msmarco | 1-1 | 409 | held-out | log rank (lower is better) | 2321 | 3.2317 | 3.2094 | 0.0223 | -0.0109 to 0.0567 |
| msmarco | 1-1 | 409 | held-out | worst rank in the set | 2321 | 672.1563 | 1249.5958 | -577.4395 | -1440.6037 to -49.6912 |
| msmarco | 1-1 | 409 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1889 | 0.2225 | -0.0336 | -0.0393 to -0.0280 |
| msmarco | 1-1 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1974 | 0.2002 | -0.0028 | -0.0091 to 0.0031 |
| msmarco | 1-1 | 512 | held-out | reciprocal rank | 2321 | 0.1460 | 0.1848 | -0.0388 | -0.0443 to -0.0336 |
| msmarco | 1-1 | 512 | held-out | in top 10 | 2321 | 0.3961 | 0.3913 | 0.0048 | -0.0046 to 0.0141 |
| msmarco | 1-1 | 512 | held-out | in top 100 | 2321 | 0.7615 | 0.7461 | 0.0154 | 0.0064 to 0.0236 |
| msmarco | 1-1 | 512 | held-out | log rank (lower is better) | 2321 | 3.2125 | 3.2094 | 0.0031 | -0.0283 to 0.0359 |
| msmarco | 1-1 | 512 | held-out | worst rank in the set | 2321 | 698.4171 | 1249.5958 | -551.1787 | -1395.4973 to -54.7193 |
| msmarco | 1-1 | 512 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1934 | 0.2225 | -0.0291 | -0.0344 to -0.0238 |
| msmarco | 1-1 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1981 | 0.2002 | -0.0021 | -0.0081 to 0.0037 |
| msmarco | 1-1 | 614 | held-out | reciprocal rank | 2321 | 0.1509 | 0.1848 | -0.0340 | -0.0393 to -0.0289 |
| msmarco | 1-1 | 614 | held-out | in top 10 | 2321 | 0.3942 | 0.3913 | 0.0029 | -0.0064 to 0.0119 |
| msmarco | 1-1 | 614 | held-out | in top 100 | 2321 | 0.7663 | 0.7461 | 0.0201 | 0.0118 to 0.0282 |
| msmarco | 1-1 | 614 | held-out | log rank (lower is better) | 2321 | 3.1972 | 3.2094 | -0.0123 | -0.0425 to 0.0186 |
| msmarco | 1-1 | 614 | held-out | worst rank in the set | 2321 | 657.2932 | 1249.5958 | -592.3026 | -1439.9340 to -78.5581 |
| msmarco | 1-1 | 614 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.1965 | 0.2225 | -0.0260 | -0.0313 to -0.0207 |
| msmarco | 1-1 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1989 | 0.2002 | -0.0014 | -0.0069 to 0.0040 |
| msmarco | 1-1 | 716 | held-out | reciprocal rank | 2321 | 0.1550 | 0.1848 | -0.0298 | -0.0350 to -0.0251 |
| msmarco | 1-1 | 716 | held-out | in top 10 | 2321 | 0.3967 | 0.3913 | 0.0054 | -0.0035 to 0.0140 |
| msmarco | 1-1 | 716 | held-out | in top 100 | 2321 | 0.7671 | 0.7461 | 0.0210 | 0.0130 to 0.0286 |
| msmarco | 1-1 | 716 | held-out | log rank (lower is better) | 2321 | 3.1874 | 3.2094 | -0.0221 | -0.0503 to 0.0066 |
| msmarco | 1-1 | 716 | held-out | worst rank in the set | 2321 | 653.9015 | 1249.5958 | -595.6943 | -1433.8428 to -90.5098 |
| msmarco | 1-1 | 716 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.2004 | 0.2225 | -0.0221 | -0.0272 to -0.0173 |
| msmarco | 1-1 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1997 | 0.2002 | -0.0005 | -0.0057 to 0.0046 |
| msmarco | 1-1 | 819 | held-out | reciprocal rank | 2321 | 0.1576 | 0.1848 | -0.0272 | -0.0320 to -0.0229 |
| msmarco | 1-1 | 819 | held-out | in top 10 | 2321 | 0.3959 | 0.3913 | 0.0046 | -0.0039 to 0.0132 |
| msmarco | 1-1 | 819 | held-out | in top 100 | 2321 | 0.7681 | 0.7461 | 0.0220 | 0.0145 to 0.0296 |
| msmarco | 1-1 | 819 | held-out | log rank (lower is better) | 2321 | 3.1793 | 3.2094 | -0.0301 | -0.0563 to -0.0040 |
| msmarco | 1-1 | 819 | held-out | worst rank in the set | 2321 | 670.3369 | 1249.5958 | -579.2589 | -1396.5575 to -99.3513 |
| msmarco | 1-1 | 819 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.2021 | 0.2225 | -0.0204 | -0.0251 to -0.0158 |
| msmarco | 1-1 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.1990 | 0.2002 | -0.0012 | -0.0061 to 0.0035 |
| msmarco | 1-1 | 921 | held-out | reciprocal rank | 2321 | 0.1674 | 0.1848 | -0.0174 | -0.0217 to -0.0135 |
| msmarco | 1-1 | 921 | held-out | in top 10 | 2321 | 0.3981 | 0.3913 | 0.0067 | -0.0004 to 0.0140 |
| msmarco | 1-1 | 921 | held-out | in top 100 | 2321 | 0.7685 | 0.7461 | 0.0224 | 0.0158 to 0.0288 |
| msmarco | 1-1 | 921 | held-out | log rank (lower is better) | 2321 | 3.1629 | 3.2094 | -0.0465 | -0.0667 to -0.0267 |
| msmarco | 1-1 | 921 | held-out | worst rank in the set | 2321 | 711.3463 | 1249.5958 | -538.2496 | -1253.9266 to -111.8249 |
| msmarco | 1-1 | 921 | held-out | nDCG@10, restricted to held-out labels | 2321 | 0.2102 | 0.2225 | -0.0123 | -0.0164 to -0.0084 |
| msmarco | 1-1 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 2321 | 0.2000 | 0.2002 | -0.0003 | -0.0043 to 0.0038 |
| msmarco | 2-2 | 102 | held-out | reciprocal rank | 81 | 0.0833 | 0.1012 | -0.0179 | -0.0332 to -0.0028 |
| msmarco | 2-2 | 102 | held-out | in top 10 | 81 | 0.2001 | 0.2215 | -0.0213 | -0.0479 to 0.0035 |
| msmarco | 2-2 | 102 | held-out | in top 100 | 81 | 0.6402 | 0.5934 | 0.0468 | -0.0022 to 0.0996 |
| msmarco | 2-2 | 102 | held-out | log rank (lower is better) | 81 | 3.9734 | 4.0342 | -0.0608 | -0.2206 to 0.0798 |
| msmarco | 2-2 | 102 | held-out | worst rank in the set | 81 | 1450.7356 | 1595.1883 | -144.4527 | -911.6489 to 504.9452 |
| msmarco | 2-2 | 102 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1183 | 0.1441 | -0.0258 | -0.0440 to -0.0080 |
| msmarco | 2-2 | 102 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1159 | 0.1095 | 0.0064 | -0.0099 to 0.0233 |
| msmarco | 2-2 | 204 | held-out | reciprocal rank | 81 | 0.0887 | 0.1012 | -0.0125 | -0.0277 to 0.0023 |
| msmarco | 2-2 | 204 | held-out | in top 10 | 81 | 0.2173 | 0.2215 | -0.0041 | -0.0307 to 0.0205 |
| msmarco | 2-2 | 204 | held-out | in top 100 | 81 | 0.6602 | 0.5934 | 0.0669 | 0.0211 to 0.1174 |
| msmarco | 2-2 | 204 | held-out | log rank (lower is better) | 81 | 3.8795 | 4.0342 | -0.1547 | -0.3101 to -0.0248 |
| msmarco | 2-2 | 204 | held-out | worst rank in the set | 81 | 1143.8601 | 1595.1883 | -451.3282 | -1431.4914 to 132.1880 |
| msmarco | 2-2 | 204 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1283 | 0.1441 | -0.0158 | -0.0331 to 0.0019 |
| msmarco | 2-2 | 204 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1198 | 0.1095 | 0.0103 | -0.0045 to 0.0252 |
| msmarco | 2-2 | 307 | held-out | reciprocal rank | 81 | 0.0892 | 0.1012 | -0.0119 | -0.0258 to 0.0017 |
| msmarco | 2-2 | 307 | held-out | in top 10 | 81 | 0.2225 | 0.2215 | 0.0010 | -0.0248 to 0.0254 |
| msmarco | 2-2 | 307 | held-out | in top 100 | 81 | 0.6597 | 0.5934 | 0.0664 | 0.0239 to 0.1141 |
| msmarco | 2-2 | 307 | held-out | log rank (lower is better) | 81 | 3.8557 | 4.0342 | -0.1785 | -0.3224 to -0.0568 |
| msmarco | 2-2 | 307 | held-out | worst rank in the set | 81 | 1056.6543 | 1595.1883 | -538.5339 | -1639.3839 to 92.5523 |
| msmarco | 2-2 | 307 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1305 | 0.1441 | -0.0136 | -0.0294 to 0.0028 |
| msmarco | 2-2 | 307 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1199 | 0.1095 | 0.0104 | -0.0026 to 0.0240 |
| msmarco | 2-2 | 409 | held-out | reciprocal rank | 81 | 0.0930 | 0.1012 | -0.0082 | -0.0209 to 0.0047 |
| msmarco | 2-2 | 409 | held-out | in top 10 | 81 | 0.2281 | 0.2215 | 0.0067 | -0.0184 to 0.0304 |
| msmarco | 2-2 | 409 | held-out | in top 100 | 81 | 0.6590 | 0.5934 | 0.0656 | 0.0256 to 0.1092 |
| msmarco | 2-2 | 409 | held-out | log rank (lower is better) | 81 | 3.8583 | 4.0342 | -0.1759 | -0.3088 to -0.0615 |
| msmarco | 2-2 | 409 | held-out | worst rank in the set | 81 | 1102.4218 | 1595.1883 | -492.7665 | -1508.5731 to 102.1527 |
| msmarco | 2-2 | 409 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1360 | 0.1441 | -0.0081 | -0.0234 to 0.0078 |
| msmarco | 2-2 | 409 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1203 | 0.1095 | 0.0108 | -0.0022 to 0.0243 |
| msmarco | 2-2 | 512 | held-out | reciprocal rank | 81 | 0.0938 | 0.1012 | -0.0074 | -0.0188 to 0.0049 |
| msmarco | 2-2 | 512 | held-out | in top 10 | 81 | 0.2258 | 0.2215 | 0.0044 | -0.0206 to 0.0275 |
| msmarco | 2-2 | 512 | held-out | in top 100 | 81 | 0.6590 | 0.5934 | 0.0656 | 0.0266 to 0.1072 |
| msmarco | 2-2 | 512 | held-out | log rank (lower is better) | 81 | 3.8712 | 4.0342 | -0.1630 | -0.2889 to -0.0542 |
| msmarco | 2-2 | 512 | held-out | worst rank in the set | 81 | 1146.7454 | 1595.1883 | -448.4429 | -1400.6514 to 122.4809 |
| msmarco | 2-2 | 512 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1364 | 0.1441 | -0.0077 | -0.0216 to 0.0070 |
| msmarco | 2-2 | 512 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1199 | 0.1095 | 0.0104 | -0.0012 to 0.0225 |
| msmarco | 2-2 | 614 | held-out | reciprocal rank | 81 | 0.0946 | 0.1012 | -0.0066 | -0.0178 to 0.0048 |
| msmarco | 2-2 | 614 | held-out | in top 10 | 81 | 0.2261 | 0.2215 | 0.0046 | -0.0197 to 0.0276 |
| msmarco | 2-2 | 614 | held-out | in top 100 | 81 | 0.6515 | 0.5934 | 0.0581 | 0.0211 to 0.0976 |
| msmarco | 2-2 | 614 | held-out | log rank (lower is better) | 81 | 3.8744 | 4.0342 | -0.1598 | -0.2793 to -0.0578 |
| msmarco | 2-2 | 614 | held-out | worst rank in the set | 81 | 1156.5288 | 1595.1883 | -438.6595 | -1369.5680 to 122.2087 |
| msmarco | 2-2 | 614 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1374 | 0.1441 | -0.0067 | -0.0207 to 0.0074 |
| msmarco | 2-2 | 614 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1189 | 0.1095 | 0.0094 | -0.0015 to 0.0208 |
| msmarco | 2-2 | 716 | held-out | reciprocal rank | 81 | 0.0946 | 0.1012 | -0.0066 | -0.0174 to 0.0041 |
| msmarco | 2-2 | 716 | held-out | in top 10 | 81 | 0.2225 | 0.2215 | 0.0010 | -0.0211 to 0.0216 |
| msmarco | 2-2 | 716 | held-out | in top 100 | 81 | 0.6438 | 0.5934 | 0.0504 | 0.0171 to 0.0872 |
| msmarco | 2-2 | 716 | held-out | log rank (lower is better) | 81 | 3.8796 | 4.0342 | -0.1546 | -0.2698 to -0.0571 |
| msmarco | 2-2 | 716 | held-out | worst rank in the set | 81 | 1131.4485 | 1595.1883 | -463.7397 | -1422.3527 to 95.8645 |
| msmarco | 2-2 | 716 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1363 | 0.1441 | -0.0078 | -0.0212 to 0.0057 |
| msmarco | 2-2 | 716 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1174 | 0.1095 | 0.0079 | -0.0027 to 0.0192 |
| msmarco | 2-2 | 819 | held-out | reciprocal rank | 81 | 0.0975 | 0.1012 | -0.0036 | -0.0139 to 0.0067 |
| msmarco | 2-2 | 819 | held-out | in top 10 | 81 | 0.2209 | 0.2215 | -0.0005 | -0.0222 to 0.0189 |
| msmarco | 2-2 | 819 | held-out | in top 100 | 81 | 0.6394 | 0.5934 | 0.0460 | 0.0130 to 0.0833 |
| msmarco | 2-2 | 819 | held-out | log rank (lower is better) | 81 | 3.8863 | 4.0342 | -0.1479 | -0.2512 to -0.0585 |
| msmarco | 2-2 | 819 | held-out | worst rank in the set | 81 | 1151.1919 | 1595.1883 | -443.9964 | -1370.1628 to 102.2465 |
| msmarco | 2-2 | 819 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1387 | 0.1441 | -0.0054 | -0.0185 to 0.0073 |
| msmarco | 2-2 | 819 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1178 | 0.1095 | 0.0083 | -0.0018 to 0.0190 |
| msmarco | 2-2 | 921 | held-out | reciprocal rank | 81 | 0.0983 | 0.1012 | -0.0029 | -0.0115 to 0.0055 |
| msmarco | 2-2 | 921 | held-out | in top 10 | 81 | 0.2209 | 0.2215 | -0.0005 | -0.0234 to 0.0199 |
| msmarco | 2-2 | 921 | held-out | in top 100 | 81 | 0.6381 | 0.5934 | 0.0448 | 0.0141 to 0.0791 |
| msmarco | 2-2 | 921 | held-out | log rank (lower is better) | 81 | 3.9156 | 4.0342 | -0.1186 | -0.1989 to -0.0471 |
| msmarco | 2-2 | 921 | held-out | worst rank in the set | 81 | 1174.1800 | 1595.1883 | -421.0082 | -1258.6538 to 55.7784 |
| msmarco | 2-2 | 921 | held-out | nDCG@10, restricted to held-out labels | 81 | 0.1400 | 0.1441 | -0.0041 | -0.0163 to 0.0075 |
| msmarco | 2-2 | 921 | held-out | reciprocal rank, construction positives skipped (secondary) | 81 | 0.1146 | 0.1095 | 0.0051 | -0.0031 to 0.0140 |

## Gains and losses per query

| environment | arm | k | queries | mean fraction of positives improved | mean fraction worsened | worst positive's rank change (median) | queries whose worst positive worsened |
|---|---|---|---|---|---|---|---|
| scifact | prefix | 102 | 69 | 0.1652 | 0.5643 | 3.0 | 0.6957 |
| scifact | prefix | 204 | 69 | 0.1737 | 0.4556 | 2.0 | 0.6087 |
| scifact | prefix | 307 | 69 | 0.1845 | 0.3911 | 1.0 | 0.5507 |
| scifact | prefix | 409 | 69 | 0.1845 | 0.3901 | 1.0 | 0.5217 |
| scifact | prefix | 512 | 69 | 0.2046 | 0.3640 | 0.0 | 0.4493 |
| scifact | prefix | 614 | 69 | 0.1766 | 0.3266 | 0.0 | 0.3623 |
| scifact | prefix | 716 | 69 | 0.1705 | 0.2510 | 0.0 | 0.3188 |
| scifact | prefix | 819 | 69 | 0.1853 | 0.2449 | 0.0 | 0.3043 |
| scifact | prefix | 921 | 69 | 0.1628 | 0.1326 | 0.0 | 0.1594 |
| scifact | shared | 102 | 69 | 0.6162 | 0.0831 | -3.0 | 0.0000 |
| scifact | shared | 204 | 69 | 0.6167 | 0.0536 | -2.0 | 0.0000 |
| scifact | shared | 307 | 69 | 0.5957 | 0.0459 | -2.0 | 0.0000 |
| scifact | shared | 409 | 69 | 0.5667 | 0.0338 | -2.0 | 0.0000 |
| scifact | shared | 512 | 69 | 0.5558 | 0.0229 | -2.0 | 0.0000 |
| scifact | shared | 614 | 69 | 0.5510 | 0.0145 | -2.0 | 0.0000 |
| scifact | shared | 716 | 69 | 0.5437 | 0.0145 | -2.0 | 0.0000 |
| scifact | shared | 819 | 69 | 0.5425 | 0.0217 | -2.0 | 0.0000 |
| scifact | shared | 921 | 69 | 0.5208 | 0.0217 | -1.0 | 0.0000 |
| scifact | own | 102 | 69 | 0.7278 | 0.0000 | -2.5 | 0.0000 |
| scifact | own | 204 | 69 | 0.7350 | 0.0000 | -2.5 | 0.0000 |
| scifact | own | 307 | 69 | 0.7350 | 0.0000 | -2.5 | 0.0000 |
| scifact | own | 409 | 69 | 0.7024 | 0.0000 | -2.5 | 0.0000 |
| scifact | own | 512 | 69 | 0.7024 | 0.0000 | -2.5 | 0.0000 |
| scifact | own | 614 | 69 | 0.7097 | 0.0000 | -2.5 | 0.0000 |
| scifact | own | 716 | 69 | 0.6915 | 0.0000 | -2.5 | 0.0000 |
| scifact | own | 819 | 69 | 0.6843 | 0.0000 | -2.0 | 0.0000 |
| scifact | own | 921 | 69 | 0.6722 | 0.0000 | -2.0 | 0.0000 |
| scifact | 1-1 | 102 | 69 | 0.2971 | 0.4366 | 0.2 | 0.5217 |
| scifact | 1-1 | 204 | 69 | 0.3056 | 0.3986 | 0.2 | 0.5507 |
| scifact | 1-1 | 307 | 69 | 0.3176 | 0.3986 | 0.2 | 0.5072 |
| scifact | 1-1 | 409 | 69 | 0.3104 | 0.3418 | 0.0 | 0.4783 |
| scifact | 1-1 | 512 | 69 | 0.3152 | 0.3164 | 0.0 | 0.4493 |
| scifact | 1-1 | 614 | 69 | 0.3062 | 0.3056 | 0.0 | 0.4493 |
| scifact | 1-1 | 716 | 69 | 0.3007 | 0.3007 | 0.0 | 0.4928 |
| scifact | 1-1 | 819 | 69 | 0.3104 | 0.2748 | 0.0 | 0.4203 |
| scifact | 1-1 | 921 | 69 | 0.2965 | 0.2210 | 0.0 | 0.3623 |
| scifact | 2-2 | 102 | 6 | 0.3090 | 0.4965 | 0.1 | 0.5000 |
| scifact | 2-2 | 204 | 6 | 0.2535 | 0.4861 | 0.0 | 0.5000 |
| scifact | 2-2 | 307 | 6 | 0.2674 | 0.4653 | -0.2 | 0.1667 |
| scifact | 2-2 | 409 | 6 | 0.2951 | 0.4097 | -0.2 | 0.1667 |
| scifact | 2-2 | 512 | 6 | 0.2674 | 0.3542 | -0.1 | 0.3333 |
| scifact | 2-2 | 614 | 6 | 0.2708 | 0.3125 | -0.2 | 0.3333 |
| scifact | 2-2 | 716 | 6 | 0.2465 | 0.2986 | -0.2 | 0.1667 |
| scifact | 2-2 | 819 | 6 | 0.2743 | 0.2812 | -0.5 | 0.1667 |
| scifact | 2-2 | 921 | 6 | 0.2743 | 0.1701 | -0.4 | 0.1667 |
| nfcorpus | prefix | 102 | 2162 | 0.3692 | 0.5948 | 37.0 | 0.5800 |
| nfcorpus | prefix | 204 | 2162 | 0.3903 | 0.5611 | 17.5 | 0.5587 |
| nfcorpus | prefix | 307 | 2162 | 0.3918 | 0.5499 | 14.0 | 0.5564 |
| nfcorpus | prefix | 409 | 2162 | 0.4040 | 0.5326 | 17.0 | 0.5620 |
| nfcorpus | prefix | 512 | 2162 | 0.4168 | 0.5143 | 7.0 | 0.5472 |
| nfcorpus | prefix | 614 | 2162 | 0.4219 | 0.4993 | 3.0 | 0.5241 |
| nfcorpus | prefix | 716 | 2162 | 0.4223 | 0.4908 | 3.0 | 0.5231 |
| nfcorpus | prefix | 819 | 2162 | 0.4201 | 0.4803 | 2.0 | 0.5153 |
| nfcorpus | prefix | 921 | 2162 | 0.4120 | 0.4709 | 4.0 | 0.5370 |
| nfcorpus | shared | 102 | 2162 | 0.7805 | 0.1699 | -506.5 | 0.1156 |
| nfcorpus | shared | 204 | 2162 | 0.8109 | 0.1335 | -563.0 | 0.0916 |
| nfcorpus | shared | 307 | 2162 | 0.8225 | 0.1185 | -553.0 | 0.0796 |
| nfcorpus | shared | 409 | 2162 | 0.8279 | 0.1118 | -554.5 | 0.0759 |
| nfcorpus | shared | 512 | 2162 | 0.8308 | 0.1074 | -511.0 | 0.0685 |
| nfcorpus | shared | 614 | 2162 | 0.8313 | 0.1038 | -480.5 | 0.0666 |
| nfcorpus | shared | 716 | 2162 | 0.8305 | 0.1019 | -464.0 | 0.0666 |
| nfcorpus | shared | 819 | 2162 | 0.8301 | 0.0998 | -419.5 | 0.0620 |
| nfcorpus | shared | 921 | 2162 | 0.8236 | 0.0982 | -317.5 | 0.0712 |
| nfcorpus | own | 102 | 2162 | 0.9630 | 0.0000 | -886.7 | 0.0000 |
| nfcorpus | own | 204 | 2162 | 0.9629 | 0.0000 | -886.7 | 0.0000 |
| nfcorpus | own | 307 | 2162 | 0.9629 | 0.0000 | -886.2 | 0.0000 |
| nfcorpus | own | 409 | 2162 | 0.9627 | 0.0000 | -885.6 | 0.0000 |
| nfcorpus | own | 512 | 2162 | 0.9625 | 0.0000 | -885.4 | 0.0000 |
| nfcorpus | own | 614 | 2162 | 0.9622 | 0.0000 | -885.0 | 0.0000 |
| nfcorpus | own | 716 | 2162 | 0.9617 | 0.0000 | -884.2 | 0.0000 |
| nfcorpus | own | 819 | 2162 | 0.9606 | 0.0000 | -880.2 | 0.0000 |
| nfcorpus | own | 921 | 2162 | 0.9570 | 0.0000 | -837.7 | 0.0000 |
| nfcorpus | 1-1 | 102 | 2162 | 0.5488 | 0.4296 | -82.0 | 0.3599 |
| nfcorpus | 1-1 | 204 | 2162 | 0.5642 | 0.4131 | -106.2 | 0.3131 |
| nfcorpus | 1-1 | 307 | 2162 | 0.5750 | 0.3964 | -114.3 | 0.2979 |
| nfcorpus | 1-1 | 409 | 2162 | 0.5765 | 0.3904 | -111.9 | 0.2831 |
| nfcorpus | 1-1 | 512 | 2162 | 0.5812 | 0.3817 | -109.5 | 0.2655 |
| nfcorpus | 1-1 | 614 | 2162 | 0.5840 | 0.3751 | -108.5 | 0.2581 |
| nfcorpus | 1-1 | 716 | 2162 | 0.5902 | 0.3653 | -107.8 | 0.2521 |
| nfcorpus | 1-1 | 819 | 2162 | 0.5965 | 0.3493 | -102.5 | 0.2410 |
| nfcorpus | 1-1 | 921 | 2162 | 0.6028 | 0.3253 | -85.1 | 0.2239 |
| nfcorpus | 2-2 | 102 | 1922 | 0.6009 | 0.3756 | -189.9 | 0.2794 |
| nfcorpus | 2-2 | 204 | 1922 | 0.6234 | 0.3483 | -216.2 | 0.2263 |
| nfcorpus | 2-2 | 307 | 1922 | 0.6354 | 0.3334 | -221.6 | 0.2040 |
| nfcorpus | 2-2 | 409 | 1922 | 0.6416 | 0.3244 | -230.2 | 0.1930 |
| nfcorpus | 2-2 | 512 | 1922 | 0.6452 | 0.3184 | -228.1 | 0.1800 |
| nfcorpus | 2-2 | 614 | 1922 | 0.6474 | 0.3135 | -216.2 | 0.1816 |
| nfcorpus | 2-2 | 716 | 1922 | 0.6508 | 0.3071 | -209.4 | 0.1722 |
| nfcorpus | 2-2 | 819 | 1922 | 0.6534 | 0.2996 | -196.5 | 0.1675 |
| nfcorpus | 2-2 | 921 | 1922 | 0.6569 | 0.2826 | -158.1 | 0.1623 |
| nfcorpus | 5-5 | 102 | 1517 | 0.6228 | 0.3542 | -277.4 | 0.2281 |
| nfcorpus | 5-5 | 204 | 1517 | 0.6568 | 0.3162 | -321.9 | 0.1575 |
| nfcorpus | 5-5 | 307 | 1517 | 0.6686 | 0.3010 | -330.1 | 0.1299 |
| nfcorpus | 5-5 | 409 | 1517 | 0.6770 | 0.2900 | -328.9 | 0.1259 |
| nfcorpus | 5-5 | 512 | 1517 | 0.6818 | 0.2827 | -323.4 | 0.1200 |
| nfcorpus | 5-5 | 614 | 1517 | 0.6844 | 0.2772 | -318.5 | 0.1101 |
| nfcorpus | 5-5 | 716 | 1517 | 0.6871 | 0.2723 | -299.1 | 0.1154 |
| nfcorpus | 5-5 | 819 | 1517 | 0.6885 | 0.2670 | -269.4 | 0.1074 |
| nfcorpus | 5-5 | 921 | 1517 | 0.6891 | 0.2566 | -206.9 | 0.0962 |
| nfcorpus | 10-10 | 102 | 1105 | 0.6172 | 0.3634 | -241.0 | 0.2172 |
| nfcorpus | 10-10 | 204 | 1105 | 0.6577 | 0.3191 | -287.1 | 0.1557 |
| nfcorpus | 10-10 | 307 | 1105 | 0.6751 | 0.2989 | -282.8 | 0.1348 |
| nfcorpus | 10-10 | 409 | 1105 | 0.6830 | 0.2882 | -281.1 | 0.1258 |
| nfcorpus | 10-10 | 512 | 1105 | 0.6878 | 0.2810 | -268.5 | 0.1104 |
| nfcorpus | 10-10 | 614 | 1105 | 0.6902 | 0.2768 | -249.0 | 0.1113 |
| nfcorpus | 10-10 | 716 | 1105 | 0.6914 | 0.2741 | -241.8 | 0.1104 |
| nfcorpus | 10-10 | 819 | 1105 | 0.6930 | 0.2691 | -221.1 | 0.1113 |
| nfcorpus | 10-10 | 921 | 1105 | 0.6900 | 0.2624 | -167.5 | 0.1104 |
| nfcorpus | 20-20 | 102 | 611 | 0.5945 | 0.3900 | -140.6 | 0.2226 |
| nfcorpus | 20-20 | 204 | 611 | 0.6349 | 0.3453 | -156.0 | 0.1784 |
| nfcorpus | 20-20 | 307 | 611 | 0.6525 | 0.3253 | -155.6 | 0.1604 |
| nfcorpus | 20-20 | 409 | 611 | 0.6643 | 0.3116 | -164.5 | 0.1391 |
| nfcorpus | 20-20 | 512 | 611 | 0.6699 | 0.3036 | -160.9 | 0.1293 |
| nfcorpus | 20-20 | 614 | 611 | 0.6724 | 0.2992 | -151.6 | 0.1326 |
| nfcorpus | 20-20 | 716 | 611 | 0.6738 | 0.2969 | -147.9 | 0.1277 |
| nfcorpus | 20-20 | 819 | 611 | 0.6747 | 0.2926 | -129.1 | 0.1309 |
| nfcorpus | 20-20 | 921 | 611 | 0.6708 | 0.2894 | -97.2 | 0.1408 |
| msmarco | prefix | 102 | 2321 | 0.2192 | 0.7158 | 79.0 | 0.8018 |
| msmarco | prefix | 204 | 2321 | 0.2756 | 0.6257 | 19.0 | 0.7036 |
| msmarco | prefix | 307 | 2321 | 0.2972 | 0.5822 | 8.0 | 0.6609 |
| msmarco | prefix | 409 | 2321 | 0.3224 | 0.5360 | 4.0 | 0.6036 |
| msmarco | prefix | 512 | 2321 | 0.3374 | 0.4994 | 2.0 | 0.5597 |
| msmarco | prefix | 614 | 2321 | 0.3405 | 0.4715 | 1.0 | 0.5330 |
| msmarco | prefix | 716 | 2321 | 0.3592 | 0.4277 | 0.0 | 0.4838 |
| msmarco | prefix | 819 | 2321 | 0.3572 | 0.3951 | 0.0 | 0.4649 |
| msmarco | prefix | 921 | 2321 | 0.3443 | 0.3506 | 0.0 | 0.4115 |
| msmarco | shared | 102 | 2321 | 0.8487 | 0.0267 | -43.0 | 0.0099 |
| msmarco | shared | 204 | 2321 | 0.8593 | 0.0154 | -43.0 | 0.0017 |
| msmarco | shared | 307 | 2321 | 0.8634 | 0.0090 | -42.0 | 0.0013 |
| msmarco | shared | 409 | 2321 | 0.8605 | 0.0071 | -40.0 | 0.0013 |
| msmarco | shared | 512 | 2321 | 0.8559 | 0.0049 | -38.0 | 0.0009 |
| msmarco | shared | 614 | 2321 | 0.8503 | 0.0039 | -37.0 | 0.0013 |
| msmarco | shared | 716 | 2321 | 0.8469 | 0.0028 | -35.0 | 0.0000 |
| msmarco | shared | 819 | 2321 | 0.8417 | 0.0023 | -33.0 | 0.0000 |
| msmarco | shared | 921 | 2321 | 0.8269 | 0.0029 | -29.0 | 0.0000 |
| msmarco | own | 102 | 2321 | 0.9072 | 0.0004 | -34.0 | 0.0000 |
| msmarco | own | 204 | 2321 | 0.9079 | 0.0002 | -34.0 | 0.0000 |
| msmarco | own | 307 | 2321 | 0.9070 | 0.0002 | -34.0 | 0.0000 |
| msmarco | own | 409 | 2321 | 0.9056 | 0.0000 | -34.0 | 0.0000 |
| msmarco | own | 512 | 2321 | 0.9020 | 0.0002 | -33.5 | 0.0000 |
| msmarco | own | 614 | 2321 | 0.8990 | 0.0000 | -33.0 | 0.0000 |
| msmarco | own | 716 | 2321 | 0.8970 | 0.0000 | -32.0 | 0.0000 |
| msmarco | own | 819 | 2321 | 0.8942 | 0.0000 | -31.5 | 0.0000 |
| msmarco | own | 921 | 2321 | 0.8840 | 0.0000 | -28.0 | 0.0000 |
| msmarco | 1-1 | 102 | 2321 | 0.3631 | 0.5658 | 1.5 | 0.5872 |
| msmarco | 1-1 | 204 | 2321 | 0.3782 | 0.5360 | 1.0 | 0.5575 |
| msmarco | 1-1 | 307 | 2321 | 0.3904 | 0.5203 | 0.5 | 0.5330 |
| msmarco | 1-1 | 409 | 2321 | 0.3961 | 0.5063 | 0.5 | 0.5157 |
| msmarco | 1-1 | 512 | 2321 | 0.4005 | 0.4894 | 0.5 | 0.5075 |
| msmarco | 1-1 | 614 | 2321 | 0.4039 | 0.4775 | 0.0 | 0.4972 |
| msmarco | 1-1 | 716 | 2321 | 0.4079 | 0.4585 | 0.0 | 0.4907 |
| msmarco | 1-1 | 819 | 2321 | 0.4127 | 0.4479 | 0.0 | 0.4739 |
| msmarco | 1-1 | 921 | 2321 | 0.4220 | 0.3989 | 0.0 | 0.4347 |
| msmarco | 2-2 | 102 | 81 | 0.5010 | 0.4511 | 1.5 | 0.5185 |
| msmarco | 2-2 | 204 | 81 | 0.5442 | 0.3966 | -5.0 | 0.3704 |
| msmarco | 2-2 | 307 | 81 | 0.5576 | 0.3871 | -8.0 | 0.3457 |
| msmarco | 2-2 | 409 | 81 | 0.5661 | 0.3719 | -8.7 | 0.3210 |
| msmarco | 2-2 | 512 | 81 | 0.5543 | 0.3711 | -11.5 | 0.3210 |
| msmarco | 2-2 | 614 | 81 | 0.5471 | 0.3673 | -12.0 | 0.2963 |
| msmarco | 2-2 | 716 | 81 | 0.5504 | 0.3583 | -9.1 | 0.2840 |
| msmarco | 2-2 | 819 | 81 | 0.5445 | 0.3524 | -10.2 | 0.2840 |
| msmarco | 2-2 | 921 | 81 | 0.5381 | 0.3266 | -10.0 | 0.2963 |

## Similarity against competition

Differences from the full query, on the same documents. A positive whose score rises but whose rank does not, means other documents rose as much or more.

| environment | arm | k | positive score change | score change at rank 10 of the list | score change at rank 100 | corpus mean score change | positive-minus-rank-10 margin change | positive z-score change | mean rank change (log) |
|---|---|---|---|---|---|---|---|---|---|
| scifact | prefix | 102 | -0.5627 | -0.4833 | -0.3726 | -0.2196 | -0.0794 | -1.665 | 0.577 |
| scifact | prefix | 204 | -0.5122 | -0.4393 | -0.3367 | -0.1969 | -0.0729 | -1.247 | 0.314 |
| scifact | prefix | 307 | -0.4514 | -0.3858 | -0.2933 | -0.1717 | -0.0656 | -1.132 | 0.208 |
| scifact | prefix | 409 | -0.3809 | -0.3244 | -0.2452 | -0.1429 | -0.0565 | -0.852 | 0.119 |
| scifact | prefix | 512 | -0.3048 | -0.2584 | -0.1935 | -0.1110 | -0.0463 | -0.660 | 0.102 |
| scifact | prefix | 614 | -0.2434 | -0.2066 | -0.1542 | -0.0882 | -0.0368 | -0.454 | 0.063 |
| scifact | prefix | 716 | -0.1825 | -0.1543 | -0.1149 | -0.0657 | -0.0283 | -0.335 | 0.040 |
| scifact | prefix | 819 | -0.1167 | -0.0974 | -0.0713 | -0.0399 | -0.0193 | -0.230 | 0.024 |
| scifact | prefix | 921 | -0.0535 | -0.0440 | -0.0304 | -0.0141 | -0.0095 | -0.100 | 0.013 |
| scifact | shared | 102 | -0.3094 | -0.2922 | -0.2245 | -0.1281 | -0.0172 | 0.361 | -0.738 |
| scifact | shared | 204 | -0.1590 | -0.1729 | -0.1319 | -0.0718 | 0.0139 | 0.611 | -0.724 |
| scifact | shared | 307 | -0.0681 | -0.0994 | -0.0764 | -0.0392 | 0.0312 | 0.703 | -0.687 |
| scifact | shared | 409 | -0.0135 | -0.0535 | -0.0413 | -0.0182 | 0.0401 | 0.714 | -0.666 |
| scifact | shared | 512 | 0.0187 | -0.0239 | -0.0181 | -0.0043 | 0.0426 | 0.669 | -0.647 |
| scifact | shared | 614 | 0.0353 | -0.0080 | -0.0058 | 0.0032 | 0.0433 | 0.633 | -0.621 |
| scifact | shared | 716 | 0.0417 | -0.0009 | 0.0002 | 0.0063 | 0.0426 | 0.604 | -0.584 |
| scifact | shared | 819 | 0.0418 | 0.0019 | 0.0023 | 0.0071 | 0.0399 | 0.547 | -0.540 |
| scifact | shared | 921 | 0.0349 | 0.0027 | 0.0033 | 0.0065 | 0.0321 | 0.425 | -0.416 |
| scifact | own | 102 | -0.2949 | n/a | n/a | -0.1335 | n/a | 0.987 | -1.265 |
| scifact | own | 204 | -0.1391 | n/a | n/a | -0.0789 | n/a | 1.220 | -1.270 |
| scifact | own | 307 | -0.0465 | n/a | n/a | -0.0458 | n/a | 1.285 | -1.261 |
| scifact | own | 409 | 0.0087 | n/a | n/a | -0.0248 | n/a | 1.262 | -1.204 |
| scifact | own | 512 | 0.0406 | n/a | n/a | -0.0120 | n/a | 1.211 | -1.159 |
| scifact | own | 614 | 0.0561 | n/a | n/a | -0.0045 | n/a | 1.134 | -1.124 |
| scifact | own | 716 | 0.0613 | n/a | n/a | -0.0010 | n/a | 1.067 | -1.098 |
| scifact | own | 819 | 0.0598 | n/a | n/a | 0.0005 | n/a | 0.976 | -0.998 |
| scifact | own | 921 | 0.0485 | n/a | n/a | 0.0019 | n/a | 0.735 | -0.804 |
| scifact | 1-1 | 102 | -0.3454 | -0.3017 | -0.2320 | -0.1335 | -0.0437 | -0.294 | 0.111 |
| scifact | 1-1 | 204 | -0.2084 | -0.1856 | -0.1420 | -0.0790 | -0.0228 | -0.052 | 0.067 |
| scifact | 1-1 | 307 | -0.1238 | -0.1147 | -0.0871 | -0.0458 | -0.0091 | 0.071 | 0.044 |
| scifact | 1-1 | 409 | -0.0698 | -0.0688 | -0.0520 | -0.0249 | -0.0010 | 0.130 | 0.009 |
| scifact | 1-1 | 512 | -0.0369 | -0.0405 | -0.0303 | -0.0120 | 0.0036 | 0.153 | -0.018 |
| scifact | 1-1 | 614 | -0.0180 | -0.0241 | -0.0175 | -0.0045 | 0.0062 | 0.152 | -0.021 |
| scifact | 1-1 | 716 | -0.0093 | -0.0161 | -0.0114 | -0.0010 | 0.0068 | 0.145 | 0.000 |
| scifact | 1-1 | 819 | -0.0051 | -0.0120 | -0.0082 | 0.0005 | 0.0069 | 0.139 | -0.038 |
| scifact | 1-1 | 921 | -0.0005 | -0.0066 | -0.0043 | 0.0019 | 0.0061 | 0.114 | -0.033 |
| scifact | 2-2 | 102 | -0.3560 | -0.3111 | -0.2147 | -0.1187 | -0.0449 | -0.373 | 0.205 |
| scifact | 2-2 | 204 | -0.2084 | -0.1798 | -0.1243 | -0.0677 | -0.0286 | -0.093 | 0.191 |
| scifact | 2-2 | 307 | -0.1165 | -0.1001 | -0.0711 | -0.0385 | -0.0164 | 0.030 | 0.162 |
| scifact | 2-2 | 409 | -0.0618 | -0.0519 | -0.0377 | -0.0198 | -0.0099 | 0.050 | 0.135 |
| scifact | 2-2 | 512 | -0.0277 | -0.0225 | -0.0161 | -0.0067 | -0.0051 | 0.056 | 0.110 |
| scifact | 2-2 | 614 | -0.0071 | -0.0054 | -0.0039 | 0.0004 | -0.0017 | 0.069 | 0.077 |
| scifact | 2-2 | 716 | 0.0027 | 0.0024 | 0.0013 | 0.0035 | 0.0003 | 0.084 | 0.065 |
| scifact | 2-2 | 819 | 0.0069 | 0.0063 | 0.0036 | 0.0047 | 0.0006 | 0.086 | 0.038 |
| scifact | 2-2 | 921 | 0.0085 | 0.0077 | 0.0046 | 0.0048 | 0.0008 | 0.075 | 0.021 |
| nfcorpus | prefix | 102 | -0.2861 | -0.3904 | -0.3111 | -0.2024 | 0.1043 | -0.434 | 0.325 |
| nfcorpus | prefix | 204 | -0.2567 | -0.3515 | -0.2790 | -0.1808 | 0.0948 | -0.308 | 0.201 |
| nfcorpus | prefix | 307 | -0.2282 | -0.3121 | -0.2475 | -0.1607 | 0.0839 | -0.235 | 0.144 |
| nfcorpus | prefix | 409 | -0.1884 | -0.2578 | -0.2037 | -0.1320 | 0.0694 | -0.172 | 0.092 |
| nfcorpus | prefix | 512 | -0.1514 | -0.2086 | -0.1640 | -0.1053 | 0.0572 | -0.118 | 0.054 |
| nfcorpus | prefix | 614 | -0.1180 | -0.1636 | -0.1280 | -0.0815 | 0.0455 | -0.082 | 0.034 |
| nfcorpus | prefix | 716 | -0.0909 | -0.1254 | -0.0984 | -0.0629 | 0.0344 | -0.060 | 0.027 |
| nfcorpus | prefix | 819 | -0.0579 | -0.0806 | -0.0627 | -0.0393 | 0.0228 | -0.037 | 0.015 |
| nfcorpus | prefix | 921 | -0.0254 | -0.0369 | -0.0278 | -0.0155 | 0.0115 | -0.019 | 0.010 |
| nfcorpus | shared | 102 | -0.1153 | -0.2196 | -0.1625 | -0.0872 | 0.1043 | 0.839 | -1.401 |
| nfcorpus | shared | 204 | -0.0243 | -0.1192 | -0.0786 | -0.0267 | 0.0948 | 0.949 | -1.462 |
| nfcorpus | shared | 307 | 0.0280 | -0.0593 | -0.0293 | 0.0080 | 0.0873 | 0.968 | -1.445 |
| nfcorpus | shared | 409 | 0.0577 | -0.0233 | -0.0001 | 0.0278 | 0.0810 | 0.954 | -1.401 |
| nfcorpus | shared | 512 | 0.0734 | -0.0024 | 0.0164 | 0.0385 | 0.0758 | 0.926 | -1.349 |
| nfcorpus | shared | 614 | 0.0798 | 0.0079 | 0.0242 | 0.0430 | 0.0719 | 0.896 | -1.298 |
| nfcorpus | shared | 716 | 0.0803 | 0.0119 | 0.0267 | 0.0438 | 0.0683 | 0.856 | -1.237 |
| nfcorpus | shared | 819 | 0.0749 | 0.0132 | 0.0264 | 0.0415 | 0.0617 | 0.764 | -1.091 |
| nfcorpus | shared | 921 | 0.0582 | 0.0117 | 0.0214 | 0.0330 | 0.0465 | 0.562 | -0.765 |
| nfcorpus | own | 102 | -0.0598 | n/a | n/a | -0.1145 | n/a | 3.853 | -5.315 |
| nfcorpus | own | 204 | 0.0528 | n/a | n/a | -0.0636 | n/a | 4.017 | -5.288 |
| nfcorpus | own | 307 | 0.1156 | n/a | n/a | -0.0334 | n/a | 3.973 | -5.228 |
| nfcorpus | own | 409 | 0.1494 | n/a | n/a | -0.0154 | n/a | 3.860 | -5.148 |
| nfcorpus | own | 512 | 0.1654 | n/a | n/a | -0.0051 | n/a | 3.725 | -5.049 |
| nfcorpus | own | 614 | 0.1701 | n/a | n/a | -0.0001 | n/a | 3.599 | -4.954 |
| nfcorpus | own | 716 | 0.1670 | n/a | n/a | 0.0028 | n/a | 3.391 | -4.772 |
| nfcorpus | own | 819 | 0.1522 | n/a | n/a | 0.0055 | n/a | 2.941 | -4.284 |
| nfcorpus | own | 921 | 0.1150 | n/a | n/a | 0.0068 | n/a | 2.073 | -3.024 |
| nfcorpus | 1-1 | 102 | -0.1642 | -0.2497 | -0.1919 | -0.1145 | 0.0856 | 0.190 | -0.419 |
| nfcorpus | 1-1 | 204 | -0.0913 | -0.1613 | -0.1190 | -0.0635 | 0.0700 | 0.287 | -0.491 |
| nfcorpus | 1-1 | 307 | -0.0480 | -0.1081 | -0.0755 | -0.0333 | 0.0601 | 0.321 | -0.511 |
| nfcorpus | 1-1 | 409 | -0.0221 | -0.0758 | -0.0493 | -0.0154 | 0.0537 | 0.334 | -0.515 |
| nfcorpus | 1-1 | 512 | -0.0072 | -0.0568 | -0.0340 | -0.0050 | 0.0496 | 0.337 | -0.511 |
| nfcorpus | 1-1 | 614 | -0.0001 | -0.0474 | -0.0265 | -0.0001 | 0.0473 | 0.335 | -0.504 |
| nfcorpus | 1-1 | 716 | 0.0041 | -0.0409 | -0.0215 | 0.0028 | 0.0450 | 0.325 | -0.485 |
| nfcorpus | 1-1 | 819 | 0.0077 | -0.0329 | -0.0159 | 0.0055 | 0.0406 | 0.297 | -0.438 |
| nfcorpus | 1-1 | 921 | 0.0092 | -0.0221 | -0.0093 | 0.0068 | 0.0313 | 0.227 | -0.317 |
| nfcorpus | 2-2 | 102 | -0.1478 | -0.2411 | -0.1819 | -0.1044 | 0.0933 | 0.258 | -0.516 |
| nfcorpus | 2-2 | 204 | -0.0710 | -0.1472 | -0.1043 | -0.0493 | 0.0763 | 0.350 | -0.586 |
| nfcorpus | 2-2 | 307 | -0.0256 | -0.0909 | -0.0580 | -0.0170 | 0.0653 | 0.380 | -0.599 |
| nfcorpus | 2-2 | 409 | 0.0011 | -0.0568 | -0.0304 | 0.0020 | 0.0579 | 0.388 | -0.595 |
| nfcorpus | 2-2 | 512 | 0.0164 | -0.0366 | -0.0143 | 0.0127 | 0.0530 | 0.387 | -0.584 |
| nfcorpus | 2-2 | 614 | 0.0234 | -0.0267 | -0.0064 | 0.0176 | 0.0501 | 0.380 | -0.570 |
| nfcorpus | 2-2 | 716 | 0.0262 | -0.0214 | -0.0026 | 0.0195 | 0.0476 | 0.367 | -0.545 |
| nfcorpus | 2-2 | 819 | 0.0269 | -0.0159 | 0.0007 | 0.0201 | 0.0428 | 0.332 | -0.482 |
| nfcorpus | 2-2 | 921 | 0.0231 | -0.0097 | 0.0026 | 0.0173 | 0.0328 | 0.254 | -0.349 |
| nfcorpus | 5-5 | 102 | -0.1367 | -0.2374 | -0.1745 | -0.0959 | 0.1007 | 0.233 | -0.464 |
| nfcorpus | 5-5 | 204 | -0.0556 | -0.1370 | -0.0912 | -0.0361 | 0.0814 | 0.324 | -0.536 |
| nfcorpus | 5-5 | 307 | -0.0082 | -0.0769 | -0.0419 | -0.0014 | 0.0687 | 0.351 | -0.547 |
| nfcorpus | 5-5 | 409 | 0.0195 | -0.0405 | -0.0126 | 0.0186 | 0.0600 | 0.357 | -0.539 |
| nfcorpus | 5-5 | 512 | 0.0350 | -0.0190 | 0.0044 | 0.0297 | 0.0540 | 0.353 | -0.524 |
| nfcorpus | 5-5 | 614 | 0.0420 | -0.0081 | 0.0126 | 0.0345 | 0.0501 | 0.346 | -0.507 |
| nfcorpus | 5-5 | 716 | 0.0439 | -0.0034 | 0.0156 | 0.0357 | 0.0473 | 0.334 | -0.484 |
| nfcorpus | 5-5 | 819 | 0.0424 | -0.0000 | 0.0166 | 0.0343 | 0.0424 | 0.302 | -0.427 |
| nfcorpus | 5-5 | 921 | 0.0341 | 0.0021 | 0.0142 | 0.0276 | 0.0319 | 0.229 | -0.305 |
| nfcorpus | 10-10 | 102 | -0.1324 | -0.2401 | -0.1728 | -0.0921 | 0.1077 | 0.151 | -0.324 |
| nfcorpus | 10-10 | 204 | -0.0504 | -0.1357 | -0.0860 | -0.0296 | 0.0853 | 0.231 | -0.387 |
| nfcorpus | 10-10 | 307 | -0.0024 | -0.0729 | -0.0346 | 0.0065 | 0.0704 | 0.256 | -0.399 |
| nfcorpus | 10-10 | 409 | 0.0254 | -0.0348 | -0.0042 | 0.0272 | 0.0602 | 0.262 | -0.394 |
| nfcorpus | 10-10 | 512 | 0.0408 | -0.0124 | 0.0133 | 0.0384 | 0.0532 | 0.260 | -0.381 |
| nfcorpus | 10-10 | 614 | 0.0477 | -0.0009 | 0.0217 | 0.0432 | 0.0485 | 0.255 | -0.366 |
| nfcorpus | 10-10 | 716 | 0.0493 | 0.0038 | 0.0245 | 0.0441 | 0.0455 | 0.246 | -0.349 |
| nfcorpus | 10-10 | 819 | 0.0471 | 0.0066 | 0.0246 | 0.0418 | 0.0406 | 0.225 | -0.312 |
| nfcorpus | 10-10 | 921 | 0.0376 | 0.0073 | 0.0202 | 0.0332 | 0.0303 | 0.172 | -0.226 |
| nfcorpus | 20-20 | 102 | -0.1268 | -0.2445 | -0.1706 | -0.0870 | 0.1177 | 0.075 | -0.204 |
| nfcorpus | 20-20 | 204 | -0.0455 | -0.1380 | -0.0818 | -0.0229 | 0.0925 | 0.144 | -0.263 |
| nfcorpus | 20-20 | 307 | 0.0019 | -0.0735 | -0.0292 | 0.0140 | 0.0754 | 0.166 | -0.272 |
| nfcorpus | 20-20 | 409 | 0.0295 | -0.0342 | 0.0020 | 0.0349 | 0.0637 | 0.175 | -0.271 |
| nfcorpus | 20-20 | 512 | 0.0446 | -0.0108 | 0.0198 | 0.0460 | 0.0554 | 0.176 | -0.262 |
| nfcorpus | 20-20 | 614 | 0.0513 | 0.0014 | 0.0284 | 0.0507 | 0.0499 | 0.172 | -0.251 |
| nfcorpus | 20-20 | 716 | 0.0528 | 0.0066 | 0.0311 | 0.0513 | 0.0462 | 0.167 | -0.239 |
| nfcorpus | 20-20 | 819 | 0.0502 | 0.0092 | 0.0305 | 0.0483 | 0.0410 | 0.154 | -0.213 |
| nfcorpus | 20-20 | 921 | 0.0398 | 0.0096 | 0.0248 | 0.0381 | 0.0302 | 0.118 | -0.155 |
| msmarco | prefix | 102 | -0.5714 | -0.5881 | -0.5004 | -0.1017 | 0.0167 | -3.803 | 1.013 |
| msmarco | prefix | 204 | -0.5201 | -0.5359 | -0.4557 | -0.0922 | 0.0158 | -2.907 | 0.503 |
| msmarco | prefix | 307 | -0.4610 | -0.4750 | -0.4032 | -0.0825 | 0.0140 | -2.258 | 0.310 |
| msmarco | prefix | 409 | -0.3840 | -0.3957 | -0.3344 | -0.0680 | 0.0117 | -1.754 | 0.200 |
| msmarco | prefix | 512 | -0.3130 | -0.3226 | -0.2720 | -0.0564 | 0.0096 | -1.332 | 0.130 |
| msmarco | prefix | 614 | -0.2470 | -0.2547 | -0.2142 | -0.0439 | 0.0078 | -0.990 | 0.084 |
| msmarco | prefix | 716 | -0.1835 | -0.1897 | -0.1593 | -0.0319 | 0.0061 | -0.699 | 0.042 |
| msmarco | prefix | 819 | -0.1184 | -0.1227 | -0.1022 | -0.0180 | 0.0043 | -0.464 | 0.021 |
| msmarco | prefix | 921 | -0.0567 | -0.0591 | -0.0485 | -0.0054 | 0.0025 | -0.249 | 0.008 |
| msmarco | shared | 102 | -0.3179 | -0.3631 | -0.3088 | -0.0528 | 0.0451 | -0.186 | -1.641 |
| msmarco | shared | 204 | -0.1669 | -0.2204 | -0.1873 | -0.0257 | 0.0535 | 0.515 | -1.622 |
| msmarco | shared | 307 | -0.0756 | -0.1319 | -0.1120 | -0.0095 | 0.0562 | 0.770 | -1.537 |
| msmarco | shared | 409 | -0.0201 | -0.0761 | -0.0645 | 0.0003 | 0.0559 | 0.862 | -1.436 |
| msmarco | shared | 512 | 0.0130 | -0.0408 | -0.0345 | 0.0059 | 0.0538 | 0.878 | -1.329 |
| msmarco | shared | 614 | 0.0302 | -0.0204 | -0.0171 | 0.0087 | 0.0507 | 0.855 | -1.227 |
| msmarco | shared | 716 | 0.0372 | -0.0101 | -0.0083 | 0.0096 | 0.0473 | 0.811 | -1.127 |
| msmarco | shared | 819 | 0.0377 | -0.0060 | -0.0047 | 0.0095 | 0.0437 | 0.753 | -1.031 |
| msmarco | shared | 921 | 0.0317 | -0.0026 | -0.0018 | 0.0078 | 0.0343 | 0.594 | -0.794 |
| msmarco | own | 102 | -0.3011 | n/a | n/a | -0.0559 | n/a | 0.671 | -2.934 |
| msmarco | own | 204 | -0.1440 | n/a | n/a | -0.0295 | n/a | 1.396 | -2.898 |
| msmarco | own | 307 | -0.0502 | n/a | n/a | -0.0137 | n/a | 1.631 | -2.816 |
| msmarco | own | 409 | 0.0059 | n/a | n/a | -0.0040 | n/a | 1.685 | -2.707 |
| msmarco | own | 512 | 0.0384 | n/a | n/a | 0.0018 | n/a | 1.655 | -2.563 |
| msmarco | own | 614 | 0.0544 | n/a | n/a | 0.0047 | n/a | 1.585 | -2.411 |
| msmarco | own | 716 | 0.0599 | n/a | n/a | 0.0059 | n/a | 1.501 | -2.267 |
| msmarco | own | 819 | 0.0585 | n/a | n/a | 0.0061 | n/a | 1.376 | -2.058 |
| msmarco | own | 921 | 0.0476 | n/a | n/a | 0.0051 | n/a | 1.052 | -1.537 |
| msmarco | 1-1 | 102 | -0.3616 | -0.3740 | -0.3187 | -0.0559 | 0.0123 | -1.257 | 0.197 |
| msmarco | 1-1 | 204 | -0.2263 | -0.2354 | -0.2009 | -0.0295 | 0.0092 | -0.578 | 0.096 |
| msmarco | 1-1 | 307 | -0.1421 | -0.1493 | -0.1277 | -0.0137 | 0.0072 | -0.281 | 0.052 |
| msmarco | 1-1 | 409 | -0.0884 | -0.0946 | -0.0811 | -0.0040 | 0.0062 | -0.123 | 0.022 |
| msmarco | 1-1 | 512 | -0.0541 | -0.0597 | -0.0513 | 0.0018 | 0.0056 | -0.033 | 0.003 |
| msmarco | 1-1 | 614 | -0.0340 | -0.0393 | -0.0339 | 0.0047 | 0.0053 | 0.017 | -0.012 |
| msmarco | 1-1 | 716 | -0.0237 | -0.0289 | -0.0250 | 0.0059 | 0.0052 | 0.040 | -0.022 |
| msmarco | 1-1 | 819 | -0.0180 | -0.0231 | -0.0199 | 0.0061 | 0.0051 | 0.054 | -0.030 |
| msmarco | 1-1 | 921 | -0.0104 | -0.0152 | -0.0129 | 0.0051 | 0.0048 | 0.067 | -0.047 |
| msmarco | 2-2 | 102 | -0.3348 | -0.3747 | -0.3224 | -0.0545 | 0.0398 | -1.090 | -0.061 |
| msmarco | 2-2 | 204 | -0.2010 | -0.2312 | -0.1978 | -0.0263 | 0.0302 | -0.482 | -0.155 |
| msmarco | 2-2 | 307 | -0.1184 | -0.1424 | -0.1204 | -0.0093 | 0.0240 | -0.222 | -0.178 |
| msmarco | 2-2 | 409 | -0.0668 | -0.0862 | -0.0717 | 0.0007 | 0.0194 | -0.088 | -0.176 |
| msmarco | 2-2 | 512 | -0.0347 | -0.0507 | -0.0407 | 0.0064 | 0.0160 | -0.011 | -0.163 |
| msmarco | 2-2 | 614 | -0.0157 | -0.0297 | -0.0226 | 0.0093 | 0.0140 | 0.038 | -0.160 |
| msmarco | 2-2 | 716 | -0.0066 | -0.0193 | -0.0137 | 0.0101 | 0.0127 | 0.061 | -0.155 |
| msmarco | 2-2 | 819 | -0.0027 | -0.0144 | -0.0096 | 0.0100 | 0.0116 | 0.069 | -0.148 |
| msmarco | 2-2 | 921 | 0.0005 | -0.0088 | -0.0053 | 0.0081 | 0.0092 | 0.071 | -0.119 |

## Construction size on common cohorts

Cells are compared on the same queries: those with enough positives for the largest cell shown. Partitions are drawn separately per cell; this describes high-count queries only.

| environment | queries with at least | queries | cell | k | measure | cell value | 1-1 value, same queries | difference from 1-1 | 95% interval (pointwise) |
|---|---|---|---|---|---|---|---|---|---|
| scifact | 4 | 6 | 1-1 | 102 | held-out reciprocal rank | 0.2658 | 0.2658 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 102 | held-out log rank | 1.4671 | 1.4671 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 102 | nDCG@10, full judgments | 0.8839 | 0.8839 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 204 | held-out reciprocal rank | 0.2502 | 0.2502 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 204 | held-out log rank | 1.4806 | 1.4806 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 204 | nDCG@10, full judgments | 0.9017 | 0.9017 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 307 | held-out reciprocal rank | 0.2777 | 0.2777 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 307 | held-out log rank | 1.3854 | 1.3854 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 307 | nDCG@10, full judgments | 0.8994 | 0.8994 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 409 | held-out reciprocal rank | 0.2772 | 0.2772 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 409 | held-out log rank | 1.3820 | 1.3820 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 409 | nDCG@10, full judgments | 0.8995 | 0.8995 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 512 | held-out reciprocal rank | 0.2903 | 0.2903 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 512 | held-out log rank | 1.3504 | 1.3504 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 512 | nDCG@10, full judgments | 0.8928 | 0.8928 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 614 | held-out reciprocal rank | 0.2819 | 0.2819 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 614 | held-out log rank | 1.3842 | 1.3842 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 614 | nDCG@10, full judgments | 0.8747 | 0.8747 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 716 | held-out reciprocal rank | 0.2787 | 0.2787 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 716 | held-out log rank | 1.3971 | 1.3971 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 716 | nDCG@10, full judgments | 0.8782 | 0.8782 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 819 | held-out reciprocal rank | 0.3025 | 0.3025 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 819 | held-out log rank | 1.3508 | 1.3508 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 819 | nDCG@10, full judgments | 0.8754 | 0.8754 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 921 | held-out reciprocal rank | 0.3025 | 0.3025 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 921 | held-out log rank | 1.3590 | 1.3590 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 1-1 | 921 | nDCG@10, full judgments | 0.8412 | 0.8412 | 0.0000 | 0.0000 to 0.0000 |
| scifact | 4 | 6 | 2-2 | 102 | held-out reciprocal rank | 0.2633 | 0.2658 | -0.0025 | -0.0539 to 0.0658 |
| scifact | 4 | 6 | 2-2 | 102 | held-out log rank | 1.4698 | 1.4671 | 0.0027 | -0.1970 to 0.1309 |
| scifact | 4 | 6 | 2-2 | 102 | nDCG@10, full judgments | 0.8845 | 0.8839 | 0.0006 | -0.0530 to 0.0588 |
| scifact | 4 | 6 | 2-2 | 204 | held-out reciprocal rank | 0.2693 | 0.2502 | 0.0191 | -0.0194 to 0.0579 |
| scifact | 4 | 6 | 2-2 | 204 | held-out log rank | 1.4565 | 1.4806 | -0.0242 | -0.1956 to 0.1335 |
| scifact | 4 | 6 | 2-2 | 204 | nDCG@10, full judgments | 0.8712 | 0.9017 | -0.0305 | -0.1020 to 0.0309 |
| scifact | 4 | 6 | 2-2 | 307 | held-out reciprocal rank | 0.2836 | 0.2777 | 0.0059 | -0.0331 to 0.0484 |
| scifact | 4 | 6 | 2-2 | 307 | held-out log rank | 1.4269 | 1.3854 | 0.0415 | -0.0700 to 0.1702 |
| scifact | 4 | 6 | 2-2 | 307 | nDCG@10, full judgments | 0.8753 | 0.8994 | -0.0240 | -0.0711 to 0.0145 |
| scifact | 4 | 6 | 2-2 | 409 | held-out reciprocal rank | 0.2931 | 0.2772 | 0.0158 | -0.0340 to 0.0703 |
| scifact | 4 | 6 | 2-2 | 409 | held-out log rank | 1.4002 | 1.3820 | 0.0182 | -0.0967 to 0.1471 |
| scifact | 4 | 6 | 2-2 | 409 | nDCG@10, full judgments | 0.8755 | 0.8995 | -0.0240 | -0.0676 to 0.0090 |
| scifact | 4 | 6 | 2-2 | 512 | held-out reciprocal rank | 0.3057 | 0.2903 | 0.0154 | -0.0424 to 0.0891 |
| scifact | 4 | 6 | 2-2 | 512 | held-out log rank | 1.3750 | 1.3504 | 0.0246 | -0.1167 to 0.1719 |
| scifact | 4 | 6 | 2-2 | 512 | nDCG@10, full judgments | 0.8690 | 0.8928 | -0.0238 | -0.0664 to 0.0016 |
| scifact | 4 | 6 | 2-2 | 614 | held-out reciprocal rank | 0.3218 | 0.2819 | 0.0399 | -0.0198 to 0.1046 |
| scifact | 4 | 6 | 2-2 | 614 | held-out log rank | 1.3427 | 1.3842 | -0.0415 | -0.1921 to 0.1183 |
| scifact | 4 | 6 | 2-2 | 614 | nDCG@10, full judgments | 0.8651 | 0.8747 | -0.0096 | -0.0624 to 0.0233 |
| scifact | 4 | 6 | 2-2 | 716 | held-out reciprocal rank | 0.3256 | 0.2787 | 0.0469 | -0.0135 to 0.1125 |
| scifact | 4 | 6 | 2-2 | 716 | held-out log rank | 1.3298 | 1.3971 | -0.0672 | -0.2172 to 0.1021 |
| scifact | 4 | 6 | 2-2 | 716 | nDCG@10, full judgments | 0.8648 | 0.8782 | -0.0134 | -0.0636 to 0.0171 |
| scifact | 4 | 6 | 2-2 | 819 | held-out reciprocal rank | 0.3338 | 0.3025 | 0.0313 | -0.0753 to 0.1197 |
| scifact | 4 | 6 | 2-2 | 819 | held-out log rank | 1.3030 | 1.3508 | -0.0478 | -0.2187 to 0.1893 |
| scifact | 4 | 6 | 2-2 | 819 | nDCG@10, full judgments | 0.8652 | 0.8754 | -0.0102 | -0.0604 to 0.0190 |
| scifact | 4 | 6 | 2-2 | 921 | held-out reciprocal rank | 0.3404 | 0.3025 | 0.0378 | -0.0787 to 0.1264 |
| scifact | 4 | 6 | 2-2 | 921 | held-out log rank | 1.2862 | 1.3590 | -0.0728 | -0.2656 to 0.2211 |
| scifact | 4 | 6 | 2-2 | 921 | nDCG@10, full judgments | 0.8534 | 0.8412 | 0.0121 | -0.0020 to 0.0292 |
| nfcorpus | 4 | 1922 | 1-1 | 102 | held-out reciprocal rank | 0.0534 | 0.0534 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 102 | held-out log rank | 5.1643 | 5.1643 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 102 | nDCG@10, full judgments | 0.5259 | 0.5259 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 204 | held-out reciprocal rank | 0.0555 | 0.0555 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 204 | held-out log rank | 5.0958 | 5.0958 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 204 | nDCG@10, full judgments | 0.5400 | 0.5400 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 307 | held-out reciprocal rank | 0.0578 | 0.0578 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 307 | held-out log rank | 5.0755 | 5.0755 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 307 | nDCG@10, full judgments | 0.5426 | 0.5426 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 409 | held-out reciprocal rank | 0.0598 | 0.0598 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 409 | held-out log rank | 5.0696 | 5.0696 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 409 | nDCG@10, full judgments | 0.5412 | 0.5412 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 512 | held-out reciprocal rank | 0.0612 | 0.0612 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 512 | held-out log rank | 5.0723 | 5.0723 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 512 | nDCG@10, full judgments | 0.5360 | 0.5360 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 614 | held-out reciprocal rank | 0.0623 | 0.0623 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 614 | held-out log rank | 5.0771 | 5.0771 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 614 | nDCG@10, full judgments | 0.5311 | 0.5311 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 716 | held-out reciprocal rank | 0.0631 | 0.0631 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 716 | held-out log rank | 5.0936 | 5.0936 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 716 | nDCG@10, full judgments | 0.5199 | 0.5199 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 819 | held-out reciprocal rank | 0.0640 | 0.0640 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 819 | held-out log rank | 5.1371 | 5.1371 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 819 | nDCG@10, full judgments | 0.4890 | 0.4890 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 921 | held-out reciprocal rank | 0.0639 | 0.0639 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 921 | held-out log rank | 5.2453 | 5.2453 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 1-1 | 921 | nDCG@10, full judgments | 0.4257 | 0.4257 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 4 | 1922 | 2-2 | 102 | held-out reciprocal rank | 0.0588 | 0.0534 | 0.0054 | 0.0030 to 0.0078 |
| nfcorpus | 4 | 1922 | 2-2 | 102 | held-out log rank | 5.0398 | 5.1643 | -0.1246 | -0.1593 to -0.0908 |
| nfcorpus | 4 | 1922 | 2-2 | 102 | nDCG@10, full judgments | 0.5487 | 0.5259 | 0.0228 | 0.0189 to 0.0263 |
| nfcorpus | 4 | 1922 | 2-2 | 204 | held-out reciprocal rank | 0.0620 | 0.0555 | 0.0065 | 0.0042 to 0.0089 |
| nfcorpus | 4 | 1922 | 2-2 | 204 | held-out log rank | 4.9699 | 5.0958 | -0.1259 | -0.1607 to -0.0918 |
| nfcorpus | 4 | 1922 | 2-2 | 204 | nDCG@10, full judgments | 0.5629 | 0.5400 | 0.0228 | 0.0192 to 0.0263 |
| nfcorpus | 4 | 1922 | 2-2 | 307 | held-out reciprocal rank | 0.0637 | 0.0578 | 0.0059 | 0.0034 to 0.0084 |
| nfcorpus | 4 | 1922 | 2-2 | 307 | held-out log rank | 4.9562 | 5.0755 | -0.1192 | -0.1541 to -0.0855 |
| nfcorpus | 4 | 1922 | 2-2 | 307 | nDCG@10, full judgments | 0.5575 | 0.5426 | 0.0150 | 0.0114 to 0.0183 |
| nfcorpus | 4 | 1922 | 2-2 | 409 | held-out reciprocal rank | 0.0646 | 0.0598 | 0.0048 | 0.0022 to 0.0075 |
| nfcorpus | 4 | 1922 | 2-2 | 409 | held-out log rank | 4.9601 | 5.0696 | -0.1096 | -0.1445 to -0.0748 |
| nfcorpus | 4 | 1922 | 2-2 | 409 | nDCG@10, full judgments | 0.5480 | 0.5412 | 0.0068 | 0.0032 to 0.0102 |
| nfcorpus | 4 | 1922 | 2-2 | 512 | held-out reciprocal rank | 0.0654 | 0.0612 | 0.0042 | 0.0015 to 0.0069 |
| nfcorpus | 4 | 1922 | 2-2 | 512 | held-out log rank | 4.9713 | 5.0723 | -0.1010 | -0.1358 to -0.0662 |
| nfcorpus | 4 | 1922 | 2-2 | 512 | nDCG@10, full judgments | 0.5352 | 0.5360 | -0.0008 | -0.0044 to 0.0025 |
| nfcorpus | 4 | 1922 | 2-2 | 614 | held-out reciprocal rank | 0.0658 | 0.0623 | 0.0035 | 0.0007 to 0.0063 |
| nfcorpus | 4 | 1922 | 2-2 | 614 | held-out log rank | 4.9858 | 5.0771 | -0.0914 | -0.1265 to -0.0567 |
| nfcorpus | 4 | 1922 | 2-2 | 614 | nDCG@10, full judgments | 0.5240 | 0.5311 | -0.0071 | -0.0106 to -0.0038 |
| nfcorpus | 4 | 1922 | 2-2 | 716 | held-out reciprocal rank | 0.0658 | 0.0631 | 0.0027 | -0.0001 to 0.0055 |
| nfcorpus | 4 | 1922 | 2-2 | 716 | held-out log rank | 5.0106 | 5.0936 | -0.0830 | -0.1192 to -0.0483 |
| nfcorpus | 4 | 1922 | 2-2 | 716 | nDCG@10, full judgments | 0.5079 | 0.5199 | -0.0120 | -0.0152 to -0.0088 |
| nfcorpus | 4 | 1922 | 2-2 | 819 | held-out reciprocal rank | 0.0647 | 0.0640 | 0.0007 | -0.0022 to 0.0037 |
| nfcorpus | 4 | 1922 | 2-2 | 819 | held-out log rank | 5.0733 | 5.1371 | -0.0638 | -0.0990 to -0.0296 |
| nfcorpus | 4 | 1922 | 2-2 | 819 | nDCG@10, full judgments | 0.4738 | 0.4890 | -0.0152 | -0.0180 to -0.0122 |
| nfcorpus | 4 | 1922 | 2-2 | 921 | held-out reciprocal rank | 0.0628 | 0.0639 | -0.0011 | -0.0042 to 0.0019 |
| nfcorpus | 4 | 1922 | 2-2 | 921 | held-out log rank | 5.2067 | 5.2453 | -0.0386 | -0.0746 to -0.0042 |
| nfcorpus | 4 | 1922 | 2-2 | 921 | nDCG@10, full judgments | 0.4215 | 0.4257 | -0.0042 | -0.0062 to -0.0021 |
| nfcorpus | 10 | 1517 | 1-1 | 102 | held-out reciprocal rank | 0.0400 | 0.0400 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 102 | held-out log rank | 5.4362 | 5.4362 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 102 | nDCG@10, full judgments | 0.5351 | 0.5351 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 204 | held-out reciprocal rank | 0.0414 | 0.0414 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 204 | held-out log rank | 5.3708 | 5.3708 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 204 | nDCG@10, full judgments | 0.5507 | 0.5507 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 307 | held-out reciprocal rank | 0.0433 | 0.0433 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 307 | held-out log rank | 5.3473 | 5.3473 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 307 | nDCG@10, full judgments | 0.5539 | 0.5539 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 409 | held-out reciprocal rank | 0.0449 | 0.0449 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 409 | held-out log rank | 5.3394 | 5.3394 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 409 | nDCG@10, full judgments | 0.5532 | 0.5532 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 512 | held-out reciprocal rank | 0.0457 | 0.0457 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 512 | held-out log rank | 5.3388 | 5.3388 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 512 | nDCG@10, full judgments | 0.5490 | 0.5490 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 614 | held-out reciprocal rank | 0.0463 | 0.0463 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 614 | held-out log rank | 5.3409 | 5.3409 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 614 | nDCG@10, full judgments | 0.5444 | 0.5444 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 716 | held-out reciprocal rank | 0.0472 | 0.0472 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 716 | held-out log rank | 5.3502 | 5.3502 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 716 | nDCG@10, full judgments | 0.5336 | 0.5336 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 819 | held-out reciprocal rank | 0.0484 | 0.0484 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 819 | held-out log rank | 5.3790 | 5.3790 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 819 | nDCG@10, full judgments | 0.5050 | 0.5050 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 921 | held-out reciprocal rank | 0.0485 | 0.0485 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 921 | held-out log rank | 5.4587 | 5.4587 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 1-1 | 921 | nDCG@10, full judgments | 0.4504 | 0.4504 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 10 | 1517 | 2-2 | 102 | held-out reciprocal rank | 0.0437 | 0.0400 | 0.0037 | 0.0016 to 0.0061 |
| nfcorpus | 10 | 1517 | 2-2 | 102 | held-out log rank | 5.3143 | 5.4362 | -0.1219 | -0.1595 to -0.0841 |
| nfcorpus | 10 | 1517 | 2-2 | 102 | nDCG@10, full judgments | 0.5471 | 0.5351 | 0.0120 | 0.0078 to 0.0157 |
| nfcorpus | 10 | 1517 | 2-2 | 204 | held-out reciprocal rank | 0.0458 | 0.0414 | 0.0044 | 0.0021 to 0.0067 |
| nfcorpus | 10 | 1517 | 2-2 | 204 | held-out log rank | 5.2460 | 5.3708 | -0.1248 | -0.1642 to -0.0876 |
| nfcorpus | 10 | 1517 | 2-2 | 204 | nDCG@10, full judgments | 0.5627 | 0.5507 | 0.0120 | 0.0082 to 0.0158 |
| nfcorpus | 10 | 1517 | 2-2 | 307 | held-out reciprocal rank | 0.0473 | 0.0433 | 0.0039 | 0.0015 to 0.0064 |
| nfcorpus | 10 | 1517 | 2-2 | 307 | held-out log rank | 5.2292 | 5.3473 | -0.1181 | -0.1566 to -0.0802 |
| nfcorpus | 10 | 1517 | 2-2 | 307 | nDCG@10, full judgments | 0.5588 | 0.5539 | 0.0049 | 0.0012 to 0.0088 |
| nfcorpus | 10 | 1517 | 2-2 | 409 | held-out reciprocal rank | 0.0480 | 0.0449 | 0.0031 | 0.0006 to 0.0057 |
| nfcorpus | 10 | 1517 | 2-2 | 409 | held-out log rank | 5.2292 | 5.3394 | -0.1102 | -0.1495 to -0.0719 |
| nfcorpus | 10 | 1517 | 2-2 | 409 | nDCG@10, full judgments | 0.5504 | 0.5532 | -0.0028 | -0.0067 to 0.0009 |
| nfcorpus | 10 | 1517 | 2-2 | 512 | held-out reciprocal rank | 0.0487 | 0.0457 | 0.0031 | 0.0004 to 0.0057 |
| nfcorpus | 10 | 1517 | 2-2 | 512 | held-out log rank | 5.2352 | 5.3388 | -0.1036 | -0.1428 to -0.0659 |
| nfcorpus | 10 | 1517 | 2-2 | 512 | nDCG@10, full judgments | 0.5394 | 0.5490 | -0.0096 | -0.0134 to -0.0060 |
| nfcorpus | 10 | 1517 | 2-2 | 614 | held-out reciprocal rank | 0.0490 | 0.0463 | 0.0027 | 0.0000 to 0.0053 |
| nfcorpus | 10 | 1517 | 2-2 | 614 | held-out log rank | 5.2459 | 5.3409 | -0.0950 | -0.1336 to -0.0570 |
| nfcorpus | 10 | 1517 | 2-2 | 614 | nDCG@10, full judgments | 0.5294 | 0.5444 | -0.0150 | -0.0186 to -0.0113 |
| nfcorpus | 10 | 1517 | 2-2 | 716 | held-out reciprocal rank | 0.0491 | 0.0472 | 0.0019 | -0.0008 to 0.0047 |
| nfcorpus | 10 | 1517 | 2-2 | 716 | held-out log rank | 5.2634 | 5.3502 | -0.0868 | -0.1260 to -0.0495 |
| nfcorpus | 10 | 1517 | 2-2 | 716 | nDCG@10, full judgments | 0.5157 | 0.5336 | -0.0179 | -0.0213 to -0.0145 |
| nfcorpus | 10 | 1517 | 2-2 | 819 | held-out reciprocal rank | 0.0482 | 0.0484 | -0.0003 | -0.0031 to 0.0026 |
| nfcorpus | 10 | 1517 | 2-2 | 819 | held-out log rank | 5.3100 | 5.3790 | -0.0690 | -0.1079 to -0.0311 |
| nfcorpus | 10 | 1517 | 2-2 | 819 | nDCG@10, full judgments | 0.4882 | 0.5050 | -0.0168 | -0.0200 to -0.0138 |
| nfcorpus | 10 | 1517 | 2-2 | 921 | held-out reciprocal rank | 0.0475 | 0.0485 | -0.0010 | -0.0039 to 0.0021 |
| nfcorpus | 10 | 1517 | 2-2 | 921 | held-out log rank | 5.4088 | 5.4587 | -0.0499 | -0.0877 to -0.0115 |
| nfcorpus | 10 | 1517 | 2-2 | 921 | nDCG@10, full judgments | 0.4466 | 0.4504 | -0.0037 | -0.0058 to -0.0016 |
| nfcorpus | 10 | 1517 | 5-5 | 102 | held-out reciprocal rank | 0.0505 | 0.0400 | 0.0106 | 0.0085 to 0.0127 |
| nfcorpus | 10 | 1517 | 5-5 | 102 | held-out log rank | 5.2090 | 5.4362 | -0.2272 | -0.2636 to -0.1920 |
| nfcorpus | 10 | 1517 | 5-5 | 102 | nDCG@10, full judgments | 0.5171 | 0.5351 | -0.0181 | -0.0235 to -0.0127 |
| nfcorpus | 10 | 1517 | 5-5 | 204 | held-out reciprocal rank | 0.0529 | 0.0414 | 0.0114 | 0.0093 to 0.0136 |
| nfcorpus | 10 | 1517 | 5-5 | 204 | held-out log rank | 5.1363 | 5.3708 | -0.2345 | -0.2700 to -0.1990 |
| nfcorpus | 10 | 1517 | 5-5 | 204 | nDCG@10, full judgments | 0.5349 | 0.5507 | -0.0159 | -0.0211 to -0.0105 |
| nfcorpus | 10 | 1517 | 5-5 | 307 | held-out reciprocal rank | 0.0538 | 0.0433 | 0.0105 | 0.0082 to 0.0128 |
| nfcorpus | 10 | 1517 | 5-5 | 307 | held-out log rank | 5.1261 | 5.3473 | -0.2212 | -0.2573 to -0.1852 |
| nfcorpus | 10 | 1517 | 5-5 | 307 | nDCG@10, full judgments | 0.5322 | 0.5539 | -0.0217 | -0.0268 to -0.0167 |
| nfcorpus | 10 | 1517 | 5-5 | 409 | held-out reciprocal rank | 0.0539 | 0.0449 | 0.0090 | 0.0066 to 0.0114 |
| nfcorpus | 10 | 1517 | 5-5 | 409 | held-out log rank | 5.1340 | 5.3394 | -0.2054 | -0.2422 to -0.1690 |
| nfcorpus | 10 | 1517 | 5-5 | 409 | nDCG@10, full judgments | 0.5258 | 0.5532 | -0.0274 | -0.0323 to -0.0226 |
| nfcorpus | 10 | 1517 | 5-5 | 512 | held-out reciprocal rank | 0.0540 | 0.0457 | 0.0083 | 0.0059 to 0.0107 |
| nfcorpus | 10 | 1517 | 5-5 | 512 | held-out log rank | 5.1488 | 5.3388 | -0.1899 | -0.2266 to -0.1543 |
| nfcorpus | 10 | 1517 | 5-5 | 512 | nDCG@10, full judgments | 0.5178 | 0.5490 | -0.0312 | -0.0361 to -0.0267 |
| nfcorpus | 10 | 1517 | 5-5 | 614 | held-out reciprocal rank | 0.0540 | 0.0463 | 0.0076 | 0.0051 to 0.0101 |
| nfcorpus | 10 | 1517 | 5-5 | 614 | held-out log rank | 5.1656 | 5.3409 | -0.1753 | -0.2117 to -0.1389 |
| nfcorpus | 10 | 1517 | 5-5 | 614 | nDCG@10, full judgments | 0.5096 | 0.5444 | -0.0348 | -0.0394 to -0.0305 |
| nfcorpus | 10 | 1517 | 5-5 | 716 | held-out reciprocal rank | 0.0535 | 0.0472 | 0.0063 | 0.0037 to 0.0089 |
| nfcorpus | 10 | 1517 | 5-5 | 716 | held-out log rank | 5.1889 | 5.3502 | -0.1612 | -0.1977 to -0.1251 |
| nfcorpus | 10 | 1517 | 5-5 | 716 | nDCG@10, full judgments | 0.5003 | 0.5336 | -0.0334 | -0.0377 to -0.0292 |
| nfcorpus | 10 | 1517 | 5-5 | 819 | held-out reciprocal rank | 0.0525 | 0.0484 | 0.0041 | 0.0014 to 0.0068 |
| nfcorpus | 10 | 1517 | 5-5 | 819 | held-out log rank | 5.2454 | 5.3790 | -0.1336 | -0.1690 to -0.0985 |
| nfcorpus | 10 | 1517 | 5-5 | 819 | nDCG@10, full judgments | 0.4805 | 0.5050 | -0.0246 | -0.0281 to -0.0211 |
| nfcorpus | 10 | 1517 | 5-5 | 921 | held-out reciprocal rank | 0.0503 | 0.0485 | 0.0019 | -0.0009 to 0.0048 |
| nfcorpus | 10 | 1517 | 5-5 | 921 | held-out log rank | 5.3678 | 5.4587 | -0.0909 | -0.1262 to -0.0559 |
| nfcorpus | 10 | 1517 | 5-5 | 921 | nDCG@10, full judgments | 0.4478 | 0.4504 | -0.0025 | -0.0048 to -0.0002 |
| nfcorpus | 20 | 1105 | 1-1 | 102 | held-out reciprocal rank | 0.0303 | 0.0303 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 102 | held-out log rank | 5.6891 | 5.6891 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 102 | nDCG@10, full judgments | 0.5633 | 0.5633 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 204 | held-out reciprocal rank | 0.0315 | 0.0315 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 204 | held-out log rank | 5.6262 | 5.6262 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 204 | nDCG@10, full judgments | 0.5808 | 0.5808 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 307 | held-out reciprocal rank | 0.0334 | 0.0334 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 307 | held-out log rank | 5.6034 | 5.6034 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 307 | nDCG@10, full judgments | 0.5854 | 0.5854 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 409 | held-out reciprocal rank | 0.0343 | 0.0343 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 409 | held-out log rank | 5.5955 | 5.5955 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 409 | nDCG@10, full judgments | 0.5853 | 0.5853 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 512 | held-out reciprocal rank | 0.0350 | 0.0350 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 512 | held-out log rank | 5.5914 | 5.5914 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 512 | nDCG@10, full judgments | 0.5817 | 0.5817 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 614 | held-out reciprocal rank | 0.0358 | 0.0358 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 614 | held-out log rank | 5.5889 | 5.5889 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 614 | nDCG@10, full judgments | 0.5776 | 0.5776 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 716 | held-out reciprocal rank | 0.0365 | 0.0365 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 716 | held-out log rank | 5.5902 | 5.5902 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 716 | nDCG@10, full judgments | 0.5678 | 0.5678 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 819 | held-out reciprocal rank | 0.0379 | 0.0379 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 819 | held-out log rank | 5.6042 | 5.6042 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 819 | nDCG@10, full judgments | 0.5425 | 0.5425 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 921 | held-out reciprocal rank | 0.0386 | 0.0386 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 921 | held-out log rank | 5.6512 | 5.6512 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 1-1 | 921 | nDCG@10, full judgments | 0.4965 | 0.4965 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 20 | 1105 | 2-2 | 102 | held-out reciprocal rank | 0.0334 | 0.0303 | 0.0030 | 0.0007 to 0.0053 |
| nfcorpus | 20 | 1105 | 2-2 | 102 | held-out log rank | 5.5790 | 5.6891 | -0.1100 | -0.1546 to -0.0648 |
| nfcorpus | 20 | 1105 | 2-2 | 102 | nDCG@10, full judgments | 0.5689 | 0.5633 | 0.0056 | 0.0007 to 0.0101 |
| nfcorpus | 20 | 1105 | 2-2 | 204 | held-out reciprocal rank | 0.0350 | 0.0315 | 0.0035 | 0.0011 to 0.0059 |
| nfcorpus | 20 | 1105 | 2-2 | 204 | held-out log rank | 5.5162 | 5.6262 | -0.1100 | -0.1570 to -0.0642 |
| nfcorpus | 20 | 1105 | 2-2 | 204 | nDCG@10, full judgments | 0.5862 | 0.5808 | 0.0054 | 0.0007 to 0.0098 |
| nfcorpus | 20 | 1105 | 2-2 | 307 | held-out reciprocal rank | 0.0359 | 0.0334 | 0.0025 | -0.0000 to 0.0050 |
| nfcorpus | 20 | 1105 | 2-2 | 307 | held-out log rank | 5.4982 | 5.6034 | -0.1052 | -0.1517 to -0.0591 |
| nfcorpus | 20 | 1105 | 2-2 | 307 | nDCG@10, full judgments | 0.5831 | 0.5854 | -0.0022 | -0.0067 to 0.0021 |
| nfcorpus | 20 | 1105 | 2-2 | 409 | held-out reciprocal rank | 0.0364 | 0.0343 | 0.0021 | -0.0005 to 0.0047 |
| nfcorpus | 20 | 1105 | 2-2 | 409 | held-out log rank | 5.4950 | 5.5955 | -0.1004 | -0.1472 to -0.0537 |
| nfcorpus | 20 | 1105 | 2-2 | 409 | nDCG@10, full judgments | 0.5762 | 0.5853 | -0.0092 | -0.0137 to -0.0050 |
| nfcorpus | 20 | 1105 | 2-2 | 512 | held-out reciprocal rank | 0.0372 | 0.0350 | 0.0022 | -0.0006 to 0.0049 |
| nfcorpus | 20 | 1105 | 2-2 | 512 | held-out log rank | 5.4965 | 5.5914 | -0.0949 | -0.1415 to -0.0490 |
| nfcorpus | 20 | 1105 | 2-2 | 512 | nDCG@10, full judgments | 0.5669 | 0.5817 | -0.0148 | -0.0192 to -0.0108 |
| nfcorpus | 20 | 1105 | 2-2 | 614 | held-out reciprocal rank | 0.0373 | 0.0358 | 0.0016 | -0.0012 to 0.0043 |
| nfcorpus | 20 | 1105 | 2-2 | 614 | held-out log rank | 5.5044 | 5.5889 | -0.0845 | -0.1320 to -0.0389 |
| nfcorpus | 20 | 1105 | 2-2 | 614 | nDCG@10, full judgments | 0.5581 | 0.5776 | -0.0195 | -0.0237 to -0.0155 |
| nfcorpus | 20 | 1105 | 2-2 | 716 | held-out reciprocal rank | 0.0378 | 0.0365 | 0.0013 | -0.0015 to 0.0041 |
| nfcorpus | 20 | 1105 | 2-2 | 716 | held-out log rank | 5.5145 | 5.5902 | -0.0757 | -0.1234 to -0.0301 |
| nfcorpus | 20 | 1105 | 2-2 | 716 | nDCG@10, full judgments | 0.5467 | 0.5678 | -0.0211 | -0.0251 to -0.0174 |
| nfcorpus | 20 | 1105 | 2-2 | 819 | held-out reciprocal rank | 0.0374 | 0.0379 | -0.0005 | -0.0035 to 0.0024 |
| nfcorpus | 20 | 1105 | 2-2 | 819 | held-out log rank | 5.5428 | 5.6042 | -0.0614 | -0.1085 to -0.0151 |
| nfcorpus | 20 | 1105 | 2-2 | 819 | nDCG@10, full judgments | 0.5249 | 0.5425 | -0.0176 | -0.0211 to -0.0142 |
| nfcorpus | 20 | 1105 | 2-2 | 921 | held-out reciprocal rank | 0.0377 | 0.0386 | -0.0009 | -0.0040 to 0.0023 |
| nfcorpus | 20 | 1105 | 2-2 | 921 | held-out log rank | 5.6045 | 5.6512 | -0.0468 | -0.0950 to -0.0014 |
| nfcorpus | 20 | 1105 | 2-2 | 921 | nDCG@10, full judgments | 0.4936 | 0.4965 | -0.0029 | -0.0053 to -0.0004 |
| nfcorpus | 20 | 1105 | 5-5 | 102 | held-out reciprocal rank | 0.0376 | 0.0303 | 0.0072 | 0.0050 to 0.0093 |
| nfcorpus | 20 | 1105 | 5-5 | 102 | held-out log rank | 5.5060 | 5.6891 | -0.1830 | -0.2246 to -0.1412 |
| nfcorpus | 20 | 1105 | 5-5 | 102 | nDCG@10, full judgments | 0.5347 | 0.5633 | -0.0286 | -0.0349 to -0.0225 |
| nfcorpus | 20 | 1105 | 5-5 | 204 | held-out reciprocal rank | 0.0396 | 0.0315 | 0.0082 | 0.0059 to 0.0104 |
| nfcorpus | 20 | 1105 | 5-5 | 204 | held-out log rank | 5.4355 | 5.6262 | -0.1906 | -0.2319 to -0.1503 |
| nfcorpus | 20 | 1105 | 5-5 | 204 | nDCG@10, full judgments | 0.5546 | 0.5808 | -0.0261 | -0.0321 to -0.0202 |
| nfcorpus | 20 | 1105 | 5-5 | 307 | held-out reciprocal rank | 0.0404 | 0.0334 | 0.0070 | 0.0046 to 0.0093 |
| nfcorpus | 20 | 1105 | 5-5 | 307 | held-out log rank | 5.4216 | 5.6034 | -0.1818 | -0.2233 to -0.1417 |
| nfcorpus | 20 | 1105 | 5-5 | 307 | nDCG@10, full judgments | 0.5535 | 0.5854 | -0.0319 | -0.0377 to -0.0263 |
| nfcorpus | 20 | 1105 | 5-5 | 409 | held-out reciprocal rank | 0.0405 | 0.0343 | 0.0062 | 0.0037 to 0.0086 |
| nfcorpus | 20 | 1105 | 5-5 | 409 | held-out log rank | 5.4247 | 5.5955 | -0.1708 | -0.2117 to -0.1301 |
| nfcorpus | 20 | 1105 | 5-5 | 409 | nDCG@10, full judgments | 0.5493 | 0.5853 | -0.0361 | -0.0416 to -0.0306 |
| nfcorpus | 20 | 1105 | 5-5 | 512 | held-out reciprocal rank | 0.0408 | 0.0350 | 0.0058 | 0.0032 to 0.0083 |
| nfcorpus | 20 | 1105 | 5-5 | 512 | held-out log rank | 5.4326 | 5.5914 | -0.1588 | -0.1990 to -0.1193 |
| nfcorpus | 20 | 1105 | 5-5 | 512 | nDCG@10, full judgments | 0.5432 | 0.5817 | -0.0385 | -0.0439 to -0.0332 |
| nfcorpus | 20 | 1105 | 5-5 | 614 | held-out reciprocal rank | 0.0409 | 0.0358 | 0.0052 | 0.0026 to 0.0077 |
| nfcorpus | 20 | 1105 | 5-5 | 614 | held-out log rank | 5.4437 | 5.5889 | -0.1452 | -0.1863 to -0.1051 |
| nfcorpus | 20 | 1105 | 5-5 | 614 | nDCG@10, full judgments | 0.5368 | 0.5776 | -0.0407 | -0.0458 to -0.0356 |
| nfcorpus | 20 | 1105 | 5-5 | 716 | held-out reciprocal rank | 0.0408 | 0.0365 | 0.0043 | 0.0017 to 0.0069 |
| nfcorpus | 20 | 1105 | 5-5 | 716 | held-out log rank | 5.4589 | 5.5902 | -0.1314 | -0.1724 to -0.0906 |
| nfcorpus | 20 | 1105 | 5-5 | 716 | nDCG@10, full judgments | 0.5299 | 0.5678 | -0.0379 | -0.0426 to -0.0331 |
| nfcorpus | 20 | 1105 | 5-5 | 819 | held-out reciprocal rank | 0.0405 | 0.0379 | 0.0026 | -0.0002 to 0.0053 |
| nfcorpus | 20 | 1105 | 5-5 | 819 | held-out log rank | 5.4954 | 5.6042 | -0.1088 | -0.1493 to -0.0688 |
| nfcorpus | 20 | 1105 | 5-5 | 819 | nDCG@10, full judgments | 0.5157 | 0.5425 | -0.0268 | -0.0308 to -0.0230 |
| nfcorpus | 20 | 1105 | 5-5 | 921 | held-out reciprocal rank | 0.0396 | 0.0386 | 0.0009 | -0.0021 to 0.0039 |
| nfcorpus | 20 | 1105 | 5-5 | 921 | held-out log rank | 5.5758 | 5.6512 | -0.0754 | -0.1163 to -0.0352 |
| nfcorpus | 20 | 1105 | 5-5 | 921 | nDCG@10, full judgments | 0.4936 | 0.4965 | -0.0028 | -0.0053 to -0.0003 |
| nfcorpus | 20 | 1105 | 10-10 | 102 | held-out reciprocal rank | 0.0387 | 0.0303 | 0.0083 | 0.0064 to 0.0103 |
| nfcorpus | 20 | 1105 | 10-10 | 102 | held-out log rank | 5.4751 | 5.6891 | -0.2139 | -0.2557 to -0.1745 |
| nfcorpus | 20 | 1105 | 10-10 | 102 | nDCG@10, full judgments | 0.5290 | 0.5633 | -0.0342 | -0.0417 to -0.0271 |
| nfcorpus | 20 | 1105 | 10-10 | 204 | held-out reciprocal rank | 0.0402 | 0.0315 | 0.0087 | 0.0066 to 0.0108 |
| nfcorpus | 20 | 1105 | 10-10 | 204 | held-out log rank | 5.4119 | 5.6262 | -0.2143 | -0.2554 to -0.1755 |
| nfcorpus | 20 | 1105 | 10-10 | 204 | nDCG@10, full judgments | 0.5469 | 0.5808 | -0.0339 | -0.0409 to -0.0272 |
| nfcorpus | 20 | 1105 | 10-10 | 307 | held-out reciprocal rank | 0.0408 | 0.0334 | 0.0074 | 0.0052 to 0.0095 |
| nfcorpus | 20 | 1105 | 10-10 | 307 | held-out log rank | 5.3999 | 5.6034 | -0.2035 | -0.2437 to -0.1651 |
| nfcorpus | 20 | 1105 | 10-10 | 307 | nDCG@10, full judgments | 0.5469 | 0.5854 | -0.0385 | -0.0453 to -0.0320 |
| nfcorpus | 20 | 1105 | 10-10 | 409 | held-out reciprocal rank | 0.0407 | 0.0343 | 0.0064 | 0.0040 to 0.0086 |
| nfcorpus | 20 | 1105 | 10-10 | 409 | held-out log rank | 5.4056 | 5.5955 | -0.1898 | -0.2309 to -0.1513 |
| nfcorpus | 20 | 1105 | 10-10 | 409 | nDCG@10, full judgments | 0.5432 | 0.5853 | -0.0421 | -0.0485 to -0.0360 |
| nfcorpus | 20 | 1105 | 10-10 | 512 | held-out reciprocal rank | 0.0408 | 0.0350 | 0.0058 | 0.0034 to 0.0081 |
| nfcorpus | 20 | 1105 | 10-10 | 512 | held-out log rank | 5.4186 | 5.5914 | -0.1728 | -0.2130 to -0.1337 |
| nfcorpus | 20 | 1105 | 10-10 | 512 | nDCG@10, full judgments | 0.5379 | 0.5817 | -0.0438 | -0.0499 to -0.0381 |
| nfcorpus | 20 | 1105 | 10-10 | 614 | held-out reciprocal rank | 0.0407 | 0.0358 | 0.0050 | 0.0025 to 0.0073 |
| nfcorpus | 20 | 1105 | 10-10 | 614 | held-out log rank | 5.4335 | 5.5889 | -0.1554 | -0.1957 to -0.1159 |
| nfcorpus | 20 | 1105 | 10-10 | 614 | nDCG@10, full judgments | 0.5326 | 0.5776 | -0.0450 | -0.0509 to -0.0394 |
| nfcorpus | 20 | 1105 | 10-10 | 716 | held-out reciprocal rank | 0.0406 | 0.0365 | 0.0041 | 0.0016 to 0.0065 |
| nfcorpus | 20 | 1105 | 10-10 | 716 | held-out log rank | 5.4500 | 5.5902 | -0.1402 | -0.1810 to -0.1001 |
| nfcorpus | 20 | 1105 | 10-10 | 716 | nDCG@10, full judgments | 0.5271 | 0.5678 | -0.0407 | -0.0461 to -0.0356 |
| nfcorpus | 20 | 1105 | 10-10 | 819 | held-out reciprocal rank | 0.0404 | 0.0379 | 0.0025 | -0.0002 to 0.0051 |
| nfcorpus | 20 | 1105 | 10-10 | 819 | held-out log rank | 5.4878 | 5.6042 | -0.1164 | -0.1574 to -0.0759 |
| nfcorpus | 20 | 1105 | 10-10 | 819 | nDCG@10, full judgments | 0.5160 | 0.5425 | -0.0265 | -0.0309 to -0.0221 |
| nfcorpus | 20 | 1105 | 10-10 | 921 | held-out reciprocal rank | 0.0396 | 0.0386 | 0.0009 | -0.0019 to 0.0037 |
| nfcorpus | 20 | 1105 | 10-10 | 921 | held-out log rank | 5.5733 | 5.6512 | -0.0779 | -0.1183 to -0.0375 |
| nfcorpus | 20 | 1105 | 10-10 | 921 | nDCG@10, full judgments | 0.4950 | 0.4965 | -0.0015 | -0.0043 to 0.0013 |
| nfcorpus | 40 | 611 | 1-1 | 102 | held-out reciprocal rank | 0.0226 | 0.0226 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 102 | held-out log rank | 5.9319 | 5.9319 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 102 | nDCG@10, full judgments | 0.6104 | 0.6104 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 204 | held-out reciprocal rank | 0.0238 | 0.0238 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 204 | held-out log rank | 5.8740 | 5.8740 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 204 | nDCG@10, full judgments | 0.6301 | 0.6301 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 307 | held-out reciprocal rank | 0.0256 | 0.0256 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 307 | held-out log rank | 5.8491 | 5.8491 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 307 | nDCG@10, full judgments | 0.6359 | 0.6359 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 409 | held-out reciprocal rank | 0.0261 | 0.0261 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 409 | held-out log rank | 5.8404 | 5.8404 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 409 | nDCG@10, full judgments | 0.6370 | 0.6370 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 512 | held-out reciprocal rank | 0.0267 | 0.0267 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 512 | held-out log rank | 5.8341 | 5.8341 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 512 | nDCG@10, full judgments | 0.6348 | 0.6348 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 614 | held-out reciprocal rank | 0.0273 | 0.0273 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 614 | held-out log rank | 5.8314 | 5.8314 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 614 | nDCG@10, full judgments | 0.6315 | 0.6315 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 716 | held-out reciprocal rank | 0.0279 | 0.0279 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 716 | held-out log rank | 5.8272 | 5.8272 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 716 | nDCG@10, full judgments | 0.6232 | 0.6232 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 819 | held-out reciprocal rank | 0.0292 | 0.0292 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 819 | held-out log rank | 5.8291 | 5.8291 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 819 | nDCG@10, full judgments | 0.6021 | 0.6021 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 921 | held-out reciprocal rank | 0.0300 | 0.0300 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 921 | held-out log rank | 5.8530 | 5.8530 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 1-1 | 921 | nDCG@10, full judgments | 0.5655 | 0.5655 | 0.0000 | 0.0000 to 0.0000 |
| nfcorpus | 40 | 611 | 2-2 | 102 | held-out reciprocal rank | 0.0251 | 0.0226 | 0.0025 | -0.0002 to 0.0053 |
| nfcorpus | 40 | 611 | 2-2 | 102 | held-out log rank | 5.8362 | 5.9319 | -0.0958 | -0.1584 to -0.0369 |
| nfcorpus | 40 | 611 | 2-2 | 102 | nDCG@10, full judgments | 0.6157 | 0.6104 | 0.0053 | -0.0014 to 0.0115 |
| nfcorpus | 40 | 611 | 2-2 | 204 | held-out reciprocal rank | 0.0267 | 0.0238 | 0.0029 | 0.0001 to 0.0058 |
| nfcorpus | 40 | 611 | 2-2 | 204 | held-out log rank | 5.7726 | 5.8740 | -0.1014 | -0.1636 to -0.0407 |
| nfcorpus | 40 | 611 | 2-2 | 204 | nDCG@10, full judgments | 0.6352 | 0.6301 | 0.0051 | -0.0007 to 0.0111 |
| nfcorpus | 40 | 611 | 2-2 | 307 | held-out reciprocal rank | 0.0271 | 0.0256 | 0.0015 | -0.0016 to 0.0045 |
| nfcorpus | 40 | 611 | 2-2 | 307 | held-out log rank | 5.7547 | 5.8491 | -0.0945 | -0.1576 to -0.0344 |
| nfcorpus | 40 | 611 | 2-2 | 307 | nDCG@10, full judgments | 0.6339 | 0.6359 | -0.0021 | -0.0078 to 0.0037 |
| nfcorpus | 40 | 611 | 2-2 | 409 | held-out reciprocal rank | 0.0277 | 0.0261 | 0.0017 | -0.0015 to 0.0049 |
| nfcorpus | 40 | 611 | 2-2 | 409 | held-out log rank | 5.7483 | 5.8404 | -0.0921 | -0.1551 to -0.0317 |
| nfcorpus | 40 | 611 | 2-2 | 409 | nDCG@10, full judgments | 0.6290 | 0.6370 | -0.0080 | -0.0137 to -0.0023 |
| nfcorpus | 40 | 611 | 2-2 | 512 | held-out reciprocal rank | 0.0287 | 0.0267 | 0.0020 | -0.0014 to 0.0054 |
| nfcorpus | 40 | 611 | 2-2 | 512 | held-out log rank | 5.7445 | 5.8341 | -0.0896 | -0.1512 to -0.0309 |
| nfcorpus | 40 | 611 | 2-2 | 512 | nDCG@10, full judgments | 0.6212 | 0.6348 | -0.0136 | -0.0192 to -0.0082 |
| nfcorpus | 40 | 611 | 2-2 | 614 | held-out reciprocal rank | 0.0287 | 0.0273 | 0.0015 | -0.0019 to 0.0051 |
| nfcorpus | 40 | 611 | 2-2 | 614 | held-out log rank | 5.7473 | 5.8314 | -0.0841 | -0.1460 to -0.0244 |
| nfcorpus | 40 | 611 | 2-2 | 614 | nDCG@10, full judgments | 0.6143 | 0.6315 | -0.0172 | -0.0226 to -0.0119 |
| nfcorpus | 40 | 611 | 2-2 | 716 | held-out reciprocal rank | 0.0291 | 0.0279 | 0.0012 | -0.0023 to 0.0048 |
| nfcorpus | 40 | 611 | 2-2 | 716 | held-out log rank | 5.7504 | 5.8272 | -0.0768 | -0.1386 to -0.0169 |
| nfcorpus | 40 | 611 | 2-2 | 716 | nDCG@10, full judgments | 0.6050 | 0.6232 | -0.0183 | -0.0232 to -0.0132 |
| nfcorpus | 40 | 611 | 2-2 | 819 | held-out reciprocal rank | 0.0290 | 0.0292 | -0.0002 | -0.0040 to 0.0037 |
| nfcorpus | 40 | 611 | 2-2 | 819 | held-out log rank | 5.7642 | 5.8291 | -0.0649 | -0.1266 to -0.0046 |
| nfcorpus | 40 | 611 | 2-2 | 819 | nDCG@10, full judgments | 0.5873 | 0.6021 | -0.0148 | -0.0192 to -0.0104 |
| nfcorpus | 40 | 611 | 2-2 | 921 | held-out reciprocal rank | 0.0298 | 0.0300 | -0.0002 | -0.0041 to 0.0035 |
| nfcorpus | 40 | 611 | 2-2 | 921 | held-out log rank | 5.7968 | 5.8530 | -0.0562 | -0.1182 to 0.0023 |
| nfcorpus | 40 | 611 | 2-2 | 921 | nDCG@10, full judgments | 0.5645 | 0.5655 | -0.0010 | -0.0042 to 0.0025 |
| nfcorpus | 40 | 611 | 5-5 | 102 | held-out reciprocal rank | 0.0286 | 0.0226 | 0.0060 | 0.0035 to 0.0083 |
| nfcorpus | 40 | 611 | 5-5 | 102 | held-out log rank | 5.7748 | 5.9319 | -0.1571 | -0.2115 to -0.1032 |
| nfcorpus | 40 | 611 | 5-5 | 102 | nDCG@10, full judgments | 0.5834 | 0.6104 | -0.0270 | -0.0356 to -0.0185 |
| nfcorpus | 40 | 611 | 5-5 | 204 | held-out reciprocal rank | 0.0294 | 0.0238 | 0.0056 | 0.0031 to 0.0081 |
| nfcorpus | 40 | 611 | 5-5 | 204 | held-out log rank | 5.7110 | 5.8740 | -0.1631 | -0.2164 to -0.1084 |
| nfcorpus | 40 | 611 | 5-5 | 204 | nDCG@10, full judgments | 0.6056 | 0.6301 | -0.0245 | -0.0322 to -0.0163 |
| nfcorpus | 40 | 611 | 5-5 | 307 | held-out reciprocal rank | 0.0299 | 0.0256 | 0.0043 | 0.0015 to 0.0070 |
| nfcorpus | 40 | 611 | 5-5 | 307 | held-out log rank | 5.6957 | 5.8491 | -0.1534 | -0.2070 to -0.0994 |
| nfcorpus | 40 | 611 | 5-5 | 307 | nDCG@10, full judgments | 0.6062 | 0.6359 | -0.0297 | -0.0372 to -0.0220 |
| nfcorpus | 40 | 611 | 5-5 | 409 | held-out reciprocal rank | 0.0302 | 0.0261 | 0.0041 | 0.0013 to 0.0069 |
| nfcorpus | 40 | 611 | 5-5 | 409 | held-out log rank | 5.6932 | 5.8404 | -0.1472 | -0.2019 to -0.0937 |
| nfcorpus | 40 | 611 | 5-5 | 409 | nDCG@10, full judgments | 0.6035 | 0.6370 | -0.0334 | -0.0407 to -0.0262 |
| nfcorpus | 40 | 611 | 5-5 | 512 | held-out reciprocal rank | 0.0305 | 0.0267 | 0.0038 | 0.0009 to 0.0066 |
| nfcorpus | 40 | 611 | 5-5 | 512 | held-out log rank | 5.6960 | 5.8341 | -0.1381 | -0.1918 to -0.0843 |
| nfcorpus | 40 | 611 | 5-5 | 512 | nDCG@10, full judgments | 0.5988 | 0.6348 | -0.0361 | -0.0433 to -0.0289 |
| nfcorpus | 40 | 611 | 5-5 | 614 | held-out reciprocal rank | 0.0304 | 0.0273 | 0.0031 | 0.0002 to 0.0061 |
| nfcorpus | 40 | 611 | 5-5 | 614 | held-out log rank | 5.7028 | 5.8314 | -0.1287 | -0.1827 to -0.0759 |
| nfcorpus | 40 | 611 | 5-5 | 614 | nDCG@10, full judgments | 0.5944 | 0.6315 | -0.0370 | -0.0439 to -0.0302 |
| nfcorpus | 40 | 611 | 5-5 | 716 | held-out reciprocal rank | 0.0304 | 0.0279 | 0.0025 | -0.0006 to 0.0055 |
| nfcorpus | 40 | 611 | 5-5 | 716 | held-out log rank | 5.7116 | 5.8272 | -0.1156 | -0.1698 to -0.0617 |
| nfcorpus | 40 | 611 | 5-5 | 716 | nDCG@10, full judgments | 0.5893 | 0.6232 | -0.0339 | -0.0403 to -0.0276 |
| nfcorpus | 40 | 611 | 5-5 | 819 | held-out reciprocal rank | 0.0302 | 0.0292 | 0.0010 | -0.0023 to 0.0043 |
| nfcorpus | 40 | 611 | 5-5 | 819 | held-out log rank | 5.7341 | 5.8291 | -0.0950 | -0.1503 to -0.0405 |
| nfcorpus | 40 | 611 | 5-5 | 819 | nDCG@10, full judgments | 0.5792 | 0.6021 | -0.0229 | -0.0280 to -0.0176 |
| nfcorpus | 40 | 611 | 5-5 | 921 | held-out reciprocal rank | 0.0299 | 0.0300 | -0.0001 | -0.0036 to 0.0033 |
| nfcorpus | 40 | 611 | 5-5 | 921 | held-out log rank | 5.7834 | 5.8530 | -0.0696 | -0.1257 to -0.0161 |
| nfcorpus | 40 | 611 | 5-5 | 921 | nDCG@10, full judgments | 0.5637 | 0.5655 | -0.0018 | -0.0050 to 0.0017 |
| nfcorpus | 40 | 611 | 10-10 | 102 | held-out reciprocal rank | 0.0283 | 0.0226 | 0.0057 | 0.0032 to 0.0080 |
| nfcorpus | 40 | 611 | 10-10 | 102 | held-out log rank | 5.7559 | 5.9319 | -0.1760 | -0.2299 to -0.1231 |
| nfcorpus | 40 | 611 | 10-10 | 102 | nDCG@10, full judgments | 0.5796 | 0.6104 | -0.0308 | -0.0411 to -0.0214 |
| nfcorpus | 40 | 611 | 10-10 | 204 | held-out reciprocal rank | 0.0292 | 0.0238 | 0.0055 | 0.0030 to 0.0078 |
| nfcorpus | 40 | 611 | 10-10 | 204 | held-out log rank | 5.6987 | 5.8740 | -0.1754 | -0.2291 to -0.1232 |
| nfcorpus | 40 | 611 | 10-10 | 204 | nDCG@10, full judgments | 0.6008 | 0.6301 | -0.0293 | -0.0385 to -0.0200 |
| nfcorpus | 40 | 611 | 10-10 | 307 | held-out reciprocal rank | 0.0296 | 0.0256 | 0.0040 | 0.0014 to 0.0065 |
| nfcorpus | 40 | 611 | 10-10 | 307 | held-out log rank | 5.6854 | 5.8491 | -0.1637 | -0.2175 to -0.1124 |
| nfcorpus | 40 | 611 | 10-10 | 307 | nDCG@10, full judgments | 0.6018 | 0.6359 | -0.0342 | -0.0426 to -0.0254 |
| nfcorpus | 40 | 611 | 10-10 | 409 | held-out reciprocal rank | 0.0297 | 0.0261 | 0.0036 | 0.0009 to 0.0061 |
| nfcorpus | 40 | 611 | 10-10 | 409 | held-out log rank | 5.6863 | 5.8404 | -0.1541 | -0.2090 to -0.1029 |
| nfcorpus | 40 | 611 | 10-10 | 409 | nDCG@10, full judgments | 0.6002 | 0.6370 | -0.0368 | -0.0448 to -0.0284 |
| nfcorpus | 40 | 611 | 10-10 | 512 | held-out reciprocal rank | 0.0297 | 0.0267 | 0.0030 | 0.0003 to 0.0057 |
| nfcorpus | 40 | 611 | 10-10 | 512 | held-out log rank | 5.6950 | 5.8341 | -0.1392 | -0.1928 to -0.0883 |
| nfcorpus | 40 | 611 | 10-10 | 512 | nDCG@10, full judgments | 0.5962 | 0.6348 | -0.0387 | -0.0463 to -0.0306 |
| nfcorpus | 40 | 611 | 10-10 | 614 | held-out reciprocal rank | 0.0296 | 0.0273 | 0.0023 | -0.0006 to 0.0051 |
| nfcorpus | 40 | 611 | 10-10 | 614 | held-out log rank | 5.7049 | 5.8314 | -0.1265 | -0.1804 to -0.0750 |
| nfcorpus | 40 | 611 | 10-10 | 614 | nDCG@10, full judgments | 0.5920 | 0.6315 | -0.0395 | -0.0468 to -0.0317 |
| nfcorpus | 40 | 611 | 10-10 | 716 | held-out reciprocal rank | 0.0295 | 0.0279 | 0.0015 | -0.0016 to 0.0044 |
| nfcorpus | 40 | 611 | 10-10 | 716 | held-out log rank | 5.7159 | 5.8272 | -0.1113 | -0.1650 to -0.0589 |
| nfcorpus | 40 | 611 | 10-10 | 716 | nDCG@10, full judgments | 0.5885 | 0.6232 | -0.0347 | -0.0416 to -0.0278 |
| nfcorpus | 40 | 611 | 10-10 | 819 | held-out reciprocal rank | 0.0294 | 0.0292 | 0.0002 | -0.0031 to 0.0033 |
| nfcorpus | 40 | 611 | 10-10 | 819 | held-out log rank | 5.7414 | 5.8291 | -0.0877 | -0.1424 to -0.0353 |
| nfcorpus | 40 | 611 | 10-10 | 819 | nDCG@10, full judgments | 0.5811 | 0.6021 | -0.0210 | -0.0264 to -0.0153 |
| nfcorpus | 40 | 611 | 10-10 | 921 | held-out reciprocal rank | 0.0291 | 0.0300 | -0.0008 | -0.0043 to 0.0024 |
| nfcorpus | 40 | 611 | 10-10 | 921 | held-out log rank | 5.7961 | 5.8530 | -0.0569 | -0.1110 to -0.0047 |
| nfcorpus | 40 | 611 | 10-10 | 921 | nDCG@10, full judgments | 0.5657 | 0.5655 | 0.0002 | -0.0033 to 0.0039 |
| nfcorpus | 40 | 611 | 20-20 | 102 | held-out reciprocal rank | 0.0297 | 0.0226 | 0.0071 | 0.0048 to 0.0093 |
| nfcorpus | 40 | 611 | 20-20 | 102 | held-out log rank | 5.7292 | 5.9319 | -0.2027 | -0.2562 to -0.1490 |
| nfcorpus | 40 | 611 | 20-20 | 102 | nDCG@10, full judgments | 0.5777 | 0.6104 | -0.0327 | -0.0430 to -0.0229 |
| nfcorpus | 40 | 611 | 20-20 | 204 | held-out reciprocal rank | 0.0309 | 0.0238 | 0.0071 | 0.0047 to 0.0094 |
| nfcorpus | 40 | 611 | 20-20 | 204 | held-out log rank | 5.6702 | 5.8740 | -0.2038 | -0.2569 to -0.1504 |
| nfcorpus | 40 | 611 | 20-20 | 204 | nDCG@10, full judgments | 0.5993 | 0.6301 | -0.0309 | -0.0402 to -0.0213 |
| nfcorpus | 40 | 611 | 20-20 | 307 | held-out reciprocal rank | 0.0312 | 0.0256 | 0.0056 | 0.0030 to 0.0081 |
| nfcorpus | 40 | 611 | 20-20 | 307 | held-out log rank | 5.6610 | 5.8491 | -0.1882 | -0.2402 to -0.1339 |
| nfcorpus | 40 | 611 | 20-20 | 307 | nDCG@10, full judgments | 0.6000 | 0.6359 | -0.0359 | -0.0447 to -0.0268 |
| nfcorpus | 40 | 611 | 20-20 | 409 | held-out reciprocal rank | 0.0313 | 0.0261 | 0.0052 | 0.0025 to 0.0077 |
| nfcorpus | 40 | 611 | 20-20 | 409 | held-out log rank | 5.6621 | 5.8404 | -0.1783 | -0.2306 to -0.1250 |
| nfcorpus | 40 | 611 | 20-20 | 409 | nDCG@10, full judgments | 0.5984 | 0.6370 | -0.0386 | -0.0473 to -0.0300 |
| nfcorpus | 40 | 611 | 20-20 | 512 | held-out reciprocal rank | 0.0313 | 0.0267 | 0.0047 | 0.0020 to 0.0073 |
| nfcorpus | 40 | 611 | 20-20 | 512 | held-out log rank | 5.6707 | 5.8341 | -0.1635 | -0.2158 to -0.1104 |
| nfcorpus | 40 | 611 | 20-20 | 512 | nDCG@10, full judgments | 0.5948 | 0.6348 | -0.0400 | -0.0483 to -0.0317 |
| nfcorpus | 40 | 611 | 20-20 | 614 | held-out reciprocal rank | 0.0312 | 0.0273 | 0.0040 | 0.0011 to 0.0066 |
| nfcorpus | 40 | 611 | 20-20 | 614 | held-out log rank | 5.6818 | 5.8314 | -0.1496 | -0.2015 to -0.0973 |
| nfcorpus | 40 | 611 | 20-20 | 614 | nDCG@10, full judgments | 0.5914 | 0.6315 | -0.0401 | -0.0481 to -0.0321 |
| nfcorpus | 40 | 611 | 20-20 | 716 | held-out reciprocal rank | 0.0312 | 0.0279 | 0.0032 | 0.0002 to 0.0060 |
| nfcorpus | 40 | 611 | 20-20 | 716 | held-out log rank | 5.6938 | 5.8272 | -0.1334 | -0.1860 to -0.0800 |
| nfcorpus | 40 | 611 | 20-20 | 716 | nDCG@10, full judgments | 0.5878 | 0.6232 | -0.0354 | -0.0426 to -0.0281 |
| nfcorpus | 40 | 611 | 20-20 | 819 | held-out reciprocal rank | 0.0310 | 0.0292 | 0.0018 | -0.0015 to 0.0047 |
| nfcorpus | 40 | 611 | 20-20 | 819 | held-out log rank | 5.7199 | 5.8291 | -0.1092 | -0.1641 to -0.0571 |
| nfcorpus | 40 | 611 | 20-20 | 819 | nDCG@10, full judgments | 0.5807 | 0.6021 | -0.0214 | -0.0272 to -0.0153 |
| nfcorpus | 40 | 611 | 20-20 | 921 | held-out reciprocal rank | 0.0305 | 0.0300 | 0.0005 | -0.0029 to 0.0036 |
| nfcorpus | 40 | 611 | 20-20 | 921 | held-out log rank | 5.7781 | 5.8530 | -0.0749 | -0.1279 to -0.0233 |
| nfcorpus | 40 | 611 | 20-20 | 921 | nDCG@10, full judgments | 0.5663 | 0.5655 | 0.0008 | -0.0028 to 0.0048 |
| msmarco | 4 | 81 | 1-1 | 102 | held-out reciprocal rank | 0.0785 | 0.0785 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 102 | held-out log rank | 4.0643 | 4.0643 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 102 | nDCG@10, full judgments | 0.4507 | 0.4507 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 204 | held-out reciprocal rank | 0.0809 | 0.0809 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 204 | held-out log rank | 3.9746 | 3.9746 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 204 | nDCG@10, full judgments | 0.4449 | 0.4449 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 307 | held-out reciprocal rank | 0.0829 | 0.0829 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 307 | held-out log rank | 3.9394 | 3.9394 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 307 | nDCG@10, full judgments | 0.4361 | 0.4361 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 409 | held-out reciprocal rank | 0.0846 | 0.0846 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 409 | held-out log rank | 3.9303 | 3.9303 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 409 | nDCG@10, full judgments | 0.4131 | 0.4131 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 512 | held-out reciprocal rank | 0.0841 | 0.0841 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 512 | held-out log rank | 3.9317 | 3.9317 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 512 | nDCG@10, full judgments | 0.3854 | 0.3854 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 614 | held-out reciprocal rank | 0.0864 | 0.0864 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 614 | held-out log rank | 3.9298 | 3.9298 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 614 | nDCG@10, full judgments | 0.3596 | 0.3596 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 716 | held-out reciprocal rank | 0.0893 | 0.0893 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 716 | held-out log rank | 3.9207 | 3.9207 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 716 | nDCG@10, full judgments | 0.3397 | 0.3397 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 819 | held-out reciprocal rank | 0.0951 | 0.0951 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 819 | held-out log rank | 3.9174 | 3.9174 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 819 | nDCG@10, full judgments | 0.3114 | 0.3114 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 921 | held-out reciprocal rank | 0.0974 | 0.0974 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 921 | held-out log rank | 3.9414 | 3.9414 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 1-1 | 921 | nDCG@10, full judgments | 0.2678 | 0.2678 | 0.0000 | 0.0000 to 0.0000 |
| msmarco | 4 | 81 | 2-2 | 102 | held-out reciprocal rank | 0.0833 | 0.0785 | 0.0048 | -0.0083 to 0.0184 |
| msmarco | 4 | 81 | 2-2 | 102 | held-out log rank | 3.9734 | 4.0643 | -0.0909 | -0.2236 to 0.0482 |
| msmarco | 4 | 81 | 2-2 | 102 | nDCG@10, full judgments | 0.4033 | 0.4507 | -0.0475 | -0.0742 to -0.0209 |
| msmarco | 4 | 81 | 2-2 | 204 | held-out reciprocal rank | 0.0887 | 0.0809 | 0.0077 | -0.0060 to 0.0214 |
| msmarco | 4 | 81 | 2-2 | 204 | held-out log rank | 3.8795 | 3.9746 | -0.0951 | -0.2178 to 0.0345 |
| msmarco | 4 | 81 | 2-2 | 204 | nDCG@10, full judgments | 0.3990 | 0.4449 | -0.0459 | -0.0732 to -0.0199 |
| msmarco | 4 | 81 | 2-2 | 307 | held-out reciprocal rank | 0.0892 | 0.0829 | 0.0064 | -0.0069 to 0.0205 |
| msmarco | 4 | 81 | 2-2 | 307 | held-out log rank | 3.8557 | 3.9394 | -0.0837 | -0.2081 to 0.0469 |
| msmarco | 4 | 81 | 2-2 | 307 | nDCG@10, full judgments | 0.3813 | 0.4361 | -0.0548 | -0.0794 to -0.0313 |
| msmarco | 4 | 81 | 2-2 | 409 | held-out reciprocal rank | 0.0930 | 0.0846 | 0.0084 | -0.0045 to 0.0223 |
| msmarco | 4 | 81 | 2-2 | 409 | held-out log rank | 3.8583 | 3.9303 | -0.0720 | -0.2000 to 0.0575 |
| msmarco | 4 | 81 | 2-2 | 409 | nDCG@10, full judgments | 0.3609 | 0.4131 | -0.0521 | -0.0727 to -0.0323 |
| msmarco | 4 | 81 | 2-2 | 512 | held-out reciprocal rank | 0.0938 | 0.0841 | 0.0097 | -0.0033 to 0.0236 |
| msmarco | 4 | 81 | 2-2 | 512 | held-out log rank | 3.8712 | 3.9317 | -0.0605 | -0.1867 to 0.0737 |
| msmarco | 4 | 81 | 2-2 | 512 | nDCG@10, full judgments | 0.3417 | 0.3854 | -0.0438 | -0.0612 to -0.0250 |
| msmarco | 4 | 81 | 2-2 | 614 | held-out reciprocal rank | 0.0946 | 0.0864 | 0.0081 | -0.0078 to 0.0237 |
| msmarco | 4 | 81 | 2-2 | 614 | held-out log rank | 3.8744 | 3.9298 | -0.0554 | -0.1852 to 0.0832 |
| msmarco | 4 | 81 | 2-2 | 614 | nDCG@10, full judgments | 0.3190 | 0.3596 | -0.0406 | -0.0570 to -0.0236 |
| msmarco | 4 | 81 | 2-2 | 716 | held-out reciprocal rank | 0.0946 | 0.0893 | 0.0052 | -0.0100 to 0.0208 |
| msmarco | 4 | 81 | 2-2 | 716 | held-out log rank | 3.8796 | 3.9207 | -0.0411 | -0.1744 to 0.0971 |
| msmarco | 4 | 81 | 2-2 | 716 | nDCG@10, full judgments | 0.3045 | 0.3397 | -0.0352 | -0.0511 to -0.0200 |
| msmarco | 4 | 81 | 2-2 | 819 | held-out reciprocal rank | 0.0975 | 0.0951 | 0.0024 | -0.0163 to 0.0203 |
| msmarco | 4 | 81 | 2-2 | 819 | held-out log rank | 3.8863 | 3.9174 | -0.0311 | -0.1670 to 0.1115 |
| msmarco | 4 | 81 | 2-2 | 819 | nDCG@10, full judgments | 0.2912 | 0.3114 | -0.0203 | -0.0339 to -0.0066 |
| msmarco | 4 | 81 | 2-2 | 921 | held-out reciprocal rank | 0.0983 | 0.0974 | 0.0009 | -0.0172 to 0.0186 |
| msmarco | 4 | 81 | 2-2 | 921 | held-out log rank | 3.9156 | 3.9414 | -0.0258 | -0.1608 to 0.1135 |
| msmarco | 4 | 81 | 2-2 | 921 | nDCG@10, full judgments | 0.2637 | 0.2678 | -0.0040 | -0.0138 to 0.0055 |

## Small sample of competing scores

A seeded sample of queries, fixed before results: the worst-ranked evaluated positive against the mask's top non-positive documents, scored under both queries. Diagnostic only.

| environment | query position | arm | k | rank of the worst evaluated positive | its rank, full query | its score, full query | its score, mask | competitors' mean score, full query | competitors' mean score, mask | competitors |
|---|---|---|---|---|---|---|---|---|---|---|
| scifact | 7 | shared | 102 | 3 | 4 | 0.71977 | 0.37764 | 0.53502 | 0.27425 | 10 |
| scifact | 7 | shared | 204 | 4 | 4 | 0.71977 | 0.53535 | 0.53061 | 0.39409 | 10 |
| scifact | 7 | shared | 307 | 4 | 4 | 0.71977 | 0.63226 | 0.53502 | 0.45032 | 10 |
| scifact | 7 | shared | 409 | 4 | 4 | 0.71977 | 0.68667 | 0.53502 | 0.49874 | 10 |
| scifact | 7 | shared | 512 | 4 | 4 | 0.71977 | 0.72002 | 0.53733 | 0.52379 | 10 |
| scifact | 7 | shared | 614 | 4 | 4 | 0.71977 | 0.73604 | 0.53733 | 0.53800 | 10 |
| scifact | 7 | shared | 716 | 4 | 4 | 0.71977 | 0.74567 | 0.53733 | 0.54321 | 10 |
| scifact | 7 | shared | 819 | 4 | 4 | 0.71977 | 0.75085 | 0.53733 | 0.54582 | 10 |
| scifact | 7 | shared | 921 | 4 | 4 | 0.71977 | 0.74663 | 0.53733 | 0.54523 | 10 |
| scifact | 7 | 1-1 | 102 | 2 | 3 | 0.74493 | 0.37752 | 0.53379 | 0.27454 | 10 |
| scifact | 7 | 1-1 | 204 | 3 | 3 | 0.74493 | 0.52818 | 0.53379 | 0.38814 | 10 |
| scifact | 7 | 1-1 | 307 | 3 | 3 | 0.74493 | 0.61854 | 0.53502 | 0.45038 | 10 |
| scifact | 7 | 1-1 | 409 | 3 | 3 | 0.74493 | 0.67693 | 0.53502 | 0.49418 | 10 |
| scifact | 7 | 1-1 | 512 | 3 | 3 | 0.74493 | 0.71513 | 0.53502 | 0.52569 | 10 |
| scifact | 7 | 1-1 | 614 | 3 | 3 | 0.74493 | 0.73288 | 0.53502 | 0.53799 | 10 |
| scifact | 7 | 1-1 | 716 | 3 | 3 | 0.74493 | 0.74368 | 0.53502 | 0.54534 | 10 |
| scifact | 7 | 1-1 | 819 | 3 | 3 | 0.74493 | 0.74651 | 0.53733 | 0.54533 | 10 |
| scifact | 7 | 1-1 | 921 | 3 | 3 | 0.74493 | 0.74959 | 0.53733 | 0.54270 | 10 |
| scifact | 10 | shared | 102 | 4 | 4 | 0.76872 | 0.37494 | 0.63207 | 0.31114 | 10 |
| scifact | 10 | shared | 204 | 4 | 4 | 0.76872 | 0.54660 | 0.63734 | 0.44838 | 10 |
| scifact | 10 | shared | 307 | 4 | 3 | 0.77427 | 0.65536 | 0.63734 | 0.53746 | 10 |
| scifact | 10 | shared | 409 | 4 | 4 | 0.76872 | 0.71533 | 0.63841 | 0.58948 | 10 |
| scifact | 10 | shared | 512 | 4 | 4 | 0.76872 | 0.75341 | 0.63734 | 0.61832 | 10 |
| scifact | 10 | shared | 614 | 4 | 4 | 0.76872 | 0.77533 | 0.63872 | 0.63533 | 10 |
| scifact | 10 | shared | 716 | 4 | 4 | 0.76872 | 0.78458 | 0.63872 | 0.64400 | 10 |
| scifact | 10 | shared | 819 | 4 | 4 | 0.76872 | 0.78794 | 0.63872 | 0.64718 | 10 |
| scifact | 10 | shared | 921 | 4 | 4 | 0.76872 | 0.78596 | 0.63872 | 0.64581 | 10 |
| scifact | 10 | 1-1 | 102 | 3 | 2 | 0.79300 | 0.35885 | 0.63199 | 0.29707 | 10 |
| scifact | 10 | 1-1 | 204 | 3 | 2 | 0.79300 | 0.52870 | 0.63734 | 0.43660 | 10 |
| scifact | 10 | 1-1 | 307 | 3 | 2 | 0.79300 | 0.63216 | 0.63734 | 0.51871 | 10 |
| scifact | 10 | 1-1 | 409 | 3 | 2 | 0.79300 | 0.69898 | 0.63841 | 0.56933 | 10 |
| scifact | 10 | 1-1 | 512 | 3 | 2 | 0.79300 | 0.74285 | 0.63734 | 0.60230 | 10 |
| scifact | 10 | 1-1 | 614 | 3 | 2 | 0.79300 | 0.76726 | 0.63872 | 0.62050 | 10 |
| scifact | 10 | 1-1 | 716 | 3 | 2 | 0.79300 | 0.78161 | 0.63872 | 0.63072 | 10 |
| scifact | 10 | 1-1 | 819 | 3 | 2 | 0.79300 | 0.78716 | 0.63872 | 0.63509 | 10 |
| scifact | 10 | 1-1 | 921 | 3 | 2 | 0.79300 | 0.78940 | 0.63872 | 0.63603 | 10 |
| scifact | 19 | shared | 102 | 3 | 79 | 0.50131 | 0.30683 | 0.57941 | 0.29770 | 10 |
| scifact | 19 | shared | 204 | 3 | 79 | 0.50131 | 0.42245 | 0.58338 | 0.41214 | 10 |
| scifact | 19 | shared | 307 | 5 | 79 | 0.50131 | 0.50198 | 0.59752 | 0.49240 | 10 |
| scifact | 19 | shared | 409 | 6 | 79 | 0.50131 | 0.54202 | 0.59399 | 0.53656 | 10 |
| scifact | 19 | shared | 512 | 7 | 79 | 0.50131 | 0.56365 | 0.59449 | 0.57296 | 10 |
| scifact | 19 | shared | 614 | 12 | 79 | 0.50131 | 0.56784 | 0.59876 | 0.58653 | 10 |
| scifact | 19 | shared | 716 | 14 | 79 | 0.50131 | 0.56928 | 0.59823 | 0.59163 | 10 |
| scifact | 19 | shared | 819 | 15 | 79 | 0.50131 | 0.56759 | 0.60127 | 0.59249 | 10 |
| scifact | 19 | shared | 921 | 21 | 79 | 0.50131 | 0.55466 | 0.60127 | 0.59472 | 10 |
| scifact | 19 | 1-1 | 102 | 9 | 1 | 0.68594 | 0.26700 | 0.54836 | 0.27027 | 10 |
| scifact | 19 | 1-1 | 204 | 7 | 1 | 0.68594 | 0.37663 | 0.54715 | 0.38237 | 10 |
| scifact | 19 | 1-1 | 307 | 4 | 1 | 0.68594 | 0.45805 | 0.54949 | 0.44881 | 10 |
| scifact | 19 | 1-1 | 409 | 4 | 1 | 0.68594 | 0.51056 | 0.55957 | 0.49055 | 10 |
| scifact | 19 | 1-1 | 512 | 2 | 1 | 0.68594 | 0.55544 | 0.56465 | 0.51530 | 10 |
| scifact | 19 | 1-1 | 614 | 2 | 1 | 0.68594 | 0.57375 | 0.56699 | 0.52819 | 10 |
| scifact | 19 | 1-1 | 716 | 2 | 1 | 0.68594 | 0.58379 | 0.57243 | 0.53740 | 10 |
| scifact | 19 | 1-1 | 819 | 1 | 1 | 0.68594 | 0.60631 | 0.59055 | 0.54557 | 10 |
| scifact | 19 | 1-1 | 921 | 1 | 1 | 0.68594 | 0.63084 | 0.59423 | 0.56248 | 10 |
| scifact | 30 | shared | 102 | 2 | 2 | 0.65216 | 0.37716 | 0.55391 | 0.30445 | 10 |
| scifact | 30 | shared | 204 | 2 | 2 | 0.65216 | 0.51646 | 0.55391 | 0.41519 | 10 |
| scifact | 30 | shared | 307 | 2 | 2 | 0.65216 | 0.60527 | 0.55915 | 0.47603 | 10 |
| scifact | 30 | shared | 409 | 2 | 2 | 0.65216 | 0.65795 | 0.55915 | 0.51737 | 10 |
| scifact | 30 | shared | 512 | 2 | 2 | 0.65216 | 0.68790 | 0.55914 | 0.55039 | 10 |
| scifact | 30 | shared | 614 | 2 | 2 | 0.65216 | 0.70070 | 0.55958 | 0.56228 | 10 |
| scifact | 30 | shared | 716 | 2 | 2 | 0.65216 | 0.70369 | 0.55958 | 0.56747 | 10 |
| scifact | 30 | shared | 819 | 2 | 2 | 0.65216 | 0.70008 | 0.56052 | 0.56693 | 10 |
| scifact | 30 | shared | 921 | 2 | 2 | 0.65216 | 0.69284 | 0.56212 | 0.57114 | 10 |
| scifact | 30 | 1-1 | 102 | 2 | 2 | 0.65216 | 0.36377 | 0.55312 | 0.30291 | 10 |
| scifact | 30 | 1-1 | 204 | 2 | 2 | 0.65216 | 0.49408 | 0.55371 | 0.41027 | 10 |
| scifact | 30 | 1-1 | 307 | 2 | 2 | 0.65216 | 0.57043 | 0.55704 | 0.47539 | 10 |
| scifact | 30 | 1-1 | 409 | 2 | 2 | 0.65216 | 0.62017 | 0.55704 | 0.51523 | 10 |
| scifact | 30 | 1-1 | 512 | 2 | 2 | 0.65216 | 0.65065 | 0.56052 | 0.53651 | 10 |
| scifact | 30 | 1-1 | 614 | 2 | 2 | 0.65216 | 0.66886 | 0.56052 | 0.55128 | 10 |
| scifact | 30 | 1-1 | 716 | 2 | 2 | 0.65216 | 0.67687 | 0.56052 | 0.55713 | 10 |
| scifact | 30 | 1-1 | 819 | 2 | 2 | 0.65216 | 0.67661 | 0.56052 | 0.55814 | 10 |
| scifact | 30 | 1-1 | 921 | 2 | 2 | 0.65216 | 0.67533 | 0.56052 | 0.56570 | 10 |
| scifact | 33 | shared | 102 | 6 | 12 | 0.57117 | 0.33345 | 0.70336 | 0.32831 | 10 |
| scifact | 33 | shared | 204 | 7 | 12 | 0.57117 | 0.46500 | 0.70336 | 0.46275 | 10 |
| scifact | 33 | shared | 307 | 9 | 12 | 0.57117 | 0.53686 | 0.70336 | 0.55705 | 10 |
| scifact | 33 | shared | 409 | 9 | 12 | 0.57117 | 0.58235 | 0.70336 | 0.61221 | 10 |
| scifact | 33 | shared | 512 | 11 | 12 | 0.57117 | 0.61320 | 0.70336 | 0.65076 | 10 |
| scifact | 33 | shared | 614 | 11 | 12 | 0.57117 | 0.61968 | 0.70336 | 0.67626 | 10 |
| scifact | 33 | shared | 716 | 11 | 12 | 0.57117 | 0.62449 | 0.70336 | 0.68583 | 10 |
| scifact | 33 | shared | 819 | 11 | 12 | 0.57117 | 0.62402 | 0.70336 | 0.69067 | 10 |
| scifact | 33 | shared | 921 | 12 | 12 | 0.57117 | 0.61315 | 0.70336 | 0.69679 | 10 |
| scifact | 33 | 1-1 | 102 | 12 | 12 | 0.57117 | 0.26296 | 0.70336 | 0.32145 | 10 |
| scifact | 33 | 1-1 | 204 | 12 | 12 | 0.57117 | 0.36709 | 0.70336 | 0.47005 | 10 |
| scifact | 33 | 1-1 | 307 | 12 | 12 | 0.57117 | 0.44180 | 0.70336 | 0.55916 | 10 |
| scifact | 33 | 1-1 | 409 | 12 | 12 | 0.57117 | 0.48106 | 0.70336 | 0.61648 | 10 |
| scifact | 33 | 1-1 | 512 | 12 | 12 | 0.57117 | 0.50958 | 0.70336 | 0.65531 | 10 |
| scifact | 33 | 1-1 | 614 | 12 | 12 | 0.57117 | 0.53152 | 0.70336 | 0.67666 | 10 |
| scifact | 33 | 1-1 | 716 | 12 | 12 | 0.57117 | 0.54320 | 0.70336 | 0.68792 | 10 |
| scifact | 33 | 1-1 | 819 | 12 | 12 | 0.57117 | 0.54971 | 0.70336 | 0.69267 | 10 |
| scifact | 33 | 1-1 | 921 | 12 | 12 | 0.57117 | 0.55681 | 0.70336 | 0.69594 | 10 |
| scifact | 37 | shared | 102 | 2 | 5 | 0.60341 | 0.34201 | 0.58614 | 0.29235 | 10 |
| scifact | 37 | shared | 204 | 3 | 5 | 0.60341 | 0.48167 | 0.58941 | 0.42442 | 10 |
| scifact | 37 | shared | 307 | 3 | 5 | 0.60341 | 0.57363 | 0.59296 | 0.49891 | 10 |
| scifact | 37 | shared | 409 | 3 | 5 | 0.60341 | 0.62289 | 0.59296 | 0.54624 | 10 |
| scifact | 37 | shared | 512 | 3 | 5 | 0.60341 | 0.64818 | 0.59296 | 0.57610 | 10 |
| scifact | 37 | shared | 614 | 3 | 5 | 0.60341 | 0.66435 | 0.59296 | 0.59241 | 10 |
| scifact | 37 | shared | 716 | 3 | 5 | 0.60341 | 0.66625 | 0.59296 | 0.60105 | 10 |
| scifact | 37 | shared | 819 | 3 | 5 | 0.60341 | 0.66433 | 0.59296 | 0.60405 | 10 |
| scifact | 37 | shared | 921 | 3 | 5 | 0.60341 | 0.65510 | 0.59296 | 0.60301 | 10 |
| scifact | 37 | 1-1 | 102 | 1 | 1 | 0.73174 | 0.35998 | 0.58614 | 0.29034 | 10 |
| scifact | 37 | 1-1 | 204 | 1 | 1 | 0.73174 | 0.50770 | 0.59296 | 0.41102 | 10 |
| scifact | 37 | 1-1 | 307 | 1 | 1 | 0.73174 | 0.59263 | 0.59296 | 0.48465 | 10 |
| scifact | 37 | 1-1 | 409 | 1 | 1 | 0.73174 | 0.64880 | 0.59296 | 0.53011 | 10 |
| scifact | 37 | 1-1 | 512 | 1 | 1 | 0.73174 | 0.68412 | 0.59296 | 0.55828 | 10 |
| scifact | 37 | 1-1 | 614 | 1 | 1 | 0.73174 | 0.70348 | 0.59296 | 0.57364 | 10 |
| scifact | 37 | 1-1 | 716 | 1 | 1 | 0.73174 | 0.71360 | 0.59296 | 0.58020 | 10 |
| scifact | 37 | 1-1 | 819 | 1 | 1 | 0.73174 | 0.71771 | 0.59296 | 0.58485 | 10 |
| scifact | 37 | 1-1 | 921 | 1 | 1 | 0.73174 | 0.72396 | 0.59296 | 0.58822 | 10 |
| scifact | 38 | shared | 102 | 3 | 18 | 0.48309 | 0.29961 | 0.56655 | 0.27671 | 10 |
| scifact | 38 | shared | 204 | 3 | 18 | 0.48309 | 0.42700 | 0.56473 | 0.39766 | 10 |
| scifact | 38 | shared | 307 | 3 | 18 | 0.48309 | 0.50624 | 0.56655 | 0.47918 | 10 |
| scifact | 38 | shared | 409 | 4 | 18 | 0.48309 | 0.55261 | 0.56655 | 0.53115 | 10 |
| scifact | 38 | shared | 512 | 4 | 18 | 0.48309 | 0.57444 | 0.56655 | 0.55937 | 10 |
| scifact | 38 | shared | 614 | 4 | 18 | 0.48309 | 0.58457 | 0.56655 | 0.57271 | 10 |
| scifact | 38 | shared | 716 | 5 | 18 | 0.48309 | 0.58239 | 0.56655 | 0.57856 | 10 |
| scifact | 38 | shared | 819 | 8 | 18 | 0.48309 | 0.57196 | 0.56655 | 0.57990 | 10 |
| scifact | 38 | shared | 921 | 10 | 18 | 0.48309 | 0.55216 | 0.56655 | 0.58111 | 10 |
| scifact | 38 | 1-1 | 102 | 2 | 6 | 0.57138 | 0.29409 | 0.56326 | 0.27210 | 10 |
| scifact | 38 | 1-1 | 204 | 3 | 6 | 0.57138 | 0.41712 | 0.56655 | 0.39053 | 10 |
| scifact | 38 | 1-1 | 307 | 3 | 6 | 0.57138 | 0.49099 | 0.56655 | 0.46638 | 10 |
| scifact | 38 | 1-1 | 409 | 3 | 6 | 0.57138 | 0.54888 | 0.56655 | 0.51375 | 10 |
| scifact | 38 | 1-1 | 512 | 3 | 6 | 0.57138 | 0.57538 | 0.56655 | 0.54186 | 10 |
| scifact | 38 | 1-1 | 614 | 3 | 6 | 0.57138 | 0.59085 | 0.56655 | 0.55653 | 10 |
| scifact | 38 | 1-1 | 716 | 3 | 6 | 0.57138 | 0.59185 | 0.56655 | 0.55885 | 10 |
| scifact | 38 | 1-1 | 819 | 3 | 6 | 0.57138 | 0.58889 | 0.56655 | 0.56325 | 10 |
| scifact | 38 | 1-1 | 921 | 3 | 6 | 0.57138 | 0.58531 | 0.56655 | 0.56592 | 10 |
| scifact | 47 | shared | 102 | 2 | 5 | 0.61218 | 0.33839 | 0.58044 | 0.28695 | 10 |
| scifact | 47 | shared | 204 | 2 | 5 | 0.61218 | 0.47216 | 0.58909 | 0.41122 | 10 |
| scifact | 47 | shared | 307 | 2 | 5 | 0.61218 | 0.56538 | 0.59402 | 0.49118 | 10 |
| scifact | 47 | shared | 409 | 2 | 5 | 0.61218 | 0.61992 | 0.59398 | 0.54321 | 10 |
| scifact | 47 | shared | 512 | 2 | 5 | 0.61218 | 0.64418 | 0.59336 | 0.57711 | 10 |
| scifact | 47 | shared | 614 | 2 | 5 | 0.61218 | 0.65939 | 0.59336 | 0.59670 | 10 |
| scifact | 47 | shared | 716 | 2 | 5 | 0.61218 | 0.66403 | 0.59336 | 0.60438 | 10 |
| scifact | 47 | shared | 819 | 2 | 5 | 0.61218 | 0.66425 | 0.59402 | 0.60817 | 10 |
| scifact | 47 | shared | 921 | 2 | 5 | 0.61218 | 0.65215 | 0.59650 | 0.60419 | 10 |
| scifact | 47 | 1-1 | 102 | 2 | 1 | 0.76908 | 0.31807 | 0.56689 | 0.27123 | 10 |
| scifact | 47 | 1-1 | 204 | 2 | 1 | 0.76908 | 0.46430 | 0.57877 | 0.39062 | 10 |
| scifact | 47 | 1-1 | 307 | 2 | 1 | 0.76908 | 0.56529 | 0.58945 | 0.46764 | 10 |
| scifact | 47 | 1-1 | 409 | 1 | 1 | 0.76908 | 0.64159 | 0.59438 | 0.52194 | 10 |
| scifact | 47 | 1-1 | 512 | 1 | 1 | 0.76908 | 0.68318 | 0.59560 | 0.55384 | 10 |
| scifact | 47 | 1-1 | 614 | 1 | 1 | 0.76908 | 0.70572 | 0.59653 | 0.57051 | 10 |
| scifact | 47 | 1-1 | 716 | 1 | 1 | 0.76908 | 0.71951 | 0.59560 | 0.58131 | 10 |
| scifact | 47 | 1-1 | 819 | 1 | 1 | 0.76908 | 0.72784 | 0.59560 | 0.58334 | 10 |
| scifact | 47 | 1-1 | 921 | 1 | 1 | 0.76908 | 0.73602 | 0.59715 | 0.58931 | 10 |
| scifact | 61 | shared | 102 | 7 | 13 | 0.50814 | 0.31480 | 0.58247 | 0.30413 | 10 |
| scifact | 61 | shared | 204 | 8 | 13 | 0.50814 | 0.43968 | 0.58312 | 0.43060 | 10 |
| scifact | 61 | shared | 307 | 8 | 13 | 0.50814 | 0.50631 | 0.58454 | 0.50578 | 10 |
| scifact | 61 | shared | 409 | 8 | 13 | 0.50814 | 0.54396 | 0.58454 | 0.55037 | 10 |
| scifact | 61 | shared | 512 | 8 | 13 | 0.50814 | 0.56327 | 0.58247 | 0.57434 | 10 |
| scifact | 61 | shared | 614 | 9 | 13 | 0.50814 | 0.57434 | 0.58982 | 0.58771 | 10 |
| scifact | 61 | shared | 716 | 9 | 13 | 0.50814 | 0.57854 | 0.58982 | 0.59170 | 10 |
| scifact | 61 | shared | 819 | 10 | 13 | 0.50814 | 0.57648 | 0.58982 | 0.59541 | 10 |
| scifact | 61 | shared | 921 | 10 | 13 | 0.50814 | 0.56487 | 0.59047 | 0.59426 | 10 |
| scifact | 61 | 1-1 | 102 | 11 | 13 | 0.50814 | 0.26608 | 0.56920 | 0.28163 | 10 |
| scifact | 61 | 1-1 | 204 | 10 | 13 | 0.50814 | 0.37168 | 0.58840 | 0.38914 | 10 |
| scifact | 61 | 1-1 | 307 | 11 | 13 | 0.50814 | 0.43173 | 0.58840 | 0.46187 | 10 |
| scifact | 61 | 1-1 | 409 | 9 | 13 | 0.50814 | 0.47700 | 0.58840 | 0.50014 | 10 |
| scifact | 61 | 1-1 | 512 | 10 | 13 | 0.50814 | 0.49666 | 0.58840 | 0.52841 | 10 |
| scifact | 61 | 1-1 | 614 | 10 | 13 | 0.50814 | 0.50378 | 0.58840 | 0.54248 | 10 |
| scifact | 61 | 1-1 | 716 | 10 | 13 | 0.50814 | 0.50922 | 0.58840 | 0.54824 | 10 |
| scifact | 61 | 1-1 | 819 | 10 | 13 | 0.50814 | 0.51550 | 0.58840 | 0.55851 | 10 |
| scifact | 61 | 1-1 | 921 | 11 | 13 | 0.50814 | 0.52355 | 0.59047 | 0.57456 | 10 |
| scifact | 68 | shared | 102 | 5 | 386 | 0.45305 | 0.29058 | 0.56808 | 0.28949 | 10 |
| scifact | 68 | shared | 204 | 19 | 386 | 0.45305 | 0.38605 | 0.59162 | 0.40351 | 10 |
| scifact | 68 | shared | 307 | 18 | 386 | 0.45305 | 0.45536 | 0.58547 | 0.47648 | 10 |
| scifact | 68 | shared | 409 | 16 | 386 | 0.45305 | 0.49933 | 0.58542 | 0.51714 | 10 |
| scifact | 68 | shared | 512 | 19 | 386 | 0.45305 | 0.51971 | 0.59081 | 0.54305 | 10 |
| scifact | 68 | shared | 614 | 23 | 386 | 0.45305 | 0.52763 | 0.60077 | 0.56054 | 10 |
| scifact | 68 | shared | 716 | 36 | 386 | 0.45305 | 0.52501 | 0.60077 | 0.57006 | 10 |
| scifact | 68 | shared | 819 | 51 | 386 | 0.45305 | 0.52309 | 0.60077 | 0.57951 | 10 |
| scifact | 68 | shared | 921 | 133 | 386 | 0.45305 | 0.50874 | 0.60296 | 0.59528 | 10 |
| scifact | 68 | 1-1 | 102 | 569 | 386 | 0.45305 | 0.22451 | 0.59375 | 0.29164 | 10 |
| scifact | 68 | 1-1 | 204 | 553 | 386 | 0.45305 | 0.31205 | 0.59850 | 0.40949 | 10 |
| scifact | 68 | 1-1 | 307 | 527 | 386 | 0.45305 | 0.35978 | 0.59337 | 0.47259 | 10 |
| scifact | 68 | 1-1 | 409 | 510 | 386 | 0.45305 | 0.39098 | 0.59838 | 0.51139 | 10 |
| scifact | 68 | 1-1 | 512 | 610 | 386 | 0.45305 | 0.39848 | 0.60142 | 0.53493 | 10 |
| scifact | 68 | 1-1 | 614 | 536 | 386 | 0.45305 | 0.41786 | 0.60077 | 0.55288 | 10 |
| scifact | 68 | 1-1 | 716 | 523 | 386 | 0.45305 | 0.42232 | 0.60077 | 0.56172 | 10 |
| scifact | 68 | 1-1 | 819 | 461 | 386 | 0.45305 | 0.43376 | 0.60077 | 0.57152 | 10 |
| scifact | 68 | 1-1 | 921 | 434 | 386 | 0.45305 | 0.44377 | 0.60352 | 0.58611 | 10 |
| nfcorpus | 109 | shared | 102 | 3077 | 2026 | 0.26185 | 0.14968 | 0.43348 | 0.25286 | 10 |
| nfcorpus | 109 | shared | 204 | 3236 | 3055 | 0.22116 | 0.20574 | 0.46203 | 0.35412 | 10 |
| nfcorpus | 109 | shared | 307 | 3121 | 2726 | 0.23659 | 0.24823 | 0.46392 | 0.41152 | 10 |
| nfcorpus | 109 | shared | 409 | 3117 | 2750 | 0.23541 | 0.27056 | 0.45991 | 0.44553 | 10 |
| nfcorpus | 109 | shared | 512 | 3038 | 2750 | 0.23541 | 0.28410 | 0.46554 | 0.46612 | 10 |
| nfcorpus | 109 | shared | 614 | 3178 | 2750 | 0.23541 | 0.28064 | 0.46473 | 0.47527 | 10 |
| nfcorpus | 109 | shared | 716 | 3141 | 2750 | 0.23541 | 0.28145 | 0.46491 | 0.47646 | 10 |
| nfcorpus | 109 | shared | 819 | 3042 | 3267 | 0.20700 | 0.28091 | 0.46591 | 0.48294 | 10 |
| nfcorpus | 109 | shared | 921 | 3251 | 3267 | 0.20700 | 0.25741 | 0.46765 | 0.48329 | 10 |
| nfcorpus | 109 | 1-1 | 102 | 2161 | 1496 | 0.27939 | 0.12662 | 0.30336 | 0.20788 | 10 |
| nfcorpus | 109 | 1-1 | 204 | 1995 | 1496 | 0.27939 | 0.17961 | 0.34538 | 0.28393 | 10 |
| nfcorpus | 109 | 1-1 | 307 | 2855 | 1496 | 0.27939 | 0.19041 | 0.39125 | 0.33154 | 10 |
| nfcorpus | 109 | 1-1 | 409 | 2769 | 1496 | 0.27939 | 0.21781 | 0.38841 | 0.36627 | 10 |
| nfcorpus | 109 | 1-1 | 512 | 2665 | 1496 | 0.27939 | 0.23008 | 0.42715 | 0.38195 | 10 |
| nfcorpus | 109 | 1-1 | 614 | 2517 | 1496 | 0.27939 | 0.23919 | 0.44312 | 0.39123 | 10 |
| nfcorpus | 109 | 1-1 | 716 | 2520 | 1496 | 0.27939 | 0.24239 | 0.45936 | 0.39746 | 10 |
| nfcorpus | 109 | 1-1 | 819 | 2561 | 1496 | 0.27939 | 0.24841 | 0.46181 | 0.41568 | 10 |
| nfcorpus | 109 | 1-1 | 921 | 2272 | 1496 | 0.27939 | 0.25969 | 0.47001 | 0.43141 | 10 |
| nfcorpus | 167 | shared | 102 | 3087 | 3341 | 0.17428 | 0.12941 | 0.35903 | 0.24057 | 10 |
| nfcorpus | 167 | shared | 204 | 3098 | 3341 | 0.17428 | 0.17918 | 0.36472 | 0.33110 | 10 |
| nfcorpus | 167 | shared | 307 | 3224 | 3341 | 0.17428 | 0.20398 | 0.39708 | 0.38096 | 10 |
| nfcorpus | 167 | shared | 409 | 3371 | 3341 | 0.17428 | 0.21380 | 0.40188 | 0.42295 | 10 |
| nfcorpus | 167 | shared | 512 | 3394 | 3341 | 0.17428 | 0.22214 | 0.40188 | 0.44332 | 10 |
| nfcorpus | 167 | shared | 614 | 3364 | 3341 | 0.17428 | 0.22515 | 0.42018 | 0.45617 | 10 |
| nfcorpus | 167 | shared | 716 | 3406 | 3341 | 0.17428 | 0.22015 | 0.41983 | 0.46072 | 10 |
| nfcorpus | 167 | shared | 819 | 3410 | 3341 | 0.17428 | 0.21802 | 0.42864 | 0.45996 | 10 |
| nfcorpus | 167 | shared | 921 | 3403 | 3559 | 0.13888 | 0.20907 | 0.44125 | 0.45374 | 10 |
| nfcorpus | 167 | 1-1 | 102 | 1138 | 2135 | 0.23448 | 0.13393 | 0.38094 | 0.20320 | 10 |
| nfcorpus | 167 | 1-1 | 204 | 868 | 2135 | 0.23448 | 0.19701 | 0.41272 | 0.28463 | 10 |
| nfcorpus | 167 | 1-1 | 307 | 1192 | 2135 | 0.23448 | 0.22394 | 0.41082 | 0.33300 | 10 |
| nfcorpus | 167 | 1-1 | 409 | 1588 | 2135 | 0.23448 | 0.22845 | 0.41108 | 0.36081 | 10 |
| nfcorpus | 167 | 1-1 | 512 | 1689 | 2135 | 0.23448 | 0.23505 | 0.41108 | 0.38108 | 10 |
| nfcorpus | 167 | 1-1 | 614 | 1545 | 2135 | 0.23448 | 0.24569 | 0.41108 | 0.38589 | 10 |
| nfcorpus | 167 | 1-1 | 716 | 1394 | 2135 | 0.23448 | 0.25511 | 0.41108 | 0.39215 | 10 |
| nfcorpus | 167 | 1-1 | 819 | 1459 | 2135 | 0.23448 | 0.25799 | 0.41801 | 0.40214 | 10 |
| nfcorpus | 167 | 1-1 | 921 | 1640 | 2135 | 0.23448 | 0.26575 | 0.43716 | 0.42481 | 10 |
| nfcorpus | 545 | shared | 102 | 2813 | 1570 | 0.35048 | 0.16265 | 0.64256 | 0.30606 | 10 |
| nfcorpus | 545 | shared | 204 | 2604 | 2841 | 0.27863 | 0.23498 | 0.63880 | 0.43881 | 10 |
| nfcorpus | 545 | shared | 307 | 2861 | 2860 | 0.27705 | 0.26600 | 0.64251 | 0.51873 | 10 |
| nfcorpus | 545 | shared | 409 | 2870 | 2860 | 0.27705 | 0.29100 | 0.64251 | 0.57827 | 10 |
| nfcorpus | 545 | shared | 512 | 2782 | 2860 | 0.27705 | 0.31235 | 0.64251 | 0.60897 | 10 |
| nfcorpus | 545 | shared | 614 | 2753 | 2841 | 0.27863 | 0.31895 | 0.64442 | 0.62931 | 10 |
| nfcorpus | 545 | shared | 716 | 2818 | 2855 | 0.27748 | 0.31543 | 0.64438 | 0.63889 | 10 |
| nfcorpus | 545 | shared | 819 | 2816 | 2855 | 0.27748 | 0.31309 | 0.64438 | 0.64605 | 10 |
| nfcorpus | 545 | shared | 921 | 2831 | 2855 | 0.27748 | 0.30606 | 0.64510 | 0.64840 | 10 |
| nfcorpus | 545 | 1-1 | 102 | 2198 | 1666 | 0.34432 | 0.14580 | 0.62625 | 0.26004 | 10 |
| nfcorpus | 545 | 1-1 | 204 | 2496 | 1666 | 0.34432 | 0.19482 | 0.62685 | 0.36079 | 10 |
| nfcorpus | 545 | 1-1 | 307 | 2243 | 1666 | 0.34432 | 0.23460 | 0.62761 | 0.41904 | 10 |
| nfcorpus | 545 | 1-1 | 409 | 1877 | 1666 | 0.34432 | 0.27061 | 0.63547 | 0.46931 | 10 |
| nfcorpus | 545 | 1-1 | 512 | 1994 | 1666 | 0.34432 | 0.28076 | 0.63486 | 0.49733 | 10 |
| nfcorpus | 545 | 1-1 | 614 | 2112 | 1666 | 0.34432 | 0.28608 | 0.63497 | 0.51779 | 10 |
| nfcorpus | 545 | 1-1 | 716 | 2096 | 1666 | 0.34432 | 0.29182 | 0.63302 | 0.52805 | 10 |
| nfcorpus | 545 | 1-1 | 819 | 1991 | 1666 | 0.34432 | 0.30044 | 0.63486 | 0.53792 | 10 |
| nfcorpus | 545 | 1-1 | 921 | 2038 | 1666 | 0.34432 | 0.30499 | 0.63960 | 0.56455 | 10 |
| nfcorpus | 813 | shared | 102 | 813 | 2792 | 0.16203 | 0.15980 | 0.37528 | 0.23789 | 10 |
| nfcorpus | 813 | shared | 204 | 742 | 2792 | 0.16203 | 0.22288 | 0.40963 | 0.32891 | 10 |
| nfcorpus | 813 | shared | 307 | 733 | 2792 | 0.16203 | 0.26369 | 0.42559 | 0.39137 | 10 |
| nfcorpus | 813 | shared | 409 | 794 | 2792 | 0.16203 | 0.28463 | 0.42559 | 0.42949 | 10 |
| nfcorpus | 813 | shared | 512 | 735 | 2792 | 0.16203 | 0.29696 | 0.43223 | 0.44710 | 10 |
| nfcorpus | 813 | shared | 614 | 775 | 2792 | 0.16203 | 0.30038 | 0.43914 | 0.45546 | 10 |
| nfcorpus | 813 | shared | 716 | 741 | 2792 | 0.16203 | 0.30237 | 0.43751 | 0.45865 | 10 |
| nfcorpus | 813 | shared | 819 | 907 | 2792 | 0.16203 | 0.29381 | 0.43751 | 0.46099 | 10 |
| nfcorpus | 813 | shared | 921 | 1163 | 2792 | 0.16203 | 0.26907 | 0.44167 | 0.45573 | 10 |
| nfcorpus | 813 | 1-1 | 102 | 1621 | 2792 | 0.16203 | 0.12700 | 0.40886 | 0.20966 | 10 |
| nfcorpus | 813 | 1-1 | 204 | 2684 | 2792 | 0.16203 | 0.14612 | 0.41440 | 0.29037 | 10 |
| nfcorpus | 813 | 1-1 | 307 | 2413 | 2792 | 0.16203 | 0.17700 | 0.40454 | 0.33786 | 10 |
| nfcorpus | 813 | 1-1 | 409 | 2393 | 2792 | 0.16203 | 0.19111 | 0.41856 | 0.37081 | 10 |
| nfcorpus | 813 | 1-1 | 512 | 2055 | 2792 | 0.16203 | 0.21415 | 0.43129 | 0.39396 | 10 |
| nfcorpus | 813 | 1-1 | 614 | 2150 | 2792 | 0.16203 | 0.21775 | 0.43129 | 0.40282 | 10 |
| nfcorpus | 813 | 1-1 | 716 | 2152 | 2792 | 0.16203 | 0.21524 | 0.43129 | 0.40479 | 10 |
| nfcorpus | 813 | 1-1 | 819 | 2086 | 2792 | 0.16203 | 0.21990 | 0.43296 | 0.41468 | 10 |
| nfcorpus | 813 | 1-1 | 921 | 2062 | 2792 | 0.16203 | 0.21303 | 0.44240 | 0.42335 | 10 |
| nfcorpus | 833 | shared | 102 | 1117 | 1554 | 0.26373 | 0.17281 | 0.42652 | 0.26942 | 10 |
| nfcorpus | 833 | shared | 204 | 1875 | 1554 | 0.26373 | 0.21285 | 0.43533 | 0.38055 | 10 |
| nfcorpus | 833 | shared | 307 | 2154 | 1554 | 0.26373 | 0.23517 | 0.44142 | 0.44293 | 10 |
| nfcorpus | 833 | shared | 409 | 2323 | 1554 | 0.26373 | 0.24889 | 0.44144 | 0.47912 | 10 |
| nfcorpus | 833 | shared | 512 | 2091 | 1554 | 0.26373 | 0.27055 | 0.45247 | 0.50019 | 10 |
| nfcorpus | 833 | shared | 614 | 2170 | 1554 | 0.26373 | 0.27129 | 0.45247 | 0.50642 | 10 |
| nfcorpus | 833 | shared | 716 | 2247 | 1554 | 0.26373 | 0.26731 | 0.45247 | 0.50930 | 10 |
| nfcorpus | 833 | shared | 819 | 2129 | 1554 | 0.26373 | 0.27115 | 0.46416 | 0.50733 | 10 |
| nfcorpus | 833 | shared | 921 | 2110 | 1554 | 0.26373 | 0.26211 | 0.47051 | 0.50046 | 10 |
| nfcorpus | 833 | 1-1 | 102 | 14 | 18 | 0.44862 | 0.24884 | 0.43614 | 0.25502 | 10 |
| nfcorpus | 833 | 1-1 | 204 | 15 | 18 | 0.44862 | 0.34891 | 0.43815 | 0.35777 | 10 |
| nfcorpus | 833 | 1-1 | 307 | 12 | 18 | 0.44862 | 0.40604 | 0.44075 | 0.40979 | 10 |
| nfcorpus | 833 | 1-1 | 409 | 13 | 18 | 0.44862 | 0.43861 | 0.43738 | 0.44771 | 10 |
| nfcorpus | 833 | 1-1 | 512 | 14 | 18 | 0.44862 | 0.45403 | 0.43808 | 0.46655 | 10 |
| nfcorpus | 833 | 1-1 | 614 | 13 | 18 | 0.44862 | 0.46285 | 0.43886 | 0.47343 | 10 |
| nfcorpus | 833 | 1-1 | 716 | 15 | 18 | 0.44862 | 0.46250 | 0.44178 | 0.47594 | 10 |
| nfcorpus | 833 | 1-1 | 819 | 15 | 18 | 0.44862 | 0.46396 | 0.45909 | 0.47664 | 10 |
| nfcorpus | 833 | 1-1 | 921 | 14 | 18 | 0.44862 | 0.45995 | 0.46966 | 0.47357 | 10 |
| nfcorpus | 1383 | shared | 102 | 2311 | 1441 | 0.31027 | 0.16450 | 0.58724 | 0.27203 | 10 |
| nfcorpus | 1383 | shared | 204 | 2159 | 2919 | 0.23533 | 0.23975 | 0.59694 | 0.40402 | 10 |
| nfcorpus | 1383 | shared | 307 | 1737 | 2919 | 0.23533 | 0.28880 | 0.60099 | 0.47140 | 10 |
| nfcorpus | 1383 | shared | 409 | 2063 | 2919 | 0.23533 | 0.29861 | 0.60246 | 0.51786 | 10 |
| nfcorpus | 1383 | shared | 512 | 1985 | 2919 | 0.23533 | 0.31542 | 0.59920 | 0.54809 | 10 |
| nfcorpus | 1383 | shared | 614 | 2085 | 2919 | 0.23533 | 0.31768 | 0.59920 | 0.56251 | 10 |
| nfcorpus | 1383 | shared | 716 | 2163 | 2919 | 0.23533 | 0.31538 | 0.61255 | 0.56532 | 10 |
| nfcorpus | 1383 | shared | 819 | 2209 | 2919 | 0.23533 | 0.31010 | 0.61255 | 0.57392 | 10 |
| nfcorpus | 1383 | shared | 921 | 2439 | 2919 | 0.23533 | 0.29293 | 0.61935 | 0.59093 | 10 |
| nfcorpus | 1383 | 1-1 | 102 | 1119 | 1441 | 0.31027 | 0.15431 | 0.53834 | 0.26320 | 10 |
| nfcorpus | 1383 | 1-1 | 204 | 1281 | 1441 | 0.31027 | 0.21551 | 0.58124 | 0.37724 | 10 |
| nfcorpus | 1383 | 1-1 | 307 | 1383 | 1441 | 0.31027 | 0.25173 | 0.58425 | 0.44405 | 10 |
| nfcorpus | 1383 | 1-1 | 409 | 1388 | 1441 | 0.31027 | 0.26966 | 0.59204 | 0.48306 | 10 |
| nfcorpus | 1383 | 1-1 | 512 | 1668 | 1441 | 0.31027 | 0.26999 | 0.59331 | 0.51122 | 10 |
| nfcorpus | 1383 | 1-1 | 614 | 1535 | 1441 | 0.31027 | 0.28167 | 0.60336 | 0.52412 | 10 |
| nfcorpus | 1383 | 1-1 | 716 | 1598 | 1441 | 0.31027 | 0.28171 | 0.60603 | 0.53064 | 10 |
| nfcorpus | 1383 | 1-1 | 819 | 1695 | 1441 | 0.31027 | 0.28098 | 0.60603 | 0.54361 | 10 |
| nfcorpus | 1383 | 1-1 | 921 | 1664 | 1441 | 0.31027 | 0.29139 | 0.61237 | 0.57306 | 10 |
| nfcorpus | 1390 | shared | 102 | 350 | 849 | 0.30747 | 0.20975 | 0.45355 | 0.27329 | 10 |
| nfcorpus | 1390 | shared | 204 | 263 | 387 | 0.34352 | 0.29630 | 0.45069 | 0.37301 | 10 |
| nfcorpus | 1390 | shared | 307 | 249 | 387 | 0.34352 | 0.34644 | 0.46198 | 0.43057 | 10 |
| nfcorpus | 1390 | shared | 409 | 293 | 387 | 0.34352 | 0.36437 | 0.47161 | 0.46602 | 10 |
| nfcorpus | 1390 | shared | 512 | 309 | 387 | 0.34352 | 0.37681 | 0.47602 | 0.48694 | 10 |
| nfcorpus | 1390 | shared | 614 | 380 | 387 | 0.34352 | 0.37466 | 0.48755 | 0.50054 | 10 |
| nfcorpus | 1390 | shared | 716 | 392 | 387 | 0.34352 | 0.37530 | 0.49866 | 0.50438 | 10 |
| nfcorpus | 1390 | shared | 819 | 464 | 387 | 0.34352 | 0.36891 | 0.49866 | 0.50895 | 10 |
| nfcorpus | 1390 | shared | 921 | 489 | 387 | 0.34352 | 0.35917 | 0.51317 | 0.51251 | 10 |
| nfcorpus | 1390 | 1-1 | 102 | 884 | 312 | 0.35136 | 0.17523 | 0.43673 | 0.23251 | 10 |
| nfcorpus | 1390 | 1-1 | 204 | 500 | 312 | 0.35136 | 0.25511 | 0.46669 | 0.33123 | 10 |
| nfcorpus | 1390 | 1-1 | 307 | 584 | 312 | 0.35136 | 0.28249 | 0.46152 | 0.37666 | 10 |
| nfcorpus | 1390 | 1-1 | 409 | 619 | 312 | 0.35136 | 0.30476 | 0.44648 | 0.40944 | 10 |
| nfcorpus | 1390 | 1-1 | 512 | 538 | 312 | 0.35136 | 0.32227 | 0.46726 | 0.43149 | 10 |
| nfcorpus | 1390 | 1-1 | 614 | 687 | 312 | 0.35136 | 0.31914 | 0.48345 | 0.44261 | 10 |
| nfcorpus | 1390 | 1-1 | 716 | 572 | 312 | 0.35136 | 0.33013 | 0.49322 | 0.45305 | 10 |
| nfcorpus | 1390 | 1-1 | 819 | 452 | 312 | 0.35136 | 0.33821 | 0.50224 | 0.45871 | 10 |
| nfcorpus | 1390 | 1-1 | 921 | 456 | 312 | 0.35136 | 0.33760 | 0.51403 | 0.47381 | 10 |
| nfcorpus | 1481 | shared | 102 | 3012 | 3624 | 0.09175 | 0.13174 | 0.39256 | 0.25812 | 10 |
| nfcorpus | 1481 | shared | 204 | 2898 | 3624 | 0.09175 | 0.18110 | 0.37325 | 0.35943 | 10 |
| nfcorpus | 1481 | shared | 307 | 2668 | 3624 | 0.09175 | 0.21951 | 0.37182 | 0.41425 | 10 |
| nfcorpus | 1481 | shared | 409 | 3008 | 3624 | 0.09175 | 0.22737 | 0.37182 | 0.44422 | 10 |
| nfcorpus | 1481 | shared | 512 | 2987 | 3624 | 0.09175 | 0.23227 | 0.37182 | 0.45909 | 10 |
| nfcorpus | 1481 | shared | 614 | 3107 | 3624 | 0.09175 | 0.23002 | 0.38298 | 0.46647 | 10 |
| nfcorpus | 1481 | shared | 716 | 3230 | 3624 | 0.09175 | 0.22363 | 0.38298 | 0.46752 | 10 |
| nfcorpus | 1481 | shared | 819 | 3468 | 3624 | 0.09175 | 0.20174 | 0.41253 | 0.46256 | 10 |
| nfcorpus | 1481 | shared | 921 | 3569 | 3624 | 0.09175 | 0.17424 | 0.41757 | 0.44977 | 10 |
| nfcorpus | 1481 | 1-1 | 102 | 42 | 346 | 0.29969 | 0.20042 | 0.39064 | 0.21754 | 10 |
| nfcorpus | 1481 | 1-1 | 204 | 39 | 346 | 0.29969 | 0.27177 | 0.41104 | 0.29234 | 10 |
| nfcorpus | 1481 | 1-1 | 307 | 39 | 346 | 0.29969 | 0.31309 | 0.41104 | 0.34038 | 10 |
| nfcorpus | 1481 | 1-1 | 409 | 44 | 346 | 0.29969 | 0.33354 | 0.39912 | 0.36447 | 10 |
| nfcorpus | 1481 | 1-1 | 512 | 41 | 346 | 0.29969 | 0.35272 | 0.40698 | 0.38242 | 10 |
| nfcorpus | 1481 | 1-1 | 614 | 35 | 346 | 0.29969 | 0.36404 | 0.41339 | 0.39061 | 10 |
| nfcorpus | 1481 | 1-1 | 716 | 37 | 346 | 0.29969 | 0.36870 | 0.41187 | 0.39549 | 10 |
| nfcorpus | 1481 | 1-1 | 819 | 54 | 346 | 0.29969 | 0.35902 | 0.42677 | 0.40402 | 10 |
| nfcorpus | 1481 | 1-1 | 921 | 82 | 346 | 0.29969 | 0.33697 | 0.43134 | 0.40647 | 10 |
| nfcorpus | 1800 | shared | 102 | 3302 | 3260 | 0.12167 | 0.10188 | 0.43758 | 0.27896 | 10 |
| nfcorpus | 1800 | shared | 204 | 3278 | 3260 | 0.12167 | 0.14917 | 0.43839 | 0.37412 | 10 |
| nfcorpus | 1800 | shared | 307 | 2974 | 3260 | 0.12167 | 0.19279 | 0.44671 | 0.43383 | 10 |
| nfcorpus | 1800 | shared | 409 | 3051 | 3260 | 0.12167 | 0.19832 | 0.44671 | 0.46607 | 10 |
| nfcorpus | 1800 | shared | 512 | 2993 | 3260 | 0.12167 | 0.20534 | 0.44714 | 0.48788 | 10 |
| nfcorpus | 1800 | shared | 614 | 2944 | 3260 | 0.12167 | 0.20900 | 0.44714 | 0.49734 | 10 |
| nfcorpus | 1800 | shared | 716 | 2970 | 3260 | 0.12167 | 0.20523 | 0.44714 | 0.50044 | 10 |
| nfcorpus | 1800 | shared | 819 | 3003 | 3260 | 0.12167 | 0.19711 | 0.44716 | 0.49964 | 10 |
| nfcorpus | 1800 | shared | 921 | 3207 | 3260 | 0.12167 | 0.17445 | 0.45365 | 0.49089 | 10 |
| nfcorpus | 1800 | 1-1 | 102 | 16 | 6 | 0.48670 | 0.17237 | 0.36077 | 0.18096 | 10 |
| nfcorpus | 1800 | 1-1 | 204 | 26 | 6 | 0.48670 | 0.22270 | 0.41374 | 0.23756 | 10 |
| nfcorpus | 1800 | 1-1 | 307 | 15 | 6 | 0.48670 | 0.26881 | 0.42808 | 0.27510 | 10 |
| nfcorpus | 1800 | 1-1 | 409 | 15 | 6 | 0.48670 | 0.29283 | 0.42539 | 0.29858 | 10 |
| nfcorpus | 1800 | 1-1 | 512 | 10 | 6 | 0.48670 | 0.31826 | 0.43623 | 0.31105 | 10 |
| nfcorpus | 1800 | 1-1 | 614 | 9 | 6 | 0.48670 | 0.32207 | 0.42769 | 0.31690 | 10 |
| nfcorpus | 1800 | 1-1 | 716 | 12 | 6 | 0.48670 | 0.33654 | 0.43015 | 0.33365 | 10 |
| nfcorpus | 1800 | 1-1 | 819 | 9 | 6 | 0.48670 | 0.36990 | 0.45092 | 0.36285 | 10 |
| nfcorpus | 1800 | 1-1 | 921 | 6 | 6 | 0.48670 | 0.40119 | 0.45436 | 0.38074 | 10 |
| nfcorpus | 2072 | shared | 102 | 4 | 28 | 0.41858 | 0.25344 | 0.45655 | 0.24774 | 10 |
| nfcorpus | 2072 | shared | 204 | 4 | 28 | 0.41858 | 0.35276 | 0.45443 | 0.34083 | 10 |
| nfcorpus | 2072 | shared | 307 | 4 | 28 | 0.41858 | 0.41828 | 0.45443 | 0.40505 | 10 |
| nfcorpus | 2072 | shared | 409 | 4 | 28 | 0.41858 | 0.46134 | 0.46177 | 0.44961 | 10 |
| nfcorpus | 2072 | shared | 512 | 4 | 28 | 0.41858 | 0.48501 | 0.46177 | 0.47492 | 10 |
| nfcorpus | 2072 | shared | 614 | 4 | 28 | 0.41858 | 0.49751 | 0.46177 | 0.48831 | 10 |
| nfcorpus | 2072 | shared | 716 | 3 | 28 | 0.41858 | 0.50668 | 0.46177 | 0.49117 | 10 |
| nfcorpus | 2072 | shared | 819 | 4 | 28 | 0.41858 | 0.50389 | 0.47137 | 0.49543 | 10 |
| nfcorpus | 2072 | shared | 921 | 8 | 28 | 0.41858 | 0.48449 | 0.47425 | 0.49389 | 10 |
| nfcorpus | 2072 | 1-1 | 102 | 70 | 28 | 0.41858 | 0.18674 | 0.45024 | 0.23952 | 10 |
| nfcorpus | 2072 | 1-1 | 204 | 47 | 28 | 0.41858 | 0.26987 | 0.44875 | 0.33213 | 10 |
| nfcorpus | 2072 | 1-1 | 307 | 49 | 28 | 0.41858 | 0.32041 | 0.44875 | 0.39161 | 10 |
| nfcorpus | 2072 | 1-1 | 409 | 49 | 28 | 0.41858 | 0.35312 | 0.44084 | 0.42800 | 10 |
| nfcorpus | 2072 | 1-1 | 512 | 47 | 28 | 0.41858 | 0.37938 | 0.44875 | 0.45396 | 10 |
| nfcorpus | 2072 | 1-1 | 614 | 42 | 28 | 0.41858 | 0.39709 | 0.44875 | 0.46673 | 10 |
| nfcorpus | 2072 | 1-1 | 716 | 43 | 28 | 0.41858 | 0.40382 | 0.44875 | 0.47210 | 10 |
| nfcorpus | 2072 | 1-1 | 819 | 39 | 28 | 0.41858 | 0.41361 | 0.45443 | 0.47657 | 10 |
| nfcorpus | 2072 | 1-1 | 921 | 38 | 28 | 0.41858 | 0.41218 | 0.46858 | 0.47779 | 10 |
| msmarco | 31 | shared | 102 | 4 | 14 | 0.59070 | 0.33801 | 0.59670 | 0.30295 | 10 |
| msmarco | 31 | shared | 204 | 4 | 14 | 0.59070 | 0.48951 | 0.60789 | 0.44184 | 10 |
| msmarco | 31 | shared | 307 | 4 | 14 | 0.59070 | 0.57830 | 0.60174 | 0.53105 | 10 |
| msmarco | 31 | shared | 409 | 4 | 14 | 0.59070 | 0.62647 | 0.60174 | 0.57896 | 10 |
| msmarco | 31 | shared | 512 | 4 | 14 | 0.59070 | 0.65374 | 0.60174 | 0.60762 | 10 |
| msmarco | 31 | shared | 614 | 4 | 14 | 0.59070 | 0.66633 | 0.61945 | 0.62200 | 10 |
| msmarco | 31 | shared | 716 | 4 | 14 | 0.59070 | 0.66801 | 0.61945 | 0.63224 | 10 |
| msmarco | 31 | shared | 819 | 4 | 14 | 0.59070 | 0.66547 | 0.61945 | 0.63389 | 10 |
| msmarco | 31 | shared | 921 | 5 | 14 | 0.59070 | 0.65096 | 0.61945 | 0.63246 | 10 |
| msmarco | 31 | 1-1 | 102 | 1 | 1 | 0.66914 | 0.34713 | 0.60174 | 0.30051 | 10 |
| msmarco | 31 | 1-1 | 204 | 1 | 1 | 0.66914 | 0.50442 | 0.60789 | 0.43714 | 10 |
| msmarco | 31 | 1-1 | 307 | 1 | 1 | 0.66914 | 0.60007 | 0.60174 | 0.52375 | 10 |
| msmarco | 31 | 1-1 | 409 | 1 | 1 | 0.66914 | 0.65255 | 0.60826 | 0.57499 | 10 |
| msmarco | 31 | 1-1 | 512 | 1 | 1 | 0.66914 | 0.68771 | 0.61945 | 0.60057 | 10 |
| msmarco | 31 | 1-1 | 614 | 1 | 1 | 0.66914 | 0.70588 | 0.60826 | 0.61484 | 10 |
| msmarco | 31 | 1-1 | 716 | 1 | 1 | 0.66914 | 0.70727 | 0.61945 | 0.62230 | 10 |
| msmarco | 31 | 1-1 | 819 | 1 | 1 | 0.66914 | 0.70920 | 0.61945 | 0.62710 | 10 |
| msmarco | 31 | 1-1 | 921 | 1 | 1 | 0.66914 | 0.70187 | 0.61945 | 0.63022 | 10 |
| msmarco | 397 | shared | 102 | 22 | 118 | 0.60713 | 0.33052 | 0.64691 | 0.34628 | 10 |
| msmarco | 397 | shared | 204 | 13 | 118 | 0.60713 | 0.47083 | 0.64668 | 0.48817 | 10 |
| msmarco | 397 | shared | 307 | 15 | 118 | 0.60713 | 0.55950 | 0.66095 | 0.57814 | 10 |
| msmarco | 397 | shared | 409 | 18 | 118 | 0.60713 | 0.61029 | 0.65830 | 0.63277 | 10 |
| msmarco | 397 | shared | 512 | 16 | 118 | 0.60713 | 0.63847 | 0.66168 | 0.65931 | 10 |
| msmarco | 397 | shared | 614 | 15 | 118 | 0.60713 | 0.65241 | 0.66168 | 0.67532 | 10 |
| msmarco | 397 | shared | 716 | 19 | 118 | 0.60713 | 0.66158 | 0.66480 | 0.68212 | 10 |
| msmarco | 397 | shared | 819 | 23 | 118 | 0.60713 | 0.65857 | 0.67282 | 0.68655 | 10 |
| msmarco | 397 | shared | 921 | 45 | 118 | 0.60713 | 0.64417 | 0.67654 | 0.68889 | 10 |
| msmarco | 397 | 1-1 | 102 | 313 | 118 | 0.60713 | 0.28323 | 0.64610 | 0.34597 | 10 |
| msmarco | 397 | 1-1 | 204 | 171 | 118 | 0.60713 | 0.42397 | 0.64784 | 0.49238 | 10 |
| msmarco | 397 | 1-1 | 307 | 184 | 118 | 0.60713 | 0.49503 | 0.64619 | 0.58090 | 10 |
| msmarco | 397 | 1-1 | 409 | 169 | 118 | 0.60713 | 0.54642 | 0.65425 | 0.63514 | 10 |
| msmarco | 397 | 1-1 | 512 | 161 | 118 | 0.60713 | 0.57375 | 0.65809 | 0.66603 | 10 |
| msmarco | 397 | 1-1 | 614 | 167 | 118 | 0.60713 | 0.58686 | 0.65809 | 0.68132 | 10 |
| msmarco | 397 | 1-1 | 716 | 168 | 118 | 0.60713 | 0.59263 | 0.65809 | 0.68638 | 10 |
| msmarco | 397 | 1-1 | 819 | 158 | 118 | 0.60713 | 0.59933 | 0.66168 | 0.68878 | 10 |
| msmarco | 397 | 1-1 | 921 | 164 | 118 | 0.60713 | 0.60101 | 0.66480 | 0.68515 | 10 |
| msmarco | 437 | shared | 102 | 56 | 118 | 0.79225 | 0.38776 | 0.85970 | 0.40581 | 10 |
| msmarco | 437 | shared | 204 | 69 | 118 | 0.79225 | 0.54635 | 0.86796 | 0.57906 | 10 |
| msmarco | 437 | shared | 307 | 67 | 118 | 0.79225 | 0.64861 | 0.86880 | 0.69298 | 10 |
| msmarco | 437 | shared | 409 | 74 | 118 | 0.79225 | 0.71742 | 0.86903 | 0.76864 | 10 |
| msmarco | 437 | shared | 512 | 59 | 118 | 0.79225 | 0.76459 | 0.87230 | 0.81360 | 10 |
| msmarco | 437 | shared | 614 | 70 | 148 | 0.77569 | 0.78499 | 0.87230 | 0.84319 | 10 |
| msmarco | 437 | shared | 716 | 92 | 148 | 0.77569 | 0.79307 | 0.87334 | 0.85783 | 10 |
| msmarco | 437 | shared | 819 | 100 | 148 | 0.77569 | 0.79527 | 0.87334 | 0.86684 | 10 |
| msmarco | 437 | shared | 921 | 116 | 148 | 0.77569 | 0.79284 | 0.87334 | 0.86988 | 10 |
| msmarco | 437 | 1-1 | 102 | 80 | 51 | 0.82097 | 0.37265 | 0.84867 | 0.39583 | 10 |
| msmarco | 437 | 1-1 | 204 | 68 | 51 | 0.82097 | 0.52695 | 0.82389 | 0.55981 | 10 |
| msmarco | 437 | 1-1 | 307 | 69 | 51 | 0.82097 | 0.62532 | 0.85222 | 0.66633 | 10 |
| msmarco | 437 | 1-1 | 409 | 68 | 51 | 0.82097 | 0.69706 | 0.86489 | 0.74269 | 10 |
| msmarco | 437 | 1-1 | 512 | 69 | 51 | 0.82097 | 0.74582 | 0.86916 | 0.79450 | 10 |
| msmarco | 437 | 1-1 | 614 | 73 | 51 | 0.82097 | 0.77261 | 0.87018 | 0.82531 | 10 |
| msmarco | 437 | 1-1 | 716 | 72 | 51 | 0.82097 | 0.78829 | 0.87018 | 0.84241 | 10 |
| msmarco | 437 | 1-1 | 819 | 68 | 51 | 0.82097 | 0.79415 | 0.87018 | 0.84911 | 10 |
| msmarco | 437 | 1-1 | 921 | 60 | 51 | 0.82097 | 0.80048 | 0.87340 | 0.85448 | 10 |
| msmarco | 684 | shared | 102 | 46 | 124 | 0.72854 | 0.35607 | 0.82204 | 0.37646 | 10 |
| msmarco | 684 | shared | 204 | 39 | 124 | 0.72854 | 0.52330 | 0.81878 | 0.54550 | 10 |
| msmarco | 684 | shared | 307 | 39 | 124 | 0.72854 | 0.62330 | 0.82462 | 0.65690 | 10 |
| msmarco | 684 | shared | 409 | 42 | 124 | 0.72854 | 0.68896 | 0.84074 | 0.73449 | 10 |
| msmarco | 684 | shared | 512 | 45 | 167 | 0.71661 | 0.72585 | 0.84389 | 0.77652 | 10 |
| msmarco | 684 | shared | 614 | 55 | 167 | 0.71661 | 0.74204 | 0.84389 | 0.80288 | 10 |
| msmarco | 684 | shared | 716 | 57 | 167 | 0.71661 | 0.75163 | 0.84389 | 0.81917 | 10 |
| msmarco | 684 | shared | 819 | 61 | 167 | 0.71661 | 0.75356 | 0.84389 | 0.82270 | 10 |
| msmarco | 684 | shared | 921 | 64 | 167 | 0.71661 | 0.74975 | 0.84389 | 0.82885 | 10 |
| msmarco | 684 | 1-1 | 102 | 185 | 167 | 0.71661 | 0.31746 | 0.78081 | 0.37166 | 10 |
| msmarco | 684 | 1-1 | 204 | 139 | 167 | 0.71661 | 0.47684 | 0.79578 | 0.53920 | 10 |
| msmarco | 684 | 1-1 | 307 | 99 | 167 | 0.71661 | 0.58211 | 0.80596 | 0.64634 | 10 |
| msmarco | 684 | 1-1 | 409 | 109 | 167 | 0.71661 | 0.63958 | 0.79879 | 0.71481 | 10 |
| msmarco | 684 | 1-1 | 512 | 136 | 167 | 0.71661 | 0.68171 | 0.84389 | 0.76617 | 10 |
| msmarco | 684 | 1-1 | 614 | 126 | 167 | 0.71661 | 0.70369 | 0.84264 | 0.79424 | 10 |
| msmarco | 684 | 1-1 | 716 | 127 | 167 | 0.71661 | 0.71272 | 0.84415 | 0.80849 | 10 |
| msmarco | 684 | 1-1 | 819 | 132 | 167 | 0.71661 | 0.71600 | 0.84415 | 0.81502 | 10 |
| msmarco | 684 | 1-1 | 921 | 133 | 167 | 0.71661 | 0.72132 | 0.84415 | 0.82538 | 10 |
| msmarco | 771 | shared | 102 | 2 | 9 | 0.76720 | 0.36764 | 0.78007 | 0.35560 | 10 |
| msmarco | 771 | shared | 204 | 4 | 7 | 0.77126 | 0.53622 | 0.78501 | 0.52163 | 10 |
| msmarco | 771 | shared | 307 | 7 | 9 | 0.76720 | 0.64313 | 0.78522 | 0.63206 | 10 |
| msmarco | 771 | shared | 409 | 6 | 9 | 0.76720 | 0.71295 | 0.78344 | 0.69989 | 10 |
| msmarco | 771 | shared | 512 | 7 | 9 | 0.76720 | 0.75627 | 0.78538 | 0.74297 | 10 |
| msmarco | 771 | shared | 614 | 7 | 9 | 0.76720 | 0.77653 | 0.78825 | 0.76662 | 10 |
| msmarco | 771 | shared | 716 | 7 | 9 | 0.76720 | 0.78973 | 0.78825 | 0.77889 | 10 |
| msmarco | 771 | shared | 819 | 7 | 9 | 0.76720 | 0.79326 | 0.78825 | 0.78396 | 10 |
| msmarco | 771 | shared | 921 | 7 | 9 | 0.76720 | 0.79072 | 0.79019 | 0.78421 | 10 |
| msmarco | 771 | 1-1 | 102 | 11 | 7 | 0.77126 | 0.34299 | 0.77724 | 0.35190 | 10 |
| msmarco | 771 | 1-1 | 204 | 9 | 7 | 0.77126 | 0.49933 | 0.78149 | 0.51204 | 10 |
| msmarco | 771 | 1-1 | 307 | 9 | 7 | 0.77126 | 0.60287 | 0.78501 | 0.61751 | 10 |
| msmarco | 771 | 1-1 | 409 | 9 | 7 | 0.77126 | 0.67765 | 0.78377 | 0.68956 | 10 |
| msmarco | 771 | 1-1 | 512 | 9 | 7 | 0.77126 | 0.72010 | 0.78344 | 0.73382 | 10 |
| msmarco | 771 | 1-1 | 614 | 9 | 7 | 0.77126 | 0.74976 | 0.78517 | 0.76357 | 10 |
| msmarco | 771 | 1-1 | 716 | 9 | 7 | 0.77126 | 0.75946 | 0.78538 | 0.77624 | 10 |
| msmarco | 771 | 1-1 | 819 | 8 | 7 | 0.77126 | 0.76330 | 0.78627 | 0.78078 | 10 |
| msmarco | 771 | 1-1 | 921 | 7 | 7 | 0.77126 | 0.76940 | 0.78627 | 0.78431 | 10 |
| msmarco | 981 | shared | 102 | 6 | 16 | 0.74452 | 0.37245 | 0.75300 | 0.37082 | 10 |
| msmarco | 981 | shared | 204 | 8 | 65 | 0.66866 | 0.51689 | 0.75067 | 0.52372 | 10 |
| msmarco | 981 | shared | 307 | 17 | 65 | 0.66866 | 0.59827 | 0.76780 | 0.62060 | 10 |
| msmarco | 981 | shared | 409 | 23 | 65 | 0.66866 | 0.64595 | 0.76814 | 0.68082 | 10 |
| msmarco | 981 | shared | 512 | 24 | 65 | 0.66866 | 0.67557 | 0.77202 | 0.71895 | 10 |
| msmarco | 981 | shared | 614 | 28 | 65 | 0.66866 | 0.68942 | 0.77176 | 0.74505 | 10 |
| msmarco | 981 | shared | 716 | 30 | 65 | 0.66866 | 0.69758 | 0.77176 | 0.75476 | 10 |
| msmarco | 981 | shared | 819 | 31 | 65 | 0.66866 | 0.69872 | 0.77176 | 0.76030 | 10 |
| msmarco | 981 | shared | 921 | 41 | 65 | 0.66866 | 0.69494 | 0.77185 | 0.76656 | 10 |
| msmarco | 981 | 1-1 | 102 | 22 | 16 | 0.74452 | 0.33927 | 0.73425 | 0.36375 | 10 |
| msmarco | 981 | 1-1 | 204 | 18 | 16 | 0.74452 | 0.47152 | 0.75059 | 0.50097 | 10 |
| msmarco | 981 | 1-1 | 307 | 19 | 16 | 0.74452 | 0.55870 | 0.75926 | 0.59394 | 10 |
| msmarco | 981 | 1-1 | 409 | 18 | 16 | 0.74452 | 0.61596 | 0.76188 | 0.65334 | 10 |
| msmarco | 981 | 1-1 | 512 | 20 | 16 | 0.74452 | 0.65018 | 0.76307 | 0.69169 | 10 |
| msmarco | 981 | 1-1 | 614 | 20 | 16 | 0.74452 | 0.67738 | 0.76307 | 0.71772 | 10 |
| msmarco | 981 | 1-1 | 716 | 18 | 16 | 0.74452 | 0.69721 | 0.76188 | 0.72987 | 10 |
| msmarco | 981 | 1-1 | 819 | 19 | 16 | 0.74452 | 0.70338 | 0.76307 | 0.73555 | 10 |
| msmarco | 981 | 1-1 | 921 | 17 | 16 | 0.74452 | 0.71558 | 0.77160 | 0.74677 | 10 |
| msmarco | 1073 | shared | 102 | 5 | 5 | 0.61750 | 0.31707 | 0.56850 | 0.28976 | 10 |
| msmarco | 1073 | shared | 204 | 5 | 26 | 0.51458 | 0.44983 | 0.57361 | 0.41988 | 10 |
| msmarco | 1073 | shared | 307 | 6 | 26 | 0.51458 | 0.51112 | 0.57615 | 0.48820 | 10 |
| msmarco | 1073 | shared | 409 | 6 | 26 | 0.51458 | 0.54563 | 0.57615 | 0.53596 | 10 |
| msmarco | 1073 | shared | 512 | 6 | 26 | 0.51458 | 0.56058 | 0.57729 | 0.56400 | 10 |
| msmarco | 1073 | shared | 614 | 8 | 26 | 0.51458 | 0.56305 | 0.57798 | 0.57978 | 10 |
| msmarco | 1073 | shared | 716 | 8 | 26 | 0.51458 | 0.56215 | 0.57798 | 0.58654 | 10 |
| msmarco | 1073 | shared | 819 | 8 | 26 | 0.51458 | 0.55933 | 0.57798 | 0.59048 | 10 |
| msmarco | 1073 | shared | 921 | 11 | 26 | 0.51458 | 0.54968 | 0.57950 | 0.58764 | 10 |
| msmarco | 1073 | 1-1 | 102 | 12 | 5 | 0.61750 | 0.26321 | 0.55156 | 0.27979 | 10 |
| msmarco | 1073 | 1-1 | 204 | 7 | 5 | 0.61750 | 0.38583 | 0.55130 | 0.39356 | 10 |
| msmarco | 1073 | 1-1 | 307 | 6 | 5 | 0.61750 | 0.45009 | 0.56044 | 0.45014 | 10 |
| msmarco | 1073 | 1-1 | 409 | 6 | 5 | 0.61750 | 0.49773 | 0.56012 | 0.49581 | 10 |
| msmarco | 1073 | 1-1 | 512 | 6 | 5 | 0.61750 | 0.52532 | 0.56234 | 0.52041 | 10 |
| msmarco | 1073 | 1-1 | 614 | 5 | 5 | 0.61750 | 0.53985 | 0.56859 | 0.52858 | 10 |
| msmarco | 1073 | 1-1 | 716 | 5 | 5 | 0.61750 | 0.54510 | 0.56859 | 0.53339 | 10 |
| msmarco | 1073 | 1-1 | 819 | 6 | 5 | 0.61750 | 0.55680 | 0.57133 | 0.54188 | 10 |
| msmarco | 1073 | 1-1 | 921 | 6 | 5 | 0.61750 | 0.57230 | 0.57798 | 0.55624 | 10 |
| msmarco | 1108 | shared | 102 | 68 | 284 | 0.47726 | 0.25890 | 0.57352 | 0.28437 | 10 |
| msmarco | 1108 | shared | 204 | 15 | 284 | 0.47726 | 0.37727 | 0.57370 | 0.39747 | 10 |
| msmarco | 1108 | shared | 307 | 29 | 284 | 0.47726 | 0.44427 | 0.57589 | 0.47486 | 10 |
| msmarco | 1108 | shared | 409 | 39 | 284 | 0.47726 | 0.48342 | 0.57510 | 0.52143 | 10 |
| msmarco | 1108 | shared | 512 | 33 | 284 | 0.47726 | 0.51321 | 0.57370 | 0.54868 | 10 |
| msmarco | 1108 | shared | 614 | 38 | 284 | 0.47726 | 0.52645 | 0.57887 | 0.56743 | 10 |
| msmarco | 1108 | shared | 716 | 41 | 284 | 0.47726 | 0.53113 | 0.58105 | 0.57372 | 10 |
| msmarco | 1108 | shared | 819 | 45 | 284 | 0.47726 | 0.53004 | 0.58105 | 0.57949 | 10 |
| msmarco | 1108 | shared | 921 | 71 | 284 | 0.47726 | 0.52394 | 0.59047 | 0.58089 | 10 |
| msmarco | 1108 | 1-1 | 102 | 3 | 159 | 0.49577 | 0.27117 | 0.54807 | 0.26573 | 10 |
| msmarco | 1108 | 1-1 | 204 | 8 | 159 | 0.49577 | 0.37413 | 0.56400 | 0.37805 | 10 |
| msmarco | 1108 | 1-1 | 307 | 8 | 159 | 0.49577 | 0.43867 | 0.57213 | 0.44098 | 10 |
| msmarco | 1108 | 1-1 | 409 | 7 | 159 | 0.49577 | 0.48574 | 0.57039 | 0.48858 | 10 |
| msmarco | 1108 | 1-1 | 512 | 8 | 159 | 0.49577 | 0.50197 | 0.57831 | 0.51003 | 10 |
| msmarco | 1108 | 1-1 | 614 | 13 | 159 | 0.49577 | 0.51858 | 0.57831 | 0.53530 | 10 |
| msmarco | 1108 | 1-1 | 716 | 14 | 159 | 0.49577 | 0.51992 | 0.57831 | 0.54037 | 10 |
| msmarco | 1108 | 1-1 | 819 | 29 | 159 | 0.49577 | 0.51776 | 0.58196 | 0.55528 | 10 |
| msmarco | 1108 | 1-1 | 921 | 54 | 159 | 0.49577 | 0.51383 | 0.58710 | 0.56793 | 10 |
| msmarco | 1153 | shared | 102 | 2 | 1 | 0.73813 | 0.36636 | 0.62455 | 0.30529 | 10 |
| msmarco | 1153 | shared | 204 | 2 | 1 | 0.73813 | 0.52860 | 0.64191 | 0.44549 | 10 |
| msmarco | 1153 | shared | 307 | 2 | 1 | 0.73813 | 0.63495 | 0.64500 | 0.53122 | 10 |
| msmarco | 1153 | shared | 409 | 2 | 2 | 0.73182 | 0.68831 | 0.64500 | 0.58122 | 10 |
| msmarco | 1153 | shared | 512 | 2 | 2 | 0.73182 | 0.72707 | 0.64500 | 0.61511 | 10 |
| msmarco | 1153 | shared | 614 | 2 | 2 | 0.73182 | 0.74501 | 0.64500 | 0.63305 | 10 |
| msmarco | 1153 | shared | 716 | 2 | 2 | 0.73182 | 0.75231 | 0.64500 | 0.64313 | 10 |
| msmarco | 1153 | shared | 819 | 2 | 2 | 0.73182 | 0.75365 | 0.64500 | 0.64395 | 10 |
| msmarco | 1153 | shared | 921 | 2 | 2 | 0.73182 | 0.75097 | 0.64500 | 0.64321 | 10 |
| msmarco | 1153 | 1-1 | 102 | 2 | 1 | 0.73813 | 0.32209 | 0.62673 | 0.29307 | 10 |
| msmarco | 1153 | 1-1 | 204 | 2 | 1 | 0.73813 | 0.48142 | 0.63745 | 0.42786 | 10 |
| msmarco | 1153 | 1-1 | 307 | 2 | 1 | 0.73813 | 0.56962 | 0.63767 | 0.51084 | 10 |
| msmarco | 1153 | 1-1 | 409 | 2 | 1 | 0.73813 | 0.62332 | 0.64500 | 0.55681 | 10 |
| msmarco | 1153 | 1-1 | 512 | 2 | 1 | 0.73813 | 0.66433 | 0.64500 | 0.59371 | 10 |
| msmarco | 1153 | 1-1 | 614 | 2 | 1 | 0.73813 | 0.68469 | 0.64500 | 0.61175 | 10 |
| msmarco | 1153 | 1-1 | 716 | 2 | 1 | 0.73813 | 0.69660 | 0.64500 | 0.62080 | 10 |
| msmarco | 1153 | 1-1 | 819 | 2 | 1 | 0.73813 | 0.70021 | 0.64500 | 0.62464 | 10 |
| msmarco | 1153 | 1-1 | 921 | 2 | 1 | 0.73813 | 0.71334 | 0.64500 | 0.63305 | 10 |
| msmarco | 1797 | shared | 102 | 4 | 3 | 0.69655 | 0.35279 | 0.61208 | 0.30740 | 10 |
| msmarco | 1797 | shared | 204 | 4 | 3 | 0.69655 | 0.50788 | 0.61208 | 0.44065 | 10 |
| msmarco | 1797 | shared | 307 | 4 | 3 | 0.69655 | 0.59644 | 0.62178 | 0.51791 | 10 |
| msmarco | 1797 | shared | 409 | 4 | 3 | 0.69655 | 0.65611 | 0.62178 | 0.57105 | 10 |
| msmarco | 1797 | shared | 512 | 4 | 4 | 0.68103 | 0.69255 | 0.62659 | 0.60795 | 10 |
| msmarco | 1797 | shared | 614 | 4 | 4 | 0.68103 | 0.71311 | 0.62659 | 0.62656 | 10 |
| msmarco | 1797 | shared | 716 | 4 | 4 | 0.68103 | 0.72305 | 0.62659 | 0.63732 | 10 |
| msmarco | 1797 | shared | 819 | 4 | 4 | 0.68103 | 0.72283 | 0.62659 | 0.63998 | 10 |
| msmarco | 1797 | shared | 921 | 4 | 4 | 0.68103 | 0.71568 | 0.63178 | 0.64317 | 10 |
| msmarco | 1797 | 1-1 | 102 | 5 | 3 | 0.69655 | 0.32714 | 0.61208 | 0.30370 | 10 |
| msmarco | 1797 | 1-1 | 204 | 5 | 3 | 0.69655 | 0.45644 | 0.62178 | 0.42820 | 10 |
| msmarco | 1797 | 1-1 | 307 | 4 | 3 | 0.69655 | 0.54450 | 0.62178 | 0.50542 | 10 |
| msmarco | 1797 | 1-1 | 409 | 4 | 3 | 0.69655 | 0.59433 | 0.62178 | 0.55461 | 10 |
| msmarco | 1797 | 1-1 | 512 | 4 | 3 | 0.69655 | 0.63732 | 0.62178 | 0.58950 | 10 |
| msmarco | 1797 | 1-1 | 614 | 4 | 3 | 0.69655 | 0.66785 | 0.62359 | 0.61321 | 10 |
| msmarco | 1797 | 1-1 | 716 | 4 | 3 | 0.69655 | 0.68448 | 0.62659 | 0.62429 | 10 |
| msmarco | 1797 | 1-1 | 819 | 4 | 3 | 0.69655 | 0.68752 | 0.62659 | 0.62859 | 10 |
| msmarco | 1797 | 1-1 | 921 | 4 | 3 | 0.69655 | 0.69154 | 0.63178 | 0.62993 | 10 |

## Coverage, checks, and cost

| environment | training queries | with a positive | eligible (2+ positives) | ranked | capped | documents ranked | positives | orderings | masks | queries sharing an id with a document | positives out of the competition | ties met by tracked positives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact | 728 | 728 | 69 | 69 | False | 5183 | 163 | 630 | 5118 | 0 | 0 | 0 |
| nfcorpus | 2331 | 2331 | 2162 | 2162 | False | 3633 | 100664 | 164508 | 1463276 | 0 | 0 | 953742 |
| msmarco | 45000 | 45000 | 2321 | 2321 | False | 8841823 | 5054 | 18575 | 148607 | 2321 | 0 | 0 |

| environment | stored regrets compared | largest difference from the shared run | source |
|---|---|---|---|
| scifact | 1296 | 0.00000001 | 20261107T025534Z-d8bad5ff |
| nfcorpus | 85311 | 0.00000001 | 20261107T025534Z-d8bad5ff |
| msmarco | 42507 | 0.00000001 | 20261107T025534Z-d8bad5ff |

| environment | device | seconds | masks | documents | mask-documents per second | plan | thresholds | read | scan | metrics | write | peak device MiB | peak RSS MiB | artifact MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact | NVIDIA GeForce RTX 4090 | 0.9 | 5118 | 5183 | 203M | 0.2 | 0.0 | 0.0 | 0.1 | 0.4 | 0.0 | 115 | 1371 | 1.6 |
| nfcorpus | NVIDIA GeForce RTX 4090 | 194.2 | 1463276 | 3633 | 334M | 4.6 | 64.8 | 0.0 | 15.9 | 106.8 | 0.6 | 794 | 9463 | 971.4 |
| msmarco | NVIDIA GeForce RTX 4090 | 2727.3 | 148607 | 8841823 | 488M | 1.9 | 1.0 | 2.2 | 2690.2 | 11.6 | 0.0 | 559 | 27014 | 48.2 |
