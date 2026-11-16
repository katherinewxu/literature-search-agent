# Judge calibration under covariate shift

Eligible share of never-screened (abstract-bearing) records the LLM agents included, with the
judge's error rates corrected for differences between calibration and target records.
Similarity = within-review percentile of BM25(review question + criteria, title + abstract);
bins at 0.25, 0.5, 0.75. CI: cluster bootstrap over reviews.

## gemini-3.8-flash

Where the records sit (share of each class per similarity bin) and the judge's yes-rate:

| similarity bin | gold n | gold yes | excluded n | excluded yes | unscreened n | unscreened yes |
|---|---|---|---|---|---|---|
| < .25 | 59 | 0.85 | 145 | 0.17 | 406 | 0.38 |
| .25-.50 | 64 | 0.84 | 133 | 0.20 | 406 | 0.45 |
| .50-.75 | 92 | 0.85 | 129 | 0.28 | 394 | 0.49 |
| >= .75 | 115 | 0.89 | 117 | 0.32 | 385 | 0.46 |

32 reviews; 1591 unscreened records.

| estimate | eligible share | 95% CI | P(eligible \| judged eligible) | 95% CI |
|---|---|---|---|---|
| naive (pooled calibration) | 0.33 | [0.21, 0.41] | 0.64 | [0.46, 0.74] |
| similarity-stratified | 0.33 | [0.20, 0.41] | 0.64 | [0.44, 0.73] |
| review-reweighted | 0.07 | [0.00, 0.23] | 0.13 | [0.00, 0.45] |
| both | 0.11 | [0.05, 0.25] | 0.22 | [0.10, 0.47] |
| per-review, shrunk (k=10) | 0.26 | [0.18, 0.33] | 0.53 | [0.39, 0.65] |

## gpt-5.6-luna

Where the records sit (share of each class per similarity bin) and the judge's yes-rate:

| similarity bin | gold n | gold yes | excluded n | excluded yes | unscreened n | unscreened yes |
|---|---|---|---|---|---|---|
| < .25 | 59 | 0.81 | 145 | 0.18 | 406 | 0.38 |
| .25-.50 | 64 | 0.89 | 133 | 0.29 | 406 | 0.48 |
| .50-.75 | 92 | 0.87 | 129 | 0.31 | 394 | 0.48 |
| >= .75 | 115 | 0.88 | 117 | 0.36 | 385 | 0.47 |

32 reviews; 1591 unscreened records.

| estimate | eligible share | 95% CI | P(eligible \| judged eligible) | 95% CI |
|---|---|---|---|---|
| naive (pooled calibration) | 0.29 | [0.14, 0.39] | 0.56 | [0.31, 0.68] |
| similarity-stratified | 0.29 | [0.13, 0.39] | 0.55 | [0.29, 0.67] |
| review-reweighted | 0.00 | [0.00, 0.13] | 0.00 | [0.00, 0.25] |
| both | 0.02 | [0.00, 0.16] | 0.03 | [0.00, 0.28] |
| per-review, shrunk (k=10) | 0.24 | [0.15, 0.32] | 0.48 | [0.32, 0.60] |

Reading: if the corrected shares sit inside the naive CI, the observable shift does not change
the conclusion. The beyond-search counts in `web_robust.md` section 4 scale with
P(eligible | judged eligible), so multiply them by (corrected PPV / naive PPV).
Remaining assumption: given similarity and review, the judge errs on never-screened records
as it does on screened ones. Only human labels on never-screened records can test that.
