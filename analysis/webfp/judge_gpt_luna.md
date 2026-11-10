# LLM judge on live-web inclusions (openai/gpt-5.6-luna)

Calibration and pool use records with an abstract only.

| quantity | value | n |
|---|---|---|
| sensitivity (gold judged eligible) | 0.87 | 330 |
| specificity (screened-excluded judged ineligible) | 0.72 | 524 |
| unscreened judged eligible (raw) | 0.45 | 1591 |
| unscreened truly eligible (Rogan-Gladen) | 0.29 | |
| P(eligible \| judged eligible), unscreened | 0.56 | |

- gold, with abstract: 0.87 judged eligible (n=330)
- gold, without abstract: 0.78 judged eligible (n=49)
- excluded, with abstract: 0.28 judged eligible (n=524)
- excluded, without abstract: 0.23 judged eligible (n=48)
- unscreened, with abstract: 0.45 judged eligible (n=1591)
- unscreened, without abstract: 0.34 judged eligible (n=313)

Specificity is measured on hard negatives (records the review's own search retrieved), so it is
probably lower than on unscreened records, which makes the Rogan-Gladen share conservative.

## eval (8 reviews)

Pool: 150 judged-eligible unscreened works, weighted by P(eligible|judged)=0.56.

| agent | mode | B | recall (gold) | recall (gold + pool) | precision (gold) | precision (adjusted) |
|---|---|---|---|---|---|---|
| haiku-5.5 | citation | 10 | 0.050 | 0.053 | 0.120 | 0.162 |
| haiku-5.5 | citation | 20 | 0.087 | 0.076 | 0.161 | 0.175 |
| haiku-5.5 | citation | 40 | 0.108 | 0.093 | 0.161 | 0.173 |
| haiku-5.5 | hybrid | 10 | 0.141 | 0.152 | 0.177 | 0.240 |
| haiku-5.5 | hybrid | 20 | 0.165 | 0.183 | 0.191 | 0.248 |
| haiku-5.5 | hybrid | 40 | 0.171 | 0.178 | 0.148 | 0.196 |
| openalex-search | keyword | 10 | 0.010 | 0.016 | 0.003 | 0.006 |
| openalex-search | keyword | 20 | 0.010 | 0.016 | 0.002 | 0.004 |
| openalex-search | keyword | 40 | 0.010 | 0.016 | 0.002 | 0.004 |
| sonnet-5.5 | citation | 40 | 0.148 | 0.146 | 0.180 | 0.232 |
| sonnet-5.5 | hybrid | 10 | 0.215 | 0.222 | 0.428 | 0.503 |
| sonnet-5.5 | hybrid | 20 | 0.295 | 0.307 | 0.389 | 0.466 |
| sonnet-5.5 | hybrid | 40 | 0.324 | 0.351 | 0.349 | 0.424 |

## transfer (24 reviews)

Pool: 568 judged-eligible unscreened works, weighted by P(eligible|judged)=0.56.

| agent | mode | B | recall (gold) | recall (gold + pool) | precision (gold) | precision (adjusted) |
|---|---|---|---|---|---|---|
| haiku-5.5 | hybrid | 20 | 0.084 | 0.160 | 0.152 | 0.291 |
| haiku-5.5 | hybrid | 40 | 0.098 | 0.202 | 0.124 | 0.259 |
| openalex-search | keyword | 20 | 0.018 | 0.030 | 0.004 | 0.009 |
| openalex-search | keyword | 40 | 0.018 | 0.030 | 0.004 | 0.009 |

