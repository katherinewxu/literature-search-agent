# LLM judge on live-web inclusions (google/gemini-3.8-flash)

Calibration and pool use records with an abstract only.

| quantity | value | n |
|---|---|---|
| sensitivity (gold judged eligible) | 0.86 | 330 |
| specificity (screened-excluded judged ineligible) | 0.76 | 524 |
| unscreened judged eligible (raw) | 0.45 | 1591 |
| unscreened truly eligible (Rogan-Gladen) | 0.33 | |
| P(eligible \| judged eligible), unscreened | 0.64 | |

- gold, with abstract: 0.86 judged eligible (n=330)
- gold, without abstract: 0.96 judged eligible (n=49)
- excluded, with abstract: 0.24 judged eligible (n=524)
- excluded, without abstract: 0.54 judged eligible (n=48)
- unscreened, with abstract: 0.45 judged eligible (n=1591)
- unscreened, without abstract: 0.59 judged eligible (n=313)

Specificity is measured on hard negatives (records the review's own search retrieved), so it is
probably lower than on unscreened records, which makes the Rogan-Gladen share conservative.

## eval (8 reviews)

Pool: 144 judged-eligible unscreened works, weighted by P(eligible|judged)=0.64.

| agent | mode | B | recall (gold) | recall (gold + pool) | precision (gold) | precision (adjusted) |
|---|---|---|---|---|---|---|
| haiku-5.5 | citation | 10 | 0.050 | 0.055 | 0.120 | 0.169 |
| haiku-5.5 | citation | 20 | 0.087 | 0.076 | 0.161 | 0.177 |
| haiku-5.5 | citation | 40 | 0.108 | 0.094 | 0.161 | 0.175 |
| haiku-5.5 | hybrid | 10 | 0.141 | 0.158 | 0.177 | 0.316 |
| haiku-5.5 | hybrid | 20 | 0.165 | 0.183 | 0.191 | 0.278 |
| haiku-5.5 | hybrid | 40 | 0.171 | 0.181 | 0.148 | 0.210 |
| openalex-search | keyword | 10 | 0.010 | 0.011 | 0.003 | 0.006 |
| openalex-search | keyword | 20 | 0.010 | 0.011 | 0.002 | 0.004 |
| openalex-search | keyword | 40 | 0.010 | 0.011 | 0.002 | 0.004 |
| sonnet-5.5 | citation | 40 | 0.148 | 0.150 | 0.180 | 0.239 |
| sonnet-5.5 | hybrid | 10 | 0.215 | 0.221 | 0.428 | 0.515 |
| sonnet-5.5 | hybrid | 20 | 0.295 | 0.312 | 0.389 | 0.482 |
| sonnet-5.5 | hybrid | 40 | 0.324 | 0.354 | 0.349 | 0.435 |

## transfer (24 reviews)

Pool: 565 judged-eligible unscreened works, weighted by P(eligible|judged)=0.64.

| agent | mode | B | recall (gold) | recall (gold + pool) | precision (gold) | precision (adjusted) |
|---|---|---|---|---|---|---|
| haiku-5.5 | hybrid | 20 | 0.084 | 0.166 | 0.152 | 0.307 |
| haiku-5.5 | hybrid | 40 | 0.098 | 0.220 | 0.124 | 0.289 |
| openalex-search | keyword | 20 | 0.018 | 0.032 | 0.004 | 0.010 |
| openalex-search | keyword | 40 | 0.018 | 0.032 | 0.004 | 0.010 |

