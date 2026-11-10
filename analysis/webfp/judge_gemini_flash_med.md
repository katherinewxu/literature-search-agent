# LLM judge on live-web inclusions (google/gemini-3.8-flash)

Calibration and pool use records with an abstract only.

| quantity | value | n |
|---|---|---|
| sensitivity (gold judged eligible) | 0.91 | 22 |
| specificity (screened-excluded judged ineligible) | 0.68 | 41 |
| unscreened judged eligible (raw) | 0.33 | 117 |
| unscreened truly eligible (Rogan-Gladen) | 0.03 | |
| P(eligible \| judged eligible), unscreened | 0.07 | |

- gold, with abstract: 0.91 judged eligible (n=22)
- gold, without abstract: 1.00 judged eligible (n=5)
- excluded, with abstract: 0.32 judged eligible (n=41)
- excluded, without abstract: 0.75 judged eligible (n=4)
- unscreened, with abstract: 0.33 judged eligible (n=117)
- unscreened, without abstract: 0.42 judged eligible (n=12)

Specificity is measured on hard negatives (records the review's own search retrieved), so it is
probably lower than on unscreened records, which makes the Rogan-Gladen share conservative.

## eval (8 reviews)

Pool: 3 judged-eligible unscreened works, weighted by P(eligible|judged)=0.07.

| agent | mode | B | recall (gold) | recall (gold + pool) | precision (gold) | precision (adjusted) |
|---|---|---|---|---|---|---|
| haiku-5.5 | citation | 10 | 0.050 | 0.050 | 0.120 | 0.120 |
| haiku-5.5 | citation | 20 | 0.087 | 0.087 | 0.161 | 0.161 |
| haiku-5.5 | citation | 40 | 0.108 | 0.108 | 0.161 | 0.161 |
| haiku-5.5 | hybrid | 10 | 0.141 | 0.141 | 0.177 | 0.177 |
| haiku-5.5 | hybrid | 20 | 0.165 | 0.165 | 0.191 | 0.191 |
| haiku-5.5 | hybrid | 40 | 0.171 | 0.171 | 0.148 | 0.148 |
| openalex-search | keyword | 10 | 0.010 | 0.010 | 0.003 | 0.003 |
| openalex-search | keyword | 20 | 0.010 | 0.010 | 0.002 | 0.002 |
| openalex-search | keyword | 40 | 0.010 | 0.010 | 0.002 | 0.002 |
| sonnet-5.5 | citation | 40 | 0.148 | 0.148 | 0.180 | 0.180 |
| sonnet-5.5 | hybrid | 10 | 0.215 | 0.215 | 0.428 | 0.428 |
| sonnet-5.5 | hybrid | 20 | 0.295 | 0.296 | 0.389 | 0.389 |
| sonnet-5.5 | hybrid | 40 | 0.324 | 0.324 | 0.349 | 0.349 |

## transfer (24 reviews)

Pool: 36 judged-eligible unscreened works, weighted by P(eligible|judged)=0.07.

| agent | mode | B | recall (gold) | recall (gold + pool) | precision (gold) | precision (adjusted) |
|---|---|---|---|---|---|---|
| haiku-5.5 | hybrid | 20 | 0.084 | 0.085 | 0.152 | 0.153 |
| haiku-5.5 | hybrid | 40 | 0.098 | 0.099 | 0.124 | 0.125 |
| openalex-search | keyword | 20 | 0.018 | 0.018 | 0.004 | 0.005 |
| openalex-search | keyword | 40 | 0.018 | 0.018 | 0.004 | 0.005 |

