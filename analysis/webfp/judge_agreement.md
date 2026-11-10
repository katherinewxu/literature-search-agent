# Two-judge agreement and circularity checks

A = judgements_gemini-3.8-flash.jsonl, B = judgements_gpt-5.6-luna.jsonl; 2445 records with an abstract judged by both.

| judge | sensitivity | specificity | unscreened yes | eligible share (RG) | P(elig \| yes) |
|---|---|---|---|---|---|
| A | 0.86 | 0.76 | 0.45 | 0.33 | 0.64 |
| B | 0.87 | 0.72 | 0.45 | 0.29 | 0.56 |
| A and B | 0.82 | 0.78 | 0.40 | 0.30 | 0.62 |

Cohen's kappa A vs B: all records 0.80; unscreened only 0.79.

## eval (8 reviews)

Recall of LLM agents: gold only, then gold + pool for each pool definition.

| agent | mode | B | gold | all (A) | loao (A) | all (B) | both |
|---|---|---|---|---|---|---|---|
| haiku-5.5 | citation | 10 | 0.050 | 0.055 | 0.056 | 0.053 | 0.055 |
| haiku-5.5 | citation | 20 | 0.087 | 0.076 | 0.077 | 0.076 | 0.079 |
| haiku-5.5 | citation | 40 | 0.108 | 0.094 | 0.095 | 0.093 | 0.096 |
| haiku-5.5 | hybrid | 10 | 0.141 | 0.158 | 0.153 | 0.152 | 0.151 |
| haiku-5.5 | hybrid | 20 | 0.165 | 0.183 | 0.182 | 0.183 | 0.179 |
| haiku-5.5 | hybrid | 40 | 0.171 | 0.181 | 0.179 | 0.178 | 0.177 |
| sonnet-5.5 | citation | 40 | 0.148 | 0.150 | 0.142 | 0.146 | 0.153 |
| sonnet-5.5 | hybrid | 10 | 0.215 | 0.221 | 0.229 | 0.222 | 0.217 |
| sonnet-5.5 | hybrid | 20 | 0.295 | 0.312 | 0.306 | 0.307 | 0.308 |
| sonnet-5.5 | hybrid | 40 | 0.324 | 0.354 | 0.343 | 0.351 | 0.349 |

## transfer (24 reviews)

Recall of LLM agents: gold only, then gold + pool for each pool definition.

| agent | mode | B | gold | all (A) | loao (A) | all (B) | both |
|---|---|---|---|---|---|---|---|
| haiku-5.5 | hybrid | 20 | 0.084 | 0.166 | 0.097 | 0.160 | 0.154 |
| haiku-5.5 | hybrid | 40 | 0.098 | 0.220 | 0.113 | 0.202 | 0.201 |

