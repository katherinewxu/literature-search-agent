# Counterfactual: relevance-ranked citation expansion

Every expand the LLM web agents made is replayed with the same paper and k; columns count gold
studies in the returned slices (union per run, summed over runs), and recall if each were included.

## eval (8 reviews)

| agent | mode | B | runs | expands/run | gold in slices: actual | sim | bm25 | recall: actual | upper bound actual | upper bound bm25 |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku-5.5 | citation | 10 | 8 | 0.9 | 6 | 6 | 7 | 0.050 | 0.050 | 0.052 |
| haiku-5.5 | citation | 20 | 8 | 0.8 | 12 | 13 | 13 | 0.087 | 0.112 | 0.121 |
| haiku-5.5 | citation | 40 | 8 | 0.8 | 14 | 21 | 21 | 0.108 | 0.117 | 0.171 |
| haiku-5.5 | hybrid | 10 | 8 | 0.5 | 0 | 0 | 1 | 0.141 | 0.141 | 0.154 |
| haiku-5.5 | hybrid | 20 | 8 | 0.5 | 3 | 11 | 11 | 0.165 | 0.165 | 0.173 |
| haiku-5.5 | hybrid | 40 | 8 | 1.0 | 5 | 6 | 4 | 0.171 | 0.176 | 0.176 |
| sonnet-5.5 | citation | 40 | 8 | 2.8 | 21 | 37 | 40 | 0.148 | 0.160 | 0.201 |
| sonnet-5.5 | hybrid | 10 | 8 | 0.2 | 1 | 22 | 20 | 0.215 | 0.215 | 0.247 |
| sonnet-5.5 | hybrid | 20 | 8 | 0.1 | 1 | 19 | 18 | 0.295 | 0.295 | 0.326 |
| sonnet-5.5 | hybrid | 40 | 8 | 0.4 | 1 | 4 | 3 | 0.324 | 0.324 | 0.328 |

Share of a paper's neighbours present in the corpus graph (median over expands): 1.00

## transfer (24 reviews)

| agent | mode | B | runs | expands/run | gold in slices: actual | sim | bm25 | recall: actual | upper bound actual | upper bound bm25 |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku-5.5 | hybrid | 20 | 24 | 0.6 | 26 | 51 | 50 | 0.084 | 0.087 | 0.099 |
| haiku-5.5 | hybrid | 40 | 24 | 0.7 | 12 | 28 | 33 | 0.098 | 0.098 | 0.101 |

Share of a paper's neighbours present in the corpus graph (median over expands): 1.00

