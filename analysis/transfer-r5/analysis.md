# Trace analysis

Runs: 216 from /scratch/users/kwx04/literature-search-agent/runs/ext-transfer-baselines-r3, /scratch/users/kwx04/literature-search-agent/runs/ext-transfer-api-r4

## Search vs. screening (mean [95% bootstrap CI] over episodes)

| system | B | search recall (gold exposed) | recall (included) | screen recall | precision | exposed | included |
|---|---|---|---|---|---|---|---|
| bm25/keyword | 20 | 0.321 [0.224, 0.429] | 0.321 [0.224, 0.429] | 1.000 | 0.087 | 195 | 195 |
| bm25/keyword | 40 | 0.474 [0.350, 0.599] | 0.474 [0.350, 0.599] | 1.000 | 0.071 | 392 | 392 |
| haiku-5.5/hybrid | 20 | 0.458 [0.354, 0.570] | 0.370 [0.277, 0.467] | 0.805 | 0.413 | 140 | 56 |
| haiku-5.5/hybrid | 40 | 0.595 [0.494, 0.700] | 0.477 [0.385, 0.574] | 0.792 | 0.390 | 260 | 88 |
| qwen3-32b/hybrid | 20 | 0.324 [0.223, 0.428] | 0.271 [0.176, 0.368] | 0.768 | 0.326 | 121 | 51 |
| qwen3-32b/hybrid | 40 | 0.424 [0.318, 0.535] | 0.351 [0.255, 0.455] | 0.818 | 0.302 | 232 | 83 |
| snowball-tuned/hybrid | 20 | 0.065 [0.033, 0.101] | 0.057 [0.029, 0.090] | 0.674 | 0.158 | 18 | 16 |
| snowball-tuned/hybrid | 40 | 0.106 [0.055, 0.166] | 0.095 [0.049, 0.149] | 0.679 | 0.163 | 32 | 27 |
| sonnet-5.5/hybrid | 40 | 0.697 [0.600, 0.789] | 0.561 [0.461, 0.659] | 0.777 | 0.441 | 351 | 90 |

## Paired per-review differences (system - reference; 95% bootstrap CI over reviews)

| system | reference | B | Δrecall | Δaurc | Δf1 | wins/losses (recall) |
|---|---|---|---|---|---|---|
| haiku-5.5/hybrid | bm25/keyword | 20 | +0.048 [-0.009, +0.103] | +0.054 [+0.018, +0.092] | +0.223 [+0.178, +0.272] | 16/6 of 24 |
| haiku-5.5/hybrid | bm25/keyword | 40 | +0.003 [-0.078, +0.077] | +0.055 [+0.002, +0.105] | +0.279 [+0.229, +0.336] | 14/10 of 24 |
| qwen3-32b/hybrid | bm25/keyword | 20 | -0.051 [-0.109, +0.003] | -0.016 [-0.050, +0.017] | +0.123 [+0.083, +0.166] | 8/14 of 24 |
| qwen3-32b/hybrid | bm25/keyword | 40 | -0.123 [-0.212, -0.040] | -0.052 [-0.106, +0.003] | +0.155 [+0.113, +0.201] | 9/15 of 24 |
| snowball-tuned/hybrid | bm25/keyword | 20 | -0.264 [-0.349, -0.186] | -0.156 [-0.209, -0.107] | -0.043 [-0.073, -0.009] | 0/24 of 24 |
| snowball-tuned/hybrid | bm25/keyword | 40 | -0.379 [-0.478, -0.279] | -0.244 [-0.314, -0.178] | -0.006 [-0.050, +0.042] | 0/24 of 24 |
| sonnet-5.5/hybrid | bm25/keyword | 40 | +0.087 [+0.016, +0.158] | +0.105 [+0.061, +0.152] | +0.337 [+0.284, +0.394] | 15/9 of 24 |

## Matched exposure: mean recall once N papers have been exposed

### B=20

| system | N=0 | N=10 | N=25 | N=50 | N=100 | N=200 | N=300 | N=400 |
|---|---|---|---|---|---|---|---|---|
| bm25/keyword | 0.000 | 0.025 | 0.088 | 0.132 | 0.210 | - | - | - |
| haiku-5.5/hybrid | 0.000 | 0.024 | 0.111 | 0.206 | 0.284 | - | - | - |
| qwen3-32b/hybrid | 0.000 | 0.021 | 0.102 | 0.159 | 0.289 | - | - | - |
| snowball-tuned/hybrid | 0.000 | 0.000 | 0.057 | - | - | - | - | - |

### B=40

| system | N=0 | N=10 | N=25 | N=50 | N=100 | N=200 | N=300 | N=400 |
|---|---|---|---|---|---|---|---|---|
| bm25/keyword | 0.000 | 0.025 | 0.088 | 0.132 | 0.210 | 0.321 | 0.414 | - |
| haiku-5.5/hybrid | 0.000 | 0.023 | 0.118 | 0.211 | 0.286 | 0.406 | 0.546 | - |
| qwen3-32b/hybrid | 0.000 | 0.021 | 0.098 | 0.141 | 0.229 | 0.364 | 0.361 | - |
| snowball-tuned/hybrid | 0.000 | 0.000 | 0.086 | - | - | - | - | - |
| sonnet-5.5/hybrid | 0.000 | 0.023 | 0.144 | 0.230 | 0.342 | 0.472 | 0.515 | - |

## Gold reachability from the 5 BM25 seeds (fraction of gold within h hops)

| graph | review | universe | gold | h=1 | h=2 | h=3 | h=4 | isolated gold | mean gold degree |
|---|---|---|---|---|---|---|---|---|---|
| closed | Abgaz_2023 | 733 | 33 | 0.48 | 0.97 | 1.00 | 1.00 | 0.00 | 16.9 |
| closed | Adamo_2021 | 4506 | 63 | 0.05 | 0.21 | 0.60 | 0.83 | 0.10 | 7.9 |
| closed | Ali_2024 | 2140 | 113 | 0.15 | 0.48 | 0.74 | 0.77 | 0.16 | 5.4 |
| closed | Anmarkrud_2021 | 2276 | 64 | 0.33 | 0.92 | 0.92 | 0.94 | 0.03 | 26.8 |
| closed | Attai_2022 | 3793 | 234 | 0.02 | 0.06 | 0.16 | 0.34 | 0.20 | 3.3 |
| closed | Deckers_2022 | 494 | 16 | 0.06 | 0.19 | 0.25 | 0.25 | 0.44 | 1.2 |
| closed | Filges_2018 | 6006 | 121 | 0.02 | 0.17 | 0.70 | 0.81 | 0.17 | 16.7 |
| closed | Filges_2022 | 4548 | 28 | 0.04 | 0.04 | 0.25 | 0.36 | 0.54 | 3.8 |
| closed | Harms_2024 | 1548 | 71 | 0.03 | 0.04 | 0.17 | 0.25 | 0.27 | 1.7 |
| closed | Kapuka_2021 | 427 | 27 | 0.00 | 0.00 | 0.00 | 0.00 | 0.41 | 0.9 |
| closed | Moran_2020 | 5242 | 142 | 0.06 | 0.22 | 0.51 | 0.74 | 0.13 | 5.5 |
| closed | Nobrega-Dos_Santos_2023 | 981 | 23 | 0.52 | 1.00 | 1.00 | 1.00 | 0.00 | 13.4 |
| closed | Noetel_2021 | 1077 | 29 | 0.03 | 0.28 | 0.76 | 0.79 | 0.14 | 6.8 |
| closed | Quevedo_2023 | 455 | 42 | 0.17 | 0.45 | 0.76 | 0.81 | 0.17 | 3.4 |
| closed | Radjenovic_2013 | 5842 | 48 | 0.21 | 0.88 | 0.98 | 0.98 | 0.02 | 18.8 |
| closed | Rinne_2021 | 1765 | 407 | 0.11 | 0.43 | 0.83 | 0.89 | 0.10 | 6.1 |
| closed | Seghers_2022 | 4521 | 28 | 0.14 | 0.36 | 0.46 | 0.54 | 0.32 | 3.4 |
| closed | Smid_2019 | 2888 | 27 | 0.41 | 0.93 | 0.96 | 0.96 | 0.04 | 53.5 |
| closed | Taschner_2024 | 3828 | 109 | 0.05 | 0.34 | 0.67 | 0.77 | 0.23 | 8.3 |
| closed | Theobald_2021 | 350 | 35 | 0.09 | 0.26 | 0.43 | 0.43 | 0.43 | 2.7 |
| closed | Verdugo-Castro_2022 | 3306 | 26 | 0.04 | 0.23 | 0.58 | 0.65 | 0.23 | 3.3 |
| closed | Viguie_2020 | 5978 | 321 | 0.01 | 0.02 | 0.26 | 0.79 | 0.08 | 12.7 |
| closed | Wassenaar_2017 | 7548 | 44 | 0.39 | 1.00 | 1.00 | 1.00 | 0.00 | 59.1 |
| closed | van_Ballegooijen_2024 | 8800 | 182 | 0.16 | 0.77 | 0.95 | 0.97 | 0.02 | 33.2 |

## Strategy profile (share of charged budget by action)

| system | B | search | expand_refs | expand_citers | read | noop | unique queries | first expand step | leftover budget | stops with >=25% left |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25/keyword | 20 | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 1.0 | - | 0.00 | 0.00 |
| bm25/keyword | 40 | 0.97 | 0.00 | 0.00 | 0.00 | 0.00 | 1.0 | - | 0.00 | 0.00 |
| haiku-5.5/hybrid | 20 | 0.75 | 0.02 | 0.15 | 0.02 | 0.00 | 12.4 | 3.3 | 0.17 | 0.21 |
| haiku-5.5/hybrid | 40 | 0.80 | 0.01 | 0.12 | 0.03 | 0.00 | 23.9 | 3.8 | 0.25 | 0.42 |
| qwen3-32b/hybrid | 20 | 0.71 | 0.00 | 0.01 | 0.21 | 0.01 | 3.9 | 7.0 | 0.19 | 0.42 |
| qwen3-32b/hybrid | 40 | 0.78 | 0.00 | 0.00 | 0.18 | 0.01 | 5.0 | - | 0.27 | 0.42 |
| snowball-tuned/hybrid | 20 | 0.05 | 0.07 | 0.03 | 0.79 | 0.00 | 1.0 | 16.0 | 0.00 | 0.00 |
| snowball-tuned/hybrid | 40 | 0.03 | 0.12 | 0.10 | 0.72 | 0.00 | 1.0 | 16.0 | 0.00 | 0.00 |
| sonnet-5.5/hybrid | 40 | 0.92 | 0.00 | 0.01 | 0.05 | 0.00 | 34.0 | 16.8 | 0.05 | 0.08 |

## Missed gold at B=40 (counts over episodes)

| system | exposed, not included | not reached in budget | unreached, BM25 rank<=100 | unreachable & rank>100 | no lexical overlap | missing abstract |
|---|---|---|---|---|---|---|
| bm25/keyword|B40 | 0 | 1117 | 0 | 393 | 59 | 548 |
| haiku-5.5/hybrid|B40 | 198 | 779 | 7 | 349 | 49 | 466 |
| qwen3-32b/hybrid|B40 | 123 | 1113 | 10 | 408 | 55 | 539 |
| snowball-tuned/hybrid|B40 | 15 | 1503 | 39 | 508 | 58 | 605 |
| sonnet-5.5/hybrid|B40 | 271 | 692 | 5 | 276 | 46 | 450 |

## Figures

![curves_b20.png](curves_b20.png)
![curves_b40.png](curves_b40.png)
![search_vs_screening.png](search_vs_screening.png)
![reachability.png](reachability.png)
