# Closed graph: Haiku agent vs no-loop Haiku query list

Paired over reviews; diff = no-loop - agent; 95% bootstrap CI; exact two-sided sign test.

## eval, B=20: haiku-5.5-static vs haiku-5.5 hybrid agent

| metric | n | diff | 95% CI | up/down | sign p |
|---|---|---|---|---|---|
| recall | 8 | +0.123 | [+0.019, +0.232] | 5/2 | 0.453 |
| precision | 8 | -0.097 | [-0.135, -0.052] | 1/7 | 0.070 |
| f1 | 8 | -0.023 | [-0.092, +0.056] | 2/6 | 0.289 |
| included_count | 8 | +46.000 | [+29.375, +65.000] | 8/0 | 0.008 |
| charged_cost | 8 | +5.875 | [+3.250, +8.875] | 8/0 | 0.008 |

| system | n | recall | precision | F1 | included | spent |
|---|---|---|---|---|---|---|
| agent | 8 | 0.432 | 0.321 | 0.323 | 40.5 | 14.1 |
| haiku-5.5-static | 8 | 0.555 | 0.224 | 0.300 | 86.5 | 20.0 |

## eval, B=40: haiku-5.5-static vs haiku-5.5 hybrid agent

| metric | n | diff | 95% CI | up/down | sign p |
|---|---|---|---|---|---|
| recall | 8 | +0.025 | [-0.023, +0.075] | 4/2 | 0.688 |
| precision | 8 | -0.101 | [-0.132, -0.074] | 0/8 | 0.008 |
| f1 | 8 | -0.100 | [-0.130, -0.072] | 0/8 | 0.008 |
| included_count | 8 | +56.125 | [+33.125, +80.875] | 8/0 | 0.008 |
| charged_cost | 8 | +7.750 | [+4.500, +11.000] | 8/0 | 0.008 |

| system | n | recall | precision | F1 | included | spent |
|---|---|---|---|---|---|---|
| agent | 8 | 0.626 | 0.294 | 0.385 | 75.2 | 32.2 |
| haiku-5.5-static | 8 | 0.651 | 0.193 | 0.285 | 131.4 | 40.0 |

## eval, B=40: haiku-5.5-static-all vs haiku-5.5 hybrid agent

| metric | n | diff | 95% CI | up/down | sign p |
|---|---|---|---|---|---|
| recall | 8 | +0.060 | [-0.002, +0.133] | 5/2 | 0.453 |
| precision | 8 | -0.221 | [-0.280, -0.166] | 0/8 | 0.008 |
| f1 | 8 | -0.259 | [-0.303, -0.202] | 0/8 | 0.008 |
| included_count | 8 | +301.250 | [+263.375, +332.875] | 8/0 | 0.008 |
| charged_cost | 8 | +7.750 | [+4.625, +11.000] | 8/0 | 0.008 |

| system | n | recall | precision | F1 | included | spent |
|---|---|---|---|---|---|---|
| agent | 8 | 0.626 | 0.294 | 0.385 | 75.2 | 32.2 |
| haiku-5.5-static-all | 8 | 0.686 | 0.073 | 0.126 | 376.5 | 40.0 |

## transfer, B=20: haiku-5.5-static vs haiku-5.5 hybrid agent

| metric | n | diff | 95% CI | up/down | sign p |
|---|---|---|---|---|---|
| recall | 24 | +0.093 | [+0.060, +0.127] | 19/2 | 0.000 |
| precision | 24 | -0.121 | [-0.159, -0.087] | 2/22 | 0.000 |
| f1 | 24 | -0.033 | [-0.072, +0.003] | 9/15 | 0.307 |
| included_count | 24 | +43.667 | [+36.458, +51.042] | 24/0 | 0.000 |
| charged_cost | 24 | +3.458 | [+2.458, +4.708] | 24/0 | 0.000 |

| system | n | recall | precision | F1 | included | spent |
|---|---|---|---|---|---|---|
| agent | 24 | 0.370 | 0.413 | 0.343 | 55.6 | 16.5 |
| haiku-5.5-static | 24 | 0.462 | 0.292 | 0.310 | 99.3 | 20.0 |

## transfer, B=40: haiku-5.5-static vs haiku-5.5 hybrid agent

| metric | n | diff | 95% CI | up/down | sign p |
|---|---|---|---|---|---|
| recall | 24 | +0.112 | [+0.076, +0.149] | 22/2 | 0.000 |
| precision | 24 | -0.144 | [-0.206, -0.095] | 0/24 | 0.000 |
| f1 | 24 | -0.081 | [-0.130, -0.039] | 3/21 | 0.000 |
| included_count | 24 | +77.917 | [+62.083, +97.167] | 23/1 | 0.000 |
| charged_cost | 24 | +10.042 | [+6.708, +13.583] | 24/0 | 0.000 |

| system | n | recall | precision | F1 | included | spent |
|---|---|---|---|---|---|---|
| agent | 24 | 0.477 | 0.390 | 0.390 | 87.7 | 30.0 |
| haiku-5.5-static | 24 | 0.589 | 0.246 | 0.309 | 165.6 | 40.0 |

## transfer, B=40: haiku-5.5-static-all vs haiku-5.5 hybrid agent

| metric | n | diff | 95% CI | up/down | sign p |
|---|---|---|---|---|---|
| recall | 24 | +0.195 | [+0.142, +0.250] | 22/1 | 0.000 |
| precision | 24 | -0.266 | [-0.335, -0.208] | 0/24 | 0.000 |
| f1 | 24 | -0.207 | [-0.276, -0.147] | 1/23 | 0.000 |
| included_count | 24 | +303.417 | [+282.125, +323.167] | 24/0 | 0.000 |
| charged_cost | 24 | +10.042 | [+6.708, +13.625] | 24/0 | 0.000 |

| system | n | recall | precision | F1 | included | spent |
|---|---|---|---|---|---|---|
| agent | 24 | 0.477 | 0.390 | 0.390 | 87.7 | 30.0 |
| haiku-5.5-static-all | 24 | 0.671 | 0.123 | 0.182 | 391.1 | 40.0 |

