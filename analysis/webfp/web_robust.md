# Live-web robustness: paired tests, judge sensitivity, cost, beyond-search yield

## 1. Paired comparisons over reviews (recall, B - A)

CI: bootstrap over reviews; up/down: reviews where B beats / trails A; p: exact two-sided sign test.

| comparison | reviews | mean diff | 95% CI | up/down | sign-test p |
|---|---|---|---|---|---|
| eval: v2 - original, haiku citation B=10 | 8 | +0.036 | [+0.002, +0.084] | 3/0 | 0.250 |
| eval: v2 - original, haiku citation B=20 | 8 | +0.047 | [+0.008, +0.100] | 3/0 | 0.250 |
| eval: v2 - original, haiku citation B=40 | 8 | +0.042 | [+0.000, +0.100] | 2/0 | 0.500 |
| eval: v2 - original, sonnet citation B=40 | 8 | +0.070 | [+0.026, +0.121] | 6/0 | 0.031 |
| eval: v2 - original, haiku hybrid B=40 | 8 | +0.032 | [-0.037, +0.133] | 2/2 | 1.000 |
| HELD-OUT transfer: v2 - original, haiku citation B=40 | 24 | +0.001 | [-0.026, +0.034] | 2/4 | 0.688 |
| HELD-OUT transfer: v2 - original, haiku hybrid B=20 | 24 | +0.026 | [+0.002, +0.051] | 14/6 | 0.115 |
| HELD-OUT transfer: v2 - original, haiku hybrid B=40 | 24 | +0.037 | [+0.007, +0.071] | 13/7 | 0.263 |
| eval: sonnet hybrid - snowball hybrid, B=40 | 8 | +0.316 | [+0.194, +0.447] | 8/0 | 0.008 |
| eval: sonnet hybrid - OpenAlex search, B=40 | 8 | +0.315 | [+0.192, +0.449] | 8/0 | 0.008 |
| eval: haiku citation - snowball citation, B=40 | 8 | +0.096 | [-0.002, +0.231] | 3/1 | 0.625 |
| transfer: haiku hybrid - snowball hybrid, B=40 | 24 | +0.087 | [+0.061, +0.114] | 22/0 | 0.000 |
| transfer: haiku hybrid - OpenAlex search, B=40 | 24 | +0.080 | [+0.056, +0.105] | 22/0 | 0.000 |
| P1 transfer: haiku hybrid agent - haiku no-loop (static queries), B=40 | 21 | -0.086 | [-0.121, -0.055] | 0/18 | 0.000 |
| transfer: haiku hybrid agent - haiku no-loop, B=20 | 22 | -0.057 | [-0.084, -0.031] | 3/17 | 0.003 |
| transfer: haiku no-loop screened - include-all, B=40 | 21 | -0.020 | [-0.032, -0.010] | 0/13 | 0.000 |
| eval: haiku hybrid agent - haiku no-loop, B=40 | 8 | -0.198 | [-0.310, -0.103] | 0/8 | 0.008 |
| eval: sonnet hybrid agent - sonnet no-loop, B=40 | 8 | +0.324 | [+0.193, +0.477] | 8/0 | 0.008 |
| transfer: trained 1.7B (GRPO) - haiku hybrid, B=40 | 0 | - | - | - | - |
| transfer: trained 1.7B (GRPO) - SFT, B=40 | 0 | - | - | - | - |
| transfer: trained 1.7B (GRPO) - haiku no-loop, B=40 | 0 | - | - | - | - |
| eval: trained 1.7B (GRPO) - haiku hybrid, B=40 | 0 | - | - | - | - |

Mean recall per web system (original environment unless noted):

| split | system | mode | B | n | recall |
|---|---|---|---|---|---|
| eval | haiku-5.5 | citation | 10 | 8 | 0.050 |
| eval | haiku-5.5 | citation | 20 | 8 | 0.087 |
| eval | haiku-5.5 | citation | 40 | 8 | 0.108 |
| eval | haiku-5.5 | hybrid | 10 | 8 | 0.141 |
| eval | haiku-5.5 | hybrid | 20 | 8 | 0.165 |
| eval | haiku-5.5 | hybrid | 40 | 8 | 0.171 |
| eval | openalex-search | keyword | 10 | 8 | 0.010 |
| eval | openalex-search | keyword | 20 | 8 | 0.010 |
| eval | openalex-search | keyword | 40 | 8 | 0.010 |
| eval | sonnet-5.5 | citation | 40 | 8 | 0.148 |
| eval | sonnet-5.5 | hybrid | 10 | 8 | 0.215 |
| eval | sonnet-5.5 | hybrid | 20 | 8 | 0.295 |
| eval | sonnet-5.5 | hybrid | 40 | 8 | 0.324 |
| eval | snowball-tuned | citation | 10 | 8 | 0.008 |
| eval | snowball-tuned | citation | 20 | 8 | 0.010 |
| eval | snowball-tuned | citation | 40 | 8 | 0.012 |
| eval | snowball-tuned | hybrid | 10 | 8 | 0.008 |
| eval | snowball-tuned | hybrid | 20 | 8 | 0.008 |
| eval | snowball-tuned | hybrid | 40 | 8 | 0.008 |
| eval | haiku-5.5 (v2 env) | citation | 10 | 8 | 0.086 |
| eval | haiku-5.5 (v2 env) | citation | 20 | 8 | 0.134 |
| eval | haiku-5.5 (v2 env) | citation | 40 | 8 | 0.150 |
| eval | haiku-5.5 (v2 env) | hybrid | 40 | 8 | 0.203 |
| eval | sonnet-5.5 (v2 env) | citation | 40 | 8 | 0.218 |
| eval | haiku-5.5-static | keyword | 20 | 8 | 0.331 |
| eval | haiku-5.5-static | keyword | 40 | 8 | 0.368 |
| eval | haiku-5.5-static-all | keyword | 40 | 8 | 0.364 |
| eval | sonnet-5.5-static | keyword | 40 | 8 | 0.000 |
| transfer | haiku-5.5 | hybrid | 20 | 24 | 0.084 |
| transfer | haiku-5.5 | hybrid | 40 | 24 | 0.098 |
| transfer | openalex-search | keyword | 20 | 24 | 0.018 |
| transfer | openalex-search | keyword | 40 | 24 | 0.018 |
| transfer | snowball-tuned | citation | 20 | 24 | 0.005 |
| transfer | snowball-tuned | citation | 40 | 24 | 0.007 |
| transfer | snowball-tuned | hybrid | 20 | 24 | 0.011 |
| transfer | snowball-tuned | hybrid | 40 | 24 | 0.011 |
| transfer | haiku-5.5 | citation | 40 | 24 | 0.036 |
| transfer | haiku-5.5 (v2 env) | citation | 40 | 24 | 0.037 |
| transfer | haiku-5.5 (v2 env) | hybrid | 20 | 24 | 0.110 |
| transfer | haiku-5.5 (v2 env) | hybrid | 40 | 24 | 0.134 |
| transfer | haiku-5.5-static | keyword | 20 | 22 | 0.142 |
| transfer | haiku-5.5-static | keyword | 40 | 21 | 0.176 |
| transfer | haiku-5.5-static-all | keyword | 40 | 21 | 0.196 |

## 2. Judge sensitivity (Rogan-Gladen eligible share of never-screened records)

**gemini-3.8-flash**: measured sensitivity 0.86, specificity 0.76, unscreened yes-rate 0.45; eligible share 0.33, bootstrap 95% CI [0.27, 0.39] (calibration sampling only).

If specificity on never-screened records differs from the screened-excluded estimate:

| sensitivity \ specificity | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|
| 0.80 | 0.21 | 0.29 | 0.36 | 0.41 | 0.45 | 0.49 |
| 0.86 | 0.19 | 0.26 | 0.32 | 0.37 | 0.42 | 0.45 |
| 0.95 | 0.16 | 0.22 | 0.28 | 0.33 | 0.37 | 0.41 |

**gpt-5.6-luna**: measured sensitivity 0.87, specificity 0.72, unscreened yes-rate 0.45; eligible share 0.29, bootstrap 95% CI [0.22, 0.36] (calibration sampling only).

If specificity on never-screened records differs from the screened-excluded estimate:

| sensitivity \ specificity | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|
| 0.80 | 0.23 | 0.30 | 0.37 | 0.42 | 0.46 | 0.50 |
| 0.87 | 0.19 | 0.27 | 0.32 | 0.38 | 0.42 | 0.46 |
| 0.95 | 0.17 | 0.23 | 0.29 | 0.34 | 0.38 | 0.41 |

Eligibility here is abstract-level: gold studies were included after full-text review, so these shares are upper bounds on full-text eligibility.

## 3. Cost per episode and per gold study found (B=40, original environment)

Dollars are recovered from the response cache; 'matched' is the share of turns whose cached usage was found.

| split | system | mode | n | LLM calls/ep | $/episode | gold found/ep | $/gold found | wall s/ep | matched |
|---|---|---|---|---|---|---|---|---|---|
| eval | haiku-5.5 | citation | 8 | 3.6 | 0.001 | 1.9 | 0.000 | 4 | 1.00 |
| eval | haiku-5.5 | hybrid | 8 | 20.4 | 0.006 | 3.8 | 0.002 | 38 | 1.00 |
| eval | haiku-5.5-static | keyword | 8 | 39.5 | 0.004 | 12.1 | 0.000 | 37 | 1.00 |
| eval | haiku-5.5-static-all | keyword | 8 | 1.0 | 0.000 | 12.4 | 0.000 | 17 | 1.00 |
| eval | openalex-search | keyword | 8 | 0.0 | 0.000 | 0.2 | 0.000 | 1 | 1.00 |
| eval | snowball-tuned | citation | 8 | 0.0 | 0.000 | 0.4 | 0.000 | 2 | 1.00 |
| eval | snowball-tuned | hybrid | 8 | 0.0 | 0.000 | 0.1 | 0.000 | 2 | 1.00 |
| eval | sonnet-5.5 | citation | 8 | 9.0 | 0.058 | 3.5 | 0.017 | 25 | 1.00 |
| eval | sonnet-5.5 | hybrid | 8 | 36.5 | 0.234 | 10.5 | 0.022 | 99 | 1.00 |
| eval | sonnet-5.5-static | keyword | 8 | 14.2 | 0.000 | 0.0 | - | 2 | 0.00 |
| transfer | haiku-5.5 | hybrid | 24 | 25.0 | 0.007 | 9.0 | 0.001 | 51 | 1.00 |
| transfer | haiku-5.5-static | keyword | 21 | 41.0 | 0.004 | 9.5 | 0.000 | 38 | 1.00 |
| transfer | haiku-5.5-static-all | keyword | 21 | 1.0 | 0.000 | 10.8 | 0.000 | 17 | 1.00 |
| transfer | openalex-search | keyword | 24 | 0.0 | 0.000 | 0.7 | 0.000 | 1 | 1.00 |
| transfer | snowball-tuned | citation | 24 | 0.0 | 0.000 | 0.2 | 0.000 | 3 | 1.00 |
| transfer | snowball-tuned | hybrid | 24 | 0.0 | 0.000 | 0.3 | 0.000 | 2 | 1.00 |

## 4. Beyond the review's own search (B=40, original environment, judge = Gemini)

Per episode: gold found, and never-screened records included by the system that the judge calls eligible
(expected eligible = judged-eligible x PPV). Abstract-level eligibility; records without an abstract are not counted.

PPV used: 0.64.

| split | system | mode | n | gold found/ep | judged-eligible new/ep | expected eligible new/ep | new as % of gold found |
|---|---|---|---|---|---|---|---|
| eval | haiku-5.5 | citation | 8 | 1.9 | 0.2 | 0.2 | 9% |
| eval | haiku-5.5 | hybrid | 8 | 3.8 | 4.4 | 2.8 | 75% |
| eval | openalex-search | keyword | 8 | 0.2 | 0.5 | 0.3 | 129% |
| eval | sonnet-5.5 | citation | 8 | 3.5 | 3.8 | 2.4 | 69% |
| eval | sonnet-5.5 | hybrid | 8 | 10.5 | 12.0 | 7.7 | 73% |
| transfer | haiku-5.5 | hybrid | 24 | 9.0 | 19.5 | 12.5 | 139% |
| transfer | openalex-search | keyword | 24 | 0.7 | 1.2 | 0.8 | 113% |
