# Original vs v2 web environment (eval, 8 reviews)

| agent | mode | B | env | n | recall | precision | budget spent | expands/run | dead-end stops | zero-seed runs |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku-5.5 | citation | 10 | original | 8 | 0.050 | 0.120 | 0.47 | 2.2 | 4 | 1 |
| haiku-5.5 | citation | 10 | v2 | 8 | 0.086 | 0.194 | 0.72 | 3.9 | 1 | 0 |
| | | | per-review recall change | 8 | +0.036 | up in 3, down in 0 | | | | |
| haiku-5.5 | citation | 20 | original | 8 | 0.087 | 0.161 | 0.22 | 2.0 | 8 | 1 |
| haiku-5.5 | citation | 20 | v2 | 8 | 0.134 | 0.221 | 0.45 | 4.8 | 3 | 0 |
| | | | per-review recall change | 8 | +0.047 | up in 3, down in 0 | | | | |
| haiku-5.5 | citation | 40 | original | 8 | 0.108 | 0.161 | 0.11 | 1.8 | 6 | 1 |
| haiku-5.5 | citation | 40 | v2 | 8 | 0.150 | 0.178 | 0.23 | 4.8 | 6 | 0 |
| | | | per-review recall change | 8 | +0.042 | up in 2, down in 0 | | | | |
| haiku-5.5 | hybrid | 40 | original | 8 | 0.171 | 0.148 | 0.53 | 1.9 | 1 | 1 |
| haiku-5.5 | hybrid | 40 | v2 | 8 | 0.203 | 0.185 | 0.64 | 2.8 | 2 | 0 |
| | | | per-review recall change | 8 | +0.032 | up in 2, down in 2 | | | | |
| sonnet-5.5 | citation | 40 | original | 8 | 0.148 | 0.180 | 0.31 | 6.0 | 4 | 1 |
| sonnet-5.5 | citation | 40 | v2 | 8 | 0.218 | 0.279 | 0.62 | 14.4 | 0 | 0 |
| | | | per-review recall change | 8 | +0.070 | up in 6, down in 0 | | | | |
