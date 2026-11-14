# Why web agents stop early (repeat 0)

left = mean share of budget unspent; empty = final turn saw no unscreened papers;
stop_empty_left = stopped on an empty frontier with more than half the budget left; zero_seed = episode had no seeds;
dead_expand = share of expands that returned no new paper.

## eval

| agent | mode | B | n | left | empty | stop_empty_left | zero_seed | dead_expand |
|---|---|---|---|---|---|---|---|---|
| haiku-5.5 | citation | 10 | 8 | 0.53 | 0.62 | 0.50 | 0.12 | 0.19 |
| haiku-5.5 | citation | 20 | 8 | 0.78 | 1.00 | 1.00 | 0.12 | 0.21 |
| haiku-5.5 | citation | 40 | 8 | 0.89 | 0.75 | 0.75 | 0.12 | 0.14 |
| haiku-5.5 | hybrid | 10 | 8 | 0.23 | 0.12 | 0.00 | 0.12 | 0.00 |
| haiku-5.5 | hybrid | 20 | 8 | 0.26 | 0.25 | 0.00 | 0.12 | 0.42 |
| haiku-5.5 | hybrid | 40 | 8 | 0.47 | 0.38 | 0.12 | 0.12 | 0.02 |
| sonnet-5.5 | citation | 40 | 8 | 0.69 | 0.75 | 0.50 | 0.12 | 0.27 |
| sonnet-5.5 | hybrid | 10 | 8 | 0.14 | 0.00 | 0.00 | 0.12 | 0.00 |
| sonnet-5.5 | hybrid | 20 | 8 | 0.10 | 0.00 | 0.00 | 0.12 | 0.50 |
| sonnet-5.5 | hybrid | 40 | 8 | 0.08 | 0.25 | 0.00 | 0.12 | 0.33 |

## transfer

| agent | mode | B | n | left | empty | stop_empty_left | zero_seed | dead_expand |
|---|---|---|---|---|---|---|---|---|
| haiku-5.5 | hybrid | 20 | 24 | 0.23 | 0.08 | 0.00 | 0.00 | 0.14 |
| haiku-5.5 | hybrid | 40 | 24 | 0.35 | 0.42 | 0.17 | 0.00 | 0.28 |

