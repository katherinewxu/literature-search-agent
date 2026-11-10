# Live-web inclusions: how many already have a human label?

Per cell: all inclusions, then the share that are gold, screened-and-excluded by the review
authors, or never screened. `abs-incl` = screened-excluded records that passed the review's
title/abstract stage (excluded only at full text; eval reviews only). Pooled over reviews.

## eval (8 reviews)

| agent | mode | B | inclusions | gold | screened-excluded | unscreened | abs-incl (of staged excl) |
|---|---|---|---|---|---|---|---|
| haiku-5.5 | citation | 10 | 28 | 29% | 25% | 46% | 0/0 |
| haiku-5.5 | citation | 20 | 42 | 31% | 38% | 31% | 0/0 |
| haiku-5.5 | citation | 40 | 39 | 38% | 31% | 31% | 0/0 |
| haiku-5.5 | hybrid | 10 | 113 | 18% | 25% | 58% | 0/3 |
| haiku-5.5 | hybrid | 20 | 195 | 14% | 19% | 66% | 0/1 |
| haiku-5.5 | hybrid | 40 | 226 | 13% | 24% | 62% | 1/6 |
| openalex-search | keyword | 10 | 665 | 0% | 9% | 91% | 0/1 |
| openalex-search | keyword | 20 | 990 | 0% | 8% | 92% | 0/1 |
| openalex-search | keyword | 40 | 990 | 0% | 8% | 92% | 0/1 |
| sonnet-5.5 | citation | 40 | 106 | 26% | 11% | 62% | 0/0 |
| sonnet-5.5 | hybrid | 10 | 135 | 33% | 14% | 53% | 0/0 |
| sonnet-5.5 | hybrid | 20 | 242 | 30% | 15% | 55% | 1/1 |
| sonnet-5.5 | hybrid | 40 | 345 | 24% | 15% | 61% | 2/3 |

LLM agents pooled: 1471 inclusions, 1128 not gold; of those, 24% screened-excluded (human negative) and 76% unscreened.

Per review (LLM agents, non-gold inclusions): screened-excluded / unscreened

Appenzeller-Herzog_2019 27/9, Bos_2018 27/26, Brouwer_2019 63/112, Donners_2021 33/88, Hall_2011 30/505, Leenaars_2019 51/50, Meijboom_2021 16/34, Muthu_2020 28/29

## transfer (24 reviews)

| agent | mode | B | inclusions | gold | screened-excluded | unscreened | abs-incl (of staged excl) |
|---|---|---|---|---|---|---|---|
| haiku-5.5 | hybrid | 20 | 996 | 15% | 12% | 73% | 0/0 |
| haiku-5.5 | hybrid | 40 | 1532 | 14% | 13% | 73% | 0/0 |
| openalex-search | keyword | 20 | 3571 | 0% | 4% | 96% | 0/0 |
| openalex-search | keyword | 40 | 3571 | 0% | 4% | 96% | 0/0 |

LLM agents pooled: 2528 inclusions, 2159 not gold; of those, 14% screened-excluded (human negative) and 86% unscreened.

Per review (LLM agents, non-gold inclusions): screened-excluded / unscreened

Abgaz_2023 19/77, Adamo_2021 15/22, Ali_2024 17/56, Anmarkrud_2021 14/62, Attai_2022 18/76, Deckers_2022 11/251, Filges_2018 28/57, Filges_2022 10/31, Harms_2024 4/89, Kapuka_2021 2/53, Moran_2020 4/19, Nobrega-Dos_Santos_2023 6/24, Noetel_2021 2/20, Quevedo_2023 8/225, Radjenovic_2013 6/110, Rinne_2021 5/165, Seghers_2022 8/59, Smid_2019 8/47, Taschner_2024 7/57, Theobald_2021 5/85, Verdugo-Castro_2022 5/35, Viguie_2020 19/161, Wassenaar_2017 11/36, van_Ballegooijen_2024 78/32

- unscreened.jsonl: 1904 unique (review, work) pairs written; 0 without a cached title skipped.
- screened_excluded.jsonl: 572 unique (review, work) pairs written; 0 without a cached title skipped.
