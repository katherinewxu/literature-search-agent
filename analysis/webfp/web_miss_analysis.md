# Why web agents miss gold studies

## eval (8 reviews)

Share of gold studies by cause (summed over reviews).

| agent | mode | B | n gold | found | rejected_read | rejected_unread | seen_undecided | one_hop | other_system | never_surfaced |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku-5.5 | citation | 10 | 341 | 0.02 | 0.00 | 0.00 | 0.00 | 0.09 | 0.31 | 0.58 |
| haiku-5.5 | citation | 20 | 341 | 0.04 | 0.00 | 0.01 | 0.00 | 0.09 | 0.30 | 0.55 |
| haiku-5.5 | citation | 40 | 341 | 0.04 | 0.00 | 0.01 | 0.00 | 0.12 | 0.28 | 0.54 |
| haiku-5.5 | hybrid | 10 | 341 | 0.06 | 0.00 | 0.01 | 0.00 | 0.33 | 0.18 | 0.42 |
| haiku-5.5 | hybrid | 20 | 341 | 0.08 | 0.00 | 0.02 | 0.00 | 0.34 | 0.16 | 0.40 |
| haiku-5.5 | hybrid | 40 | 341 | 0.09 | 0.00 | 0.02 | 0.00 | 0.35 | 0.16 | 0.38 |
| sonnet-5.5 | citation | 40 | 341 | 0.08 | 0.00 | 0.01 | 0.00 | 0.23 | 0.21 | 0.47 |
| sonnet-5.5 | hybrid | 10 | 341 | 0.13 | 0.00 | 0.02 | 0.00 | 0.30 | 0.11 | 0.44 |
| sonnet-5.5 | hybrid | 20 | 341 | 0.21 | 0.00 | 0.02 | 0.00 | 0.29 | 0.06 | 0.41 |
| sonnet-5.5 | hybrid | 40 | 341 | 0.25 | 0.01 | 0.04 | 0.00 | 0.31 | 0.02 | 0.37 |

one_hop: size of the smallest citation neighbourhood (refs + citers) linking the gold study to an included paper;
an expand returns at most 20, so large neighbourhoods are not one cheap step away.

| agent | mode | B | n | median | share <= 20 | share <= 100 |
|---|---|---|---|---|---|---|
| haiku-5.5 | citation | 10 | 29 | 56 | 0.00 | 0.79 |
| haiku-5.5 | citation | 20 | 32 | 65 | 0.00 | 0.84 |
| haiku-5.5 | citation | 40 | 42 | 80 | 0.00 | 0.76 |
| haiku-5.5 | hybrid | 10 | 111 | 85 | 0.03 | 0.98 |
| haiku-5.5 | hybrid | 20 | 115 | 85 | 0.03 | 0.98 |
| haiku-5.5 | hybrid | 40 | 118 | 85 | 0.05 | 0.98 |
| sonnet-5.5 | citation | 40 | 77 | 56 | 0.00 | 1.00 |
| sonnet-5.5 | hybrid | 10 | 102 | 53 | 0.13 | 0.86 |
| sonnet-5.5 | hybrid | 20 | 99 | 35 | 0.13 | 0.95 |
| sonnet-5.5 | hybrid | 40 | 107 | 40 | 0.09 | 0.93 |

Gold studies surfaced by at least one web run vs by none:

| group | n | has abstract | median year | median citation degree | median title words |
|---|---|---|---|---|---|
| surfaced by some run | 136 | 0.94 | 2010.0 | 40.0 | 13.0 |
| never surfaced | 205 | 0.80 | 2009 | 33 | 12 |

## transfer (24 reviews)

Share of gold studies by cause (summed over reviews).

| agent | mode | B | n gold | found | rejected_read | rejected_unread | seen_undecided | one_hop | other_system | never_surfaced |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku-5.5 | hybrid | 20 | 2233 | 0.07 | 0.00 | 0.01 | 0.00 | 0.26 | 0.02 | 0.64 |
| haiku-5.5 | hybrid | 40 | 2233 | 0.10 | 0.00 | 0.01 | 0.00 | 0.28 | 0.01 | 0.60 |

one_hop: size of the smallest citation neighbourhood (refs + citers) linking the gold study to an included paper;
an expand returns at most 20, so large neighbourhoods are not one cheap step away.

| agent | mode | B | n | median | share <= 20 | share <= 100 |
|---|---|---|---|---|---|---|
| haiku-5.5 | hybrid | 20 | 584 | 54 | 0.07 | 0.85 |
| haiku-5.5 | hybrid | 40 | 634 | 52 | 0.09 | 0.86 |

Gold studies surfaced by at least one web run vs by none:

| group | n | has abstract | median year | median citation degree | median title words |
|---|---|---|---|---|---|
| surfaced by some run | 288 | 0.84 | 2016.0 | 47.0 | 14.0 |
| never surfaced | 1945 | 0.70 | 2015.0 | 39 | 13 |

## Example gold studies the agent rejected

- [eval] haiku-5.5 hybrid B=20, Appenzeller-Herzog_2019, rejected_unread: Serum Transaminases in Children with Wilson???s Disease
- [eval] haiku-5.5 hybrid B=20, Appenzeller-Herzog_2019, rejected_unread: Treatment of Wilson Disease With Ammonium Tetrathiomolybdate
- [eval] haiku-5.5 hybrid B=20, Appenzeller-Herzog_2019, rejected_unread: Wilson disease: Histopathological correlations with treatment on follow-up liver biopsies
- [eval] haiku-5.5 hybrid B=40, Appenzeller-Herzog_2019, rejected_unread: Hepatic manifestations of Wilson’s disease: 12-year experience in a Swiss tertiary referral centre
- [eval] haiku-5.5 hybrid B=40, Appenzeller-Herzog_2019, rejected_unread: Early neurological worsening in patients with Wilson's disease
- [eval] sonnet-5.5 hybrid B=10, Appenzeller-Herzog_2019, rejected_unread: Zinc monotherapy is effective in Wilson’s disease patients with mild liver disease diagnosed in childhood: a retrospective study
- [eval] sonnet-5.5 hybrid B=40, Appenzeller-Herzog_2019, rejected_read: Long‐term follow‐up of Wilson Disease: natural history, treatment, mutations analysis and phenotypic correlation
- [eval] sonnet-5.5 hybrid B=40, Appenzeller-Herzog_2019, rejected_read: Long-term outcome in Serbian patients with Wilson disease
- [eval] sonnet-5.5 hybrid B=20, Bos_2018, rejected_unread: Brain Imaging and Cognitive Predictors of Stroke and Alzheimer Disease in the Framingham Heart Study
- [eval] haiku-5.5 citation B=20, Brouwer_2019, rejected_unread: Relapse after cognitive behavior therapy of depression: potential implications for longer courses of treatment
- [eval] haiku-5.5 hybrid B=20, Brouwer_2019, rejected_unread: Recurrence of major depressive disorder and its predictors in the general population: results from The Netherlands Mental Health Survey and Incidence Study (NEMESIS)
- [eval] haiku-5.5 hybrid B=40, Brouwer_2019, rejected_unread: Personality and cognitive vulnerability in remitted recurrently depressed patients
- [eval] sonnet-5.5 citation B=40, Brouwer_2019, rejected_unread: Prevention of relapse/recurrence in major depressive disorder with either mindfulness-based cognitive therapy or cognitive therapy.
- [eval] sonnet-5.5 hybrid B=20, Brouwer_2019, rejected_unread: Factors predicting the long-term illness course in a cohort of depressed inpatients
- [eval] sonnet-5.5 hybrid B=40, Brouwer_2019, rejected_unread: Is Experiential Avoidance a Mediating, Moderating, Independent, Overlapping, or Proxy Risk Factor in the Onset, Relapse and Maintenance of Depressive Disorders?
- [eval] sonnet-5.5 hybrid B=40, Brouwer_2019, rejected_unread: Personality Disorders Predict Relapse After Remission From an Episode of Major Depressive Disorder
- [eval] sonnet-5.5 hybrid B=40, Brouwer_2019, rejected_unread: Repetitive negative thinking as a predictor of depression and anxiety: A longitudinal cohort study
- [eval] haiku-5.5 citation B=20, Donners_2021, rejected_unread: Emicizumab Prophylaxis in Patients Who Have Hemophilia A without Inhibitors
- [eval] haiku-5.5 citation B=20, Donners_2021, rejected_unread: Emicizumab Prophylaxis in Hemophilia A with Inhibitors
- [eval] haiku-5.5 citation B=20, Donners_2021, rejected_unread: Factor VIII–Mimetic Function of Humanized Bispecific Antibody in Hemophilia A
- [eval] haiku-5.5 hybrid B=10, Donners_2021, rejected_unread: Long-term safety and efficacy of emicizumab in a phase 1/2 study in patients with hemophilia A with or without inhibitors
- [eval] haiku-5.5 hybrid B=10, Donners_2021, rejected_unread: A multicentre, open‐label study of emicizumab given every 2 or 4 weeks in children with severe haemophilia A without inhibitors
- [eval] haiku-5.5 hybrid B=20, Donners_2021, rejected_unread: Emicizumab treatment and monitoring in a paediatric cohort: real‐world data
- [eval] sonnet-5.5 hybrid B=20, Donners_2021, rejected_unread: A multicenter, open-label phase 3 study of emicizumab prophylaxis in children with hemophilia A with inhibitors
- [eval] haiku-5.5 hybrid B=10, Hall_2011, rejected_unread: Quantitative analysis of faults and failures in a complex software system
