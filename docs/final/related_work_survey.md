# Related work and novelty survey: budgeted LLM literature-search agents on SYNERGY

Compiled 2026-11-07. Every paper listed was checked to exist through arXiv, ACL Anthology, CEUR, PMC, or the Crossref API (DOI resolves, and title, authors and venue match). Each BibTeX entry contains only fields I verified. A `note` field marks anything I could not verify.

**Possible inconsistency in your setup:** the SYNERGY release (De Bruin et al. 2023, doi:10.34894/HE6NAQ; GitHub asreview/synergy-dataset) has **26** systematic reviews, not 72 (169,288 records, 2,834 inclusions). Check where "72 reviews" comes from: other collections added, splits, or a newer release. Reconcile this before the paper says "SYNERGY (72 reviews)".

---

## 1. Novelty verdicts

### Claim (1): scoring web search agents against a fixed review's gold list penalizes them, so pooled evaluation is needed
**Verdict: partially shown before. The general principle is old. Your instantiation (web or LLM search agents on systematic-review benchmarks, with false positives split into "never screened" and "author-rejected", plus re-scoring on a pooled qrel set) appears new.**
- **Closest prior work: Cormack & Grossman (2018), Waterloo at CLEF eHealth TAR** (CEUR Vol-2125, paper 89). They ran BMI over all of MEDLINE, not the Boolean-retrieved pool. They found that "no assessments were available for many, if not most, of the highly ranked documents ... these documents were indistinguishable from those for which 'relevant' assessments were provided." They also argued that the CLEF architecture "tacitly assumes ... that the search and screening phases ... were infallible, and each attained 100% recall." This is the same phenomenon, observed for a whole-database TAR system, eight years earlier. However, it was anecdotal: they did not quantify it, did not calibrate a judge, and did not build a pooled qrel set. You should cite it prominently and frame your work as the first quantification of it for agents.
- **Sahu, Charlin & Pal (2026), arXiv:2605.29234**, "Human Citation Lists Are Not a Ground Truth". Their domain is related-work citation lists of ML papers (RollingEval), not systematic reviews. A neutral LLM judge rates only 51% of human citations as relevant, versus 86–88% for AI re-rankers, and they argue for reporting recall together with judged relevance. This is close in spirit, and it is recent, so reviewers may know it. Ways yours differs:
  - A systematic review's gold set is defined by explicit eligibility criteria, so "relevant" means eligible, not merely topical.
  - SYNERGY's screened pool gives you a natural set of judged negatives. That lets you calibrate the judge (sensitivity and specificity) and separate unscreened records from author-rejected ones. They have no counterpart to this.
  - You re-score with pooled qrels and report the change in recall and precision.
- **Rajit et al. (2025, Research Synthesis Methods)** evaluated automated citation searching over OpenAlex and Semantic Scholar on 27 reviews. They state outright that treating the included studies as the full relevant set "may overstate" the Boolean searches' recall and understate the automated searches. They raise the caveat but do not measure it.
- **The IR principle is old:** see the pooling bias papers in area D (Zobel 1998; Buckley & Voorhees 2004; Buckley et al. 2007; Yilmaz & Aslam 2006; Sakai 2007). LLM hole-filling is also established (Upadhyay et al. 2024; Abbasiantaeb et al. 2024; Gienapp et al. 2025). Do not claim "pooled evaluation is needed" as a new idea. Claim the measurement and the systematic-review-specific protocol.

### Claim (2): agents and human searches are complementary
**Verdict: already shown qualitatively, in small studies. Your large-scale quantification is new-ish.**
- **otto-SR, Cao et al. 2025 (medRxiv preprint).** An LLM workflow reproduced 12 Cochrane reviews. After blinded adjudication it found a median of 2.0 (IQR 1–6.5) eligible studies per review that the original authors likely missed. It is not peer reviewed. It reran or updated the reviews, which differs from your setup of a web agent with a budget.
- **Carrier et al. 2026 (Arthroplasty Today).** ChatGPT-5 Deep Research recovered 47.5% of the gold studies across 5 reviews, against 85.2% for human fellows. It also surfaced 3 "newly identified eligible studies" outside the gold set, and the authors explicitly call the two complementary. The sample is small (n=5 reviews).
- Cormack & Grossman 2018 put this forward as a hypothesis (TAR as a "second assessor").
- Methodology literature shows that human reviews do miss eligible studies. Waffenschmidt et al. 2019 report a median of 5% of eligible studies missed under single screening. Sampson et al. 2006 introduced relative recall, which is pooled-recall thinking within the systematic-review community.
- **What is new in yours:** you cover 72 (or 26) reviews, you separate "not found by the human search" from "rejected by the human screen" (76–86% are the former), and you use a calibrated judge.

### Claim (3): the low web recall is a real search failure, not an evaluation artifact
**Verdict: consistent with prior findings, and the robustness check appears new.** Prior work reports low recall for automated or LLM search against systematic-review gold sets:
- Rajit et al. 2025: median recall 35.8% for automated citation searching.
- Carrier et al. 2026: 47.5% and 40.9% for ChatGPT Deep Research and Agent Mode.
- ResearchArena (Kang & Xiong 2024): LLM agents underperform keyword retrieval.
- Wang et al. SIGIR 2023: ChatGPT Boolean queries trade recall for precision.

None of these shows that the recall gap survives pooled re-judging. Your result that recall goes from .32 to .35 under pooling is a new and useful robustness result.

**Two caveats a reviewer will raise:**
- **(a) Judge prevalence correction.** With sensitivity .86 and specificity .76, the raw rate of "eligible" verdicts is biased upward. A Rogan–Gladen correction gives p = (p_obs + spec − 1)/(sens + spec − 1). At p_obs ≈ .33 this gives p ≈ (.33 − .24)/.62 ≈ .15. The share of truly eligible records may therefore be closer to 15% than to a third, unless specificity on unscreened records is higher than on author-rejected ones. That is plausible, since author-rejected records are hard negatives that passed the authors' Boolean search. Report both numbers or justify the choice.
- **(b) Pool circularity.** The pool is built from the evaluated agents' own outputs and judged by an LLM. Clarke & Dietz (2024) and Soboroff (2025) show this can inflate the systems that contributed to the pool. The jump from .10 to .22 on the transfer reviews is the number most exposed to this. Possible mitigations:
  - Leave-one-agent-out pools.
  - A judge from a different model family than the agents.
  - Human spot-checks of a sample of the pooled additions.

### Closest 3 papers to the overall project
1. **PaSa (He et al., ACL 2025).** An RL-trained (PPO) Crawler and Selector agent that does keyword search plus citation expansion over arXiv/Google, evaluated on recall against synthetic and real CS queries. Yours differs in four ways: tool calls have an explicit cost; the benchmark is systematic-review eligibility rather than topical queries; you compare three evaluation universes (screened pool, one-hop graph, live OpenAlex); and you add recall estimation and stopping.
2. **Sahu, Charlin & Pal (arXiv:2605.29234, 2026).** A deep-research literature search agent that expands bibliographies breadth-first, plus the argument that human reference lists are not ground truth, checked with an LLM judge. Theirs uses related-work lists with no explicit criteria. Yours uses systematic-review eligibility criteria, has the screened-pool negatives for judge calibration, re-scores with pooled qrels, uses a cost budget, and trains a small model with RL.
3. **Cormack & Grossman (CLEF eHealth 2018).** Whole-MEDLINE TAR retrieves unassessed records that look relevant, and CLEF qrels assume the original search was infallible. Yours replaces their anecdote with a quantified and calibrated measurement on LLM agents with a web tool, and adds pooled re-scoring.

Runners-up: otto-SR (Cao et al. 2025) and TrialMind (Wang et al. 2025, npj Digit. Med.) are end-to-end LLM systematic-review pipelines that include search, scored against gold included studies. DeepRetrieval (Jiang et al. 2025) is RL-trained query generation with a recall reward on PubMed.

---

## 2. Papers by area

### A. LLM agents for academic or systematic-review literature search
| Paper | One-line summary | Relation to ours |
|---|---|---|
| He et al. 2025, PaSa (ACL) | Crawler and Selector LLM agents, trained with session-level PPO; search plus citation expansion; AutoScholarQuery and RealScholarQuery; beats Google+GPT-4o on recall@20/50 | Closest agent. RL-trained paper search. No budget, no systematic-review eligibility, no pooled evaluation. |
| Shi et al. 2025, SPAR (arXiv:2507.15245) | Training-free multi-agent: query understanding, multi-source retrieval, "RefChain" citation exploration, reranking; SPARBench with expert labels | Citation-chain agent baseline. I could not verify whether SPARBench used pooled annotation. |
| Ajith et al. 2024, LitSearch (EMNLP) | 597 literature-search queries on ML/NLP papers; dense retrievers beat BM25 by 24.8 recall@5 | A static retrieval benchmark, contrasting with your agentic, budgeted, systematic-review setting. |
| Kang & Xiong 2024, ResearchArena (arXiv:2406.10291) | Survey-construction benchmark (discovery, selection, organization) on S2ORC; LLM agents underperform keyword retrieval | Supports claim 3 (LLM search recall is genuinely weak). |
| Jiang et al. 2025, DeepRetrieval (arXiv:2503.00223) | RL (GRPO-style) training of query generation on live search engines, with a recall reward; PubMed publication-search Recall@3K 65.1% vs 24.7% previous best | The closest RL-for-literature-recall work. It is single-shot query generation, not a multi-tool agent with a budget. |
| Sahu, Charlin & Pal 2026 (arXiv:2605.29234) | Deep-research pipeline with breadth-first bibliography expansion, recall <20% to >80% on RollingEval; LLM judge shows human citation lists are weak ground truth | Closest on evaluation (see section 1). |
| Asai et al. 2026, OpenScholar (Nature) | Retrieval-augmented LM over 45M open-access papers; ScholarQABench | Deep-research literature synthesis. Its goal is answer quality, not exhaustive recall. |
| Skarlinski et al. 2024, PaperQA2 (arXiv:2409.13740) | Literature agent that matches or exceeds experts on LitQA2 retrieval and synthesis tasks | Agentic literature search baseline aimed at QA rather than review inclusion. |

No paper or system called "MetaSyn" turned up in my searches, so do not cite one unless you have its exact reference. Tools for snowballing with an LLM in the loop (ProfOlaf, arXiv:2510.26750; SnowBallSLR, a recent preprint) exist but report no recall evaluation. I left them out of the BibTeX.

### B. LLMs for systematic-review screening, Boolean query generation, and automated snowballing
| Paper | One-line summary | Relation |
|---|---|---|
| Wang, Scells, Koopman, Zuccon 2023 (SIGIR) | ChatGPT-generated Boolean queries for systematic reviews: high precision, lower recall | The standard query-generation baseline. Supports claim 3. |
| Wang et al. 2024 (ECIR) | Zero-shot generative LLMs for title and abstract screening, with calibrated thresholds for a recall target | Your include decisions are LLM screening, and you could cite their calibration. |
| Guo et al. 2024 (JMIR) | GPT-4 screening on clinical reviews | An LLM screening accuracy reference. |
| Khraisha et al. 2024 (Res. Synth. Methods) | GPT-4 screening and extraction across languages and grey literature; mixed reliability | Evidence on LLM screening reliability. |
| Rajit et al. 2025 (Res. Synth. Methods) | Automated citation searching (OpenAlex and Semantic Scholar) on 27 reviews: median recall 35.8%, higher precision than Boolean searches; notes the gold-set-completeness assumption | Closest automated-snowballing baseline. Supports claims 1 and 3. |
| Wang et al. 2025, TrialMind (npj Digit. Med.) | LLM pipeline for search, screening and extraction over 100 systematic reviews; search recall 0.71–0.83 | An end-to-end systematic-review LLM system scored against the gold included set. |
| Cao et al. 2025, otto-SR (medRxiv) | LLM workflow reproduced 12 Cochrane reviews; 96.7% screening sensitivity; median of 2 eligible studies per review missed by the authors | Supports claims 1 and 2. A preprint. |
| Kusa et al. 2023, CSMeD (NeurIPS D&B) | Meta-dataset of 325 systematic reviews for citation screening, plus full-text screening data (CSMeD-FT) | A benchmark you could use for transfer, and a related-benchmark citation. |

### C. TAR stopping rules and recall estimation
| Paper | One-line summary | Relation |
|---|---|---|
| Cormack & Grossman 2016 (SIGIR) | Knee and target stopping methods for continuous active learning | The canonical stopping rules your yield-curve stopping should compare against. |
| Callaghan & Müller-Hansen 2020 (Syst. Rev.) | Hypergeometric-test stopping with a recall target and confidence level | The closest analogue to your sample-based Clopper–Pearson bound. |
| Callaghan et al. 2024 (Syst. Rev.) | Commentary: stopping criteria must be robust and well evaluated | Use to motivate reporting calibration and coverage of your bound. |
| Li & Kanoulas 2020 (TOIS) | Stopping by estimating recall from sampling of ranked lists | A sampling-based recall estimation baseline. |
| Yang, Lewis & Frieder 2021 (DocEng) | Heuristic stopping rules for TAR, evaluated | Heuristic baselines, such as yield-curve stopping. |
| Lewis, Yang & Frieder 2021 (CIKM) | Certifying one-phase TAR: valid recall confidence intervals from sampling | The most direct precedent for a statistically certified recall bound. |
| Stevenson & Bin-Hezam 2023 (TOIS) | Point-process (counting) models for TAR stopping | A strong yield-curve modelling baseline. |
| Kanoulas et al. 2017 (CLEF Working Notes) | CLEF eHealth TAR task: 58 Cochrane DTA reviews; qrels exist only for Boolean-retrieved records | A benchmark whose unjudged documents count as non-relevant (relevant to claim 1). |

Note on LLM-based stopping: I found no peer-reviewed work that does LLM-judged sample-based recall estimation for an agent's open-web search. The closest are the sampling certifiers (Lewis et al. 2021; Callaghan & Müller-Hansen 2020). Applying such a bound when the "screened" items are LLM judgments appears new. A caveat: if the bound uses LLM labels, it inherits the judge's error, so propagate sensitivity and specificity.

### D. Incomplete relevance judgments and gold-set incompleteness (most important for novelty)
| Paper | One-line summary | Relation |
|---|---|---|
| Zobel 1998 (SIGIR) | TREC pools miss many relevant documents, so recall is overestimated; system rankings remain mostly stable | The foundational pooling-bias citation. |
| Buckley & Voorhees 2004 (SIGIR) | bpref: an evaluation measure robust to incomplete judgments | An alternative to treating unjudged records as non-relevant. |
| Yilmaz & Aslam 2006 (CIKM) | infAP: estimating average precision from incomplete or sampled judgments | Statistical estimation under incomplete qrels, a sibling of your sampled judge. |
| Buckley, Dimmick, Soboroff, Voorhees 2007 (Inf. Retr.) | Small pools relative to the collection bias judgments toward title-word documents (AQUAINT) | Directly analogous: a review's Boolean search is a biased "pool". |
| Sakai 2007 (SIGIR) | Condensed lists (drop unjudged documents) as an alternative to bpref | A metric option for your screened-pool universe. |
| Cormack & Grossman 2018 (CLEF eHealth) | Whole-MEDLINE TAR retrieves many unassessed records that look relevant; CLEF assumes the search was infallible | **The closest prior observation of your finding.** |
| Thomas et al. 2024 (SIGIR) | LLM relevance labels as accurate as human assessors, at Bing scale | Justifies the LLM judge. |
| Faggioli et al. 2023 (ICTIR) | Perspectives on how far LLMs can replace human relevance judgment | Framing for the human–LLM judging spectrum. |
| Upadhyay et al. 2024, UMBRELA (arXiv:2406.06519) | Open-source reproduction of the Bing LLM assessor | A judge implementation reference. |
| Upadhyay, Kamalloo & Lin 2024 (arXiv:2405.04727) | LLMs fill simulated holes in qrels to restore evaluation | Directly supports pooled hole-filling. |
| Abbasiantaeb et al. 2024 (EMTCIR @ SIGIR-AP, CEUR 3854) | Can LLMs fill relevance judgment holes? (conversational search) | Same. |
| Gienapp et al. 2025 (arXiv:2510.04633) | Topic-specific classifiers trained on about 128 human judgments beat prompted LLM judges for unjudged documents | Suggests calibrating or fine-tuning the judge per review using the author-screened labels, which SYNERGY gives you. |
| Clarke & Dietz 2024 (arXiv:2412.17156) | LLM judging can be gamed: a run that used LLM judgments ranked 5th under UMBRELA and 28th under humans | **Circularity critique you must address.** |
| Alaofi et al. 2024 (SIGIR-AP) | LLM judges over-label documents containing query words as relevant ("gullibility") | Relevant to judge specificity (.76). |
| Soboroff 2025 (IRRJ) | Do not use prompted LLMs to generate ground-truth judgments | A counterpoint to cite. |
| Sampson et al. 2006 (BMC Med. Res. Methodol.) | Relative recall: use the union of included studies across reviews as a pseudo-gold set | The systematic-review community's pooled-recall precedent. |
| Waffenschmidt et al. 2019 (BMC Med. Res. Methodol.) | Single screening misses a median of 5% of eligible studies (range 0–58%) | Evidence that human gold sets are incomplete. |
| Carrier et al. 2026 (Arthroplasty Today) | ChatGPT-5 Deep Research recovers 47.5% of the gold set and also finds eligible studies outside it | Supports claims 2 and 3. Small study. |
| Scells et al. 2017 (SIGIR) | Test collection for retrieving systematic-review studies; qrels come from the review's Boolean search | Another systematic-review benchmark with the same incompleteness assumption. |

---

## 3. BibTeX (verified fields only)

```bibtex
@misc{debruin2023synergy,
  author = {De Bruin, Jonathan and Ma, Yongchao and Ferdinands, Gerbrich and Teijema, Jelle and Van de Schoot, Rens},
  title = {{SYNERGY} -- Open machine learning dataset on study selection in systematic reviews},
  year = {2023},
  publisher = {DataverseNL},
  version = {V1},
  doi = {10.34894/HE6NAQ},
  note = {Dataset; 26 systematic reviews. Author first names as given in the repo citation; verify against the repo's own BibTeX}
}

@inproceedings{he2025pasa,
  author = {He, Yichen and Huang, Guanhua and Feng, Peiyuan and Lin, Yuan and Zhang, Yuchen and Li, Hang and E, Weinan},
  title = {{PaSa}: An {LLM} Agent for Comprehensive Academic Paper Search},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2025},
  pages = {11663--11679},
  doi = {10.18653/v1/2025.acl-long.572}
}

@misc{shi2025spar,
  author = {Shi, Xiaofeng and Li, Yuduo and Kou, Qian and Yu, Longbin and Xie, Jinxin and Zhou, Hua},
  title = {{SPAR}: Scholar Paper Retrieval with {LLM}-based Agents for Enhanced Academic Search},
  year = {2025},
  eprint = {2507.15245},
  archivePrefix = {arXiv}
}

@inproceedings{ajith2024litsearch,
  author = {Ajith, Anirudh and Xia, Mengzhou and Chevalier, Alexis and Goyal, Tanya and Chen, Danqi and Gao, Tianyu},
  title = {{LitSearch}: A Retrieval Benchmark for Scientific Literature Search},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year = {2024},
  pages = {15068--15083},
  doi = {10.18653/v1/2024.emnlp-main.840},
  note = {Author first names other than Ajith/Chevalier from memory; surnames verified via Crossref}
}

@misc{kang2024researcharena,
  author = {Kang, Hao and Xiong, Chenyan},
  title = {{ResearchArena}: Benchmarking Large Language Models' Ability to Collect and Organize Information as Research Agents},
  year = {2024},
  eprint = {2406.10291},
  archivePrefix = {arXiv}
}

@misc{jiang2025deepretrieval,
  author = {Jiang, Pengcheng and others},
  title = {{DeepRetrieval}: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning},
  year = {2025},
  eprint = {2503.00223},
  archivePrefix = {arXiv},
  note = {Full author list and any later venue not verified}
}

@misc{sahu2026rethinking,
  author = {Sahu, Gaurav and Charlin, Laurent and Pal, Christopher},
  title = {Rethinking Literature Search Evaluation: Deep Research Helps, and Human Citation Lists Are Not a Ground Truth},
  year = {2026},
  eprint = {2605.29234},
  archivePrefix = {arXiv}
}

@article{asai2026openscholar,
  author = {Asai, Akari and others and Hajishirzi, Hannaneh},
  title = {Synthesizing scientific literature with retrieval-augmented language models},
  journal = {Nature},
  year = {2026},
  pages = {857--863},
  doi = {10.1038/s41586-025-10072-4},
  note = {28 authors (Asai, He, Shao, Shi, Singh, ... Koh, Hajishirzi); volume not verified. arXiv:2411.14199 (OpenScholar)}
}

@misc{skarlinski2024paperqa2,
  author = {Skarlinski, Michael D. and Cox, Sam and Laurent, Jon M. and Braza, James D. and Hinks, Michaela and Hammerling, Michael J. and Ponnapati, Manvitha and Rodriques, Samuel G. and White, Andrew D.},
  title = {Language agents achieve superhuman synthesis of scientific knowledge},
  year = {2024},
  eprint = {2409.13740},
  archivePrefix = {arXiv}
}

@inproceedings{wang2023chatgptboolean,
  author = {Wang, Shuai and Scells, Harrisen and Koopman, Bevan and Zuccon, Guido},
  title = {Can {ChatGPT} Write a Good {Boolean} Query for Systematic Review Literature Search?},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2023},
  pages = {1426--1436},
  doi = {10.1145/3539618.3591703}
}

@inproceedings{wang2024zeroshot,
  author = {Wang, Shuai and Scells, Harrisen and Zhuang, Shengyao and Potthast, Martin and Koopman, Bevan and Zuccon, Guido},
  title = {Zero-Shot Generative Large Language Models for Systematic Review Screening Automation},
  booktitle = {Advances in Information Retrieval (ECIR 2024)},
  series = {Lecture Notes in Computer Science},
  year = {2024},
  pages = {403--420},
  doi = {10.1007/978-3-031-56027-9_25},
  note = {Shengyao first name from memory}
}

@article{guo2024screening,
  author = {Guo, Eddie and Gupta, Mehul and Deng, Jiawen and Park, Ye-Jean and Paget, Michael and Naugler, Christopher},
  title = {Automated Paper Screening for Clinical Reviews Using Large Language Models: Data Analysis Study},
  journal = {Journal of Medical Internet Research},
  year = {2024},
  pages = {e48996},
  doi = {10.2196/48996},
  note = {Given names from memory; surnames verified. Volume (26) not verified}
}

@article{khraisha2024gpt4,
  author = {Khraisha, Qusai and Put, Sophie and Kappenberg, Johanna and Warraitch, Azza and Hadfield, Kristin},
  title = {Can large language models replace humans in systematic reviews? Evaluating {GPT}-4's efficacy in screening and extracting data from peer-reviewed and grey literature in multiple languages},
  journal = {Research Synthesis Methods},
  year = {2024},
  pages = {616--626},
  doi = {10.1002/jrsm.1715},
  note = {Given names from memory; surnames verified}
}

@article{rajit2025citation,
  author = {Rajit, Darren and Du, Lan and Teede, Helena and Enticott, Joanne},
  title = {Automated citation searching in systematic review production: A simulation study},
  journal = {Research Synthesis Methods},
  year = {2025},
  volume = {16},
  number = {1},
  pages = {211--227},
  doi = {10.1017/rsm.2024.15}
}

@article{wang2025trialmind,
  author = {Wang, Zifeng and Cao, Lang and Danek, Benjamin and Jin, Qiao and Lu, Zhiyong and Sun, Jimeng},
  title = {Accelerating clinical evidence synthesis with large language models},
  journal = {npj Digital Medicine},
  year = {2025},
  volume = {8},
  pages = {509},
  doi = {10.1038/s41746-025-01840-7}
}

@misc{cao2025ottosr,
  author = {Cao, Christian and Arora, Rohit and Cento, Paul and others and Moher, David and Bobrovitz, Niklas},
  title = {Automation of Systematic Reviews with Large Language Models},
  year = {2025},
  howpublished = {medRxiv preprint},
  doi = {10.1101/2025.06.13.25329541},
  note = {36 authors; not peer reviewed}
}

@inproceedings{kusa2023csmed,
  author = {Kusa, Wojciech and Mendoza, Oscar E. and Samwald, Matthias and Knoth, Petr and Hanbury, Allan},
  title = {{CSMeD}: Bridging the Dataset Gap in Automated Citation Screening for Systematic Literature Reviews},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track},
  year = {2023},
  eprint = {2311.12474},
  archivePrefix = {arXiv}
}

@inproceedings{cormack2016engineering,
  author = {Cormack, Gordon V. and Grossman, Maura R.},
  title = {Engineering Quality and Reliability in Technology-Assisted Review},
  booktitle = {Proceedings of the 39th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2016},
  pages = {75--84},
  doi = {10.1145/2911451.2911510}
}

@article{callaghan2020stopping,
  author = {Callaghan, Max W. and M{\"u}ller-Hansen, Finn},
  title = {Statistical stopping criteria for automated screening in systematic reviews},
  journal = {Systematic Reviews},
  year = {2020},
  volume = {9},
  pages = {273},
  doi = {10.1186/s13643-020-01521-4}
}

@article{callaghan2024robust,
  author = {Callaghan, Max and M{\"u}ller-Hansen, Finn and Bond, Melissa and Hamel, Candyce and Devane, Declan and Kusa, Wojciech and O'Mara-Eves, Alison and Spijker, Ren{\'e} and Stevenson, Mark and Stansfield, Claire and Thomas, James and Minx, Jan C.},
  title = {Computer-assisted screening in systematic evidence synthesis requires robust and well-evaluated stopping criteria},
  journal = {Systematic Reviews},
  year = {2024},
  volume = {13},
  pages = {284},
  doi = {10.1186/s13643-024-02699-7},
  note = {Given names from memory; surnames verified}
}

@article{li2020stop,
  author = {Li, Dan and Kanoulas, Evangelos},
  title = {When to Stop Reviewing in Technology-Assisted Reviews: Sampling from an Adaptive Distribution to Estimate Residual Relevant Documents},
  journal = {ACM Transactions on Information Systems},
  year = {2020},
  doi = {10.1145/3411755},
  note = {Subtitle from memory (Crossref title shows main title only); pages 1--36 per Crossref; volume 38(4) not verified}
}

@inproceedings{yang2021heuristic,
  author = {Yang, Eugene and Lewis, David D. and Frieder, Ophir},
  title = {Heuristic Stopping Rules for Technology-Assisted Review},
  booktitle = {Proceedings of the 21st ACM Symposium on Document Engineering},
  year = {2021},
  doi = {10.1145/3469096.3469873}
}

@inproceedings{lewis2021certifying,
  author = {Lewis, David D. and Yang, Eugene and Frieder, Ophir},
  title = {Certifying One-Phase Technology-Assisted Reviews},
  booktitle = {Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
  year = {2021},
  pages = {893--902},
  doi = {10.1145/3459637.3482415}
}

@article{stevenson2023pointprocess,
  author = {Stevenson, Mark and Bin-Hezam, Reem},
  title = {Stopping Methods for Technology-assisted Reviews Based on Point Processes},
  journal = {ACM Transactions on Information Systems},
  year = {2023},
  doi = {10.1145/3631990}
}

@inproceedings{kanoulas2017clef,
  author = {Kanoulas, Evangelos and Li, Dan and Azzopardi, Leif and Spijker, Ren{\'e}},
  title = {{CLEF} 2017 Technologically Assisted Reviews in Empirical Medicine Overview},
  booktitle = {Working Notes of CLEF 2017 -- Conference and Labs of the Evaluation Forum},
  series = {CEUR Workshop Proceedings},
  volume = {1866},
  year = {2017},
  url = {https://ceur-ws.org/Vol-1866/invited_paper_12.pdf}
}

@inproceedings{cormack2018clef,
  author = {Cormack, Gordon V. and Grossman, Maura R.},
  title = {Technology-Assisted Review in Empirical Medicine: {Waterloo} Participation in {CLEF} {eHealth} 2018},
  booktitle = {Working Notes of CLEF 2018 -- Conference and Labs of the Evaluation Forum},
  series = {CEUR Workshop Proceedings},
  volume = {2125},
  year = {2018},
  url = {https://ceur-ws.org/Vol-2125/paper_89.pdf}
}

@inproceedings{zobel1998reliable,
  author = {Zobel, Justin},
  title = {How Reliable Are the Results of Large-Scale Information Retrieval Experiments?},
  booktitle = {Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {1998},
  pages = {307--314},
  doi = {10.1145/290941.291014}
}

@inproceedings{buckley2004bpref,
  author = {Buckley, Chris and Voorhees, Ellen M.},
  title = {Retrieval Evaluation with Incomplete Information},
  booktitle = {Proceedings of the 27th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2004},
  pages = {25--32},
  doi = {10.1145/1008992.1009000}
}

@inproceedings{yilmaz2006infap,
  author = {Yilmaz, Emine and Aslam, Javed A.},
  title = {Estimating Average Precision with Incomplete and Imperfect Judgments},
  booktitle = {Proceedings of the 15th ACM International Conference on Information and Knowledge Management (CIKM '06)},
  year = {2006},
  doi = {10.1145/1183614.1183633}
}

@article{buckley2007bias,
  author = {Buckley, Chris and Dimmick, Darrin and Soboroff, Ian and Voorhees, Ellen},
  title = {Bias and the Limits of Pooling for Large Collections},
  journal = {Information Retrieval},
  year = {2007},
  volume = {10},
  number = {6},
  pages = {491--508},
  doi = {10.1007/s10791-007-9032-x}
}

@inproceedings{sakai2007bpref,
  author = {Sakai, Tetsuya},
  title = {Alternatives to Bpref},
  booktitle = {Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2007},
  pages = {71--78},
  doi = {10.1145/1277741.1277756}
}

@inproceedings{thomas2024llmjudge,
  author = {Thomas, Paul and Spielman, Seth and Craswell, Nick and Mitra, Bhaskar},
  title = {Large Language Models can Accurately Predict Searcher Preferences},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2024},
  pages = {1930--1940},
  doi = {10.1145/3626772.3657707}
}

@inproceedings{faggioli2023perspectives,
  author = {Faggioli, Guglielmo and Dietz, Laura and Clarke, Charles L. A. and Demartini, Gianluca and Hagen, Matthias and Hauff, Claudia and Kando, Noriko and Kanoulas, Evangelos and Potthast, Martin and Stein, Benno and Wachsmuth, Henning},
  title = {Perspectives on Large Language Models for Relevance Judgment},
  booktitle = {Proceedings of the 2023 ACM SIGIR International Conference on Theory of Information Retrieval},
  year = {2023},
  pages = {39--50},
  doi = {10.1145/3578337.3605136}
}

@misc{upadhyay2024umbrela,
  author = {Upadhyay, Shivani and Pradeep, Ronak and Thakur, Nandan and Craswell, Nick and Lin, Jimmy},
  title = {{UMBRELA}: {UMbrela} is the (Open-Source Reproduction of the) {Bing} {RELevance} Assessor},
  year = {2024},
  eprint = {2406.06519},
  archivePrefix = {arXiv}
}

@misc{upadhyay2024patch,
  author = {Upadhyay, Shivani and Kamalloo, Ehsan and Lin, Jimmy},
  title = {{LLMs} Can Patch Up Missing Relevance Judgments in Evaluation},
  year = {2024},
  eprint = {2405.04727},
  archivePrefix = {arXiv}
}

@inproceedings{abbasiantaeb2024holes,
  author = {Abbasiantaeb, Zahra and Meng, Chuan and Azzopardi, Leif and Aliannejadi, Mohammad},
  title = {Can We Use Large Language Models to Fill Relevance Judgment Holes?},
  booktitle = {Joint Proceedings of the SIGIR-AP 2024 Workshops EMTCIR 2024 and UM-CIR 2024},
  series = {CEUR Workshop Proceedings},
  volume = {3854},
  year = {2024},
  url = {https://ceur-ws.org/Vol-3854/emtcir-2.pdf}
}

@misc{gienapp2025topicclassifiers,
  author = {Gienapp, Lukas and Potthast, Martin and Yates, Andrew and Scells, Harrisen and Yang, Eugene},
  title = {Topic-Specific Classifiers are Better Relevance Judges than Prompted {LLMs}},
  year = {2025},
  eprint = {2510.04633},
  archivePrefix = {arXiv}
}

@misc{clarke2024cantreplace,
  author = {Clarke, Charles L. A. and Dietz, Laura},
  title = {{LLM}-based relevance assessment still can't replace human relevance assessment},
  year = {2024},
  eprint = {2412.17156},
  archivePrefix = {arXiv}
}

@inproceedings{alaofi2024fooled,
  author = {Alaofi, Marwah and Thomas, Paul and Scholer, Falk and Sanderson, Mark},
  title = {{LLMs} can be Fooled into Labelling a Document as Relevant: best caf{\'e} near me; this paper is perfectly relevant},
  booktitle = {Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region},
  year = {2024},
  pages = {32--41},
  doi = {10.1145/3673791.3698431}
}

@article{soboroff2025dontuse,
  author = {Soboroff, Ian},
  title = {Don't Use {LLMs} to Make Relevance Judgments},
  journal = {Information Retrieval Research},
  year = {2025},
  volume = {1},
  number = {1},
  pages = {29--46},
  doi = {10.54195/irrj.19625}
}

@article{sampson2006relativerecall,
  author = {Sampson, Margaret and Zhang, Li and Morrison, Andra and Barrowman, Nicholas J. and Clifford, Tammy J. and Platt, Robert W. and Klassen, Terry P. and Moher, David},
  title = {An alternative to the hand searching gold standard: validating methodological search filters using relative recall},
  journal = {BMC Medical Research Methodology},
  year = {2006},
  volume = {6},
  pages = {33},
  doi = {10.1186/1471-2288-6-33},
  note = {Given names from memory; surnames verified via Crossref}
}

@article{waffenschmidt2019single,
  author = {Waffenschmidt, Siw and Knelangen, Marco and Sieben, Wiebke and B{\"u}hn, Stefanie and Pieper, Dawid},
  title = {Single screening versus conventional double screening for study selection in systematic reviews: a methodological systematic review},
  journal = {BMC Medical Research Methodology},
  year = {2019},
  volume = {19},
  pages = {132},
  doi = {10.1186/s12874-019-0782-0}
}

@article{carrier2026aisr,
  author = {Carrier, Robert and Lopez, Leonardo and Moya, Andrew and Hernandez, Victor H. and D'Apuzzo, Michele R. and McNamara, Colin A.},
  title = {Evaluating the Utility of Artificial Intelligence in Conducting Systematic Reviews},
  journal = {Arthroplasty Today},
  year = {2026},
  volume = {39},
  pages = {102035},
  doi = {10.1016/j.artd.2026.102035}
}

@inproceedings{scells2017testcollection,
  author = {Scells, Harrisen and Zuccon, Guido and Koopman, Bevan and Deacon, Anthony and Azzopardi, Leif and Geva, Shlomo},
  title = {A Test Collection for Evaluating Retrieval of Studies for Inclusion in Systematic Reviews},
  booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2017},
  pages = {1237--1240},
  doi = {10.1145/3077136.3080707}
}
```
