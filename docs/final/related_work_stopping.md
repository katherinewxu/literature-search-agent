# New related work (verified 2026-11-23)

Every entry was checked against arXiv API metadata or Crossref DOI records (title, authors, year). Venues are marked "arXiv only" unless arXiv comments or Crossref confirm them.

## 1. Budget forcing / forcing continuation

```bibtex
@article{muennighoff2025s1,
  title={s1: Simple test-time scaling},
  author={Muennighoff, Niklas and Yang, Zitong and Shi, Weijia and Li, Xiang Lisa and Fei-Fei, Li and Hajishirzi, Hannaneh and Zettlemoyer, Luke and Liang, Percy and Cand{\`e}s, Emmanuel and Hashimoto, Tatsunori},
  journal={arXiv preprint arXiv:2501.19393}, year={2025}}
```
https://arxiv.org/abs/2501.19393 (venue UNVERIFIED: possibly EMNLP 2025, not confirmed)
Relation: budget forcing suppresses end-of-thinking and appends "Wait". Our no-stop intervention is the same move applied to an agent's STOP action, and it recovers recall.

```bibtex
@article{shen2025thinking,
  title={Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction},
  author={Shen, Junhong and Bai, Hao and Zhang, Lunjun and Zhou, Yifei and Setlur, Amrith and Tong, Shengbang and Caples, Diego and Jiang, Nan and Zhang, Tong and Talwalkar, Ameet and Kumar, Aviral},
  journal={arXiv preprint arXiv:2506.07976}, year={2025}}
```
https://arxiv.org/abs/2506.07976 (arXiv only)
Relation: shows that prompting web agents to interact longer, with no training, improves success. This is the agentic analogue of budget forcing and parallels our finding that the interaction horizon, not per-step reasoning, is what binds.

## 2. Simple non-agentic pipelines matching agents

```bibtex
@article{xia2024agentless,
  title={Agentless: Demystifying LLM-based Software Engineering Agents},
  author={Xia, Chunqiu Steven and Deng, Yinlin and Dunn, Soren and Zhang, Lingming},
  journal={arXiv preprint arXiv:2407.01489}, year={2024}}
```
https://arxiv.org/abs/2407.01489 (venue UNVERIFIED: believed FSE 2025)
Relation: a fixed localize-repair-validate pipeline matched agentic SWE systems. Our fixed 8-query round-robin list plays the same role for literature search.

```bibtex
@article{kapoor2024agents,
  title={AI Agents That Matter},
  author={Kapoor, Sayash and Stroebl, Benedikt and Siegel, Zachary S. and Nadgir, Nitya and Narayanan, Arvind},
  journal={arXiv preprint arXiv:2407.01502}, year={2024}}
```
https://arxiv.org/abs/2407.01502 (arXiv only)
Relation: argues that agents must be compared with simple, cost-matched baselines. Our equal-budget static baseline follows that prescription, and the agent loses to it.

## 3. Early stopping and budget use in tool/search agents

```bibtex
@article{liu2025budgetaware,
  title={Budget-Aware Tool Use Enables Effective Agent Scaling},
  author={Liu, Tengxiao and Wang, Zifeng and Miao, Jin and Hsu, I-Hung and Yan, Jun and Chen, Jiefeng and Han, Rujun and Xu, Fangyuan and Chen, Yanfei and Jiang, Ke and Daruki, Samira and Liang, Yi and Wang, William Yang and Pfister, Tomas and Lee, Chen-Yu},
  journal={arXiv preprint arXiv:2511.17006}, year={2025}, note={COLM 2026}}
```
https://arxiv.org/abs/2511.17006 (COLM 2026 per arXiv comment)
Relation: finds that ReAct agents plateau when given more tool calls unless the agent is told how much budget remains (Budget Tracker). This matches our observation that the agent leaves budget unspent and that the "N budget left" re-prompt helps.

```bibtex
@article{yen2025lost,
  title={Lost in the Maze: Overcoming Context Limitations in Long-Horizon Agentic Search},
  author={Yen, Howard and Lee, Yoonsang and Paranjape, Ashwin and Xia, Mengzhou and Venkatesh, Thejas and Hessel, Jack and Chen, Danqi and Zhang, Yuhao},
  journal={arXiv preprint arXiv:2510.18939}, year={2025}, note={COLM 2026}}
```
https://arxiv.org/abs/2510.18939 (COLM 2026 per arXiv comment)
Relation: its error taxonomy of deep-research agents names "stop early" as a major failure mode alongside context and budget exhaustion.

```bibtex
@article{zhang2026judged,
  title={Judged Useless, Queried Anyway: Tool-Using Agents Rarely Turn Their Own Evidence Judgments into Stopping Decisions},
  author={Zhang, Chubin and Wan, Zhenglin and Yu, Xingrui and Wu, Jingxuan and Zhou, Yaxin and Tsang, Ivor and An, Bo},
  journal={arXiv preprint arXiv:2610.06191}, year={2026}}
```
https://arxiv.org/abs/2610.06191 (arXiv only)
Relation: shows that agent stopping is decoupled from the agent's own evidence judgments and that only a harness-enforced stopping rule fixes it. This supports treating STOP as a controller decision rather than an LLM decision.

Optional extra: Zhang et al. 2026, "To Search or Not to Search" (under-search vs over-search in deep search agents), https://arxiv.org/abs/2602.03304 (verified, WWW 2026 per repo name only).

## 4. Searcher stopping behaviour / foraging

```bibtex
@inproceedings{maxwell2015searching,
  title={Searching and Stopping: An Analysis of Stopping Rules and Strategies},
  author={Maxwell, David and Azzopardi, Leif and J{\"a}rvelin, Kalervo and Keskustalo, Heikki},
  booktitle={Proceedings of the 24th ACM International Conference on Information and Knowledge Management (CIKM)},
  pages={313--322}, year={2015}, doi={10.1145/2806416.2806476}}
```
https://doi.org/10.1145/2806416.2806476
Relation: simulated searchers using frustration/disgust-point stopping rules did best. Our agent behaves like a simulated searcher whose session-level stopping rule is too conservative for a recall-oriented task.

```bibtex
@article{pirolli1999foraging,
  title={Information Foraging},
  author={Pirolli, Peter and Card, Stuart},
  journal={Psychological Review}, volume={106}, number={4}, pages={643--675}, year={1999},
  doi={10.1037/0033-295X.106.4.643}}
```
https://doi.org/10.1037/0033-295X.106.4.643
Relation: in foraging theory, a forager leaves a patch when the marginal gain rate falls below the average rate. The agent's early STOP is a leave-the-environment decision made while the marginal yield is still positive.

## 5. LLMs for systematic-review search (2024-2026, not already cited)

```bibtex
@inproceedings{wang2025reassessing,
  title={Reassessing Large Language Model Boolean Query Generation for Systematic Reviews},
  author={Wang, Shuai and Scells, Harrisen and Koopman, Bevan and Zuccon, Guido},
  booktitle={SIGIR}, year={2025}, note={arXiv:2505.07155}}
```
https://arxiv.org/abs/2505.07155 (SIGIR 2025 per arXiv comment)
Relation: LLM-generated queries get higher precision but lower recall than baselines. This is the same precision-over-recall bias we see in our agent's stricter screening.

```bibtex
@article{rajit2025citation,
  title={Automated citation searching in systematic review production: A simulation study},
  author={Rajit, Darren and Du, Lan and Teede, Helena and Enticott, Joanne},
  journal={Research Synthesis Methods}, volume={16}, pages={211--227}, year={2025},
  doi={10.1017/rsm.2024.15}}
```
https://doi.org/10.1017/rsm.2024.15
Relation: automated citation chasing has lower recall than standard searches and works best as a supplement. This bears directly on our agent's citation-expansion action.

```bibtex
@article{padarha2026agentslr,
  title={Evaluating AI-based Scientific Knowledge Synthesis with Epidemiological Systematic Reviews},
  author={Padarha, Shreyansh and Kearns, Ryan Othniel and Naidoo, Tristan and others},
  journal={arXiv preprint arXiv:2603.22327}, year={2026}}
```
https://arxiv.org/abs/2603.22327 (arXiv only; 16 authors, see the page for the full list)
Relation: an AgentSLR harness that evaluates frontier models stage by stage on SLRs. It is complementary because it does not study adaptive budgeted search or stopping.

I could not find any 2024-2026 paper that directly compares an adaptive LLM search agent with a static or Boolean query list on SR recall at a matched budget. That gap supports our novelty claim, but it rests on a non-exhaustive search.

## Suggested positioning paragraph (~150 words)

Technology-assisted review has long treated *when to stop* as a first-class problem, separate from *what to retrieve*. Knee and target methods, along with statistically certified stopping rules [Cormack & Grossman 2016; Callaghan & Müller-Hansen 2020; Lewis 2021; Stevenson 2023], exist because recall-oriented search fails mainly by quitting too soon. LLM literature-search agents fold this decision into the policy: the model itself chooses STOP. We find that it chooses STOP badly. Agents abandon 30-50% of their budget, much like a simulated searcher whose stopping rule is tuned for precision [Maxwell et al. 2015; Pirolli & Card 1999]. Rejecting STOP while budget remains, the agentic analogue of s1's budget forcing [Muennighoff et al. 2025; cf. Shen et al. 2025; Liu et al. 2025], recovers roughly half of the recall gap. Even a model that spends its whole budget only ties a static 8-query list. That echoes Agentless-style results [Xia et al. 2024; Kapoor et al. 2024]: in this setting, adaptivity adds little beyond a good stopping rule.
