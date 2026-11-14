# Anticipated reviewer critiques and what we did (2026-11-13)

Numbers: `analysis/webfp/web_robust.md` (script `scripts/web_robust.py`), `analysis/webfp/env_v2_compare.md`,
`docs/RESULTS.md` (memorization, closed-graph baselines). All live-web tables reproduce offline with
`python scripts/reproduce_web.py results/web`.

## 1. The v2 environment fix was designed and tested on the same 8 reviews
**Done:** held-out test on the 24 transfer reviews (Haiku; never used to design the fix).
Partial: the OpenAlex daily request budget ran out, so 56/72 held-out episodes finished
(citation B=40: 22 reviews, hybrid B=20: 19, hybrid B=40: 15). Fill-in command below.

| v2 - original (recall) | reviews | mean | 95% CI | up/down |
|---|---|---|---|---|
| eval, sonnet citation B=40 | 8 | +.070 | [+.026, +.121] | 6/0 (sign p=.03) |
| eval, haiku citation B=10/20/40 | 8 | +.036/+.047/+.042 | all CIs >= 0 | 3/0, 3/0, 2/0 |
| **held-out** transfer, haiku citation B=40 | 22 | +.000 | [-.028, +.037] | 1/4 |
| **held-out** transfer, haiku hybrid B=20 | 19 | +.026 | [-.004, +.057] | 10/6 |
| **held-out** transfer, haiku hybrid B=40 | 15 | +.044 | [+.001, +.094] | 9/4 |

**How to report it:** the fix does not replicate for Haiku citation mode on held-out reviews; the
eval gains for Haiku come from 2-3 reviews (Meijboom via the seed fallback, Donners, Brouwer).
The Sonnet eval gain is the only result that is significant by sign test, and it was not
tested on held-out reviews (too expensive under the cap). Present v2 as a diagnosed environment
flaw with a fix that helps on the reviews where the dead end occurs, not as a general improvement.

## 2. Thin evidence: 8 reviews, single runs, no CIs
**Done:** every live-web comparison now has a bootstrap CI over reviews and an exact sign test
(`web_robust.md` section 1). Prefer the 24-review transfer set for headline claims. Run-to-run
noise is already sized in `docs/RESULTS.md` (Haiku about +-.05 between repeats; Sonnet within .02).

## 3. Judge calibrated on a different population; abstract-level only
**Done:** sensitivity analysis and bootstrap CI (`web_robust.md` section 2).
- Gemini: eligible share .33, 95% CI [.27, .39] from calibration sampling; .19-.45 if specificity
  on never-screened records were anywhere from .65 to .90. GPT-5.6 Luna: .29 [.22, .36].
- **Wording:** say "abstract-level eligible" everywhere; full-text eligibility is lower.
- **Still needed:** the 30 hand labels (`analysis/webfp/handlabel.html`). With n=30 a 95% CI is
  about +-.17, so they check direction, not the exact share.

## 4. "Budget" ignores LLM cost
**Done** (`web_robust.md` section 3, from per-call usage in the response cache, 100% matched):

| B=40 | $/episode | gold found/episode | $/gold found | wall s/episode |
|---|---|---|---|---|
| eval, sonnet hybrid | .234 | 10.5 | .022 | 99 |
| eval, haiku hybrid | .006 | 3.8 | .002 | 38 |
| transfer, haiku hybrid | .007 | 9.0 | .001 | 51 |
| OpenAlex search / snowball | 0 | 0.1-0.7 | - | 1-3 |

## 5. Weak web baselines
**Done:** added the tuned scripted snowball agent (no LLM; automated citation chasing) on the web
for both splits. It reaches ~.01 recall, like raw OpenAlex search. LLM agents beat both on every
review that differs: eval sonnet hybrid - snowball +.316 [+.196, +.447] (8/0); transfer haiku
hybrid - snowball +.087 [+.060, +.114] (22/0). Closed-graph BM25, SPECTER2, snowball and GRPO
baselines with CIs are already in `docs/RESULTS.md`. Cite Rajit et al. (2025, 35.8% median recall
from automated citation searching) as an external reference, not a like-for-like comparison.

## 6. Memorization
Already done (`docs/RESULTS.md`, "Robustness"): closed-book probe; removing every gold study any
model names from memory changes no conclusion (each system loses at most .02 recall).

## 7. Recall of .1-.35 is far from usable
**Reframe** the agent as a supplement to the review's own search. `web_robust.md` section 4: per
B=40 episode, Sonnet hybrid includes 10.5 gold plus about 7.7 abstract-level-eligible studies the
review's search never retrieved (+73%); Haiku hybrid on transfer 9.0 gold plus about 12.5 (+139%).
Caveat: abstract-level and judge-estimated (see 3).

## 8. Live web is not reproducible
**Done:** README quick start verified on a fresh clone (Python 3.15 venv, `pip install -e .`,
build/audit/run/demo/report, 57 tests OK). Compact live-web records ship in `results/web/`, and
`scripts/reproduce_web.py` recomputes the web tables offline (stdlib only).

## 9. Wording and consistency fixes for the report
- Error-analysis shares are pooled over gold studies (micro); recall tables average per review
  (macro). Label each table.
- Pooled recall uses a pool built only from our systems; say so next to the leave-one-agent-out result.
- "One citation hop" uses the corpus citation graph, not the live graph.
- Do **not** claim the agents "underuse" citation chasing: forcing expansion never helps and hurts
  Haiku (`docs/RESULTS.md`, "Forcing citation chasing"). The defensible claim: misses are search
  coverage failures; screening rejects at most 5% of gold; in citation-only mode agents stop
  early, partly because of a dead end in the environment (excluded papers were not offered for
  expansion; one review had no seeds).

## Fill in the held-out runs (after the OpenAlex budget resets at 00:00 UTC)
```bash
ssh ... 'cd ~/literature-search-agent && MATRIX_CONFIG=configs/web/transfer_web_v2_haiku.json WORKERS=3 \
  sbatch --exclude=barley-03,barley-04 --time=04:00:00 --export=ALL farmshare/api_matrix.sbatch'
```
then rerun `scripts/web_robust.py` and `scripts/export_web_results.py` and update section 1.
