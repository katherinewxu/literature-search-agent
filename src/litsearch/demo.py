"""Self-contained HTML replay of one episode (no server, CDN, or external fonts).

``build_html_replay(run)`` works from the run file alone (paper ids only). Passing ``papers``
(id -> {"title", "year", "gold"}) adds titles and gold markers, and ``estimate`` (one row of
``analysis/*/recall_estimates.jsonl``) adds the agent's end-of-episode recall estimate next to the
true recall. Gold markers are evaluator information shown to the viewer, never to the agent.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

UNIVERSE_LABEL = {
    "pool": "closed pool (the review's own candidate papers)",
    "open": "open graph (pool + cited works)",
    "openalex": "live OpenAlex (~250M works)",
}

CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--ink:#16202a;--ink2:#566170;--rule:#dde1e6;--accent:#2a78d6;
--gold:#0f8a5f;--goldbg:#e3f4ec;--miss:#b4bcc6;--est:#eb6834;--track:#e9edf1}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#181c21;--ink:#e6e9ed;--ink2:#9aa5b1;
--rule:#2a3038;--accent:#5a9be6;--gold:#3cc48f;--goldbg:#173328;--miss:#56606b;--est:#f08a5d;--track:#262c33}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:18px max(16px,3vw);border-bottom:1px solid var(--rule);background:var(--panel)}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:600}
h1{font-size:22px;margin:4px 0 2px;line-height:1.25;text-wrap:balance}
.sub{color:var(--ink2);font-size:14px}
main{display:grid;grid-template-columns:minmax(280px,360px) minmax(0,1fr);gap:16px;padding:16px max(16px,3vw);max-width:1400px;margin:0 auto}
@media(max-width:860px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px;min-width:0}
h2{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin:0 0 10px;font-weight:600}
.section+.section{margin-top:18px;padding-top:16px;border-top:1px solid var(--rule)}
details summary{cursor:pointer;color:var(--ink2);font-size:13px}
.criteria{font-size:13px;color:var(--ink2);max-height:180px;overflow:auto;margin-top:6px}
.meter{height:10px;background:var(--track);border-radius:5px;overflow:hidden;margin:6px 0 4px}
.meter>div{height:100%;background:var(--accent);transition:width .2s}
.kv{display:flex;justify-content:space-between;gap:8px;font-variant-numeric:tabular-nums;padding:3px 0}
.kv b{font-weight:600}
.readout{display:grid;gap:8px}
.big{font-size:30px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.1}
.tile{border:1px solid var(--rule);border-radius:8px;padding:10px 12px}
.tile .lab{font-size:12px;color:var(--ink2)}
.tile.est .big{color:var(--est)}
.tile.true .big{color:var(--gold)}
.note{font-size:12px;color:var(--ink2)}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{font:inherit;font-size:14px;border:1px solid var(--rule);background:var(--panel);color:var(--ink);padding:6px 12px;border-radius:7px;cursor:pointer}
button:hover,button:focus-visible{border-color:var(--accent);outline:none}
input[type=range]{width:100%;accent-color:var(--accent)}
#curve{width:100%;height:auto;display:block}
.steps{display:flex;flex-direction:column;gap:8px;margin-top:12px;max-height:62vh;overflow:auto;padding-right:4px}
.step{border:1px solid var(--rule);border-left:3px solid var(--accent);border-radius:8px;padding:8px 10px}
.step.current{box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 35%,transparent)}
.step.kind-expand_refs,.step.kind-expand_citers{border-left-color:#8a63d2}
.step.kind-read{border-left-color:var(--ink2)}.step.kind-stop{border-left-color:var(--est)}
.step .head{display:flex;justify-content:space-between;gap:8px;font-size:14px}
.step .head .cost{color:var(--ink2);font-variant-numeric:tabular-nums;white-space:nowrap}
.q{font-family:ui-monospace,Menlo,monospace;font-size:13px;background:var(--track);padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
ul.papers{list-style:none;margin:6px 0 0;padding:0;font-size:13px}
ul.papers li{display:flex;gap:6px;align-items:baseline;padding:1px 0;color:var(--ink2)}
ul.papers li .dot{flex:0 0 auto;width:8px;height:8px;border-radius:50%;background:var(--miss);transform:translateY(-1px)}
ul.papers li.gold{color:var(--ink)}ul.papers li.gold .dot{background:var(--gold)}
.tag{font-size:11px;border-radius:4px;padding:0 5px;white-space:nowrap}
.tag.inc{background:var(--goldbg);color:var(--gold);font-weight:600}
.tag.fp{background:var(--track);color:var(--ink2)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin-top:6px}
.legend span{display:inline-flex;gap:5px;align-items:center}
.legend i{width:9px;height:9px;border-radius:50%;display:inline-block}
"""

SCRIPT = r"""
const D=JSON.parse(document.getElementById('run-data').textContent);
const run=D.run, P=D.papers||{}, E=D.estimate, trace=(run.trace||[]).filter(e=>e.action&&e.action.kind!=='seed');
const budget=run.episode.budget, gold=new Set(Object.keys(P).filter(k=>P[k].gold)), nGold=D.gold_count||0;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const title=id=>P[id]?`${esc(P[id].title)}${P[id].year?` <span class="note">(${P[id].year})</span>`:''}`:esc(id);
// charged actions become steps; screening events fold into the step that follows them
const steps=[];let pend=[];
for(const e of trace){if(['include','exclude'].includes(e.action.kind)){pend.push(e);continue}
 steps.push({e,screen:pend});pend=[];}
if(pend.length&&steps.length)steps.at(-1).screen.push(...pend);
const seeds=run.seed_ids||[];
const slider=document.getElementById('slider');slider.max=steps.length;slider.value=steps.length;
function stateAt(n){let spent=run.seed_cost||0, inc=new Set(), rec=[[spent,0]], seen=new Set(seeds);
 for(const s of steps.slice(0,n)){spent+=s.e.charged_cost||0;inc=new Set(s.e.included_ids||[]);
  for(const x of s.screen)inc=new Set(x.included_ids||[]);(s.e.result_ids||[]).forEach(i=>seen.add(i));
  rec.push([spent,nGold?[...inc].filter(i=>gold.has(i)).length/nGold:(s.e.recall_after||0)])}
 return {spent,inc,rec,seen}}
function curve(st){const W=760,H=200,L=40,B=26,T=12,x=c=>L+(W-L-10)*c/budget,y=r=>H-B-(H-B-T)*r;
 let g=`<line x1="${L}" y1="${y(0)}" x2="${W-10}" y2="${y(0)}" stroke="var(--rule)"/>`;
 for(const v of [0.25,0.5,0.75,1]){g+=`<line x1="${L}" y1="${y(v)}" x2="${W-10}" y2="${y(v)}" stroke="var(--rule)" stroke-dasharray="2 4"/><text x="${L-6}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="var(--ink2)">${v.toFixed(2)}</text>`}
 for(const c of [0,budget/2,budget])g+=`<text x="${x(c)}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--ink2)">${c}</text>`;
 g+=`<text x="${W-10}" y="${H-8}" text-anchor="end" font-size="11" fill="var(--ink2)">charged calls</text>`;
 if(E&&+slider.value===steps.length){
  if(E.lower!=null)g+=`<rect x="${L}" y="${y(Math.min(1,E.hat??E.lower))}" width="${W-10-L}" height="${Math.max(0,y(E.lower)-y(Math.min(1,E.hat??E.lower)))}" fill="var(--est)" opacity=".10"/>`;
  for(const [v,lab,dash] of [[E.yield,'yield-curve estimate','6 4'],[E.hat,'sampled estimate',''],[E.lower,'95% lower bound','2 3']])
   if(v!=null)g+=`<line x1="${L}" y1="${y(Math.min(1,v))}" x2="${W-10}" y2="${y(Math.min(1,v))}" stroke="var(--est)" stroke-width="1.5" stroke-dasharray="${dash}"/><text x="${L+6}" y="${y(Math.min(1,v))-4}" font-size="11" fill="var(--est)">${lab} ${v.toFixed(2)}</text>`}
 const pts=st.rec.map(([c,r])=>`${x(c)},${y(r)}`).join(' ');
 g+=`<polyline points="${pts}" fill="none" stroke="var(--gold)" stroke-width="2.5" stroke-linejoin="round"/>`;
 const last=st.rec.at(-1);g+=`<circle cx="${x(last[0])}" cy="${y(last[1])}" r="4.5" fill="var(--gold)" stroke="var(--panel)" stroke-width="2"/>`;
 document.getElementById('curve').innerHTML=g}
function paperList(ids,inc,limit=10){const shown=ids.slice(0,limit);
 return `<ul class="papers">${shown.map(id=>`<li class="${gold.has(id)?'gold':''}"><span class="dot"></span><span>${title(id)} ${inc.has(id)?(gold.has(id)?'<span class="tag inc">included ✓</span>':'<span class="tag fp">included, not gold</span>'):''}</span></li>`).join('')}${ids.length>limit?`<li class="note">… ${ids.length-limit} more</li>`:''}</ul>`}
function describe(s){const a=s.e.action,k=a.kind;
 if(k==='search')return `search <span class="q">${esc(a.query)}</span>`;
 if(k==='expand_refs')return `references of “${P[a.paper_id]?esc(P[a.paper_id].title):esc(a.paper_id)}”`;
 if(k==='expand_citers')return `papers citing “${P[a.paper_id]?esc(P[a.paper_id].title):esc(a.paper_id)}”`;
 if(k==='read')return `read abstract of “${P[a.paper_id]?esc(P[a.paper_id].title):esc(a.paper_id)}”`;
 if(k==='stop')return 'stop';if(k==='noop')return 'invalid action (charged)';return esc(k)}
function render(){const n=+slider.value,st=stateAt(n);
 document.getElementById('stepno').textContent=`Step ${n} of ${steps.length}`;
 document.getElementById('budgetbar').style.width=`${100*Math.min(1,st.spent/budget)}%`;
 document.getElementById('budgettxt').textContent=`${st.spent} of ${budget} calls spent`;
 const tp=[...st.inc].filter(i=>gold.has(i)).length;
 document.getElementById('found').innerHTML=`<div class="kv"><span>Papers seen</span><b>${st.seen.size}</b></div><div class="kv"><span>Included</span><b>${st.inc.size}</b></div><div class="kv"><span>Included that are gold</span><b>${tp} of ${nGold}</b></div>`;
 document.getElementById('truerec').textContent=nGold?(tp/nGold).toFixed(2):'–';
 const estEl=document.getElementById('estrec');
 if(estEl){const done=n===steps.length;estEl.textContent=done&&E&&E.hat!=null?Math.min(1,E.hat).toFixed(2):'…';
  document.getElementById('estnote').textContent=done?(E&&E.lower!=null?`95% lower bound ${E.lower.toFixed(2)} · yield curve ${E.yield!=null?Math.min(1,E.yield).toFixed(2):'–'}`:''):'computed when the agent stops'}
 curve(st);
 const list=document.getElementById('steps');
 list.innerHTML=(n===0?'':'')+steps.slice(0,n).map((s,i)=>{const inc=new Set(s.screen.length?s.screen.at(-1).included_ids:s.e.included_ids);
  const added=s.screen.filter(x=>x.action.kind==='include').map(x=>x.action.paper_id);
  return `<div class="step kind-${esc(s.e.action.kind)} ${i===n-1?'current':''}"><div class="head"><span><b>${i+1}.</b> ${describe(s)}</span><span class="cost">cost ${s.e.charged_cost} · ${s.e.budget_after} left</span></div>
  ${added.length?`<div class="note" style="margin-top:4px">screened in from earlier results:</div>${paperList(added,inc,6)}`:''}
  ${(s.e.result_ids||[]).length?`<div class="note" style="margin-top:4px">${s.e.result_ids.length} new papers returned:</div>${paperList(s.e.result_ids,new Set(),6)}`:''}
  ${s.e.error?`<div class="note">${esc(s.e.error.message)}</div>`:''}</div>`}).reverse().join('')||'<p class="note">Seeds shown; press Next or Play.</p>'}
let timer=null;
const play=document.getElementById('play');
play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='Play';return}
 if(+slider.value>=steps.length)slider.value=0;play.textContent='Pause';
 timer=setInterval(()=>{if(+slider.value>=steps.length){clearInterval(timer);timer=null;play.textContent='Play';return}slider.value=+slider.value+1;render()},900)};
slider.oninput=render;
document.getElementById('prev').onclick=()=>{slider.value=Math.max(0,+slider.value-1);render()};
document.getElementById('next').onclick=()=>{slider.value=Math.min(steps.length,+slider.value+1);render()};
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')document.getElementById('next').click();if(e.key==='ArrowLeft')document.getElementById('prev').click();if(e.key===' '&&e.target===document.body){e.preventDefault();play.click()}});
render();
"""


def build_html_replay(
    run: dict[str, Any],
    *,
    papers: dict[str, dict[str, Any]] | None = None,
    estimate: dict[str, Any] | None = None,
    gold_count: int | None = None,
    question: str | None = None,
    criteria: str | None = None,
    label: str | None = None,
) -> str:
    episode = run.get("episode", {})
    universe = (run.get("matrix") or {}).get("universe") or (run.get("matrix") or {}).get("index_scope", "pool")
    if "open" in str((run.get("matrix") or {}).get("data_dir", "")) and universe == "pool":
        universe = "open"
    heading = label or f"{run.get('agent', 'agent')} · {episode.get('review_id', 'episode')}"
    sub = (
        f"{run.get('agent')} · {episode.get('mode')} tools · budget {episode.get('budget')} calls · "
        f"{UNIVERSE_LABEL.get(universe, universe)}"
    )
    est = None
    if estimate:
        est = {"hat": estimate.get("llm_hat"), "lower": estimate.get("llm_lower"), "yield": estimate.get("saturation")}
    data = {"run": run, "papers": papers or {}, "estimate": est, "gold_count": gold_count or 0}
    payload = json.dumps(data).replace("</", "<\\/")
    q = html.escape(question or episode.get("question", ""))
    c = html.escape(criteria or episode.get("eligibility_criteria", ""))
    est_tile = (
        """<div class="tile est"><div class="lab">Agent's own estimate (no gold labels)</div>
<div class="big" id="estrec">…</div><div class="note" id="estnote"></div></div>"""
        if est
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(heading)}</title><style>{CSS}</style></head><body>
<header><div class="eyebrow">Literature-search replay · budgeted agent</div>
<h1>{q}</h1><div class="sub">{html.escape(sub)}</div></header>
<main>
<aside class="panel">
 <div class="section"><h2>Eligibility criteria</h2><details><summary>Show criteria</summary><div class="criteria">{c}</div></details></div>
 <div class="section"><h2>Budget</h2><div class="meter"><div id="budgetbar"></div></div><div class="note" id="budgettxt"></div></div>
 <div class="section"><h2>Recall</h2><div class="readout">
  <div class="tile true"><div class="lab">True recall (gold studies included)</div><div class="big" id="truerec">–</div></div>
  {est_tile}
 </div><div id="found" style="margin-top:10px"></div></div>
 <div class="section"><h2>Playback</h2><input id="slider" type="range" min="0" aria-label="step">
  <div class="controls" style="margin-top:8px"><button id="prev">‹ Prev</button><button id="play">Play</button><button id="next">Next ›</button><span class="note" id="stepno"></span></div>
  <div class="note" style="margin-top:6px">Keys: ← → step, space play.</div></div>
</aside>
<section class="panel">
 <h2>Recall over charged calls</h2><svg id="curve" viewBox="0 0 760 200" role="img" aria-label="recall over cost"></svg>
 <div class="legend"><span><i style="background:var(--gold)"></i>true recall</span>{'<span><i style="background:var(--est)"></i>agent estimate at stop</span>' if est else ''}<span><i style="background:var(--gold)"></i>gold paper</span><span><i style="background:var(--miss)"></i>not gold</span></div>
 <div class="steps" id="steps"></div>
</section></main>
<script type="application/json" id="run-data">{payload}</script>
<script>{SCRIPT}</script></body></html>"""


def write_html_replay(
    run_path: str | Path,
    output_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    estimates: str | Path | None = None,
    label: str | None = None,
    openalex_cache: str | Path | None = None,
) -> Path:
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    papers: dict[str, dict[str, Any]] | None = None
    gold_count = None
    if data_dir:
        from .store import CorpusStore

        store = CorpusStore.load(data_dir)
        review = store.review(run["episode"]["review_id"])
        ids = set(run.get("seed_ids") or []) | set(run.get("predictions") or [])
        for event in run.get("trace") or []:
            ids.update(event.get("result_ids") or [])
            if event.get("action", {}).get("paper_id"):
                ids.add(event["action"]["paper_id"])
        papers = {}
        missing = []
        for paper_id in ids:
            paper = store.papers.get(paper_id)
            if paper:
                papers[paper_id] = {"title": paper.title, "year": paper.publication_year, "gold": paper_id in review.gold_ids}
            else:
                missing.append(paper_id)
        if missing and openalex_cache:  # live-web runs: titles of works outside the frozen store
            from .openalex import OpenAlexClient

            client = OpenAlexClient(openalex_cache, delay_seconds=0.12)
            for start in range(0, len(missing), 50):
                for work in client.works_batch(missing[start : start + 50], select="id,display_name,publication_year"):
                    wid = str(work.get("id", "")).rsplit("/", 1)[-1]
                    papers[wid] = {"title": work.get("display_name") or wid, "year": work.get("publication_year"), "gold": False}
        gold_count = len(review.gold_ids)
    estimate = None
    if estimates:
        for line in Path(estimates).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["agent"] == run["agent"] and row["review"] == run["episode"]["review_id"]:
                estimate = row
                break
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_html_replay(run, papers=papers, estimate=estimate, gold_count=gold_count, label=label), encoding="utf-8"
    )
    return output
