from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def build_html_replay(run: dict[str, Any]) -> str:
    payload = json.dumps(run).replace("</", "<\\/")
    title = html.escape(
        f"{run.get('agent', 'agent')} · {run.get('episode', {}).get('review_id', 'episode')}"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title>
<style>
:root{{--ink:#17212b;--muted:#667085;--line:#d0d5dd;--accent:#175cd3;--good:#067647;--bg:#f8fafc}}
*{{box-sizing:border-box}}body{{margin:0;font:15px system-ui;background:var(--bg);color:var(--ink)}}
header{{padding:20px 28px;background:#fff;border-bottom:1px solid var(--line)}}
main{{display:grid;grid-template-columns:300px 1fr;gap:18px;padding:18px;max-width:1200px;margin:auto}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}}
.metric{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #eee}}
#timeline{{max-height:360px;overflow:auto}}#graph,#curve{{width:100%;background:#fbfdff;border:1px solid #e4e7ec;border-radius:8px}}
button{{border:1px solid var(--line);background:white;padding:8px 12px;border-radius:8px}}
button:hover{{border-color:var(--accent)}}input{{width:100%}}.step{{padding:10px;border-left:3px solid var(--accent);margin:8px 0;background:#f8fafc}}
.papers{{font-size:13px;color:var(--muted)}}@media(max-width:760px){{main{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1 style="margin:0 0 4px">Literature-search replay</h1><div>{title}</div></header>
<main><aside class="card"><h2>Outcome</h2><div id="metrics"></div><h2>Playback</h2>
<input id="slider" type="range" min="0" value="0"><p><button id="prev">Previous</button> <button id="next">Next</button></p>
<div id="status"></div></aside><section class="card"><h2>Traversal graph</h2><svg id="graph" viewBox="0 0 800 330"></svg>
<h2>Recall over charged cost</h2><svg id="curve" viewBox="0 0 800 180"></svg>
<h2>Action timeline</h2><div id="timeline"></div></section></main>
<script type="application/json" id="run-data">{payload}</script>
<script>
const run=JSON.parse(document.getElementById('run-data').textContent), trace=run.trace||[];
const slider=document.getElementById('slider');slider.max=trace.length;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
document.getElementById('metrics').innerHTML=Object.entries(run.metrics||{{}}).map(([k,v])=>`<div class="metric"><span>${{esc(k)}}</span><b>${{typeof v==='number'?v.toFixed(3):esc(v)}}</b></div>`).join('');
function graph(rows){{const ids=[...new Set([...(run.seed_ids||[]),...rows.flatMap(e=>e.result_ids||[]),...rows.flatMap(e=>e.included_ids||[])])], included=new Set(rows.at(-1)?.included_ids||[]), pos={{}};
ids.forEach((id,i)=>{{const a=2*Math.PI*i/Math.max(ids.length,1);pos[id]=[400+260*Math.cos(a),165+125*Math.sin(a)]}});
const edges=rows.filter(e=>['expand_refs','expand_citers'].includes(e.action.kind)).flatMap(e=>(e.result_ids||[]).map(x=>[e.action.paper_id,x]));
document.getElementById('graph').innerHTML=edges.filter(e=>pos[e[0]]&&pos[e[1]]).map(e=>`<line x1="${{pos[e[0]][0]}}" y1="${{pos[e[0]][1]}}" x2="${{pos[e[1]][0]}}" y2="${{pos[e[1]][1]}}" stroke="#d0d5dd"/>`).join('')+ids.map(id=>`<g><circle cx="${{pos[id][0]}}" cy="${{pos[id][1]}}" r="${{included.has(id)?10:7}}" fill="${{included.has(id)?'#067647':'#175cd3'}}"/><text x="${{pos[id][0]+11}}" y="${{pos[id][1]+4}}" font-size="11">${{esc(id)}}</text></g>`).join('')}}
function curve(rows){{let cost=0, pts=[[30,150]];rows.forEach(e=>{{cost+=e.charged_cost;pts.push([30+740*cost/Math.max(1,run.episode.budget),150-125*(e.recall_after||0)])}});document.getElementById('curve').innerHTML=`<line x1="30" y1="150" x2="770" y2="150" stroke="#667085"/><line x1="30" y1="20" x2="30" y2="150" stroke="#667085"/><polyline points="${{pts.map(p=>p.join(',')).join(' ')}}" fill="none" stroke="#175cd3" stroke-width="3"/><text x="380" y="173" font-size="12">charged cost</text><text x="4" y="18" font-size="12">1.0</text>`}}
function render(){{const n=Number(slider.value), rows=trace.slice(0,n);document.getElementById('status').textContent=`Step ${{n}} / ${{trace.length}}`;graph(rows);curve(rows);
document.getElementById('timeline').innerHTML=rows.map(e=>`<div class="step"><b>${{e.step}}. ${{esc(e.action.kind)}}</b> · cost ${{e.charged_cost}} · budget ${{e.budget_after}}<div class="papers">${{esc((e.result_ids||[]).join(', '))}}</div>${{e.error?`<div>${{esc(e.error.message)}}</div>`:''}}</div>`).join('')||'<p>No actions yet.</p>';}}
slider.oninput=render;document.getElementById('prev').onclick=()=>{{slider.value=Math.max(0,+slider.value-1);render()}};document.getElementById('next').onclick=()=>{{slider.value=Math.min(trace.length,+slider.value+1);render()}};render();
</script></body></html>"""


def write_html_replay(run_path: str | Path, output_path: str | Path) -> Path:
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html_replay(run), encoding="utf-8")
    return output

