"""GRPO v2 vs. matching SFT and GRPO v1, on held-out reviews. Usage: python scripts/grpo2_tables.py OUT.md

Final checkpoints only. Pre-stated success criteria (written before the evaluation ran):
(a) precision >= .25 with recall within .02 of the current policy (GRPO v1, same seed);
(b) GRPO gain over SFT larger than seed noise.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from robustness import ROOT, boot, fmt

SETS = [
    ("closed pool (8 eval reviews)", ["synergy-v3-small-trained-r4", "synergy-v3-small-trained-s2-r4", "synergy-v3-grpo2-r5"], ("hybrid", "citation", "keyword"), (10, 20, 40)),
    ("open graph (8 eval reviews)", ["synergy-v3-open-small-r4", "synergy-v3-open-small-s2-r4", "synergy-v3-open-grpo2-r5"], ("hybrid", "citation"), (10, 20, 40)),
    ("transfer (24 reviews)", ["ext-transfer-small-r4", "ext-transfer-small-s2-r4", "ext-transfer-grpo2-r5"], ("hybrid",), (20, 40)),
]
S1, S2 = "qwen3-1.7b-sft", "qwen3-1.7b-sft-s2"
G1, G2 = "qwen3-1.7b-grpo", "qwen3-1.7b-grpo-s2"
T1, T2, F1 = "grpo2-turn-s1", "grpo2-turn-s2", "grpo2-fbeta-s1"
AGENTS = [S1, G1, T1, F1, S2, G2, T2]
# (new, reference, what the comparison means)
PAIRS = [(T1, S1, "v2 turn s1 - SFT s1"), (T2, S2, "v2 turn s2 - SFT s2"), (F1, S1, "v2 fbeta s1 - SFT s1"),
         (T1, G1, "v2 turn s1 - v1 s1"), (T2, G2, "v2 turn s2 - v1 s2"), (F1, G1, "v2 fbeta s1 - v1 s1"),
         (T1, F1, "turn credit effect (turn s1 - fbeta s1)"), (G1, G2, "seed noise, v1 (s1 - s2)"), (T1, T2, "seed noise, v2 (s1 - s2)")]


def load(dirs):
    g = defaultdict(dict)
    for d in dirs:
        for path in sorted((ROOT / "runs" / d).glob("*.json")):
            run = json.loads(path.read_text())
            ep = run["episode"]
            if run["matrix"].get("repeat", 0) != 0 or run["agent"] not in AGENTS:
                continue
            mix = Counter()
            for event in run["trace"]:
                action = event.get("action") or {}
                if int(event.get("charged_cost") or 0) > 0 and action.get("kind") != "seed":
                    mix[action.get("kind")] += int(event["charged_cost"])
            g[(run["agent"], ep["mode"], ep["budget"])][ep["review_id"]] = {
                **{k: run["metrics"][k] for k in ("recall", "precision", "f1", "charged_cost", "included_count")}, "mix": mix}
    return g


def mean(cell, key):
    return st.mean(r[key] for r in cell.values())


def main():
    L = ["# GRPO v2 evaluation", "", "Final checkpoints. Hybrid unless stated. CIs are paired bootstrap over reviews.", ""]
    verdicts = []
    for title, dirs, modes, budgets in SETS:
        g = load(dirs)
        L += [f"## {title}", ""]
        for mode in modes:
            for b in budgets:
                if not any((a, mode, b) in g for a in AGENTS):
                    continue
                L += [f"### {mode}, B={b}", "", "| agent | n | recall | precision | F1 | included | spent | search | refs | citers | read |", "|---|---|---|---|---|---|---|---|---|---|---|"]
                for a in AGENTS:
                    c = g.get((a, mode, b))
                    if not c:
                        continue
                    mix = Counter()
                    for r in c.values():
                        mix.update(r["mix"])
                    tot = sum(mix.values()) or 1
                    L.append(f"| {a} | {len(c)} | {mean(c,'recall'):.3f} | {mean(c,'precision'):.3f} | {mean(c,'f1'):.3f} | {mean(c,'included_count'):.0f} | {mean(c,'charged_cost'):.1f} | "
                             + " | ".join(f"{mix[k]/tot:.0%}" for k in ("search", "expand_refs", "expand_citers", "read")) + " |")
                L.append("")
                for new, ref, what in PAIRS:
                    A, B = g.get((new, mode, b), {}), g.get((ref, mode, b), {})
                    ks = sorted(set(A) & set(B))
                    if len(ks) < 2:
                        continue
                    parts = []
                    for m in ("recall", "precision", "f1"):
                        d, lo, hi = boot([A[k][m] - B[k][m] for k in ks])
                        parts.append(f"{m} {fmt(d)} [{fmt(lo)}, {fmt(hi)}]")
                    L.append(f"- {what} (n={len(ks)}): " + "; ".join(parts))
                L.append("")
                for new, v1, sft in ((T1, G1, S1), (T2, G2, S2), (F1, G1, S1)):
                    A, V, S = g.get((new, mode, b), {}), g.get((v1, mode, b), {}), g.get((sft, mode, b), {})
                    ks = sorted(set(A) & set(V) & set(S))
                    if not ks:
                        continue
                    p = st.mean(A[k]["precision"] for k in ks)
                    dv = st.mean(A[k]["recall"] - V[k]["recall"] for k in ks)
                    ds = st.mean(A[k]["recall"] - S[k]["recall"] for k in ks)
                    noise = abs(st.mean(g[(G1, mode, b)][k]["recall"] - g[(G2, mode, b)][k]["recall"] for k in set(g.get((G1, mode, b), {})) & set(g.get((G2, mode, b), {})))) if (G1, mode, b) in g and (G2, mode, b) in g else float("nan")
                    verdicts.append(f"| {title.split(' (')[0]} | {mode} | {b} | {new} | {p:.3f} | {'yes' if p >= .25 else 'no'} | {dv:+.3f} | {'yes' if dv >= -.02 else 'no'} | {ds:+.3f} | {noise:.3f} | {'yes' if ds > noise else 'no'} |")
    L += ["## Pre-stated criteria", "", "(a) precision >= .25 and recall within .02 of v1 (same seed); (b) gain over SFT exceeds the v1 seed gap in that cell.", "",
          "| set | mode | B | agent | precision | >=.25 | recall - v1 | within .02 | recall - SFT | v1 seed gap | beyond noise |", "|---|---|---|---|---|---|---|---|---|---|---|"] + verdicts + [""]
    Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[1]).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
