"""Stopping ablation robustness: T=0.7, 3 repeats, closed graph, B=40 (configs/llm/*_ablation*.json).

Usage: python scripts/ablation_repeats.py OUT.md
Arms: agent, no-stop (early STOP rejected), prompt-only ("use your full budget" in the system
prompt), no-loop (static LLM query list). Each review's metric is averaged over its repeats, then
compared paired over reviews (bootstrap CI + exact sign test). "repeat SD" is the SD across the 3
repeats of the mean-over-reviews recall. Also reports the prompt-only arm at T=0 against the T=0
agent and no-stop runs.
"""
import json
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from web_robust import RUNS, boot_ci, sign_test  # noqa: E402

METRICS = ("recall", "precision", "f1", "included_count", "charged_cost")
SETS = {  # (split, model): (dir, label prefix)
    ("transfer", "Haiku"): ("ext-transfer-ablation", "haiku-5.5"),
    ("eval", "Haiku"): ("synergy-v3-ablation", "haiku-5.5"),
    ("transfer", "Qwen3-1.7B GRPO"): ("ext-transfer-ablation-qwen", "qwen3-1.7b-grpo"),
    ("eval", "Qwen3-1.7B GRPO"): ("synergy-v3-ablation-qwen", "qwen3-1.7b-grpo"),
}
ARMS = {"agent": "{p}-t07", "no-stop": "{p}-nostop-t07", "prompt-only": "{p}-prompt-t07", "no-loop": "{p}-static-t07"}
T0 = {  # T=0 references for the prompt-only arm: (dir, label)
    ("transfer", "Haiku"): {"agent": ("ext-transfer-api-r4", "haiku-5.5"), "no-stop": ("ext-transfer-nostop", "haiku-5.5-nostop"),
                            "prompt-only": ("ext-transfer-ablation", "haiku-5.5-prompt")},
    ("eval", "Haiku"): {"agent": ("synergy-v3-api-r4", "haiku-5.5"), "no-stop": ("synergy-v3-nostop", "haiku-5.5-nostop"),
                        "prompt-only": ("synergy-v3-ablation", "haiku-5.5-prompt")},
    ("transfer", "Qwen3-1.7B GRPO"): {"agent": ("ext-transfer-small-r4", "qwen3-1.7b-grpo"),
                                      "no-stop": ("ext-transfer-nostop-qwen", "qwen3-1.7b-grpo-nostop"),
                                      "prompt-only": ("ext-transfer-ablation-qwen", "qwen3-1.7b-grpo-prompt")},
    ("eval", "Qwen3-1.7B GRPO"): {"agent": ("synergy-v3-small-trained-r4", "qwen3-1.7b-grpo"),
                                  "no-stop": ("synergy-v3-nostop-qwen", "qwen3-1.7b-grpo-nostop"),
                                  "prompt-only": ("synergy-v3-ablation-qwen", "qwen3-1.7b-grpo-prompt")},
}


def runs(d, label, budget=40):
    """review -> repeat -> metrics (also counts rejected stops)."""
    out = defaultdict(dict)
    for p in sorted((RUNS / d).glob(f"*__{label}__*__b{budget}__r*.json")):
        r = json.loads(p.read_text())
        if r["agent"] != label:
            continue
        m = dict(r["metrics"])
        m["stops_rejected"] = (r.get("agent_summary") or {}).get("stops_rejected", 0)
        out[r["episode"]["review_id"]][r["matrix"].get("repeat", 0)] = m
    return out


def per_review(cell, key):
    return {rev: statistics.fmean(m[key] for m in reps.values()) for rev, reps in cell.items()}


def paired(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        return "| 0 | | | | |"
    d = [b[k] - a[k] for k in keys]
    lo, hi = boot_ci(d)
    return f"| {len(d)} | {statistics.fmean(d):+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} | {sign_test(d):.3f} |"


def main(out):
    L = ["# Stopping ablation robustness (closed graph, B=40)", ""]
    for key, (d, prefix) in SETS.items():
        cells = {arm: runs(d, tmpl.format(p=prefix)) for arm, tmpl in ARMS.items()}
        if not any(cells.values()):
            continue
        L += [f"## {key[1]}, {key[0]}: T=0.7, 3 repeats (per-review mean over repeats)", "",
              "| arm | reviews | runs | " + " | ".join(METRICS) + " | stops rejected | repeat SD of recall |", "|---" * (len(METRICS) + 5) + "|"]
        for arm, cell in cells.items():
            if not cell:
                L.append(f"| {arm} | 0 | 0 |" + " |" * (len(METRICS) + 2))
                continue
            vals = [statistics.fmean(per_review(cell, m).values()) for m in METRICS + ("stops_rejected",)]
            reps = sorted({r for v in cell.values() for r in v})
            by_rep = [statistics.fmean(v[r]["recall"] for v in cell.values() if r in v) for r in reps]
            sd = statistics.stdev(by_rep) if len(by_rep) > 1 else float("nan")
            L.append(f"| {arm} | {len(cell)} | {sum(len(v) for v in cell.values())} | " + " | ".join(f"{v:.3f}" if v < 2 else f"{v:.1f}" for v in vals) + f" | {sd:.3f} |")
        L += ["", "| comparison (B - A) | metric | n | diff | 95% CI | up/down | sign p |", "|---|---|---|---|---|---|---|"]
        for a, b in (("agent", "no-stop"), ("agent", "prompt-only"), ("prompt-only", "no-stop"), ("no-loop", "agent"), ("no-loop", "no-stop")):
            for m in ("recall", "f1"):
                if cells[a] and cells[b]:
                    L.append(f"| {b} - {a} | {m} " + paired(per_review(cells[a], m), per_review(cells[b], m)))
        L.append("")
        t0 = {arm: runs(*spec) for arm, spec in T0[key].items()}
        t0 = {arm: {rev: {0: reps[0]} for rev, reps in c.items() if 0 in reps} for arm, c in t0.items()}
        if t0["prompt-only"]:
            L += [f"### {key[1]}, {key[0]}: T=0 (single run)", "", "| arm | n | recall | f1 | included | spent | stops rejected |", "|---|---|---|---|---|---|---|"]
            for arm, cell in t0.items():
                v = [statistics.fmean(per_review(cell, m).values()) for m in ("recall", "f1", "included_count", "charged_cost", "stops_rejected")]
                L.append(f"| {arm} | {len(cell)} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.1f} | {v[3]:.1f} | {v[4]:.2f} |")
            L += ["", "| comparison | metric | n | diff | 95% CI | up/down | sign p |", "|---|---|---|---|---|---|---|"]
            for a, b in (("agent", "prompt-only"), ("prompt-only", "no-stop")):
                for m in ("recall", "f1"):
                    L.append(f"| {b} - {a} | {m} " + paired(per_review(t0[a], m), per_review(t0[b], m)))
            L.append("")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
