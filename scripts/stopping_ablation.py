"""Stopping ablation: agent vs the same agent with early STOP rejected vs the no-loop baseline.

Usage: python scripts/stopping_ablation.py OUT.md
Closed graph, paired over reviews, repeat 0. If the no-stop agent reaches no-loop recall, early
stopping explains the agent's recall gap (P1); see docs/final/paper_plan.md.
"""
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from web_robust import boot_ci, load, sign_test  # noqa: E402

SYSTEMS = {  # (split, model) -> {arm: (dir, agent, mode)}
    ("transfer", "Haiku"): {"agent": ("ext-transfer-api-r4", "haiku-5.5", "hybrid"), "no-stop": ("ext-transfer-nostop", "haiku-5.5-nostop", "hybrid"),
                            "no-loop": ("ext-transfer-static", "haiku-5.5-static", "keyword")},
    ("eval", "Haiku"): {"agent": ("synergy-v3-api-r4", "haiku-5.5", "hybrid"), "no-stop": ("synergy-v3-nostop", "haiku-5.5-nostop", "hybrid"),
                        "no-loop": ("synergy-v3-static", "haiku-5.5-static", "keyword")},
    ("transfer", "Qwen3-1.7B GRPO"): {"agent": ("ext-transfer-small-r4", "qwen3-1.7b-grpo", "hybrid"),
                                      "no-stop": ("ext-transfer-nostop-qwen", "qwen3-1.7b-grpo-nostop", "hybrid"),
                                      "no-loop": ("ext-transfer-nostop-qwen", "qwen3-1.7b-grpo-static", "keyword")},
    ("eval", "Qwen3-1.7B GRPO"): {"agent": ("synergy-v3-small-trained-r4", "qwen3-1.7b-grpo", "hybrid"),
                                  "no-stop": ("synergy-v3-nostop-qwen", "qwen3-1.7b-grpo-nostop", "hybrid"),
                                  "no-loop": ("synergy-v3-nostop-qwen", "qwen3-1.7b-grpo-static", "keyword")},
    ("web transfer", "Haiku"): {"agent": ("ext-transfer-web-r5", "haiku-5.5", "hybrid"), "no-stop": ("ext-transfer-web-nostop", "haiku-5.5-nostop", "hybrid"),
                                "no-loop": ("ext-transfer-web-static", "haiku-5.5-static", "keyword")},
    ("web eval", "Haiku"): {"agent": ("synergy-v3-web-r5", "haiku-5.5", "hybrid"), "no-stop": ("synergy-v3-web-nostop", "haiku-5.5-nostop", "hybrid"),
                            "no-loop": ("synergy-v3-web-static", "haiku-5.5-static", "keyword")},
}
METRICS = ("recall", "precision", "f1", "included_count", "charged_cost")


def cell(spec, budget):
    return {r["episode"]["review_id"]: r for r in load(spec[0], agent=spec[1], mode=spec[2], budget=budget)}


def diff_row(label, a, b, key):
    keys = sorted(set(a) & set(b))
    if not keys:
        return f"| {label} | {key} | 0 | | | | |"
    d = [b[k]["metrics"][key] - a[k]["metrics"][key] for k in keys]
    lo, hi = boot_ci(d)
    return f"| {label} | {key} | {len(d)} | {statistics.fmean(d):+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} | {sign_test(d):.3f} |"


def main(out):
    L = ["# Stopping ablation (closed graph; web rows: live OpenAlex)", ""]
    for (split, model), arms in SYSTEMS.items():
        for budget in (20, 40):
            cells = {arm: cell(spec, budget) for arm, spec in arms.items()}
            if not all(cells.values()):
                continue
            L += [f"## {model}, {split}, B={budget}", "", "| arm | n | " + " | ".join(METRICS) + " | stops rejected | early stops honoured |",
                  "|---" * (len(METRICS) + 4) + "|"]
            for arm, runs in cells.items():
                vals = [statistics.fmean(r["metrics"][m] for r in runs.values()) for m in METRICS]
                summ = [r.get("agent_summary") or {} for r in runs.values()]
                rej = statistics.fmean(s.get("stops_rejected", 0) for s in summ) if arm == "no-stop" else float("nan")
                hon = statistics.fmean(s.get("early_stops_honoured", 0) for s in summ) if arm == "no-stop" else float("nan")
                L.append(f"| {arm} | {len(runs)} | " + " | ".join(f"{v:.3f}" if v < 2 else f"{v:.1f}" for v in vals) + f" | {rej:.2f} | {hon:.2f} |")
            L += ["", "| comparison (B - A) | metric | n | diff | 95% CI | up/down | sign p |", "|---|---|---|---|---|---|---|"]
            for label, a, b in (("no-stop - agent", "agent", "no-stop"), ("no-stop - no-loop", "no-loop", "no-stop")):
                for key in ("recall", "f1", "included_count"):
                    L.append(diff_row(label, cells[a], cells[b], key))
            L.append("")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
