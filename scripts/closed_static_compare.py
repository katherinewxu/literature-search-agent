"""No-loop LLM baseline vs the Haiku agent on the closed graph (paired over reviews).

Usage: python scripts/closed_static_compare.py OUT.md
Compares runs/{synergy-v3,ext-transfer}-static (configs/llm/*_static.json) with the closed-graph
Haiku hybrid agent in runs/{synergy-v3,ext-transfer}-api-r4, repeat 0, same seed and reviews.
"""
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from web_robust import boot_ci, load, paired, sign_test  # noqa: E402

SPLITS = {"eval": ("synergy-v3-api-r4", "synergy-v3-static"), "transfer": ("ext-transfer-api-r4", "ext-transfer-static")}


def by_review(runs, key):
    return {r["episode"]["review_id"]: r["metrics"][key] for r in runs}


def main(out):
    lines = ["# Closed graph: Haiku agent vs no-loop Haiku query list", "",
             "Paired over reviews; diff = no-loop - agent; 95% bootstrap CI; exact two-sided sign test.", ""]
    for split, (agent_dir, static_dir) in SPLITS.items():
        for budget in (20, 40):
            agent = load(agent_dir, agent="haiku-5.5", mode="hybrid", budget=budget)
            for label in ("haiku-5.5-static", "haiku-5.5-static-all"):
                static = load(static_dir, agent=label, budget=budget)
                if not static or not agent:
                    continue
                lines += [f"## {split}, B={budget}: {label} vs haiku-5.5 hybrid agent", "",
                          "| metric | n | diff | 95% CI | up/down | sign p |", "|---|---|---|---|---|---|"]
                for key in ("recall", "precision", "f1", "included_count", "charged_cost"):
                    lines.append(paired(key, by_review(agent, key), by_review(static, key)))
                means = lambda runs, k: statistics.fmean(r["metrics"][k] for r in runs)
                lines += ["", "| system | n | recall | precision | F1 | included | spent |", "|---|---|---|---|---|---|---|"]
                for name, runs in (("agent", agent), (label, static)):
                    lines.append(f"| {name} | {len(runs)} | {means(runs, 'recall'):.3f} | {means(runs, 'precision'):.3f} | "
                                 f"{means(runs, 'f1'):.3f} | {means(runs, 'included_count'):.1f} | {means(runs, 'charged_cost'):.1f} |")
                lines.append("")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
