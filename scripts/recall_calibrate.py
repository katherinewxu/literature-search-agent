"""Screener-calibrated recall estimation. Usage: python scripts/recall_calibrate.py analysis/recall-est

Calibrates the LLM screener's true/false-positive rates on the CALIBRATION set (the 40 training
reviews, where using gold labels is allowed) and applies the Rogan-Gladen correction
p = (p_obs - FPR) / (TPR - FPR) to every stratum of the evaluation episodes (8 evaluation +
24 transfer reviews). The lower bound maps the Clopper-Pearson upper bound of the observed
"yes" rate through the same correction (calibration uncertainty is not propagated).
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from litsearch.recall_estimation import cp_upper  # noqa: E402

CALIBRATION = "life-science 40"


def rates(rows):
    """Pooled TPR (over all strata) and per-stratum FPR."""
    tp = pos = 0
    fp, neg = defaultdict(int), defaultdict(int)
    for r in rows:
        for name, s in r["strata"].items():
            yes, gold, sampled = set(s["yes"]), set(s["gold"]), s["sampled"]
            tp += len(yes & gold)
            pos += len(gold)
            fp[name] += len(yes - gold)
            neg[name] += len(sampled) - len(gold)
    tpr = tp / pos if pos else 1.0
    return tpr, {k: fp[k] / neg[k] if neg[k] else 0.0 for k in fp}


def corrected(rows_or_row, tpr, fpr, alpha=0.05):
    r = rows_or_row
    found = total = upper = 0.0
    for name, s in r["strata"].items():
        n = len(s["sampled"])
        if not n:
            continue
        k = len(s["yes"])
        denom = max(tpr - fpr[name], 1e-3)
        p = min(1.0, max(0.0, (k / n - fpr[name]) / denom))
        p_up = min(1.0, max(0.0, (cp_upper(k, n, alpha / 2) - fpr[name]) / denom))
        total += s["n_total"] * p
        if name == "included":
            found = s["n_total"] * p
            upper += s["n_total"] * p
        else:
            upper += s["n_total"] * p_up
    return (found / total if total else None), (found / upper if upper else None)


def corr(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")


def main():
    out = Path(sys.argv[1])
    rows = [json.loads(l) for l in (out / "recall_estimates.jsonl").read_text().splitlines()]
    calib = [r for r in rows if r["set"] == CALIBRATION]
    tpr, fpr = rates(calib)
    lines = ["# Calibrated recall estimation", "",
             f"Screener rates fit on the {len(calib)} calibration episodes ({CALIBRATION}): "
             f"TPR {tpr:.2f}; FPR included {fpr.get('included', 0):.3f}, excluded {fpr.get('excluded', 0):.3f}, "
             f"unexposed {fpr.get('unexposed', 0):.3f}.", "",
             "| set | agent | n | true recall | estimator | mean estimate | MAE | corr | lower bound holds | mean lower bound |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in rows:
        if r["set"] != CALIBRATION:
            groups[(r["set"], r["agent"])].append(r)
    summary = {}
    for (name, agent), rs in sorted(groups.items()):
        for label, fn in (
            ("sample, gold screener", lambda r: (r["oracle_hat"], r["oracle_lower"])),
            ("sample, Haiku raw", lambda r: (r["llm_hat"], r["llm_lower"])),
            ("sample, Haiku calibrated", lambda r: corrected(r, tpr, fpr)),
            ("yield curve", lambda r: (r["saturation"], None)),
        ):
            vals = [(fn(r), r["true_recall"]) for r in rs]
            vals = [((h, lo), t) for (h, lo), t in vals if h is not None]
            if not vals:
                continue
            est = [h for (h, _), _ in vals]
            truth = [t for _, t in vals]
            lows = [lo for (_, lo), _ in vals if lo is not None]
            holds = [t >= lo for (_, lo), t in vals if lo is not None]
            mae = st.mean(abs(a - b) for a, b in zip(est, truth))
            lines.append(
                f"| {name} | {agent} | {len(vals)} | {st.mean(truth):.3f} | {label} | {st.mean(est):.3f} | {mae:.3f} | "
                f"{corr(est, truth):.2f} | {(f'{sum(holds) / len(holds):.0%}' if holds else '–')} | "
                f"{(f'{st.mean(lows):.3f}' if lows else '–')} |"
            )
            summary.setdefault(label, []).extend(zip(est, truth))
    lines += ["", "Pooled over all evaluation episodes (8 + 24 reviews, all agents):", ""]
    for label, pairs in summary.items():
        e, t = zip(*pairs)
        lines.append(f"- {label}: MAE {st.mean(abs(a - b) for a, b in pairs):.3f}, corr {corr(list(e), list(t)):.2f}, "
                     f"mean estimate {st.mean(e):.3f} vs true {st.mean(t):.3f} (n={len(pairs)})")
    (out / "recall_calibrated.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
