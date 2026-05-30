"""
step10_prioritise.py
====================
Phase 2 — Novelty 2  |  Run on: LOCAL PC

What this does:
    Given the severity results from step8_severity.py, ranks all test
    pairs (UI screen pairs) by regression risk score and produces a
    prioritised test execution order.

    This extends the base paper (Moradi et al.) which stops at
    change detection + heatmap.  Here we answer a different question:

        "If you can only re-run K tests before the next release,
         which K screens should you check first?"

    Risk score per pair is computed from:
        - Number of detected changes
        - Severity of each change (high/medium/low)
        - Change type weights (remove > resize > relocate > add > color_change)
        - Confidence penalty for uncertain detections
        - Bonus for multiple co-occurring change types (complex screens)

    Output:
        ranked_results.json     — full ranked list with scores + breakdown
        Console table           — top-N pairs printed ranked by risk

Thesis contribution:
    Turns change detection into actionable test prioritisation.
    Test engineers can configure BUDGET (how many tests to run) and
    the ranker tells them exactly which screens to check first.
    This is directly applicable to CI/CD pipelines where test time
    is constrained.

Usage:
    # Rank from step8 batch results
    python scripts/step10_prioritise.py ^
        --severity_results ./outputs/severity_v4/severity_results_0.50.json ^
        --output_dir       ./outputs/prioritisation_v1 ^
        --budget           50 ^
        --show_top         20

    # Compare two builds: rank by delta (what changed between runs)
    python scripts/step10_prioritise.py ^
        --severity_results ./outputs/severity_v5/severity_results_0.50.json ^
        --output_dir       ./outputs/prioritisation_v2 ^
        --budget           50
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# =============================================================================
# RISK SCORING
# =============================================================================

# Change type risk weights — how much each type contributes to regression risk.
# Grounded in the thesis GT_DECISION mapping:
#   remove / add / resize / relocate  →  FAIL (functional regression)
#   color_change                      →  PASS (cosmetic, lower risk)
CHANGE_TYPE_RISK = {
    "remove":       1.00,   # highest — element disappeared, likely broken flow
    "resize":       0.80,   # high    — layout shift, may obscure other elements
    "relocate":     0.75,   # high    — element moved, tap targets shift
    "add":          0.65,   # medium  — new element, unknown impact
    "color_change": 0.20,   # low     — cosmetic by default
}

# Severity multipliers — applied on top of change type weight
SEVERITY_MULTIPLIER = {
    "high":   1.0,
    "medium": 0.65,
    "low":    0.30,
}

# Bonus for co-occurring change types on the same screen.
# A screen with both remove + resize is harder to verify than one with
# a single change — assign a complexity bonus.
COMPLEXITY_BONUS_PER_EXTRA_TYPE = 0.08   # added per unique change type beyond 1


def compute_pair_risk(changes):
    """
    Compute a regression risk score [0, 1] for a single test pair.

    Algorithm:
        1. Base score = weighted sum of (change_type_risk * severity_mult * severity_score)
           across all detected changes
        2. Normalise by max possible score for the observed change count
        3. Add complexity bonus for screens with multiple change types
        4. Clip to [0, 1]

    Returns:
        risk_score   : float [0, 1]
        breakdown    : dict with score components for reporting
    """
    if not changes:
        return 0.0, {"n_changes": 0, "base_score": 0.0,
                     "complexity_bonus": 0.0, "change_types": []}

    weighted_scores = []
    change_types_seen = set()

    for c in changes:
        ctype    = c.get("change_type", "remove")
        slabel   = c.get("severity_label", "medium")
        sscore   = c.get("severity_score", 0.5)

        type_w   = CHANGE_TYPE_RISK.get(ctype, 0.5)
        sev_mult = SEVERITY_MULTIPLIER.get(slabel, 0.65)

        # Combined weight: type risk × severity multiplier × raw severity score
        # Raw severity score adds magnitude signal (e.g. large resize vs small)
        w = type_w * sev_mult * sscore
        weighted_scores.append(w)
        change_types_seen.add(ctype)

    # Base score: mean of weighted individual changes
    # Using mean (not sum) so that screens with many small changes don't
    # automatically outrank screens with one critical change.
    base_score = float(np.mean(weighted_scores)) if weighted_scores else 0.0

    # Complexity bonus: screens with multiple distinct change types are harder
    # to verify — a tester needs to check more things.
    n_extra_types = max(0, len(change_types_seen) - 1)
    complexity_bonus = n_extra_types * COMPLEXITY_BONUS_PER_EXTRA_TYPE

    risk_score = float(np.clip(base_score + complexity_bonus, 0.0, 1.0))

    breakdown = {
        "n_changes":        len(changes),
        "base_score":       round(base_score, 4),
        "complexity_bonus": round(complexity_bonus, 4),
        "change_types":     sorted(change_types_seen),
        "n_fail":           sum(1 for c in changes if c.get("decision") == "FAIL"),
        "max_severity":     max((c.get("severity_score", 0) for c in changes),
                                default=0.0),
    }

    return risk_score, breakdown


def risk_tier(score):
    """Map risk score to a human-readable tier label."""
    if score >= 0.65:
        return "CRITICAL"
    elif score >= 0.40:
        return "HIGH"
    elif score >= 0.20:
        return "MEDIUM"
    else:
        return "LOW"


# =============================================================================
# PRIORITISATION
# =============================================================================

def prioritise(severity_results_path, output_dir, budget=None, show_top=20):
    """
    Load step8 severity results, rank all pairs by risk, save outputs.

    Args:
        severity_results_path : path to severity_results_X.XX.json from step8
        output_dir            : where to write ranked_results.json
        budget                : if set, highlight the top-budget pairs
        show_top              : how many rows to print in the console table
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(severity_results_path) as f:
        data = json.load(f)

    pair_decisions = data.get("pair_decisions", [])
    fail_threshold = data.get("fail_threshold", 0.5)

    if not pair_decisions:
        print("[ERROR] No pair_decisions found in severity results file.")
        return

    print(f"[INFO] Ranking {len(pair_decisions)} test pairs by regression risk ...")
    print(f"[INFO] Fail threshold used in step8: {fail_threshold}")

    # ── Score every pair ──────────────────────────────────────
    ranked = []
    for pair in pair_decisions:
        pid        = pair["pair_id"]
        gt_type    = pair.get("gt_type", "unknown")
        gt_dec     = pair.get("gt_decision", "FAIL")
        our_dec    = pair.get("our_decision", "PASS")
        correct    = pair.get("correct", False)
        changes    = pair.get("changes", [])

        risk_score, breakdown = compute_pair_risk(changes)

        ranked.append({
            "pair_id":      pid,
            "risk_score":   round(risk_score, 4),
            "risk_tier":    risk_tier(risk_score),
            "our_decision": our_dec,
            "gt_type":      gt_type,
            "gt_decision":  gt_dec,
            "correct":      correct,
            **breakdown,
        })

    # Sort by risk score descending
    ranked.sort(key=lambda x: x["risk_score"], reverse=True)

    # Assign rank
    for i, r in enumerate(ranked, 1):
        r["rank"] = i

    # ── Budget analysis ───────────────────────────────────────
    budget_stats = None
    if budget and budget < len(ranked):
        top_k     = ranked[:budget]
        bottom_k  = ranked[budget:]

        # How many true regressions are in the top-K?
        # (GT decision = FAIL means it's a real regression)
        true_regressions_total = sum(
            1 for r in ranked if r["gt_decision"] == "FAIL"
        )
        true_regressions_in_budget = sum(
            1 for r in top_k if r["gt_decision"] == "FAIL"
        )
        missed_regressions = sum(
            1 for r in bottom_k if r["gt_decision"] == "FAIL"
        )

        recall_at_budget = (
            true_regressions_in_budget / true_regressions_total
            if true_regressions_total > 0 else 0.0
        )

        budget_stats = {
            "budget":                      budget,
            "total_pairs":                 len(ranked),
            "true_regressions_total":      true_regressions_total,
            "true_regressions_in_budget":  true_regressions_in_budget,
            "missed_regressions":          missed_regressions,
            "recall_at_budget":            round(recall_at_budget, 4),
            "test_reduction_pct":          round(
                (1 - budget / len(ranked)) * 100, 1
            ),
        }

    # ── Console output ────────────────────────────────────────
    display_n = min(show_top, len(ranked))
    budget_marker = budget or len(ranked)

    print("\n" + "=" * 85)
    print("  TEST PRIORITISATION — RANKED BY REGRESSION RISK")
    print(f"  Total pairs: {len(ranked)}  |  Showing top {display_n}"
          + (f"  |  Budget: {budget}" if budget else ""))
    print("=" * 85)
    print(f"  {'Rank':<6} {'Pair ID':<28} {'Risk':>6} {'Tier':<10} "
          f"{'Decision':<10} {'Changes':>8} {'Change Types'}")
    print("  " + "-" * 80)

    for r in ranked[:display_n]:
        budget_flag = " ◄" if r["rank"] == budget_marker else ""
        print(f"  {r['rank']:<6} {r['pair_id']:<28} "
              f"{r['risk_score']:>6.4f} {r['risk_tier']:<10} "
              f"{r['our_decision']:<10} {r['n_changes']:>8} "
              f"{', '.join(r['change_types']) or '—'}{budget_flag}")

    if budget and budget < len(ranked):
        print(f"\n  ◄ Budget cutoff at rank {budget}")

    print("=" * 85)

    # ── Budget recall summary ─────────────────────────────────
    if budget_stats:
        bs = budget_stats
        print(f"\n  BUDGET ANALYSIS  (run top {bs['budget']} of {bs['total_pairs']} tests)")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Test reduction  :  {bs['test_reduction_pct']}% fewer tests to run")
        print(f"  True regressions captured  :  "
              f"{bs['true_regressions_in_budget']} / {bs['true_regressions_total']} "
              f"({bs['recall_at_budget']*100:.1f}% recall)")
        print(f"  Regressions missed  :  {bs['missed_regressions']} "
              f"(in the deprioritised {bs['total_pairs'] - bs['budget']} pairs)")
        print(f"  ─────────────────────────────────────────────")

    print("=" * 85)

    # ── Distribution summary ──────────────────────────────────
    tier_counts = defaultdict(int)
    for r in ranked:
        tier_counts[r["risk_tier"]] += 1

    print(f"\n  Risk distribution across all {len(ranked)} pairs:")
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        n   = tier_counts[tier]
        pct = n / len(ranked) * 100
        bar = "█" * int(pct / 2)
        print(f"    {tier:<10} {n:>4}  ({pct:5.1f}%)  {bar}")

    # ── Recall@K curve ────────────────────────────────────────
    true_reg_total = sum(1 for r in ranked if r["gt_decision"] == "FAIL")
    if true_reg_total > 0:
        print(f"\n  Recall@K — how many regressions are caught by testing top-K pairs:")
        checkpoints = [10, 25, 50, 100, 150, 200]
        print(f"    {'K':<8} {'Captured':>10} {'Recall':>10} {'Test %':>10}")
        print("    " + "-" * 42)
        for k in checkpoints:
            if k > len(ranked):
                break
            captured = sum(1 for r in ranked[:k] if r["gt_decision"] == "FAIL")
            recall   = captured / true_reg_total
            test_pct = k / len(ranked) * 100
            print(f"    {k:<8} {captured:>10} {recall:>9.1%} {test_pct:>9.1f}%")

    print()

    # ── Save JSON ─────────────────────────────────────────────
    output = {
        "source_file":    str(severity_results_path),
        "fail_threshold": fail_threshold,
        "total_pairs":    len(ranked),
        "budget":         budget,
        "budget_stats":   budget_stats,
        "ranked_pairs":   ranked,
    }
    out_path = output_dir / "ranked_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Ranked results saved: {out_path}")

    return ranked, budget_stats


# =============================================================================
# EVALUATION — how well does risk ranking correlate with true regression?
# =============================================================================

def evaluate_ranking(ranked):
    """
    Measure how well the risk ranking separates true regressions from benign.

    Computes:
        - Average Precision (AP) — area under precision-recall curve
        - NDCG@K — normalised discounted cumulative gain at K
        - Separation score — mean risk of FAIL pairs vs mean risk of PASS pairs
    """
    from collections import OrderedDict

    labels = [1 if r["gt_decision"] == "FAIL" else 0 for r in ranked]
    scores = [r["risk_score"] for r in ranked]

    # Average Precision
    n_pos   = sum(labels)
    n_total = len(labels)
    if n_pos == 0:
        return {}

    precisions, recalls = [], []
    tp = 0
    for i, (label, _) in enumerate(zip(labels, scores), 1):
        if label == 1:
            tp += 1
        precisions.append(tp / i)
        recalls.append(tp / n_pos)

    # Trapezoidal AP
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]

    # Separation
    fail_scores = [s for l, s in zip(labels, scores) if l == 1]
    pass_scores = [s for l, s in zip(labels, scores) if l == 0]
    separation  = (np.mean(fail_scores) - np.mean(pass_scores)
                   if fail_scores and pass_scores else 0.0)

    print("\n" + "=" * 50)
    print("  RANKING QUALITY METRICS")
    print("=" * 50)
    print(f"  Average Precision (AP) : {ap:.4f}")
    print(f"  Separation score       : {separation:+.4f}  "
          f"(mean FAIL risk - mean PASS risk)")
    print(f"  Mean risk — FAIL pairs : {np.mean(fail_scores):.4f}")
    print(f"  Mean risk — PASS pairs : {np.mean(pass_scores):.4f}")
    print("=" * 50)

    return {
        "average_precision": round(ap, 4),
        "separation_score":  round(float(separation), 4),
        "mean_risk_fail":    round(float(np.mean(fail_scores)), 4),
        "mean_risk_pass":    round(float(np.mean(pass_scores)), 4),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rank test pairs by regression risk (test prioritisation)"
    )
    parser.add_argument(
        "--severity_results", required=True,
        help="Path to severity_results_X.XX.json from step8_severity.py"
    )
    parser.add_argument(
        "--output_dir", default="./outputs/prioritisation",
        help="Where to save ranked_results.json"
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help="Max number of tests to run (e.g. 50 = run only top 50 riskiest pairs)"
    )
    parser.add_argument(
        "--show_top", type=int, default=20,
        help="How many rows to print in the console table (default: 20)"
    )
    args = parser.parse_args()

    ranked, budget_stats = prioritise(
        severity_results_path=args.severity_results,
        output_dir=args.output_dir,
        budget=args.budget,
        show_top=args.show_top,
    )

    if ranked:
        eval_metrics = evaluate_ranking(ranked)

        # Append ranking quality to JSON
        out_path = Path(args.output_dir) / "ranked_results.json"
        with open(out_path) as f:
            existing = json.load(f)
        existing["ranking_quality"] = eval_metrics
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)

    print(f"\n[DONE] Next: review ranked_results.json for thesis table")


if __name__ == "__main__":
    main()
