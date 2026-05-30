"""
step8_severity.py
=================
Phase 2 - Week 6  |  Run on: LOCAL PC

What this does:
    Scores each detected change as regression (FAIL) or benign (PASS).
    Takes the change type from step7_classifier.py and the raw feature
    vector from the graph matcher, and produces a severity score 0-1.

    Severity rules (configurable thresholds):
        remove        -> always HIGH   (element gone = likely regression)
        add           -> MEDIUM        (new element, may or may not matter)
        resize        -> magnitude-dependent (area_ratio far from 1.0 = HIGH)
        relocate      -> distance-dependent  (centre_dist high = HIGH)
        color_change  -> LOW by default      (cosmetic unless large shift)

    Output per change:
        severity_score  : float 0.0 - 1.0
        severity_label  : "low" / "medium" / "high"
        decision        : "PASS" / "FAIL"

    Thesis contribution:
        Configurable FAIL threshold lets test engineers tune sensitivity
        per project — ignore cosmetic updates, catch structural regressions.

Usage:
    # Evaluate severity on test pairs (requires classifier.pkl from step7)
    python scripts/step8_severity.py ^
        --model          ./outputs/yolo_runs/rico_ui_v2/weights/best.pt ^
        --classifier     ./outputs/classifier/classifier.pkl ^
        --test_dir       ./outputs/change_dataset/test/pairs ^
        --test_manifest  ./outputs/change_dataset/test/manifest.json ^
        --output_dir     ./outputs/severity ^
        --fail_threshold 0.5

    # Run on a single pair
    python scripts/step8_severity.py ^
        --model      ./outputs/yolo_runs/rico_ui_v2/weights/best.pt ^
        --classifier ./outputs/classifier/classifier.pkl ^
        --image1     ./outputs/change_dataset/test/pairs/10703_v00_original.jpg ^
        --image2     ./outputs/change_dataset/test/pairs/10703_v00_changed.jpg ^
        --fail_threshold 0.5
"""

import cv2
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).parent))
from graph_builder import build_graph, run_yolo
from graph_matcher import match_graphs, phash_similarity, colour_similarity
from step7_classifier import (
    compute_bbox_area_ratio,
    compute_centre_distance,
    compute_region_pixel_diff,
    extract_features_from_pair,
    SIM_THRESHOLD,          # single source of truth — must match training threshold
)


# =============================================================================
# SEVERITY RULES
# =============================================================================

# Base severity score per change type (before magnitude adjustment)
BASE_SEVERITY = {
    "remove":       0.85,   # high — element gone
    "add":          0.50,   # medium — new element appeared
    "resize":       0.55,   # medium-high — magnitude-adjusted below
    "relocate":     0.55,   # medium-high — distance-adjusted below
    "color_change": 0.10,   # low — cosmetic by default (lowered from 0.20 to reduce FP rate)
}

# Severity labels by score range
def severity_label(score):
    if score >= 0.65:
        return "high"
    elif score >= 0.35:
        return "medium"
    else:
        return "low"


def compute_severity(change_type, features):
    """
    Compute severity score for a single detected change.

    Args:
        change_type : predicted change type string
        features    : [phash_dist, colour_dist, area_ratio,
                       centre_dist, class_match, similarity, region_pixel_diff]

    Returns:
        score : float 0.0 - 1.0
    """
    phash_dist, colour_dist, area_ratio, centre_dist, class_match, sim, *_ = features

    base = BASE_SEVERITY.get(change_type, 0.5)

    if change_type == "remove":
        # Always high — slight reduction if similarity was medium
        # (could be a partial match rather than full removal)
        score = base - (0.1 * sim)

    elif change_type == "add":
        # Medium — slightly higher if class_match is 0 (unknown element type)
        score = base + (0.1 * (1.0 - class_match))

    elif change_type == "resize":
        # Magnitude: how far is area_ratio from 0.5 (the neutral log-scale point)?
        # area_ratio=0.5 means no size change, 0 or 1 means extreme change
        magnitude = abs(area_ratio - 0.5) * 2.0   # 0 = no change, 1 = max change
        score = base + (0.3 * magnitude) - (0.1 * sim)

    elif change_type == "relocate":
        # Distance: higher centre_dist = more severe
        score = base + (0.3 * centre_dist) - (0.05 * sim)

    elif change_type == "color_change":
        # Colour distance drives severity — large hue shifts are more notable
        score = base + (0.4 * colour_dist) - (0.1 * sim)

    else:
        score = base

    return float(np.clip(score, 0.0, 1.0))


# =============================================================================
# SINGLE PAIR SCORING
# =============================================================================

def score_pair(model, classifier_data, img1_path, img2_path,
               fail_threshold=0.5):
    """
    Full pipeline for one pair:
        YOLO → Graph → Match → Classify → Score → PASS/FAIL

    Returns list of change dicts, each with:
        change_type, features, severity_score, severity_label, decision
    """
    clf = classifier_data["classifier"]
    le  = classifier_data["label_encoder"]

    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        return []

    h, w = img1.shape[:2]

    # Build graphs
    det1 = run_yolo(model, img1)
    det2 = run_yolo(model, img2)
    G1 = build_graph(img1, det1, extract_ocr=False, extract_clip=True)
    G2 = build_graph(img2, det2, extract_ocr=False, extract_clip=True)

    if G1.number_of_nodes() == 0 and G2.number_of_nodes() == 0:
        return []

    # Match graphs — threshold MUST match SIM_THRESHOLD in step7_classifier.py.
    # SIM_THRESHOLD is imported directly so both training (step7) and inference
    # (step8) always use the same value.  Any manual change to SIM_THRESHOLD in
    # step7 propagates here automatically, preventing distribution mismatch.
    matches, changed_nodes, added_nodes, removed_nodes, sim_matrix = \
        match_graphs(G1, G2, similarity_threshold=SIM_THRESHOLD)

    results = []

    # ── Changed nodes ──────────────────────────────────────────
    for n1_id, n2_id, sim in changed_nodes:
        d1 = G1.nodes[n1_id]
        d2 = G2.nodes[n2_id]

        phash_dist  = 1.0 - phash_similarity(d1.get("phash"), d2.get("phash"))
        colour_dist = 1.0 - colour_similarity(
            d1.get("mean_colour"), d2.get("mean_colour")
        )
        area_ratio  = compute_bbox_area_ratio(d1["bbox"], d2["bbox"])
        centre_dist = compute_centre_distance(d1["bbox"], d2["bbox"], w, h)
        class_match = 1.0 if d1["class_name"] == d2["class_name"] else 0.0
        pixel_diff  = compute_region_pixel_diff(img1, img2, d1["bbox"], d2["bbox"])
        # 8-feature vector — must match classifier_v4 training vector exactly
        features    = [phash_dist, colour_dist, area_ratio,
                       centre_dist, class_match, float(sim),
                       pixel_diff, 1.0]   # is_matched=1.0

        change_type = le.inverse_transform(
            clf.predict([features])
        )[0]
        score  = compute_severity(change_type, features)
        label  = severity_label(score)
        decision = "FAIL" if score >= fail_threshold else "PASS"

        results.append({
            "type":           "changed",
            "change_type":    change_type,
            "box_original":   d1["bbox"],
            "box_changed":    d2["bbox"],
            "class":          d1["class_name"],
            "features":       features,
            "severity_score": score,
            "severity_label": label,
            "decision":       decision,
        })

    # ── Removed nodes ──────────────────────────────────────────
    for n1_id in removed_nodes:
        d1 = G1.nodes[n1_id]
        # is_matched=0.0 — this node had no match in G2
        features = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        change_type = "remove"
        score    = compute_severity(change_type, features)
        label    = severity_label(score)
        decision = "FAIL" if score >= fail_threshold else "PASS"

        results.append({
            "type":           "removed",
            "change_type":    change_type,
            "box_original":   d1["bbox"],
            "box_changed":    None,
            "class":          d1["class_name"],
            "features":       features,
            "severity_score": score,
            "severity_label": label,
            "decision":       decision,
        })

    # ── Added nodes ────────────────────────────────────────────
    for n2_id in added_nodes:
        d2 = G2.nodes[n2_id]
        # is_matched=0.0 — this node had no match in G1
        features = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]

        change_type = "add"
        score    = compute_severity(change_type, features)
        label    = severity_label(score)
        decision = "FAIL" if score >= fail_threshold else "PASS"

        results.append({
            "type":           "added",
            "change_type":    change_type,
            "box_original":   None,
            "box_changed":    d2["bbox"],
            "class":          d2["class_name"],
            "features":       features,
            "severity_score": score,
            "severity_label": label,
            "decision":       decision,
        })

    return results


# =============================================================================
# BATCH EVALUATION
# =============================================================================

def evaluate_severity(model, classifier_data, test_dir, manifest_path,
                      output_dir, fail_threshold=0.5):
    """
    Run full pipeline on all test pairs.
    Evaluates how well severity scoring maps to actual change types.
    """
    test_dir   = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    pairs = manifest["pairs"]
    print(f"[INFO] Scoring {len(pairs)} test pairs "
          f"(fail_threshold={fail_threshold})...")

    # Ground truth severity mapping
    # remove/add = regression (FAIL expected)
    # resize/relocate = medium (FAIL expected for large changes)
    # color_change = benign (PASS expected)
    GT_DECISION = {
        "remove":       "FAIL",
        "add":          "FAIL",
        "resize":       "FAIL",
        "relocate":     "FAIL",
        "color_change": "PASS",
    }

    all_results   = []
    stats         = defaultdict(lambda: defaultdict(int))
    pair_decisions = []

    for i, pair in enumerate(pairs):
        pid         = pair["pair_id"]
        gt_type     = pair["change_type"]
        orig_path   = test_dir / f"{pid}_original.jpg"
        chng_path   = test_dir / f"{pid}_changed.jpg"

        if not orig_path.exists() or not chng_path.exists():
            continue

        changes = score_pair(
            model, classifier_data, orig_path, chng_path, fail_threshold
        )

        # Pair-level decision: FAIL if any change is FAIL
        pair_decision = "FAIL" if any(
            c["decision"] == "FAIL" for c in changes
        ) else "PASS"

        gt_decision = GT_DECISION.get(gt_type, "FAIL")
        correct     = pair_decision == gt_decision

        pair_decisions.append({
            "pair_id":      pid,
            "gt_type":      gt_type,
            "gt_decision":  gt_decision,
            "our_decision": pair_decision,
            "correct":      correct,
            "n_changes":    len(changes),
            "changes":      changes,
        })

        stats[gt_type][pair_decision] += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)} pairs scored...")

    # ── Print results table ────────────────────────────────────
    total   = len(pair_decisions)
    correct = sum(1 for p in pair_decisions if p["correct"])

    print("\n" + "=" * 60)
    print("SEVERITY SCORER — EVALUATION RESULTS")
    print(f"Fail threshold: {fail_threshold}")
    print("=" * 60)
    print(f"\nPair-level accuracy: {correct}/{total} = "
          f"{correct/total:.4f}" if total > 0 else "0 pairs")

    print(f"\n{'Change Type':<15} {'PASS':>8} {'FAIL':>8} "
          f"{'GT Decision':>15} {'Correct %':>10}")
    print("-" * 60)
    for gt_type in sorted(stats.keys()):
        gt_dec  = GT_DECISION.get(gt_type, "FAIL")
        n_pass  = stats[gt_type]["PASS"]
        n_fail  = stats[gt_type]["FAIL"]
        total_t = n_pass + n_fail
        n_correct = stats[gt_type][gt_dec]
        pct = n_correct / total_t * 100 if total_t > 0 else 0
        print(f"{gt_type:<15} {n_pass:>8} {n_fail:>8} "
              f"{gt_dec:>15} {pct:>9.1f}%")

    print("=" * 60)

    # False positive / false negative analysis
    fp = sum(1 for p in pair_decisions
             if p["gt_decision"] == "PASS" and p["our_decision"] == "FAIL")
    fn = sum(1 for p in pair_decisions
             if p["gt_decision"] == "FAIL" and p["our_decision"] == "PASS")
    tp = sum(1 for p in pair_decisions
             if p["gt_decision"] == "FAIL" and p["our_decision"] == "FAIL")
    tn = sum(1 for p in pair_decisions
             if p["gt_decision"] == "PASS" and p["our_decision"] == "PASS")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\nRegression detection (FAIL = positive class):")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"\n  False positive rate: {fp}/{tn+fp} "
          f"({fp/(tn+fp)*100:.1f}% of benign flagged as regression)"
          if (tn + fp) > 0 else "")
    print(f"  False negative rate: {fn}/{tp+fn} "
          f"({fn/(tp+fn)*100:.1f}% of regressions missed)"
          if (tp + fn) > 0 else "")
    print("=" * 60)

    # Save results
    evaluation = {
        "fail_threshold":    fail_threshold,
        "total_pairs":       total,
        "correct":           correct,
        "accuracy":          correct / total if total > 0 else 0,
        "precision":         precision,
        "recall":            recall,
        "f1":                f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "per_type_stats":    {k: dict(v) for k, v in stats.items()},
        "pair_decisions":    pair_decisions,
    }
    out_path = output_dir / f"severity_results_{fail_threshold:.2f}.json"
    with open(out_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"\n[INFO] Results saved: {out_path}")

    return evaluation


# =============================================================================
# VISUALISATION
# =============================================================================

def visualise_severity(img1, img2, changes, output_path):
    """Draw severity scores on side-by-side visualisation."""
    vis1 = img1.copy()
    vis2 = img2.copy()

    COLOURS = {"high": (0, 0, 255), "medium": (0, 165, 255), "low": (0, 255, 0)}

    for change in changes:
        colour = COLOURS.get(change["severity_label"], (255, 255, 255))
        label  = (f"{change['change_type']} "
                  f"{change['severity_score']:.2f} "
                  f"[{change['decision']}]")

        if change.get("box_original"):
            x1, y1, x2, y2 = [int(v) for v in change["box_original"]]
            cv2.rectangle(vis1, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(vis1, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        if change.get("box_changed"):
            x1, y1, x2, y2 = [int(v) for v in change["box_changed"]]
            cv2.rectangle(vis2, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(vis2, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    combined = np.hstack([vis1, vis2])
    cv2.imwrite(str(output_path), combined)
    print(f"[INFO] Severity visualisation saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score detected changes as regression or benign"
    )
    parser.add_argument("--model",          required=True)
    parser.add_argument("--classifier",     required=True,
                        help="Path to classifier.pkl from step7")
    parser.add_argument("--test_dir",       default=None,
                        help="Test pairs directory (batch mode)")
    parser.add_argument("--test_manifest",  default=None,
                        help="Test manifest.json (batch mode)")
    parser.add_argument("--image1",         default=None,
                        help="Original image (single mode)")
    parser.add_argument("--image2",         default=None,
                        help="Changed image (single mode)")
    parser.add_argument("--output_dir",     default="./outputs/severity")
    parser.add_argument("--fail_threshold", type=float, default=0.5,
                        help="Severity score >= this = FAIL (default: 0.5)")
    parser.add_argument("--visualise",      action="store_true")
    args = parser.parse_args()

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Loading classifier: {args.classifier}")
    with open(args.classifier, "rb") as f:
        classifier_data = pickle.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image1 and args.image2:
        # Single pair mode
        changes = score_pair(
            model, classifier_data,
            args.image1, args.image2,
            args.fail_threshold
        )
        print(f"\n[INFO] Detected {len(changes)} change(s):")
        for c in changes:
            print(f"  {c['change_type']:<15} "
                  f"severity={c['severity_score']:.3f} "
                  f"[{c['severity_label']}]  →  {c['decision']}")

        if args.visualise and changes:
            img1 = cv2.imread(args.image1)
            img2 = cv2.imread(args.image2)
            stem = Path(args.image1).stem.replace("_original", "")
            visualise_severity(
                img1, img2, changes,
                output_dir / f"{stem}_severity.jpg"
            )

        overall = "FAIL" if any(c["decision"] == "FAIL" for c in changes) else "PASS"
        print(f"\n  Overall decision: {overall}")

    elif args.test_dir and args.test_manifest:
        # Batch evaluation mode
        evaluate_severity(
            model=model,
            classifier_data=classifier_data,
            test_dir=args.test_dir,
            manifest_path=args.test_manifest,
            output_dir=output_dir,
            fail_threshold=args.fail_threshold,
        )

    else:
        print("[ERROR] Provide --image1/--image2 or --test_dir/--test_manifest")
        parser.print_help()


if __name__ == "__main__":
    main()
