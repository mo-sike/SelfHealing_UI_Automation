"""
step9_demo.py
=============
Phase 2 - Week 8  |  Run on: LOCAL PC

What this does:
    End-to-end pipeline demo.
    Takes two screenshots (before / after) and produces:
        - Console summary table of detected changes
        - Side-by-side visualisation with severity colour coding
        - JSON result file
        - Overall PASS / FAIL verdict

    Colour coding in visualisation:
        RED    — high severity  (score >= 0.65)  → likely FAIL
        ORANGE — medium severity (score >= 0.35)
        GREEN  — low severity   (score <  0.35)  → likely PASS

    Full pipeline per pair:
        YOLO detection
            → KNN graph construction
            → recursive graph matching
            → change type classification  (step7 Random Forest)
            → severity scoring            (step8 rules)
            → PASS / FAIL decision

Usage:
    # Single pair
    python scripts/step9_demo.py ^
        --model          ./outputs/yolo_runs/rico_ui_v2/weights/best.pt ^
        --classifier     ./outputs/classifier/classifier.pkl ^
        --image1         ./outputs/change_dataset/test/pairs/10703_v00_original.jpg ^
        --image2         ./outputs/change_dataset/test/pairs/10703_v00_changed.jpg ^
        --output_dir     ./outputs/demo ^
        --fail_threshold 0.5 ^
        --threshold      0.75

    # Batch mode — run on N random test pairs and show aggregate results
    python scripts/step9_demo.py ^
        --model          ./outputs/yolo_runs/rico_ui_v2/weights/best.pt ^
        --classifier     ./outputs/classifier/classifier.pkl ^
        --pairs_dir      ./outputs/change_dataset/test/pairs ^
        --manifest       ./outputs/change_dataset/test/manifest.json ^
        --output_dir     ./outputs/demo ^
        --fail_threshold 0.5 ^
        --threshold      0.75 ^
        --n_pairs        20 ^
        --visualise
"""

import cv2
import json
import pickle
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).parent))
from graph_builder import build_graph, run_yolo
from graph_matcher import (
    match_graphs, phash_similarity, colour_similarity,
    position_similarity, RELOCATE_POSITION_THRESH,
)
from step7_classifier import compute_bbox_area_ratio, compute_centre_distance


def compute_region_pixel_diff(img1, img2, box1, box2):
    """
    Mean absolute pixel difference between two crops.
    Normalised to [0, 1]. Catches colour shifts phash misses.
    Added for classifier_v3 compatibility (7-feature vector).
    """
    if box1 is None or box2 is None:
        return 0.0
    x1, y1, x2, y2 = [int(v) for v in box1]
    crop1 = img1[y1:y2, x1:x2]
    x1, y1, x2, y2 = [int(v) for v in box2]
    crop2 = img2[y1:y2, x1:x2]
    if crop1.size == 0 or crop2.size == 0:
        return 0.0
    h = min(crop1.shape[0], crop2.shape[0])
    w = min(crop1.shape[1], crop2.shape[1])
    if h < 2 or w < 2:
        return 0.0
    c1 = cv2.resize(crop1, (w, h)).astype(float)
    c2 = cv2.resize(crop2, (w, h)).astype(float)
    return float(np.mean(np.abs(c1 - c2)) / 255.0)
from step8_severity import compute_severity, severity_label


# =============================================================================
# COLOURS
# =============================================================================

SEVERITY_COLOURS = {
    "high":   (0,   0,   255),   # red
    "medium": (0,   165, 255),   # orange
    "low":    (0,   255, 0),     # green
}

CHANGE_TYPE_COLOURS = {
    "remove":       (0,   0,   255),
    "add":          (255, 100, 0),
    "resize":       (255, 0,   255),
    "relocate":     (0,   200, 255),
    "color_change": (0,   255, 200),
}


# =============================================================================
# CORE PIPELINE
# =============================================================================

def run_pipeline(model, classifier_data, img1_path, img2_path,
                 fail_threshold=0.5, sim_threshold=0.6):
    """
    Full end-to-end pipeline for one image pair.

    Returns:
        changes  : list of change dicts with type, severity, decision
        decision : "PASS" or "FAIL"
        meta     : graph stats dict
    """
    clf = classifier_data["classifier"]
    le  = classifier_data["label_encoder"]

    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        print(f"[ERROR] Cannot read images")
        return [], "ERROR", {}

    h, w = img1.shape[:2]

    # ── Step 4: YOLO + graph construction ─────────────────────
    det1 = run_yolo(model, img1)
    det2 = run_yolo(model, img2)
    G1   = build_graph(img1, det1, extract_ocr=False, extract_clip=True)
    G2   = build_graph(img2, det2, extract_ocr=False, extract_clip=True)

    meta = {
        "image_size":   [w, h],
        "g1_nodes":     G1.number_of_nodes(),
        "g1_edges":     G1.number_of_edges(),
        "g2_nodes":     G2.number_of_nodes(),
        "g2_edges":     G2.number_of_edges(),
        "g1_detections": len(det1),
        "g2_detections": len(det2),
    }

    if G1.number_of_nodes() == 0 and G2.number_of_nodes() == 0:
        return [], "PASS", meta

    # ── Step 5: Graph matching ─────────────────────────────────
    matches, changed_nodes, added_nodes, removed_nodes, sim_matrix = \
        match_graphs(G1, G2, similarity_threshold=sim_threshold)

    changes = []

    # Track flagged G1 nodes to avoid double-counting with C3
    flagged_g1 = set()

    # Changed nodes (matched but below threshold)
    for n1_id, n2_id, sim in changed_nodes:
        flagged_g1.add(n1_id)
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
        # 8-feature vector — is_matched=1.0 (this node was matched)
        features    = [phash_dist, colour_dist, area_ratio,
                       centre_dist, class_match, float(sim), pixel_diff, 1.0]

        # ── Step 7: Classify ───────────────────────────────────
        change_type = le.inverse_transform(clf.predict([features]))[0]

        # ── Step 8: Score severity ─────────────────────────────
        score    = compute_severity(change_type, features)
        label    = severity_label(score)
        decision = "FAIL" if score >= fail_threshold else "PASS"

        changes.append({
            "match_type":     "changed",
            "change_type":    change_type,
            "class":          d1["class_name"],
            "box_original":   d1["bbox"],
            "box_changed":    d2["bbox"],
            "similarity":     float(sim),
            "features":       features,
            "severity_score": score,
            "severity_label": label,
            "decision":       decision,
        })

    # Removed nodes — is_matched=0.0
    for n1_id in removed_nodes:
        flagged_g1.add(n1_id)
        d1       = G1.nodes[n1_id]
        features = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        score    = compute_severity("remove", features)
        label    = severity_label(score)
        decision = "FAIL" if score >= fail_threshold else "PASS"

        changes.append({
            "match_type":     "removed",
            "change_type":    "remove",
            "class":          d1["class_name"],
            "box_original":   d1["bbox"],
            "box_changed":    None,
            "similarity":     0.0,
            "features":       features,
            "severity_score": score,
            "severity_label": label,
            "decision":       decision,
        })

    # Added nodes — is_matched=0.0
    for n2_id in added_nodes:
        d2       = G2.nodes[n2_id]
        features = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        score    = compute_severity("add", features)
        label    = severity_label(score)
        decision = "FAIL" if score >= fail_threshold else "PASS"

        changes.append({
            "match_type":     "added",
            "change_type":    "add",
            "class":          d2["class_name"],
            "box_original":   None,
            "box_changed":    d2["bbox"],
            "similarity":     0.0,
            "features":       features,
            "severity_score": score,
            "severity_label": label,
            "decision":       decision,
        })

    # ── Phase C3: post-match relocate detection ────────────────
    # Good matches (sim >= threshold) with a large position shift =
    # relocated element.  Run through classifier so it can be labelled
    # "relocate" and given an appropriate severity score.
    for n1_id, n2_id, sim in matches:
        if sim < sim_threshold:
            continue                    # already in changed_nodes
        if n1_id in flagged_g1:
            continue                    # already handled
        d1 = G1.nodes[n1_id]
        d2 = G2.nodes[n2_id]
        pos_sim = position_similarity(d1.get("bbox"), d2.get("bbox"))
        if pos_sim >= RELOCATE_POSITION_THRESH:
            continue                    # not relocated enough

        flagged_g1.add(n1_id)
        phash_dist  = 1.0 - phash_similarity(d1.get("phash"), d2.get("phash"))
        colour_dist = 1.0 - colour_similarity(
            d1.get("mean_colour"), d2.get("mean_colour")
        )
        area_ratio  = compute_bbox_area_ratio(d1["bbox"], d2["bbox"])
        centre_dist = compute_centre_distance(d1["bbox"], d2["bbox"], w, h)
        class_match = 1.0 if d1["class_name"] == d2["class_name"] else 0.0
        pixel_diff  = compute_region_pixel_diff(img1, img2, d1["bbox"], d2["bbox"])
        features    = [phash_dist, colour_dist, area_ratio,
                       centre_dist, class_match, float(sim), pixel_diff, 1.0]

        change_type = le.inverse_transform(clf.predict([features]))[0]
        score    = compute_severity(change_type, features)
        label    = severity_label(score)
        decision = "FAIL" if score >= fail_threshold else "PASS"

        changes.append({
            "match_type":     "relocated",
            "change_type":    change_type,
            "class":          d1["class_name"],
            "box_original":   d1["bbox"],
            "box_changed":    d2["bbox"],
            "similarity":     float(sim),
            "position_sim":   float(pos_sim),
            "features":       features,
            "severity_score": score,
            "severity_label": label,
            "decision":       decision,
        })

    overall = "FAIL" if any(c["decision"] == "FAIL" for c in changes) else "PASS"
    return changes, overall, meta


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_summary(pair_id, changes, overall, meta, fail_threshold):
    """Print a clean summary table to console."""
    print("\n" + "=" * 65)
    print(f"  VISUAL CHANGE DETECTION REPORT")
    print(f"  Pair:      {pair_id}")
    print(f"  Threshold: {fail_threshold}  |  "
          f"G1: {meta['g1_nodes']} nodes  G2: {meta['g2_nodes']} nodes")
    print("=" * 65)

    if not changes:
        print("  No changes detected.")
    else:
        print(f"  {'#':<4} {'Type':<15} {'Class':<12} "
              f"{'Score':>7} {'Severity':<10} {'Decision'}")
        print("  " + "-" * 60)
        for i, c in enumerate(changes, 1):
            print(f"  {i:<4} {c['change_type']:<15} {c['class']:<12} "
                  f"{c['severity_score']:>7.3f} {c['severity_label']:<10} "
                  f"{c['decision']}")

    print("  " + "-" * 65)
    verdict_str = f"  ▶  OVERALL VERDICT:  {overall}"
    print(verdict_str)
    print("=" * 65 + "\n")


# =============================================================================
# VISUALISATION
# =============================================================================

def visualise_pair(img1, img2, changes, overall, pair_id, output_path):
    """
    Side-by-side visualisation with severity colour-coded boxes.
    Red = high, Orange = medium, Green = low.
    """
    vis1 = img1.copy()
    vis2 = img2.copy()
    h, w = img1.shape[:2]

    for c in changes:
        colour = SEVERITY_COLOURS.get(c["severity_label"], (255, 255, 255))
        label  = f"{c['change_type']} {c['severity_score']:.2f} [{c['decision']}]"

        if c.get("box_original"):
            x1, y1, x2, y2 = [int(v) for v in c["box_original"]]
            cv2.rectangle(vis1, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(vis1, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1,
                        cv2.LINE_AA)

        if c.get("box_changed"):
            x1, y1, x2, y2 = [int(v) for v in c["box_changed"]]
            cv2.rectangle(vis2, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(vis2, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1,
                        cv2.LINE_AA)

    # Header bars
    verdict_colour = (0, 0, 220) if overall == "FAIL" else (0, 180, 0)
    header_h = 36

    def add_header(img, text, colour):
        out = np.zeros((header_h, img.shape[1], 3), dtype=np.uint8)
        out[:] = colour
        cv2.putText(out, text, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([out, img])

    n_fail = sum(1 for c in changes if c["decision"] == "FAIL")
    vis1 = add_header(vis1, "ORIGINAL", (50, 50, 50))
    vis2 = add_header(vis2,
                      f"CHANGED  |  {len(changes)} change(s) detected  "
                      f"|  {n_fail} FAIL  |  {overall}",
                      verdict_colour)

    combined = np.hstack([vis1, vis2])
    cv2.imwrite(str(output_path), combined)
    print(f"[INFO] Visualisation saved: {output_path}")


# =============================================================================
# SINGLE PAIR MODE
# =============================================================================

def run_single(model, classifier_data, img1_path, img2_path,
               output_dir, fail_threshold, visualise, sim_threshold=0.6):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(img1_path).stem.replace("_original", "")

    changes, overall, meta = run_pipeline(
        model, classifier_data, img1_path, img2_path,
        fail_threshold, sim_threshold
    )

    print_summary(stem, changes, overall, meta, fail_threshold)

    # Save JSON
    result = {
        "pair_id":       stem,
        "overall":       overall,
        "fail_threshold": fail_threshold,
        "n_changes":     len(changes),
        "changes":       changes,
        "meta":          meta,
    }
    json_path = output_dir / f"{stem}_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Result saved: {json_path}")

    # Visualise
    if visualise:
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        if img1 is not None and img2 is not None:
            visualise_pair(
                img1, img2, changes, overall, stem,
                output_dir / f"{stem}_demo.jpg"
            )

    return result


# =============================================================================
# BATCH MODE
# =============================================================================

def run_batch(model, classifier_data, pairs_dir, manifest_path,
              output_dir, fail_threshold, n_pairs, visualise, seed=42,
              sim_threshold=0.6):
    """
    Run pipeline on N pairs from manifest.
    Prints aggregate results table — useful for thesis defence.
    """
    pairs_dir  = Path(pairs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_pairs = manifest["pairs"]
    random.seed(seed)
    selected  = random.sample(all_pairs, min(n_pairs, len(all_pairs)))

    print(f"\n[INFO] Running batch demo on {len(selected)} pairs "
          f"(fail_threshold={fail_threshold})...")

    # Ground truth severity mapping
    GT_DECISION = {
        "remove": "FAIL", "add": "FAIL", "resize": "FAIL",
        "relocate": "FAIL", "color_change": "PASS",
    }

    results      = []
    type_stats   = defaultdict(lambda: {"correct": 0, "total": 0})
    tp = fp = tn = fn = 0

    for i, pair in enumerate(selected):
        pid     = pair["pair_id"]
        gt_type = pair["change_type"]
        orig    = pairs_dir / f"{pid}_original.jpg"
        chng    = pairs_dir / f"{pid}_changed.jpg"

        if not orig.exists() or not chng.exists():
            continue

        changes, overall, meta = run_pipeline(
            model, classifier_data, orig, chng,
            fail_threshold, sim_threshold
        )

        gt_decision = GT_DECISION.get(gt_type, "FAIL")
        correct     = overall == gt_decision

        type_stats[gt_type]["total"]   += 1
        type_stats[gt_type]["correct"] += int(correct)

        if gt_decision == "FAIL" and overall == "FAIL":   tp += 1
        elif gt_decision == "PASS" and overall == "FAIL": fp += 1
        elif gt_decision == "PASS" and overall == "PASS": tn += 1
        elif gt_decision == "FAIL" and overall == "PASS": fn += 1

        results.append({
            "pair_id": pid, "gt_type": gt_type,
            "gt_decision": gt_decision, "our_decision": overall,
            "correct": correct, "n_changes": len(changes),
        })

        # Visualise if requested
        if visualise:
            img1 = cv2.imread(str(orig))
            img2 = cv2.imread(str(chng))
            if img1 is not None and img2 is not None:
                visualise_pair(
                    img1, img2, changes, overall, pid,
                    output_dir / f"{pid}_demo.jpg"
                )

        status = "✓" if correct else "✗"
        print(f"  [{status}] {pid:<25} gt={gt_type:<15} "
              f"pred={overall}  changes={len(changes)}")

    # ── Aggregate results ──────────────────────────────────────
    total   = len(results)
    correct_total = sum(1 for r in results if r["correct"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print("\n" + "=" * 65)
    print("  BATCH DEMO — AGGREGATE RESULTS")
    print(f"  Pairs: {total}  |  Fail threshold: {fail_threshold}")
    print("=" * 65)
    print(f"\n  Overall accuracy: {correct_total}/{total} = "
          f"{correct_total/total:.4f}" if total > 0 else "")
    print(f"\n  {'Change Type':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "-" * 45)
    for gt_type in sorted(type_stats.keys()):
        s   = type_stats[gt_type]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"  {gt_type:<15} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    print(f"\n  Regression detection (FAIL = positive class):")
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1        : {f1:.4f}")
    print("=" * 65)

    # Save batch results
    batch_result = {
        "n_pairs":       total,
        "fail_threshold": fail_threshold,
        "accuracy":      correct_total / total if total > 0 else 0,
        "precision":     precision,
        "recall":        recall,
        "f1":            f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "per_type":      {k: dict(v) for k, v in type_stats.items()},
        "pairs":         results,
    }
    out_path = output_dir / "batch_results.json"
    with open(out_path, "w") as f:
        json.dump(batch_result, f, indent=2)
    print(f"\n[INFO] Batch results saved: {out_path}")

    return batch_result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end visual change detection demo"
    )
    parser.add_argument("--model",          required=True,
                        help="Path to best.pt")
    parser.add_argument("--classifier",     required=True,
                        help="Path to classifier.pkl from step7")

    # Single pair mode
    parser.add_argument("--image1",         default=None,
                        help="Original screenshot path")
    parser.add_argument("--image2",         default=None,
                        help="Changed screenshot path")

    # Batch mode
    parser.add_argument("--pairs_dir",      default=None,
                        help="Test pairs directory (batch mode)")
    parser.add_argument("--manifest",       default=None,
                        help="manifest.json (batch mode)")
    parser.add_argument("--n_pairs",        type=int, default=20,
                        help="Number of pairs to sample in batch mode")

    # Shared
    parser.add_argument("--output_dir",     default="./outputs/demo")
    parser.add_argument("--fail_threshold", type=float, default=0.5,
                        help="Severity score >= this = FAIL (default: 0.5)")
    parser.add_argument("--threshold",      type=float, default=0.6,
                        help="Graph similarity threshold — nodes below this = changed (default: 0.6, try 0.7-0.8 to catch subtle changes)")
    parser.add_argument("--visualise",      action="store_true",
                        help="Save visualisation images")
    args = parser.parse_args()

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Loading classifier: {args.classifier}")
    with open(args.classifier, "rb") as f:
        classifier_data = pickle.load(f)

    if args.image1 and args.image2:
        run_single(
            model=model,
            classifier_data=classifier_data,
            img1_path=args.image1,
            img2_path=args.image2,
            output_dir=args.output_dir,
            fail_threshold=args.fail_threshold,
            visualise=args.visualise,
            sim_threshold=args.threshold,
        )

    elif args.pairs_dir and args.manifest:
        run_batch(
            model=model,
            classifier_data=classifier_data,
            pairs_dir=args.pairs_dir,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            fail_threshold=args.fail_threshold,
            n_pairs=args.n_pairs,
            visualise=args.visualise,
            sim_threshold=args.threshold,
        )

    else:
        print("[ERROR] Provide --image1/--image2 OR --pairs_dir/--manifest")
        parser.print_help()


if __name__ == "__main__":
    main()
