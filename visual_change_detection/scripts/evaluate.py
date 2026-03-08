"""
step6_evaluate.py
=================
Phase 1 - Week 4  |  Run on: LOCAL PC

What this does:
    Computes Precision, Recall, F-score at IOU thresholds 0.25, 0.50, 0.75.
    Mirrors Tables 2, 3, 4 from the base paper (Moradi et al.)

    Also computes:
        - Per change-type breakdown
        - Comparison vs pixel-wise baseline
        - Summary table ready for thesis

Usage:
    # Evaluate graph pipeline results
    python scripts/step6_evaluate.py \
        --results_dir ./outputs/results/test \
        --output_dir  ./outputs/evaluation

    # Also run pixel-wise baseline for comparison
    python scripts/step6_evaluate.py \
        --results_dir ./outputs/results/test \
        --output_dir  ./outputs/evaluation \
        --run_baseline \
        --pairs_dir   ./outputs/change_dataset/test/pairs \
        --manifest    ./outputs/change_dataset/test/manifest.json
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# =============================================================================
# IOU
# =============================================================================

def compute_iou(box1, box2):
    """
    Compute Intersection over Union between two boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(all_results, iou_threshold=0.5):
    """
    Compute Precision, Recall, F-score at given IOU threshold.

    Args:
        all_results    : list of result dicts from step5
        iou_threshold  : IOU threshold for a detection to count as TP

    Returns:
        dict with precision, recall, f1, tp, fp, fn counts
    """
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for result in all_results:
        pred_boxes = result.get("changed_boxes", [])
        gt_boxes   = result.get("gt_boxes", [])

        if not gt_boxes and not pred_boxes:
            continue  # true negative — no change, none predicted

        matched_gt = set()

        for pred in pred_boxes:
            best_iou = 0.0
            best_gt  = -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt  = gi

            if best_iou >= iou_threshold and best_gt >= 0:
                tp_total  += 1
                matched_gt.add(best_gt)
            else:
                fp_total += 1

        fn_total += len(gt_boxes) - len(matched_gt)

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        tp_total,
        "fp":        fp_total,
        "fn":        fn_total,
    }


def compute_per_type_metrics(all_results, iou_threshold=0.5):
    """Compute metrics broken down by change type."""
    by_type = defaultdict(list)
    for result in all_results:
        t = result.get("gt_change_type", "unknown")
        by_type[t].append(result)

    per_type = {}
    for change_type, results in by_type.items():
        per_type[change_type] = compute_metrics(results, iou_threshold)

    return per_type


# =============================================================================
# PIXEL-WISE BASELINE
# =============================================================================

def pixel_diff_boxes(img1, img2, threshold=30, min_area=500):
    """
    Simple pixel-wise difference baseline.
    Returns bounding boxes of significantly changed regions.
    """
    # Convert to grayscale
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(g1, g2)

    # Threshold
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask   = cv2.dilate(mask, kernel, iterations=2)
    mask   = cv2.erode(mask,  kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            boxes.append([x, y, x + w, y + h])

    return boxes


def run_baseline_evaluation(pairs_dir, manifest_path, output_dir):
    """Run pixel-wise baseline on all test pairs."""
    pairs_dir  = Path(pairs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"[INFO] Running pixel-wise baseline on {len(manifest['pairs'])} pairs...")
    baseline_results = []

    for pair in manifest["pairs"]:
        pid     = pair["pair_id"]
        orig    = cv2.imread(str(pairs_dir / f"{pid}_original.jpg"))
        changed = cv2.imread(str(pairs_dir / f"{pid}_changed.jpg"))

        with open(pairs_dir / f"{pid}_gt.json") as f:
            gt = json.load(f)

        if orig is None or changed is None:
            continue

        pred_boxes = pixel_diff_boxes(orig, changed)
        gt_boxes   = [
            c["changed_box"] for c in gt["changes"]
            if c.get("changed_box")
        ]

        baseline_results.append({
            "pair_id":         pid,
            "changed_boxes":   pred_boxes,
            "gt_boxes":        gt_boxes,
            "gt_change_type":  gt["change_type"],
        })

    return baseline_results


# =============================================================================
# REPORT
# =============================================================================

def print_results_table(graph_metrics, baseline_metrics=None):
    """Print evaluation results in thesis-ready table format."""
    thresholds = [0.25, 0.50, 0.75]

    print("\n" + "=" * 70)
    print("CHANGE DETECTION EVALUATION — PHASE 1 RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<25} {'IOU':>6} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print("-" * 70)

    for iou in thresholds:
        m = graph_metrics[iou]
        prefix = "Graph-based (ours)" if iou == thresholds[0] else ""
        print(f"{prefix:<25} {iou:>6.2f} {m['precision']:>12.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}")

    if baseline_metrics:
        print()
        for iou in thresholds:
            m = baseline_metrics[iou]
            prefix = "Pixel-wise (baseline)" if iou == thresholds[0] else ""
            print(f"{prefix:<25} {iou:>6.2f} {m['precision']:>12.4f} "
                  f"{m['recall']:>10.4f} {m['f1']:>10.4f}")

    print("=" * 70)


def print_per_type_table(per_type_metrics, iou=0.5):
    """Print per change-type breakdown."""
    print(f"\n{'Change Type':<20} {'Precision':>12} {'Recall':>10} "
          f"{'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 70)
    for change_type, m in sorted(per_type_metrics.items()):
        print(f"{change_type:<20} {m['precision']:>12.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f} "
              f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6}")
    print("-" * 70)


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(results_dir, output_dir, pairs_dir=None,
                   manifest_path=None, run_baseline=False):
    results_dir = Path(results_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_file = results_dir / "all_results.json"
    if not results_file.exists():
        print(f"[ERROR] Results file not found: {results_file}")
        print("[INFO]  Run step5_graph_matcher.py first")
        return

    with open(results_file) as f:
        all_results = json.load(f)

    print(f"[INFO] Loaded {len(all_results)} pair results")

    # ── Graph-based metrics at 3 IOU thresholds ────────────────
    graph_metrics = {}
    for iou in [0.25, 0.50, 0.75]:
        graph_metrics[iou] = compute_metrics(all_results, iou)

    # ── Per change-type breakdown at IOU=0.5 ──────────────────
    per_type = compute_per_type_metrics(all_results, iou_threshold=0.5)

    # ── Pixel-wise baseline (optional) ────────────────────────
    baseline_metrics = None
    if run_baseline and pairs_dir and manifest_path:
        baseline_results = run_baseline_evaluation(
            pairs_dir, manifest_path, output_dir / "baseline"
        )
        baseline_metrics = {}
        for iou in [0.25, 0.50, 0.75]:
            baseline_metrics[iou] = compute_metrics(baseline_results, iou)

    # ── Print tables ───────────────────────────────────────────
    print_results_table(graph_metrics, baseline_metrics)
    print(f"\n--- Per Change Type (IOU=0.50) ---")
    print_per_type_table(per_type, iou=0.5)

    # ── Save full results ──────────────────────────────────────
    evaluation = {
        "graph_based": {
            str(iou): graph_metrics[iou] for iou in graph_metrics
        },
        "per_change_type": per_type,
        "baseline": {
            str(iou): baseline_metrics[iou]
            for iou in baseline_metrics
        } if baseline_metrics else None,
        "n_pairs": len(all_results)
    }
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"\n[DONE] Full results saved: {output_dir}/evaluation_results.json")
    return evaluation


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate change detection pipeline"
    )
    parser.add_argument("--results_dir",  required=True,
                        help="Directory with all_results.json from step5")
    parser.add_argument("--output_dir",   default="./outputs/evaluation")
    parser.add_argument("--run_baseline", action="store_true",
                        help="Also run pixel-wise baseline for comparison")
    parser.add_argument("--pairs_dir",    default=None,
                        help="Test pairs directory (needed for baseline)")
    parser.add_argument("--manifest",     default=None,
                        help="manifest.json (needed for baseline)")
    args = parser.parse_args()

    run_evaluation(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        pairs_dir=args.pairs_dir,
        manifest_path=args.manifest,
        run_baseline=args.run_baseline
    )

if __name__ == "__main__":
    main()
