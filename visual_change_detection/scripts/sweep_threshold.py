"""
sweep_threshold.py
==================
Phase A1 — Threshold Sweep

Runs graph_matcher + evaluate at multiple similarity thresholds.
Prints a ranked summary table so you can pick the best threshold.

Usage:
    python scripts/sweep_threshold.py \
        --model      ./outputs/yolo_runs/rico_ui_v3/weights/best.pt \
        --pairs_dir  ./outputs/change_dataset_v2/test/pairs \
        --manifest   ./outputs/change_dataset_v2/test/manifest.json \
        --output_dir ./outputs/sweep/threshold

Time estimate: ~5-8 min per threshold on CPU, ~2-3 min with GPU.
Total: ~35-55 min for 7 thresholds.
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path


# =============================================================================
# THRESHOLDS TO SWEEP
# =============================================================================

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


# =============================================================================
# INLINE EVALUATION (no subprocess — faster, reuses loaded model)
# =============================================================================

def run_sweep(model_path, pairs_dir, manifest_path, output_dir):
    """Run full sweep inline — loads model once, sweeps all thresholds."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_dir = Path(pairs_dir)
    manifest_path = Path(manifest_path)

    # ── Imports (after arg parse so errors are clear) ──────────
    print("[INFO] Loading pipeline modules...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from ultralytics import YOLO
    import cv2
    import numpy as np
    from collections import defaultdict

    from graph_builder import build_graph, run_yolo
    from graph_matcher import match_graphs, detect_changes

    # ── Load model once ────────────────────────────────────────
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    # ── Load manifest ──────────────────────────────────────────
    with open(manifest_path) as f:
        manifest = json.load(f)
    pairs = manifest["pairs"]
    print(f"[INFO] Loaded {len(pairs)} test pairs")

    # ── Pre-build all graphs (do this once, reuse across thresholds) ──
    print(f"\n[INFO] Pre-building graphs for all {len(pairs)} pairs...")
    print("       (This runs YOLO once — avoids repeating detection per threshold)")

    graph_cache = {}   # pair_id → (G1, G2, gt_boxes, gt_change_type)
    skipped = 0

    for i, pair in enumerate(pairs):
        pid = pair["pair_id"]
        orig_path = pairs_dir / f"{pid}_original.jpg"
        chng_path = pairs_dir / f"{pid}_changed.jpg"
        gt_path = pairs_dir / f"{pid}_gt.json"

        if not orig_path.exists() or not chng_path.exists():
            skipped += 1
            continue

        img1 = cv2.imread(str(orig_path))
        img2 = cv2.imread(str(chng_path))
        if img1 is None or img2 is None:
            skipped += 1
            continue

        with open(gt_path) as f:
            gt = json.load(f)

        det1 = run_yolo(model, img1)
        det2 = run_yolo(model, img2)
        G1 = build_graph(img1, det1, extract_ocr=False)
        G2 = build_graph(img2, det2, extract_ocr=False)

        gt_boxes = [
            c["changed_box"] for c in gt.get("changes", [])
            if c.get("changed_box")
        ]

        graph_cache[pid] = {
            "G1":             G1,
            "G2":             G2,
            "gt_boxes":       gt_boxes,
            "gt_change_type": gt.get("change_type", "unknown"),
        }

        if (i + 1) % 50 == 0:
            print(f"  Built {i+1}/{len(pairs)} graphs...")

    print(f"[INFO] Graph cache ready: {len(graph_cache)} pairs "
          f"({skipped} skipped)")

    # ── Sweep thresholds ───────────────────────────────────────
    summary = []

    for threshold in THRESHOLDS:
        print(f"\n{'='*55}")
        print(f"  THRESHOLD = {threshold:.2f}")
        print(f"{'='*55}")

        thresh_dir = output_dir / f"thresh_{threshold:.2f}"
        thresh_dir.mkdir(parents=True, exist_ok=True)

        all_results = []

        for pid, cache in graph_cache.items():
            G1 = cache["G1"]
            G2 = cache["G2"]

            changed_boxes, change_details, _ = detect_changes(
                G1, G2, threshold
            )

            all_results.append({
                "pair_id":        pid,
                "changed_boxes":  changed_boxes,
                "gt_boxes":       cache["gt_boxes"],
                "gt_change_type": cache["gt_change_type"],
            })

        # Save results for this threshold
        with open(thresh_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Compute metrics at IOU 0.25, 0.50, 0.75
        metrics = {}
        for iou in [0.25, 0.50, 0.75]:
            metrics[iou] = compute_metrics(all_results, iou)

        # Per-type breakdown at IOU=0.50
        per_type = compute_per_type_metrics(all_results, iou_threshold=0.50)

        # Print
        print(f"  {'IOU':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
        print(f"  {'-'*40}")
        for iou in [0.25, 0.50, 0.75]:
            m = metrics[iou]
            print(f"  {iou:>6.2f}  {m['precision']:>10.4f}  "
                  f"{m['recall']:>8.4f}  {m['f1']:>8.4f}")

        print(f"\n  Per-type F1 @ IOU=0.50:")
        for ct, m in sorted(per_type.items()):
            bar = "█" * int(m['f1'] * 20)
            print(f"    {ct:<15} {m['f1']:.4f}  {bar}")

        summary.append({
            "threshold": threshold,
            "metrics":   {str(iou): metrics[iou] for iou in metrics},
            "per_type":  per_type,
        })

    # ── Summary table ──────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  SWEEP SUMMARY — ranked by F1 @ IOU=0.50")
    print(f"{'='*65}")
    print(f"  {'Threshold':>10}  {'F1@0.25':>8}  {'F1@0.50':>8}  "
          f"{'F1@0.75':>8}  {'P@0.50':>8}  {'R@0.50':>8}")
    print(f"  {'-'*60}")

    ranked = sorted(summary, key=lambda x: x["metrics"]["0.5"]["f1"],
                    reverse=True)
    for r in ranked:
        t = r["threshold"]
        m25 = r["metrics"]["0.25"]["f1"]
        m50 = r["metrics"]["0.5"]["f1"]
        m75 = r["metrics"]["0.75"]["f1"]
        p50 = r["metrics"]["0.5"]["precision"]
        rc50 = r["metrics"]["0.5"]["recall"]
        marker = "  ← BEST" if r == ranked[0] else ""
        print(f"  {t:>10.2f}  {m25:>8.4f}  {m50:>8.4f}  "
              f"{m75:>8.4f}  {p50:>8.4f}  {rc50:>8.4f}{marker}")

    print(f"{'='*65}")
    best = ranked[0]
    print(f"\n  Best threshold: {best['threshold']:.2f} "
          f"→ F1@0.50 = {best['metrics']['0.5']['f1']:.4f}")

    # Save full summary
    with open(output_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] Full results saved: {output_dir}/sweep_summary.json")
    print(
        f"[DONE] Per-threshold results: {output_dir}/thresh_*/all_results.json")

    return ranked[0]["threshold"]


# =============================================================================
# METRIC HELPERS (inline — no import dependency on evaluate.py)
# =============================================================================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def compute_metrics(all_results, iou_threshold=0.5):
    tp = fp = fn = 0
    for result in all_results:
        pred = result.get("changed_boxes", [])
        gt = result.get("gt_boxes", [])
        if not gt and not pred:
            continue
        matched_gt = set()
        for p in pred:
            best_iou, best_gt = 0.0, -1
            for gi, g in enumerate(gt):
                if gi in matched_gt:
                    continue
                iou = compute_iou(p, g)
                if iou > best_iou:
                    best_iou, best_gt = iou, gi
            if best_iou >= iou_threshold and best_gt >= 0:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1
        fn += len(gt) - len(matched_gt)
    precision = tp / (tp+fp) if (tp+fp) > 0 else 0.0
    recall = tp / (tp+fn) if (tp+fn) > 0 else 0.0
    f1 = 2*precision*recall / \
        (precision+recall) if (precision+recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def compute_per_type_metrics(all_results, iou_threshold=0.5):
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r.get("gt_change_type", "unknown")].append(r)
    return {ct: compute_metrics(res, iou_threshold)
            for ct, res in by_type.items()}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sweep similarity threshold for graph matcher"
    )
    parser.add_argument("--model",       required=True,
                        help="Path to best.pt (rico_ui_v3)")
    parser.add_argument("--pairs_dir",   required=True,
                        help="Test pairs directory")
    parser.add_argument("--manifest",    required=True,
                        help="Test manifest.json")
    parser.add_argument("--output_dir",  default="./outputs/sweep/threshold")
    args = parser.parse_args()

    best = run_sweep(
        model_path=args.model,
        pairs_dir=args.pairs_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print(
        f"\n[NEXT] Use --threshold {best:.2f} in your next graph_matcher run")
    print(f"[NEXT] Then proceed to Phase A3 — weight sweep")


if __name__ == "__main__":
    main()
