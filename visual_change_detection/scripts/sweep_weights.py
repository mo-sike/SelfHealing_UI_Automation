"""
sweep_weights.py
================
Phase A3 — Similarity Weight Sweep

Tests 8 weight configurations for the graph matcher at threshold=0.85.
Builds graphs once, sweeps all configs in-memory.

Each config adjusts: visual, colour, text, class, structural weights.
All weights in each config sum to 1.0.

Usage:
    python scripts/sweep_weights.py --model ./outputs/yolo_runs/rico_ui_v3/weights/best.pt --pairs_dir ./outputs/change_dataset_v2/test/pairs --manifest ./outputs/change_dataset_v2/test/manifest.json --output_dir ./outputs/sweep/weights

Time estimate: ~15-25 min total (graphs built once, ~2 min per config).
"""

import json
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict


# =============================================================================
# WEIGHT CONFIGS TO SWEEP
# Each entry: (name, weights_dict)
# All weights must sum to 1.0
# =============================================================================

WEIGHT_CONFIGS = [
    (
        "current_baseline",
        {"visual": 0.40, "colour": 0.25, "text": 0.05,
         "class": 0.20, "structural": 0.10}
    ),
    (
        "boost_structural",         # structural is too low at 0.10
        {"visual": 0.35, "colour": 0.15, "text": 0.05,
         "class": 0.20, "structural": 0.25}
    ),
    (
        "paper_emphasis",           # base paper: visual + structural dominant
        {"visual": 0.45, "colour": 0.10, "text": 0.10,
         "class": 0.15, "structural": 0.20}
    ),
    (
        "class_heavy",              # class label is a strong signal for UI
        {"visual": 0.30, "colour": 0.10, "text": 0.05,
         "class": 0.35, "structural": 0.20}
    ),
    (
        "visual_structural",        # drop colour, boost visual+structural
        {"visual": 0.40, "colour": 0.05, "text": 0.05,
         "class": 0.20, "structural": 0.30}
    ),
    (
        "balanced",                 # equalise all meaningful signals
        {"visual": 0.30, "colour": 0.15, "text": 0.10,
         "class": 0.25, "structural": 0.20}
    ),
    (
        "structural_dominant",      # max structural — context is everything
        {"visual": 0.25, "colour": 0.10, "text": 0.05,
         "class": 0.25, "structural": 0.35}
    ),
    (
        "no_colour",                # colour is noisy for RICO mobile UIs
        {"visual": 0.45, "colour": 0.00, "text": 0.05,
         "class": 0.25, "structural": 0.25}
    ),
]

# Fixed threshold from Phase A1 result
THRESHOLD = 0.85


# =============================================================================
# METRIC HELPERS
# =============================================================================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def compute_metrics(all_results, iou_threshold=0.5):
    tp = fp = fn = 0
    for result in all_results:
        pred = result.get("changed_boxes", [])
        gt   = result.get("gt_boxes", [])
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
                tp += 1; matched_gt.add(best_gt)
            else:
                fp += 1
        fn += len(gt) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def compute_per_type_metrics(all_results, iou_threshold=0.5):
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r.get("gt_change_type", "unknown")].append(r)
    return {ct: compute_metrics(res, iou_threshold)
            for ct, res in by_type.items()}


# =============================================================================
# SIMILARITY FUNCTIONS (self-contained — no import from graph_matcher)
# =============================================================================

def phash_similarity(hash1, hash2):
    if hash1 is None or hash2 is None:
        return 0.5
    try:
        import imagehash
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return max(0.0, 1.0 - (h1 - h2) / 64.0)
    except Exception:
        return 0.5


def colour_similarity(c1, c2):
    if not c1 or not c2:
        return 0.5
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    dist = np.linalg.norm(c1 - c2)
    return max(0.0, 1.0 - dist / 441.0)


def text_similarity(t1, t2):
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    t1_set = set(t1.lower().split())
    t2_set = set(t2.lower().split())
    if not t1_set and not t2_set:
        return 1.0
    intersection = len(t1_set & t2_set)
    union = len(t1_set | t2_set)
    return intersection / union if union > 0 else 0.0


def class_similarity(c1, c2):
    return 1.0 if c1 == c2 else 0.0


def node_similarity_weighted(G1, G2, n1, n2, weights,
                              depth=0, max_depth=2, memo=None):
    """Node similarity with configurable weights."""
    if memo is None:
        memo = {}
    key = (n1, n2, depth)
    if key in memo:
        return memo[key]

    d1 = G1.nodes[n1]
    d2 = G2.nodes[n2]

    s_visual   = phash_similarity(d1.get("phash"),       d2.get("phash"))
    s_colour   = colour_similarity(d1.get("mean_colour"), d2.get("mean_colour"))
    s_text     = text_similarity(d1.get("ocr_text", ""),  d2.get("ocr_text", ""))
    s_class    = class_similarity(d1.get("class_name"),   d2.get("class_name"))

    s_structural = 0.0
    if depth < max_depth:
        nbrs1 = list(G1.neighbors(n1))
        nbrs2 = list(G2.neighbors(n2))
        if nbrs1 and nbrs2:
            scores = []
            for nb1 in nbrs1:
                best = max(
                    node_similarity_weighted(G1, G2, nb1, nb2,
                                             weights, depth + 1,
                                             max_depth, memo)
                    for nb2 in nbrs2
                )
                scores.append(best)
            s_structural = float(np.mean(scores)) if scores else 0.0

    sim = (
        weights["visual"]     * s_visual +
        weights["colour"]     * s_colour +
        weights["text"]       * s_text +
        weights["class"]      * s_class +
        weights["structural"] * s_structural
    )

    memo[key] = sim
    return sim


def match_graphs_weighted(G1, G2, weights, threshold):
    """Graph matching with configurable weights and threshold."""
    nodes1 = list(G1.nodes())
    nodes2 = list(G2.nodes())

    if not nodes1 and not nodes2:
        return [], [], [], [], np.array([])
    if not nodes1:
        return [], [], list(nodes2), [], np.array([])
    if not nodes2:
        return [], [], [], list(nodes1), np.array([])

    n1_count = len(nodes1)
    n2_count = len(nodes2)
    sim_matrix = np.zeros((n1_count, n2_count))
    memo = {}

    for i, nd1 in enumerate(nodes1):
        for j, nd2 in enumerate(nodes2):
            sim_matrix[i, j] = node_similarity_weighted(
                G1, G2, nd1, nd2, weights, depth=0, memo=memo
            )

    # Greedy matching
    matched1 = set()
    matched2 = set()
    matches  = []

    flat_indices = np.argsort(sim_matrix.ravel())[::-1]
    for idx in flat_indices:
        i, j = divmod(idx, n2_count)
        if i in matched1 or j in matched2:
            continue
        sim = sim_matrix[i, j]
        if sim >= threshold:
            matches.append((nodes1[i], nodes2[j], sim))
            matched1.add(i)
            matched2.add(j)

    changed_nodes = []
    removed_nodes = []
    for i, nd1 in enumerate(nodes1):
        if i not in matched1:
            # Find best match even below threshold
            if n2_count > 0:
                best_j   = int(np.argmax(sim_matrix[i]))
                best_sim = sim_matrix[i, best_j]
                if best_j not in matched2:
                    changed_nodes.append((nd1, nodes2[best_j], best_sim))
                    matched1.add(i)
                    matched2.add(best_j)
                else:
                    removed_nodes.append(nd1)
            else:
                removed_nodes.append(nd1)

    added_nodes = [nodes2[j] for j in range(n2_count) if j not in matched2]

    return matches, changed_nodes, added_nodes, removed_nodes, sim_matrix


def detect_changes_weighted(G1, G2, weights, threshold):
    """Detect changed boxes using weighted similarity."""
    matches, changed_nodes, added_nodes, removed_nodes, _ = \
        match_graphs_weighted(G1, G2, weights, threshold)

    changed_boxes = []

    for n1_id, n2_id, sim in changed_nodes:
        box = G2.nodes[n2_id]["bbox"]
        changed_boxes.append(box)

    for n1_id in removed_nodes:
        box = G1.nodes[n1_id]["bbox"]
        changed_boxes.append(box)

    for n2_id in added_nodes:
        box = G2.nodes[n2_id]["bbox"]
        changed_boxes.append(box)

    return changed_boxes


# =============================================================================
# MAIN SWEEP
# =============================================================================

def run_sweep(model_path, pairs_dir, manifest_path, output_dir):
    output_dir    = Path(output_dir)
    pairs_dir     = Path(pairs_dir)
    manifest_path = Path(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import cv2
    from ultralytics import YOLO
    from graph_builder import build_graph, run_yolo

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    with open(manifest_path) as f:
        manifest = json.load(f)
    pairs = manifest["pairs"]
    print(f"[INFO] {len(pairs)} test pairs | threshold fixed at {THRESHOLD}")

    # ── Build graph cache once ─────────────────────────────────
    print(f"\n[INFO] Building graphs for all pairs (runs YOLO once)...")
    graph_cache = {}
    skipped = 0

    for i, pair in enumerate(pairs):
        pid       = pair["pair_id"]
        orig_path = pairs_dir / f"{pid}_original.jpg"
        chng_path = pairs_dir / f"{pid}_changed.jpg"
        gt_path   = pairs_dir / f"{pid}_gt.json"

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
        G1   = build_graph(img1, det1, extract_ocr=False)
        G2   = build_graph(img2, det2, extract_ocr=False)

        gt_boxes = [
            c["changed_box"] for c in gt.get("changes", [])
            if c.get("changed_box")
        ]

        graph_cache[pid] = {
            "G1": G1, "G2": G2,
            "gt_boxes": gt_boxes,
            "gt_change_type": gt.get("change_type", "unknown"),
        }

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)} graphs built...")

    print(f"[INFO] Cache ready: {len(graph_cache)} pairs ({skipped} skipped)\n")

    # ── Sweep weight configs ───────────────────────────────────
    summary = []

    for config_name, weights in WEIGHT_CONFIGS:
        # Validate weights sum to ~1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"[WARN] {config_name}: weights sum to {total:.3f} (not 1.0)")

        print(f"\n{'='*60}")
        print(f"  CONFIG: {config_name}")
        print(f"  Weights: visual={weights['visual']}  colour={weights['colour']}  "
              f"text={weights['text']}  class={weights['class']}  "
              f"structural={weights['structural']}")
        print(f"{'='*60}")

        all_results = []
        for pid, cache in graph_cache.items():
            changed_boxes = detect_changes_weighted(
                cache["G1"], cache["G2"], weights, THRESHOLD
            )
            all_results.append({
                "pair_id":        pid,
                "changed_boxes":  changed_boxes,
                "gt_boxes":       cache["gt_boxes"],
                "gt_change_type": cache["gt_change_type"],
            })

        # Save per-config results
        config_dir = output_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        metrics  = {iou: compute_metrics(all_results, iou)
                    for iou in [0.25, 0.50, 0.75]}
        per_type = compute_per_type_metrics(all_results, 0.50)

        print(f"  {'IOU':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
        print(f"  {'-'*42}")
        for iou in [0.25, 0.50, 0.75]:
            m = metrics[iou]
            print(f"  {iou:>6.2f}  {m['precision']:>10.4f}  "
                  f"{m['recall']:>8.4f}  {m['f1']:>8.4f}")

        print(f"\n  Per-type F1 @ IOU=0.50:")
        for ct, m in sorted(per_type.items()):
            bar = "█" * int(m["f1"] * 20)
            print(f"    {ct:<15} {m['f1']:.4f}  {bar}")

        summary.append({
            "config":   config_name,
            "weights":  weights,
            "metrics":  {str(iou): metrics[iou] for iou in metrics},
            "per_type": per_type,
        })

    # ── Final ranked summary ───────────────────────────────────
    ranked = sorted(summary, key=lambda x: x["metrics"]["0.5"]["f1"],
                    reverse=True)

    print(f"\n\n{'='*75}")
    print(f"  WEIGHT SWEEP SUMMARY — threshold={THRESHOLD} — ranked by F1@0.50")
    print(f"{'='*75}")
    print(f"  {'Config':<25}  {'F1@0.25':>8}  {'F1@0.50':>8}  "
          f"{'F1@0.75':>8}  {'P@0.50':>8}  {'R@0.50':>8}")
    print(f"  {'-'*70}")

    for r in ranked:
        m25  = r["metrics"]["0.25"]["f1"]
        m50  = r["metrics"]["0.5"]["f1"]
        m75  = r["metrics"]["0.75"]["f1"]
        p50  = r["metrics"]["0.5"]["precision"]
        rc50 = r["metrics"]["0.5"]["recall"]
        marker = "  ← BEST" if r == ranked[0] else ""
        print(f"  {r['config']:<25}  {m25:>8.4f}  {m50:>8.4f}  "
              f"{m75:>8.4f}  {p50:>8.4f}  {rc50:>8.4f}{marker}")

    print(f"{'='*75}")

    best = ranked[0]
    print(f"\n  Best config : {best['config']}")
    print(f"  Best weights: {best['weights']}")
    print(f"  F1@0.50     : {best['metrics']['0.5']['f1']:.4f}  "
          f"(vs baseline {summary[0]['metrics']['0.5']['f1']:.4f})")

    print(f"\n  Per-type breakdown for best config:")
    for ct, m in sorted(best["per_type"].items()):
        bar = "█" * int(m["f1"] * 20)
        print(f"    {ct:<15} {m['f1']:.4f}  {bar}")

    # Save summary
    with open(output_dir / "weight_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Results saved: {output_dir}/weight_sweep_summary.json")
    print(f"[DONE] Per-config results: {output_dir}/<config_name>/all_results.json")
    print(f"\n[NEXT] Update WEIGHTS in graph_matcher.py with best config")
    print(f"[NEXT] Then proceed to Phase C — position feature + colour fix")

    return best


def main():
    parser = argparse.ArgumentParser(
        description="Sweep similarity weights for graph matcher"
    )
    parser.add_argument("--model",      required=True)
    parser.add_argument("--pairs_dir",  required=True)
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--output_dir", default="./outputs/sweep/weights")
    args = parser.parse_args()

    run_sweep(
        model_path=args.model,
        pairs_dir=args.pairs_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
