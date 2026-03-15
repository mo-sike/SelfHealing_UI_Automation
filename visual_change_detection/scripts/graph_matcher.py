"""
step5_graph_matcher.py
======================
Phase 1 - Week 3, Days 18-21  |  Run on: LOCAL PC

What this does:
    Compares two graphs (original vs changed screenshot) using a
    recursive similarity algorithm from the base paper (Moradi et al.).

    Similarity between two nodes is computed from:
        1. Visual similarity   — perceptual hash Hamming distance
        2. Colour similarity   — Euclidean distance in BGR colour space
        3. Text similarity     — character overlap of OCR text
        4. Class similarity    — same class = 1, different = 0
        5. Structural similarity — recursive: similarity of neighbours

    Nodes with no good match, or whose best match has low similarity,
    are flagged as CHANGED.

    Output:
        - List of changed regions (bounding boxes)
        - Similarity score per node pair
        - Heatmap image overlay

Usage:
    # Compare two images
    python scripts/step5_graph_matcher.py \
        --model   ./outputs/yolo_runs/rico_ui_v2/weights/best.pt \
        --image1  ./outputs/change_dataset/test/pairs/10037_v02_original.jpg \
        --image2  ./outputs/change_dataset/test/pairs/10037_v02_changed.jpg \
        --output  ./outputs/results/10037_v02 \
        --visualise

    # Evaluate on full test set
    python scripts/step5_graph_matcher.py \
        --model       ./outputs/yolo_runs/rico_ui_v2/weights/best.pt \
        --pairs_dir   ./outputs/change_dataset/test/pairs \
        --manifest    ./outputs/change_dataset/test/manifest.json \
        --output_dir  ./outputs/results/test
"""

import os
import cv2
import json
import pickle
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict

from graph_builder import build_graph, run_yolo, visualise_graph
from ultralytics import YOLO


# =============================================================================
# SIMILARITY FUNCTIONS
# =============================================================================

def phash_similarity(hash1, hash2):
    """
    Compute similarity from perceptual hash strings.
    Returns float in [0, 1]. 1 = identical, 0 = completely different.
    """
    if hash1 is None or hash2 is None:
        return 0.5  # neutral when hash unavailable
    try:
        import imagehash
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        # phash distance is 0-64; normalise to similarity
        dist = h1 - h2
        return max(0.0, 1.0 - dist / 64.0)
    except Exception:
        return 0.5


def colour_similarity(c1, c2):
    """
    Compute similarity from mean BGR colour vectors.
    Returns float in [0, 1]. 1 = same colour.
    """
    if not c1 or not c2:
        return 0.5
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    dist = np.linalg.norm(c1 - c2)
    # Max possible distance in BGR space = sqrt(3 * 255^2) ≈ 441
    return max(0.0, 1.0 - dist / 441.0)


def text_similarity(t1, t2):
    """
    Compute similarity between OCR text strings.
    Uses character-level Jaccard similarity.
    Returns float in [0, 1].
    """
    if not t1 and not t2:
        return 1.0   # both empty = same
    if not t1 or not t2:
        return 0.0   # one empty, one not
    t1_set = set(t1.lower().split())
    t2_set = set(t2.lower().split())
    if not t1_set and not t2_set:
        return 1.0
    intersection = len(t1_set & t2_set)
    union = len(t1_set | t2_set)
    return intersection / union if union > 0 else 0.0


def class_similarity(c1, c2):
    """1.0 if same class, 0.0 otherwise."""
    return 1.0 if c1 == c2 else 0.0


# =============================================================================
# RECURSIVE GRAPH MATCHING (Moradi et al. Algorithm)
# =============================================================================

# Weights for combining similarity components
# Tuned to match base paper emphasis on visual + structural
WEIGHTS = {
    "visual":     0.40,   # perceptual hash
    "colour":     0.25,   # mean colour
    "text":       0.05,   # OCR text
    "class":      0.20,   # same class label
    "structural": 0.10,   # neighbourhood overlap
}


def node_similarity(G1, G2, n1, n2, depth=0, max_depth=2, memo=None):
    """
    Compute recursive similarity between node n1 in G1 and node n2 in G2.

    depth=0    : includes structural (neighbourhood) similarity
    depth>0    : excludes structural to prevent infinite recursion
    max_depth  : how deep to recurse (2 matches base paper)
    memo       : cache to avoid recomputing same pairs
    """
    if memo is None:
        memo = {}

    key = (n1, n2, depth)
    if key in memo:
        return memo[key]

    d1 = G1.nodes[n1]
    d2 = G2.nodes[n2]

    # ── Component similarities ─────────────────────────────────
    s_visual = phash_similarity(d1.get("phash"), d2.get("phash"))
    s_colour = colour_similarity(d1.get("mean_colour"), d2.get("mean_colour"))
    s_text = text_similarity(d1.get("ocr_text", ""), d2.get("ocr_text", ""))
    s_class = class_similarity(d1.get("class_name"), d2.get("class_name"))

    # ── Structural similarity (recursive) ─────────────────────
    s_structural = 0.0
    if depth < max_depth:
        nbrs1 = list(G1.neighbors(n1))
        nbrs2 = list(G2.neighbors(n2))
        if nbrs1 and nbrs2:
            # For each neighbour of n1, find best matching neighbour in n2
            scores = []
            for nb1 in nbrs1:
                best = max(
                    node_similarity(G1, G2, nb1, nb2,
                                    depth + 1, max_depth, memo)
                    for nb2 in nbrs2
                )
                scores.append(best)
            s_structural = np.mean(scores) if scores else 0.0

    sim = (
        WEIGHTS["visual"] * s_visual +
        WEIGHTS["colour"] * s_colour +
        WEIGHTS["text"] * s_text +
        WEIGHTS["class"] * s_class +
        WEIGHTS["structural"] * s_structural
    )

    memo[key] = sim
    return sim


def match_graphs(G1, G2, similarity_threshold=0.5):
    """
    Match nodes between G1 (original) and G2 (changed).

    Algorithm:
        1. Compute pairwise similarity for all node pairs
        2. Greedy match: assign each G1 node to its best G2 match
        3. Nodes with best match below threshold = changed
        4. G2 nodes with no match = added
        5. G1 nodes with no match = removed

    Returns:
        matches        : list of (n1, n2, similarity) tuples
        changed_nodes  : list of n1 node IDs flagged as changed
        added_nodes    : list of n2 node IDs with no match in G1
        removed_nodes  : list of n1 node IDs with no match in G2
        sim_matrix     : 2D numpy array of pairwise similarities
    """
    nodes1 = list(G1.nodes())
    nodes2 = list(G2.nodes())

    if not nodes1 or not nodes2:
        return [], [], list(nodes2), list(nodes1), np.array([])

    n1, n2 = len(nodes1), len(nodes2)
    sim_matrix = np.zeros((n1, n2))
    memo = {}

    for i, nd1 in enumerate(nodes1):
        for j, nd2 in enumerate(nodes2):
            sim_matrix[i, j] = node_similarity(
                G1, G2, nd1, nd2, depth=0, memo=memo
            )

    # Greedy matching (highest similarity first)
    matched1 = set()
    matched2 = set()
    matches = []

    flat_indices = np.argsort(sim_matrix.ravel())[::-1]
    for idx in flat_indices:
        i, j = divmod(idx, n2)
        if i in matched1 or j in matched2:
            continue
        matched1.add(i)
        matched2.add(j)
        matches.append((nodes1[i], nodes2[j], sim_matrix[i, j]))

    # Classify results
    # changed_nodes: matched pairs whose similarity is below threshold
    changed_nodes = [
        (n1_id, n2_id, sim) for n1_id, n2_id, sim in matches
        if sim < similarity_threshold
    ]
    removed_nodes = [
        nodes1[i] for i in range(n1) if i not in matched1
    ]
    added_nodes = [
        nodes2[j] for j in range(n2) if j not in matched2
    ]

    return matches, changed_nodes, added_nodes, removed_nodes, sim_matrix


# =============================================================================
# CHANGE DETECTION
# =============================================================================

def detect_changes(G1, G2, similarity_threshold=0.5):
    """
    Detect changed regions between two graphs.

    Returns:
        changed_boxes  : list of [x1,y1,x2,y2] bounding boxes of changes
        change_details : list of dicts with full change info
        match_result   : raw output from match_graphs
    """
    matches, changed_nodes, added_nodes, removed_nodes, sim_matrix = \
        match_graphs(G1, G2, similarity_threshold)

    changed_boxes = []
    change_details = []

    # Changed nodes — now tuples of (n1_id, n2_id, sim)
    for n1_id, n2_id, sim in changed_nodes:
        box = G2.nodes[n2_id]["bbox"]
        changed_boxes.append(box)
        change_details.append({
            "type":         "changed",
            "node_g1":      n1_id,
            "node_g2":      n2_id,
            "similarity":   sim,
            "class":        G1.nodes[n1_id]["class_name"],
            "box_original": G1.nodes[n1_id]["bbox"],
            "box_changed":  G2.nodes[n2_id]["bbox"],
        })

    # Removed nodes — box from G1
    for n1_id in removed_nodes:
        box = G1.nodes[n1_id]["bbox"]
        changed_boxes.append(box)
        change_details.append({
            "type":     "removed",
            "node_g1":  n1_id,
            "node_g2":  None,
            "class":    G1.nodes[n1_id]["class_name"],
            "box_original": box,
            "box_changed":  None,
        })

    # Added nodes — box from G2
    for n2_id in added_nodes:
        box = G2.nodes[n2_id]["bbox"]
        changed_boxes.append(box)
        change_details.append({
            "type":     "added",
            "node_g1":  None,
            "node_g2":  n2_id,
            "class":    G2.nodes[n2_id]["class_name"],
            "box_original": None,
            "box_changed":  box,
        })

    changed_boxes = nms_boxes(changed_boxes, iou_threshold=0.4)

    return changed_boxes, change_details, (
        matches, changed_nodes, added_nodes, removed_nodes, sim_matrix
    )


def make_heatmap(img_h, img_w, boxes):
    """Create binary heatmap from changed bounding boxes."""
    heatmap = np.zeros((img_h, img_w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        heatmap[y1:y2, x1:x2] = 255
    return heatmap


def nms_boxes(boxes, iou_threshold=0.4):
    """
    Merge overlapping changed boxes using Non-Maximum Suppression.
    Reduces duplicate detections of the same changed region.
    """
    if not boxes:
        return boxes

    boxes_arr = np.array([[b[0], b[1], b[2], b[3]] for b in boxes],
                         dtype=np.float32)

    areas = ((boxes_arr[:, 2] - boxes_arr[:, 0]) *
             (boxes_arr[:, 3] - boxes_arr[:, 1]))

    order = areas.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(boxes_arr[i, 0], boxes_arr[order[1:], 0])
        yy1 = np.maximum(boxes_arr[i, 1], boxes_arr[order[1:], 1])
        xx2 = np.minimum(boxes_arr[i, 2], boxes_arr[order[1:], 2])
        yy2 = np.minimum(boxes_arr[i, 3], boxes_arr[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        overlap = (w * h) / (areas[order[1:]] + 1e-6)
        order = order[np.where(overlap <= iou_threshold)[0] + 1]

    return [boxes[i] for i in keep]


# =============================================================================
# VISUALISATION
# =============================================================================

def visualise_changes(img1, img2, G1, G2, changed_boxes,
                      change_details, output_path):
    """Create side-by-side visualisation of detected changes."""
    vis1 = img1.copy()
    vis2 = img2.copy()

    # Draw all graph nodes lightly
    for _, data in G1.nodes(data=True):
        x1, y1, x2, y2 = data["bbox"]
        cv2.rectangle(vis1, (x1, y1), (x2, y2), (100, 100, 100), 1)
    for _, data in G2.nodes(data=True):
        x1, y1, x2, y2 = data["bbox"]
        cv2.rectangle(vis2, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Draw changes
    for detail in change_details:
        if detail["box_original"]:
            x1, y1, x2, y2 = [int(v) for v in detail["box_original"]]
            cv2.rectangle(vis1, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis1, detail["type"], (x1, max(y1-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if detail["box_changed"]:
            x1, y1, x2, y2 = [int(v) for v in detail["box_changed"]]
            cv2.rectangle(vis2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis2, detail["type"], (x1, max(y1-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    label1 = f"ORIGINAL  nodes:{G1.number_of_nodes()}"
    label2 = f"CHANGED   changes:{len(changed_boxes)}"
    cv2.putText(vis1, label1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis2, label2, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    combined = np.hstack([vis1, vis2])
    cv2.imwrite(str(output_path), combined)
    print(f"[INFO] Visualisation saved: {output_path}")


# =============================================================================
# SINGLE PAIR
# =============================================================================

def process_pair(model, image1_path, image2_path, output_dir,
                 similarity_threshold=0.5, visualise=False,
                 extract_ocr=False):
    """
    Full pipeline for one image pair:
        YOLO → Graph → Match → Detect changes → Save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img1 = cv2.imread(str(image1_path))
    img2 = cv2.imread(str(image2_path))
    if img1 is None or img2 is None:
        print(f"[ERROR] Cannot read images")
        return None

    h, w = img1.shape[:2]

    # Detect + build graphs
    det1 = run_yolo(model, img1)
    det2 = run_yolo(model, img2)
    G1 = build_graph(img1, det1, extract_ocr=extract_ocr)
    G2 = build_graph(img2, det2, extract_ocr=extract_ocr)

    # Match + detect changes
    changed_boxes, change_details, match_result = detect_changes(
        G1, G2, similarity_threshold
    )

    # Build heatmap
    heatmap = make_heatmap(h, w, changed_boxes)

    # Save outputs
    stem = Path(image1_path).stem.replace("_original", "")
    result = {
        "pair_id":       stem,
        "n_changes":     len(changed_boxes),
        "changed_boxes": changed_boxes,
        "change_details": change_details,
        "g1_nodes":      G1.number_of_nodes(),
        "g2_nodes":      G2.number_of_nodes(),
    }
    with open(output_dir / f"{stem}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    cv2.imwrite(str(output_dir / f"{stem}_heatmap.png"), heatmap)

    if visualise:
        visualise_changes(
            img1, img2, G1, G2, changed_boxes, change_details,
            output_dir / f"{stem}_vis.jpg"
        )

    return result


# =============================================================================
# BATCH EVALUATION
# =============================================================================

def evaluate_batch(model_path, pairs_dir, manifest_path, output_dir,
                   similarity_threshold=0.5, max_pairs=None):
    """
    Run change detection on all test pairs and collect results
    for evaluation in step6_evaluate.py.
    """
    pairs_dir = Path(pairs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    pairs = manifest["pairs"]
    if max_pairs:
        pairs = pairs[:max_pairs]

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] Evaluating {len(pairs)} pairs ...")
    all_results = []

    for i, pair in enumerate(pairs):
        pid = pair["pair_id"]
        orig = pairs_dir / f"{pid}_original.jpg"
        changed = pairs_dir / f"{pid}_changed.jpg"
        gt_path = pairs_dir / f"{pid}_gt.json"

        if not orig.exists() or not changed.exists():
            continue

        result = process_pair(
            model=model,
            image1_path=orig,
            image2_path=changed,
            output_dir=output_dir,
            similarity_threshold=similarity_threshold,
            visualise=False,
            extract_ocr=False   # faster for batch
        )

        if result is None:
            continue

        # Load ground truth
        with open(gt_path) as f:
            gt = json.load(f)

        result["gt_change_type"] = gt["change_type"]
        result["gt_boxes"] = [
            c["changed_box"] for c in gt["changes"]
            if c.get("changed_box")
        ]
        all_results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(pairs)} pairs ...")

    # Save all results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[DONE] Results saved to: {output_dir}/all_results.json")
    print(f"[DONE] Total pairs processed: {len(all_results)}")
    print("[INFO] Next: run step6_evaluate.py to compute metrics")
    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Graph matching based change detection"
    )
    parser.add_argument("--model",       required=True,
                        help="Path to best.pt")
    parser.add_argument("--image1",      default=None,
                        help="Original image path (single mode)")
    parser.add_argument("--image2",      default=None,
                        help="Changed image path (single mode)")
    parser.add_argument("--output",      default=None,
                        help="Output directory (single mode)")
    parser.add_argument("--pairs_dir",   default=None,
                        help="Pairs directory (batch mode)")
    parser.add_argument("--manifest",    default=None,
                        help="manifest.json path (batch mode)")
    parser.add_argument("--output_dir",  default=None,
                        help="Output directory (batch mode)")
    parser.add_argument("--threshold",   type=float, default=0.6,
                        help="Similarity threshold (default: 0.5)")
    parser.add_argument("--visualise",   action="store_true")
    parser.add_argument("--max_pairs",   type=int, default=None,
                        help="Limit pairs for quick testing")
    args = parser.parse_args()

    if args.image1 and args.image2:
        model = YOLO(args.model)
        output = args.output or "./outputs/results/single"
        process_pair(
            model=model,
            image1_path=args.image1,
            image2_path=args.image2,
            output_dir=output,
            similarity_threshold=args.threshold,
            visualise=args.visualise
        )

    elif args.pairs_dir and args.manifest:
        output_dir = args.output_dir or "./outputs/results/test"
        evaluate_batch(
            model_path=args.model,
            pairs_dir=args.pairs_dir,
            manifest_path=args.manifest,
            output_dir=output_dir,
            similarity_threshold=args.threshold,
            max_pairs=args.max_pairs
        )
    else:
        print("[ERROR] Provide --image1/--image2 or --pairs_dir/--manifest")
        parser.print_help()


if __name__ == "__main__":
    main()
