"""
step10_healer.py
================
Phase 2 — Novelty 2 + 3  |  Run on: LOCAL PC

What this does:
    Visual Self-Healing — screenshot-only test locator healing.
    No DOM, no XPath, no internet required.

    Given:
        old screenshot (image1) + new screenshot (image2)
        broken test locators JSON {name, x, y} for each element in image1

    Output:
        Updated coordinates + healing verdict for each locator.

    Four components:
        C1: Locator Mapper    — find which G1 node the broken coordinate belongs to
        C2: Heal Suggester    — locate that node in G2 using Phase 1 match results
        C3: Confidence Est.   — entropy of XGBoost class probabilities → [0, 1]
        C4: Triage            — AUTO-HEAL / REVIEW / CANNOT-HEAL + reason

    Phase 2 Novelty 2:
        First screenshot-only self-healing pipeline for mobile UI test automation.
        Works on any platform using only two screenshots + broken test coordinates.

    Phase 2 Novelty 3:
        Entropy-based uncertainty quantification drives a three-tier verdict that
        prevents wrong auto-heals from silently breaking tests.

Evaluation metrics (batch mode):
    Heal Accuracy    — correct_auto_heals / total_auto_heals  (target: > 85%)
    False Heal Rate  — wrong_auto_heals / total_auto_heals    (target: < 5%)
    Review Reduction — (1 - REVIEW_count / total_changes) %  (target: > 60%)
    Heal Coverage    — (AUTO-HEAL + REVIEW) / total           (target: > 75%)

Usage:
    # Single pair — with locators file
    python scripts/step10_healer.py ^
        --model          ./outputs/yolo_runs/rico_ui_v4/weights/best.pt ^
        --classifier     ./outputs/classifier_v6/classifier.pkl ^
        --image1         ./outputs/change_dataset_v3/test/pairs/12136_v01_original.jpg ^
        --image2         ./outputs/change_dataset_v3/test/pairs/12136_v01_changed.jpg ^
        --locators       ./inputs/locators_sample.json ^
        --output_dir     ./outputs/healer ^
        --threshold      0.85 ^
        --auto_heal_conf 0.80 ^
        --visualise

    # Batch evaluation — 400 test pairs (auto-generates locators from GT)
    python scripts/step10_healer.py ^
        --model          ./outputs/yolo_runs/rico_ui_v4/weights/best.pt ^
        --classifier     ./outputs/classifier_v6/classifier.pkl ^
        --pairs_dir      ./outputs/change_dataset_v3/test/pairs ^
        --manifest       ./outputs/change_dataset_v3/test/manifest.json ^
        --output_dir     ./outputs/healer ^
        --threshold      0.85 ^
        --auto_heal_conf 0.80
"""

import os
import sys
import cv2
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import scipy.stats
from ultralytics import YOLO

# Import from Phase 1 pipeline
sys.path.insert(0, str(Path(__file__).parent))
from graph_builder import build_graph, run_yolo
from graph_matcher import (
    match_graphs,
    phash_similarity,
    colour_similarity,
    position_similarity,
    RELOCATE_POSITION_THRESH,
)
from step7_classifier import compute_bbox_area_ratio, compute_centre_distance


# =============================================================================
# CONSTANTS
# =============================================================================

# Heal accuracy threshold — predicted centre must be within this many pixels
# of the GT changed_box centre to count as a correct heal.
HEAL_ACCURACY_RADIUS_PX = 20

# Pixel-diff sanity check for "unchanged" matches.
# A high-sim matched node pair whose crops differ by more than this fraction
# is likely a wrong greedy match (e.g. remove case matched to same-position
# G2 element).  Demote such pairs to the classifier path instead of
# hard-coding confidence=1.0.
UNCHANGED_PIXEL_DIFF_THRESH = 0.35

VERDICT_COLOURS = {
    "AUTO-HEAL":   (0, 200,  50),    # green
    "REVIEW":      (0, 165, 255),    # orange
    "CANNOT-HEAL": (0,   0, 255),    # red
}

VERDICT_ICONS = {
    "AUTO-HEAL":   "✅",
    "REVIEW":      "⚠️ ",
    "CANNOT-HEAL": "❌",
}


# =============================================================================
# UTILITY — region pixel diff (imported concept from step9_demo)
# =============================================================================

def compute_region_pixel_diff(img1, img2, box1, box2):
    """
    Mean absolute pixel difference between two element crops, normalised [0, 1].
    Catches colour shifts that phash misses.
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


def _build_feature_vector(G1, G2, img1, img2, n1, n2, sim, img_w, img_h,
                           is_matched=1.0):
    """
    Build the 8-feature vector used by the XGBoost classifier.

    Features (same order as step7_classifier.py):
        0: phash_dist       — 1 - phash_similarity
        1: colour_dist      — 1 - colour_similarity
        2: area_ratio       — log-normalised bbox area ratio
        3: centre_dist      — normalised centre distance
        4: class_match      — 1.0 same class, 0.0 different
        5: similarity_score — raw graph matcher similarity
        6: region_pixel_diff — mean absolute pixel diff (normalised)
        7: is_matched       — 1.0 matched node, 0.0 unmatched
    """
    d1 = G1.nodes[n1]
    d2 = G2.nodes[n2] if n2 is not None else None

    phash_dist  = 1.0 - phash_similarity(
        d1.get("phash"), d2.get("phash") if d2 else None
    )
    colour_dist = 1.0 - colour_similarity(
        d1.get("mean_colour"), d2.get("mean_colour") if d2 else None
    )
    area_ratio  = compute_bbox_area_ratio(
        d1["bbox"], d2["bbox"] if d2 else None
    )
    centre_dist = compute_centre_distance(
        d1["bbox"], d2["bbox"] if d2 else None, img_w, img_h
    )
    class_match = (1.0 if d2 and d1["class_name"] == d2["class_name"] else 0.0)
    pixel_diff  = compute_region_pixel_diff(
        img1, img2, d1["bbox"], d2["bbox"] if d2 else None
    )
    return [phash_dist, colour_dist, area_ratio,
            centre_dist, class_match, float(sim), pixel_diff, float(is_matched)]


# =============================================================================
# COMPONENT 1 — Locator Mapper
# =============================================================================

def find_node_at_coordinate(G, x, y):
    """
    Find which graph node's bounding box contains the point (x, y).

    When multiple nodes overlap, the one with the highest YOLO confidence
    is returned (most precise detection wins).

    Returns:
        node_id : int, or None if no node contains the coordinate.
    """
    best_node = None
    best_conf = -1.0

    for node_id, data in G.nodes(data=True):
        x1, y1, x2, y2 = data["bbox"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            conf = data.get("confidence", 0.0)
            if conf > best_conf:
                best_conf = conf
                best_node = node_id

    return best_node


def find_best_iou_node(G, box):
    """
    Find the graph node with highest IOU overlap with a given bounding box.

    Used in batch evaluation to anchor each GT-annotated change to the
    correct YOLO-detected G1 node, instead of relying on the centre-coordinate
    lookup which can land on a small child element inside a large GT bbox.

    Returns:
        (node_id, iou) — node_id is None if G is empty.
    """
    best_node, best_iou = None, 0.0
    bx1, by1, bx2, by2 = [float(v) for v in box]
    b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    for node_id, data in G.nodes(data=True):
        nx1, ny1, nx2, ny2 = [float(v) for v in data["bbox"]]
        ix1 = max(nx1, bx1); iy1 = max(ny1, by1)
        ix2 = min(nx2, bx2); iy2 = min(ny2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        n_area = max(1.0, (nx2 - nx1) * (ny2 - ny1))
        iou = inter / (b_area + n_area - inter + 1e-6)
        if iou > best_iou:
            best_iou = iou
            best_node = node_id
    return best_node, float(best_iou)


# =============================================================================
# COMPONENT 2 — Heal Suggester
# =============================================================================

def suggest_heal(G1, G2, all_matches, changed_nodes, removed_nodes,
                 old_x, old_y):
    """
    Find where the element at (old_x, old_y) in G1 maps to in G2.

    Uses Phase 1 graph match results directly — no re-running the pipeline.

    Priority ordering:
        1. Element was removed (not matched) → CANNOT-HEAL
        2. Element was matched (below threshold) → changed, give new coords
        3. Element was matched (above threshold) → unchanged, locator still valid

    Returns dict with keys:
        status      : "unchanged" | "changed" | "removed" | "not_found"
        new_x       : float or None
        new_y       : float or None
        new_bbox    : [x1,y1,x2,y2] or None
        similarity  : float
        old_class   : str or None
        new_class   : str or None
        n1          : int or None  (G1 node id)
        n2          : int or None  (G2 node id)
    """
    n1 = find_node_at_coordinate(G1, old_x, old_y)

    if n1 is None:
        return {
            "status": "not_found", "new_x": None, "new_y": None,
            "new_bbox": None, "similarity": 0.0,
            "old_class": None, "new_class": None, "n1": None, "n2": None,
        }

    # Build a quick lookup: n1_id → (n2_id, sim) from all_matches
    match_lookup = {m_n1: (m_n2, sim) for m_n1, m_n2, sim in all_matches}

    # Check removed first — element has no counterpart in G2
    if n1 in removed_nodes:
        return {
            "status": "removed", "new_x": None, "new_y": None,
            "new_bbox": None, "similarity": 0.0,
            "old_class": G1.nodes[n1].get("class_name"),
            "new_class": None, "n1": n1, "n2": None,
        }

    # Element was matched — retrieve G2 node
    if n1 not in match_lookup:
        # Should not happen if not in removed_nodes, but guard it
        return {
            "status": "removed", "new_x": None, "new_y": None,
            "new_bbox": None, "similarity": 0.0,
            "old_class": G1.nodes[n1].get("class_name"),
            "new_class": None, "n1": n1, "n2": None,
        }

    n2, sim = match_lookup[n1]
    bbox2 = G2.nodes[n2]["bbox"]
    new_x = (bbox2[0] + bbox2[2]) / 2.0
    new_y = (bbox2[1] + bbox2[3]) / 2.0

    # Determine status: is this node in the below-threshold changed set?
    changed_n1_set = {cn1 for cn1, cn2, s in changed_nodes}
    status = "changed" if n1 in changed_n1_set else "unchanged"

    return {
        "status":    status,
        "new_x":     new_x,
        "new_y":     new_y,
        "new_bbox":  list(bbox2),
        "similarity": float(sim),
        "old_class": G1.nodes[n1].get("class_name"),
        "new_class": G2.nodes[n2].get("class_name"),
        "n1": n1,
        "n2": n2,
    }


# =============================================================================
# COMPONENT 3 — Confidence Estimator
# =============================================================================

def compute_confidence(clf, le, features):
    """
    Predict change type and compute entropy-based confidence.

    Confidence = 1 - normalised_entropy
        → 1.0 : classifier strongly favours one class (high certainty)
        → 0.0 : probability mass spread equally across all classes (maximum uncertainty)

    Normalising by max_entropy gives [0, 1] regardless of number of classes,
    and requires no additional training — uses classifier_v6 directly.

    Args:
        clf      : trained XGBoostClassifier (loaded from .pkl)
        le       : LabelEncoder (loaded from .pkl)
        features : 8-element feature vector

    Returns:
        change_type : str   — predicted change type label
        confidence  : float — [0, 1] certainty score
        proba       : np.ndarray — full probability distribution
    """
    proba     = clf.predict_proba([features])[0]
    entropy   = scipy.stats.entropy(proba)
    n_classes = len(proba)
    max_entr  = scipy.stats.entropy([1.0 / n_classes] * n_classes)

    # Avoid division by zero if all probs equal (degenerate case)
    confidence = 0.0 if max_entr == 0 else float(1.0 - (entropy / max_entr))
    confidence = float(np.clip(confidence, 0.0, 1.0))

    change_type = le.inverse_transform([int(proba.argmax())])[0]
    return change_type, confidence, proba


# =============================================================================
# COMPONENT 4 — Triage
# =============================================================================

def triage(heal_result, change_type, confidence,
           auto_heal_confidence=0.80, review_confidence=0.50):
    """
    Three-tier verdict for a single locator.

    Verdict table:
        AUTO-HEAL    — element unchanged (sim ≥ threshold), OR
                       change in [relocate, resize] AND confidence ≥ 0.80
        REVIEW       — any change AND confidence 0.50–0.79, OR
                       change in [color_change, add] at any confidence ≥ 0.50
        CANNOT-HEAL  — element removed / not detected, OR confidence < 0.50

    Returns:
        verdict : str  — "AUTO-HEAL" | "REVIEW" | "CANNOT-HEAL"
        reason  : str  — human-readable explanation
    """
    status = heal_result["status"]

    # Unchanged — locator still valid as-is (but coordinates may have shifted)
    if status == "unchanged":
        return "AUTO-HEAL", "element unchanged, locator still valid"

    # No element to heal
    if status in ("removed", "not_found"):
        reason = (
            "element removed from UI"
            if status == "removed"
            else "element not detected at old coordinate"
        )
        return "CANNOT-HEAL", reason

    # Low confidence — system is unsure, do not auto-heal
    if confidence < review_confidence:
        return (
            "CANNOT-HEAL",
            f"confidence too low ({confidence:.2f} < {review_confidence:.2f})"
        )

    # Positional changes at high confidence — safe to auto-heal
    if change_type in ("relocate", "resize") and confidence >= auto_heal_confidence:
        return (
            "AUTO-HEAL",
            f"element {change_type}d with high confidence ({confidence:.2f})"
        )

    # Everything else — new coordinates are available but need human sign-off
    reason = (
        f"change type '{change_type}' requires human review"
        if confidence >= auto_heal_confidence
        else f"change type '{change_type}', confidence ({confidence:.2f}) "
             f"below auto-heal threshold ({auto_heal_confidence:.2f})"
    )
    return "REVIEW", reason


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_report(test_name, image1_path, image2_path,
                    locators, heal_results, output_dir,
                    threshold, auto_heal_conf):
    """
    Print formatted table to console and save heal_report.json.

    heal_results is a list of dicts, one per locator, with keys:
        locator_name, old_x, old_y,
        new_x, new_y, new_bbox,
        status, change_type, confidence, proba,
        verdict, reason
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    auto  = sum(1 for r in heal_results if r["verdict"] == "AUTO-HEAL")
    rev   = sum(1 for r in heal_results if r["verdict"] == "REVIEW")
    cant  = sum(1 for r in heal_results if r["verdict"] == "CANNOT-HEAL")
    total = len(heal_results)
    coverage = (auto + rev) / total * 100 if total else 0.0

    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  SELF-HEALING REPORT")
    print(f"  Test:      {test_name}")
    print(f"  Old:       {Path(image1_path).name}  |  New: {Path(image2_path).name}")
    print(f"  Threshold: {threshold}  |  Auto-heal confidence: {auto_heal_conf}")
    print(f"{sep}\n")

    for r in heal_results:
        icon = VERDICT_ICONS.get(r["verdict"], "?")
        print(f"  {icon} {r['locator_name']}")
        print(f"     Old position : ({r['old_x']:.0f}, {r['old_y']:.0f})")

        if r["new_x"] is not None:
            print(f"     New position : ({r['new_x']:.0f}, {r['new_y']:.0f})")
        if r.get("new_bbox"):
            b = r["new_bbox"]
            print(f"     New bbox     : [{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}]")

        if r["change_type"]:
            print(f"     Change type  : {r['change_type']}")
        if r["confidence"] is not None:
            print(f"     Confidence   : {r['confidence']:.2f}")

        print(f"     Verdict      : {r['verdict']} — {r['reason']}")
        print()

    print(f"{sep}")
    print(f"  Summary: {auto} auto-healed  |  {rev} need review  |  {cant} cannot heal")
    print(f"  Heal coverage : {coverage:.1f}%  ({total} locators total)")
    print(f"{sep}\n")

    # Serialise (remove numpy arrays for JSON)
    def _serialisable(r):
        out = dict(r)
        out.pop("proba", None)    # numpy array — not JSON serialisable
        out.pop("features", None)
        return out

    report = {
        "test_name":       test_name,
        "image1":          str(image1_path),
        "image2":          str(image2_path),
        "threshold":       threshold,
        "auto_heal_conf":  auto_heal_conf,
        "summary": {
            "total":    total,
            "auto_heal": auto,
            "review":   rev,
            "cannot_heal": cant,
            "coverage_pct": round(coverage, 2),
        },
        "results": [_serialisable(r) for r in heal_results],
    }

    report_path = output_dir / "heal_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Report saved: {report_path}")
    return report


# =============================================================================
# VISUALISATION
# =============================================================================

def visualise_healer(img1, img2, heal_results, output_path):
    """
    Side-by-side annotated image.

    Left  (img1): circles at old locator positions, colour = verdict
    Right (img2): circles at new positions + arrows for AUTO-HEAL cases
    """
    vis1 = img1.copy()
    vis2 = img2.copy()
    h, w = img1.shape[:2]

    for r in heal_results:
        colour = VERDICT_COLOURS.get(r["verdict"], (200, 200, 200))
        old_pt = (int(r["old_x"]), int(r["old_y"]))

        # ── Left image: old position ───────────────────────────────────────
        cv2.circle(vis1, old_pt, 14, colour, 2)
        cv2.circle(vis1, old_pt, 3, colour, -1)
        cv2.putText(
            vis1, r["locator_name"][:16],
            (old_pt[0] + 16, old_pt[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA
        )

        # ── Right image: new position ──────────────────────────────────────
        if r["new_x"] is not None:
            new_pt = (int(r["new_x"]), int(r["new_y"]))
            cv2.circle(vis2, new_pt, 14, colour, 2)
            cv2.circle(vis2, new_pt, 3, colour, -1)

            if r["verdict"] == "AUTO-HEAL":
                # Draw a small arrow indicating movement direction
                dx = new_pt[0] - old_pt[0]
                dy = new_pt[1] - old_pt[1]
                dist = max(1.0, np.sqrt(dx * dx + dy * dy))
                # Offset start point slightly outside the circle
                start = (
                    int(new_pt[0] - dx / dist * 20),
                    int(new_pt[1] - dy / dist * 20),
                )
                if 0 <= start[0] < w and 0 <= start[1] < h:
                    cv2.arrowedLine(vis2, start, new_pt, colour, 1,
                                    tipLength=0.4, line_type=cv2.LINE_AA)

            verdict_short = {"AUTO-HEAL": "AUTO", "REVIEW": "REV",
                             "CANNOT-HEAL": "NO"}[r["verdict"]]
            cv2.putText(
                vis2, f"{r['locator_name'][:14]} [{verdict_short}]",
                (new_pt[0] + 16, new_pt[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA
            )

    # Headers
    cv2.putText(vis1, "BEFORE (old locators)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis2, "AFTER  (healed locators)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.hstack([vis1, vis2])
    cv2.imwrite(str(output_path), combined)
    print(f"[INFO] Visualisation saved: {output_path}")


# =============================================================================
# HEALING-OPTIMISED GRAPH MATCHING
# -----------------------------------------------------------------------------
# The detection matcher (graph_matcher.py) uses CLIP (weight=0.15) to
# identify colour changes.  For healing this is actively harmful: a
# colour-changed element receives lower CLIP similarity and the greedy
# matcher prefers a different G2 node that looks more visually similar
# (same colour) rather than the true counterpart at the same position.
#
# Healing weights instead emphasise structural layout (phash 0.30) and
# spatial position (position 0.30) so the element is tracked across frames
# regardless of colour, size, or minor appearance shifts.  CLIP is zeroed
# out and colour weight is minimised.
#
# This function is used ONLY by the healer pipeline.  Phase 1 detection
# (graph_matcher.detect_changes) continues to use the original weights.
# =============================================================================

from graph_matcher import (phash_similarity as _phash_sim,
                           colour_similarity as _colour_sim,
                           text_similarity   as _text_sim,
                           class_similarity  as _class_sim,
                           position_similarity as _pos_sim)

HEALING_WEIGHTS = {
    "visual":     0.30,   # phash — structure regardless of colour
    "clip":       0.00,   # DISABLED — hurts matching for colour-changed elements
    "colour":     0.05,   # minimal — colour may legitimately change
    "text":       0.05,
    "class":      0.20,   # same YOLO class = same UI element type
    "structural": 0.10,   # neighbourhood topology
    "position":   0.30,   # where the element is — key for relocation tracking
}


def _healing_node_similarity(G1, G2, n1, n2, depth=0, max_depth=2, memo=None):
    """Node similarity using healing-optimised weights (structure + position)."""
    if memo is None:
        memo = {}
    key = (n1, n2, depth)
    if key in memo:
        return memo[key]

    d1, d2 = G1.nodes[n1], G2.nodes[n2]

    s_visual   = _phash_sim(d1.get("phash"),         d2.get("phash"))
    s_colour   = _colour_sim(d1.get("mean_colour"),   d2.get("mean_colour"))
    s_text     = _text_sim(d1.get("ocr_text", ""),    d2.get("ocr_text", ""))
    s_class    = _class_sim(d1.get("class_name"),     d2.get("class_name"))
    s_position = _pos_sim(d1.get("bbox"),             d2.get("bbox"))

    s_structural = 0.0
    if depth < max_depth:
        nbrs1, nbrs2 = list(G1.neighbors(n1)), list(G2.neighbors(n2))
        if nbrs1 and nbrs2:
            scores = [
                max(
                    _healing_node_similarity(G1, G2, nb1, nb2,
                                             depth + 1, max_depth, memo)
                    for nb2 in nbrs2
                )
                for nb1 in nbrs1
            ]
            s_structural = float(np.mean(scores)) if scores else 0.0

    sim = (
        HEALING_WEIGHTS["visual"]     * s_visual     +
        HEALING_WEIGHTS["colour"]     * s_colour     +
        HEALING_WEIGHTS["text"]       * s_text       +
        HEALING_WEIGHTS["class"]      * s_class      +
        HEALING_WEIGHTS["structural"] * s_structural +
        HEALING_WEIGHTS["position"]   * s_position
        # clip: 0.00 — intentionally omitted
    )
    memo[key] = sim
    return sim


def _healing_match_graphs(G1, G2, similarity_threshold=0.85):
    """
    Greedy graph matching using healing-optimised weights.

    NOTE: The 0.85 default here is INTENTIONALLY independent of
    step7_classifier.SIM_THRESHOLD (currently 0.65).  The healer's goal is
    element tracking (find where this locator moved), not change detection.
    It uses position-heavy weights (0.30) and omits CLIP (0.00), so a higher
    threshold is appropriate — a 0.85 score here means "almost certainly the
    same element at a new position", which is exactly what healing needs.
    Do NOT sync this to SIM_THRESHOLD.

    Returns the same tuple as match_graphs():
        all_matches, changed_nodes, added_nodes, removed_nodes, sim_matrix
    """
    nodes1, nodes2 = list(G1.nodes()), list(G2.nodes())
    if not nodes1 or not nodes2:
        return [], [], list(nodes2), list(nodes1), np.array([])

    n1_cnt, n2_cnt = len(nodes1), len(nodes2)
    sim_matrix = np.zeros((n1_cnt, n2_cnt))
    memo = {}

    for i, nd1 in enumerate(nodes1):
        for j, nd2 in enumerate(nodes2):
            sim_matrix[i, j] = _healing_node_similarity(
                G1, G2, nd1, nd2, depth=0, max_depth=2, memo=memo
            )

    matched1, matched2 = set(), set()
    all_matches = []
    for idx in np.argsort(sim_matrix.ravel())[::-1]:
        i, j = divmod(idx, n2_cnt)
        if i in matched1 or j in matched2:
            continue
        matched1.add(i)
        matched2.add(j)
        all_matches.append((nodes1[i], nodes2[j], float(sim_matrix[i, j])))

    changed_nodes = [(n1, n2, s) for n1, n2, s in all_matches
                     if s < similarity_threshold]
    removed_nodes = [nodes1[i] for i in range(n1_cnt) if i not in matched1]
    added_nodes   = [nodes2[j] for j in range(n2_cnt) if j not in matched2]

    return all_matches, changed_nodes, added_nodes, removed_nodes, sim_matrix


# =============================================================================
# CORE PIPELINE — single pair
# =============================================================================

def run_healer_single(model, clf, le, image1_path, image2_path,
                      locators_data, output_dir,
                      sim_threshold=0.85, auto_heal_conf=0.80,
                      visualise=False):
    """
    Run full self-healing pipeline on one image pair.

    Args:
        model         : loaded YOLO model
        clf           : loaded XGBoost classifier
        le            : loaded LabelEncoder
        image1_path   : path to old screenshot
        image2_path   : path to new screenshot
        locators_data : dict with keys "test_name" and "locators" list [{name, x, y}, ...]
        output_dir    : where to save outputs
        sim_threshold : graph matcher similarity threshold (default 0.85)
        auto_heal_conf: confidence required for AUTO-HEAL verdict (default 0.80)
        visualise     : save annotated side-by-side image

    Returns:
        list of heal result dicts, one per locator
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img1 = cv2.imread(str(image1_path))
    img2 = cv2.imread(str(image2_path))
    if img1 is None or img2 is None:
        print(f"[ERROR] Cannot read images: {image1_path}, {image2_path}")
        return []

    h, w = img1.shape[:2]
    test_name = locators_data.get("test_name", "unknown")
    locators  = locators_data.get("locators", [])

    print(f"\n[INFO] Running self-healer on: {Path(image1_path).name}")
    print(f"[INFO] YOLO detection + graph construction ...")

    # ── Phase 1 pipeline ──────────────────────────────────────────────────────
    det1 = run_yolo(model, img1)
    det2 = run_yolo(model, img2)
    G1   = build_graph(img1, det1, extract_ocr=False, extract_clip=True)
    G2   = build_graph(img2, det2, extract_ocr=False, extract_clip=True)

    print(f"[INFO] G1 nodes: {G1.number_of_nodes()}  G2 nodes: {G2.number_of_nodes()}")
    print(f"[INFO] Graph matching — healing weights (threshold={sim_threshold}) ...")

    all_matches, changed_nodes, added_nodes, removed_nodes, _ = \
        _healing_match_graphs(G1, G2, similarity_threshold=sim_threshold)

    print(f"[INFO] Matched: {len(all_matches)}  Changed: {len(changed_nodes)}  "
          f"Removed: {len(removed_nodes)}  Added: {len(added_nodes)}")
    print(f"[INFO] Healing {len(locators)} locators ...")

    # ── Heal each locator ─────────────────────────────────────────────────────
    heal_results = []

    for loc in locators:
        name  = loc.get("name", "unknown")
        old_x = float(loc["x"])
        old_y = float(loc["y"])

        # C2: find element in G2
        heal = suggest_heal(
            G1, G2, all_matches, changed_nodes, removed_nodes, old_x, old_y
        )

        n1 = heal["n1"]
        n2 = heal["n2"]

        # C3: compute confidence (skip if element not found / removed)
        if heal["status"] == "unchanged":
            # Pixel-diff sanity check: high-sim match with very different pixel
            # content signals a wrong greedy pairing (e.g. a removed element
            # matched to whatever G2 node happens to sit at the same position).
            # Demote such pairs to the classifier path; keep confidence=1.0 only
            # when the crops are genuinely similar.
            if n1 is not None and n2 is not None:
                pxd = compute_region_pixel_diff(
                    img1, img2, G1.nodes[n1]["bbox"], G2.nodes[n2]["bbox"]
                )
                if pxd > UNCHANGED_PIXEL_DIFF_THRESH:
                    # Visually different despite high similarity score — classify
                    features    = _build_feature_vector(
                        G1, G2, img1, img2, n1, n2, heal["similarity"], w, h, 1.0
                    )
                    change_type, confidence, proba = compute_confidence(clf, le, features)
                else:
                    change_type = "no_change"
                    confidence  = 1.0
                    proba       = None
                    features    = None
            else:
                change_type = "no_change"
                confidence  = 1.0
                proba       = None
                features    = None
        elif heal["status"] == "changed" and n1 is not None and n2 is not None:
            features   = _build_feature_vector(
                G1, G2, img1, img2, n1, n2, heal["similarity"], w, h, is_matched=1.0
            )
            change_type, confidence, proba = compute_confidence(clf, le, features)
        else:
            # removed / not_found — no classifier input available
            change_type = "remove" if heal["status"] == "removed" else None
            confidence  = 0.0
            proba       = None
            features    = None

        # C4: triage
        verdict, reason = triage(
            heal, change_type, confidence,
            auto_heal_confidence=auto_heal_conf
        )

        heal_results.append({
            "locator_name": name,
            "old_x":        old_x,
            "old_y":        old_y,
            "new_x":        heal["new_x"],
            "new_y":        heal["new_y"],
            "new_bbox":     heal["new_bbox"],
            "status":       heal["status"],
            "old_class":    heal["old_class"],
            "new_class":    heal["new_class"],
            "similarity":   heal["similarity"],
            "change_type":  change_type,
            "confidence":   confidence if confidence is not None else None,
            "proba":        proba,
            "features":     features,
            "verdict":      verdict,
            "reason":       reason,
        })

    # ── Report ────────────────────────────────────────────────────────────────
    generate_report(
        test_name=test_name,
        image1_path=image1_path,
        image2_path=image2_path,
        locators=locators,
        heal_results=heal_results,
        output_dir=output_dir,
        threshold=sim_threshold,
        auto_heal_conf=auto_heal_conf,
    )

    # ── Visualisation ─────────────────────────────────────────────────────────
    if visualise:
        stem = Path(image1_path).stem.replace("_original", "")
        vis_path = output_dir / f"{stem}_heal_vis.jpg"
        visualise_healer(img1, img2, heal_results, vis_path)

    return heal_results


# =============================================================================
# BATCH EVALUATION — 400 test pairs
# =============================================================================

def run_healer_batch(model, clf, le, pairs_dir, manifest_path, output_dir,
                     sim_threshold=0.85, auto_heal_conf=0.80,
                     max_pairs=None):
    """
    Evaluate the healer on all test pairs.

    For each pair, the GT _gt.json provides:
        original_box  → used as the broken locator coordinate (centre)
        changed_box   → used as ground truth new position
        change_type   → expected verdict type

    Expected verdict by change type:
        relocate, resize  → expect AUTO-HEAL
        color_change, add → expect REVIEW
        remove            → expect CANNOT-HEAL

    Primary metrics:
        Heal Accuracy    = correct_auto_heals / total_auto_heals
        False Heal Rate  = wrong_auto_heals / total_auto_heals
        Review Reduction = (1 - REVIEW_count / total) * 100%
        Heal Coverage    = (AUTO-HEAL + REVIEW) / total * 100%
    """
    pairs_dir  = Path(pairs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    pairs = manifest["pairs"]
    if max_pairs:
        pairs = pairs[:max_pairs]

    print(f"\n[INFO] Batch evaluation — {len(pairs)} pairs")
    print(f"[INFO] Threshold: {sim_threshold}  Auto-heal conf: {auto_heal_conf}")
    print("-" * 60)

    # ── Aggregate counters ────────────────────────────────────────────────────
    total           = 0
    auto_heals      = 0
    correct_autos   = 0   # correct auto-heals (semantics vary by type)
    wrong_autos     = 0   # wrong auto-heals
    reviews         = 0
    cant_heals      = 0
    remove_cant     = 0   # remove cases correctly sent to CANNOT-HEAL
    locators_skipped = 0  # GT changes with no overlapping YOLO detection

    # Per-change-type breakdown
    type_stats = defaultdict(lambda: {
        "total": 0, "auto": 0, "review": 0, "cant": 0, "correct_auto": 0
    })

    all_pair_results = []

    for i, pair in enumerate(pairs):
        pid       = pair["pair_id"]
        orig_path = pairs_dir / f"{pid}_original.jpg"
        chng_path = pairs_dir / f"{pid}_changed.jpg"
        gt_path   = pairs_dir / f"{pid}_gt.json"

        if not orig_path.exists() or not chng_path.exists() or not gt_path.exists():
            continue

        with open(gt_path) as f:
            gt = json.load(f)

        # Load images + build graphs first (needed for IOU-based locator lookup)
        gt_changes = gt.get("changes", [])
        if not gt_changes:
            continue

        img1 = cv2.imread(str(orig_path))
        img2 = cv2.imread(str(chng_path))
        if img1 is None or img2 is None:
            continue

        _h, _w = img1.shape[:2]

        det1 = run_yolo(model, img1)
        det2 = run_yolo(model, img2)
        G1   = build_graph(img1, det1, extract_ocr=False, extract_clip=True)
        G2   = build_graph(img2, det2, extract_ocr=False, extract_clip=True)

        all_matches_i, changed_nodes_i, added_nodes_i, removed_nodes_i, _ = \
            _healing_match_graphs(G1, G2, similarity_threshold=sim_threshold)

        # Build locators using IOU-based G1 node lookup.
        # This anchors each GT-annotated change to the correct YOLO-detected
        # node rather than relying on the original_box centre, which can land
        # on a small child element inside a large GT bbox.
        locators = []
        for idx, chg in enumerate(gt_changes):
            orig_box = chg.get("original_box") or chg.get("box_original")
            if orig_box is None:
                continue
            g1_node, g1_iou = find_best_iou_node(G1, orig_box)
            if g1_node is None or g1_iou < 0.10:
                # No YOLO detection overlaps the GT changed element — skip
                locators_skipped += 1
                continue
            d1 = G1.nodes[g1_node]
            cx = (d1["bbox"][0] + d1["bbox"][2]) / 2.0
            cy = (d1["bbox"][1] + d1["bbox"][3]) / 2.0
            locators.append({
                "name":            f"{chg.get('change_type', 'change')}_{idx}",
                "x":               cx,
                "y":               cy,
                "_gt_changed_box": chg.get("changed_box"),
                "_gt_change_type": chg.get("change_type", "unknown"),
                "_g1_iou":         g1_iou,
            })

        if not locators:
            continue

        pair_row = {"pair_id": pid, "results": []}

        for loc in locators:
            gt_type    = loc["_gt_change_type"]
            gt_new_box = loc.get("_gt_changed_box")

            heal = suggest_heal(
                G1, G2, all_matches_i, changed_nodes_i, removed_nodes_i,
                loc["x"], loc["y"]
            )

            n1 = heal["n1"]
            n2 = heal["n2"]

            if heal["status"] == "unchanged":
                # Pixel-diff sanity check — same logic as run_healer_single
                if n1 is not None and n2 is not None:
                    pxd = compute_region_pixel_diff(
                        img1, img2, G1.nodes[n1]["bbox"], G2.nodes[n2]["bbox"]
                    )
                    if pxd > UNCHANGED_PIXEL_DIFF_THRESH:
                        features    = _build_feature_vector(
                            G1, G2, img1, img2, n1, n2,
                            heal["similarity"], _w, _h, 1.0
                        )
                        change_type, confidence, _ = compute_confidence(clf, le, features)
                    else:
                        change_type = "no_change"
                        confidence  = 1.0
                        features    = None
                else:
                    change_type = "no_change"
                    confidence  = 1.0
                    features    = None
            elif heal["status"] == "changed" and n1 is not None and n2 is not None:
                features    = _build_feature_vector(
                    G1, G2, img1, img2, n1, n2, heal["similarity"], _w, _h, 1.0
                )
                change_type, confidence, _ = compute_confidence(clf, le, features)
            else:
                change_type = "remove" if heal["status"] == "removed" else None
                confidence  = 0.0
                features    = None

            verdict, reason = triage(
                heal, change_type, confidence,
                auto_heal_confidence=auto_heal_conf
            )

            # ── Evaluate correctness of AUTO-HEALs ────────────────────────
            # Correctness semantics by change type:
            #   add          → AUTO-HEAL is always correct: the template element
            #                  is unchanged in G2; the locator remains valid.
            #                  GT changed_box is the ADDED element, not the template.
            #   remove       → CANNOT-HEAL is correct; AUTO-HEAL is always wrong.
            #   relocate / resize / color_change
            #                → AUTO-HEAL is correct iff predicted new coord is
            #                  within HEAL_ACCURACY_RADIUS_PX of GT changed_box centre.
            correct = None
            if verdict == "AUTO-HEAL":
                if gt_type == "add":
                    correct = True    # template element unchanged — locator still valid
                elif gt_type == "remove":
                    correct = False   # element removed — should have been CANNOT-HEAL
                elif heal["new_x"] is not None and gt_new_box is not None:
                    gt_cx = (gt_new_box[0] + gt_new_box[2]) / 2.0
                    gt_cy = (gt_new_box[1] + gt_new_box[3]) / 2.0
                    dist  = np.sqrt(
                        (heal["new_x"] - gt_cx) ** 2 +
                        (heal["new_y"] - gt_cy) ** 2
                    )
                    correct = bool(dist <= HEAL_ACCURACY_RADIUS_PX)
                else:
                    correct = False

            # ── Aggregate ─────────────────────────────────────────────────
            total += 1
            type_stats[gt_type]["total"] += 1

            if verdict == "AUTO-HEAL":
                auto_heals += 1
                type_stats[gt_type]["auto"] += 1
                if correct is True:
                    correct_autos += 1
                    type_stats[gt_type]["correct_auto"] += 1
                elif correct is False:
                    wrong_autos += 1
            elif verdict == "REVIEW":
                reviews += 1
                type_stats[gt_type]["review"] += 1
            else:
                cant_heals += 1
                type_stats[gt_type]["cant"] += 1
                if gt_type == "remove":
                    remove_cant += 1   # correctly abstained

            pair_row["results"].append({
                "locator":      loc["name"],
                "gt_type":      gt_type,
                "verdict":      verdict,
                "change_type":  change_type,
                "confidence":   float(round(confidence, 4)) if confidence is not None else None,
                "correct_heal": correct,
                "g1_iou":       round(loc.get("_g1_iou", 0.0), 3),
            })

        all_pair_results.append(pair_row)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(pairs)} pairs ...")

    # ── Compute metrics ───────────────────────────────────────────────────────
    heal_accuracy   = correct_autos / auto_heals * 100 if auto_heals else 0.0
    false_heal_rate = wrong_autos   / auto_heals * 100 if auto_heals else 0.0
    review_reduction = (1.0 - reviews / total) * 100 if total else 0.0
    coverage        = (auto_heals + reviews) / total * 100 if total else 0.0

    remove_total = type_stats.get("remove", {}).get("total", 0)
    remove_cant_rate = remove_cant / remove_total * 100 if remove_total else 0.0

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  BATCH EVALUATION RESULTS — {total} locators "
          f"(+{locators_skipped} skipped: no YOLO overlap)")
    print(f"  Healing weights: phash=0.30 pos=0.30 class=0.20 "
          f"struct=0.10 colour=0.05 text=0.05 CLIP=0.00")
    print(f"{sep}")
    print(f"  Heal Accuracy    : {heal_accuracy:.1f}%  "
          f"({correct_autos}/{auto_heals} auto-heals correct)")
    print(f"  False Heal Rate  : {false_heal_rate:.1f}%  "
          f"({wrong_autos}/{auto_heals} auto-heals wrong)")
    print(f"  Review Reduction : {review_reduction:.1f}%  "
          f"({total - reviews}/{total} not sent to review)")
    print(f"  Heal Coverage    : {coverage:.1f}%  "
          f"({auto_heals + reviews}/{total} with suggestions)")
    print(f"  Remove Abstain   : {remove_cant_rate:.1f}%  "
          f"({remove_cant}/{remove_total} removes → CANNOT-HEAL)")
    print()
    print(f"  Verdict breakdown:")
    print(f"    AUTO-HEAL  : {auto_heals:4d}  ({auto_heals/total*100:.1f}%)")
    print(f"    REVIEW     : {reviews:4d}  ({reviews/total*100:.1f}%)")
    print(f"    CANNOT-HEAL: {cant_heals:4d}  ({cant_heals/total*100:.1f}%)")
    print()
    print(f"  Per-change-type breakdown  (correct_auto semantics: add=unchanged "
          f"correct, remove=cant-heal correct, others=20px):")
    for ct in ["relocate", "resize", "color_change", "add", "remove"]:
        s = type_stats.get(ct, {})
        if not s or s.get("total", 0) == 0:
            continue
        n  = s["total"]
        a  = s.get("auto", 0)
        r  = s.get("review", 0)
        c  = s.get("cant", 0)
        ca = s.get("correct_auto", 0)
        acc = ca / a * 100 if a else 0.0
        print(f"    {ct:<14}: n={n:3d}  auto={a:3d}({acc:.0f}%)  "
              f"review={r:3d}  cant={c:3d}")
    print(f"{sep}\n")

    # ── Save evaluation JSON ──────────────────────────────────────────────────
    eval_summary = {
        "pairs_evaluated":   len(all_pair_results),
        "total_locators":    total,
        "locators_skipped":  locators_skipped,
        "auto_heals":        auto_heals,
        "correct_auto_heals": correct_autos,
        "wrong_auto_heals":  wrong_autos,
        "reviews":           reviews,
        "cannot_heals":      cant_heals,
        "remove_cant_heals": remove_cant,
        "metrics": {
            "heal_accuracy_pct":    round(heal_accuracy,    2),
            "false_heal_rate_pct":  round(false_heal_rate,  2),
            "review_reduction_pct": round(review_reduction, 2),
            "heal_coverage_pct":    round(coverage,         2),
            "remove_abstain_pct":   round(remove_cant_rate, 2),
        },
        "per_type": {
            ct: dict(s) for ct, s in type_stats.items()
        },
        "pair_results": all_pair_results,
    }

    eval_path = output_dir / "batch_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[INFO] Evaluation saved: {eval_path}")

    return eval_summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visual Self-Healing — Phase 2 Novelties 2 & 3"
    )

    # ── Model / classifier ────────────────────────────────────────────────────
    parser.add_argument(
        "--model", required=True,
        help="Path to YOLOv8 best.pt"
    )
    parser.add_argument(
        "--classifier", required=True,
        help="Path to classifier.pkl (from step7_classifier.py)"
    )

    # ── Single-pair mode ──────────────────────────────────────────────────────
    parser.add_argument("--image1",   default=None,
                        help="Old screenshot (before update)")
    parser.add_argument("--image2",   default=None,
                        help="New screenshot (after update)")
    parser.add_argument("--locators", default=None,
                        help="JSON file with test locators to heal")

    # ── Batch evaluation mode ─────────────────────────────────────────────────
    parser.add_argument("--pairs_dir", default=None,
                        help="Directory containing *_original.jpg and *_changed.jpg pairs")
    parser.add_argument("--manifest",  default=None,
                        help="manifest.json for batch evaluation")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of pairs (for quick testing)")

    # ── Shared options ────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", default="./outputs/healer",
        help="Directory to save outputs (default: ./outputs/healer)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Graph match similarity threshold (default: 0.85)"
    )
    parser.add_argument(
        "--auto_heal_conf", type=float, default=0.80,
        help="Confidence threshold for AUTO-HEAL verdict (default: 0.80)"
    )
    parser.add_argument(
        "--visualise", action="store_true",
        help="Save annotated side-by-side visualisation (single-pair mode)"
    )

    args = parser.parse_args()

    # ── Load model + classifier ───────────────────────────────────────────────
    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Loading classifier: {args.classifier}")
    with open(args.classifier, "rb") as f:
        classifier_data = pickle.load(f)
    clf = classifier_data["classifier"]
    le  = classifier_data["label_encoder"]

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.image1 and args.image2:
        # Single-pair mode
        if args.locators:
            with open(args.locators) as f:
                locators_data = json.load(f)
        else:
            # Auto-generate a single centre-of-image locator as a fallback
            print("[WARN] No --locators file provided. Using image centre as demo.")
            img_tmp = cv2.imread(str(args.image1))
            _h, _w = img_tmp.shape[:2] if img_tmp is not None else (1920, 1080)
            locators_data = {
                "test_name": Path(args.image1).stem,
                "locators": [{"name": "demo_centre", "x": _w / 2, "y": _h / 2}]
            }

        run_healer_single(
            model=model,
            clf=clf,
            le=le,
            image1_path=args.image1,
            image2_path=args.image2,
            locators_data=locators_data,
            output_dir=args.output_dir,
            sim_threshold=args.threshold,
            auto_heal_conf=args.auto_heal_conf,
            visualise=args.visualise,
        )

    elif args.pairs_dir and args.manifest:
        # Batch evaluation mode
        run_healer_batch(
            model=model,
            clf=clf,
            le=le,
            pairs_dir=args.pairs_dir,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            sim_threshold=args.threshold,
            auto_heal_conf=args.auto_heal_conf,
            max_pairs=args.max_pairs,
        )

    else:
        print("[ERROR] Provide either:")
        print("        --image1 --image2 (single-pair mode)")
        print("        --pairs_dir --manifest (batch evaluation mode)")
        parser.print_help()


if __name__ == "__main__":
    main()
