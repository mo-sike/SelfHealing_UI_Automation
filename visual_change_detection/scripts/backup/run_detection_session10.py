"""
run_detection.py
================
Run the full detection pipeline on all test pairs and save all_results.json
in the format expected by evaluate.py (step6_evaluate).

This is the missing "step5 runner" — feeds into evaluate.py for detection F1.

Usage:
    python scripts/run_detection.py ^
        --model       ./outputs/yolo_runs/rico_ui_v4/weights/best.pt ^
        --classifier  ./outputs/classifier_v6/classifier.pkl ^
        --pairs_dir   ./outputs/change_dataset_v3/test/pairs ^
        --manifest    ./outputs/change_dataset_v3/test/manifest.json ^
        --output_dir  ./outputs/results_v3 ^
        --threshold   0.85

Then evaluate:
    python scripts/evaluate.py ^
        --results_dir ./outputs/results_v3 ^
        --output_dir  ./outputs/evaluation_v3
"""

import cv2
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))
from graph_builder  import build_graph, run_yolo
from graph_matcher  import (
    match_graphs, phash_similarity, colour_similarity,
    position_similarity, RELOCATE_POSITION_THRESH,
)
from step7_classifier import compute_bbox_area_ratio, compute_centre_distance
from ultralytics import YOLO


def compute_region_pixel_diff(img1, img2, box1, box2):
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


def run_pipeline_detection(model, clf, le, img1_path, img2_path,
                            sim_threshold=0.85):
    """
    Run YOLO → graph → match on one pair.
    Returns list of dicts: [{box_original, box_changed, change_type}, ...]
    """
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        return []

    h, w = img1.shape[:2]

    det1 = run_yolo(model, img1)
    det2 = run_yolo(model, img2)
    G1   = build_graph(img1, det1, extract_ocr=False, extract_clip=True)
    G2   = build_graph(img2, det2, extract_ocr=False, extract_clip=True)

    if G1.number_of_nodes() == 0 and G2.number_of_nodes() == 0:
        return []

    matches, changed_nodes, added_nodes, removed_nodes, _ = \
        match_graphs(G1, G2, similarity_threshold=sim_threshold)

    detections = []
    flagged_g1 = set()

    # Changed nodes (matched, below threshold)
    for n1_id, n2_id, sim in changed_nodes:
        flagged_g1.add(n1_id)
        d1 = G1.nodes[n1_id]
        d2 = G2.nodes[n2_id]
        phash_dist  = 1.0 - phash_similarity(d1.get("phash"), d2.get("phash"))
        colour_dist = 1.0 - colour_similarity(d1.get("mean_colour"), d2.get("mean_colour"))
        area_ratio  = compute_bbox_area_ratio(d1["bbox"], d2["bbox"])
        centre_dist = compute_centre_distance(d1["bbox"], d2["bbox"], w, h)
        class_match = 1.0 if d1["class_name"] == d2["class_name"] else 0.0
        pixel_diff  = compute_region_pixel_diff(img1, img2, d1["bbox"], d2["bbox"])
        features    = [phash_dist, colour_dist, area_ratio, centre_dist,
                       class_match, float(sim), pixel_diff, 1.0]
        change_type = le.inverse_transform(clf.predict([features]))[0]
        detections.append({
            "box_original": list(d1["bbox"]),
            "box_changed":  list(d2["bbox"]),
            "change_type":  change_type,
        })

    # Removed nodes
    for n1_id in removed_nodes:
        flagged_g1.add(n1_id)
        d1 = G1.nodes[n1_id]
        detections.append({
            "box_original": list(d1["bbox"]),
            "box_changed":  None,
            "change_type":  "remove",
        })

    # Added nodes
    for n2_id in added_nodes:
        d2 = G2.nodes[n2_id]
        detections.append({
            "box_original": None,
            "box_changed":  list(d2["bbox"]),
            "change_type":  "add",
        })

    # Phase C3 — post-match relocate detection
    for n1_id, n2_id, sim in matches:
        if sim < sim_threshold:
            continue
        if n1_id in flagged_g1:
            continue
        d1 = G1.nodes[n1_id]
        d2 = G2.nodes[n2_id]
        pos_sim = position_similarity(d1.get("bbox"), d2.get("bbox"))
        if pos_sim >= RELOCATE_POSITION_THRESH:
            continue
        flagged_g1.add(n1_id)
        phash_dist  = 1.0 - phash_similarity(d1.get("phash"), d2.get("phash"))
        colour_dist = 1.0 - colour_similarity(d1.get("mean_colour"), d2.get("mean_colour"))
        area_ratio  = compute_bbox_area_ratio(d1["bbox"], d2["bbox"])
        centre_dist = compute_centre_distance(d1["bbox"], d2["bbox"], w, h)
        class_match = 1.0 if d1["class_name"] == d2["class_name"] else 0.0
        pixel_diff  = compute_region_pixel_diff(img1, img2, d1["bbox"], d2["bbox"])
        features    = [phash_dist, colour_dist, area_ratio, centre_dist,
                       class_match, float(sim), pixel_diff, 1.0]
        change_type = le.inverse_transform(clf.predict([features]))[0]
        detections.append({
            "box_original": list(d1["bbox"]),
            "box_changed":  list(d2["bbox"]),
            "change_type":  change_type,
        })

    return detections


def main():
    parser = argparse.ArgumentParser(description="Generate all_results.json for evaluate.py")
    parser.add_argument("--model",      required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--pairs_dir",  required=True)
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold",  type=float, default=0.85)
    args = parser.parse_args()

    pairs_dir  = Path(args.pairs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading YOLO: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Loading classifier: {args.classifier}")
    with open(args.classifier, "rb") as f:
        cd = pickle.load(f)
    clf = cd["classifier"]
    le  = cd["label_encoder"]

    with open(args.manifest) as f:
        manifest = json.load(f)
    pairs = manifest["pairs"]

    print(f"[INFO] Running detection on {len(pairs)} pairs (threshold={args.threshold}) ...")
    print("-" * 60)

    all_results   = []
    type_counters = defaultdict(lambda: {"gt": 0, "pred": 0})

    for i, pair in enumerate(pairs):
        pid      = pair["pair_id"]
        gt_type  = pair.get("change_type", "unknown")
        orig     = pairs_dir / f"{pid}_original.jpg"
        chng     = pairs_dir / f"{pid}_changed.jpg"
        gt_path  = pairs_dir / f"{pid}_gt.json"

        if not orig.exists() or not chng.exists() or not gt_path.exists():
            continue

        with open(gt_path) as f:
            gt_data = json.load(f)

        # Build GT boxes list — use original_box (G1) as detection target.
        # For add changes (no original), fall back to changed_box (G2).
        gt_boxes = []
        gt_change_types = []
        for chg in gt_data.get("changes", []):
            ct   = chg.get("change_type", gt_type)
            obox = chg.get("original_box") or chg.get("box_original")
            cbox = chg.get("changed_box")  or chg.get("box_changed")
            box  = obox if obox is not None else cbox
            if box is not None:
                gt_boxes.append(list(box))
                gt_change_types.append(ct)

        # Run detection
        detections = run_pipeline_detection(
            model, clf, le, orig, chng, sim_threshold=args.threshold
        )

        # Build predicted changed_boxes: box_original for all except pure add
        changed_boxes = []
        for d in detections:
            b = d["box_original"] if d["box_original"] is not None else d["box_changed"]
            if b is not None:
                changed_boxes.append(b)

        # One result entry per GT change (standard for per-change evaluation)
        for gi, gbox in enumerate(gt_boxes):
            ct = gt_change_types[gi] if gi < len(gt_change_types) else gt_type
            all_results.append({
                "pair_id":        pid,
                "gt_change_type": ct,
                "gt_boxes":       [gbox],
                "changed_boxes":  changed_boxes,
            })
            type_counters[ct]["gt"] += 1
            type_counters[ct]["pred"] += len(changed_boxes)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)} pairs processed ...")

    # Save
    out_file = output_dir / "all_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[INFO] Saved {len(all_results)} result entries → {out_file}")
    print(f"[INFO] Per-type GT counts:")
    for ct, s in sorted(type_counters.items()):
        print(f"  {ct:<15}: {s['gt']} GT  |  avg {s['pred']/max(1,s['gt']):.1f} pred/pair")
    print(f"\nNow run:")
    print(f"  python scripts/evaluate.py --results_dir {args.output_dir} --output_dir {str(output_dir).replace('results', 'evaluation')}")


if __name__ == "__main__":
    main()
