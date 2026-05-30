"""
step12_annotate_ui_diff.py
==========================
Phase A — Ground-Truth Annotation via UIAutomator Diff

Generates ground-truth change annotations by diffing the UIAutomator XML
layout dumps captured by step11_collect_real_pairs.py.

Algorithm (three-pass matching):
    Pass 1  — Match by resource-id  (same widget, most reliable)
    Pass 2  — Match by (class, text) (same text content in same widget type)
    Pass 3  — Match by IOU ≥ 0.50   (same visual region, positional fallback)

    Unmatched in dump1 → "remove"
    Unmatched in dump2 → "add"

Change classification for matched pairs:
    position shift > MIN_MOVE_PX        → "relocate"
    size change    > MIN_SIZE_CHANGE    → "resize"
    both                                → dominant dimension decides
    pixel diff     > COLOR_CHANGE_THRESH AND position/size stable → "color_change"
    None of the above                   → no change (pair skipped)

Boxes for GT:
    By default the UIAutomator bounds are used directly.
    With --yolo_model: the closest YOLO detection (IOU-based) replaces each box,
    giving GT boxes that are more compatible with the YOLO pipeline's predictions.

Output (same format as generate_changes.py):
    {pair_id}_gt.json  — {"pair_id", "change_type", "n_changes", "changes", "image_size", "source"}

Usage:
    # Annotate all pairs in a directory (requires v*_dump.xml files)
    python scripts/step12_annotate_ui_diff.py --pairs_dir ./real_data/AntennaPod/pairs --output_dir ./real_data/AntennaPod/pairs

    # With YOLO box refinement
    python scripts/step12_annotate_ui_diff.py --pairs_dir ./real_data/AntennaPod/pairs --output_dir ./real_data/AntennaPod/pairs --yolo_model ./outputs/yolo_runs/rico_ui_v4/weights/best.pt

    # Annotate all apps
    python scripts/step12_annotate_ui_diff.py --root_dir ./real_data --yolo_model ./outputs/yolo_runs/rico_ui_v4/weights/best.pt

    # Dry run — print what would be annotated without writing files
    python scripts/step12_annotate_ui_diff.py --pairs_dir ./real_data/AntennaPod/pairs --dry_run
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter


# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum pixel shift (in either axis) to classify as a positional relocate
MIN_MOVE_PX = 15

# Minimum fractional size change to classify as resize (0.10 = 10%)
MIN_SIZE_CHANGE = 0.10

# Mean absolute pixel diff threshold for colour change detection
COLOR_CHANGE_THRESH = 0.08

# Minimum IOU to count as a position-based match (Pass 3)
IOU_MATCH_THRESH = 0.50

# Element area filter — skip elements outside [MIN_AREA_FRAC, MAX_AREA_FRAC]
# of the total screen area. Avoids root containers and 1-pixel artefacts.
MIN_AREA_FRAC = 0.0008   # 0.08% of screen — roughly 8×8px on 1080×1920
MAX_AREA_FRAC = 0.90     # 90% of screen — skip full-screen containers

# Minimum YOLO box IOU to accept YOLO refinement of GT box
YOLO_REFINE_MIN_IOU = 0.40


# =============================================================================
# ELEMENT PARSING
# =============================================================================

def parse_ui_dump(xml_path, img_w, img_h):
    """
    Parse a UIAutomator XML dump and return a list of element dicts.

    Each dict has:
        resource_id   : str  — e.g. "com.example:id/submit"
        class_name    : str  — e.g. "android.widget.Button"
        text          : str  — visible text content
        content_desc  : str  — accessibility description
        bounds        : [x1, y1, x2, y2]  — absolute pixel coords
        checkable     : bool
        checked       : bool
        enabled       : bool

    Filters out elements that are too small or too large (containers).
    """
    if not Path(xml_path).exists():
        return []
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[WARN] XML parse error in {xml_path}: {e}")
        return []

    screen_area = img_w * img_h

    elements = []
    _parse_node(root, elements, screen_area)
    return elements


def _parse_node(node, elements, screen_area):
    """Recursively collect element data from UIAutomator XML nodes."""
    bounds_str = node.attrib.get("bounds", "")
    bounds = _parse_bounds(bounds_str)
    if bounds:
        x1, y1, x2, y2 = bounds
        w = x2 - x1
        h = y2 - y1
        area = w * h
        area_frac = area / max(screen_area, 1)

        if (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC and w > 4 and h > 4):
            elements.append({
                "resource_id":  node.attrib.get("resource-id", ""),
                "class_name":   node.attrib.get("class", ""),
                "text":         node.attrib.get("text", "").strip(),
                "content_desc": node.attrib.get("content-desc", "").strip(),
                "bounds":       [x1, y1, x2, y2],
                "checkable":    node.attrib.get("checkable", "false") == "true",
                "checked":      node.attrib.get("checked",   "false") == "true",
                "enabled":      node.attrib.get("enabled",   "true")  == "true",
                "clickable":    node.attrib.get("clickable", "false") == "true",
            })

    for child in node:
        _parse_node(child, elements, screen_area)


def _parse_bounds(bounds_str):
    """Parse '[x1,y1][x2,y2]' → (x1,y1,x2,y2) or None."""
    try:
        s = bounds_str.replace("][", ",").strip("[]")
        parts = s.split(",")
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except Exception:
        return None


# =============================================================================
# ELEMENT MATCHING — three passes
# =============================================================================

def _iou(b1, b2):
    """IOU of two [x1,y1,x2,y2] boxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(1, (b1[2]-b1[0]) * (b1[3]-b1[1]))
    a2 = max(1, (b2[2]-b2[0]) * (b2[3]-b2[1]))
    return inter / (a1 + a2 - inter + 1e-6)


def match_elements(elems1, elems2):
    """
    Match elements across two UIAutomator dumps.

    Returns:
        matched   : list of (elem1, elem2) tuples
        removed   : list of elem1 dicts not matched to anything in dump2
        added     : list of elem2 dicts not matched to anything in dump1
    """
    unmatched1 = list(range(len(elems1)))
    unmatched2 = list(range(len(elems2)))
    matched_pairs = []

    def consume(i1, i2):
        unmatched1.remove(i1)
        unmatched2.remove(i2)
        matched_pairs.append((elems1[i1], elems2[i2]))

    # ── Pass 1: resource-id match ─────────────────────────────────────────────
    rid1 = {}
    for i in unmatched1[:]:
        rid = elems1[i]["resource_id"]
        if rid:
            rid1.setdefault(rid, []).append(i)

    for j in unmatched2[:]:
        rid = elems2[j]["resource_id"]
        if rid and rid in rid1 and rid1[rid]:
            candidates = rid1[rid]
            # If multiple candidates, pick by IOU
            best_i = max(candidates, key=lambda i: _iou(elems1[i]["bounds"], elems2[j]["bounds"]))
            consume(best_i, j)
            rid1[rid].remove(best_i)

    # ── Pass 2: (class_name, text) match ─────────────────────────────────────
    ct1 = {}
    for i in unmatched1[:]:
        e = elems1[i]
        key = (e["class_name"], e["text"])
        if key[1]:   # only meaningful if text is non-empty
            ct1.setdefault(key, []).append(i)

    for j in unmatched2[:]:
        e = elems2[j]
        key = (e["class_name"], e["text"])
        if key[1] and key in ct1 and ct1[key]:
            candidates = ct1[key]
            best_i = max(candidates, key=lambda i: _iou(elems1[i]["bounds"], elems2[j]["bounds"]))
            if _iou(elems1[best_i]["bounds"], e["bounds"]) > 0.10:   # any spatial proximity
                consume(best_i, j)
                ct1[key].remove(best_i)

    # ── Pass 3: IOU-based positional match (IOU ≥ 0.50) ──────────────────────
    for j in unmatched2[:]:
        if j not in unmatched2:
            continue
        best_i, best_iou = None, IOU_MATCH_THRESH
        for i in unmatched1:
            v = _iou(elems1[i]["bounds"], elems2[j]["bounds"])
            if v > best_iou:
                best_iou, best_i = v, i
        if best_i is not None:
            consume(best_i, j)

    removed = [elems1[i] for i in unmatched1]
    added   = [elems2[j] for j in unmatched2]
    return matched_pairs, removed, added


# =============================================================================
# CHANGE CLASSIFICATION
# =============================================================================

def _crop_diff(img1, img2, box1, box2):
    """Mean absolute pixel difference (normalised 0–1) between two element crops."""
    try:
        x1,y1,x2,y2 = [int(v) for v in box1]
        c1 = img1[y1:y2, x1:x2]
        x1,y1,x2,y2 = [int(v) for v in box2]
        c2 = img2[y1:y2, x1:x2]
        if c1.size == 0 or c2.size == 0:
            return 0.0
        h = min(c1.shape[0], c2.shape[0])
        w = min(c1.shape[1], c2.shape[1])
        if h < 2 or w < 2:
            return 0.0
        c1 = cv2.resize(c1, (w, h)).astype(float)
        c2 = cv2.resize(c2, (w, h)).astype(float)
        return float(np.mean(np.abs(c1 - c2)) / 255.0)
    except Exception:
        return 0.0


def classify_change(e1, e2, img1=None, img2=None):
    """
    Classify the type of change between two matched elements.

    Returns: change_type str or None (no significant change).
    """
    b1 = e1["bounds"]
    b2 = e2["bounds"]

    cx1 = (b1[0] + b1[2]) / 2.0;  cy1 = (b1[1] + b1[3]) / 2.0
    cx2 = (b2[0] + b2[2]) / 2.0;  cy2 = (b2[1] + b2[3]) / 2.0
    w1 = b1[2] - b1[0];  h1 = b1[3] - b1[1]
    w2 = b2[2] - b2[0];  h2 = b2[3] - b2[1]

    dx = abs(cx2 - cx1)
    dy = abs(cy2 - cy1)
    moved = max(dx, dy) > MIN_MOVE_PX

    dw = abs(w2 - w1) / max(w1, 1)
    dh = abs(h2 - h1) / max(h1, 1)
    resized = max(dw, dh) > MIN_SIZE_CHANGE

    if moved and resized:
        # Dominant change wins
        return "relocate" if max(dx, dy) > max(dw, dh) * min(w1, h1) else "resize"

    if moved:
        return "relocate"

    if resized:
        return "resize"

    # Check pixel-level colour change
    if img1 is not None and img2 is not None:
        pdiff = _crop_diff(img1, img2, b1, b2)
        if pdiff > COLOR_CHANGE_THRESH:
            return "color_change"

    return None   # no detectable change


# =============================================================================
# YOLO BOX REFINEMENT  (optional)
# =============================================================================

def _load_yolo(model_path):
    """Lazily load YOLO model. Returns model or None."""
    try:
        from ultralytics import YOLO
        return YOLO(str(model_path))
    except Exception as e:
        print(f"[WARN] Cannot load YOLO model: {e}")
        return None


def _run_yolo(model, img):
    """Run YOLO on an image, return list of [x1,y1,x2,y2] boxes."""
    results = model(img, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1, y1, x2, y2])
    return boxes


def refine_box_with_yolo(gt_box, yolo_boxes):
    """
    Replace gt_box with the YOLO detection that overlaps it most,
    provided that IOU ≥ YOLO_REFINE_MIN_IOU.
    Returns refined box or original gt_box.
    """
    best_box, best_iou = gt_box, 0.0
    for yb in yolo_boxes:
        v = _iou(gt_box, yb)
        if v > best_iou:
            best_iou, best_box = v, yb
    if best_iou >= YOLO_REFINE_MIN_IOU:
        return [int(v) for v in best_box]
    return gt_box


# =============================================================================
# PAIR ANNOTATOR
# =============================================================================

def annotate_pair(pair_id, img1_path, img2_path, dump1_path, dump2_path,
                  output_dir, yolo_model=None, dry_run=False):
    """
    Generate GT JSON for one before/after pair.

    Args:
        pair_id    : unique identifier, e.g. "antennapod_main"
        img1_path  : path to original (v1) screenshot
        img2_path  : path to changed (v2) screenshot
        dump1_path : UIAutomator XML for v1 (may be None)
        dump2_path : UIAutomator XML for v2 (may be None)
        output_dir : where to save {pair_id}_gt.json
        yolo_model : optional loaded YOLO model for box refinement
        dry_run    : if True, print results without writing files

    Returns gt dict or None if no changes found.
    """
    # Check whether dumps exist
    has_dumps = (dump1_path and Path(dump1_path).exists() and
                 dump2_path and Path(dump2_path).exists())

    if not has_dumps:
        print(f"[WARN] Missing UI dumps for {pair_id} — cannot annotate.")
        return None

    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        print(f"[WARN] Cannot read images for {pair_id}")
        return None

    h, w = img1.shape[:2]

    # ── Parse dumps ───────────────────────────────────────────────────────────
    elems1 = parse_ui_dump(dump1_path, w, h)
    elems2 = parse_ui_dump(dump2_path, w, h)

    if not elems1 and not elems2:
        print(f"[WARN] Both dumps empty for {pair_id}")
        return None

    # ── Match elements ────────────────────────────────────────────────────────
    matched, removed, added = match_elements(elems1, elems2)

    # ── YOLO boxes (optional) ─────────────────────────────────────────────────
    yolo1_boxes, yolo2_boxes = [], []
    if yolo_model:
        yolo1_boxes = _run_yolo(yolo_model, img1)
        yolo2_boxes = _run_yolo(yolo_model, img2)

    # ── Build change records ──────────────────────────────────────────────────
    change_records = []

    for e1, e2 in matched:
        ct = classify_change(e1, e2, img1, img2)
        if ct is None:
            continue

        orig_box = e1["bounds"]
        chng_box = e2["bounds"]

        # Optionally refine to nearest YOLO detection
        if yolo_model:
            orig_box = refine_box_with_yolo(orig_box, yolo1_boxes)
            chng_box = refine_box_with_yolo(chng_box, yolo2_boxes)

        change_records.append({
            "change_type":  ct,
            "original_box": orig_box,
            "changed_box":  chng_box,
        })

    for e in removed:
        orig_box = e["bounds"]
        if yolo_model:
            orig_box = refine_box_with_yolo(orig_box, yolo1_boxes)
        change_records.append({
            "change_type":  "remove",
            "original_box": orig_box,
            "changed_box":  None,
        })

    for e in added:
        chng_box = e["bounds"]
        if yolo_model:
            chng_box = refine_box_with_yolo(chng_box, yolo2_boxes)
        change_records.append({
            "change_type":  "add",
            "original_box": None,
            "changed_box":  chng_box,
        })

    if not change_records:
        print(f"[INFO] No changes detected for {pair_id} — pair skipped.")
        return None

    # Dominant change type for the pair
    type_counts  = Counter(r["change_type"] for r in change_records)
    primary_type = type_counts.most_common(1)[0][0]

    gt = {
        "pair_id":      pair_id,
        "change_type":  primary_type,
        "n_changes":    len(change_records),
        "changes":      change_records,
        "image_size":   [w, h],
        "source":       "real",
        "type_counts":  dict(type_counts),
    }

    if dry_run:
        print(f"  [DRY] {pair_id}: {len(change_records)} changes  "
              f"({', '.join(f'{k}={v}' for k,v in type_counts.items())})")
        return gt

    # ── Write GT file ─────────────────────────────────────────────────────────
    out_path = Path(output_dir) / f"{pair_id}_gt.json"
    with open(out_path, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"  [OK] {pair_id}: {len(change_records)} changes → {out_path.name}  "
          f"({', '.join(f'{k}={v}' for k,v in type_counts.items())})")
    return gt


# =============================================================================
# BATCH ANNOTATOR
# =============================================================================

def annotate_pairs_dir(pairs_dir, output_dir=None, yolo_model=None, dry_run=False):
    """
    Annotate all screen pairs in a directory.

    Expects files matching:
        {pair_id}_original.jpg   ← before screenshot
        {pair_id}_changed.jpg    ← after screenshot
        {pair_id}_v1_dump.xml    ← UIAutomator dump for before
        {pair_id}_v2_dump.xml    ← UIAutomator dump for after

    Skips pairs that already have a _gt.json (unless --overwrite is set).
    """
    pairs_dir  = Path(pairs_dir)
    output_dir = Path(output_dir) if output_dir else pairs_dir

    # Discover pairs by looking for *_original.jpg files
    orig_files = sorted(pairs_dir.glob("*_original.jpg"))
    if not orig_files:
        print(f"[WARN] No *_original.jpg found in {pairs_dir}")
        return []

    results = []
    skip = 0
    total = len(orig_files)

    print(f"\n[ANNOTATE] {total} pairs found in {pairs_dir}")
    if yolo_model:
        print(f"[ANNOTATE] YOLO box refinement enabled")

    for orig_path in orig_files:
        # Derive pair_id from filename by stripping _original.jpg
        pair_id   = orig_path.stem.replace("_original", "")
        chng_path = pairs_dir / f"{pair_id}_changed.jpg"
        dump1     = pairs_dir / f"{pair_id}_v1_dump.xml"
        dump2     = pairs_dir / f"{pair_id}_v2_dump.xml"
        gt_path   = output_dir / f"{pair_id}_gt.json"

        if not chng_path.exists():
            print(f"[WARN] No matching _changed.jpg for {pair_id}")
            continue

        if gt_path.exists() and not dry_run:
            skip += 1
            continue

        gt = annotate_pair(
            pair_id    = pair_id,
            img1_path  = orig_path,
            img2_path  = chng_path,
            dump1_path = str(dump1) if dump1.exists() else None,
            dump2_path = str(dump2) if dump2.exists() else None,
            output_dir = output_dir,
            yolo_model = yolo_model,
            dry_run    = dry_run,
        )
        if gt:
            results.append(gt)

    print(f"\n[ANNOTATE] Done. Annotated: {len(results)}  Skipped (existing): {skip}  "
          f"Total: {total}")
    return results


def annotate_all_apps(root_dir, yolo_model=None, dry_run=False):
    """
    Walk root_dir and annotate every app's pairs/ subdirectory.
    Expected structure: root_dir/{app_name}/pairs/
    """
    root_dir = Path(root_dir)
    all_results = []
    app_dirs = [d for d in root_dir.iterdir()
                if d.is_dir() and (d / "pairs").exists()]

    if not app_dirs:
        print(f"[WARN] No app directories with 'pairs/' found under {root_dir}")
        return []

    for app_dir in sorted(app_dirs):
        pairs_dir = app_dir / "pairs"
        print(f"\n═══ {app_dir.name} ══════════════════════════════════════")
        results = annotate_pairs_dir(pairs_dir, pairs_dir, yolo_model, dry_run)
        all_results.extend(results)

    print(f"\n[DONE] Total annotated across all apps: {len(all_results)}")
    return all_results


# =============================================================================
# STATS PRINTER
# =============================================================================

def print_stats(results):
    if not results:
        return
    from collections import defaultdict
    type_counts = defaultdict(int)
    for gt in results:
        for ct, n in gt.get("type_counts", {}).items():
            type_counts[ct] += n
    total = sum(type_counts.values())
    print("\n  Change type distribution across all annotated pairs:")
    for ct in ["relocate", "resize", "color_change", "add", "remove"]:
        n = type_counts.get(ct, 0)
        bar = "█" * int(n / max(total, 1) * 40)
        print(f"    {ct:<14}: {n:4d}  {bar}")
    print(f"    {'TOTAL':<14}: {total}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase A — UIAutomator diff → GT annotation"
    )

    parser.add_argument(
        "--pairs_dir", default=None,
        help="Directory of *_original.jpg + *_v1_dump.xml pairs (single app)"
    )
    parser.add_argument(
        "--root_dir", default=None,
        help="Root directory containing multiple app subdirs with pairs/ (all apps)"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save GT JSONs (default: same as pairs_dir)"
    )
    parser.add_argument(
        "--yolo_model", default=None,
        help="YOLOv8 model for GT box refinement (optional but recommended)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be annotated without writing any files"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-annotate pairs that already have a _gt.json"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print change type distribution after annotation"
    )

    args = parser.parse_args()

    if not args.pairs_dir and not args.root_dir:
        print("[ERROR] Provide --pairs_dir (single app) or --root_dir (all apps)")
        parser.print_help()
        sys.exit(1)

    # ── Load YOLO model (optional) ────────────────────────────────────────────
    yolo_model = None
    if args.yolo_model:
        print(f"[INFO] Loading YOLO model: {args.yolo_model}")
        yolo_model = _load_yolo(args.yolo_model)
        if yolo_model:
            print("[INFO] YOLO model loaded — GT boxes will be refined.")
        else:
            print("[WARN] YOLO load failed — using raw UIAutomator bounds.")

    # ── Annotate ─────────────────────────────────────────────────────────────
    if args.root_dir:
        results = annotate_all_apps(args.root_dir, yolo_model, args.dry_run)
    else:
        out = args.output_dir or args.pairs_dir
        results = annotate_pairs_dir(args.pairs_dir, out, yolo_model, args.dry_run)

    if args.stats and results:
        print_stats(results)


if __name__ == "__main__":
    main()
