"""
step2_generate_changes.py
=========================
Phase 1 - Week 1, Days 4-7  |  Run on: LOCAL PC

What this does:
    Generates synthetic before/after image pairs for change detection training.
    For each source image it creates N pairs, each with 1-2 applied changes.

    5 change types:
        remove        - delete a control, fill with background colour
        color_change  - shift hue of a control region by 60-120 degrees
        relocate      - move a control to a new non-overlapping position
        resize        - scale a control (shrink 0.4-0.7x or grow 1.3-1.8x)
        add           - duplicate a control to a new position

    Each pair produces:
        {id}_original.jpg   - original image
        {id}_changed.jpg    - modified image
        {id}_gt.json        - ground truth: change type + bounding box
        {id}_heatmap.png    - binary heatmap mask of changed region

Usage:
    # Training pairs
    python scripts/step2_generate_changes.py \
        --images_dir ./outputs/rico_yolo_dataset/images/train \
        --labels_dir ./outputs/rico_yolo_dataset/labels/train \
        --output_dir ./outputs/change_dataset/train \
        --pairs_per_image 3 --max_changes 2 --target_pairs 1600

    # Test pairs
    python scripts/step2_generate_changes.py \
        --images_dir ./outputs/rico_yolo_dataset/images/test \
        --labels_dir ./outputs/rico_yolo_dataset/labels/test \
        --output_dir ./outputs/change_dataset/test \
        --pairs_per_image 3 --max_changes 2 --target_pairs 400

    # Visualise a specific pair (after generation)
    python scripts/step2_generate_changes.py \
        --images_dir ./outputs/rico_yolo_dataset/images/train \
        --labels_dir ./outputs/rico_yolo_dataset/labels/train \
        --output_dir ./outputs/change_dataset/train \
        --target_pairs 0 --visualise PAIR_ID
"""

import os
import cv2
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import Counter


# =============================================================================
# CHANGE FUNCTIONS
# =============================================================================

def get_background_colour(img, x1, y1, x2, y2):
    """Sample background colour from image corners."""
    h, w = img.shape[:2]
    samples = []
    for py in [5, h - 5]:
        for px in [5, w - 5]:
            if 0 <= py < h and 0 <= px < w:
                samples.append(img[py, px].tolist())
    return samples[0] if samples else [240, 240, 240]


def apply_remove(img, box):
    """Remove a control by filling with background colour."""
    x1, y1, x2, y2 = box
    changed = img.copy()
    bg = get_background_colour(img, x1, y1, x2, y2)
    changed[y1:y2, x1:x2] = bg
    return changed, [x1, y1, x2, y2]


def apply_color_change(img, box):
    """Shift hue of a control region."""
    x1, y1, x2, y2 = box
    changed = img.copy()
    region  = changed[y1:y2, x1:x2].copy()
    hsv     = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.int32)
    shift   = random.randint(60, 120) * random.choice([-1, 1])
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    changed[y1:y2, x1:x2] = cv2.cvtColor(
        hsv.astype(np.uint8), cv2.COLOR_HSV2BGR
    )
    return changed, [x1, y1, x2, y2]


def apply_relocate(img, box, all_boxes):
    """Move a control to a new non-overlapping location."""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1

    def overlaps(nx1, ny1, nx2, ny2):
        for bx1, by1, bx2, by2 in all_boxes:
            if nx1 < bx2 and nx2 > bx1 and ny1 < by2 and ny2 > by1:
                return True
        return False

    for _ in range(40):
        nx1 = random.randint(0, max(0, w - bw - 1))
        ny1 = random.randint(0, max(0, h - bh - 1))
        nx2, ny2 = nx1 + bw, ny1 + bh
        if not overlaps(nx1, ny1, nx2, ny2):
            changed = img.copy()
            bg = get_background_colour(img, x1, y1, x2, y2)
            changed[y1:y2, x1:x2] = bg
            changed[ny1:ny2, nx1:nx2] = img[y1:y2, x1:x2]
            return changed, [nx1, ny1, nx2, ny2]

    return None, None  # could not find empty space


def apply_resize(img, box):
    """Scale a control region."""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1

    scale = random.choice([
        random.uniform(0.4, 0.7),   # shrink
        random.uniform(1.3, 1.8)    # grow
    ])
    nw = max(10, int(bw * scale))
    nh = max(10, int(bh * scale))
    nw = min(nw, w - x1)
    nh = min(nh, h - y1)
    nx2, ny2 = x1 + nw, y1 + nh

    region   = img[y1:y2, x1:x2]
    resized  = cv2.resize(region, (nw, nh))
    changed  = img.copy()
    bg = get_background_colour(img, x1, y1, x2, y2)
    changed[y1:y2, x1:x2] = bg
    changed[y1:ny2, x1:nx2] = resized
    return changed, [x1, y1, nx2, ny2]


def apply_add(img, box, all_boxes):
    """Duplicate a control to a new non-overlapping location."""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1

    def overlaps(nx1, ny1, nx2, ny2):
        for bx1, by1, bx2, by2 in all_boxes:
            if nx1 < bx2 and nx2 > bx1 and ny1 < by2 and ny2 > by1:
                return True
        return False

    for _ in range(40):
        nx1 = random.randint(0, max(0, w - bw - 1))
        ny1 = random.randint(0, max(0, h - bh - 1))
        nx2, ny2 = nx1 + bw, ny1 + bh
        if not overlaps(nx1, ny1, nx2, ny2):
            changed = img.copy()
            changed[ny1:ny2, nx1:nx2] = img[y1:y2, x1:x2]
            return changed, [nx1, ny1, nx2, ny2]

    return None, None


CHANGE_FUNCTIONS = {
    "remove":       apply_remove,
    "color_change": apply_color_change,
    "relocate":     apply_relocate,
    "resize":       apply_resize,
    "add":          apply_add,
}


# =============================================================================
# BOX HELPERS
# =============================================================================

def load_boxes(label_path, img_w, img_h):
    """Load YOLO label file and return pixel boxes."""
    boxes = []
    if not Path(label_path).exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                boxes.append([x1, y1, x2, y2])
    return boxes


def make_heatmap(img_h, img_w, changed_boxes):
    """Create binary heatmap for changed regions."""
    heatmap = np.zeros((img_h, img_w), dtype=np.uint8)
    for x1, y1, x2, y2 in changed_boxes:
        heatmap[y1:y2, x1:x2] = 255
    return heatmap


# =============================================================================
# PAIR GENERATION
# =============================================================================

def generate_pair(img_path, label_path, pair_id, output_dir,
                  max_changes=2):
    """Generate one before/after pair. Returns change_type or None on failure."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    boxes = load_boxes(label_path, w, h)
    if len(boxes) < 2:
        return None

    n_changes    = random.randint(1, min(max_changes, len(boxes)))
    change_boxes = random.sample(boxes, n_changes)
    all_boxes    = boxes.copy()

    changed_img    = img.copy()
    change_records = []
    changed_regions = []

    for box in change_boxes:
        change_type = random.choice(list(CHANGE_FUNCTIONS.keys()))
        fn = CHANGE_FUNCTIONS[change_type]

        if change_type in ["relocate", "add"]:
            result_img, changed_box = fn(changed_img, box, all_boxes)
        else:
            result_img, changed_box = fn(changed_img, box)

        if result_img is None:
            continue

        changed_img = result_img
        change_records.append({
            "change_type":    change_type,
            "original_box":   box,
            "changed_box":    changed_box,
        })
        changed_regions.append(changed_box)
        if changed_box not in all_boxes:
            all_boxes.append(changed_box)

    if not change_records:
        return None

    # Determine dominant change type for this pair
    type_counts  = Counter(r["change_type"] for r in change_records)
    primary_type = type_counts.most_common(1)[0][0]

    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(pairs_dir / f"{pair_id}_original.jpg"), img)
    cv2.imwrite(str(pairs_dir / f"{pair_id}_changed.jpg"),  changed_img)

    gt = {
        "pair_id":       pair_id,
        "change_type":   primary_type,
        "n_changes":     len(change_records),
        "changes":       change_records,
        "image_size":    [w, h]
    }
    with open(pairs_dir / f"{pair_id}_gt.json", "w") as f:
        json.dump(gt, f, indent=2)

    heatmap = make_heatmap(h, w, changed_regions)
    cv2.imwrite(str(pairs_dir / f"{pair_id}_heatmap.png"), heatmap)

    return primary_type


# =============================================================================
# VISUALISATION
# =============================================================================

def visualise_pair(pair_id, output_dir):
    """Create a side-by-side visualisation of a pair with change boxes drawn."""
    pairs_dir = output_dir / "pairs"
    orig_path = pairs_dir / f"{pair_id}_original.jpg"
    chng_path = pairs_dir / f"{pair_id}_changed.jpg"
    gt_path   = pairs_dir / f"{pair_id}_gt.json"

    for p in [orig_path, chng_path, gt_path]:
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            return

    orig = cv2.imread(str(orig_path))
    chng = cv2.imread(str(chng_path))
    with open(gt_path) as f:
        gt = json.load(f)

    # Draw change boxes
    for change in gt["changes"]:
        ob = change["original_box"]
        cb = change["changed_box"]
        cv2.rectangle(orig, (ob[0], ob[1]), (ob[2], ob[3]), (0, 0, 255), 2)
        cv2.rectangle(chng, (cb[0], cb[1]), (cb[2], cb[3]), (0, 255, 0), 2)

    # Add labels
    label = f"Type: {gt['change_type']}  Changes: {gt['n_changes']}"
    cv2.putText(orig, "ORIGINAL", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(chng, f"CHANGED [{label}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    vis = np.hstack([orig, chng])
    out_path = output_dir / f"vis_{pair_id}.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"[INFO] Saved visualisation: {out_path}")


# =============================================================================
# MAIN GENERATION
# =============================================================================

def generate_dataset(images_dir, labels_dir, output_dir,
                     pairs_per_image=3, max_changes=2, target_pairs=1600):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(images_dir.glob("*.jpg"))
    print(f"[INFO] Found {len(all_images)} images in {images_dir}")
    print(f"[INFO] Target pairs: {target_pairs}")
    print(f"[INFO] Pairs per image: {pairs_per_image}")

    pairs         = []
    type_counts   = Counter()
    generated     = 0

    for img_path in all_images:
        if target_pairs > 0 and generated >= target_pairs:
            break
        label_path = labels_dir / f"{img_path.stem}.txt"
        img_id     = img_path.stem

        for v in range(pairs_per_image):
            if target_pairs > 0 and generated >= target_pairs:
                break
            pair_id     = f"{img_id}_v{v:02d}"
            change_type = generate_pair(img_path, label_path, pair_id,
                                        output_dir, max_changes)
            if change_type:
                pairs.append({"pair_id": pair_id, "change_type": change_type})
                type_counts[change_type] += 1
                generated += 1
                if generated % 100 == 0:
                    print(f"  Generated {generated} pairs...")

    manifest = {
        "total_pairs":               generated,
        "change_type_distribution":  dict(type_counts),
        "pairs":                     pairs
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DONE] Generated {generated} image pairs")
    print(f"[INFO] Manifest written: {output_dir}/manifest.json")
    print(f"[INFO] Total pairs generated: {generated}")
    print("[INFO] Change type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<25} {c}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic before/after image pairs"
    )
    parser.add_argument("--images_dir",     required=True)
    parser.add_argument("--labels_dir",     required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--pairs_per_image", type=int, default=3)
    parser.add_argument("--max_changes",    type=int, default=2)
    parser.add_argument("--target_pairs",   type=int, default=1600)
    parser.add_argument("--visualise",      type=str, default=None,
                        help="Pair ID to visualise (e.g. 10037_v02)")
    args = parser.parse_args()

    if args.visualise:
        visualise_pair(args.visualise, Path(args.output_dir))
    else:
        generate_dataset(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            pairs_per_image=args.pairs_per_image,
            max_changes=args.max_changes,
            target_pairs=args.target_pairs
        )

if __name__ == "__main__":
    main()
