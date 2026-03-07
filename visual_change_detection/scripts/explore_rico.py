"""
step1_explore_rico.py
=====================
Phase 1 - Week 1, Days 1-3  |  Run on: LOCAL PC

What this does:
    1. Explores RICO JSON structure        (--explore_only flag)
    2. Curates 4000 images from 66k RICO
    3. Converts annotations to YOLO format
    4. Creates 80/10/10 train/val/test split
    5. Writes dataset.yaml ready for Kaggle training

Usage:
    # Step A - always explore first
    python scripts/step1_explore_rico.py --rico_dir ./data/rico --explore_only

    # Step B - full processing
    python scripts/step1_explore_rico.py --rico_dir ./data/rico --output_dir ./outputs/rico_yolo_dataset

Expected output (Step B):
    outputs/rico_yolo_dataset/
        images/train/    ~3165 images
        images/val/      ~386  images
        images/test/     ~391  images
        labels/train/    ~3165 .txt files
        labels/val/      ~386  .txt files
        labels/test/     ~391  .txt files
        dataset.yaml     YOLOv8 config
        summary.json     Stats
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import cv2


# =============================================================================
# CLASS MAPPING: RICO labels → 10 target classes
# =============================================================================

RICO_TO_TARGET = {
    # button
    "Button":                   "button",
    "Toggle Button":            "button",
    "Radio Button":             "button",
    "Floating Action Button":   "button",
    # text
    "Text":                     "text",
    "Text Button":              "text",
    "Label":                    "text",
    # input
    "Edit Text":                "input",
    "Number Picker":            "input",
    "Date Picker":              "input",
    "Time Picker":              "input",
    "Seek Bar":                 "input",
    # image
    "Image":                    "image",
    "Image Button":             "image",
    "Image View":               "image",
    "Background Image":         "image",
    "Advertisement":            "image",
    # icon
    "Icon":                     "icon",
    # checkbox
    "CheckBox":                 "checkbox",
    "Switch":                   "checkbox",
    "Toggle":                   "checkbox",
    # toolbar
    "Toolbar":                  "toolbar",
    "Action Bar":               "toolbar",
    "Navigation Bar":           "toolbar",
    "Status Bar":               "toolbar",
    # list_item
    "List Item":                "list_item",
    "ListView":                 "list_item",
    "RecyclerView":             "list_item",
    # card
    "Card View":                "card",
    "Pager":                    "card",
    "Modal":                    "card",
    "Web View":                 "card",
    # menu
    "Drawer":                   "menu",
    "Bottom Navigation":        "menu",
    "Tab":                      "menu",
    "Tab Layout":               "menu",
    "Multi-Tab":                "menu",
    "Checkbox":             "checkbox",   # capital C, no capital B
    "Card":                 "card",       # without 'View' suffix
    "Input":                "input",      # plain 'Input' label
    "Pager Indicator":      "card",       # found in your data
    "Map View":             "image",
}

TARGET_CLASSES = [
    "button",    # 0
    "text",      # 1
    "input",     # 2
    "image",     # 3
    "icon",      # 4
    "checkbox",  # 5
    "toolbar",   # 6
    "list_item",  # 7
    "card",      # 8
    "menu",      # 9
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}


# =============================================================================
# JSON PARSER
# =============================================================================

def debug_json_structure(json_path):
    """Print raw JSON structure — call this if empty parses occur."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def show(obj, depth=0, max_depth=3):
        indent = "  " * depth
        if depth > max_depth:
            print(f"{indent}...")
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items())[:8]:
                if isinstance(v, (dict, list)):
                    print(f"{indent}[key] {k}:")
                    show(v, depth + 1, max_depth)
                else:
                    print(f"{indent}[key] {k}: {str(v)[:80]}")
        elif isinstance(obj, list):
            print(f"{indent}[list of {len(obj)}]")
            if obj:
                show(obj[0], depth + 1, max_depth)

    print(f"\n=== STRUCTURE: {Path(json_path).name} ===")
    show(data)
    print("===\n")


def parse_rico_annotation(json_path):
    """
    Parse RICO JSON — handles all known RICO format variants.

    Confirmed RICO structure:
        - Root dict IS the node (no wrapper 'root' key)
        - 'componentLabel' = semantic class name
        - 'bounds' = [left, top, right, bottom]
        - Children under 'children' list
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return None, str(e)

    components = []

    def get_class(node):
        for key in ["componentLabel", "class", "type", "widget_class"]:
            val = node.get(key, "")
            if val:
                # Strip Java package path: android.widget.Button -> Button
                return val.split(".")[-1].strip()
        return ""

    def traverse(node):
        if not isinstance(node, dict):
            return
        cls = get_class(node)
        bounds = (node.get("bounds") or node.get("bound") or
                  node.get("bbox") or node.get("rect"))
        if bounds and len(bounds) == 4:
            x1, y1, x2, y2 = bounds
            if x2 > x1 and y2 > y1 and cls:
                components.append({
                    "class_name": cls,
                    "bounds": [x1, y1, x2, y2]
                })
        for child in (node.get("children") or []):
            traverse(child)

    if isinstance(data, list):
        for item in data:
            traverse(item)
    elif isinstance(data, dict):
        root = data.get("root") or data.get("Root") or data.get("hierarchy")
        traverse(root if root else data)

    return components, None


def map_to_target(cls):
    """Map RICO class name to one of 10 target classes. Returns None if unmapped."""
    if cls in RICO_TO_TARGET:
        return RICO_TO_TARGET[cls]
    lower = cls.lower()
    for k, v in RICO_TO_TARGET.items():
        if k.lower() in lower or lower in k.lower():
            return v
    if any(k in lower for k in ["button", "btn"]):
        return "button"
    if any(k in lower for k in ["text", "label", "title"]):
        return "text"
    if any(k in lower for k in ["image", "img", "photo"]):
        return "image"
    if any(k in lower for k in ["icon", "drawable"]):
        return "icon"
    if any(k in lower for k in ["edit", "input", "field"]):
        return "input"
    if any(k in lower for k in ["check", "switch", "toggle"]):
        return "checkbox"
    if any(k in lower for k in ["toolbar", "bar", "action"]):
        return "toolbar"
    if any(k in lower for k in ["list", "recycler"]):
        return "list_item"
    if any(k in lower for k in ["card", "frame"]):
        return "card"
    if any(k in lower for k in ["menu", "drawer", "nav", "tab"]):
        return "menu"
    return None


def to_yolo_lines(components, img_w, img_h):
    """Convert component list to YOLO format label strings."""
    lines = []
    for comp in components:
        target = map_to_target(comp["class_name"])
        if not target:
            continue
        x1, y1, x2, y2 = comp["bounds"]
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(
            f"{CLASS_TO_IDX[target]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        )
    return lines


# =============================================================================
# EXPLORATION
# =============================================================================

def explore_rico(rico_dir, sample_n=100):
    """Quick exploration — always run before full processing."""
    annots_dir = Path(rico_dir) / "semantic_annotations"
    all_json = sorted(annots_dir.glob("*.json"))[:sample_n]

    if not all_json:
        print(f"[ERROR] No JSON files at: {annots_dir}")
        return

    print(f"[INFO] Sampling {len(all_json)} files from {annots_dir}")
    print("\n[DEBUG] First 3 file structures:")
    for p in all_json[:3]:
        debug_json_structure(p)

    all_classes = Counter()
    mapped = Counter()
    unmapped = Counter()
    counts = []
    parse_err = 0
    empty = 0

    for p in all_json:
        comps, err = parse_rico_annotation(p)
        if err:
            parse_err += 1
            continue
        if not comps:
            empty += 1
            continue
        counts.append(len(comps))
        for c in comps:
            all_classes[c["class_name"]] += 1
            t = map_to_target(c["class_name"])
            if t:
                mapped[t] += 1
            else:
                unmapped[c["class_name"]] += 1

    print("\n" + "=" * 60)
    print("RICO EXPLORATION REPORT")
    print("=" * 60)
    print(f"Sampled:          {len(all_json)}")
    print(f"Parse errors:     {parse_err}")
    print(f"Empty parses:     {empty}")
    print(f"Successful:       {len(counts)}")
    if not counts:
        print("\n[WARNING] Nothing parsed — check JSON structure above")
        return
    print(f"Avg components:   {sum(counts)/len(counts):.1f}  "
          f"(min:{min(counts)} max:{max(counts)})")
    coverage = sum(mapped.values()) / max(sum(all_classes.values()), 1) * 100
    print(f"Mapping coverage: {coverage:.1f}%")
    print("\nTop 20 RICO classes:")
    for cls, cnt in all_classes.most_common(20):
        print(f"  {cls:<42} {cnt:>5}  -> {map_to_target(cls) or '[UNMAPPED]'}")
    print("\nMapped target classes:")
    for cls, cnt in mapped.most_common():
        print(f"  {cls:<20} {cnt:>5}")
    print("\nTop unmapped classes:")
    for cls, cnt in unmapped.most_common(10):
        print(f"  {cls:<42} {cnt:>5}")
    print("=" * 60)


# =============================================================================
# DATASET CURATION
# =============================================================================

def curate_dataset(rico_dir, output_dir, target_count=4000,
                   min_components=3, seed=42):
    """Curate RICO subset and write YOLO-format dataset."""
    rico_dir = Path(rico_dir)
    output_dir = Path(output_dir)
    screens_dir = rico_dir / "combined"
    annots_dir = rico_dir / "semantic_annotations"

    for d in [screens_dir, annots_dir]:
        if not d.exists():
            print(f"[ERROR] Not found: {d}")
            return False

    print(f"\n[INFO] Scanning RICO at {rico_dir} ...")
    all_jsons = sorted(annots_dir.glob("*.json"))
    print(f"[INFO] {len(all_jsons)} annotation files found")

    valid = []
    skipped = 0

    for jp in all_jsons:
        img_path = screens_dir / f"{jp.stem}.jpg"
        if not img_path.exists():
            skipped += 1
            continue
        comps, err = parse_rico_annotation(jp)
        if err or not comps:
            skipped += 1
            continue
        mapped_count = len(
            [c for c in comps if map_to_target(c["class_name"])])
        if mapped_count < min_components:
            skipped += 1
            continue
        valid.append({
            "img_id":    jp.stem,
            "img_path":  str(img_path),
            "json_path": str(jp)
        })

    print(f"[INFO] Valid: {len(valid)}  Skipped: {skipped}")

    random.seed(seed)
    if len(valid) > target_count:
        valid = random.sample(valid, target_count)
        print(f"[INFO] Sampled down to {target_count}")
    random.shuffle(valid)

    n = len(valid)
    splits = {
        "train": valid[:int(n * 0.8)],
        "val":   valid[int(n * 0.8):int(n * 0.9)],
        "test":  valid[int(n * 0.9):]
    }
    print("[INFO] Splits:", {k: len(v) for k, v in splits.items()})

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("[INFO] Writing images and labels ...")
    stats = defaultdict(int)

    for split, samples in splits.items():
        for s in samples:
            img = cv2.imread(s["img_path"])
            if img is None:
                stats["imread_failed"] += 1
                continue
            h, w = img.shape[:2]
            comps, err = parse_rico_annotation(Path(s["json_path"]))
            if err:
                stats["parse_failed"] += 1
                continue
            lines = to_yolo_lines(comps, w, h)
            if not lines:
                stats["no_valid_labels"] += 1
                continue
            shutil.copy2(
                s["img_path"],
                output_dir / "images" / split / f"{s['img_id']}.jpg"
            )
            with open(output_dir / "labels" / split /
                      f"{s['img_id']}.txt", "w") as f:
                f.write("\n".join(lines))
            stats[f"{split}_ok"] += 1

    print("[INFO] Processing stats:", dict(stats))

    # Write dataset.yaml
    abs_path = str(output_dir.resolve()).replace("\\", "/")
    yaml_content = f"""# RICO Android UI Dataset - Visual Change Detection Project
# Generated by step1_explore_rico.py
# NOTE: Update 'path' to Kaggle input path before uploading to Kaggle
#   Kaggle path: /kaggle/input/rico-yolo-ui-dataset

path: {abs_path}
train: images/train
val:   images/val
test:  images/test

nc: 10
names:
  0: button
  1: text
  2: input
  3: image
  4: icon
  5: checkbox
  6: toolbar
  7: list_item
  8: card
  9: menu
"""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "splits": {k: len(v) for k, v in splits.items()},
            "stats":  dict(stats)
        }, f, indent=2)

    print(f"\n[DONE] Dataset written to: {output_dir}")
    print(f"[DONE] dataset.yaml path:   {abs_path}")
    print("[DONE] Next step: run step2_generate_changes.py")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Explore and curate RICO dataset for YOLO training"
    )
    parser.add_argument("--rico_dir",       required=True,
                        help="Path to RICO root folder (contains combined/ and semantic_annotations/)")
    parser.add_argument("--output_dir",     default="./outputs/rico_yolo_dataset",
                        help="Where to write YOLO dataset")
    parser.add_argument("--explore_only",   action="store_true",
                        help="Only run exploration, no file writing")
    parser.add_argument("--target_count",   type=int, default=4000,
                        help="How many images to curate")
    parser.add_argument("--min_components", type=int, default=3,
                        help="Minimum mapped components required per image")
    args = parser.parse_args()

    if args.explore_only:
        explore_rico(args.rico_dir)
    else:
        explore_rico(args.rico_dir)
        curate_dataset(
            rico_dir=args.rico_dir,
            output_dir=args.output_dir,
            target_count=args.target_count,
            min_components=args.min_components
        )


if __name__ == "__main__":
    main()
