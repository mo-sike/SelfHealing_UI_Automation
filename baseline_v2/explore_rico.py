"""
STEP 1: RICO Dataset Explorer and Annotation Converter
=======================================================
Phase 1 - Week 1, Days 1-3

What this script does:
    1. Scans your RICO dataset directory structure
    2. Explores and summarises the JSON annotation format
    3. Maps RICO component types to our 10 target classes
    4. Filters and curates a manageable subset (~3000-5000 images)
    5. Converts RICO annotations to YOLO format
    6. Generates a dataset summary report

RICO Dataset Structure expected:
    rico/
        combined/           <- screenshots (.jpg)
        semantic_annotations/  <- per-image JSON files

Usage:
    python step1_explore_rico.py --rico_dir /path/to/rico --output_dir /path/to/output
"""

import os
import json
import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
import cv2


# ─────────────────────────────────────────────
# CLASS MAPPING: RICO → Our 10 Target Classes
# ─────────────────────────────────────────────
# RICO has many fine-grained component types.
# We consolidate into 10 meaningful Android UI classes.

RICO_TO_TARGET = {
    # BUTTON
    "Button":                   "button",
    "Toggle Button":            "button",
    "Radio Button":             "button",
    "Floating Action Button":   "button",

    # TEXT
    "Text":                     "text",
    "Text Button":              "text",
    "Label":                    "text",

    # INPUT
    "Edit Text":                "input",
    "Number Picker":            "input",
    "Date Picker":              "input",
    "Time Picker":              "input",
    "Seek Bar":                 "input",

    # IMAGE
    "Image":                    "image",
    "Image Button":             "image",
    "Image View":               "image",

    # ICON
    "Icon":                     "icon",

    # CHECKBOX
    "CheckBox":                 "checkbox",
    "Switch":                   "checkbox",
    "Toggle":                   "checkbox",

    # TOOLBAR
    "Toolbar":                  "toolbar",
    "Action Bar":               "toolbar",
    "Navigation Bar":           "toolbar",
    "Status Bar":               "toolbar",

    # LIST
    "List Item":                "list_item",
    "ListView":                 "list_item",
    "RecyclerView":             "list_item",

    # CARD
    "Card View":                "card",
    "Pager":                    "card",

    # MENU / DRAWER
    "Drawer":                   "menu",
    "Bottom Navigation":        "menu",
    "Tab":                      "menu",
    "Tab Layout":               "menu",
}

# Final 10 target classes - indices used in YOLO label files
TARGET_CLASSES = [
    "button",       # 0
    "text",         # 1
    "input",        # 2
    "image",        # 3
    "icon",         # 4
    "checkbox",     # 5
    "toolbar",      # 6
    "list_item",    # 7
    "card",         # 8
    "menu",         # 9
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}


# ─────────────────────────────────────────────
# RICO JSON PARSER
# ─────────────────────────────────────────────

def parse_rico_annotation(json_path):
    """
    Parse a RICO semantic annotation JSON file.

    Handles multiple RICO JSON format variants:

    Format A - semantic_annotations (what we expect):
    {
        "activity_name": "...",
        "root": {
            "class": "android.widget.Button",
            "bounds": [x1, y1, x2, y2],
            "children": [...]
        }
    }

    Format B - some RICO versions use "componentLabel":
    {
        "componentLabel": "Button",
        "bounds": [x1, y1, x2, y2],
        "children": [...]
    }

    Format C - flat list of components:
    [
        {"class": "Button", "bounds": [...]},
        ...
    ]

    Returns: list of dicts with keys: class_name, bounds [x1,y1,x2,y2]
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return None, str(e)

    components = []

    def extract_class_name(node):
        """Try all known keys for class name in RICO formats."""
        for key in ["componentLabel", "class", "type", "widget_class"]:
            val = node.get(key, "")
            if val:
                # Strip Java package path e.g. "android.widget.Button" -> "Button"
                if "." in val:
                    val = val.split(".")[-1]
                return val.strip()
        return ""

    def traverse(node, depth=0):
        if not isinstance(node, dict):
            return

        class_name = extract_class_name(node)

        # Try multiple bounds key names
        bounds = (node.get("bounds") or
                  node.get("bound") or
                  node.get("bbox") or
                  node.get("rect"))

        if bounds and len(bounds) == 4:
            x1, y1, x2, y2 = bounds
            if x2 > x1 and y2 > y1 and class_name:
                components.append({
                    "class_name": class_name,
                    "bounds": [x1, y1, x2, y2],
                    "depth": depth
                })

        # Recurse — try multiple child key names
        children = (node.get("children") or
                    node.get("child") or
                    node.get("nodes") or [])
        if isinstance(children, list):
            for child in children:
                traverse(child, depth + 1)
        elif isinstance(children, dict):
            traverse(children, depth + 1)

    # Handle all three top-level formats
    if isinstance(data, list):
        # Format C: flat list
        for item in data:
            traverse(item)
    elif isinstance(data, dict):
        # Format A/B: try "root" key first, then traverse the dict itself
        root = data.get("root") or data.get("Root") or data.get("hierarchy")
        if root:
            traverse(root)
        else:
            # The dict itself might be the root node
            traverse(data)

    return components, None


def debug_json_structure(json_path):
    """
    Print the structure of a RICO JSON file to understand its format.
    Run this manually if parse_rico_annotation returns 0 components.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def show_structure(obj, depth=0, max_depth=3):
        indent = "  " * depth
        if depth > max_depth:
            print(f"{indent}...")
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items())[:8]:
                if isinstance(v, (dict, list)):
                    print(f"{indent}[key] {k}:")
                    show_structure(v, depth+1, max_depth)
                else:
                    print(f"{indent}[key] {k}: {str(v)[:80]}")
        elif isinstance(obj, list):
            print(f"{indent}[list of {len(obj)}]")
            if obj:
                show_structure(obj[0], depth+1, max_depth)

    print(f"\n=== JSON STRUCTURE: {Path(json_path).name} ===")
    show_structure(data)
    print("===\n")


def map_to_target_class(rico_class_name):
    """Map a RICO class name to our target class. Returns None if not mapped."""
    # Direct match
    if rico_class_name in RICO_TO_TARGET:
        return RICO_TO_TARGET[rico_class_name]

    # Partial match - check if any key is a substring
    rico_lower = rico_class_name.lower()
    for key, target in RICO_TO_TARGET.items():
        if key.lower() in rico_lower or rico_lower in key.lower():
            return target

    # Keyword fallback
    if any(k in rico_lower for k in ["button", "btn"]):
        return "button"
    if any(k in rico_lower for k in ["text", "label", "title"]):
        return "text"
    if any(k in rico_lower for k in ["image", "img", "photo"]):
        return "image"
    if any(k in rico_lower for k in ["icon", "drawable"]):
        return "icon"
    if any(k in rico_lower for k in ["edit", "input", "field"]):
        return "input"
    if any(k in rico_lower for k in ["check", "switch", "toggle"]):
        return "checkbox"
    if any(k in rico_lower for k in ["toolbar", "bar", "action"]):
        return "toolbar"
    if any(k in rico_lower for k in ["list", "recycler", "scroll"]):
        return "list_item"
    if any(k in rico_lower for k in ["card", "frame"]):
        return "card"
    if any(k in rico_lower for k in ["menu", "drawer", "nav", "tab"]):
        return "menu"

    return None


# ─────────────────────────────────────────────
# YOLO ANNOTATION CONVERTER
# ─────────────────────────────────────────────

def convert_to_yolo(components, img_width, img_height):
    """
    Convert RICO component annotations to YOLO format.

    YOLO format per line:
        class_idx  cx  cy  w  h
    All values normalized to [0, 1] by image dimensions.

    Returns: list of YOLO annotation strings
    """
    yolo_lines = []

    for comp in components:
        target_class = map_to_target_class(comp["class_name"])
        if target_class is None:
            continue

        class_idx = CLASS_TO_IDX[target_class]
        x1, y1, x2, y2 = comp["bounds"]

        # Clamp to image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))

        # Skip tiny components (likely noise)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue

        # Convert to YOLO center + size format, normalized
        cx = ((x1 + x2) / 2) / img_width
        cy = ((y1 + y2) / 2) / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height

        yolo_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


# ─────────────────────────────────────────────
# DATASET CURATION
# ─────────────────────────────────────────────

def curate_dataset(rico_dir, output_dir, target_count=4000, min_components=3, seed=42):
    """
    Curate a subset of RICO dataset for our project.

    Filtering criteria:
    - Image must have a corresponding JSON annotation
    - Image must have at least min_components mappable components
    - Random sample of target_count images for manageability

    Output structure:
        output_dir/
            images/
                train/   <- 80%
                val/     <- 10%
                test/    <- 10%
            labels/
                train/
                val/
                test/
            dataset.yaml
            summary.json
    """
    rico_dir = Path(rico_dir)
    output_dir = Path(output_dir)
    test_screenshots = r"C:\Users\saqla\Downloads\archive\unique_uis\combined"
    # screenshots_dir = rico_dir / "combined"
    screenshots_dir = Path(test_screenshots)
    annotations_dir = rico_dir / "semantic_annotations"

    # Validate RICO structure
    if not screenshots_dir.exists():
        print(f"[ERROR] Screenshots directory not found: {screenshots_dir}")
        print("Expected: rico/combined/*.jpg")
        return False

    if not annotations_dir.exists():
        print(f"[ERROR] Annotations directory not found: {annotations_dir}")
        print("Expected: rico/semantic_annotations/*.json")
        return False

    print(f"\n[INFO] Scanning RICO dataset at: {rico_dir}")

    # Find all valid pairs (image + annotation)
    all_json_files = list(annotations_dir.glob("*.json"))
    print(f"[INFO] Found {len(all_json_files)} annotation files")

    valid_samples = []
    class_distribution = Counter()
    skipped = 0

    for json_path in all_json_files:
        img_id = json_path.stem
        img_path = screenshots_dir / f"{img_id}.jpg"

        if not img_path.exists():
            skipped += 1
            continue

        # Parse annotation
        components, error = parse_rico_annotation(json_path)
        if error or not components:
            skipped += 1
            continue

        # Count mappable components
        mapped = []
        for comp in components:
            target = map_to_target_class(comp["class_name"])
            if target:
                mapped.append((comp, target))

        if len(mapped) < min_components:
            skipped += 1
            continue

        valid_samples.append({
            "img_id": img_id,
            "img_path": str(img_path),
            "json_path": str(json_path),
            "component_count": len(mapped)
        })

        for _, target in mapped:
            class_distribution[target] += 1

    print(f"[INFO] Valid samples found: {len(valid_samples)}")
    print(f"[INFO] Skipped (no pair / too few components): {skipped}")

    # Sample subset
    random.seed(seed)
    if len(valid_samples) > target_count:
        valid_samples = random.sample(valid_samples, target_count)
        print(f"[INFO] Sampled {target_count} images from valid set")

    # Train/val/test split: 80/10/10
    random.shuffle(valid_samples)
    n = len(valid_samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": valid_samples[:n_train],
        "val":   valid_samples[n_train:n_train + n_val],
        "test":  valid_samples[n_train + n_val:]
    }

    print(f"\n[INFO] Split sizes:")
    for split, samples in splits.items():
        print(f"  {split}: {len(samples)}")

    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process and copy files
    print(f"\n[INFO] Processing and copying files...")
    stats = defaultdict(int)

    for split, samples in splits.items():
        for sample in samples:
            img_path = Path(sample["img_path"])
            json_path = Path(sample["json_path"])
            img_id = sample["img_id"]

            # Read image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                stats["imread_failed"] += 1
                continue

            h, w = img.shape[:2]

            # Parse annotation and convert to YOLO
            components, error = parse_rico_annotation(json_path)
            if error:
                stats["parse_failed"] += 1
                continue

            yolo_lines = convert_to_yolo(components, w, h)
            if not yolo_lines:
                stats["no_valid_labels"] += 1
                continue

            # Copy image
            dst_img = output_dir / "images" / split / f"{img_id}.jpg"
            shutil.copy2(img_path, dst_img)

            # Write YOLO label file
            dst_label = output_dir / "labels" / split / f"{img_id}.txt"
            with open(dst_label, "w") as f:
                f.write("\n".join(yolo_lines))

            stats[f"{split}_processed"] += 1

    print(f"[INFO] Processing stats: {dict(stats)}")

    # Write dataset.yaml for YOLOv5
    yaml_content = f"""# RICO Android UI Dataset - Visual Change Detection Project
# Phase 1 - Week 1

path: {str(output_dir.absolute())}
train: images/train
val: images/val
test: images/test

nc: {len(TARGET_CLASSES)}
names: {TARGET_CLASSES}

# Class distribution (from full valid set before sampling)
# {dict(class_distribution)}
"""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    # Write summary JSON
    summary = {
        "total_valid_samples": len(valid_samples),
        "splits": {k: len(v) for k, v in splits.items()},
        "class_distribution": dict(class_distribution),
        "target_classes": TARGET_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "stats": dict(stats)
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Dataset prepared at: {output_dir}")
    print(f"[DONE] dataset.yaml written for YOLOv5 training")
    return True


# ─────────────────────────────────────────────
# QUICK EXPLORATION (run before full curation)
# ─────────────────────────────────────────────

def explore_rico(rico_dir, sample_n=50):
    """
    Quick exploration of RICO dataset structure.
    Run this FIRST to understand your data before full processing.
    """
    rico_dir = Path(rico_dir)
    annotations_dir = rico_dir / "semantic_annotations"

    all_json = list(annotations_dir.glob("*.json"))[:sample_n]

    if not all_json:
        print(f"[ERROR] No JSON files found in: {annotations_dir}")
        print("Check your --rico_dir path and that semantic_annotations/ folder exists.")
        return

    print(f"[INFO] Found {len(all_json)} JSON files to sample")

    # --- DEBUG: Print structure of first 3 files to identify format ---
    print("\n[DEBUG] Inspecting first 3 JSON files to identify format...")
    for json_path in all_json[:3]:
        debug_json_structure(json_path)
    # ------------------------------------------------------------------

    all_rico_classes = Counter()
    mapped_classes = Counter()
    unmapped_classes = Counter()
    component_counts = []
    parse_errors = 0
    empty_parses = 0

    for json_path in all_json:
        components, error = parse_rico_annotation(json_path)
        if error:
            parse_errors += 1
            continue
        if not components:
            empty_parses += 1
            continue

        component_counts.append(len(components))

        for comp in components:
            class_name = comp["class_name"]
            all_rico_classes[class_name] += 1

            target = map_to_target_class(class_name)
            if target:
                mapped_classes[target] += 1
            else:
                unmapped_classes[class_name] += 1

    print("\n" + "="*60)
    print("RICO DATASET EXPLORATION REPORT")
    print("="*60)
    print(f"\nSampled:      {len(all_json)} annotation files")
    print(f"Parse errors: {parse_errors}")
    print(
        f"Empty parses: {empty_parses}  ← if this is high, JSON format not recognised")
    print(f"Successful:   {len(component_counts)}")

    if not component_counts:
        print("\n[WARNING] Zero components parsed from all files!")
        print("The JSON structure is different from expected.")
        print(
            "Check the [DEBUG] output above and update parse_rico_annotation()")
        print("\nCommon fixes:")
        print(
            "  - RICO has different annotation folders (semantic_annotations vs ui_details)")
        print("  - Some RICO zips use 'componentLabel' others use 'class' as the key")
        print("  - Check if you downloaded the right zip: semantic_annotations.zip")
        return

    print(
        f"\nAvg components per image: {sum(component_counts)/len(component_counts):.1f}")
    print(
        f"Min/Max components:       {min(component_counts)} / {max(component_counts)}")

    print(f"\nTop 20 RICO component classes found:")
    for cls, count in all_rico_classes.most_common(20):
        target = map_to_target_class(cls)
        mapped_str = f"→ {target}" if target else "→ [UNMAPPED]"
        print(f"  {cls:<40} {count:>5}  {mapped_str}")

    print(f"\nMapped to target classes:")
    for cls, count in mapped_classes.most_common():
        print(f"  {cls:<20} {count:>5}")

    mapping_rate = sum(mapped_classes.values()) / \
        max(sum(all_rico_classes.values()), 1) * 100
    print(
        f"\n  Mapping coverage: {mapping_rate:.1f}% of all components mapped")

    print(f"\nTop unmapped classes (consider adding to RICO_TO_TARGET if significant):")
    for cls, count in unmapped_classes.most_common(15):
        print(f"  {cls:<40} {count:>5}")

    print("\n" + "="*60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RICO Dataset Explorer and Converter")
    parser.add_argument("--rico_dir", default=r"C:\Users\saqla\Downloads\archive\rico_dataset_v0.1_semantic_annotations",
                        help="Path to RICO root directory")
    parser.add_argument("--output_dir",  default="./rico_yolo_dataset",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--explore_only", action="store_true",
                        help="Only run exploration, skip full processing")
    parser.add_argument("--target_count", type=int,
                        default=4000, help="Number of images to curate")
    parser.add_argument("--min_components", type=int, default=3,
                        help="Min mappable components per image")
    args = parser.parse_args()

    if args.explore_only:
        explore_rico(args.rico_dir, sample_n=100)
    else:
        # First explore
        explore_rico(args.rico_dir, sample_n=100)
        # Then curate
        curate_dataset(
            rico_dir=args.rico_dir,
            output_dir=args.output_dir,
            target_count=args.target_count,
            min_components=args.min_components
        )


if __name__ == "__main__":
    main()
