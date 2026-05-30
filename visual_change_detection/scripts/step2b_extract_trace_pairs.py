"""
step2b_extract_trace_pairs.py
=============================
Phase 1 - Dataset Upgrade  |  Run on: LOCAL PC

What this does:
    Mines RICO Interaction Traces to extract real before/after UI pairs.
    
    For each app trace:
        1. Loads consecutive screenshot pairs (A → B)
        2. Computes perceptual hash similarity between A and B
        3. Keeps pairs where 0.72 < similarity < 0.97
           - < 0.72 = completely different screen (navigation, skip)
           - > 0.97 = identical screen, no visible change (skip)
           - 0.72-0.97 = same screen with real UI state change (KEEP)
        4. Records gesture coordinates (where user tapped on A to get B)
        5. Saves valid pairs + manifest

    Output structure mirrors change_dataset/ so all downstream scripts
    (graph_matcher, evaluate) work without modification.

Usage:
    # Explore first — see how many valid pairs exist
    python scripts/step2b_extract_trace_pairs.py ^
        --traces_dir C:\dataset\rico_dir\traces\filtered_traces ^
        --output_dir ./outputs/trace_pairs ^
        --explore_only

    # Full extraction
    python scripts/step2b_extract_trace_pairs.py ^
        --traces_dir C:\dataset\rico_dir\traces\filtered_traces ^
        --output_dir ./outputs/trace_pairs ^
        --target_pairs 2000 ^
        --sim_low 0.72 ^
        --sim_high 0.97

    # After extraction, run graph matcher directly:
    python scripts/graph_matcher.py ^
        --model      ./outputs/yolo_runs/rico_ui_v2/weights/best.pt ^
        --pairs_dir  ./outputs/trace_pairs/pairs ^
        --manifest   ./outputs/trace_pairs/manifest.json ^
        --output_dir ./outputs/results/trace_test ^
        --threshold  0.6
"""

import os
import cv2
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

try:
    import imagehash
    from PIL import Image
    IMAGEHASH_OK = True
except ImportError:
    IMAGEHASH_OK = False
    print("[ERROR] imagehash required: pip install imagehash Pillow")
    exit(1)


# =============================================================================
# PHASH SIMILARITY
# =============================================================================

def phash_similarity(img_path1, img_path2):
    """
    Compute perceptual hash similarity between two image files.
    Returns float in [0, 1]. 1 = identical, 0 = completely different.
    """
    try:
        h1 = imagehash.phash(Image.open(img_path1).convert("RGB"))
        h2 = imagehash.phash(Image.open(img_path2).convert("RGB"))
        dist = h1 - h2  # Hamming distance 0-64
        return max(0.0, 1.0 - dist / 64.0)
    except Exception as e:
        return 0.0


def classify_similarity(sim, sim_low=0.72, sim_high=0.97):
    """Classify a similarity score into a category."""
    if sim < sim_low:
        return "different_screen"   # skip
    elif sim > sim_high:
        return "identical"          # skip
    else:
        return "valid_change"       # keep


# =============================================================================
# GESTURE HELPERS
# =============================================================================

def get_gesture_coords(gestures, screen_id):
    """
    Get normalised (x, y) tap coordinates for a screen ID.
    Returns (x, y) or None if not found.
    """
    key = str(screen_id)
    if key not in gestures:
        return None
    coords = gestures[key]
    if coords and len(coords) > 0 and len(coords[0]) >= 2:
        return (float(coords[0][0]), float(coords[0][1]))
    return None


def gesture_to_pixel_box(gesture_xy, img_w, img_h, box_size=0.15):
    """
    Convert normalised gesture coordinates to a pixel bounding box.
    Creates a box of relative size around the tap point.
    Used as approximate ground truth for where the change occurred.
    """
    if gesture_xy is None:
        return None
    x, y = gesture_xy
    half = box_size / 2
    x1 = max(0.0, x - half)
    y1 = max(0.0, y - half)
    x2 = min(1.0, x + half)
    y2 = min(1.0, y + half)
    return [
        int(x1 * img_w), int(y1 * img_h),
        int(x2 * img_w), int(y2 * img_h)
    ]


# =============================================================================
# TRACE EXPLORATION
# =============================================================================

def explore_traces(traces_dir, sample_n=200, sim_low=0.72, sim_high=0.97):
    """
    Sample traces and report similarity distribution.
    Run this first to validate filtering thresholds.
    """
    traces_dir = Path(traces_dir)
    apps = sorted(os.listdir(traces_dir))
    random.shuffle(apps)

    sim_scores = []
    categories = Counter()
    pairs_seen = 0
    errors = 0

    print(f"[INFO] Sampling {sample_n} apps from {len(apps)} total ...")

    for app in apps[:sample_n]:
        app_dir = traces_dir / app
        for trace in sorted(os.listdir(app_dir)):
            if not trace.startswith("trace"):
                continue
            trace_dir = app_dir / trace
            screen_dir = trace_dir / "screenshots"
            if not screen_dir.exists():
                continue

            # Get real screenshots (ignore ._prefixed macOS artifacts)
            screens = sorted([
                f for f in os.listdir(screen_dir)
                if f.endswith(".jpg") and not f.startswith("._")
            ], key=lambda x: int(x.replace(".jpg", "")))

            if len(screens) < 2:
                continue

            # Check consecutive pairs
            for i in range(len(screens) - 1):
                p1 = screen_dir / screens[i]
                p2 = screen_dir / screens[i + 1]
                try:
                    sim = phash_similarity(p1, p2)
                    sim_scores.append(sim)
                    cat = classify_similarity(sim, sim_low, sim_high)
                    categories[cat] += 1
                    pairs_seen += 1
                except Exception:
                    errors += 1

    if not sim_scores:
        print("[ERROR] No pairs found — check traces_dir path")
        return

    print("\n" + "=" * 60)
    print("RICO TRACES EXPLORATION REPORT")
    print("=" * 60)
    print(f"Apps sampled:      {sample_n}")
    print(f"Pairs checked:     {pairs_seen}")
    print(f"Errors:            {errors}")
    print(f"\nSimilarity stats:")
    print(f"  Mean:   {np.mean(sim_scores):.3f}")
    print(f"  Median: {np.median(sim_scores):.3f}")
    print(f"  Min:    {np.min(sim_scores):.3f}")
    print(f"  Max:    {np.max(sim_scores):.3f}")
    print(
        f"\nCategory breakdown (thresholds: low={sim_low}, high={sim_high}):")
    total = sum(categories.values())
    for cat, count in categories.most_common():
        pct = count / total * 100
        print(f"  {cat:<25} {count:>5}  ({pct:.1f}%)")
    valid = categories.get("valid_change", 0)
    print(f"\nEstimated valid pairs from full dataset:")
    print(f"  {valid}/{total} = {valid/total*100:.1f}% valid in sample")
    est_total = int((valid / total) * pairs_seen / sample_n * len(apps))
    print(f"  Projected full dataset: ~{est_total} valid pairs")
    print("=" * 60)
    print("\n[INFO] If projected pairs > 2000, thresholds are good.")
    print("[INFO] Adjust --sim_low and --sim_high if needed.")


# =============================================================================
# PAIR EXTRACTION
# =============================================================================

def extract_pairs(traces_dir, output_dir, target_pairs=2000,
                  sim_low=0.72, sim_high=0.97, seed=42):
    """
    Extract valid before/after pairs from RICO traces.
    Output format mirrors change_dataset/ for compatibility with
    graph_matcher.py and evaluate.py without modification.
    """
    traces_dir = Path(traces_dir)
    output_dir = Path(output_dir)
    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    apps = sorted(os.listdir(traces_dir))
    random.seed(seed)
    random.shuffle(apps)

    print(f"[INFO] {len(apps)} apps available")
    print(f"[INFO] Target: {target_pairs} valid pairs")
    print(f"[INFO] Similarity window: [{sim_low}, {sim_high}]")

    manifest_pairs = []
    stats = Counter()
    processed_apps = 0

    for app in apps:
        if len(manifest_pairs) >= target_pairs:
            break

        app_dir = traces_dir / app
        if not app_dir.is_dir():
            continue

        for trace in sorted(os.listdir(app_dir)):
            if not trace.startswith("trace"):
                continue
            if len(manifest_pairs) >= target_pairs:
                break

            trace_dir = app_dir / trace
            screen_dir = trace_dir / "screenshots"
            vh_dir = trace_dir / "view_hierarchies"
            gest_path = trace_dir / "gestures.json"

            if not screen_dir.exists():
                stats["no_screenshots"] += 1
                continue

            # Load gestures
            gestures = {}
            if gest_path.exists():
                try:
                    with open(gest_path) as f:
                        gestures = json.load(f)
                except Exception:
                    pass

            # Get real screenshots only (skip ._prefixed macOS artifacts)
            screens = sorted([
                f for f in os.listdir(screen_dir)
                if f.endswith(".jpg") and not f.startswith("._")
            ], key=lambda x: int(x.replace(".jpg", "")))

            if len(screens) < 2:
                stats["too_few_screens"] += 1
                continue

            # Check consecutive pairs
            for i in range(len(screens) - 1):
                if len(manifest_pairs) >= target_pairs:
                    break

                screen_a_name = screens[i]
                screen_b_name = screens[i + 1]
                screen_a_id = screen_a_name.replace(".jpg", "")
                screen_b_id = screen_b_name.replace(".jpg", "")

                path_a = screen_dir / screen_a_name
                path_b = screen_dir / screen_b_name

                # Compute similarity
                sim = phash_similarity(path_a, path_b)
                cat = classify_similarity(sim, sim_low, sim_high)

                if cat != "valid_change":
                    stats[cat] += 1
                    continue

                # Load images
                img_a = cv2.imread(str(path_a))
                img_b = cv2.imread(str(path_b))
                if img_a is None or img_b is None:
                    stats["imread_failed"] += 1
                    continue

                h, w = img_a.shape[:2]

                # Get gesture ground truth box
                gesture_xy = get_gesture_coords(gestures, screen_a_id)
                gesture_box = pixel_diff_ground_truth(img_a, img_b)

                # Build pair ID
                pair_id = f"{app}__{trace}__{screen_a_id}__{screen_b_id}"
                # Truncate if too long for filesystem
                if len(pair_id) > 100:
                    pair_id = f"{len(manifest_pairs):05d}_{screen_a_id}_{screen_b_id}"

                # Save images
                if gesture_box is None:
                    stats["gt_box_too_large"] += 1
                    continue

                # Save images
                cv2.imwrite(str(pairs_dir / f"{pair_id}_original.jpg"), img_a)
                cv2.imwrite(str(pairs_dir / f"{pair_id}_changed.jpg"),  img_b)

                # Save ground truth
                # gt = {
                #     "pair_id":      pair_id,
                #     "app":          app,
                #     "trace":        trace,
                #     "screen_a_id":  screen_a_id,
                #     "screen_b_id":  screen_b_id,
                #     "similarity":   float(sim),
                #     "change_type":  "interaction",   # real interaction change
                #     "gesture_xy":   list(gesture_xy) if gesture_xy else None,
                #     "changes": [{
                #         "change_type": "interaction",
                #         "original_box": gesture_box,
                #         "changed_box":  gesture_box,
                #     }] if gesture_box else []
                # }
                gt = {
                    "changes": [{
                        "change_type":  "interaction",
                        "original_box": gesture_box,
                        "changed_box":  gesture_box,
                    }] if gesture_box else []
                }
                with open(pairs_dir / f"{pair_id}_gt.json", "w") as f:
                    json.dump(gt, f, indent=2)

                manifest_pairs.append({
                    "pair_id":     pair_id,
                    "change_type": "interaction",
                    "similarity":  float(sim),
                    "app":         app,
                })
                stats["valid_saved"] += 1

        processed_apps += 1
        if processed_apps % 500 == 0:
            print(f"  Processed {processed_apps} apps, "
                  f"{len(manifest_pairs)} pairs saved ...")

    # Write manifest (same format as change_dataset manifest)
    manifest = {
        "total_pairs":              len(manifest_pairs),
        "source":                   "rico_interaction_traces",
        "similarity_window":        [sim_low, sim_high],
        "change_type_distribution": dict(Counter(
            p["change_type"] for p in manifest_pairs
        )),
        "pairs": manifest_pairs
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DONE] Extracted {len(manifest_pairs)} valid pairs")
    print(f"[INFO] Stats: {dict(stats)}")
    print(f"[INFO] Output: {output_dir}")
    print(f"\n[NEXT] Run graph matcher:")
    print(f"  python scripts/graph_matcher.py ^")
    print(f"      --model      ./outputs/yolo_runs/rico_ui_v2/weights/best.pt ^")
    print(f"      --pairs_dir  {output_dir}/pairs ^")
    print(f"      --manifest   {output_dir}/manifest.json ^")
    print(f"      --output_dir ./outputs/results/trace_test ^")
    print(f"      --threshold  0.6")

    return len(manifest_pairs)


def pixel_diff_ground_truth(img_a, img_b, threshold=30, min_area=200):
    """
    Compute ground truth changed region from pixel difference.
    Returns single bounding box covering all changed pixels.
    More accurate than gesture coordinates as ground truth.
    """
    g1 = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    h, w = img_a.shape[:2]
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh >= min_area:
            boxes.append([x, y, x + bw, y + bh])

    if not boxes:
        return None

    # Merge all boxes into one encompassing box
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    # Reject if changed region covers more than 40% of screen
    # These are full-screen transitions, not targeted UI changes
    img_area = img_a.shape[0] * img_a.shape[1]
    box_area = (x2 - x1) * (y2 - y1)
    if box_area / img_area > 0.40:
        return None

    return [x1, y1, x2, y2]
# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract real UI change pairs from RICO interaction traces"
    )
    parser.add_argument("--traces_dir",   required=True,
                        help=r"Path to filtered_traces folder e.g. C:\dataset\rico_dir\traces\filtered_traces")
    parser.add_argument("--output_dir",   default="./outputs/trace_pairs")
    parser.add_argument("--explore_only", action="store_true",
                        help="Only report statistics, do not save files")
    parser.add_argument("--target_pairs", type=int, default=2000,
                        help="How many valid pairs to extract")
    parser.add_argument("--sim_low",      type=float, default=0.72,
                        help="Lower phash similarity bound (below = different screen)")
    parser.add_argument("--sim_high",     type=float, default=0.97,
                        help="Upper phash similarity bound (above = identical)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    if args.explore_only:
        explore_traces(
            traces_dir=args.traces_dir,
            sim_low=args.sim_low,
            sim_high=args.sim_high
        )
    else:
        # Always explore first then extract
        explore_traces(
            traces_dir=args.traces_dir,
            sim_low=args.sim_low,
            sim_high=args.sim_high
        )
        print("\n[INFO] Starting extraction ...")
        extract_pairs(
            traces_dir=args.traces_dir,
            output_dir=args.output_dir,
            target_pairs=args.target_pairs,
            sim_low=args.sim_low,
            sim_high=args.sim_high,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
