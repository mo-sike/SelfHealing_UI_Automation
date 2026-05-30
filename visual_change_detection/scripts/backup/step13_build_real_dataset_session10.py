"""
step13_build_real_dataset.py
============================
Phase A — Real Dataset Assembly

Collects all annotated real pairs (step12 output) into a single dataset with
train/test splits, manifests, and per-type balance stats.

Optionally merges with the existing synthetic change_dataset_v3 for a
hybrid dataset. A hybrid dataset gives more training signal while the
real pairs provide an evaluation set that is directly comparable to the
base paper.

Output structure (mirrors existing pipeline format):
    real_change_dataset/
    ├── train/
    │   ├── pairs/
    │   │   ├── {pair_id}_original.jpg
    │   │   ├── {pair_id}_changed.jpg
    │   │   └── {pair_id}_gt.json
    │   └── manifest.json
    └── test/
        ├── pairs/
        │   ├── {pair_id}_original.jpg
        │   ├── {pair_id}_changed.jpg
        │   └── {pair_id}_gt.json
        └── manifest.json

Usage:
    # Build dataset from real pairs only
    python scripts/step13_build_real_dataset.py --real_root ./real_data --output_dir ./outputs/real_change_dataset

    # Build hybrid: real + synthetic (recommended for training)
    python scripts/step13_build_real_dataset.py --real_root ./real_data --synthetic_dir ./outputs/change_dataset_v3 --output_dir ./outputs/hybrid_change_dataset

    # Keep all real pairs in test set (evaluate on real only)
    python scripts/step13_build_real_dataset.py --real_root ./real_data --synthetic_dir ./outputs/change_dataset_v3 --output_dir ./outputs/hybrid_change_dataset --real_in_test_only

    # Print stats without building
    python scripts/step13_build_real_dataset.py --real_root ./real_data --stats_only
"""

import os
import sys
import json
import shutil
import random
import argparse
import itertools
from pathlib import Path
from collections import Counter, defaultdict


# =============================================================================
# PAIR DISCOVERY
# =============================================================================

def discover_real_pairs(real_root):
    """
    Walk real_root and find all annotated pairs.
    A valid pair has: *_original.jpg + *_changed.jpg + *_gt.json

    Returns list of dicts:
        {pair_id, original, changed, gt_json, app_name, change_type}
    """
    real_root = Path(real_root)
    pairs = []

    # Search in root/*/pairs/ and root/pairs/ and root/ itself
    search_dirs = set()
    for d in real_root.rglob("pairs"):
        if d.is_dir():
            search_dirs.add(d)
    search_dirs.add(real_root)          # also check root directly

    for sdir in search_dirs:
        for gt_path in sorted(sdir.glob("*_gt.json")):
            # Skip if this is a synthetic pair (source != "real")
            try:
                with open(gt_path) as f:
                    gt = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            pair_id  = gt["pair_id"]
            orig     = sdir / f"{pair_id}_original.jpg"
            chng     = sdir / f"{pair_id}_changed.jpg"

            if not orig.exists() or not chng.exists():
                continue

            # Accept only real-source pairs in this function
            if gt.get("source", "real") != "real":
                continue

            # Derive app name from pair_id (e.g. "antennapod_main" → "antennapod")
            parts    = pair_id.split("_")
            app_name = parts[0] if parts else "unknown"

            pairs.append({
                "pair_id":     pair_id,
                "original":    str(orig),
                "changed":     str(chng),
                "gt_json":     str(gt_path),
                "app_name":    app_name,
                "change_type": gt.get("change_type", "unknown"),
                "n_changes":   gt.get("n_changes", 0),
                "source":      "real",
            })

    return pairs


def discover_synthetic_pairs(synthetic_split_dir):
    """
    Discover pairs from an existing change_dataset split directory.
    Reads the manifest.json if present; otherwise scans for *_gt.json.
    """
    split_dir = Path(synthetic_split_dir)
    pairs_dir = split_dir / "pairs"
    if not pairs_dir.exists():
        pairs_dir = split_dir

    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        pair_ids = [p["pair_id"] for p in manifest.get("pairs", [])]
    else:
        pair_ids = [p.stem.replace("_gt", "")
                    for p in pairs_dir.glob("*_gt.json")]

    pairs = []
    for pid in pair_ids:
        orig = pairs_dir / f"{pid}_original.jpg"
        chng = pairs_dir / f"{pid}_changed.jpg"
        gt   = pairs_dir / f"{pid}_gt.json"
        if not (orig.exists() and chng.exists() and gt.exists()):
            continue
        try:
            with open(gt) as f:
                gt_data = json.load(f)
        except Exception:
            continue
        pairs.append({
            "pair_id":     pid,
            "original":    str(orig),
            "changed":     str(chng),
            "gt_json":     str(gt),
            "app_name":    "synthetic",
            "change_type": gt_data.get("change_type", "unknown"),
            "n_changes":   gt_data.get("n_changes", 0),
            "source":      "synthetic",
        })

    return pairs


# =============================================================================
# SPLITTING
# =============================================================================

def split_pairs(pairs, test_ratio=0.20, seed=42):
    """
    Stratified split by change_type to preserve class distribution.
    Returns (train_pairs, test_pairs).
    """
    by_type = defaultdict(list)
    for p in pairs:
        by_type[p["change_type"]].append(p)

    rng = random.Random(seed)
    train, test = [], []
    for ct, type_pairs in by_type.items():
        rng.shuffle(type_pairs)
        n_test = max(1, int(len(type_pairs) * test_ratio))
        test.extend(type_pairs[:n_test])
        train.extend(type_pairs[n_test:])

    return train, test


# =============================================================================
# DATASET BUILDER
# =============================================================================

def copy_pairs_to_split(pairs, split_dir):
    """
    Copy all pairs (original.jpg, changed.jpg, gt.json) into split_dir/pairs/.
    Returns updated pair dicts with new file paths.
    """
    pairs_out = split_dir / "pairs"
    pairs_out.mkdir(parents=True, exist_ok=True)

    updated = []
    for p in pairs:
        pid      = p["pair_id"]
        orig_dst = pairs_out / f"{pid}_original.jpg"
        chng_dst = pairs_out / f"{pid}_changed.jpg"
        gt_dst   = pairs_out / f"{pid}_gt.json"

        shutil.copy2(p["original"], orig_dst)
        shutil.copy2(p["changed"],  chng_dst)
        shutil.copy2(p["gt_json"],  gt_dst)

        up = dict(p)
        up["original"] = str(orig_dst)
        up["changed"]  = str(chng_dst)
        up["gt_json"]  = str(gt_dst)
        updated.append(up)

    return updated


def write_manifest(pairs, split_dir):
    """
    Write manifest.json compatible with the existing pipeline.
    Format: {"n_pairs": N, "pairs": [{"pair_id": ...}, ...]}
    """
    manifest = {
        "n_pairs": len(pairs),
        "pairs":   [{"pair_id": p["pair_id"]} for p in pairs],
    }
    path = split_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def write_split_summary(pairs, split_dir, split_name):
    """Write a human-readable summary of the split."""
    type_counts = Counter(p["change_type"] for p in pairs)
    source_counts = Counter(p["source"] for p in pairs)
    app_counts = Counter(p["app_name"] for p in pairs)

    summary = {
        "split":        split_name,
        "total_pairs":  len(pairs),
        "by_change_type": dict(type_counts),
        "by_source":    dict(source_counts),
        "by_app":       dict(app_counts),
    }
    path = split_dir / "split_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  [{split_name.upper()}] {len(pairs)} pairs")
    for ct in ["relocate", "resize", "color_change", "add", "remove"]:
        n = type_counts.get(ct, 0)
        print(f"    {ct:<14}: {n:4d}")
    print(f"    {'real':<14}: {source_counts.get('real', 0)}")
    print(f"    {'synthetic':<14}: {source_counts.get('synthetic', 0)}")
    return summary


def build_dataset(real_pairs, synthetic_train_pairs, synthetic_test_pairs,
                  output_dir, real_in_test_only=False, test_ratio=0.20):
    """
    Build the final dataset directory structure.

    If real_in_test_only:
        train = synthetic_train  +  synthetic_test (all synthetic)
        test  = all real pairs

    Otherwise:
        Split real pairs into train/test (stratified).
        train = real_train + synthetic_train
        test  = real_test  + synthetic_test
    """
    output_dir = Path(output_dir)
    train_dir  = output_dir / "train"
    test_dir   = output_dir / "test"

    print(f"\n[BUILD] Output: {output_dir}")

    if real_in_test_only:
        print("[BUILD] Mode: all real pairs → test set; synthetic → train")
        train_pairs_all = synthetic_train_pairs + synthetic_test_pairs
        test_pairs_all  = real_pairs
    else:
        real_train, real_test = split_pairs(real_pairs, test_ratio=test_ratio)
        train_pairs_all = real_train + synthetic_train_pairs
        test_pairs_all  = real_test  + synthetic_test_pairs

    print(f"[BUILD] Train: {len(train_pairs_all)}  Test: {len(test_pairs_all)}")

    # ── Copy files ────────────────────────────────────────────────────────────
    print("[BUILD] Copying train pairs ...")
    train_final = copy_pairs_to_split(train_pairs_all, train_dir)
    print("[BUILD] Copying test pairs ...")
    test_final  = copy_pairs_to_split(test_pairs_all,  test_dir)

    # ── Manifests ─────────────────────────────────────────────────────────────
    write_manifest(train_final, train_dir)
    write_manifest(test_final,  test_dir)

    # ── Summaries ─────────────────────────────────────────────────────────────
    train_summary = write_split_summary(train_final, train_dir, "train")
    test_summary  = write_split_summary(test_final,  test_dir,  "test")

    # ── Dataset-level summary ─────────────────────────────────────────────────
    dataset_summary = {
        "output_dir":      str(output_dir),
        "train_pairs":     len(train_final),
        "test_pairs":      len(test_final),
        "train_summary":   train_summary,
        "test_summary":    test_summary,
    }
    with open(output_dir / "dataset_summary.json", "w") as f:
        json.dump(dataset_summary, f, indent=2)

    print(f"\n[DONE] Dataset built at: {output_dir}")
    print(f"       Train manifest: {train_dir / 'manifest.json'}")
    print(f"       Test  manifest: {test_dir  / 'manifest.json'}")

    # ── Pipeline command hints ────────────────────────────────────────────────
    print("\n─── Next steps ─────────────────────────────────────────────────────────")
    train_manifest = train_dir / "manifest.json"
    test_manifest  = test_dir  / "manifest.json"
    train_pairs_dir = train_dir / "pairs"
    test_pairs_dir  = test_dir  / "pairs"
    out_path        = output_dir.name

    print(f"\n  # Retrain classifier on real (or hybrid) data:")
    print(f"  venv\\Scripts\\python scripts/step7_classifier.py "
          f"--model ./outputs/yolo_runs/rico_ui_v4/weights/best.pt "
          f"--train_dir {train_pairs_dir} "
          f"--train_manifest {train_manifest} "
          f"--test_dir {test_pairs_dir} "
          f"--test_manifest {test_manifest} "
          f"--output_dir ./outputs/classifier_real")

    print(f"\n  # Evaluate severity on real test pairs:")
    print(f"  venv\\Scripts\\python scripts/step8_severity.py "
          f"--model ./outputs/yolo_runs/rico_ui_v4/weights/best.pt "
          f"--classifier ./outputs/classifier_real/classifier.pkl "
          f"--test_dir {test_pairs_dir} "
          f"--test_manifest {test_manifest} "
          f"--output_dir ./outputs/severity_real")

    print(f"\n  # Run healer batch eval on real test pairs:")
    print(f"  venv\\Scripts\\python scripts/step10_healer.py "
          f"--model ./outputs/yolo_runs/rico_ui_v4/weights/best.pt "
          f"--classifier ./outputs/classifier_real/classifier.pkl "
          f"--pairs_dir {test_pairs_dir} "
          f"--manifest {test_manifest} "
          f"--output_dir ./outputs/healer_real")

    return dataset_summary


# =============================================================================
# STATS PRINTER
# =============================================================================

def print_discovery_stats(real_pairs):
    """Print a summary of discovered real pairs before building."""
    if not real_pairs:
        print("[STATS] No real pairs found.")
        return

    type_counts = Counter(p["change_type"] for p in real_pairs)
    app_counts  = Counter(p["app_name"]    for p in real_pairs)

    print(f"\n[STATS] Real pairs discovered: {len(real_pairs)}")
    print("\n  By change type:")
    for ct in ["relocate", "resize", "color_change", "add", "remove", "unknown"]:
        n = type_counts.get(ct, 0)
        if n:
            print(f"    {ct:<14}: {n}")

    print("\n  By app:")
    for app, n in sorted(app_counts.items(), key=lambda x: -x[1]):
        print(f"    {app:<20}: {n}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase A — Assemble real (and optional hybrid) dataset"
    )

    parser.add_argument(
        "--real_root", required=True,
        help="Root directory of collected real pairs (from step11 → step12)"
    )
    parser.add_argument(
        "--synthetic_dir", default=None,
        help="Existing change_dataset_v3 root (for hybrid mode). "
             "Expects train/ and test/ subdirectories with manifest.json."
    )
    parser.add_argument(
        "--output_dir", default="./outputs/real_change_dataset",
        help="Where to build the new dataset (default: ./outputs/real_change_dataset)"
    )
    parser.add_argument(
        "--real_in_test_only", action="store_true",
        help="Put ALL real pairs in the test split; use synthetic for train. "
             "Best for maximising comparability to base paper evaluation."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.20,
        help="Fraction of real pairs to use as test (default: 0.20). "
             "Ignored if --real_in_test_only."
    )
    parser.add_argument(
        "--stats_only", action="store_true",
        help="Print discovery stats and exit without building the dataset."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified split (default: 42)"
    )

    args = parser.parse_args()

    # ── Discover real pairs ───────────────────────────────────────────────────
    print(f"[INFO] Scanning for real pairs under: {args.real_root}")
    real_pairs = discover_real_pairs(args.real_root)
    print_discovery_stats(real_pairs)

    if args.stats_only:
        return

    if not real_pairs:
        print("[ERROR] No real pairs found. "
              "Run step11_collect_real_pairs.py and step12_annotate_ui_diff.py first.")
        sys.exit(1)

    # ── Discover synthetic pairs (optional) ──────────────────────────────────
    syn_train, syn_test = [], []
    if args.synthetic_dir:
        syn_dir = Path(args.synthetic_dir)
        print(f"\n[INFO] Loading synthetic pairs from: {syn_dir}")
        syn_train = discover_synthetic_pairs(syn_dir / "train")
        syn_test  = discover_synthetic_pairs(syn_dir / "test")
        print(f"[INFO] Synthetic: {len(syn_train)} train + {len(syn_test)} test")

    # ── Build ─────────────────────────────────────────────────────────────────
    build_dataset(
        real_pairs          = real_pairs,
        synthetic_train_pairs = syn_train,
        synthetic_test_pairs  = syn_test,
        output_dir          = args.output_dir,
        real_in_test_only   = args.real_in_test_only,
        test_ratio          = args.test_ratio,
    )


if __name__ == "__main__":
    main()
