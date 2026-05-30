"""
step7_classifier.py
===================
Phase 2 - Week 5  |  Run on: LOCAL PC

What this does:
    Trains a Random Forest classifier to predict change type from
    graph matcher features. Uses the 1600 synthetic training pairs
    where ground truth change type is known exactly.

    Feature vector per detected change (6 features):
        phash_distance       - perceptual hash distance (0-1)
        colour_distance      - mean BGR colour distance (0-1)
        bbox_area_ratio      - new_area / old_area  (resize signal)
        bbox_centre_distance - normalised centre shift (relocate signal)
        class_match          - 1 if same YOLO class, 0 if different
        similarity_score     - raw graph matcher similarity score

    Output:
        outputs/classifier/classifier.pkl   - trained Random Forest
        outputs/classifier/features_train.json
        outputs/classifier/features_test.json
        outputs/classifier/evaluation.json

Usage:
    # Full pipeline: extract features, train, evaluate
    python scripts/step7_classifier.py ^
        --model      ./outputs/yolo_runs/rico_ui_v2/weights/best.pt ^
        --train_dir  ./outputs/change_dataset/train/pairs ^
        --train_manifest ./outputs/change_dataset/train/manifest.json ^
        --test_dir   ./outputs/change_dataset/test/pairs ^
        --test_manifest  ./outputs/change_dataset/test/manifest.json ^
        --output_dir ./outputs/classifier

    # Skip feature extraction if already done (reuse saved features)
    python scripts/step7_classifier.py ^
        --model      ./outputs/yolo_runs/rico_ui_v2/weights/best.pt ^
        --train_dir  ./outputs/change_dataset/train/pairs ^
        --train_manifest ./outputs/change_dataset/train/manifest.json ^
        --test_dir   ./outputs/change_dataset/test/pairs ^
        --test_manifest  ./outputs/change_dataset/test/manifest.json ^
        --output_dir ./outputs/classifier ^
        --skip_extraction
"""

from graph_matcher import match_graphs, phash_similarity, colour_similarity
from graph_builder import build_graph, run_yolo
import os
import cv2
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# Try XGBoost; fall back to sklearn GradientBoosting if not installed
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from ultralytics import YOLO

# Import from existing pipeline scripts
import sys
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# CONSTANTS
# =============================================================================

CHANGE_TYPES = ["color_change", "resize", "relocate", "remove", "add"]

# Similarity threshold for graph matching during feature extraction.
#
# This constant is the single source of truth shared by the training pipeline
# (step7), the severity scorer (step8), and the end-to-end demo (step9).
# ALL three must use the same value — mismatch creates a training/inference
# distribution shift where changed_nodes seen at training time differ from
# those seen at inference time.
#
# Why 0.85 (confirmed by experiment at 0.65):
#   The 0.85 boundary retains matched pairs with sim 0.65–0.85 in changed_nodes.
#   This range contains the discriminative examples the classifier needs:
#     - resize   : moderate area change, sim ~0.70–0.82 → area_ratio signal clear
#     - relocate : same appearance, different position, sim ~0.75–0.84 → centre_dist clear
#     - color_change: phash stable, colour shifts, sim ~0.68–0.80 → colour_dist + pixel_diff clear
#
#   Lowering to 0.65 (tested as classifier_v7) excluded this entire range,
#   leaving only extreme changes (sim < 0.65) whose feature vectors are
#   indistinguishable from genuine removes.  Result: color_change fell to
#   0 % accuracy, resize/relocate ~8–12 %, and severity F1 dropped from
#   0.829 → 0.793.  Reverted.
#
#   The classifier accuracy plateau at ~36 % is a data-quality ceiling
#   (synthetic RICO pairs vs real data), not a threshold problem.
#   Severity F1=0.829 (which uses hardcoded rules for add/remove) is the
#   correct metric — it is unaffected by this threshold value.
SIM_THRESHOLD = 0.85


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_bbox_area_ratio(box1, box2):
    """
    Ratio of new area to old area.
    >1 = grew, <1 = shrank, =1 = same size.
    Clamped to [0.1, 10] then log-normalised to [0, 1].
    """
    if box1 is None or box2 is None:
        return 1.0
    a1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    a2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    ratio = a2 / a1
    # Log scale: ratio=1 → 0.5, ratio=0.1 → ~0, ratio=10 → ~1
    log_ratio = np.log10(np.clip(ratio, 0.1, 10.0))
    return float((log_ratio + 1.0) / 2.0)


def compute_centre_distance(box1, box2, img_w, img_h):
    """
    Normalised Euclidean distance between bounding box centres.
    0 = same position, 1 = diagonal of image.
    """
    if box1 is None or box2 is None:
        return 0.0
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    dx = (cx2 - cx1) / max(img_w, 1)
    dy = (cy2 - cy1) / max(img_h, 1)
    dist = np.sqrt(dx**2 + dy**2)
    # Diagonal = sqrt(2) ≈ 1.414
    return float(np.clip(dist / 1.414, 0.0, 1.0))


def compute_region_pixel_diff(img1, img2, box1, box2):
    """
    Mean absolute pixel difference between two crops.
    Normalised to [0, 1]. Catches colour shifts phash misses.
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


def extract_features_from_pair(model, img1_path, img2_path, gt):
    """
    Run YOLO + graph builder + matcher on one pair.
    Returns a list of feature vectors, one per detected change node.

    Each feature vector: [phash_dist, colour_dist, area_ratio,
                          centre_dist, class_match, similarity]
    """
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        return []

    h, w = img1.shape[:2]

    # Build graphs
    det1 = run_yolo(model, img1)
    det2 = run_yolo(model, img2)
    G1 = build_graph(img1, det1, extract_ocr=False, extract_clip=True)
    G2 = build_graph(img2, det2, extract_ocr=False, extract_clip=True)

    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return []

    # Match graphs
    matches, changed_nodes, added_nodes, removed_nodes, sim_matrix = \
        match_graphs(G1, G2, SIM_THRESHOLD)

    feature_vectors = []

    # Changed nodes — matched but below threshold
    for n1_id, n2_id, sim in changed_nodes:
        d1 = G1.nodes[n1_id]
        d2 = G2.nodes[n2_id]

        phash_dist = 1.0 - phash_similarity(
            d1.get("phash"), d2.get("phash")
        )
        colour_dist = 1.0 - colour_similarity(
            d1.get("mean_colour"), d2.get("mean_colour")
        )
        area_ratio = compute_bbox_area_ratio(d1["bbox"], d2["bbox"])
        centre_dist = compute_centre_distance(d1["bbox"], d2["bbox"], w, h)
        class_match = 1.0 if d1["class_name"] == d2["class_name"] else 0.0

        pixel_diff = compute_region_pixel_diff(
            img1, img2, d1["bbox"], d2["bbox"])

        feature_vectors.append([
            phash_dist,
            colour_dist,
            area_ratio,
            centre_dist,
            class_match,
            float(sim),
            pixel_diff,   # 7th: raw pixel diff catches colour shifts phash misses
            1.0,          # 8th: is_matched=1 — this node was matched in both graphs
        ])

    # Removed nodes — unmatched in G2
    # is_matched=0 directly separates remove/add from matched changes
    for n1_id in removed_nodes:
        feature_vectors.append([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        # [phash=max, colour=max, area=0(gone), centre=0, class=0, sim=0,
        #  pixel_diff=1(max), is_matched=0]

    # Added nodes — unmatched in G1
    for n2_id in added_nodes:
        feature_vectors.append([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        # [phash=max, colour=max, area=1(appeared), centre=1, class=0, sim=0,
        #  pixel_diff=1(max), is_matched=0]

    # # Removed nodes — in G1 but no match in G2
    # for n1_id in removed_nodes:
    #     d1 = G1.nodes[n1_id]
    #     feature_vectors.append([
    #         1.0,   # max phash distance (no match)
    #         1.0,   # max colour distance
    #         0.0,   # area ratio → gone (log scale min)
    #         0.0,   # centre distance → unknown
    #         0.0,   # class match → no match
    #         0.0,   # similarity → 0
    #     ])

    # # Added nodes — in G2 but no match in G1
    # for n2_id in added_nodes:
    #     d2 = G2.nodes[n2_id]
    #     feature_vectors.append([
    #         1.0,   # max phash distance
    #         1.0,   # max colour distance
    #         1.0,   # area ratio → appeared (log scale max)
    #         1.0,   # centre distance → unknown
    #         0.0,   # class match → no match
    #         0.0,   # similarity → 0
    #     ])

    return feature_vectors


def extract_features(model, pairs_dir, manifest_path, output_dir, split_name):
    """
    Extract features for all pairs in a split.
    Saves to output_dir/features_{split_name}.json
    Returns X (features), y (labels).
    """
    pairs_dir = Path(pairs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    pairs = manifest["pairs"]
    print(
        f"[INFO] Extracting features from {len(pairs)} {split_name} pairs...")

    X, y = [], []
    skipped = 0

    for i, pair in enumerate(pairs):
        pid = pair["pair_id"]
        change_type = pair["change_type"]
        orig_path = pairs_dir / f"{pid}_original.jpg"
        chng_path = pairs_dir / f"{pid}_changed.jpg"
        gt_path = pairs_dir / f"{pid}_gt.json"

        if not orig_path.exists() or not chng_path.exists():
            skipped += 1
            continue

        with open(gt_path) as f:
            gt = json.load(f)

        features = extract_features_from_pair(model, orig_path, chng_path, gt)

        if not features:
            skipped += 1
            continue

        # All feature vectors from this pair get the same label
        for fv in features:
            X.append(fv)
            y.append(change_type)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pairs)} pairs processed "
                  f"({len(X)} feature vectors so far)...")

    print(f"[INFO] Done. {len(X)} feature vectors, {skipped} pairs skipped.")
    print(f"[INFO] Label distribution: {dict(Counter(y))}")

    # Save features
    features_data = {"X": X, "y": y, "split": split_name}
    out_path = output_dir / f"features_{split_name}.json"
    with open(out_path, "w") as f:
        json.dump(features_data, f)
    print(f"[INFO] Features saved: {out_path}")

    return X, y


# =============================================================================
# TRAINING
# =============================================================================

def train_classifier(X_train, y_train, output_dir):
    """
    Train classifier on 8-feature vectors.
    Uses XGBoost if installed, otherwise falls back to sklearn
    GradientBoostingClassifier (also outperforms RF on this task).
    Saves model to output_dir/classifier.pkl
    """
    output_dir = Path(output_dir)

    X = np.array(X_train)
    le = LabelEncoder()
    y = le.fit_transform(y_train)
    n_classes = len(le.classes_)

    if _XGBOOST_AVAILABLE:
        print(f"\n[INFO] Training XGBoost on {len(X_train)} samples...")
        print(f"[INFO] Classes: {list(le.classes_)}")
        # scale_pos_weight not used for multiclass; use sample_weight via
        # class balancing through eval_metric and tree boosting instead.
        # subsample + colsample_bytree add regularisation to avoid overfitting
        # on the dominant "remove" class.
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        # Compute per-sample weights to balance classes
        from collections import Counter
        counts = Counter(y.tolist())
        max_count = max(counts.values())
        sample_weights = np.array([max_count / counts[yi] for yi in y])
        clf.fit(X, y, sample_weight=sample_weights)
        classifier_name = "XGBoost"
    else:
        print(f"\n[INFO] XGBoost not found — using GradientBoostingClassifier")
        print(f"       (install xgboost for best results: pip install xgboost)")
        print(f"[INFO] Training GradientBoostingClassifier on {len(X_train)} samples...")
        print(f"[INFO] Classes: {list(le.classes_)}")
        # Train one-vs-rest via sklearn's native multiclass support
        from sklearn.multiclass import OneVsRestClassifier
        from collections import Counter
        counts = Counter(y.tolist())
        max_count = max(counts.values())
        sample_weights = np.array([max_count / counts[yi] for yi in y])
        clf = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        clf.fit(X, y, sample_weight=sample_weights)
        classifier_name = "GradientBoosting"

    # Feature importance
    feature_names = [
        "phash_distance",    # 1
        "colour_distance",   # 2
        "area_ratio",        # 3
        "centre_distance",   # 4
        "class_match",       # 5
        "similarity_score",  # 6
        "region_pixel_diff", # 7
        "is_matched",        # 8 — 1.0=matched pair, 0.0=removed/added
    ]
    importances = clf.feature_importances_
    print(f"\n[INFO] Feature importances ({classifier_name}):")
    for name, imp in sorted(zip(feature_names, importances),
                            key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:<25} {imp:.4f}  {bar}")

    # Save model + label encoder together
    model_data = {
        "classifier":    clf,
        "label_encoder": le,
        "classifier_name": classifier_name,
        "n_features":    8,
    }
    model_path = output_dir / "classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n[INFO] Classifier saved: {model_path}")

    return clf, le


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_classifier(clf, le, X_test, y_test, output_dir):
    """Evaluate classifier on test features and print results."""
    output_dir = Path(output_dir)

    X = np.array(X_test)
    y_true_enc = le.transform(y_test)
    y_pred_enc = clf.predict(X)

    y_true = le.inverse_transform(y_true_enc)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print("CHANGE TYPE CLASSIFIER — EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {acc:.4f}")
    print("\nPer-class results:")
    print(classification_report(y_true, y_pred,
                                target_names=sorted(set(y_test)),
                                zero_division=0))

    print("Confusion Matrix:")
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = f"{'':>15}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>15}" + \
            "".join(f"{cm[i,j]:>15}" for j in range(len(labels)))
        print(row)
    print("=" * 60)

    # Save evaluation
    report = classification_report(y_true, y_pred,
                                   target_names=sorted(set(y_test)),
                                   zero_division=0,
                                   output_dict=True)
    evaluation = {
        "accuracy":           acc,
        "classification_report": report,
        "confusion_matrix":   cm.tolist(),
        "labels":             labels,
        "n_test_samples":     len(y_test),
    }
    out_path = output_dir / "evaluation.json"
    with open(out_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"[INFO] Evaluation saved: {out_path}")

    return evaluation


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train change type classifier on graph matcher features"
    )
    parser.add_argument("--model",           required=True,
                        help="Path to best.pt")
    parser.add_argument("--train_dir",       required=True,
                        help="Training pairs directory")
    parser.add_argument("--train_manifest",  required=True,
                        help="Training manifest.json")
    parser.add_argument("--test_dir",        required=True,
                        help="Test pairs directory")
    parser.add_argument("--test_manifest",   required=True,
                        help="Test manifest.json")
    parser.add_argument("--output_dir",      default="./outputs/classifier")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip feature extraction, load saved features")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features_path = output_dir / "features_train.json"
    test_features_path = output_dir / "features_test.json"

    if args.skip_extraction and train_features_path.exists() and test_features_path.exists():
        print("[INFO] Loading saved features...")
        with open(train_features_path) as f:
            train_data = json.load(f)
        with open(test_features_path) as f:
            test_data = json.load(f)
        X_train, y_train = train_data["X"], train_data["y"]
        X_test,  y_test = test_data["X"],  test_data["y"]
    else:
        print("[INFO] Loading YOLO model...")
        model = YOLO(args.model)

        X_train, y_train = extract_features(
            model=model,
            pairs_dir=args.train_dir,
            manifest_path=args.train_manifest,
            output_dir=output_dir,
            split_name="train"
        )
        X_test, y_test = extract_features(
            model=model,
            pairs_dir=args.test_dir,
            manifest_path=args.test_manifest,
            output_dir=output_dir,
            split_name="test"
        )

    if not X_train:
        print("[ERROR] No training features extracted. Check paths and data.")
        return

    clf, le = train_classifier(X_train, y_train, output_dir)
    evaluate_classifier(clf, le, X_test, y_test, output_dir)

    print(f"\n[DONE] All outputs in: {output_dir}")
    print("[DONE] Next: run step8_severity.py")


if __name__ == "__main__":
    main()
