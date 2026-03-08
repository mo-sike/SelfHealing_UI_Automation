"""
step4_graph_builder.py
======================
Phase 1 - Week 3, Days 15-17  |  Run on: LOCAL PC

What this does:
    Takes YOLO detections from a screenshot and builds a spatial graph.
    Each detected control becomes a node.
    Edges connect each node to its K nearest neighbours (KNN graph).

    Node features stored:
        - class_name      : YOLO class label
        - bbox            : [x1, y1, x2, y2] pixel coordinates
        - centre          : (cx, cy)
        - crop            : image patch (for visual hash + colour)
        - phash           : perceptual hash of crop
        - mean_colour     : mean BGR colour of crop
        - ocr_text        : OCR text extracted from crop (if any)
        - confidence      : YOLO detection confidence

Usage:
    # Build graph for a single image
    python scripts/step4_graph_builder.py \
        --model   ./outputs/yolo_runs/rico_ui_v2/weights/best.pt \
        --image   ./outputs/rico_yolo_dataset/images/test/10037.jpg \
        --output  ./outputs/graphs/10037.gpickle \
        --visualise

    # Batch build graphs for all test images
    python scripts/step4_graph_builder.py \
        --model     ./outputs/yolo_runs/rico_ui_v2/weights/best.pt \
        --image_dir ./outputs/rico_yolo_dataset/images/test \
        --output_dir ./outputs/graphs/test
"""

import os
import cv2
import pickle
import argparse
import numpy as np
import networkx as nx
from pathlib import Path

# ── Optional imports with clear error messages ────────────────────────────────
try:
    import imagehash
    from PIL import Image
    IMAGEHASH_OK = True
except ImportError:
    IMAGEHASH_OK = False
    print("[WARN] imagehash not installed — perceptual hash disabled")
    print("       pip install imagehash Pillow")

try:
    import pytesseract
    # Windows default path — adjust if different
    pytesseract.pytesseract.tesseract_cmd = (
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    )
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False
    print("[WARN] pytesseract not installed — OCR disabled")
    print("       pip install pytesseract  (also install Tesseract binary)")

from ultralytics import YOLO


# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_NAMES = [
    "button", "text", "input", "image", "icon",
    "checkbox", "toolbar", "list_item", "card", "menu"
]

# KNN graph parameter — each node connects to K nearest neighbours
K_NEIGHBOURS = 5

# Minimum confidence threshold for YOLO detections
CONF_THRESHOLD = 0.25

# Minimum crop size to attempt OCR (pixels)
OCR_MIN_SIZE = 30


# =============================================================================
# NODE FEATURE EXTRACTION
# =============================================================================

def extract_phash(crop_bgr):
    """Compute perceptual hash of an image crop."""
    if not IMAGEHASH_OK or crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        return str(imagehash.phash(pil_img))
    except Exception:
        return None


def extract_ocr_text(crop_bgr):
    """Extract text from image crop using Tesseract OCR."""
    if not TESSERACT_OK or crop_bgr is None or crop_bgr.size == 0:
        return ""
    h, w = crop_bgr.shape[:2]
    if h < OCR_MIN_SIZE or w < OCR_MIN_SIZE:
        return ""
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(
            gray,
            config='--psm 6 --oem 3'
        ).strip()
        return text
    except Exception:
        return ""


def extract_mean_colour(crop_bgr):
    """Compute mean BGR colour of crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return [0, 0, 0]
    mean = cv2.mean(crop_bgr)[:3]
    return [float(mean[0]), float(mean[1]), float(mean[2])]


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_graph(image, detections, extract_ocr=True):
    """
    Build a spatial KNN graph from YOLO detections.

    Args:
        image      : BGR numpy array
        detections : list of dicts with keys: bbox, class_id, confidence
        extract_ocr: whether to run OCR on each node (slower but richer)

    Returns:
        G : networkx.Graph with node and edge attributes
    """
    G = nx.Graph()
    h, w = image.shape[:2]

    if not detections:
        return G

    # ── Add nodes ─────────────────────────────────────────────
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Extract crop
        crop = image[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

        # Node features
        phash      = extract_phash(crop)
        mean_colour = extract_mean_colour(crop)
        ocr_text   = extract_ocr_text(crop) if extract_ocr else ""

        G.add_node(i, **{
            "class_id":    det["class_id"],
            "class_name":  CLASS_NAMES[det["class_id"]],
            "bbox":        [x1, y1, x2, y2],
            "centre":      (cx, cy),
            "confidence":  det["confidence"],
            "phash":       phash,
            "mean_colour": mean_colour,
            "ocr_text":    ocr_text,
        })

    # ── Add KNN edges ──────────────────────────────────────────
    nodes      = list(G.nodes(data=True))
    n          = len(nodes)
    k_actual   = min(K_NEIGHBOURS, n - 1)

    if k_actual < 1:
        return G

    centres = np.array([data["centre"] for _, data in nodes])

    for i in range(n):
        dists = np.sqrt(
            np.sum((centres - centres[i]) ** 2, axis=1)
        )
        dists[i] = np.inf  # exclude self
        nearest = np.argsort(dists)[:k_actual]

        for j in nearest:
            if not G.has_edge(i, j):
                G.add_edge(i, j, distance=float(dists[j]))

    return G


def run_yolo(model, image, conf=CONF_THRESHOLD):
    """Run YOLO on an image and return detections list."""
    results     = model(image, conf=conf, verbose=False)
    detections  = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "class_id":   int(box.cls[0]),
                "confidence": float(box.conf[0])
            })

    return detections


# =============================================================================
# VISUALISATION
# =============================================================================

def visualise_graph(image, G, output_path):
    """Draw detections and graph edges on image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    CLASS_COLOURS = {
        "button":    (0,   255,  0),
        "text":      (255, 200,  0),
        "input":     (0,   200, 255),
        "image":     (200,   0, 255),
        "icon":      (255, 100,  0),
        "checkbox":  (0,   255, 200),
        "toolbar":   (0,    50, 255),
        "list_item": (180, 255,  0),
        "card":      (255,   0, 100),
        "menu":      (100,   0, 255),
    }

    # Draw edges
    for u, v, data in G.edges(data=True):
        cu = tuple(map(int, G.nodes[u]["centre"]))
        cv_ = tuple(map(int, G.nodes[v]["centre"]))
        cv2.line(vis, cu, cv_, (180, 180, 180), 1, cv2.LINE_AA)

    # Draw nodes
    for node_id, data in G.nodes(data=True):
        x1, y1, x2, y2 = data["bbox"]
        colour = CLASS_COLOURS.get(data["class_name"], (255, 255, 255))
        cx, cy = int(data["centre"][0]), int(data["centre"][1])

        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        cv2.circle(vis, (cx, cy), 4, colour, -1)

        label = f"{data['class_name']} {data['confidence']:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    # Stats overlay
    cv2.putText(vis,
        f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), vis)
    print(f"[INFO] Graph visualisation saved: {output_path}")


# =============================================================================
# SINGLE IMAGE
# =============================================================================

def process_single(model_path, image_path, output_path,
                   visualise=False, extract_ocr=True):
    """Build and save graph for a single image."""
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] Processing: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None

    detections = run_yolo(model, image)
    print(f"[INFO] YOLO detections: {len(detections)}")

    G = build_graph(image, detections, extract_ocr=extract_ocr)
    print(f"[INFO] Graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # Save graph
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"[INFO] Graph saved: {output_path}")

    # Visualise
    if visualise:
        vis_path = output_path.parent / f"{output_path.stem}_vis.jpg"
        visualise_graph(image, G, vis_path)

    return G


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_batch(model_path, image_dir, output_dir, extract_ocr=False):
    """
    Build graphs for all images in a directory.

    Note: extract_ocr=False by default for batch — OCR is slow (~2s/image).
    Enable for small sets or when text similarity is important.
    """
    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(image_dir.glob("*.jpg"))
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] Processing {len(all_images)} images → {output_dir}")

    stats = {"success": 0, "failed": 0, "empty": 0}

    for i, img_path in enumerate(all_images):
        out_path = output_dir / f"{img_path.stem}.gpickle"
        if out_path.exists():
            continue  # skip already processed

        image = cv2.imread(str(img_path))
        if image is None:
            stats["failed"] += 1
            continue

        detections = run_yolo(model, image)
        if not detections:
            stats["empty"] += 1

        G = build_graph(image, detections, extract_ocr=extract_ocr)

        with open(out_path, 'wb') as f:
            pickle.dump(G, f)

        stats["success"] += 1
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(all_images)} ...")

    print(f"\n[DONE] Batch complete: {stats}")
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build spatial KNN graphs from YOLO detections"
    )
    parser.add_argument("--model",       required=True,
                        help="Path to best.pt")
    parser.add_argument("--image",       default=None,
                        help="Single image path (single mode)")
    parser.add_argument("--image_dir",   default=None,
                        help="Directory of images (batch mode)")
    parser.add_argument("--output",      default=None,
                        help="Output .gpickle path (single mode)")
    parser.add_argument("--output_dir",  default=None,
                        help="Output directory (batch mode)")
    parser.add_argument("--visualise",   action="store_true",
                        help="Save visualisation image")
    parser.add_argument("--no_ocr",      action="store_true",
                        help="Disable OCR (faster)")
    args = parser.parse_args()

    extract_ocr = not args.no_ocr

    if args.image:
        output = args.output or f"./outputs/graphs/{Path(args.image).stem}.gpickle"
        process_single(
            model_path=args.model,
            image_path=args.image,
            output_path=output,
            visualise=args.visualise,
            extract_ocr=extract_ocr
        )
    elif args.image_dir:
        output_dir = args.output_dir or "./outputs/graphs"
        process_batch(
            model_path=args.model,
            image_dir=args.image_dir,
            output_dir=output_dir,
            extract_ocr=extract_ocr
        )
    else:
        print("[ERROR] Provide either --image or --image_dir")
        parser.print_help()


if __name__ == "__main__":
    main()
