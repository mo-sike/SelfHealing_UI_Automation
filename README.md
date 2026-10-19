# Visual Self-Healing UI Test Automation

**Screenshot-only change detection and locator healing for mobile UI tests. It needs no DOM, no XPath and no network access.**

Coordinate-based UI tests break whenever an app update moves, resizes, recolours, adds or removes a control. This project takes two screenshots, one **before** and one **after** an update, and works out:

1. **What changed:** which UI elements were removed, added, moved, resized or recoloured
2. **Whether it matters:** a severity score and a configurable **PASS / FAIL** regression verdict
3. **How to fix the test:** new coordinates for each broken locator, with a confidence-based verdict of **AUTO-HEAL / REVIEW / CANNOT-HEAL**

It works only from pixels, so it applies to any platform you can screenshot, including Android, iOS, embedded HMIs and infotainment systems.

---

## Pipeline

```
 old screenshot ─┐                                        ┌─► changes + type
                 ├─► YOLOv8 ─► KNN spatial graph ─► graph ─┼─► severity ─► PASS / FAIL
 new screenshot ─┘   (UI elements)  (per screen)   matcher └─► healer ─► new coordinates
                                                                 ▲        + AUTO-HEAL / REVIEW / CANNOT-HEAL
                                             broken locators ────┘
```

| Stage | Script | What it does |
|---|---|---|
| 1. Dataset | `explore_rico.py` | Curates a class-balanced (`--stratify`) subset of the [RICO](http://www.interactionmining.org/rico) dataset (66k Android screens), converts the annotations to YOLO format and makes an 80/10/10 split |
| 2. Detection model | — | YOLOv8 trained on 10 UI classes: `button`, `text`, `input`, `image`, `icon`, `checkbox`, `toolbar`, `list_item`, `card`, `menu` |
| 3. Synthetic changes | `generate_changes.py` | Makes before/after pairs with ground truth for 5 change types: **remove, add, relocate, resize, colour change** |
| 4. Graph building | `graph_builder.py` | Turns each detection into a node (bbox, perceptual hash, mean colour, OCR text, **CLIP ViT-B/32** embedding) and connects each node to its 5 nearest neighbours |
| 5. Graph matching | `graph_matcher.py` | Recursive node similarity combining visual, colour, text, class and neighbour-structure similarity (after Moradi et al.); flags unmatched or low-similarity nodes as changed |
| 6. Change classification | `step7_classifier.py` | Tree-ensemble classifier (XGBoost, or scikit-learn if XGBoost isn't installed) that predicts the change type from the matcher's features |
| 7. Severity | `step8_severity.py` | Scores each change from 0 to 1 and returns PASS / FAIL against a configurable threshold, so cosmetic changes can pass while structural regressions fail |
| 8. Demo | `step9_demo.py` | End-to-end detection, classification and severity, on one pair or a batch |
| 9. Self-healing | `step10_healer.py` | Maps each broken locator to a node, finds that node on the new screen and estimates confidence from the entropy of the classifier's probabilities, then triages it |
| Batch runner | `run_detection.py` | Runs detection over a whole test set and writes `all_results.json` |

### Healing triage

| Verdict | Meaning |
|---|---|
| **AUTO-HEAL** | High confidence (≥ `--auto_heal_conf`, default 0.80). The locator is updated automatically. |
| **REVIEW** | A candidate was found but confidence is low, so a human checks it. |
| **CANNOT-HEAL** | The element is gone (for example removed) or has no reliable match. The test needs a manual fix. |

Low-confidence fixes are never applied silently, so a wrong heal can't quietly break a test.

---

## Project structure

```
SelfHealing_UI_Automation/
├── utility/
│   └── create_directory_structure.py
└── visual_change_detection/
    └── scripts/
        ├── explore_rico.py        # 1. RICO → YOLO dataset
        ├── generate_changes.py    # 3. synthetic before/after pairs
        ├── graph_builder.py       # 4. detections → KNN graph (+ CLIP, OCR)
        ├── graph_matcher.py       # 5. recursive graph similarity
        ├── step7_classifier.py    # 6. change-type classifier
        ├── step8_severity.py      # 7. severity + PASS/FAIL
        ├── step9_demo.py          # 8. end-to-end demo
        ├── step10_healer.py       # 9. locator self-healing
        └── run_detection.py       #    batch detection runner
```

The dataset (`data/`), model weights and training runs (`runs/`, `outputs/`) are not in the repo.

---

## Setup

```bash
cd visual_change_detection
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on Linux/macOS)

pip install ultralytics opencv-python numpy scipy networkx scikit-learn xgboost imagehash pillow pytesseract torch
pip install git+https://github.com/openai/CLIP.git
```

- **OCR:** install [Tesseract](https://github.com/tesseract-ocr/tesseract). The binary path is set in `graph_builder.py`. If Tesseract is missing, OCR is turned off.
- **CLIP** is optional. Without it, CLIP embeddings are skipped and a warning is printed.
- **Dataset:** download RICO separately and point `--rico_dir` at it.

## Usage

Run these from `visual_change_detection/`.

```bash
# 1. Build the YOLO dataset from RICO
python scripts/explore_rico.py --rico_dir ./data/rico --output_dir ./outputs/rico_yolo_dataset --stratify

# 2. Generate synthetic change pairs
python scripts/generate_changes.py --images_dir ./outputs/rico_yolo_dataset/images/train --labels_dir ./outputs/rico_yolo_dataset/labels/train --output_dir ./outputs/change_dataset/train --target_pairs 1600

# 3. Train the change-type classifier
python scripts/step7_classifier.py --model <best.pt> --train_dir <train/pairs> --train_manifest <train/manifest.json> --test_dir <test/pairs> --test_manifest <test/manifest.json> --output_dir ./outputs/classifier

# 4. Detect changes and get a PASS/FAIL verdict for one screenshot pair
python scripts/step9_demo.py --model <best.pt> --classifier ./outputs/classifier/classifier.pkl --image1 before.jpg --image2 after.jpg --fail_threshold 0.5 --visualise

# 5. Heal broken locators
python scripts/step10_healer.py --model <best.pt> --classifier ./outputs/classifier/classifier.pkl --image1 before.jpg --image2 after.jpg --locators locators.json --visualise
```

Locators file format:

```json
{
  "test_name": "Login screen regression",
  "locators": [
    { "name": "Login button", "x": 540, "y": 1620 },
    { "name": "Username field", "x": 540, "y": 900 }
  ]
}
```

Each script accepts `--help` for the full list of options. Most also have a batch mode (`--pairs_dir` and `--manifest`) for evaluating on a test set.

---

## Evaluation metrics

`step10_healer.py` batch mode reports:

| Metric | Definition |
|---|---|
| Heal accuracy | correct auto-heals ÷ total auto-heals |
| False heal rate | wrong auto-heals ÷ total auto-heals |
| Review reduction | 1 − (REVIEW count ÷ total changes) |
| Heal coverage | (AUTO-HEAL + REVIEW) ÷ total |

## Status

This is an active research project. Phase 1 (detection, graph matching and change classification) and Phase 2 (CLIP similarity, self-healing and confidence triage) are implemented. Tuning on synthetic and real app-update screenshots is ongoing.

## Tech stack

Python · YOLOv8 (Ultralytics) · CLIP · PyTorch · XGBoost / scikit-learn · OpenCV · NetworkX · Tesseract OCR · imagehash

## Author

**Mohammed Saqlain**: [GitHub](https://github.com/mo-sike) · [LinkedIn](https://www.linkedin.com/in/mo-sike)
