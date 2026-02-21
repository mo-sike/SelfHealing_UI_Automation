SelfHealing_UI_Automation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ ├── raw/
│ │ ├── screenshots_A/
│ │ └── screenshots_B/
│ ├── annotations/
│ └── processed/
│
├── models/
│ ├── yolo/
│ │ └── weights/
│ └── saved_models/
│
├── src/
│ ├── detection/
│ │ ├── detect_ui.py
│ │ └── yolo_utils.py
│ │
│ ├── graph/
│ │ ├── graph_builder.py
│ │ └── graph_utils.py
│ │
│ ├── matching/
│ │ ├── similarity.py
│ │ ├── matcher.py
│ │ └── recursive_match.py
│ │
│ ├── evaluation/
│ │ ├── metrics.py
│ │ └── iou.py
│ │
│ ├── visualization/
│ │ ├── heatmap.py
│ │ └── draw_boxes.py
│ │
│ └── main.py
│
├── results/
│ ├── detections/
│ ├── graphs/
│ ├── matches/
│ └── heatmaps/
│
└── notebooks/
└── experimentation.ipynb
