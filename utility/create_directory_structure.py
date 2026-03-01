from pathlib import Path
import argparse

__version__ = "1.0.0"

def create_structure(cwd: str | None = None):
    """
    Create the Visual Change Detection project structure.

    :param cwd: Base directory where project should be created.
                If None, uses current working directory.
    """

    base_path = Path(cwd).resolve() if cwd else Path.cwd()
    BASE_DIR = base_path / "visual_change_detection"

    # -----------------------------
    # Directory Structure
    # -----------------------------
    DIRECTORIES = [
        BASE_DIR,
        BASE_DIR / "scripts",
        BASE_DIR / "data" / "rico" / "combined",
        BASE_DIR / "data" / "rico" / "semantic_annotations",
        BASE_DIR / "outputs" / "rico_yolo_dataset" / "images" / "train",
        BASE_DIR / "outputs" / "rico_yolo_dataset" / "images" / "val",
        BASE_DIR / "outputs" / "rico_yolo_dataset" / "images" / "test",
        BASE_DIR / "outputs" / "rico_yolo_dataset" / "labels" / "train",
        BASE_DIR / "outputs" / "rico_yolo_dataset" / "labels" / "val",
        BASE_DIR / "outputs" / "rico_yolo_dataset" / "labels" / "test",
        # BASE_DIR / "outputs" / "change_dataset" / "train" / "pairs",
        # BASE_DIR / "outputs" / "change_dataset" / "test" / "pairs",
        # BASE_DIR / "outputs" / "yolo_runs" / "rico_ui_v1" / "weights",
        BASE_DIR / "outputs" / "results",
    ]

    # -----------------------------
    # Files
    # -----------------------------
    FILES = {
        BASE_DIR / "README.md": "# Visual Change Detection\n\nProject initialized.\n",
        BASE_DIR / "requirements.txt": "",
        BASE_DIR / "scripts" / "step1_explore_rico.py": "",
        BASE_DIR / "scripts" / "step2_generate_changes.py": "",
        BASE_DIR / "scripts" / "step3_train_yolo.py": "",
        # BASE_DIR / "scripts" / "step4_graph_builder.py": "",
        # BASE_DIR / "scripts" / "step5_graph_matching.py": "",
        # BASE_DIR / "scripts" / "step6_evaluate.py": "",
        # BASE_DIR / "scripts" / "step7_classifier.py": "",
        # BASE_DIR / "scripts" / "step8_regression_scorer.py": "",
        # BASE_DIR / "scripts" / "step9_demo.py": "",
        # BASE_DIR / "outputs" / "rico_yolo_dataset" / "dataset.yaml": "",
        # BASE_DIR / "outputs" / "rico_yolo_dataset" / "summary.json": "{}",
        # BASE_DIR / "outputs" / "change_dataset" / "train" / "manifest.json": "{}",
        # BASE_DIR / "outputs" / "change_dataset" / "test" / "manifest.json": "{}",
        # BASE_DIR / "outputs" / "yolo_runs" / "rico_ui_v1" / "weights" / "best.pt": "",
        # BASE_DIR / "outputs" / "yolo_runs" / "rico_ui_v1" / "weights" / "last.pt": "",
    }

    # Create directories
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

    # Create files
    for file_path, content in FILES.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.write_text(content)

    print(f"✅ Project created at: {BASE_DIR}")


# -----------------------------
# CLI Support
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Visual Change Detection project structure.")
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Base directory where project should be created (default: current working directory)"
    )

    args = parser.parse_args()
    create_structure(args.cwd)