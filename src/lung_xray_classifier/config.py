from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATHS = {
    "train_normal": PROJECT_ROOT / "TRAIN_NORMAL",
    "train_pneumonia": PROJECT_ROOT / "TRAIN_PNEUMONIA",
    "test_normal": PROJECT_ROOT / "TEST_NORMAL",
    "test_pneumonia": PROJECT_ROOT / "TEST_PNEUMONIA",
}

MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "lung_xray_classifier.keras"


def ensure_dataset_available():
    missing = [name for name, path in DATASET_PATHS.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset directories: " + ", ".join(missing)
        )
