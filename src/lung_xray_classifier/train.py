from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from lung_xray_classifier.config import DATASET_PATHS, MODEL_OUTPUT_PATH, ensure_dataset_available
    from lung_xray_classifier.data import make_class_dataset, prepare_dataset
    from lung_xray_classifier.model import build_model
else:
    from .config import DATASET_PATHS, MODEL_OUTPUT_PATH, ensure_dataset_available
    from .data import make_class_dataset, prepare_dataset
    from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lung X-ray classifier.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default=str(MODEL_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dataset_available()

    train_data_normal = make_class_dataset(DATASET_PATHS["train_normal"], 0)
    train_data_pneumonia = make_class_dataset(DATASET_PATHS["train_pneumonia"], 1)
    test_data_normal = make_class_dataset(DATASET_PATHS["test_normal"], 0)
    test_data_pneumonia = make_class_dataset(DATASET_PATHS["test_pneumonia"], 1)

    train_dataset = prepare_dataset(train_data_normal, train_data_pneumonia, batch_size=args.batch_size, training=True)
    test_dataset = prepare_dataset(test_data_normal, test_data_pneumonia, batch_size=args.batch_size, training=False)

    model = build_model()
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=args.epochs,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    print(f"Training complete. Model saved to {output_path}")
    print(f"Best validation accuracy: {max(history.history.get('val_accuracy', [0])):.4f}")


if __name__ == "__main__":
    main()
