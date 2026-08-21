from __future__ import annotations

import argparse
import base64
from pathlib import Path

import pandas as pd


def encode_directory(directory: str | Path, label: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    directory = Path(directory)
    for image_path in sorted(directory.iterdir()):
        if image_path.is_file():
            encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            records.append({"image": encoded, "label": label})
    return records


def encode_dataset(train_normal: str | Path, train_pneumonia: str | Path, test_normal: str | Path, test_pneumonia: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = encode_directory(train_normal, 0) + encode_directory(train_pneumonia, 1)
    test_records = encode_directory(test_normal, 0) + encode_directory(test_pneumonia, 1)

    pd.DataFrame(train_records).sample(frac=1, random_state=42).to_csv(output_dir / "train_images.csv", index=False)
    pd.DataFrame(test_records).sample(frac=1, random_state=42).to_csv(output_dir / "test_images.csv", index=False)

    print(f"Saved training CSV to {output_dir / 'train_images.csv'}")
    print(f"Saved testing CSV to {output_dir / 'test_images.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode chest X-ray images into base64 CSV files.")
    parser.add_argument("--train-normal", type=str, default="TRAIN_NORMAL")
    parser.add_argument("--train-pneumonia", type=str, default="TRAIN_PNEUMONIA")
    parser.add_argument("--test-normal", type=str, default="TEST_NORMAL")
    parser.add_argument("--test-pneumonia", type=str, default="TEST_PNEUMONIA")
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()
    encode_dataset(
        args.train_normal,
        args.train_pneumonia,
        args.test_normal,
        args.test_pneumonia,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
