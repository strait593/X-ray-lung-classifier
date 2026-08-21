from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import tensorflow as tf


def list_image_files(directory: str | Path) -> list[str]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    files = sorted(str(path) for path in directory.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"No image files found in dataset directory: {directory}")
    return files


def make_class_dataset(directory: str | Path, label: int, image_size: tuple[int, int] = (224, 224)) -> tf.data.Dataset:
    image_paths = list_image_files(directory)
    labels = [label] * len(image_paths)

    def load_image(path: str, label_value: int) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_size)
        return image, tf.cast(label_value, tf.float32)

    return tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )


def prepare_dataset(
    data_normal: tf.data.Dataset,
    data_pneumonia: tf.data.Dataset,
    batch_size: int = 32,
    *,
    training: bool,
) -> tf.data.Dataset:
    dataset = data_normal.concatenate(data_pneumonia)
    if training:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
