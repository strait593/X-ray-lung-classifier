from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Sequential, layers


def build_model(input_shape: tuple[int, int, int] = (224, 224, 3)) -> Sequential:
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
