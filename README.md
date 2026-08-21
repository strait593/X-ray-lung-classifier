# Lung X-ray classifier

This project trains and runs a binary classifier that distinguishes between normal chest X-rays and pneumonia-affected X-rays.

## Project structure

- `TRAIN_NORMAL/` and `TRAIN_PNEUMONIA/` contain training images.
- `TEST_NORMAL/` and `TEST_PNEUMONIA/` contain validation/test images.
- `src/lung_xray_classifier/` contains the reusable Python package.
- `models/` stores trained model artifacts.
- `notebooks/` holds exploratory work.
- `single prediction/` contains sample images for manual testing.

## Setup

```bash
python -m pip install -r requirements.txt
```

Optional local install:

```bash
python -m pip install -e .
```

## Train a model

```bash
python -m src.lung_xray_classifier.train --epochs 20 --batch-size 32
```

You can also point the output file somewhere else:

```bash
python -m src.lung_xray_classifier.train --output models/lung_xray_classifier.keras
```

## Launch the dashboard

```bash
streamlit run dashboard.py
```

## Encoding raw images to CSV

```bash
python -m src.lung_xray_classifier.data_encoding --output-dir data
```

## Notes

- The project uses a simple CNN built in TensorFlow/Keras.
- Keep the dataset directories at the project root so the default configuration works without adjusting paths.
- The repo is intentionally set up to stay compatible with the original top-level scripts while also supporting a cleaner package structure.
