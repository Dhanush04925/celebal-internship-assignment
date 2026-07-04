# MNIST Denoising Autoencoder

A modular, production-style TensorFlow/Keras implementation of a **Convolutional
Denoising Autoencoder** trained on the MNIST handwritten-digit dataset. The
model learns to reconstruct clean digit images from Gaussian-noise-corrupted
inputs.

> Project layout takes inspiration from [NvsYashwanth/MNIST-Autoecncoder](https://github.com/NvsYashwanth/MNIST-Autoecncoder)
> (organization only — all source code here is an original implementation).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Training](#training)
- [Prediction](#prediction)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)

---

## Project Overview

Denoising autoencoders learn a compressed ("latent") representation of an
input and use it to reconstruct a clean version of a corrupted signal. This
project builds one from scratch for the MNIST dataset:

1. Clean MNIST digits are normalized and reshaped to `(28, 28, 1)`.
2. Gaussian noise is synthetically injected to create noisy counterparts.
3. A convolutional encoder-decoder network is trained to map
   **noisy → clean**.
4. Reconstruction quality is measured with MSE, PSNR, and SSIM.

## Objectives

- ✅ Build a clean, modular, testable TensorFlow/Keras codebase (no notebook-only spaghetti).
- ✅ Implement a Conv2D + BatchNorm + MaxPooling/UpSampling autoencoder using the Functional API.
- ✅ Train with best-practice callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard).
- ✅ Provide reproducible CLI entry points (`train.py`, `predict.py`) alongside an exploratory notebook.
- ✅ Quantify denoising quality with standard image-similarity metrics.

## Folder Structure

```
MNIST-Denoising-Autoencoder/
│
├── assets/                        # Static assets (e.g. README images)
│
├── data/
│   └── MNIST/
│       ├── raw/                   # train.csv / test.csv (Kaggle MNIST CSV)
│       └── processed/             # cached x_train.npy, y_train.npy, x_test.npy, y_test.npy
│
├── models/
│   ├── autoencoder.py             # build_autoencoder() — Functional API architecture
│   ├── trainer.py                 # Trainer — compile, callbacks, fit, save
│   ├── predictor.py                # Predictor — load model, denoise, evaluate
│   └── __init__.py
│
├── utils/
│   ├── data_loader.py             # CSV loading + processed-array caching
│   ├── preprocessing.py           # normalize / reshape / Gaussian noise
│   ├── visualization.py           # comparison plots, training-history plots
│   ├── metrics.py                 # MSE / PSNR / SSIM
│   ├── config.py                  # Config dataclass (paths + hyperparameters)
│   ├── logger.py                  # console + rotating-file logger
│   └── __init__.py
│
├── outputs/
│   ├── checkpoints/                # saved .keras model(s)
│   ├── figures/                    # saved comparison / history plots
│   ├── logs/                       # TensorBoard + rotating log file
│   └── history/                    # training_history.csv
│
├── notebooks/
│   └── Denoising_Autoencoder.ipynb # exploratory, end-to-end notebook
│
├── train.py                        # CLI: full training pipeline
├── predict.py                       # CLI: inference + evaluation
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Installation

```bash
git clone <this-repo-url>
cd MNIST-Denoising-Autoencoder

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python 3.9+, TensorFlow 2.13+, NumPy, Pandas, Matplotlib,
scikit-image, scikit-learn.

## Dataset

This project uses the Kaggle **MNIST Dataset**:
https://www.kaggle.com/datasets/awsaf49/mnist-dataset

The loader auto-detects **either** of the two common distributions of this
dataset — use whichever one you downloaded, no conversion needed.

**Option A — CSV format**

Place `train.csv` and `test.csv` at:

```
data/MNIST/raw/train.csv
data/MNIST/raw/test.csv
```

Column `0` is the integer digit label (0–9); columns `1`–`784` are the
flattened 28×28 grayscale pixel intensities (0–255).

**Option B — Image-folder format**

Place your `training/` and `testing/` folders (each containing subfolders
`0`–`9` of `.png`/`.jpg`/`.jpeg`/`.bmp` images) at:

```
data/MNIST/raw/training/0/  ...  data/MNIST/raw/training/9/
data/MNIST/raw/testing/0/   ...  data/MNIST/raw/testing/9/
```

`load_raw_dataset()` checks for the CSV files first, then falls back to the
image-folder layout automatically.

On the first run, `train.py` will normalize, reshape, and cache processed
arrays as `.npy` files under `data/MNIST/processed/` for faster subsequent
runs (add `--force-reprocess` to rebuild them from raw data).

## Architecture

A symmetric convolutional encoder-decoder built with the Keras **Functional API**:

| Stage    | Layers                                                          | Output shape   |
|----------|------------------------------------------------------------------|----------------|
| Input    | —                                                                 | `28×28×1`      |
| Encoder  | `Conv2D(32)` → `BatchNorm` → `ReLU` → `MaxPooling2D`              | `14×14×32`     |
| Encoder  | `Conv2D(64)` → `BatchNorm` → `ReLU` → `MaxPooling2D`              | `7×7×64`       |
| Latent   | `Conv2D(64, relu)`                                                | `7×7×64`       |
| Decoder  | `Conv2D(64)` → `BatchNorm` → `ReLU` → `UpSampling2D`              | `14×14×64`     |
| Decoder  | `Conv2D(32)` → `BatchNorm` → `ReLU` → `UpSampling2D`              | `28×28×32`     |
| Output   | `Conv2D(1, sigmoid)`                                              | `28×28×1`      |

Total parameters: **~112K** — small enough to train quickly on CPU while still
learning a meaningful denoising mapping.

## Training

```bash
python train.py
```

Optional CLI overrides:

```bash
python train.py --epochs 30 --batch-size 64 --noise-factor 0.3 --loss mse --seed 7
python train.py --force-reprocess     # rebuild cached arrays from raw CSVs
```

The training pipeline:

1. Loads raw CSVs (or cached `.npy` arrays if present).
2. Normalizes → reshapes → injects Gaussian noise (`noise_factor`, default `0.4`).
3. Compiles the autoencoder with **Adam** + **binary cross-entropy** (MSE tracked as a metric).
4. Trains for up to `epochs` (default `50`) with:
   - `EarlyStopping` (`patience=8`, restores best weights)
   - `ReduceLROnPlateau` (`factor=0.5`, `patience=4`)
   - `ModelCheckpoint` (saves best model by `val_loss`)
   - `TensorBoard` (per-run logs under `outputs/logs/`)
5. Saves the training history to `outputs/history/training_history.csv`.
6. Saves diagnostic plots (loss/MSE curves, sample reconstructions) to `outputs/figures/`.

Monitor training live with:

```bash
tensorboard --logdir outputs/logs
```

## Prediction

```bash
python predict.py
```

Optional CLI overrides:

```bash
python predict.py --model-path outputs/checkpoints/denoising_autoencoder_best.keras
python predict.py --noise-factor 0.6 --n-samples 15
```

This loads the saved checkpoint, denoises the test set, saves an
Original/Noisy/Reconstructed comparison grid to
`outputs/figures/test_reconstructions_sample.png`, and writes a JSON
evaluation report (`outputs/figures/evaluation_report.json`).

## Results

After training, `outputs/figures/` contains:

- `training_history.png` — training vs. validation loss and MSE curves.
- `train_reconstructions_sample.png` — Original / Noisy / Reconstructed grid on training samples.
- `test_reconstructions_sample.png` — the same comparison on the held-out test set.
- `evaluation_report.json` — aggregated MSE / PSNR / SSIM scores.

**Observations** (fill in with your own run's numbers):

- The model rapidly learns to suppress additive Gaussian noise within the
  first several epochs, with validation loss plateauing once the network
  saturates the information bottleneck of the `7×7×64` latent space.
- Reconstructions preserve stroke topology (loops, curves, junctions) well,
  but can slightly blur fine stroke edges — an expected trade-off of a
  compressive bottleneck architecture.
- Increasing `noise_factor` degrades PSNR/SSIM roughly monotonically;
  binary cross-entropy tends to produce marginally sharper edges than MSE
  loss on this pixel range, at the cost of slightly higher raw MSE.

## Evaluation Metrics

| Metric | What it measures | Direction |
|--------|-------------------|-----------|
| **MSE**  | Mean squared pixel-wise error between clean and reconstructed images | lower is better |
| **PSNR** | Peak Signal-to-Noise Ratio (dB); log-scaled inverse of MSE | higher is better |
| **SSIM** | Structural Similarity Index; perceptual similarity of local structure | higher is better (max 1.0) |

All three are computed via `utils/metrics.py`, using `scikit-image` for PSNR
and SSIM, and reported as dataset-wide averages in `EvaluationReport`.

## Screenshots

> Run `train.py` then `predict.py` to populate these placeholders with your
> own generated figures.

- `assets/training_history.png`
- `assets/reconstruction_comparison.png`

## Future Improvements

- Swap the plain convolutional autoencoder for a **U-Net-style** architecture
  with skip connections to better preserve fine digit strokes.
- Add a **variational** (VAE) or **denoising diffusion** variant for
  comparison against the deterministic autoencoder.
- Extend noise modeling beyond Gaussian (e.g. salt-and-pepper, speckle,
  occlusion masks) to test robustness to varied corruption types.
- Package the trained model behind a small Flask/FastAPI inference endpoint
  or Streamlit demo for interactive denoising.
- Add unit tests (`pytest`) for `utils/preprocessing.py` and `utils/metrics.py`.
- Track experiments (noise factor × loss function × architecture depth)
  with a lightweight experiment tracker such as MLflow or Weights & Biases.

---

## License

Released under the [MIT License](LICENSE).
