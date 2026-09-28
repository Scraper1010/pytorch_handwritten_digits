# Model — `DigitCNN`

Defined in [`src/model.py`](../src/model.py). A compact convolutional network for
28×28 grayscale digits: under 450k parameters, yet ~99.3% MNIST test accuracy.

## Architecture

Input `(N, 1, 28, 28)` → 10 class logits:

| Stage | Layers | Output shape |
|---|---|---|
| Feature block 1 | Conv2d(1→32, 3×3, pad 1) → BatchNorm → ReLU → MaxPool(2) | `(N, 32, 14, 14)` |
| Feature block 2 | Conv2d(32→64, 3×3, pad 1) → BatchNorm → ReLU → MaxPool(2) | `(N, 64, 7, 7)` |
| Classifier head | Flatten → Linear(3136→128) → ReLU → Dropout(0.3) → Linear(128→10) | `(N, 10)` |

Design notes:

- **3×3 padded convolutions** preserve spatial size so each MaxPool exactly halves it (28 → 14 → 7).
- **BatchNorm** stabilizes training and lets Adam converge in ~10 epochs.
- **Dropout (0.3)** only acts during training; the app calls `model.eval()`, which also freezes BatchNorm statistics — required for correct single-image inference.
- No softmax inside the model: [`train.py`](../train.py) uses `CrossEntropyLoss` (expects raw logits), while [`app.py`](../app.py) applies `softmax` only to display confidences.

## Sanity check

```bash
python src/model.py
# Output shape for batch size 2: torch.Size([2, 10])
# Trainable parameters: ...
```

## Weights

Trained weights live at `checkpoints/mnist_cnn.pth` (1.7 MB, git-ignored — see
[training](training.md)). The app loads them with `map_location=device`, so a
checkpoint trained on GPU also loads on CPU-only machines.
