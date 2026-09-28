# Training

Implemented in [`train.py`](../train.py) with data loading in
[`src/dataset.py`](../src/dataset.py) and curve plotting in
[`src/utils.py`](../src/utils.py).

## Data

`get_dataloaders()` downloads the four MNIST IDX archives on first run
(~11 MB into `data/`, git-ignored) and parses them directly with `gzip` +
`struct` — no `torchvision` dependency. Pixels are scaled to `[0, 1]`:

- Train: 60,000 images, batch 64, shuffled
- Test: 10,000 images, batch 256

## Loop

- Optimizer: Adam, learning rate 0.001
- Scheduler: `StepLR(step_size=3, gamma=0.5)` — halves the LR every 3 epochs
- Loss: `CrossEntropyLoss`
- Epochs: 10 (default; override with `--epochs`)
- Checkpointing: only the **best** validation accuracy is saved to
  `checkpoints/mnist_cnn.pth`

```bash
python train.py --epochs 10 --lr 0.001 --batch-size 64
```

A GPU is used automatically when available (`src/utils.py::get_device`),
otherwise training falls back to CPU.

## Results

![Training curves](../assets/training_curves.png)

10 epochs, no augmentation: **~99.3% test accuracy**, with train/val curves
tracking closely (no meaningful overfitting — BatchNorm + Dropout do their job).

## Train in Colab

No GPU at hand? Run the same pipeline in the cloud:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wXZBv274Y_Fzny7CFTi8A30CfBXjfY3U?usp=sharing)

The notebook walks through training step by step. Copy the resulting
`mnist_cnn.pth` into `checkpoints/` to use it with the app.
