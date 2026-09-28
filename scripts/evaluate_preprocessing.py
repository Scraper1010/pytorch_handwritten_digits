"""Ablation: 20px-box + padding vs direct 28x28 resize for canvas drawings.

Simulates realistic pen input by taking MNIST test digits, cropping them
tight (as if drawn edge-to-edge across the 280px canvas), and comparing:

  A. preprocess_canvas_drawing  (crop -> 20x20 box -> pad to 28x28)
  B. naive baseline             (direct resize to 28x28)

Usage:  python scripts/evaluate_preprocessing.py [--n 500]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from src.dataset import get_dataloaders
from src.model import DigitCNN
from src.utils import get_device, preprocess_canvas_drawing


def simulate_canvas(digit_28):
    """Stretch a tight bbox crop edge-to-edge over a 280px white canvas."""
    nz = np.argwhere(digit_28 > 30)
    ymin, xmin = nz.min(axis=0)
    ymax, xmax = nz.max(axis=0)
    tight = digit_28[ymin : ymax + 1, xmin : xmax + 1]
    return np.array(Image.fromarray(255 - tight).resize((280, 280), Image.BILINEAR))


def baseline_resize(canvas):
    """Naive alternative: direct resize to 28x28 + invert if needed."""
    img = Image.fromarray(canvas.astype("uint8")).convert("L").resize((28, 28), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0) / 255.0
    return 1.0 - tensor if np.mean(np.array(img)) > 128 else tensor


def main():
    parser = argparse.ArgumentParser(description="Preprocessing ablation (A vs B)")
    parser.add_argument("--n", type=int, default=500, help="Test samples to simulate")
    args = parser.parse_args()

    device = get_device()
    model = DigitCNN().to(device)
    model.load_state_dict(
        torch.load(os.path.join("checkpoints", "mnist_cnn.pth"), map_location=device)
    )
    model.eval()

    _, test_loader = get_dataloaders()
    correct_a = correct_b = 0
    conf_a = conf_b = 0.0
    seen = 0

    with torch.no_grad():
        for images, labels in test_loader:
            for img, label in zip(images, labels):
                if seen >= args.n:
                    break
                digit_28 = (img.squeeze(0).numpy() * 255).astype(np.uint8)
                canvas = simulate_canvas(digit_28)
                drawing = {"background": canvas, "layers": [], "composite": canvas}

                prob_a = torch.softmax(
                    model(preprocess_canvas_drawing(drawing).to(device)), dim=1
                ).squeeze(0)
                prob_b = torch.softmax(
                    model(baseline_resize(canvas).to(device)), dim=1
                ).squeeze(0)

                correct_a += int(prob_a.argmax()) == int(label)
                correct_b += int(prob_b.argmax()) == int(label)
                conf_a += float(prob_a[int(label)])
                conf_b += float(prob_b[int(label)])
                seen += 1
            if seen >= args.n:
                break

    print(f"n = {seen}")
    print(f"A (20px box + pad): acc={correct_a / seen * 100:.1f}%  mean-true-conf={conf_a / seen:.3f}")
    print(f"B (direct 28x28)  : acc={correct_b / seen * 100:.1f}%  mean-true-conf={conf_b / seen:.3f}")


if __name__ == "__main__":
    main()
