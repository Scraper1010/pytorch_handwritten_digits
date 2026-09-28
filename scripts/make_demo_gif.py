"""Generate assets/demo.gif: a digit drawn stroke-by-stroke with live predictions.

Each frame renders the partial drawing (as the app's canvas would show it),
runs it through the real preprocessing + trained DigitCNN, and captions the
frame with the current top prediction. No hand-waving — what you see in the
GIF is what the model actually outputs.

Usage:  python scripts/make_demo_gif.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.model import DigitCNN
from src.utils import get_device, preprocess_canvas_drawing

CANVAS = 280
PEN_WIDTH = 18
STRIP_HEIGHT = 80
N_FRAMES = 32
HOLD_FRAMES = 8
OUTPUT = os.path.join("assets", "demo.gif")


def digit_two_strokes():
    """Polyline waypoints for a handwritten-style '2' on a 280px canvas."""
    top_arc = [(70, 70), (120, 45), (180, 50), (210, 80), (205, 120)]
    diagonal = [(205, 120), (150, 170), (100, 210), (70, 225)]
    baseline = [(70, 225), (130, 225), (210, 225)]
    return [top_arc, diagonal, baseline]


def interpolate(points, total):
    """Spread `total` samples along a polyline, returning cumulative paths."""
    seg_lengths = [
        np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points[:-1], points[1:])
    ]
    total_len = sum(seg_lengths)
    path = [points[0]]
    for a, b, seg_len in zip(points[:-1], points[1:], seg_lengths):
        steps = max(2, round(total * seg_len / total_len))
        for i in range(1, steps + 1):
            t = i / steps
            path.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return path


def load_font(size):
    for candidate in (r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default()


def caption_frame(drawing, model, device, font_big, font_small):
    """Render drawing + prediction strip as one RGB frame."""
    arr = np.array(drawing)
    tensor = preprocess_canvas_drawing(
        {"background": arr, "layers": [], "composite": arr}
    )
    if tensor is None or float(tensor.sum()) == 0.0:
        top, conf = None, 0.0
    else:
        with torch.no_grad():
            probs = torch.softmax(model(tensor.to(device)), dim=1).squeeze(0)
        top = int(probs.argmax())
        conf = float(probs[top])

    frame = Image.new("RGB", (CANVAS, CANVAS + STRIP_HEIGHT), (30, 30, 30))
    frame.paste(drawing, (0, 0))
    draw = ImageDraw.Draw(frame)
    if top is None:
        draw.text((20, CANVAS + 22), "drawing...", fill=(180, 180, 180), font=font_small)
    else:
        draw.text((20, CANVAS + 8), f"{top}", fill=(255, 255, 255), font=font_big)
        draw.text((70, CANVAS + 22), f"{conf * 100:.1f}%", fill=(140, 220, 140), font=font_small)
        bar_w = int((CANVAS - 40) * conf)
        draw.rectangle([20, CANVAS + 58, 20 + bar_w, CANVAS + 66], fill=(80, 200, 120))
    return frame


def main():
    device = get_device()
    model = DigitCNN().to(device)
    model.load_state_dict(
        torch.load(os.path.join("checkpoints", "mnist_cnn.pth"), map_location=device)
    )
    model.eval()

    font_big = load_font(44)
    font_small = load_font(22)

    # One continuous pen path across all strokes of the "2".
    full_path = []
    for stroke in digit_two_strokes():
        full_path.extend(interpolate(stroke, N_FRAMES // 3))

    frames = []
    for i in range(1, N_FRAMES + 1):
        shown = full_path[: max(2, int(len(full_path) * i / N_FRAMES))]
        drawing = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        ImageDraw.Draw(drawing).line(shown, fill=(0, 0, 0), width=PEN_WIDTH, joint="curve")
        frames.append(caption_frame(drawing, model, device, font_big, font_small))

    frames.extend([frames[-1]] * HOLD_FRAMES)  # hold the final prediction

    os.makedirs("assets", exist_ok=True)
    frames[0].save(
        OUTPUT, save_all=True, append_images=frames[1:], duration=110, loop=0
    )
    print(f"Saved {OUTPUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
