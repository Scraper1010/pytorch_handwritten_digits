# Preprocessing — canvas drawing → model tensor

Implemented in `preprocess_canvas_drawing` ([`src/utils.py`](../src/utils.py)).
Its job: turn whatever the user drew in the browser into exactly what the
network was trained on — a `(1, 1, 28, 28)` float tensor in `[0, 1]` with a
white digit on a black background.

## Pipeline

1. **Extract the drawing.** Gradio 6 sends an `EditorValue` dict
   (`background` / `layers` / `composite`). The composite is preferred, then
   the first non-empty layer, then the background. Raw arrays, PIL images and
   file paths are accepted too; an empty canvas yields a zeros tensor.
2. **Grayscale with alpha handling.** RGBA drawings are composited onto a black
   background first, so transparent pixels become black instead of garbage.
3. **Auto-invert.** If the mean pixel value is above 128 the user drew
   black-on-white (like this app's white canvas); the image is inverted to
   MNIST-style white-on-black. This makes both drawing styles classify.
4. **Bounding-box crop.** Pixels above a threshold of 30 define the digit's
   box; everything else is discarded so pen position doesn't matter.
5. **MNIST normalization.** The crop is aspect-ratio-resized into a 20×20 box
   (the MNIST standard — digits never touch the border), pasted centered on a
   28×28 black canvas, scaled to `[0, 1]`, and given batch + channel dims.

## Why it matters

A raw 280×280 canvas fed straight to the network would fail: wrong size,
wrong colors, digit off-center. Steps 4–5 replicate how the MNIST dataset
itself was built, which is a large part of why a small model reaches ~99%.
