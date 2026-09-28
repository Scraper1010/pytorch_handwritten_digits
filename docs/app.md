# App — live Gradio demo

Implemented in [`app.py`](../app.py). Draw a digit on a canvas, get the
model's top-3 confidences instantly. Run it with:

```bash
python app.py
# → http://127.0.0.1:7860
```

## Flow

1. **Load once at startup.** `DigitCNN` is moved to GPU/CPU and weights are
   loaded from `checkpoints/mnist_cnn.pth` (`map_location` keeps GPU-trained
   checkpoints working on CPU). `model.eval()` freezes BatchNorm + Dropout.
2. **Draw.** A white 280×280 canvas with a black fixed-color pencil and an
   eraser — no color pickers, layers, uploads, or crop tools.
3. **Predict** (`predict_digit`):
   - canvas → tensor via [preprocessing](preprocessing.md),
   - blank canvas → empty label (never a bogus digit),
   - `torch.no_grad()` forward + softmax → `{digit: confidence}` for `gr.Label`.
4. **Clear.** A dedicated button restores the pristine white canvas *and*
   wipes the label in one handler.

## UI events

| Trigger | Behavior |
|---|---|
| `Classify Digit` click | Explicit inference |
| Canvas `change` | Live inference, collapsed to the latest stroke (`trigger_mode="always_last"`) |
| `Clear Canvas` click | Restores white canvas + empties the label |

## Reliability details

Two Gradio-on-Windows pitfalls are handled in code:

- **Temp-file races.** Every brush stroke fires an event and Gradio reads temp
  PNGs with no retry — parallel events collide on them. Events are serialized
  (`concurrency_limit=1`, `max_threads=1`) and a retry wrapper around
  `ImageEditor.convert_and_format_image` absorbs transient locks or 0-byte
  layer files, degrading to `None` (background fallback) instead of crashing.
- **Toolbar trash icon.** Gradio exposes no prop to remove it and it wipes the
  whole canvas unpredictably, so it is hidden with scoped CSS
  (`#draw-canvas button[aria-label='Clear canvas']`); clearing goes only
  through the controlled button above.
