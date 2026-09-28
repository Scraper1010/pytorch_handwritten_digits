import os
import time
import torch
import gradio as gr
from PIL import Image

from src.model import DigitCNN
from src.utils import get_device, preprocess_canvas_drawing

# ---------------------------------------------------------------------------
# Gradio ImageEditor reads its temp PNGs with plain `PIL.Image.open(path)`.
# On Windows those files can be transiently unreadable: parallel inference
# events lock them, antivirus scans hold them, and a freshly cleared layer
# can arrive as a 0-byte file PIL cannot identify. Any of these used to crash
# preprocessing before this app's code even ran. Retry briefly, then yield
# None for that image (handled downstream via the background fallback)
# instead of failing the whole event.
# ---------------------------------------------------------------------------
_orig_convert = gr.ImageEditor.convert_and_format_image


def _safe_convert_and_format_image(self, file, _attempts=6):
    last_err = None
    for attempt in range(_attempts):
        try:
            return _orig_convert(self, file)
        except OSError as e:  # PermissionError, UnidentifiedImageError, FileNotFoundError
            last_err = e
            time.sleep(0.05 * (2**attempt))  # 50ms, 100ms, ... ~1.5s total
    print(f"[!] Skipping unreadable temp image after retries: {last_err}")
    return None


gr.ImageEditor.convert_and_format_image = _safe_convert_and_format_image

# 1. Setup Device & Load Model Weights
device = get_device()
checkpoint_path = os.path.join("checkpoints", "mnist_cnn.pth")

model = DigitCNN().to(device)

if os.path.exists(checkpoint_path):
    # map_location ensures weights load seamlessly even if trained on GPU and deployed on CPU
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"[+] Loaded weights from {checkpoint_path} onto {device}")
else:
    print(f"[!] Warning: '{checkpoint_path}' not found. Run 'python train.py' first.")

model.eval()  # Freeze BatchNorm stats and disable Dropout


# 2. Prediction Pipeline Function
# Returns a {digit: confidence} dict, or None to leave the label fully empty
# (blank canvas / cleared state show nothing instead of a bogus digit).
def predict_digit(canvas_data):
    if canvas_data is None:
        return None

    # Preprocess canvas sketch to MNIST-compliant (1, 1, 28, 28) tensor.
    # Handles Gradio 6 EditorValue dicts, empty canvas (zeros tensor), etc.
    try:
        tensor = preprocess_canvas_drawing(canvas_data)
    except Exception as e:
        print(f"[!] Preprocessing failed: {e}")
        return None
    if tensor is None:
        return None

    # Blank canvas (all zeros) -> empty label instead of classifying
    # empty input into a bogus digit. This also keeps the label cleared
    # after "Clear Canvas" via the change event.
    try:
        if float(tensor.sum()) == 0.0:
            return None
    except Exception:
        pass

    # Defensive reshape: ensure 4D (N, C, H, W) for BatchNorm2d
    if tensor.dim() == 2:  # (H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:  # (1, H, W) or (C, H, W)
        tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    # Return mapping of digit label to confidence float for gr.Label
    return {str(i): float(probabilities[i]) for i in range(10)}


# 3. Construct Gradio Interface
# NOTE: In Gradio 6.0+, theme/css/js must be passed to launch(), not Blocks().
with gr.Blocks(title="pytorch_handwritten_digits") as demo:
    gr.Markdown(
        """
        # Handwritten Digit Recognizer
        Draw a single digit (0–9) with the black pencil on the white canvas.
        Use the eraser to fix mistakes, or Clear Canvas to start over.
        """
    )

    # Vertical layout: canvas on top, buttons below it, prediction at bottom.
    # Minimal setup: white canvas, black fixed-color pencil + eraser.
    # - value=white image so the canvas starts white and Clear Canvas restores
    #   that exact same state (black-on-white is always visible)
    # - toolbar bin icon is hidden via launch(css=...) below; clearing happens
    #   only through our Clear Canvas button for predictable behavior
    # - brush locked to black via color_mode="fixed" (no color picker)
    # - default_size=18 fixes the initial pen size (Gradio has no API to
    #   lock the size slider, so it can still be adjusted)
    # - layers=False / transforms=[] / sources=() / buttons=[]
    #   hide all extra tools, leaving only pencil + eraser
    # - fixed_canvas=True locks the canvas dimensions
    # NOTE: white background + black digit is inverted to MNIST-style
    # (white on black) inside preprocess_canvas_drawing.
    sketchpad = gr.Sketchpad(
        value=Image.new("RGB", (280, 280), (255, 255, 255)),
        label="Drawing Canvas",
        type="numpy",
        image_mode="RGB",
        brush=gr.Brush(
            colors=["#000000"],
            default_color="#000000",
            color_mode="fixed",
            default_size=18,
        ),
        eraser=gr.Eraser(default_size=18),
        canvas_size=(280, 280),
        fixed_canvas=True,
        layers=False,
        transforms=[],
        sources=(),
        buttons=[],
        interactive=True,
        elem_id="draw-canvas",  # scoped by CSS below to hide the toolbar bin icon
    )
    with gr.Row():
        # Plain Button instead of ClearButton: restores the pristine white
        # canvas and wipes the label in a single handler, avoiding the
        # unreliable ClearButton reset behavior on ImageEditor components.
        clear_btn = gr.Button("Clear Canvas", variant="secondary")
        classify_btn = gr.Button("Classify Digit", variant="primary")

    output_label = gr.Label(
        label="Prediction Confidences",
        num_top_classes=3,  # Displays the top 3 highest probabilities
    )

    # Button = explicit inference; canvas change = live inference collapsed to
    # the latest stroke (trigger_mode="always_last"), so fast drawing never
    # stacks up parallel preprocessing jobs over the same temp files.
    # concurrency_limit=1 serializes runs on top of that.
    classify_btn.click(
        fn=predict_digit,
        inputs=sketchpad,
        outputs=output_label,
        concurrency_limit=1,
    )
    sketchpad.change(
        fn=predict_digit,
        inputs=sketchpad,
        outputs=output_label,
        trigger_mode="always_last",
        concurrency_limit=1,
        show_progress="hidden",
    )

    def clear_all():
        # Restore pristine white canvas and fully wipe prediction + confidences
        # (None empties the Label; zeros would still render 0% rows).
        return Image.new("RGB", (280, 280), (255, 255, 255)), None

    # Clearing the canvas must also clear the label. clear_all handles both
    # outputs at once; the sketchpad.clear handler covers any other clear
    # path (change on a blank canvas also yields None via predict_digit).
    clear_btn.click(
        fn=clear_all,
        outputs=[sketchpad, output_label],
        concurrency_limit=1,
        show_progress="hidden",
    )
    sketchpad.clear(
        fn=lambda: None,
        outputs=output_label,
        concurrency_limit=1,
        show_progress="hidden",
    )

    demo.queue(default_concurrency_limit=1)


# 4. Entrypoint
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True if you want a temporary public URL
        theme=gr.themes.Soft(),
        max_threads=1,  # Serialize event processing; avoids temp-file races
        # Gradio exposes no prop to hide the toolbar bin (trash) icon, so hide
        # it with scoped CSS. The button carries aria-label="Clear canvas".
        css="#draw-canvas button[aria-label='Clear canvas'] { display: none !important; }",
    )