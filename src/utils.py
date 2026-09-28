import numpy as np
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def get_device():
    """Detects available hardware accelerator."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_canvas_drawing(canvas_data):
    """
    Transforms Gradio drawing data into the exact format MNIST models expect:
    1. Extracts the drawing (composite, layer, or background) and converts to grayscale.
    2. Crops to the bounding box of the drawn digit.
    3. Resizes proportionally to fit in a 20x20 pixel box.
    4. Centers the digit inside a 28x28 black canvas.
    5. Normalizes to a (1, 1, 28, 28) torch.Tensor.

    Accepts Gradio 6 Sketchpad/ImageEditor output (EditorValue dict with
    'background'/'layers'/'composite'), legacy dicts with 'image', raw
    numpy arrays, PIL images, or filepath strings. Returns a zeros tensor
    for empty canvas instead of None so callers don't crash.
    """
    if canvas_data is None:
        return None

    # Gradio 6 ImageEditor/Sketchpad returns an EditorValue dict:
    # {'background': ..., 'layers': [...], 'composite': ...}
    # Legacy code used {'composite': ...} or {'image': ...}.
    if isinstance(canvas_data, dict):
        img_array = canvas_data.get("composite", None)
        if img_array is None:
            layers = canvas_data.get("layers") or []
            for layer in layers:
                if layer is not None:
                    img_array = layer
                    break
        if img_array is None:
            img_array = canvas_data.get("background", None)
        if img_array is None:
            # Legacy fallback key
            img_array = canvas_data.get("image", None)
    else:
        img_array = canvas_data

    # Empty canvas (e.g. cleared Sketchpad sends composite=None, layers=[])
    if img_array is None:
        return torch.zeros((1, 1, 28, 28), dtype=torch.float32)

    # Support filepath strings (type="filepath") and PIL images (type="pil")
    if isinstance(img_array, str):
        try:
            img = Image.open(img_array).convert("L")
        except Exception:
            return torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    elif isinstance(img_array, Image.Image):
        # Handle RGBA transparency by compositing onto a black background
        if img_array.mode == "RGBA":
            bg = Image.new("RGBA", img_array.size, (0, 0, 0, 255))
            img = Image.alpha_composite(bg, img_array).convert("L")
        else:
            img = img_array.convert("L")
    else:
        try:
            img_array = np.asarray(img_array)
        except Exception:
            return torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        if img_array.size == 0:
            return torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        # Convert to PIL Image in grayscale mode ('L'), handling alpha
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            # RGBA: composite white strokes / transparent bg onto black
            rgba = Image.fromarray(img_array.astype("uint8"), mode="RGBA")
            bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
            img = Image.alpha_composite(bg, rgba).convert("L")
        else:
            img = Image.fromarray(img_array.astype("uint8")).convert("L")

    # MNIST uses white strokes on black; invert drawings made black-on-white
    # (detected via mean pixel intensity) so both styles classify correctly.
    if np.mean(img) > 128:
        img = ImageOps.invert(img)

    # Convert to NumPy array for bounding box search
    arr = np.array(img)
    non_zeros = np.argwhere(arr > 30)

    # Empty canvas guard
    if non_zeros.size == 0:
        return torch.zeros((1, 1, 28, 28), dtype=torch.float32)

    # Crop to the bounding box of the drawn strokes
    ymin, xmin = non_zeros.min(axis=0)
    ymax, xmax = non_zeros.max(axis=0)
    cropped = img.crop((xmin, ymin, xmax + 1, ymax + 1))

    # Fit inside a 20x20 area while keeping the aspect ratio
    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round((20.0 / w) * h)))
    else:
        new_h = 20
        new_w = max(1, int(round((20.0 / h) * w)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Paste into a 28x28 black canvas
    canvas_28 = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas_28.paste(resized, (paste_x, paste_y))

    # Convert to PyTorch Tensor: shape (1, 1, 28, 28), values [0.0, 1.0]
    tensor = torch.from_numpy(np.array(canvas_28)).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W): batch + channel

def save_training_curves(history, output_path="assets/training_curves.png"):
    """Generates and saves loss & accuracy curves to disk."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val", marker="s")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.grid(True, linestyle=":")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train", marker="o")
    plt.plot(epochs, history["val_acc"], label="Val", marker="s")
    plt.title("Accuracy Curve (%)")
    plt.xlabel("Epoch")
    plt.grid(True, linestyle=":")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()