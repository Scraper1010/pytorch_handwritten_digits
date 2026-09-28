"""
Digit Recognition Package
Exposes core model architecture, dataset pipelines, and drawing preprocessing utilities.
"""

from src.dataset import get_dataloaders
from src.model import DigitCNN
from src.utils import get_device, preprocess_canvas_drawing, save_training_curves

__all__ = [
    "DigitCNN",
    "get_dataloaders",
    "get_device",
    "preprocess_canvas_drawing",
    "save_training_curves",
]