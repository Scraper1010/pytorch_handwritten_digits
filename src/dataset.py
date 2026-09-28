import gzip
import os
import struct
import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, TensorDataset

BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_lbl": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_lbl": "t10k-labels-idx1-ubyte.gz",
}

def download_mnist(target_dir="data"):
    """Downloads compressed MNIST archives if not already present on disk."""
    os.makedirs(target_dir, exist_ok=True)
    for filename in FILES.values():
        path = os.path.join(target_dir, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            res = requests.get(BASE_URL + filename, stream=True, timeout=30)
            res.raise_for_status()
            with open(path, "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)

def _load_idx_images(filepath):
    """Parses binary IDX image data into float32 tensors scaled to [0, 1]."""
    with gzip.open(filepath, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        buffer = f.read()
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape(count, 1, rows, cols)
        return torch.from_numpy(arr.copy()).float() / 255.0

def _load_idx_labels(filepath):
    """Parses binary IDX label data into int64 tensors."""
    with gzip.open(filepath, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        buffer = f.read()
        arr = np.frombuffer(buffer, dtype=np.uint8)
        return torch.from_numpy(arr.copy()).long()

def get_dataloaders(data_dir="data", train_batch_size=64, test_batch_size=256):
    """Entry point returning train and test DataLoader instances."""
    download_mnist(data_dir)
    
    train_x = _load_idx_images(os.path.join(data_dir, FILES["train_img"]))
    train_y = _load_idx_labels(os.path.join(data_dir, FILES["train_lbl"]))
    test_x = _load_idx_images(os.path.join(data_dir, FILES["test_img"]))
    test_y = _load_idx_labels(os.path.join(data_dir, FILES["test_lbl"]))

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=train_batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=test_batch_size,
        shuffle=False
    )
    return train_loader, test_loader