import os
import re
import requests
import urllib.parse
from io import BytesIO

import numpy as np
import torch

from csse.lop.nets.torchvision_modified_resnet import build_resnet18

# Define a unified cache directory for both .npy and .pt files
CACHE_DIR = os.path.expanduser("~/.cache/csse/")
os.makedirs(CACHE_DIR, exist_ok=True)  # Ensure cache directory exists

def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    return urllib.parse.urlparse(path).scheme in ("http", "https")

def get_cached_path(url):
    """Generate a local cached file path based on the URL filename."""
    filename = os.path.basename(url)  # Extract filename from URL
    return os.path.join(CACHE_DIR, filename)

def load_npy(file_path):
    """Load a NumPy array from either a URL (with caching) or a local file."""
    if is_url(file_path):
        cached_file = get_cached_path(file_path)

        # Use cached file if it exists
        if os.path.exists(cached_file):
            print(f"Using cached .npy file: {cached_file}")
            return np.load(cached_file)

        # Otherwise, download and cache
        print(f"Downloading .npy file from: {file_path}")
        response = requests.get(file_path)
        response.raise_for_status()

        with open(cached_file, "wb") as f:
            f.write(response.content)  # Save to local cache

        return np.load(cached_file)  # Load from cache after downloading

    elif os.path.exists(file_path):
        return np.load(file_path)  # Load from local file
    else:
        raise ValueError(f"File or URL not found: {file_path}")

def load_state_dict(file_path: str):
    """Load PyTorch state_dict from either a URL or a local file (with caching)."""
    if is_url(file_path):
        cached_file = get_cached_path(file_path)

        # Use cached file if it exists
        if os.path.exists(cached_file):
            print(f"Using cached state_dict: {cached_file}")
            return torch.load(cached_file, map_location="cpu")

        # Otherwise, download and cache
        print(f"Downloading state_dict from: {file_path}")
        state_dict = torch.hub.load_state_dict_from_url(
            file_path, model_dir=CACHE_DIR, map_location="cpu", check_hash=False, progress=True
        )

        return state_dict

    elif os.path.exists(file_path):
        return torch.load(file_path, map_location="cpu")
    else:
        raise ValueError(f"File or URL not found: {file_path}")

def load_class_info(algo, seed, session):
    url = f"https://huggingface.co/onlytojay/lop-resnet18/resolve/main/{algo}/class_order/index-{seed}.npy"
    class_order = load_npy(url)
    return dict(class_order = class_order,
                all_classes = class_order[:session*5],
                new_classes = class_order[(session-1)*5:session*5],
                old_classes = class_order[:(session-1)*5])

def load_lop_resnet18(algo, seed, session):
    url = f"https://huggingface.co/onlytojay/lop-resnet18/resolve/main/{algo}/model_parameters/index-{seed}_epoch-{session*200}.pt"
    state_dict = load_state_dict(url)
    model = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    model.load_state_dict(state_dict)
    return model
