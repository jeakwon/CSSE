import os
import urllib.parse
import requests
import torch
import numpy as np
from io import BytesIO
from csse.lop.nets.torchvision_modified_resnet import build_resnet18

def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    return urllib.parse.urlparse(path).scheme in ("http", "https")

def load_npy(file_path):
    """Load a NumPy array from either a URL or a local file."""
    if is_url(file_path):
        response = requests.get(file_path)
        response.raise_for_status()  # Ensure the request was successful
        return np.load(BytesIO(response.content))  # Load from URL as binary
    elif os.path.exists(file_path):
        return np.load(file_path)  # Load from local file
    else:
        raise ValueError(f"File or URL not found: {file_path}")

def load_state_dict(file_path: str):
    """Load LOP ResNet18 model with support for both URLs and local files."""
    # Load model weights (either from URL or local file)
    if is_url(file_path):
        print(f"Detected URL: {file_path}")
        cache_dir = torch.hub.get_dir()  # Default cache: ~/.cache/torch/hub/checkpoints/
        print(f"Using cache directory: {cache_dir}")

        state_dict = torch.hub.load_state_dict_from_url(
            file_path, map_location="cpu", check_hash=False, progress=True
        )
    elif os.path.exists(file_path):
        print(f"Detected local file: {file_path}")
        state_dict = torch.load(file_path, map_location="cpu")
    else:
        raise ValueError(f"File or URL not found: {file_path}")
    
    return state_dict