import os
import requests
import urllib.parse

import numpy as np
import torch

# Define a unified cache directory for storing .npy and .pt files
CACHE_DIR = os.path.expanduser("~/.cache/csse/")
os.makedirs(CACHE_DIR, exist_ok=True)  # Ensure cache directory exists

def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    return urllib.parse.urlparse(path).scheme in ("http", "https")

def get_cached_path(url):
    """Generate a local cached file path using a subdirectory structure for readability."""
    parsed_url = urllib.parse.urlparse(url)
    sub_dirs = parsed_url.path.lstrip("/").split("/")[:-1]  # Extract subdirectories
    filename = os.path.basename(parsed_url.path)  # Get the actual filename

    # Create subdirectory path inside cache
    subdir_path = os.path.join(CACHE_DIR, *sub_dirs)
    os.makedirs(subdir_path, exist_ok=True)  # Ensure subdirectory exists

    return os.path.join(subdir_path, filename)  # Return full path

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
            return torch.load(cached_file, map_location="cpu", weights_only=True)

        # Otherwise, download and cache
        state_dict = torch.hub.load_state_dict_from_url(
            file_path, model_dir=os.path.dirname(cached_file), map_location="cpu", check_hash=False, progress=True
        )

        return state_dict

    elif os.path.exists(file_path):
        return torch.load(file_path, map_location="cpu")
    else:
        raise ValueError(f"File or URL not found: {file_path}")