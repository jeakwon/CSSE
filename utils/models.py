import os
import re
import requests
import urllib.parse
from io import BytesIO

import numpy as np
import torch

from huggingface_hub import HfApi

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
    """Load Pytorch state_dict from either a URL and local file."""
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

def load_lop_experiment(algo, seed):
    api = HfApi()
    repo_id = "onlytojay/lop-resnet18"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main/"
    files = api.list_repo_files(repo_id)

    class_order_npy_url = None
    model_parameters_pt_urls = []

    for f in files:
        url = base_url+f
        if f"{algo}/class_order/index-{seed}" in f:
            class_order_npy_url = url
        elif f"{algo}/model_parameters/index-{seed}" in f:
            model_parameters_pt_urls.append( url )
    extract_epoch = lambda x: int(match.group(1)) if (match := re.search(r"epoch-(\d+)\.pt$", x)) else -1
    
    model_parameters_pt_urls = sorted(model_parameters_pt_urls, key=extract_epoch)
    
    if class_order_npy_url is None:
        print(f'Experiment with algo={algo} & seed={seed} not exist.')
        print(f'Please visit {base_url} to check!')

    return class_order_npy_url, model_parameters_pt_urls