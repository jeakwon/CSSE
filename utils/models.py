import os
import urllib.parse
import torch
from csse.lop.nets.torchvision_modified_resnet import build_resnet18

def load_lop_resnet18(weight_file_path: str):
    """Load LOP ResNet18 model with support for both URLs and local files."""
    def is_url(path: str) -> bool:
        return urllib.parse.urlparse(path).scheme in ("http", "https")

    # Load model weights (either from URL or local file)
    if is_url(weight_file_path):
        print(f"Detected URL: {weight_file_path}")
        cache_dir = torch.hub.get_dir()  # Default cache: ~/.cache/torch/hub/checkpoints/
        print(f"Using cache directory: {cache_dir}")

        state_dict = torch.hub.load_state_dict_from_url(
            weight_file_path, map_location="cpu", check_hash=False, progress=True
        )
    elif os.path.exists(weight_file_path):
        print(f"Detected local file: {weight_file_path}")
        state_dict = torch.load(weight_file_path, map_location="cpu")
    else:
        raise ValueError(f"File or URL not found: {weight_file_path}")

    # Initialize model and load weights
    lop_resnet18 = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    lop_resnet18.load_state_dict(state_dict)
    
    return lop_resnet18