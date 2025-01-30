import torch
from csse.lop.nets.torchvision_modified_resnet import build_resnet18

def load_lop_resnet18(weight_url: str):
    """Load LOP ResNet18 model with automatic caching and downloading."""
    cache_dir = torch.hub.get_dir()  # Default: ~/.cache/torch/hub/
    print(f"Using cache directory: {cache_dir}")

    state_dict = torch.hub.load_state_dict_from_url(
        weight_url, map_location="cpu", check_hash=False, progress=True
    )

    lop_resnet18 = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    lop_resnet18.load_state_dict(state_dict)

    return lop_resnet18
