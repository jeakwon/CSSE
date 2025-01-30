import os
import re
import requests
import urllib.parse
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import DataLoader

from csse.external_codes.lop.torchvision_modified_resnet import build_resnet18
from csse.external_codes.mlproj_manager.cifar_data_loader import CifarDataSet
from csse.external_codes.mlproj_manager.image_transformations import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator

ALGORITHM = {
    'bp':'base_deep_learning_system',
    'cb':'continual_backpropagation',
    'hr':'head_resetting',
    'rt':'retrained_network',
    'sp':'shrink_and_perturb',
}
# Define a unified cache directory for both .npy and .pt files
CACHE_DIR = os.path.expanduser("~/.cache/csse/")
os.makedirs(CACHE_DIR, exist_ok=True)  # Ensure cache directory exists

class LopResNet18Loader:
    def __init__(self, algo, seed):
        self.algo = algo
        self.seed = seed
        self.session = None
        self.class_info = None

    def load_model(self, session):
        self.session = session
        self.class_info = load_class_info(algo=self.algo, seed=self.seed, session=self.session)
        return load_lop_resnet18(algo=self.algo, seed=self.seed, session=session)

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
    algorithm = ALGORITHM[algo]
    file_name = f'index-{seed}.npy'
    url = f"https://huggingface.co/onlytojay/lop-resnet18/resolve/main/{algorithm}/class_order/{file_name}"
    class_order = load_npy(url)
    return dict(class_order = class_order,
                learned_classes = class_order[:session*5],
                earlier_classes = class_order[:(session-1)*5],
                current_classes = class_order[(session-1)*5:session*5],
                unknown_classes = class_order[:session*5])

def load_lop_resnet18(algo, seed, session):
    algorithm = ALGORITHM[algo]
    file_name = f'index-{seed}_epoch-{session*200}.pt'
    url = f"https://huggingface.co/onlytojay/lop-resnet18/resolve/main/{algorithm}/model_parameters/{file_name}"
    state_dict = load_state_dict(url)
    model = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    model.load_state_dict(state_dict)
    return model

def load_cifar100(
    train: bool = True, 
    data_path: str = CACHE_DIR, 
    shuffle: bool = True,
    batch_size: int = 100,
    num_workers: int = 2) -> (CifarDataSet, DataLoader):
    """
    Loads the cifar 100 data set with normalization
    :param data_path: path to the directory containing the data set
    :param train: bool that indicates whether to load the train or test data
    :return: torch DataLoader object
    """
    cifar_data = CifarDataSet(root_dir=data_path,
                              train=train,
                              cifar_type=100,
                              device=None,
                              image_normalization="max",
                              label_preprocessing="one-hot",
                              use_torch=True)

    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    transformations = [
        ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
        Normalize(mean=mean, std=std),  # center by mean and divide by std
    ]
    if train:
        transformations.append(RandomHorizontalFlip(p=0.5))
        transformations.append(RandomCrop(size=32, padding=4, padding_mode="reflect"))
        transformations.append(RandomRotator(degrees=(0,15)))

    cifar_data.set_transformation(transforms.Compose(transformations))
    return DataLoader(cifar_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)