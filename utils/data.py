import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from csse.utils.tools import load_npy
from csse.external_codes.mlproj_manager.cifar_data_loader import CifarDataSet
from csse.external_codes.mlproj_manager.image_transformations import (
    ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator
)

# Mapping of algorithm abbreviations to their full names
ALGORITHM = {
    'bp': 'base_deep_learning_system',
    'cb': 'continual_backpropagation',
    'hr': 'head_resetting',
    'rt': 'retrained_network',
    'sp': 'shrink_and_perturb',
}
# Define a unified cache directory for storing .npy and .pt files
CACHE_DIR = os.path.expanduser("~/.cache/csse/")
os.makedirs(CACHE_DIR, exist_ok=True)  # Ensure cache directory exists

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

def load_class_order(algo, seed):
    algorithm = ALGORITHM[algo]
    file_name = f'index-{seed}.npy'
    url = f"https://huggingface.co/onlytojay/lop-resnet18/resolve/main/{algorithm}/class_order/{file_name}"
    class_order = load_npy(url)
    return class_order

def parse_class_order(class_order, session, num_classes_per_session):
    N = num_classes_per_session
    return dict(entire_classes = class_order[:session*N],
                former_classes = class_order[:max(0, (session-1)*N)],
                recent_classes = class_order[max(0, (session-1)*N):session*N],
                unseen_classes = class_order[session*N:])

def load_class_info(algo, seed, session, num_classes_per_session):
    class_order = load_class_order(algo, seed, session)
    return parse_class_order(class_order, session, num_classes_per_session)
