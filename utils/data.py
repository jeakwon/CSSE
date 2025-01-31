import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from csse.external_codes.mlproj_manager.cifar_data_loader import CifarDataSet
from csse.external_codes.mlproj_manager.image_transformations import (
    ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator
)

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