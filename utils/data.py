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
    num_workers: int = 2) -> DataLoader:
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
    train_indices, _ = get_validation_and_train_indices(cifar_data)
    subsample_cifar_data_set(sub_sample_indices=train_indices, cifar_data=cifar_data)
    return DataLoader(cifar_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def subsample_cifar_data_set(sub_sample_indices, cifar_data: CifarDataSet):
    """
    Sub-samples the CIFAR-100 data set according to the given indices.
    Modifies cifar_data in place.
    """
    cifar_data.data["data"] = cifar_data.data["data"][sub_sample_indices.numpy()]
    cifar_data.data["labels"] = cifar_data.data["labels"][sub_sample_indices.numpy()]
    cifar_data.integer_labels = (
        torch.tensor(cifar_data.integer_labels)[sub_sample_indices.numpy()].tolist()
    )
    cifar_data.current_data = cifar_data.partition_data()


def get_validation_and_train_indices(cifar_data: CifarDataSet, num_classes: int = 100):
    """
    Splits the CIFAR-100 data into validation (50 per class) and train (450 per class).
    Returns (train_indices, validation_indices).
    """
    num_val_samples_per_class = 50
    num_train_samples_per_class = 450
    validation_set_size = num_val_samples_per_class * num_classes
    train_set_size = num_train_samples_per_class * num_classes

    validation_indices = torch.zeros(validation_set_size, dtype=torch.int32)
    train_indices = torch.zeros(train_set_size, dtype=torch.int32)
    current_val_samples = 0
    current_train_samples = 0

    for i in range(num_classes):
        class_indices = torch.argwhere(cifar_data.data["labels"][:, i] == 1).flatten()
        validation_indices[current_val_samples:(current_val_samples + num_val_samples_per_class)] = (
            class_indices[:num_val_samples_per_class]
        )
        train_indices[current_train_samples:(current_train_samples + num_train_samples_per_class)] = (
            class_indices[num_val_samples_per_class:]
        )
        current_val_samples += num_val_samples_per_class
        current_train_samples += num_train_samples_per_class

    return train_indices, validation_indices



def load_class_order(algo, seed):
    algorithm = ALGORITHM[algo]
    file_name = f'index-{seed}.npy'
    url = f"https://huggingface.co/onlytojay/lop-resnet18/resolve/main/{algorithm}/class_order/{file_name}"
    class_order = load_npy(url)
    return class_order

def parse_class_order(class_order, session, num_classes_per_session):
    N = num_classes_per_session
    return dict(all_classes = class_order[:session*N],
                old_classes = class_order[:max(0, (session-1)*N)],
                new_classes = class_order[max(0, (session-1)*N):session*N])

def load_class_info(algo, seed, session, num_classes_per_session):
    class_order = load_class_order(algo, seed, session)
    return parse_class_order(class_order, session, num_classes_per_session)
