# Modified from
# https://github.com/JFernando4/Research_Project_Manager/blob/master/mlproj_manager/problems/supervised_learning/cifar/cifar_data_loader.py
"""
Implementation of a data loader for the data set CIFAR-100. This implementation allows the user to select specific
classes to include in or exclude from the data set.
"""
# from third party packages:
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

# from project files:

# from mlproj_manager.problems.supervised_learning.abstract_dataset import CustomDataSet
# https://github.com/JFernando4/Research_Project_Manager/blob/master/mlproj_manager/problems/supervised_learning/abstract_dataset.py
class CustomDataSet(Dataset):
    """ General purpose abstract class for data sets used in supervised learning"""

    def __init__(self, root_dir):
        """
        :param root_dir (string): path to directory containing the data
        """
        self.root_dir = root_dir
        self.data = list()

    def load_data(self):
        """
        Loads the data set
        :return: return raw data
        """
        raise NotImplementedError("This method must be implemented for each individual data set!")

    def preprocess_data(self):
        """
        Preprocesses the data according to pre-specified parameters. Modifies self.data.
        :return: None
        """
        pass

    def __len__(self) -> int:
        """

        :return: (int) Returns the sample size of the data set.
        """
        raise NotImplementedError("This method must be implemented for each individual data set!")

    def __getitem__(self, idx) -> dict:
        """
        Returns a dictionary with two keys "image" and "label" which correspond to the given idx
        :param idx: index to retrieve data from
        :return: dictionary with sample
        """
        raise NotImplementedError("This method must be implemented for each individual data set!")


# from mlproj_manager.util.data_preprocessing_and_transformations import normalize, preprocess_labels
# https://github.com/JFernando4/Research_Project_Manager/blob/master/mlproj_manager/util/data_preprocessing_and_transformations/normalization_and_preprocessing.py
"""
Helper functions for normalizing data
"""
def normalize(data, norm_type: str, **kwargs):
    """
    Normalizes the given data according to the specified type
    :param data: (np.array or torch.Tensor) array containing the data
    :param norm_type: (str) type of normalization from the following options:
                                - None: no normalization
                                - centered: subtracts the mean and divides by standard deviation
                                - max: divides the maximum of the data
                                - minus-one-to-one: divides by the maximum of the data and scales to range(-1,1)
    :param kwargs: extra optional keyword arguments for the different normalization types. Available options:
                        - None: no available keyword arguments
                        - centered: avg = average of the data, stddev = standard deviation of the data
                        - max: max_val = maximum value of the data
                        - minus-one-to-one: max_val as above
                   any necessary statistic used for normalization is computed from the data if not give.
    :return: normalized numpy or torch array
    """
    if norm_type is None:
        return data
    elif norm_type == "centered":
        avg = np.average(np.float32(data)) if "avg" not in kwargs.keys() else kwargs["avg"]
        stddev = np.std(np.float32(data), ddof=1) if "stddev" not in kwargs.keys() else kwargs["stddev"]
        return (data - avg) / stddev
    elif norm_type == "max" or norm_type == "minus-one-to-one":
        max_val = max(data) if "max_val" not in kwargs.keys() else kwargs["max_val"]
        if norm_type == "minus-one-to-one":
            return 2 * (data / max_val) - 1
        return data / max_val
    else:
        raise ValueError("Invalid normalization type: {0}".format(norm_type))


def preprocess_labels(labels, preprocessing_type: str):
    """
    Preprocesses the given labels according to preprocessing_type
    :param labels: (numpy or torch array) array containing the labels to be preprocessed
    :param preprocessing_type: (str) type of preprocessing from the following:
                                        - None: does nothing
                                        - one-hot: turns labels into one-hot labels
    :return: preprocessed labels
    """
    if preprocessing_type is None:
        return labels
    elif preprocessing_type == "one-hot":
        return from_integers_to_one_hot(labels)
    else:
        raise ValueError("Invalid label preprocessing: {0}".format(preprocessing_type))

# from mlproj_manager.definitions import CIFAR_DATA_PATH
# https://github.com/JFernando4/Research_Project_Manager/blob/master/mlproj_manager/definitions.py
ROOT = os.path.dirname(os.path.abspath(__file__))
CIFAR_DATA_PATH = os.path.join(ROOT, "problems", "supervised_learning", "cifar")



class CifarDataSet(CustomDataSet):
    """ CIFAR dataset."""

    def __init__(self,
                 root_dir=CIFAR_DATA_PATH,
                 train=True,
                 transform=None,
                 cifar_type=100,
                 classes=None,
                 device=None,
                 image_normalization=None,
                 label_preprocessing=None,
                 use_torch=False,
                 flatten=False):
        """
        :param root_dir (string): path to directory with all the images
        :param train (booL): indicates whether to load the training or testing data set
        :param transform (callable, optional): transform to be applied to each sample
        :param cifar_type (int): indicates whether to use cifar10 (cifar_type=10) or cifar100 (cifar_type=100) data set
        :param classes (np.array, optional): np array of classes to load. If None, loads all the classes
        :param device (torch.device("cpu") or torch.device("cuda:0") or any variant): device where the data is going
                       to be loaded into. Make sure to have enough memory in your gpu when using "cuda:0" as device.
        :param image_normalization (str, optional): indicates how to normalize the data. options available:
                                                        - None: no normalization
                                                        - centered: subtracts the mean and divides by standard deviation
                                                        - max: divides by 255, the maximum pixel value
                                                        - minus-one-to-one: divides by the maximum of the data and
                                                          scales to range(-1,1)
        :param label_preprocessing (str, optional): indicates how to preprocess the labels. options available:
                                                        - None: use labels as is (numbers from 0 to 9)
                                                        - one-hot: converts labels to one-hot labels
        :param use_torch (bool, optional): if true, uses torch tensors to represent the data, otherwise, uses np array
        :param flatten: whether to flatten each image
        """
        super().__init__(root_dir)
        self.train = train
        self.transform = transform
        self.cifar_type = cifar_type
        self.classes = np.array(classes, np.int32) if classes is not None else np.arange(self.cifar_type)
        assert self.classes.size <= 100
        self.device = torch.device("cpu") if device is None else device
        self.image_norm_type = image_normalization
        self.label_preprocessing = label_preprocessing
        self.use_torch = use_torch
        self.flatten = flatten

        self.data = self.load_data()
        self.integer_labels = self.data["labels"]
        self.preprocess_data()
        self.current_data = self.partition_data()

    def load_data(self):
        """
        Load the train or test files using torchvision. The loaded files are not preprocessed. It downloads the data if
        necessary and stores it in "./src/problems/supervised_learning/cifar/". It creates several files, see
        https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10 for mor info
        :return: dictionary with keys "data" and "labels"
        """
        # Note: the shape of CIFAR10 and 100 is (num_samples, 32, 32, 3)
        if self.cifar_type == 10:
            raw_data_set = torchvision.datasets.CIFAR10(root=self.root_dir, train=self.train, download=True)
        elif self.cifar_type == 100:
            raw_data_set = torchvision.datasets.CIFAR100(root=self.root_dir, train=self.train, download=True)
        else:
            raise ValueError("Invalid CIFAR type: {0}. Choose either 100 or 10".format(self.cifar_type))
        data = {"data": raw_data_set.data, "labels": raw_data_set.targets}
        return data

    def preprocess_data(self):
        """
        Reshapes the data into the correct dimensions, converts it to the correct type, and normalizes the data if
        if specified
        :return: None
        """
        if self.flatten:
            self.data["data"] = self.data["data"].reshape(self.data["data"].shape[0], -1)

        self.data["data"] = torch.tensor(self.data["data"], dtype=torch.float32) if self.use_torch \
            else np.float32(self.data["data"])
        avg_px_val = 120.70756512369792 if self.cifar_type == 10 else 121.936059453125
        stddev_px_val = 64.15007609994316 if self.cifar_type == 10 else 68.38895681157001
        self.data["data"] = normalize(self.data["data"], norm_type=self.image_norm_type,
                                      avg=avg_px_val, stddev=stddev_px_val,
                                      max_val=255)  # max value of pixels in CIFAR10 and CIFAR100 images

        new = preprocess_labels(self.data["labels"], preprocessing_type=self.label_preprocessing)
        self.data["labels"] = torch.tensor(new, dtype=torch.float32) if self.use_torch else np.float32(new)

        if self.use_torch:
            self.data["data"] = self.data["data"].to(device=self.device)
            self.data["labels"] = self.data["labels"].to(device=self.device)

    def partition_data(self):
        """
        partitions the CIFAR data sets according to the classes in self.classes
        :return (dict): contains the samples and labels corresponding to the specified classes
        """
        current_data = {}

        # get indices corresponding to the classes
        correct_rows = np.ones(self.data['data'].shape[0], dtype=bool)
        if self.classes is not None:
            correct_rows = np.in1d(self.integer_labels, self.classes)

        # store samples and labels in the new dictionary
        current_data['data'] = self.data['data'][correct_rows, :, :, :]
        if self.label_preprocessing == "one-hot":
            current_data['labels'] = self.data['labels'][correct_rows][:,self.classes]
        else:
            current_data['labels'] = self.data['labels'][correct_rows]
        return current_data

    def select_new_partition(self, new_classes):
        """
        Generates a new partition of the CIFAR-100 data set based on the classes in new_classes
        :param new_classes: (list) the classes cof the new partition
        :return: None
        """
        self.classes = np.array(new_classes, dtype=np.int32)
        self.current_data = self.partition_data()

    def __len__(self):
        """
        :return (int): number of saples in the partitioned data set
        """
        return self.current_data['data'].shape[0]

    def __getitem__(self, idx):
        """
        :param idx (int): valid index from the data set
        :return (dict): the "image" and corresponding "label"
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.current_data['data'][idx]
        label = self.current_data['labels'][idx]    # one hot labels
        # label = self.current_data['fine_labels'][idx]     # regular labels
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def set_transformation(self, new_transformation):
        self.transform = new_transformation