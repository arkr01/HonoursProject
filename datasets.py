"""
    Loads and preprocesses all data.

    Author: Adrian Rahul Kamal Rajkamal
"""
from os import environ
from os.path import abspath, dirname

import torch
from torch.utils.data import Subset, TensorDataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, ConvertImageDtype, Normalize

# Set reproducibility configurations as per below:
# https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# Get datasets - train/test split
root_dir = dirname(abspath(__file__))
data_dir = root_dir + '/data'
img_to_float64 = Compose([ToTensor(), ConvertImageDtype(torch.float64)])
fashion_training_data = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=img_to_float64)
fashion_test_data = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=img_to_float64)

cifar10_training_data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=img_to_float64)
cifar10_test_data = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=img_to_float64)

cifar100_training_data = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=img_to_float64)
cifar100_test_data = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=img_to_float64)


# Synthetic data
synthetic_true_x = torch.randn(100, dtype=torch.float64)
synthetic_data_A = torch.randn(20, 100, dtype=torch.float64)
synthetic_data_b = torch.mv(synthetic_data_A, synthetic_true_x)
synthetic_dataset = TensorDataset(synthetic_data_A, synthetic_data_b)

synthetic_data_b_sigmoid = torch.sigmoid(synthetic_data_b).round()
synthetic_dataset_sigmoid = TensorDataset(synthetic_data_A, synthetic_data_b_sigmoid)

# Get dataset information
fashion_img_length = fashion_training_data[0][0].shape[1]
fashion_classes = fashion_training_data.classes
num_fashion_classes = len(fashion_classes)

cifar_img_shape = cifar10_training_data[0][0].shape  # same for cifar10 and cifar100
cifar10_classes = cifar10_training_data.classes
cifar100_classes = cifar100_training_data.classes

# Normalise CIFAR-10 dataset
cifar_train_std, cifar_train_mean = torch.std_mean(torch.tensor(cifar10_training_data.data / 255), dim=(0, 1, 2))
cifar_test_std, cifar_test_mean = torch.std_mean(torch.tensor(cifar10_test_data.data / 255), dim=(0, 1, 2))

normalise_train = Compose([ToTensor(), Normalize(cifar_train_mean, cifar_train_std), ConvertImageDtype(torch.float64)])
normalise_test = Compose([ToTensor(), Normalize(cifar_test_mean, cifar_test_std), ConvertImageDtype(torch.float64)])

cifar10_training_data.transform = normalise_train
cifar100_test_data.transform = normalise_test

# Create subsets of train/test datasets
NUM_PER_CLASS_TRAIN = 10  # Specify how many training examples per class to include in subset
NUM_PER_CLASS_TEST = 10  # Specify how many test examples per class to include in subset


def get_subset_examples(train_data, test_data, num_classes, num_train, num_test):
    """
    Create train and test subsets with [num_train/num_test] examples of each class (for all [num_classes] classes)
    :param train_data: training dataset
    :param test_data: test dataset
    :param num_classes: number of classes in dataset
    :param num_train: number of examples per class to extract from training dataset
    :param num_test: number of examples per class to extract from test dataset
    :return: train and test subsets
    """
    train_subset_idx = torch.zeros(0, dtype=torch.long)
    test_subset_idx = torch.zeros_like(train_subset_idx)
    if not torch.is_tensor(train_data.targets):
        train_data.targets = torch.tensor(train_data.targets)
        test_data.targets = torch.tensor(test_data.targets)
    for class_num in range(num_classes):
        train_idx_to_add = (train_data.targets == class_num).nonzero().flatten()[:num_train]
        test_idx_to_add = (test_data.targets == class_num).nonzero().flatten()[:num_test]

        # Concatenate to list of indices to subset train/test datasets
        train_subset_idx = torch.cat((train_subset_idx, train_idx_to_add))
        test_subset_idx = torch.cat((test_subset_idx, test_idx_to_add))
    return Subset(train_data, train_subset_idx), Subset(test_data, test_subset_idx)


# Set up subsets for training/testing
fashion_training_subset, fashion_test_subset = get_subset_examples(fashion_training_data, fashion_test_data,
                                                                   num_fashion_classes, NUM_PER_CLASS_TRAIN,
                                                                   NUM_PER_CLASS_TEST)
fashion_training_two_class, fashion_test_two_class = get_subset_examples(fashion_training_data, fashion_test_data, 2,
                                                                         NUM_PER_CLASS_TRAIN,
                                                                         NUM_PER_CLASS_TEST)
cifar10_training_subset, cifar10_test_subset = get_subset_examples(cifar10_training_data, cifar10_test_data, 10,
                                                                   NUM_PER_CLASS_TRAIN,
                                                                   NUM_PER_CLASS_TEST)
