"""
    Loads and preprocesses all data.

    Author: Adrian Rahul Kamal Rajkamal
"""
from os.path import abspath, dirname

import torch
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get datasets - train/test split
root_dir = dirname(abspath(__file__))
data_dir = root_dir + '/data'
fashion_training_data = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=ToTensor())
fashion_test_data = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=ToTensor())

cifar10_training_data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=ToTensor())
cifar10_test_data = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=ToTensor())

cifar100_training_data = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=ToTensor())
cifar100_test_data = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=ToTensor())

# Get dataset information
fashion_img_length = fashion_training_data[0][0].shape[1]
fashion_classes = fashion_training_data.classes
num_fashion_classes = len(fashion_classes)

cifar_img_shape = cifar10_training_data[0][0].shape  # same for cifar10 and cifar100
cifar10_classes = cifar10_training_data.classes
cifar100_classes = cifar100_training_data.classes

# Create subsets of train/test datasets
NUM_PER_CLASS_TRAIN_FASHION = 100  # Specify how many training examples per class to include in subset
NUM_PER_CLASS_TEST_FASHION = 20  # Specify how many test examples per class to include in subset


def get_subset_examples(train_data, test_data, num_classes, num_train, num_test):
    """
    Create subset of dataset with [num_train] examples of each class (for all [num_classes] classes)
    :param train_data: training dataset
    :param test_data: test dataset
    :param num_classes: number of classes in dataset
    :param num_train: number of examples per class to extract from training dataset
    :param num_test: number of examples per class to extract from test dataset
    :return: train and test indices to use via subset
    """
    train_subset_idx = torch.zeros(0, dtype=torch.long)
    test_subset_idx = torch.zeros_like(train_subset_idx)
    for class_num in range(num_classes):
        train_idx_to_add = (train_data.targets == class_num).nonzero().flatten()[:num_train]
        test_idx_to_add = (test_data.targets == class_num).nonzero().flatten()[:num_test]

        # Concatenate to list of indices to subset train/test datasets
        train_subset_idx = torch.cat((train_subset_idx, train_idx_to_add))
        test_subset_idx = torch.cat((test_subset_idx, test_idx_to_add))
    return train_subset_idx, test_subset_idx


# Set up subsets for training/testing
fashion_train_subset_idx, fashion_test_subset_idx = get_subset_examples(fashion_training_data, fashion_test_data,
                                                                        num_fashion_classes,
                                                                        NUM_PER_CLASS_TRAIN_FASHION,
                                                                        NUM_PER_CLASS_TEST_FASHION)
fashion_training_data_subset = Subset(fashion_training_data, fashion_train_subset_idx)
fashion_test_data_subset = Subset(fashion_test_data, fashion_test_subset_idx)
