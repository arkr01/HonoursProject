"""
    Loads and preprocesses all data.

    Author: Adrian Rahul Kamal Rajkamal
"""
import torch
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor


# Get datasets and set up data loaders
fashion_training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
fashion_test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

cifar10_training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())
cifar10_test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=ToTensor())

cifar100_training_data = datasets.CIFAR100(root="data", train=True, download=True, transform=ToTensor())
cifar100_test_data = datasets.CIFAR100(root="data", train=False, download=True, transform=ToTensor())

# Get dataset information
fashion_img_length = fashion_training_data[0][0].shape[1]
fashion_classes = fashion_training_data.classes
num_fashion_classes = len(fashion_classes)

# Create subsets of train/test datasets
NUM_PER_CLASS_TRAIN_FASHION = 100  # Specify how many training examples per class to include in subset
NUM_PER_CLASS_TEST_FASHION = 20  # Specify how many test examples per class to include in subset
fashion_train_subset_idx = torch.zeros(0, dtype=torch.long)
fashion_test_subset_idx = torch.zeros_like(fashion_train_subset_idx)

for class_num in range(num_fashion_classes):
    # Get first NUM_PER_CLASS indices for class class_num (e.g. get first 10 indices for class 0)
    fashion_train_idx_to_add = (fashion_training_data.targets ==
                                class_num).nonzero().flatten()[:NUM_PER_CLASS_TRAIN_FASHION]
    fashion_test_idx_to_add = (fashion_test_data.targets == class_num).nonzero().flatten()[:NUM_PER_CLASS_TEST_FASHION]

    # Concatenate to list of indices to subset train/test datasets
    fashion_train_subset_idx = torch.cat((fashion_train_subset_idx, fashion_train_idx_to_add))
    fashion_test_subset_idx = torch.cat((fashion_test_subset_idx, fashion_test_idx_to_add))

fashion_training_data_subset = Subset(fashion_training_data, fashion_train_subset_idx)
fashion_test_data_subset = Subset(fashion_test_data, fashion_test_subset_idx)
