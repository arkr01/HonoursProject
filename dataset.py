"""
    Loads and preprocesses all data.

    Author: Adrian Rahul Kamal Rajkamal
"""
import torch
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor


# Get datasets and set up data loaders
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

# Get dataset information
img_length = training_data[0][0].shape[1]
classes = training_data.classes
num_classes = len(classes)

# Create subsets of train/test datasets
NUM_PER_CLASS_TRAIN = 100  # Specify how many training examples per class to include in subset
NUM_PER_CLASS_TEST = 20  # Specify how many test examples per class to include in subset
train_subset_idx = torch.zeros(0, dtype=torch.long)
test_subset_idx = torch.zeros_like(train_subset_idx)

for class_num in range(num_classes):
    # Get first NUM_PER_CLASS indices for class class_num (e.g. get first 10 indices for class 0)
    train_idx_to_add = (training_data.targets == class_num).nonzero().flatten()[:NUM_PER_CLASS_TRAIN]
    test_idx_to_add = (test_data.targets == class_num).nonzero().flatten()[:NUM_PER_CLASS_TEST]

    # Concatenate to list of indices to subset train/test datasets
    train_subset_idx = torch.cat((train_subset_idx, train_idx_to_add))
    test_subset_idx = torch.cat((test_subset_idx, test_idx_to_add))

training_data_subset = Subset(training_data, train_subset_idx)
test_data_subset = Subset(test_data, test_subset_idx)
