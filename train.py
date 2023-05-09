import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from modules import *

""" Hyperparameters and other constants """
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# Select what to compare here
COMPARING_INVEX = True  # Set to true when comparing Invex Regularisation effects
COMPARING_L2_REG = False  # Set to true when comparing L2-Regularisation effects
COMPARING_DROPOUT = False  # Set to true when comparing Dropout effects
COMPARING_BATCH_NORM = False  # Set to true when comparing Batch Normalisation effects
COMPARING_DATA_AUGMENTATION = False  # Set to true when comparing Data Augmentation effects

# Regularisation hyperparameter values
WEIGHT_DECAY = 1e-3 * COMPARING_L2_REG
INVEX_LAMBDA = 1e-3 * COMPARING_INVEX

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using " + device)

# Get datasets and set up data loaders
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


# Define model(s)
model = NeuralNetwork()
print(model)
model = ModuleWrapper(model, lamda=INVEX_LAMBDA)
model.init_ps(train_dataloader=train_dataloader)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        model.set_batch_idx(batch)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    model = NeuralNetwork()
    model = ModuleWrapper(model, lamda=INVEX_LAMBDA)
    model.init_ps(train_dataloader=train_dataloader)
    model = model.to(device)
    model.load_state_dict(torch.load("model.pth"))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
