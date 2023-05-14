"""
    Model training/testing/saving

    Author: Adrian Rahul Kamal Rajkamal
"""
from torch.utils.data import DataLoader

from modules import *
from dataset import *

""" Hyperparameters and other constants """
NUM_EPOCHS = 5000
LEARNING_RATE = 1e-3

# Select what to compare here
COMPARING_INVEX = True  # Set to true when comparing Invex Regularisation effects
COMPARING_L2_REG = False  # Set to true when comparing L2-Regularisation effects
COMPARING_DROPOUT = False  # Set to true when comparing Dropout effects
COMPARING_BATCH_NORM = False  # Set to true when comparing Batch Normalisation effects
COMPARING_DATA_AUGMENTATION = False  # Set to true when comparing Data Augmentation effects

# Name model results based on above selection
MODEL_INVEX = "_invex" if COMPARING_INVEX else ""
MODEL_L2_REG = "_l2" if COMPARING_L2_REG else ""
MODEL_DROPOUT = "_dropout" if COMPARING_DROPOUT else ""
MODEL_BATCH_NORM = "_batch_norm" if COMPARING_BATCH_NORM else ""
MODEL_DATA_AUGMENTATION = "_data_aug" if COMPARING_DATA_AUGMENTATION else ""
REGULARISATION_CHOICES = MODEL_INVEX + MODEL_L2_REG + MODEL_DROPOUT + MODEL_BATCH_NORM + MODEL_DATA_AUGMENTATION
MODEL_CONFIG = "_with" + REGULARISATION_CHOICES if len(REGULARISATION_CHOICES) else "_unregularised"

# Regularisation hyperparameter values
L2_PARAM = 1e-2 * COMPARING_L2_REG
INVEX_LAMBDA = 1e-2 * COMPARING_INVEX

""" Train/Test setup (Validation TBD) """

# Dataloaders and batch size
SGD = False  # Set to true if we want SGD instead of pure GD (GD == SGD without batching)
BATCH_SIZE = 64 if SGD else len(training_data_subset)
train_dataloader = DataLoader(training_data_subset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data_subset, batch_size=BATCH_SIZE)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (examples, targets) in enumerate(dataloader):
        examples, targets = examples.to(device), targets.to(device)

        # Compute prediction error
        model.set_batch_idx(batch)
        prediction = model(examples)
        loss = loss_fn(prediction, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(examples)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_examples = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for examples, targets in dataloader:
            examples, targets = examples.to(device), targets.to(device)
            prediction = model(examples)
            test_loss += loss_fn(prediction, targets).item()
            correct += (prediction.argmax(1) == targets).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= num_examples
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # Define model(s)
    logistic_model = MultinomialLogisticRegression(input_dim=img_length, num_classes=num_classes)
    logistic_model_name = f"{logistic_model=}".split('=')[0]  # Gives name of model variable!
    print(logistic_model)

    logistic_model = ModuleWrapper(logistic_model, lamda=INVEX_LAMBDA)
    logistic_model.init_ps(train_dataloader=train_dataloader)
    logistic_model = logistic_model.to(device)

    cross_entropy = nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(logistic_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_PARAM)

    print("\nUsing", device, "\n")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dataloader, logistic_model, cross_entropy, sgd)
        test(test_dataloader, logistic_model, cross_entropy)

    # Model saving
    saved_filename = f"{logistic_model_name}{MODEL_CONFIG}.pth"
    torch.save(logistic_model.state_dict(), saved_filename)
    print(f"Saved PyTorch Model State to {saved_filename}")
