"""
    Defines learning procedures - hyperparameters, training/testing/saving functions, setup/config, etc.

    Author: Adrian Rahul Kamal Rajkamal
"""
from os import mkdir
from os.path import exists

from torch.utils.data import DataLoader

from datasets import *

""" Hyperparameters and other constants """
NUM_EPOCHS = int(1e7)  # Large due to training until convergence of gradient norm
GRAD_NORM_TOL = 1e-8  # Tolerance value for measuring convergence of gradient norm
LEARNING_RATE = 3e-4

COMPARING_INVEX = True
COMPARING_L2_REG = False
COMPARING_DROPOUT = False
COMPARING_BATCH_NORM = False
COMPARING_DATA_AUGMENTATION = False

INVEX_VAL = 1e-2
L2_VAL = 1e-2
L2_PARAM = L2_VAL * COMPARING_L2_REG
INVEX_PARAM = INVEX_VAL * COMPARING_INVEX

# Name model results based on above selection (for saving)
MODEL_INVEX = "_invex"
MODEL_L2_REG = ""
MODEL_DROPOUT = ""
MODEL_BATCH_NORM = ""
MODEL_DATA_AUGMENTATION = ""
REGULARISATION_CHOICES = MODEL_INVEX + MODEL_L2_REG + MODEL_DROPOUT + MODEL_BATCH_NORM + MODEL_DATA_AUGMENTATION
MODEL_CONFIG = "_with" + REGULARISATION_CHOICES

MODELS_FOLDER = './Models/'
LOSS_METRICS_FOLDER = './Losses_Metrics/'
TRAIN_FOLDER = LOSS_METRICS_FOLDER + 'Train/'
TEST_FOLDER = LOSS_METRICS_FOLDER + 'Test/'

# Dataloaders and batch size
SGD = True  # Set to true if we want SGD instead of pure GD (GD == SGD without batching)
BATCH_SIZE = 64

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Plotting setup
NUM_EPOCHS_TO_SKIP = 10
epochs_to_plot = [_ for _ in range(0, NUM_EPOCHS + 1, NUM_EPOCHS_TO_SKIP)]
training_loss_to_plot = torch.zeros_like(torch.FloatTensor(epochs_to_plot)).to(device)
test_loss_to_plot = torch.zeros_like(torch.FloatTensor(epochs_to_plot)).to(device)


def experiment_setup(training_set, test_set, invex=True, l2=False, dropout=False, batch_norm=False, data_aug=False,
                     sgd=True, batch_size=64):
    """
    Set up hyperparameters and other constants for current experiment and return a DataLoader for
    said experiment.

    :param training_set: training dataset
    :param test_set: test dataset
    :param invex: True if applying invex method, False otherwise.
    :param l2: True if applying L2 regularisation, False otherwise.
    :param dropout: True if applying dropout, False otherwise.
    :param batch_norm: True if applying batch normalisation, False otherwise.
    :param data_aug: True if applying data augmentation, False otherwise.
    :param sgd: True if training via SGD, False for pure GD.
    :param batch_size: dataloader batch size
    :return: train and test dataloaders for experiment
    """
    global COMPARING_INVEX, COMPARING_L2_REG, COMPARING_DROPOUT, COMPARING_BATCH_NORM, COMPARING_DATA_AUGMENTATION, \
        L2_PARAM, INVEX_PARAM
    COMPARING_INVEX = invex
    COMPARING_L2_REG = l2
    COMPARING_DROPOUT = dropout
    COMPARING_BATCH_NORM = batch_norm
    COMPARING_DATA_AUGMENTATION = data_aug

    # Name model results based on above selection (for saving)
    global MODEL_INVEX, MODEL_L2_REG, MODEL_DROPOUT, MODEL_BATCH_NORM, MODEL_DATA_AUGMENTATION, \
        REGULARISATION_CHOICES, MODEL_CONFIG
    MODEL_INVEX = "_invex" if COMPARING_INVEX else ""
    MODEL_L2_REG = "_l2" if COMPARING_L2_REG else ""
    MODEL_DROPOUT = "_dropout" if COMPARING_DROPOUT else ""
    MODEL_BATCH_NORM = "_batch_norm" if COMPARING_BATCH_NORM else ""
    MODEL_DATA_AUGMENTATION = "_data_aug" if COMPARING_DATA_AUGMENTATION else ""
    REGULARISATION_CHOICES = MODEL_INVEX + MODEL_L2_REG + MODEL_DROPOUT + MODEL_BATCH_NORM + MODEL_DATA_AUGMENTATION
    MODEL_CONFIG = "_with" + REGULARISATION_CHOICES if len(REGULARISATION_CHOICES) else "_unregularised"

    L2_PARAM = L2_VAL * COMPARING_L2_REG
    INVEX_PARAM = INVEX_VAL * COMPARING_INVEX

    global SGD, BATCH_SIZE
    SGD = sgd
    BATCH_SIZE = batch_size if SGD else len(training_set)

    return DataLoader(training_set, batch_size=BATCH_SIZE), DataLoader(test_set, batch_size=BATCH_SIZE)


def train(dataloader, model, loss_fn, optimizer, reconstruction=False):
    size = len(dataloader.dataset)
    loss_plot_idx = 0
    model.train()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        for batch, (examples, targets) in enumerate(dataloader):
            examples, targets = examples.to(device), targets.to(device)

            # Compute prediction error
            model.set_batch_idx(batch)

            kl_divergence = 0
            if reconstruction:
                prediction, mu, logvar = model(examples)
                kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=1), dim=0)
            else:
                prediction = model(examples)
            loss = loss_fn(prediction, examples if reconstruction else targets) + kl_divergence

            # Save training loss for each epoch (only the first batch)
            if epoch in epochs_to_plot and not batch:
                training_loss_to_plot[loss_plot_idx] += loss
                loss_plot_idx += 1

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(examples)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # Check convergence (via L2 gradient norm)
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm **= 0.5
        print("Current grad norm:", grad_norm)

        if grad_norm <= GRAD_NORM_TOL:
            print(f"Training converged after {epoch} epochs.")
            break


def test(dataloader, model, loss_fn, epoch_to_plot, reconstruction=False):
    num_examples = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_plot_idx = 0
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for examples, targets in dataloader:
            examples, targets = examples.to(device), targets.to(device)
            prediction = model(examples)
            test_loss += loss_fn(prediction, examples if reconstruction else targets).item()
            correct += (prediction.argmax(1) == targets).type(torch.float).sum().item()
    test_loss /= num_batches

    # Save test loss
    if epoch_to_plot in epochs_to_plot:
        test_loss_to_plot[loss_plot_idx] += test_loss
        loss_plot_idx += 1

    correct /= num_examples
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def save(model):
    if not exists(MODELS_FOLDER):
        mkdir(MODELS_FOLDER)
    if not exists(LOSS_METRICS_FOLDER):
        mkdir(LOSS_METRICS_FOLDER)
        mkdir(TRAIN_FOLDER)
        mkdir(TEST_FOLDER)

    model_name = f"{model=}".split('=')[0]  # Gives name of model variable!
    model_type_filename = f"{model_name}{MODEL_CONFIG}"
    torch.save(model.state_dict(), f"{MODELS_FOLDER}{model_type_filename}.pth")
    torch.save(training_loss_to_plot, f"{TRAIN_FOLDER}{model_type_filename}_loss.pth")
    torch.save(test_loss_to_plot, f"{TEST_FOLDER}{model_type_filename}_loss.pth")

    # Get learned parameters (excluding p variables) and convert into single tensor - for comparison
    parameters = [parameter.data.flatten() for parameter in model.parameters()]
    parameters_no_p = parameters[:-1] if COMPARING_INVEX else parameters  # Remove p variables if they exist
    torch.save(torch.cat(parameters_no_p), f"{LOSS_METRICS_FOLDER}{model_type_filename}_parameters.pth")

    print(f"Saved PyTorch Model State and training losses for {model_type_filename}")
