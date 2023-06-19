"""
    Model training/testing/saving

    Author: Adrian Rahul Kamal Rajkamal
"""
from os import mkdir
from os.path import exists

from torch.utils.data import DataLoader

from modules import *
from datasets import *

""" Hyperparameters and other constants """
NUM_EPOCHS = 5000  # int(1e7)  # Large due to training until convergence of gradient norm
LEARNING_RATE = 3e-4

# Select what to compare here
COMPARING_INVEX = True  # Set to true when comparing Invex Regularisation effects
COMPARING_L2_REG = False  # Set to true when comparing L2-Regularisation effects
COMPARING_DROPOUT = False  # Set to true when comparing Dropout effects
COMPARING_BATCH_NORM = False  # Set to true when comparing Batch Normalisation effects
COMPARING_DATA_AUGMENTATION = False  # Set to true when comparing Data Augmentation effects

# Name model results based on above selection (for saving)
MODEL_INVEX = "_invex" if COMPARING_INVEX else ""
MODEL_L2_REG = "_l2" if COMPARING_L2_REG else ""
MODEL_DROPOUT = "_dropout" if COMPARING_DROPOUT else ""
MODEL_BATCH_NORM = "_batch_norm" if COMPARING_BATCH_NORM else ""
MODEL_DATA_AUGMENTATION = "_data_aug" if COMPARING_DATA_AUGMENTATION else ""
REGULARISATION_CHOICES = MODEL_INVEX + MODEL_L2_REG + MODEL_DROPOUT + MODEL_BATCH_NORM + MODEL_DATA_AUGMENTATION
MODEL_CONFIG = "_with" + REGULARISATION_CHOICES if len(REGULARISATION_CHOICES) else "_unregularised"
MODELS_FOLDER = './Models/'
LOSS_METRICS_FOLDER = './Losses_Metrics/'
TRAIN_FOLDER = LOSS_METRICS_FOLDER + 'Train/'
TEST_FOLDER = LOSS_METRICS_FOLDER + 'Test/'

# Regularisation hyperparameter values
INVEX_VAL = 1e-2
L2_VAL = 1e-2
L2_PARAM = L2_VAL * COMPARING_L2_REG
INVEX_LAMBDA = INVEX_VAL * COMPARING_INVEX

""" Train/Test setup (Validation TBD) """

# Dataloaders and batch size
SGD = True  # Set to true if we want SGD instead of pure GD (GD == SGD without batching)
BATCH_SIZE = 64 if SGD else len(fashion_training_data_subset)

fashion_train_dataloader = DataLoader(fashion_training_data_subset, batch_size=BATCH_SIZE)
fashion_test_dataloader = DataLoader(fashion_test_data_subset, batch_size=BATCH_SIZE)

cifar10_train_dataloader = DataLoader(cifar10_training_data, batch_size=BATCH_SIZE)
cifar10_test_dataloader = DataLoader(cifar10_test_data, batch_size=BATCH_SIZE)

cifar100_train_dataloader = DataLoader(cifar100_training_data, batch_size=BATCH_SIZE)
cifar100_test_dataloader = DataLoader(cifar100_test_data, batch_size=BATCH_SIZE)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Plotting setup
NUM_EPOCHS_TO_SKIP = 10
epochs_to_plot = [_ for _ in range(0, NUM_EPOCHS + 1, NUM_EPOCHS_TO_SKIP)]
training_loss_to_plot = torch.zeros_like(torch.FloatTensor(epochs_to_plot)).to(device)
test_loss_to_plot = torch.zeros_like(torch.FloatTensor(epochs_to_plot)).to(device)


def train(dataloader, model, loss_fn, optimizer, epoch_to_plot, reconstruction=False):
    size = len(dataloader.dataset)
    loss_plot_idx = 0
    model.train()
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
        if epoch_to_plot in epochs_to_plot and not batch:
            training_loss_to_plot[loss_plot_idx] += loss
            loss_plot_idx += 1

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(examples)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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


if __name__ == '__main__':
    # Define model(s)
    # logistic_model = MultinomialLogisticRegression(input_dim=fashion_img_length, num_classes=num_fashion_classes)
    # logistic_model_name = f"{logistic_model=}".split('=')[0]  # Gives name of model variable!
    # print(logistic_model)

    vae_model = VAE(input_dim=cifar_img_shape[1], num_input_channels=cifar_img_shape[0])
    vae_model_name = f"{vae_model=}".split('=')[0]  # Gives name of model variable!
    print(vae_model)

    # logistic_model = ModuleWrapper(logistic_model, lamda=INVEX_LAMBDA)
    # logistic_model.init_ps(train_dataloader=fashion_train_dataloader)
    # logistic_model = logistic_model.to(device)

    vae_model = ModuleWrapper(vae_model, lamda=INVEX_LAMBDA, reconstruction=True)
    vae_model.init_ps(train_dataloader=cifar10_train_dataloader)
    vae_model = vae_model.to(device)

    cross_entropy = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    sgd = torch.optim.SGD(vae_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_PARAM)

    print("\nUsing", device, "\n")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(cifar10_train_dataloader, vae_model, mse, sgd, epoch, reconstruction=True)
        # test(cifar10_test_dataloader, vae_model, mse, epoch, reconstruction=True)
        # train(fashion_train_dataloader, logistic_model, cross_entropy, sgd, epoch)
        # test(fashion_test_dataloader, logistic_model, cross_entropy, epoch)

        # Check convergence
        grad_norm = 0
        for p in vae_model.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        print("Current grad norm:", grad_norm)

        if grad_norm < 1e-8:
            print("Training converged.")
            break

    # Model and loss/metrics saving
    if not exists(MODELS_FOLDER):
        mkdir(MODELS_FOLDER)
    if not exists(LOSS_METRICS_FOLDER):
        mkdir(LOSS_METRICS_FOLDER)
        mkdir(TRAIN_FOLDER)
        mkdir(TEST_FOLDER)

    model_type_filename = f"{vae_model_name}{MODEL_CONFIG}"
    torch.save(vae_model.state_dict(), f"{MODELS_FOLDER}{model_type_filename}.pth")
    torch.save(training_loss_to_plot, f"{TRAIN_FOLDER}{model_type_filename}_loss.pth")
    torch.save(test_loss_to_plot, f"{TEST_FOLDER}{model_type_filename}_loss.pth")

    # Get learned parameters (excluding p variables) and convert into single tensor - for comparison
    parameters = [parameter.data.flatten() for parameter in vae_model.parameters()]
    parameters_no_p = parameters[:-1] if COMPARING_INVEX else parameters  # Remove p variables if they exist
    parameters_no_p_cat = torch.cat(parameters_no_p)  # Convert all parameters into one tensor
    torch.save(parameters_no_p_cat, f"{LOSS_METRICS_FOLDER}{model_type_filename}_parameters.pth")

    print(f"Saved PyTorch Model State and training losses for {model_type_filename}")
