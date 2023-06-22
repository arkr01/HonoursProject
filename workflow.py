"""
    Defines learning procedures - hyperparameters, training/testing/saving functions, setup/config, etc.

    Author: Adrian Rahul Kamal Rajkamal
"""
from os import mkdir, environ
from os.path import exists

from numpy import log10
from torch.utils.data import DataLoader

from datasets import *

MODELS_FOLDER = root_dir + '/Models/'
LOSS_METRICS_FOLDER = root_dir + '/Losses_Metrics/'
PLOTS_FOLDER = root_dir + '/Plots/'

# Set device and reproducibility configurations as per below:
# https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


class Workflow:
    """ Set up general workflow for hyperparameters, configuration constants, training, testing, and saving. """

    def __init__(self, training_set, test_set, num_epochs=int(1e7), grad_norm_tol=1e-8, lr=3e-4, compare_invex=True,
                 invex_val=1e-2, compare_l2=False, l2_val=1e-2, compare_dropout=False, compare_batch_norm=False,
                 compare_data_aug=False, sgd=True, batch_size=64):
        """
        Set up necessary constants and variables for all experiments.

        :param training_set: training dataset
        :param test_set: test dataset
        :param num_epochs: number of epochs for training/testing
        :param grad_norm_tol: tolerance value to check L2 gradient norm convergence
        :param lr: learning rate
        :param compare_invex: True if comparing invex regularisation, False otherwise
        :param invex_val: invex regularisation lambda value
        :param compare_l2: True if comparing L2 regularisation, False otherwise
        :param l2_val: L2 regularisation lambda value
        :param compare_dropout: True if comparing dropout, False otherwise
        :param compare_batch_norm: True if comparing batch normalisation, False otherwise
        :param compare_data_aug: True if comparing data augmentation, False otherwise
        :param sgd: True if performing SGD, False if performing pure GD
        :param batch_size: Batch size (length of training dataset if sgd == False)
        """
        self.training_set = training_set
        self.test_set = test_set

        self.num_epochs = num_epochs
        self.grad_norm_tol = grad_norm_tol
        self.lr = lr

        self.compare_invex = compare_invex
        self.compare_l2 = compare_l2
        self.compare_dropout = compare_dropout
        self.compare_batch_norm = compare_batch_norm
        self.compare_data_aug = compare_data_aug

        self.invex_val = invex_val
        self.invex_param = self.invex_val * self.compare_invex
        self.l2_val = l2_val
        self.l2_param = self.l2_val * self.compare_l2

        self.sgd = sgd
        self.batch_size = batch_size if self.sgd else len(self.training_set)

        self.model_invex = "_invex" if self.compare_invex else ""
        self.model_l2 = "_l2" if self.compare_l2 else ""
        self.model_dropout = "_dropout" if self.compare_dropout else ""
        self.model_batch_norm = "_batch_norm" if self.compare_batch_norm else ""
        self.model_data_aug = "_data_aug" if self.compare_data_aug else ""

        choices = self.model_invex + self.model_l2 + self.model_dropout + self.model_batch_norm + self.model_data_aug
        self.model_config = "with" + choices if len(choices) else "unregularised"

        self.epochs_to_plot = torch.logspace(0, log10(self.num_epochs), 100).long().unique() - 1
        self.avg_training_losses_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float).to(device)
        self.avg_test_losses_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float).to(device)
        self.plot_idx = 0

        self.training_loader = DataLoader(training_set, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size)

    def train(self, model, loss_fn, optimizer, epoch, reconstruction=False):
        size = len(self.training_loader.dataset)
        num_batches = len(self.training_loader)
        model.train()

        print(f"Epoch {epoch + 1}\n-------------------------------")
        total_loss = 0
        for batch, (examples, targets) in enumerate(self.training_loader):
            examples, targets = examples.to(device), targets.to(device)

            # Compute prediction error
            model.set_batch_idx(batch)
            loss = calculate_loss(model(examples), examples, targets, loss_fn, reconstruction)
            total_loss += loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(examples)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if epoch in self.epochs_to_plot:
            self.avg_training_losses_to_plot[self.plot_idx] = total_loss / num_batches

        if self.check_grad_convergence(model):
            print(f"Training converged after {epoch} epochs.")
            return True
        return False

    def check_grad_convergence(self, model):
        """
        Check training convergence via L2 gradient norm.

        :param model: model being trained
        :return: True if converged, False otherwise.
        """
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm **= 0.5
        print("Current grad norm:", grad_norm)
        return grad_norm <= self.grad_norm_tol

    def test(self, model, loss_fn, epoch, reconstruction=False):
        num_examples = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for examples, targets in self.test_loader:
                examples, targets = examples.to(device), targets.to(device)
                prediction = model(examples)
                test_loss += calculate_loss(prediction, examples, targets, loss_fn, reconstruction).item()
                if not reconstruction:
                    correct += (prediction.argmax(1) == targets).type(torch.float).sum().item()
        test_loss /= num_batches

        # Save test loss (only update plot_idx here as test is always called after train)
        if epoch in self.epochs_to_plot:
            self.avg_test_losses_to_plot[self.plot_idx] = test_loss
            self.plot_idx += 1

        correct /= num_examples
        if not reconstruction:
            print(f"Test Accuracy: {(100 * correct):>0.1f}%")
        print(f"Test loss (avg): {test_loss:>8f}\n")

    def truncate_losses_to_plot(self):
        """
        Truncate train/test losses to plot to last recorded loss values (i.e. get rid of 0 entries for loss values
        that were never computed due to early convergence)
        :return: None
        """
        if not self.avg_training_losses_to_plot[self.plot_idx]:
            self.plot_idx -= 1
        self.epochs_to_plot = torch.narrow(self.epochs_to_plot, 0, 0, self.plot_idx)
        self.avg_training_losses_to_plot = torch.narrow(self.avg_training_losses_to_plot, 0, 0, self.plot_idx)
        self.avg_test_losses_to_plot = torch.narrow(self.avg_test_losses_to_plot, 0, 0, self.plot_idx)

    def save(self, model, model_name):
        """
        Creates necessary directories and saves all models/losses/metrics for plotting and analysis.

        :param model: Trained model to save
        :param model_name: Name of trained model
        :return: None
        """
        # Create necessary directories if they do not exist
        if not exists(MODELS_FOLDER):
            mkdir(MODELS_FOLDER)
        if not exists(MODELS_FOLDER + model_name):
            mkdir(MODELS_FOLDER + model_name)
        if not exists(LOSS_METRICS_FOLDER):
            mkdir(LOSS_METRICS_FOLDER)
        if not exists(LOSS_METRICS_FOLDER + model_name):
            mkdir(LOSS_METRICS_FOLDER + model_name)
            mkdir(LOSS_METRICS_FOLDER + model_name + '/Train/')
            mkdir(LOSS_METRICS_FOLDER + model_name + '/Test/')
        if not exists(PLOTS_FOLDER):
            mkdir(PLOTS_FOLDER)
        if not exists(PLOTS_FOLDER + model_name):
            mkdir(PLOTS_FOLDER + model_name)

        # Save model state dict, avg train/test losses (to plot), and parameters
        model_type_filename = f"{model_name}/{self.model_config}"
        torch.save(model.state_dict(), f"{MODELS_FOLDER}{model_type_filename}.pth")
        torch.save(self.epochs_to_plot, f"{LOSS_METRICS_FOLDER}{model_name}/epochs_to_plot.pth")
        torch.save(self.avg_training_losses_to_plot,
                   f"{LOSS_METRICS_FOLDER}{model_name}/Train/{self.model_config}_loss.pth")
        torch.save(self.avg_test_losses_to_plot, f"{LOSS_METRICS_FOLDER}{model_name}/Test/{self.model_config}_loss.pth")

        # Get learned parameters (excluding p variables) and convert into single tensor - for comparison
        parameters = [parameter.data.flatten() for parameter in model.parameters()]

        # Remove p variables if they exist
        num_batches = len(self.training_loader)
        parameters_no_p = parameters[:-num_batches] if self.compare_invex else parameters
        torch.save(torch.cat(parameters_no_p), f"{LOSS_METRICS_FOLDER}{model_type_filename}_parameters.pth")

        print(f"Saved PyTorch Model State and training losses for {model_type_filename}")


def calculate_loss(model_output, examples, targets, loss_fn, reconstruction):
    kl_divergence = 0
    if reconstruction:
        prediction, mu, logvar = model_output
        kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=1), dim=0)
    else:
        prediction = model_output
    return loss_fn(prediction, examples if reconstruction else targets) + kl_divergence
