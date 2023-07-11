"""
    Defines learning procedures - hyperparameters, training/testing/saving functions, setup/config, etc.

    Author: Adrian Rahul Kamal Rajkamal
"""
from os import mkdir, environ
from os.path import exists
from math import inf

from numpy import log10
from torch.utils.data import DataLoader

from datasets import *

MODELS_FOLDER = root_dir + '/Models_rgp/'
LOSS_METRICS_FOLDER = root_dir + '/Losses_Metrics_rgp/'
PLOTS_RESULTS_FOLDER = root_dir + '/Plots_Results/'

# Set device and reproducibility configurations as per below:
# https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


class Workflow:
    """ Set up general workflow for hyperparameters, configuration constants, training, testing, and saving. """

    def __init__(self, training_set, test_set, num_epochs=int(1e6), grad_norm_tol=1e-8, lr=None, compare_invex=True,
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

        # If learning rate is left unspecified, set to optimal (upper bound) learning rate (from MATH3204)
        if self.lr is None:
            training_set_transformed = torch.stack(list(zip(*self.training_set))[0]).squeeze()
            training_set_transformed_matrix = training_set_transformed.reshape(len(training_set_transformed), -1)

            training_set_spectral_norm = torch.linalg.matrix_norm(training_set_transformed_matrix, 2).item()
            self.lr = 1 / (training_set_spectral_norm ** 2 / 4 + self.l2_val)

        self.sgd = sgd
        self.batch_size = batch_size if self.sgd else len(self.training_set)

        model_invex = "_invex" if self.compare_invex else ""
        model_l2 = "_l2" if self.compare_l2 else ""
        model_dropout = "_dropout" if self.compare_dropout else ""
        model_batch_norm = "_batch_norm" if self.compare_batch_norm else ""
        model_data_aug = "_data_aug" if self.compare_data_aug else ""
        model_gd = "_gd" if not self.sgd else ""
        model_lr = f"_lr{self.lr}" if lr is not None else ""
        model_lambda = f"_lambda{invex_val}" if invex_val != 1e-2 else ""

        choices = model_invex + model_l2 + model_dropout + model_batch_norm + model_data_aug + model_gd + model_lr
        choices += model_lambda
        self.model_config = "with" + choices if len(choices) else "unregularised"

        self.epochs_to_plot = torch.logspace(0, log10(self.num_epochs), 100).long().unique() - 1
        self.avg_training_losses_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float).to(device)
        self.avg_test_losses_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float).to(device)
        self.avg_training_objectives_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float).to(device)
        self.avg_test_objectives_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float).to(device)
        self.grad_l_inf_norm_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float).to(device)
        self.plot_idx = 0

        self.training_loader = DataLoader(training_set, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size)
        self.num_train_batches = len(self.training_loader)

    def train(self, model, loss_fn, optimizer, epoch, reconstruction=False):
        num_examples = len(self.training_loader.dataset)
        model.train()

        total_loss = total_objective = correct = 0
        print(f"Epoch {epoch + 1}\n-------------------------------")
        for batch, (examples, targets) in enumerate(self.training_loader):
            examples, targets = examples.to(device), targets.to(device)
            model.set_batch_idx(batch)
            prediction = model(examples)
            loss, objective = self.calculate_loss_and_objective(model, prediction, examples, targets, loss_fn,
                                                                reconstruction)
            if not reconstruction:
                correct += (prediction.argmax(1) == targets).type(torch.float).sum().item()

            # Backpropagation
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

            total_loss += loss.item()
            total_objective += objective.item()
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(examples)
                print(f"loss: {loss:>7f}  [{current:>5d}/{num_examples:>5d}]")

        # Calculate average metrics
        avg_loss = total_loss / self.num_train_batches
        avg_objective = total_objective / self.num_train_batches
        correct /= num_examples

        # Print and save
        print(f"\nTrain loss (avg): {avg_loss:>8f}")
        if not reconstruction:
            print(f"Train Accuracy: {(100 * correct):>0.1f}%")

        if epoch in self.epochs_to_plot:
            self.avg_training_losses_to_plot[self.plot_idx] = avg_loss
            self.avg_training_objectives_to_plot[self.plot_idx] = avg_objective

        if self.check_grad_convergence(model, epoch):
            print(f"Training converged after {epoch} epochs.")
            return True
        return False

    def check_grad_convergence(self, model, epoch):
        """
        Check training convergence via L_infinity gradient norm.

        :param model: model being trained
        :param epoch: current epoch (check if grad norm should be plotted)
        :return: True if converged, False otherwise.
        """
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().norm(inf).item()
                grad_norm = max(grad_norm, param_norm)
        print(f"Current grad (L_infinity) norm: {grad_norm:>8f}")
        if epoch in self.epochs_to_plot:
            self.grad_l_inf_norm_to_plot[self.plot_idx] = grad_norm
        return grad_norm <= self.grad_norm_tol

    def test(self, model, loss_fn, epoch, reconstruction=False):
        test_batches = len(self.test_loader)
        num_examples = len(self.test_loader.dataset)
        model.eval()
        test_loss = test_objective = correct = 0
        with torch.no_grad():
            for examples, targets in self.test_loader:
                examples, targets = examples.to(device), targets.to(device)
                prediction = model(examples)
                loss, objective = self.calculate_loss_and_objective(model, prediction, examples, targets, loss_fn,
                                                                    reconstruction)
                test_loss += loss.item()
                test_objective += objective.item()
                if not reconstruction:
                    correct += (prediction.argmax(1) == targets).type(torch.float).sum().item()
        test_loss /= test_batches
        test_objective /= test_batches
        correct /= num_examples

        # Save test loss (only update plot_idx here as test is always called after train)
        if epoch in self.epochs_to_plot:
            self.avg_test_losses_to_plot[self.plot_idx] = test_loss
            self.avg_test_objectives_to_plot[self.plot_idx] = test_objective
            self.plot_idx += 1

        print(f"Test loss (avg): {test_loss:>8f}" + ("\n" if reconstruction else ""))
        if not reconstruction:
            print(f"Test Accuracy: {(100 * correct):>0.1f}%\n")

    def calculate_loss_and_objective(self, model, prediction, examples, targets, loss_fn, reconstruction):
        # TODO generalise for other regularisation methods and write docstring
        invex_objective = calculate_loss(prediction, examples, targets, loss_fn, reconstruction)

        # minor code optimisation: if no invex regularisation, invex loss == invex objective. Also, exclude p
        # variables when performing L2 regularisation
        loss = invex_objective if not self.compare_invex else calculate_loss(model.module(examples), examples,
                                                                             targets, loss_fn, reconstruction)
        parameters_to_consider = list(model.parameters())[:-self.num_train_batches] if self.compare_invex else \
            model.parameters()
        objective = invex_objective + (0 if not self.compare_l2
                                       else 0.5 * self.l2_param * sum(p.pow(2.0).sum() for p in parameters_to_consider))
        return loss, objective

    def truncate_metrics_to_plot(self):
        """
        Truncate train/test metrics to plot to last recorded values (i.e. get rid of initial 0 entries for metrics that
        were never computed due to early convergence)
        :return: None
        """
        if not self.avg_training_losses_to_plot[self.plot_idx]:
            self.plot_idx -= 1
        self.epochs_to_plot = torch.narrow(self.epochs_to_plot, 0, 0, self.plot_idx)
        self.avg_training_losses_to_plot = torch.narrow(self.avg_training_losses_to_plot, 0, 0, self.plot_idx)
        self.avg_test_losses_to_plot = torch.narrow(self.avg_test_losses_to_plot, 0, 0, self.plot_idx)
        self.avg_training_objectives_to_plot = torch.narrow(self.avg_training_objectives_to_plot, 0, 0, self.plot_idx)
        self.avg_test_objectives_to_plot = torch.narrow(self.avg_test_objectives_to_plot, 0, 0, self.plot_idx)
        self.grad_l_inf_norm_to_plot = torch.narrow(self.grad_l_inf_norm_to_plot, 0, 0, self.plot_idx)

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
        if not exists(PLOTS_RESULTS_FOLDER):
            mkdir(PLOTS_RESULTS_FOLDER)
        if not exists(PLOTS_RESULTS_FOLDER + model_name):
            mkdir(PLOTS_RESULTS_FOLDER + model_name)
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/InfNormDiffs')
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Gradient Norm')

            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Train')
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Train/Loss')
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Train/Objective')
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Train/Both')

            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Test')
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Test/Loss')
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Test/Objective')
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Test/Both')

        # Save model state dict, L2 gradient norm, avg train/test losses and objectives (to plot), and parameters
        model_type_filename = f"{model_name}/{self.model_config}"
        torch.save(model.state_dict(), f"{MODELS_FOLDER}{model_type_filename}.pth")

        torch.save(self.epochs_to_plot, f"{LOSS_METRICS_FOLDER}{model_name}/epochs_to_plot.pth")
        torch.save(self.grad_l_inf_norm_to_plot, f"{LOSS_METRICS_FOLDER}{model_type_filename}_grad_norm.pth")

        torch.save(self.avg_training_losses_to_plot,
                   f"{LOSS_METRICS_FOLDER}{model_name}/Train/{self.model_config}_loss.pth")
        torch.save(self.avg_test_losses_to_plot,
                   f"{LOSS_METRICS_FOLDER}{model_name}/Test/{self.model_config}_loss.pth")

        torch.save(self.avg_training_objectives_to_plot,
                   f"{LOSS_METRICS_FOLDER}{model_name}/Train/{self.model_config}_objective.pth")
        torch.save(self.avg_test_objectives_to_plot,
                   f"{LOSS_METRICS_FOLDER}{model_name}/Test/{self.model_config}_objective.pth")

        # Get learned parameters (excluding p variables) and convert into single tensor - for comparison
        parameters = [parameter.detach().flatten() for parameter in model.parameters()]

        # Remove p variables if they exist
        parameters_no_p = parameters[:-self.num_train_batches] if self.compare_invex else parameters
        torch.save(torch.cat(parameters_no_p), f"{LOSS_METRICS_FOLDER}{model_type_filename}_parameters.pth")

        print(f"Saved PyTorch Model State and training losses for {model_type_filename}")


def calculate_loss(model_output, examples, targets, loss_fn, reconstruction):
    # TODO change reconstruction parameter to enum once all models have been implemented, and then write docstring
    kl_divergence = 0
    if reconstruction:
        prediction, mu, logvar = model_output
        kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=1), dim=0)
    else:
        prediction = model_output
    return loss_fn(prediction, examples if reconstruction else targets) + kl_divergence
