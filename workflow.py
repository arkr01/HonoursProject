"""
    Defines learning procedures - hyperparameters, training/testing/saving functions, setup/config, etc.

    Author: Adrian Rahul Kamal Rajkamal
"""
from os import mkdir
from os.path import exists
from math import inf

from numpy import log10
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomEqualize, RandomInvert

from datasets import *

LOSS_METRICS_FOLDER = root_dir + '/Losses_Metrics/'
PLOTS_RESULTS_FOLDER = root_dir + '/Plots_Results/'

device = "cuda" if torch.cuda.is_available() else "cpu"


class Workflow:
    """ Set up general workflow for hyperparameters, configuration constants, training, testing, and saving. """

    def __init__(self, training_set, test_set, num_epochs=int(1e6), grad_norm_tol=1e-16, lr=None, compare_invex=False,
                 invex_val=1e-1, invex_p_ones=False, compare_l2=False, l2_val=1e-2, compare_dropout=False,
                 dropout_val=0.5, compare_batch_norm=False, compare_data_aug=False, subset=False, reconstruction=False,
                 least_sq=False, binary_log_reg=False, synthetic=False, sgd=True, batch_size=64, lbfgs=False,
                 zero_init=False, early_converge=False):
        """
        Set up necessary constants and variables for all experiments.

        :param training_set: training dataset
        :param test_set: test dataset
        :param num_epochs: number of epochs for training/testing
        :param grad_norm_tol: tolerance value to check L2 gradient norm convergence
        :param lr: learning rate
        :param compare_invex: True if comparing invex regularisation, False otherwise
        :param invex_val: invex regularisation lambda value
        :param invex_p_ones: True if using scalar invex p multiplied by vector of ones, False if using standard vector p
        :param compare_l2: True if comparing L2 regularisation, False otherwise
        :param l2_val: L2 regularisation lambda value
        :param compare_dropout: True if comparing dropout, False otherwise
        :param dropout_val: Dropout hyperparameter value p
        :param compare_batch_norm: True if comparing batch normalisation, False otherwise
        :param compare_data_aug: True if comparing data augmentation, False otherwise
        :param subset: True if using Subset class for dataset, False otherwise (for data augmentation)
        :param reconstruction: True if performing reconstruction, False otherwise
        :param least_sq: True if performing linear least squares regression, False otherwise
        :param binary_log_reg: True if performing binary logistic regression, False otherwise
        :param synthetic: True if using synthetic data, False otherwise
        :param sgd: True if performing SGD, False if performing pure GD
        :param batch_size: Batch size (length of training dataset if sgd == False)
        :param lbfgs: True if using LBFGS, False otherwise.
        :param zero_init: True for zero initialisation of parameters, False otherwise.
        :param early_converge: True if gradient norm meeting grad_norm_tol should stop training, False otherwise.
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
        self.invex_p_ones = invex_p_ones
        self.l2_val = l2_val
        self.l2_param = self.l2_val * self.compare_l2
        self.dropout_val = dropout_val
        self.dropout_param = self.dropout_val * self.compare_dropout

        self.reconstruction = reconstruction
        self.least_sq = least_sq
        self.binary_log_reg = binary_log_reg
        self.synthetic = synthetic

        # If learning rate is left unspecified, set to optimal (upper bound) learning rate (from MATH3204)
        if self.lr is None:
            training_set_transformed = torch.stack(list(zip(*self.training_set))[0]).squeeze()
            training_set_transformed_matrix = training_set_transformed.reshape(len(training_set_transformed), -1)

            training_set_spectral_norm_sq = torch.linalg.matrix_norm(training_set_transformed_matrix, 2).item() ** 2
            lg_unreg = training_set_spectral_norm_sq * (0.25 if not self.least_sq else 1)
            self.lr = 1 / (lg_unreg + self.l2_val)

        # Ensure only one optimiser is selected
        self.sgd = sgd
        self.lbfgs = lbfgs
        if self.sgd and self.lbfgs:
            raise Exception("Incorrect Optimiser Selected")
        self.zero_init = zero_init
        self.early_converge = early_converge

        self.batch_size = batch_size if self.sgd else len(self.training_set)

        model_invex = "_invex" if self.compare_invex else ""
        model_invex_ones = "_ones" if self.invex_p_ones else ""
        model_l2 = "_l2" if self.compare_l2 else ""
        model_dropout = "_dropout" if self.compare_dropout else ""
        model_batch_norm = "_batch_norm" if self.compare_batch_norm else ""
        model_data_aug = "_data_aug" if self.compare_data_aug else ""
        model_zero_init = "_zero_init" if self.zero_init else ""
        model_optim = "_gd" if not self.sgd and not self.lbfgs else ("_lbfgs" if self.lbfgs else "")
        model_subset = f"_subset_n={len(training_set)}" if subset else ""
        model_lr = f"_lr{self.lr}" if lr is not None else ""
        model_invex_lambda = f"_lambda{invex_val}" if self.compare_invex else ""
        model_l2_lambda = f"_l2lambda{l2_val}" if self.compare_l2 else ""

        choices = model_invex + model_invex_ones + model_l2 + model_dropout + model_batch_norm + model_data_aug
        choices += model_zero_init + model_optim + model_subset + model_lr + model_invex_lambda + model_l2_lambda
        self.model_config = "with" + choices if choices else "unregularised"

        self.epochs_to_plot = torch.logspace(0, log10(self.num_epochs), 100).long().unique() - 1
        self.avg_training_losses_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float64).to(device)
        self.avg_test_losses_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float64).to(device)
        self.avg_training_accuracies_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float64).to(device)
        self.avg_test_accuracies_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float64).to(device)
        self.grad_l_inf_norm_to_plot = torch.zeros_like(self.epochs_to_plot, dtype=torch.float64).to(device)
        self.plot_idx = 0

        if self.compare_data_aug:
            augmentations = Compose([RandomHorizontalFlip(p=0.2), RandomVerticalFlip(p=0.1), RandomEqualize(p=0.1),
                                     RandomInvert(p=0.2), ToTensor(), ConvertImageDtype(torch.float64)])
            if subset:
                training_set.dataset.transform = augmentations
            else:
                training_set.transform = augmentations

        self.training_loader = DataLoader(training_set, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size)
        self.num_train_batches = len(self.training_loader)

    def train(self, model, loss_fn, optimiser, epoch):
        num_examples = len(self.training_loader.dataset)
        model.train()

        total_loss = correct = 0
        print(f"Epoch {epoch + 1}\n-------------------------------")
        for batch, (examples, targets) in enumerate(self.training_loader):
            examples, targets = examples.to(device), targets.to(device)
            model.set_batch_idx(batch)
            loss = 0

            # Handle closure() for LBFGS
            if self.lbfgs:
                # closure() may run several times per epoch, ensure to only calculate accuracy once
                num_closure = 0

                def closure():
                    nonlocal targets, loss, correct, num_closure
                    prediction_ = model(examples)
                    if self.binary_log_reg:
                        targets = targets.unsqueeze(1).to(dtype=torch.float64)
                        prediction_ = torch.clamp(prediction_, min=0.0, max=1.0)  # For numerical issues
                    loss, obj = self.calculate_loss_and_objective(model, prediction_, examples, targets, loss_fn)
                    if not self.reconstruction and not self.least_sq:
                        pred = prediction_.argmax(1) if not self.binary_log_reg else prediction_.round()
                        num_closure += 1
                        if num_closure == 1:
                            correct += (pred == targets).type(torch.float).sum().item()

                    # Backpropagation
                    optimiser.zero_grad()
                    obj.backward()
                    return obj

                optimiser.step(closure)
            else:
                prediction = model(examples)
                if self.binary_log_reg:
                    targets = targets.unsqueeze(1).to(dtype=torch.float64)
                    prediction = torch.clamp(prediction, min=0.0, max=1.0)  # For numerical issues
                loss, objective = self.calculate_loss_and_objective(model, prediction, examples, targets, loss_fn)
                if not self.reconstruction and not self.least_sq:
                    predicted = prediction.argmax(1) if not self.binary_log_reg else prediction.round()
                    correct += (predicted == targets).type(torch.float).sum().item()

                # Backpropagation
                optimiser.zero_grad()
                objective.backward()
                optimiser.step()

            total_loss += loss.item()
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(examples)
                print(f"loss: {loss:>7f}  [{current:>5d}/{num_examples:>5d}]")

        # Calculate average metrics
        avg_loss = total_loss / self.num_train_batches
        correct /= num_examples

        # Print and save
        print(f"\nTrain loss (avg): {avg_loss:>8f}")
        if not self.reconstruction and not self.least_sq:
            print(f"Train Accuracy: {(100 * correct):>0.1f}%")

        if epoch in self.epochs_to_plot:
            self.avg_training_losses_to_plot[self.plot_idx] = avg_loss
            self.avg_training_accuracies_to_plot[self.plot_idx] = correct

        # Minor code optimisation - don't bother calculating gradient convergence if we set a negative tolerance
        if self.grad_norm_tol >= 0 and self.check_grad_convergence(model, epoch):
            print(f"Training converged after {epoch} epochs.")

            # Only return True (and stop training as a result) if we want
            return self.early_converge
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
            if self.synthetic:
                self.plot_idx += 1
        return grad_norm <= self.grad_norm_tol

    def test(self, model, loss_fn, epoch):
        test_batches = len(self.test_loader)
        num_examples = len(self.test_loader.dataset)
        model.eval()
        test_loss = correct = 0
        with torch.no_grad():
            for examples, targets in self.test_loader:
                examples, targets = examples.to(device), targets.to(device)
                prediction = model(examples)
                if self.binary_log_reg:
                    targets = targets.unsqueeze(1).to(dtype=torch.float64)
                    prediction = torch.clamp(prediction, min=0.0, max=1.0)  # For numerical issues
                loss, _ = self.calculate_loss_and_objective(model, prediction, examples, targets, loss_fn)
                test_loss += loss.item()
                if not self.reconstruction and not self.least_sq:
                    predicted = prediction.argmax(1) if not self.binary_log_reg else prediction.round()
                    correct += (predicted == targets).type(torch.float).sum().item()
        test_loss /= test_batches
        correct /= num_examples

        # Save test loss (only update plot_idx here as test is always called after train)
        if epoch in self.epochs_to_plot:
            self.avg_test_losses_to_plot[self.plot_idx] = test_loss
            self.avg_test_accuracies_to_plot[self.plot_idx] = correct
            self.plot_idx += 1

        print(f"Test loss (avg): {test_loss:>8f}" + ("\n" if self.reconstruction else ""))
        if not self.reconstruction:
            print(f"Test Accuracy: {(100 * correct):>0.1f}%\n")

    def calculate_loss_and_objective(self, model, prediction, examples, targets, loss_fn):
        # TODO generalise for other regularisation methods and write docstring
        invex_objective = self.calculate_loss(prediction, examples, targets, loss_fn)

        # minor code optimisation: if no invex regularisation, invex loss == invex objective. Also, exclude p
        # variables when performing L2 regularisation
        loss = invex_objective if not self.compare_invex else self.calculate_loss(model.module(examples), examples,
                                                                                  targets, loss_fn)
        parameters_to_consider = list(model.parameters())[:-self.num_train_batches] if self.compare_invex else \
            model.parameters()
        objective = invex_objective + (0 if not self.compare_l2
                                       else 0.5 * self.l2_param * sum(p.pow(2.0).sum() for p in parameters_to_consider))
        return loss, objective

    def calculate_loss(self, model_output, examples, targets, loss_fn):
        # TODO change self.reconstruction parameter to enum once all models have been implemented, and then write
        #  docstring
        kl_divergence = 0
        if self.reconstruction:
            prediction, mu, logvar = model_output
            kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=1), dim=0)
        else:
            prediction = model_output
        main_loss = loss_fn(prediction, examples if self.reconstruction else targets)
        if self.least_sq:
            main_loss *= 0.5
        return main_loss + kl_divergence

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
        self.grad_l_inf_norm_to_plot = torch.narrow(self.grad_l_inf_norm_to_plot, 0, 0, self.plot_idx)

    def save(self, model, model_name):
        """
        Creates necessary directories and saves all models/losses/metrics for plotting and analysis.

        :param model: Trained model to save
        :param model_name: Name of trained model
        :return: None
        """
        # Create necessary directories if they do not exist
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
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Train/Accuracy')

            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Test')
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Test/Loss')
            mkdir(PLOTS_RESULTS_FOLDER + model_name + '/Test/Accuracy')

        # Save infinity gradient norm, avg train/test losses/accuracies, parameters
        model_type_filename = f"{model_name}/{self.model_config}"

        torch.save(self.epochs_to_plot, f"{LOSS_METRICS_FOLDER}{model_name}/epochs_to_plot.pth")
        torch.save(self.grad_l_inf_norm_to_plot, f"{LOSS_METRICS_FOLDER}{model_type_filename}_grad_norm.pth")

        torch.save(self.avg_training_losses_to_plot,
                   f"{LOSS_METRICS_FOLDER}{model_name}/Train/{self.model_config}_loss.pth")
        torch.save(self.avg_test_losses_to_plot,
                   f"{LOSS_METRICS_FOLDER}{model_name}/Test/{self.model_config}_loss.pth")
        torch.save(self.avg_training_accuracies_to_plot,
                   f"{LOSS_METRICS_FOLDER}{model_name}/Train/{self.model_config}_acc.pth")
        torch.save(self.avg_test_accuracies_to_plot,
                   f"{LOSS_METRICS_FOLDER}{model_name}/Test/{self.model_config}_acc.pth")

        # Get learned parameters (excluding p variables) and convert into single tensor - for comparison
        parameters = [parameter.detach().flatten() for parameter in model.parameters()]

        # Remove p variables if they exist
        parameters_no_p = parameters[:-self.num_train_batches] if self.compare_invex else parameters
        torch.save(torch.cat(parameters_no_p), f"{LOSS_METRICS_FOLDER}{model_type_filename}_parameters.pth")

        print(f"Saved PyTorch Model State and training losses for {model_type_filename}")
