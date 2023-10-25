"""
    Plotting/analysis code with **trained models**

    Comparing regularisation methods.

    Author: Adrian Rahul Kamal Rajkamal
"""
from matplotlib import rcParams
import matplotlib.pyplot as plt
from numpy import log10

import torch

from workflow import LOSS_METRICS_FOLDER, PLOTS_RESULTS_FOLDER

rcParams["text.usetex"] = True  # Allows LaTeX in titles/labels/legends

# What experiment we're plotting
model = "resnet18_model"
lr = 100.0
lambda_1 = 0.1
lambda_2 = 0.01
experiment_name = f"gd_zero_lr{lr}_lambda{lambda_1}_l2lambda{lambda_2}_all_individual"

standard_lr = experiment_name.__contains__("lrst")

unreg_or_gd = "unregularised_" if experiment_name[0] == 's' and standard_lr else "with_"
zero_or_random = "zero_init_" if experiment_name.__contains__("zero") else ""
reg_or_gd = "" if experiment_name[0] == 's' else ("lbfgs_" if experiment_name[0] == 'l' else "gd_")
subset = "subset_n=100_" if reg_or_gd else ""
lr_val = "" if standard_lr else f"lr{lr}_"

config = zero_or_random + reg_or_gd + subset + lr_val
unreg_config = unreg_or_gd + config

if "wie" in LOSS_METRICS_FOLDER or "rgp" in LOSS_METRICS_FOLDER:
    experiment_name = "full_" + experiment_name

model_title = "DDPM" if model == "diffusion_model" else "ResNet18 Classification"
optim_title = " (GD) " if reg_or_gd == "gd_" else (" (LBFGS) " if reg_or_gd == "lbfgs_" else " ")
init_title = " (Zero Initialisation) " if zero_or_random == "zero_init_" else " "

plot_title = rf"{model_title}{init_title}{optim_title}(lr$={lr}$, $\lambda_1={lambda_1}$, $\lambda_2={lambda_2}$)"

# epochs_to_plot = torch.load(LOSS_METRICS_FOLDER + f'{model}/epochs_to_plot.pth').to('cpu')
epochs_to_plot = torch.logspace(0, log10(int(1e4)), 100).long().unique() - 1

# Train/Test Losses
unreg_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/{unreg_config}loss.pth').to('cpu')
unreg_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/{unreg_config}loss.pth').to('cpu')

invex_train_loss = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_invex_{config}lambda{lambda_1}_loss.pth').to('cpu')
invex_test_loss = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/Test/with_invex_{config}lambda{lambda_1}_loss.pth').to('cpu')

invex_ones_train_loss = (torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Train/with_invex_ones_{config}lambda{lambda_1}_loss.pth').to('cpu'))
invex_ones_test_loss = (torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Test/with_invex_ones_{config}lambda{lambda_1}_loss.pth').to('cpu'))

l2_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda{lambda_2}_loss.pth').to('cpu')
l2_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda{lambda_2}_loss.pth').to('cpu')

data_aug_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_data_aug_{config}loss.pth').to('cpu')
data_aug_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_data_aug_{config}loss.pth').to('cpu')

dropout_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_dropout_{config}loss.pth').to('cpu')
dropout_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_dropout_{config}loss.pth').to('cpu')

batch_norm_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_batch_norm_{config}loss.pth').to('cpu')
batch_norm_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_batch_norm_{config}loss.pth').to('cpu')

# Train/Test Accuracies
if model != "diffusion_model":
    unreg_train_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/{unreg_config}acc.pth').to('cpu')
    unreg_test_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/{unreg_config}acc.pth').to('cpu')

    invex_train_acc = torch.load(LOSS_METRICS_FOLDER +
                                 f'{model}/Train/with_invex_{config}lambda{lambda_1}_acc.pth').to('cpu')
    invex_test_acc = torch.load(LOSS_METRICS_FOLDER +
                                f'{model}/Test/with_invex_{config}lambda{lambda_1}_acc.pth').to('cpu')

    invex_ones_train_acc = (torch.load(LOSS_METRICS_FOLDER +
                                       f'{model}/Train/with_invex_ones_{config}lambda{lambda_1}_acc.pth').to('cpu'))
    invex_ones_test_acc = (torch.load(LOSS_METRICS_FOLDER +
                                      f'{model}/Test/with_invex_ones_{config}lambda{lambda_1}_acc.pth').to('cpu'))

    l2_train_acc = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_l2_{config}l2lambda{lambda_2}_acc.pth').to('cpu')
    l2_test_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda{lambda_2}_acc.pth').to('cpu')

    data_aug_train_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_data_aug_{config}acc.pth').to('cpu')
    data_aug_test_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_data_aug_{config}acc.pth').to('cpu')

    dropout_train_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_dropout_{config}acc.pth').to('cpu')
    dropout_test_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_dropout_{config}acc.pth').to('cpu')

    batch_norm_train_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_batch_norm_{config}acc.pth').to('cpu')
    batch_norm_test_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_batch_norm_{config}acc.pth').to('cpu')
else:
    unreg_train_acc = unreg_test_acc = invex_train_acc = invex_test_acc = invex_ones_train_acc = invex_ones_test_acc =\
        l2_train_acc = l2_test_acc = data_aug_train_acc = data_aug_test_acc = dropout_train_acc = dropout_test_acc =\
        batch_norm_train_acc = batch_norm_test_acc = None

# (Infinity) Gradient Norms
unreg_grad_norm_total = torch.load(LOSS_METRICS_FOLDER + f'{model}/{unreg_config}grad_norm_total.pth').to('cpu')
invex_grad_norm_total = torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/with_invex_{config}lambda{lambda_1}_grad_norm_total.pth').to('cpu')
invex_ones_grad_norm_total = torch.load(
    LOSS_METRICS_FOLDER + f'{model}/with_invex_ones_{config}lambda{lambda_1}_grad_norm_total.pth').to('cpu')
l2_grad_norm_total = torch.load(LOSS_METRICS_FOLDER +
                                f'{model}/with_l2_{config}l2lambda{lambda_2}_grad_norm_total.pth').to('cpu')
data_aug_grad_norm_total = torch.load(LOSS_METRICS_FOLDER +
                                      f'{model}/with_data_aug_{config}grad_norm_total.pth').to('cpu')
dropout_grad_norm_total = torch.load(LOSS_METRICS_FOLDER +
                                     f'{model}/with_dropout_{config}grad_norm_total.pth').to('cpu')
batch_norm_grad_norm_total = torch.load(LOSS_METRICS_FOLDER +
                                        f'{model}/with_batch_norm_{config}grad_norm_total.pth').to('cpu')

invex_grad_norm_theta = torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/with_invex_{config}lambda{lambda_1}_grad_norm_theta.pth').to('cpu')
invex_ones_grad_norm_theta = torch.load(
    LOSS_METRICS_FOLDER + f'{model}/with_invex_ones_{config}lambda{lambda_1}_grad_norm_theta.pth').to('cpu')

invex_grad_norm_p = torch.load(LOSS_METRICS_FOLDER +
                               f'{model}/with_invex_{config}lambda{lambda_1}_grad_norm_p.pth').to('cpu')
invex_ones_grad_norm_p = torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/with_invex_ones_{config}lambda{lambda_1}_grad_norm_p.pth').to('cpu')

with torch.no_grad():
    # Plot train/test losses for different models
    plt.figure()
    plt.plot(epochs_to_plot, unreg_train_loss, 'k', ls=(0, (1, 1)))  # dotted
    plt.plot(epochs_to_plot, invex_train_loss, 'r', ls=(0, (5, 1)))  # densely dashed
    plt.plot(epochs_to_plot, invex_ones_train_loss, 'gx-')  # x
    plt.plot(epochs_to_plot, l2_train_loss, 'b', ls=(0, (5, 10)))  # loosely dashed
    plt.plot(epochs_to_plot, data_aug_train_loss, 'c', ls=(0, (3, 1, 1, 1)))  # densely dash dotted
    plt.plot(epochs_to_plot, dropout_train_loss, 'm', ls=(0, (3, 5, 1, 5)))  # dash dotted
    plt.plot(epochs_to_plot, batch_norm_train_loss, 'y', ls=(0, (3, 10, 1, 10)))  # loosely dash dotted
    plt.legend(['Unregularised', 'Invex', 'Invex Scalar', r'$\ell_2$', 'Data Augmentation', 'Dropout',
                'Batch Normalisation'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title(plot_title)
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Loss/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Loss/{experiment_name}.pdf')

    plt.figure()
    plt.plot(epochs_to_plot, unreg_test_loss, 'k', ls=(0, (1, 1)))  # dotted
    plt.plot(epochs_to_plot, invex_test_loss, 'r', ls=(0, (5, 1)))  # densely dashed
    plt.plot(epochs_to_plot, invex_ones_test_loss, 'gx-')  # x
    plt.plot(epochs_to_plot, l2_test_loss, 'b', ls=(0, (5, 10)))  # loosely dashed
    plt.plot(epochs_to_plot, data_aug_test_loss, 'c', ls=(0, (3, 1, 1, 1)))  # densely dash dotted
    plt.plot(epochs_to_plot, dropout_test_loss, 'm', ls=(0, (3, 5, 1, 5)))  # dash dotted
    plt.plot(epochs_to_plot, batch_norm_test_loss, 'y', ls=(0, (3, 10, 1, 10)))  # loosely dash dotted
    plt.legend(['Unregularised', 'Invex', 'Invex Scalar', r'$\ell_2$', 'Data Augmentation', 'Dropout',
                'Batch Normalisation'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Loss')
    plt.title(plot_title)
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Loss/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Loss/{experiment_name}.pdf')

    if model != "diffusion_model":
        # Plot train/test accuracies for different models
        plt.figure()
        plt.plot(epochs_to_plot, unreg_train_acc, 'k', ls=(0, (1, 1)))  # dotted
        plt.plot(epochs_to_plot, invex_train_acc, 'r', ls=(0, (5, 1)))  # densely dashed
        plt.plot(epochs_to_plot, invex_ones_train_acc, 'gx-')  # x
        plt.plot(epochs_to_plot, l2_train_acc, 'b', ls=(0, (5, 10)))  # loosely dashed
        plt.plot(epochs_to_plot, data_aug_train_acc, 'c', ls=(0, (3, 1, 1, 1)))  # densely dash dotted
        plt.plot(epochs_to_plot, dropout_train_acc, 'm', ls=(0, (3, 5, 1, 5)))  # dash dotted
        plt.plot(epochs_to_plot, batch_norm_train_acc, 'y', ls=(0, (3, 10, 1, 10)))  # loosely dash dotted
        plt.legend(['Unregularised', 'Invex', 'Invex Scalar', r'$\ell_2$', 'Data Augmentation', 'Dropout',
                    'Batch Normalisation'])
        plt.xlabel('Epochs')
        plt.ylabel('Avg Train Accuracy')
        plt.title(plot_title)
        plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Accuracy/{experiment_name}.jpg')
        # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Accuracy/{experiment_name}.pdf')

        plt.figure()
        plt.plot(epochs_to_plot, unreg_test_acc, 'k', ls=(0, (1, 1)))  # dotted
        plt.plot(epochs_to_plot, invex_test_acc, 'r', ls=(0, (5, 1)))  # densely dashed
        plt.plot(epochs_to_plot, invex_ones_test_acc, 'gx-')  # x
        plt.plot(epochs_to_plot, l2_test_acc, 'b', ls=(0, (5, 10)))  # loosely dashed
        plt.plot(epochs_to_plot, data_aug_test_acc, 'c', ls=(0, (3, 1, 1, 1)))  # densely dash dotted
        plt.plot(epochs_to_plot, dropout_test_acc, 'm', ls=(0, (3, 5, 1, 5)))  # dash dotted
        plt.plot(epochs_to_plot, batch_norm_test_acc, 'y', ls=(0, (3, 10, 1, 10)))  # loosely dash dotted
        plt.legend(['Unregularised', 'Invex', 'Invex Scalar', r'$\ell_2$', 'Data Augmentation', 'Dropout',
                    'Batch Normalisation'])
        plt.xlabel('Epochs')
        plt.ylabel('Avg Test Accuracy')
        plt.title(plot_title)
        plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Accuracy/{experiment_name}.jpg')
        # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Accuracy/{experiment_name}.pdf')

    # Plot infinity gradient norm convergence for different models
    plt.figure()
    plt.plot(epochs_to_plot, unreg_grad_norm_total, 'k', ls=(0, (1, 1)))  # dotted
    plt.plot(epochs_to_plot, invex_grad_norm_total, 'r', ls=(0, (5, 1)))  # densely dashed
    plt.plot(epochs_to_plot, invex_ones_grad_norm_total, 'gx-')  # x
    plt.plot(epochs_to_plot, l2_grad_norm_total, 'b', ls=(0, (5, 10)))  # loosely dashed
    plt.plot(epochs_to_plot, data_aug_grad_norm_total, 'c', ls=(0, (3, 1, 1, 1)))  # densely dash dotted
    plt.plot(epochs_to_plot, dropout_grad_norm_total, 'm', ls=(0, (3, 5, 1, 5)))  # dash dotted
    plt.plot(epochs_to_plot, batch_norm_grad_norm_total, 'y', ls=(0, (3, 10, 1, 10)))  # loosely dash dotted
    plt.legend(['Unregularised', 'Invex', 'Invex Scalar', r'$\ell_2$', 'Data Augmentation', 'Dropout',
                'Batch Normalisation'])
    plt.xlabel('Epochs')
    plt.ylabel(r'$\ell_\infty$ Gradient Norm')
    plt.title(plot_title)
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_total.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_total.pdf')

    # Theta and p separately for invex methods
    plt.figure()
    plt.plot(epochs_to_plot, invex_grad_norm_theta, 'r', ls=(0, (5, 1)))  # densely dashed
    plt.plot(epochs_to_plot, invex_ones_grad_norm_theta, 'rx')  # x
    plt.legend([r'Invex ($\theta$)', r'Invex Scalar ($\theta$)'])
    plt.xlabel('Epochs')
    plt.ylabel(r'$\ell_\infty$ Gradient Norm')
    plt.title(plot_title)
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_theta.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_theta.pdf')

    plt.figure()
    plt.plot(epochs_to_plot, invex_grad_norm_p, 'g', ls=(0, (5, 1)))  # densely dashed
    plt.plot(epochs_to_plot, invex_ones_grad_norm_p, 'gx-')  # x
    plt.legend([r'Invex ($p$)', r'Invex Scalar ($p$)'])
    plt.xlabel('Epochs')
    plt.ylabel(r'$\ell_\infty$ Gradient Norm')
    plt.title(plot_title)
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_p.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_p.pdf')

    plt.show()
