"""
    Plotting/analysis code with **trained models**

    Author: Adrian Rahul Kamal Rajkamal
"""
from matplotlib import rcParams
import matplotlib.pyplot as plt
from numpy import log10

import torch

from workflow import LOSS_METRICS_FOLDER, PLOTS_RESULTS_FOLDER

rcParams['axes.titlesize'] = 'medium'
rcParams["text.usetex"] = True  # Allows LaTeX in titles/labels/legends

# What experiment we're plotting
model = "resnet18_model"
plot_title = " - ResNet18 Classification (GD)"
lr = 0.01

experiment_name = f"gd_lr{lr}_compare"
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

# epochs_to_plot = torch.load(LOSS_METRICS_FOLDER + f'{model}/epochs_to_plot.pth').to('cpu')
epochs_to_plot = torch.logspace(0, log10(int(2e4)), 100).long().unique() - 1

# Train/Test Losses
unreg_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/{unreg_config}loss.pth').to('cpu')
unreg_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/{unreg_config}loss.pth').to('cpu')

invex_train_loss1 = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_invex_{config}lambda0.1_loss.pth').to('cpu')
invex_test_loss1 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/Test/with_invex_{config}lambda0.1_loss.pth').to('cpu')

invex_train_loss01 = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_invex_{config}lambda0.01_loss.pth').to('cpu')
invex_test_loss01 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/Test/with_invex_{config}lambda0.01_loss.pth').to('cpu')

invex_train_loss001 = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_invex_{config}lambda0.001_loss.pth').to('cpu')
invex_test_loss001 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/Test/with_invex_{config}lambda0.001_loss.pth').to('cpu')

invex_train_loss0001 = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_invex_{config}lambda0.0001_loss.pth').to('cpu')
invex_test_loss0001 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/Test/with_invex_{config}lambda0.0001_loss.pth').to('cpu')

invex_ones_train_loss1 = (torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Train/with_invex_ones_{config}lambda0.1_loss.pth').to('cpu'))
invex_ones_test_loss1 = (torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Test/with_invex_ones_{config}lambda0.1_loss.pth').to('cpu'))

invex_ones_train_loss01 = (torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Train/with_invex_ones_{config}lambda0.01_loss.pth').to('cpu'))
invex_ones_test_loss01 = (torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Test/with_invex_ones_{config}lambda0.01_loss.pth').to('cpu'))

invex_ones_train_loss001 = (torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Train/with_invex_ones_{config}lambda0.001_loss.pth').to('cpu'))
invex_ones_test_loss001 = (torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Test/with_invex_ones_{config}lambda0.001_loss.pth').to('cpu'))

invex_ones_train_loss0001 = (torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Train/with_invex_ones_{config}lambda0.0001_loss.pth').to('cpu'))
invex_ones_test_loss0001 = (torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Test/with_invex_ones_{config}lambda0.0001_loss.pth').to('cpu'))

l2_train_loss01 = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_l2_{config}l2lambda0.01_loss.pth').to('cpu')
l2_test_loss01 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/Test/with_l2_{config}l2lambda0.01_loss.pth').to('cpu')

l2_train_loss0001 = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_l2_{config}l2lambda0.0001_loss.pth').to('cpu')
l2_test_loss0001 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/Test/with_l2_{config}l2lambda0.0001_loss.pth').to('cpu')

l2_train_loss6 = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_l2_{config}l2lambda1e-06_loss.pth').to('cpu')
l2_test_loss6 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/Test/with_l2_{config}l2lambda1e-06_loss.pth').to('cpu')

l2_train_loss8 = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_l2_{config}l2lambda1e-08_loss.pth').to('cpu')
l2_test_loss8 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/Test/with_l2_{config}l2lambda1e-08_loss.pth').to('cpu')

# Train/Test Accuracies
unreg_train_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/{unreg_config}acc.pth').to('cpu')
unreg_test_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/{unreg_config}acc.pth').to('cpu')

invex_train_acc1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_invex_{config}lambda0.1_acc.pth').to('cpu')
invex_test_acc1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.1_acc.pth').to('cpu')

invex_train_acc01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_invex_{config}lambda0.01_acc.pth').to('cpu')
invex_test_acc01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.01_acc.pth').to('cpu')

invex_train_acc001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_invex_{config}lambda0.001_acc.pth').to('cpu')
invex_test_acc001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.001_acc.pth').to('cpu')

invex_train_acc0001 = torch.load(LOSS_METRICS_FOLDER +
                                 f'{model}/Train/with_invex_{config}lambda0.0001_acc.pth').to('cpu')
invex_test_acc0001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.0001_acc.pth').to('cpu')

invex_ones_train_acc1 = (torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Train/with_invex_ones_{config}lambda0.1_acc.pth').to('cpu'))
invex_ones_test_acc1 = (torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Test/with_invex_ones_{config}lambda0.1_acc.pth').to('cpu'))

invex_ones_train_acc01 = (torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Train/with_invex_ones_{config}lambda0.01_acc.pth').to('cpu'))
invex_ones_test_acc01 = (torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Test/with_invex_ones_{config}lambda0.01_acc.pth').to('cpu'))

invex_ones_train_acc001 = (torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Train/with_invex_ones_{config}lambda0.001_acc.pth').to('cpu'))
invex_ones_test_acc001 = (torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Test/with_invex_ones_{config}lambda0.001_acc.pth').to('cpu'))

invex_ones_train_acc0001 = (torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Train/with_invex_ones_{config}lambda0.0001_acc.pth').to('cpu'))
invex_ones_test_acc0001 = (torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Test/with_invex_ones_{config}lambda0.0001_acc.pth').to('cpu'))

l2_train_acc01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.01_acc.pth').to('cpu')
l2_test_acc01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.01_acc.pth').to('cpu')

l2_train_acc0001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.0001_acc.pth').to('cpu')
l2_test_acc0001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.0001_acc.pth').to('cpu')

l2_train_acc6 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda1e-06_acc.pth').to('cpu')
l2_test_acc6 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda1e-06_acc.pth').to('cpu')

l2_train_acc8 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda1e-08_acc.pth').to('cpu')
l2_test_acc8 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda1e-08_acc.pth').to('cpu')

# (Infinity) Gradient Norms
unreg_grad_norm = torch.load(LOSS_METRICS_FOLDER + f'{model}/{unreg_config}grad_norm.pth').to('cpu')

invex_grad_norm1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_invex_{config}lambda0.1_grad_norm.pth').to('cpu')
invex_grad_norm01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_invex_{config}lambda0.01_grad_norm.pth').to('cpu')
invex_grad_norm001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_invex_{config}lambda0.001_grad_norm.pth').to('cpu')
invex_grad_norm0001 = torch.load(LOSS_METRICS_FOLDER +
                                 f'{model}/with_invex_{config}lambda0.0001_grad_norm.pth').to('cpu')

invex_ones_grad_norm1 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/with_invex_ones_{config}lambda0.1_grad_norm.pth').to('cpu')
invex_ones_grad_norm01 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/with_invex_ones_{config}lambda0.01_grad_norm.pth').to('cpu')
invex_ones_grad_norm001 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/with_invex_ones_{config}lambda0.001_grad_norm.pth').to('cpu')
invex_ones_grad_norm0001 = torch.load(LOSS_METRICS_FOLDER +
                             f'{model}/with_invex_ones_{config}lambda0.0001_grad_norm.pth').to('cpu')

l2_grad_norm01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_l2_{config}l2lambda0.01_grad_norm.pth').to('cpu')
l2_grad_norm0001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_l2_{config}l2lambda0.0001_grad_norm.pth').to('cpu')
l2_grad_norm6 = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_l2_{config}l2lambda1e-06_grad_norm.pth').to('cpu')
l2_grad_norm8 = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_l2_{config}l2lambda1e-08_grad_norm.pth').to('cpu')

with torch.no_grad():
    # Plot train/test losses for different models
    plt.figure()
    plt.semilogx(epochs_to_plot, unreg_train_loss, 'k', ls=':')
    plt.semilogx(epochs_to_plot, invex_train_loss1, 'r')
    plt.semilogx(epochs_to_plot, invex_ones_train_loss1, 'rx')
    plt.semilogx(epochs_to_plot, invex_train_loss01, 'b')
    plt.semilogx(epochs_to_plot, invex_ones_train_loss01, 'bx')
    plt.semilogx(epochs_to_plot, invex_train_loss001, 'y')
    plt.semilogx(epochs_to_plot, invex_ones_train_loss001, 'yx')
    plt.semilogx(epochs_to_plot, invex_train_loss0001, 'm')
    plt.semilogx(epochs_to_plot, invex_ones_train_loss0001, 'mx')
    plt.legend(['Unregularised', r'Invex $\lambda=0.1$', r'Invex Scalar $\lambda=0.1$', r'Invex $\lambda=0.01$',
                r'Invex Scalar $\lambda=0.01$', r'Invex $\lambda=0.001$', r'Invex Scalar $\lambda=0.001$',
                r'Invex $\lambda=0.0001$', r'Invex Scalar $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title(rf'Standard vs Scalar Invex Regularisation{plot_title} (lr$\approx{lr}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Loss/{experiment_name}_invex.jpg')

    plt.figure()
    plt.semilogx(epochs_to_plot, unreg_test_loss, 'k', ls=':')
    plt.semilogx(epochs_to_plot, invex_test_loss1, 'r')
    plt.semilogx(epochs_to_plot, invex_ones_test_loss1, 'rx')
    plt.semilogx(epochs_to_plot, invex_test_loss01, 'b')
    plt.semilogx(epochs_to_plot, invex_ones_test_loss01, 'bx')
    plt.semilogx(epochs_to_plot, invex_test_loss001, 'y')
    plt.semilogx(epochs_to_plot, invex_ones_test_loss001, 'yx')
    plt.semilogx(epochs_to_plot, invex_test_loss0001, 'm')
    plt.semilogx(epochs_to_plot, invex_ones_test_loss0001, 'mx')
    plt.legend(['Unregularised', r'Invex $\lambda=0.1$', r'Invex Scalar $\lambda=0.1$', r'Invex $\lambda=0.01$',
                r'Invex Scalar $\lambda=0.01$', r'Invex $\lambda=0.001$', r'Invex Scalar $\lambda=0.001$',
                r'Invex $\lambda=0.0001$', r'Invex Scalar $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Loss')
    plt.title(rf'Standard vs Scalar Invex Regularisation{plot_title} (lr$\approx{lr}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Loss/{experiment_name}_invex.jpg')

    plt.figure()
    plt.semilogx(epochs_to_plot, unreg_train_loss, 'k', ls=':')
    plt.semilogx(epochs_to_plot, l2_train_loss01, 'r')
    plt.semilogx(epochs_to_plot, l2_train_loss0001, 'b')
    plt.semilogx(epochs_to_plot, l2_train_loss6, 'y')
    plt.semilogx(epochs_to_plot, l2_train_loss8, 'm')
    plt.legend(['Unregularised', r'$\ell_2$ $\lambda=0.01$', r'$\ell_2$ $\lambda=0.0001$', r'$\ell_2$ $\lambda=10^{-6}$',
                r'$\ell_2$ $\lambda=10^{-8}$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title(rf'$\ell_2$ Regularisation{plot_title} (lr$\approx{lr}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Loss/{experiment_name}_l2.jpg')

    plt.figure()
    plt.semilogx(epochs_to_plot, unreg_test_loss, 'k', ls=':')
    plt.semilogx(epochs_to_plot, l2_test_loss01, 'r')
    plt.semilogx(epochs_to_plot, l2_test_loss0001, 'b')
    plt.semilogx(epochs_to_plot, l2_test_loss6, 'y')
    plt.semilogx(epochs_to_plot, l2_test_loss8, 'm')
    plt.legend(['Unregularised', r'$\ell_2$ $\lambda=0.01$', r'$\ell_2$ $\lambda=0.0001$', r'$\ell_2$ $\lambda=10^{-6}$',
                r'$\ell_2$ $\lambda=10^{-8}$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Loss')
    plt.title(rf'$\ell_2$ Regularisation{plot_title} (lr$\approx{lr}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Loss/{experiment_name}_l2.jpg')
    
    # Train/Test Accuracies
    plt.figure()
    plt.semilogx(epochs_to_plot, unreg_train_acc, 'k', ls=':')
    plt.semilogx(epochs_to_plot, invex_train_acc1, 'r')
    plt.semilogx(epochs_to_plot, invex_ones_train_acc1, 'rx')
    plt.semilogx(epochs_to_plot, invex_train_acc01, 'b')
    plt.semilogx(epochs_to_plot, invex_ones_train_acc01, 'bx')
    plt.semilogx(epochs_to_plot, invex_train_acc001, 'y')
    plt.semilogx(epochs_to_plot, invex_ones_train_acc001, 'yx')
    plt.semilogx(epochs_to_plot, invex_train_acc0001, 'm')
    plt.semilogx(epochs_to_plot, invex_ones_train_acc0001, 'mx')
    plt.legend(['Unregularised', r'Invex $\lambda=0.1$', r'Invex Scalar $\lambda=0.1$', r'Invex $\lambda=0.01$',
                r'Invex Scalar $\lambda=0.01$', r'Invex $\lambda=0.001$', r'Invex Scalar $\lambda=0.001$',
                r'Invex $\lambda=0.0001$', r'Invex Scalar $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Accuracy')
    plt.title(rf'Standard vs Scalar Invex Regularisation{plot_title} (lr$\approx{lr}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Accuracy/{experiment_name}_invex.jpg')

    plt.figure()
    plt.semilogx(epochs_to_plot, unreg_test_acc, 'k', ls=':')
    plt.semilogx(epochs_to_plot, invex_test_acc1, 'r')
    plt.semilogx(epochs_to_plot, invex_ones_test_acc1, 'rx')
    plt.semilogx(epochs_to_plot, invex_test_acc01, 'b')
    plt.semilogx(epochs_to_plot, invex_ones_test_acc01, 'bx')
    plt.semilogx(epochs_to_plot, invex_test_acc001, 'y')
    plt.semilogx(epochs_to_plot, invex_ones_test_acc001, 'yx')
    plt.semilogx(epochs_to_plot, invex_test_acc0001, 'm')
    plt.semilogx(epochs_to_plot, invex_ones_test_acc0001, 'mx')
    plt.legend(['Unregularised', r'Invex $\lambda=0.1$', r'Invex Scalar $\lambda=0.1$', r'Invex $\lambda=0.01$',
                r'Invex Scalar $\lambda=0.01$', r'Invex $\lambda=0.001$', r'Invex Scalar $\lambda=0.001$',
                r'Invex $\lambda=0.0001$', r'Invex Scalar $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Accuracy')
    plt.title(rf'Standard vs Scalar Invex Regularisation{plot_title} (lr$\approx{lr}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Accuracy/{experiment_name}_invex.jpg')

    plt.figure()
    plt.semilogx(epochs_to_plot, unreg_train_acc, 'k', ls=':')
    plt.semilogx(epochs_to_plot, l2_train_acc01, 'r')
    plt.semilogx(epochs_to_plot, l2_train_acc0001, 'b')
    plt.semilogx(epochs_to_plot, l2_train_acc6, 'y')
    plt.semilogx(epochs_to_plot, l2_train_acc8, 'm')
    plt.legend(['Unregularised', r'$\ell_2$ $\lambda=0.01$', r'$\ell_2$ $\lambda=0.0001$',
                r'$\ell_2$ $\lambda=10^{-6}$', r'$\ell_2$ $\lambda=10^{-8}$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Accuracy')
    plt.title(rf'$\ell_2$ Regularisation{plot_title} (lr$\approx{lr}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Accuracy/{experiment_name}_l2.jpg')

    plt.figure()
    plt.semilogx(epochs_to_plot, unreg_test_acc, 'k', ls=':')
    plt.semilogx(epochs_to_plot, l2_test_acc01, 'r')
    plt.semilogx(epochs_to_plot, l2_test_acc0001, 'b')
    plt.semilogx(epochs_to_plot, l2_test_acc6, 'y')
    plt.semilogx(epochs_to_plot, l2_test_acc8, 'm')
    plt.legend(['Unregularised', r'$\ell_2$ $\lambda=0.01$', r'$\ell_2$ $\lambda=0.0001$',
                r'$\ell_2$ $\lambda=10^{-6}$', r'$\ell_2$ $\lambda=10^{-8}$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Accuracy')
    plt.title(rf'$\ell_2$ Regularisation{plot_title} (lr$\approx{lr}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Accuracy/{experiment_name}_l2.jpg')
    
    # Infinity Gradient Norms
    plt.figure()
    plt.semilogx(epochs_to_plot, unreg_grad_norm, 'k', ls=':')
    plt.semilogx(epochs_to_plot, invex_grad_norm1, 'r')
    plt.semilogx(epochs_to_plot, invex_ones_grad_norm1, 'rx')
    plt.semilogx(epochs_to_plot, invex_grad_norm01, 'b')
    plt.semilogx(epochs_to_plot, invex_ones_grad_norm01, 'bx')
    plt.semilogx(epochs_to_plot, invex_grad_norm001, 'y')
    plt.semilogx(epochs_to_plot, invex_ones_grad_norm001, 'yx')
    plt.semilogx(epochs_to_plot, invex_grad_norm0001, 'm')
    plt.semilogx(epochs_to_plot, invex_ones_grad_norm0001, 'mx')
    plt.legend(['Unregularised', r'Invex $\lambda=0.1$', r'Invex Scalar $\lambda=0.1$', r'Invex $\lambda=0.01$',
                r'Invex Scalar $\lambda=0.01$', r'Invex $\lambda=0.001$', r'Invex Scalar $\lambda=0.001$',
                r'Invex $\lambda=0.0001$', r'Invex Scalar $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel('$\ell_\infty$ Gradient Norm')
    plt.title(rf'Standard vs Scalar Invex Regularisation{plot_title} (lr$\approx{lr}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_invex.jpg')

    plt.figure()
    plt.semilogx(epochs_to_plot, unreg_grad_norm, 'k', ls=':')
    plt.semilogx(epochs_to_plot, l2_grad_norm01, 'r')
    plt.semilogx(epochs_to_plot, l2_grad_norm0001, 'b')
    plt.semilogx(epochs_to_plot, l2_grad_norm6, 'y')
    plt.semilogx(epochs_to_plot, l2_grad_norm8, 'm')
    plt.legend(['Unregularised', r'$\ell_2$ $\lambda=0.01$', r'$\ell_2$ $\lambda=0.0001$',
                r'$\ell_2$ $\lambda=10^{-6}$', r'$\ell_2$ $\lambda=10^{-8}$'])
    plt.xlabel('Epochs')
    plt.ylabel('$\ell_\infty$ Gradient Norm')
    plt.title(rf'$\ell_2$ Regularisation{plot_title} (lr$\approx{lr}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_l2.jpg')

    plt.show()
