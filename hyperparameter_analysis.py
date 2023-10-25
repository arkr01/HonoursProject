"""
    Plotting/analysis code with **trained models**

    Comparing hyperparameter values for invex & L2.

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
lr = 100.0

experiment_name = f"sgd_zero_lr{lr}_compare"
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

plot_title = rf"{model_title}{init_title}{optim_title}(lr$={lr}$) - "

# epochs_to_plot = torch.load(LOSS_METRICS_FOLDER + f'{model}/epochs_to_plot.pth').to('cpu')
epochs_to_plot = torch.logspace(0, log10(int(1e4)), 100).long().unique() - 1

# Train/Test Losses
unreg_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/{unreg_config}loss.pth').to('cpu')
unreg_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/{unreg_config}loss.pth').to('cpu')

invex_train_loss1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_invex_{config}lambda0.1_loss.pth').to('cpu')
invex_test_loss1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.1_loss.pth').to('cpu')

invex_train_loss01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_invex_{config}lambda0.01_loss.pth').to('cpu')
invex_test_loss01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.01_loss.pth').to('cpu')

invex_train_loss001 = torch.load(LOSS_METRICS_FOLDER + 
                              f'{model}/Train/with_invex_{config}lambda0.001_loss.pth').to('cpu')
invex_test_loss001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.001_loss.pth').to('cpu')

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

l2_train_loss1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.1_loss.pth').to('cpu')
l2_test_loss1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.1_loss.pth').to('cpu')

l2_train_loss01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.01_loss.pth').to('cpu')
l2_test_loss01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.01_loss.pth').to('cpu')

l2_train_loss001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.001_loss.pth').to('cpu')
l2_test_loss001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.001_loss.pth').to('cpu')

l2_train_loss0001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.0001_loss.pth').to('cpu')
l2_test_loss0001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.0001_loss.pth').to('cpu')

# Train/Test Accuracies
if model != "diffusion_model":
    unreg_train_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/{unreg_config}acc.pth').to('cpu')
    unreg_test_acc = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/{unreg_config}acc.pth').to('cpu')
    
    invex_train_acc1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_invex_{config}lambda0.1_acc.pth').to('cpu')
    invex_test_acc1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.1_acc.pth').to('cpu')
    
    invex_train_acc01 = torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Train/with_invex_{config}lambda0.01_acc.pth').to('cpu')
    invex_test_acc01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.01_acc.pth').to('cpu')
    
    invex_train_acc001 = torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Train/with_invex_{config}lambda0.001_acc.pth').to('cpu')
    invex_test_acc001 = torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/Test/with_invex_{config}lambda0.001_acc.pth').to('cpu')
    
    invex_train_acc0001 = torch.load(LOSS_METRICS_FOLDER +
                                     f'{model}/Train/with_invex_{config}lambda0.0001_acc.pth').to('cpu')
    invex_test_acc0001 = torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/Test/with_invex_{config}lambda0.0001_acc.pth').to('cpu')
    
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
    
    l2_train_acc1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.1_acc.pth').to('cpu')
    l2_test_acc1 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.1_acc.pth').to('cpu')
    
    l2_train_acc01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.01_acc.pth').to('cpu')
    l2_test_acc01 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.01_acc.pth').to('cpu')
    
    l2_train_acc001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.001_acc.pth').to('cpu')
    l2_test_acc001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.001_acc.pth').to('cpu')
    
    l2_train_acc0001 = torch.load(LOSS_METRICS_FOLDER +
                                  f'{model}/Train/with_l2_{config}l2lambda0.0001_acc.pth').to('cpu')
    l2_test_acc0001 = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.0001_acc.pth').to('cpu')
else:
    unreg_train_acc = unreg_test_acc = invex_train_acc1 = invex_test_acc1 = invex_train_acc01 = invex_test_acc01 =\
        invex_train_acc001 = invex_test_acc001 = invex_train_acc0001 = invex_test_acc0001 = invex_ones_train_acc1 =\
        invex_ones_test_acc1 = invex_ones_train_acc01 = invex_ones_test_acc01 = invex_ones_train_acc001 =\
        invex_ones_test_acc001 = invex_ones_train_acc0001 = invex_ones_test_acc0001 = l2_train_acc1 = l2_test_acc1 =\
        l2_train_acc01 = l2_test_acc01 = l2_train_acc001 = l2_test_acc001 = l2_train_acc0001 = l2_test_acc0001 = None

# (Infinity) Gradient Norms
unreg_grad_norm_total = torch.load(LOSS_METRICS_FOLDER + f'{model}/{unreg_config}grad_norm_total.pth').to('cpu')

invex_grad_norm1_total = torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/with_invex_{config}lambda0.1_grad_norm_total.pth').to('cpu')
invex_grad_norm01_total = torch.load(LOSS_METRICS_FOLDER +
                                     f'{model}/with_invex_{config}lambda0.01_grad_norm_total.pth').to('cpu')
invex_grad_norm001_total = torch.load(LOSS_METRICS_FOLDER +
                                      f'{model}/with_invex_{config}lambda0.001_grad_norm_total.pth').to('cpu')
invex_grad_norm0001_total = torch.load(LOSS_METRICS_FOLDER + 
                                       f'{model}/with_invex_{config}lambda0.0001_grad_norm_total.pth').to('cpu')

invex_ones_grad_norm1_total = torch.load(LOSS_METRICS_FOLDER +
                                         f'{model}/with_invex_ones_{config}lambda0.1_grad_norm_total.pth').to('cpu')
invex_ones_grad_norm01_total = torch.load(LOSS_METRICS_FOLDER +
                                          f'{model}/with_invex_ones_{config}lambda0.01_grad_norm_total.pth').to('cpu')
invex_ones_grad_norm001_total = torch.load(LOSS_METRICS_FOLDER +
                                           f'{model}/with_invex_ones_{config}lambda0.001_grad_norm_total.pth').to('cpu')
invex_ones_grad_norm0001_total = torch.load(
    LOSS_METRICS_FOLDER + f'{model}/with_invex_ones_{config}lambda0.0001_grad_norm_total.pth').to('cpu')

invex_grad_norm1_theta = torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/with_invex_{config}lambda0.1_grad_norm_theta.pth').to('cpu')
invex_grad_norm01_theta = torch.load(LOSS_METRICS_FOLDER +
                                     f'{model}/with_invex_{config}lambda0.01_grad_norm_theta.pth').to('cpu')
invex_grad_norm001_theta = torch.load(LOSS_METRICS_FOLDER +
                                      f'{model}/with_invex_{config}lambda0.001_grad_norm_theta.pth').to('cpu')
invex_grad_norm0001_theta = torch.load(LOSS_METRICS_FOLDER + 
                                       f'{model}/with_invex_{config}lambda0.0001_grad_norm_theta.pth').to('cpu')

invex_ones_grad_norm1_theta = torch.load(LOSS_METRICS_FOLDER +
                                         f'{model}/with_invex_ones_{config}lambda0.1_grad_norm_theta.pth').to('cpu')
invex_ones_grad_norm01_theta = torch.load(LOSS_METRICS_FOLDER +
                                          f'{model}/with_invex_ones_{config}lambda0.01_grad_norm_theta.pth').to('cpu')
invex_ones_grad_norm001_theta = torch.load(LOSS_METRICS_FOLDER +
                                           f'{model}/with_invex_ones_{config}lambda0.001_grad_norm_theta.pth').to('cpu')
invex_ones_grad_norm0001_theta = torch.load(
    LOSS_METRICS_FOLDER + f'{model}/with_invex_ones_{config}lambda0.0001_grad_norm_theta.pth').to('cpu')

invex_grad_norm1_p = torch.load(LOSS_METRICS_FOLDER +
                                f'{model}/with_invex_{config}lambda0.1_grad_norm_p.pth').to('cpu')
invex_grad_norm01_p = torch.load(LOSS_METRICS_FOLDER +
                                 f'{model}/with_invex_{config}lambda0.01_grad_norm_p.pth').to('cpu')
invex_grad_norm001_p = torch.load(LOSS_METRICS_FOLDER +
                                  f'{model}/with_invex_{config}lambda0.001_grad_norm_p.pth').to('cpu')
invex_grad_norm0001_p = torch.load(LOSS_METRICS_FOLDER + 
                                   f'{model}/with_invex_{config}lambda0.0001_grad_norm_p.pth').to('cpu')

invex_ones_grad_norm1_p = torch.load(LOSS_METRICS_FOLDER +
                                     f'{model}/with_invex_ones_{config}lambda0.1_grad_norm_p.pth').to('cpu')
invex_ones_grad_norm01_p = torch.load(LOSS_METRICS_FOLDER +
                                      f'{model}/with_invex_ones_{config}lambda0.01_grad_norm_p.pth').to('cpu')
invex_ones_grad_norm001_p = torch.load(LOSS_METRICS_FOLDER +
                                       f'{model}/with_invex_ones_{config}lambda0.001_grad_norm_p.pth').to('cpu')
invex_ones_grad_norm0001_p = torch.load(
    LOSS_METRICS_FOLDER + f'{model}/with_invex_ones_{config}lambda0.0001_grad_norm_p.pth').to('cpu')

l2_grad_norm1_total = torch.load(LOSS_METRICS_FOLDER +
                                 f'{model}/with_l2_{config}l2lambda0.1_grad_norm_total.pth').to('cpu')
l2_grad_norm01_total = torch.load(LOSS_METRICS_FOLDER +
                                  f'{model}/with_l2_{config}l2lambda0.01_grad_norm_total.pth').to('cpu')
l2_grad_norm001_total = torch.load(LOSS_METRICS_FOLDER +
                                   f'{model}/with_l2_{config}l2lambda0.001_grad_norm_total.pth').to('cpu')
l2_grad_norm0001_total = torch.load(LOSS_METRICS_FOLDER +
                                    f'{model}/with_l2_{config}l2lambda0.0001_grad_norm_total.pth').to('cpu')

with torch.no_grad():
    # Plot train/test losses for different models
    plt.figure()
    plt.plot(epochs_to_plot, unreg_train_loss, 'k', ls=':')
    plt.plot(epochs_to_plot, invex_train_loss1, 'r')
    plt.plot(epochs_to_plot, invex_ones_train_loss1, 'rx-')
    plt.plot(epochs_to_plot, invex_train_loss01, 'b')
    plt.plot(epochs_to_plot, invex_ones_train_loss01, 'bx-')
    plt.plot(epochs_to_plot, invex_train_loss001, 'y')
    plt.plot(epochs_to_plot, invex_ones_train_loss001, 'yx-')
    plt.plot(epochs_to_plot, invex_train_loss0001, 'm')
    plt.plot(epochs_to_plot, invex_ones_train_loss0001, 'mx-')
    plt.legend(['Unregularised', r'Invex $\lambda=0.1$', r'Invex Scalar $\lambda=0.1$', r'Invex $\lambda=0.01$',
                r'Invex Scalar $\lambda=0.01$', r'Invex $\lambda=0.001$', r'Invex Scalar $\lambda=0.001$',
                r'Invex $\lambda=0.0001$', r'Invex Scalar $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title(rf'{plot_title}Standard vs Scalar Invex Regularisation')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Loss/{experiment_name}_invex.jpg')

    plt.figure()
    plt.plot(epochs_to_plot, unreg_test_loss, 'k', ls=':')
    plt.plot(epochs_to_plot, invex_test_loss1, 'r')
    plt.plot(epochs_to_plot, invex_ones_test_loss1, 'rx-')
    plt.plot(epochs_to_plot, invex_test_loss01, 'b')
    plt.plot(epochs_to_plot, invex_ones_test_loss01, 'bx-')
    plt.plot(epochs_to_plot, invex_test_loss001, 'y')
    plt.plot(epochs_to_plot, invex_ones_test_loss001, 'yx-')
    plt.plot(epochs_to_plot, invex_test_loss0001, 'm')
    plt.plot(epochs_to_plot, invex_ones_test_loss0001, 'mx-')
    plt.legend(['Unregularised', r'Invex $\lambda=0.1$', r'Invex Scalar $\lambda=0.1$', r'Invex $\lambda=0.01$',
                r'Invex Scalar $\lambda=0.01$', r'Invex $\lambda=0.001$', r'Invex Scalar $\lambda=0.001$',
                r'Invex $\lambda=0.0001$', r'Invex Scalar $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Loss')
    plt.title(rf'{plot_title}Standard vs Scalar Invex Regularisation')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Loss/{experiment_name}_invex.jpg')

    plt.figure()
    plt.plot(epochs_to_plot, unreg_train_loss, 'k', ls=':')
    plt.plot(epochs_to_plot, l2_train_loss1, 'r')
    plt.plot(epochs_to_plot, l2_train_loss01, 'b')
    plt.plot(epochs_to_plot, l2_train_loss001, 'y')
    plt.plot(epochs_to_plot, l2_train_loss0001, 'm')
    plt.legend(['Unregularised', r'$\ell_2$ $\lambda=0.1$', r'$\ell_2$ $\lambda=0.01$', r'$\ell_2$ $\lambda=0.001$',
                r'$\ell_2$ $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title(rf'{plot_title}$\ell_2$ Regularisation')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Loss/{experiment_name}_l2.jpg')

    plt.figure()
    plt.plot(epochs_to_plot, unreg_test_loss, 'k', ls=':')
    plt.plot(epochs_to_plot, l2_test_loss1, 'r')
    plt.plot(epochs_to_plot, l2_test_loss01, 'b')
    plt.plot(epochs_to_plot, l2_test_loss001, 'y')
    plt.plot(epochs_to_plot, l2_test_loss0001, 'm')
    plt.legend(['Unregularised', r'$\ell_2$ $\lambda=0.1$', r'$\ell_2$ $\lambda=0.01$', r'$\ell_2$ $\lambda=0.001$',
                r'$\ell_2$ $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Loss')
    plt.title(rf'{plot_title}$\ell_2$ Regularisation')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Loss/{experiment_name}_l2.jpg')
    
    # Train/Test Accuracies
    if model != "diffusion_model":
        plt.figure()
        plt.plot(epochs_to_plot, unreg_train_acc, 'k', ls=':')
        plt.plot(epochs_to_plot, invex_train_acc1, 'r')
        plt.plot(epochs_to_plot, invex_ones_train_acc1, 'rx-')
        plt.plot(epochs_to_plot, invex_train_acc01, 'b')
        plt.plot(epochs_to_plot, invex_ones_train_acc01, 'bx-')
        plt.plot(epochs_to_plot, invex_train_acc001, 'y')
        plt.plot(epochs_to_plot, invex_ones_train_acc001, 'yx-')
        plt.plot(epochs_to_plot, invex_train_acc0001, 'm')
        plt.plot(epochs_to_plot, invex_ones_train_acc0001, 'mx-')
        plt.legend(['Unregularised', r'Invex $\lambda=0.1$', r'Invex Scalar $\lambda=0.1$', r'Invex $\lambda=0.01$',
                    r'Invex Scalar $\lambda=0.01$', r'Invex $\lambda=0.001$', r'Invex Scalar $\lambda=0.001$',
                    r'Invex $\lambda=0.0001$', r'Invex Scalar $\lambda=0.0001$'])
        plt.xlabel('Epochs')
        plt.ylabel('Avg Train Accuracy')
        plt.title(rf'{plot_title}Standard vs Scalar Invex Regularisation')
        plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Accuracy/{experiment_name}_invex.jpg')
    
        plt.figure()
        plt.plot(epochs_to_plot, unreg_test_acc, 'k', ls=':')
        plt.plot(epochs_to_plot, invex_test_acc1, 'r')
        plt.plot(epochs_to_plot, invex_ones_test_acc1, 'rx-')
        plt.plot(epochs_to_plot, invex_test_acc01, 'b')
        plt.plot(epochs_to_plot, invex_ones_test_acc01, 'bx-')
        plt.plot(epochs_to_plot, invex_test_acc001, 'y')
        plt.plot(epochs_to_plot, invex_ones_test_acc001, 'yx-')
        plt.plot(epochs_to_plot, invex_test_acc0001, 'm')
        plt.plot(epochs_to_plot, invex_ones_test_acc0001, 'mx-')
        plt.legend(['Unregularised', r'Invex $\lambda=0.1$', r'Invex Scalar $\lambda=0.1$', r'Invex $\lambda=0.01$',
                    r'Invex Scalar $\lambda=0.01$', r'Invex $\lambda=0.001$', r'Invex Scalar $\lambda=0.001$',
                    r'Invex $\lambda=0.0001$', r'Invex Scalar $\lambda=0.0001$'])
        plt.xlabel('Epochs')
        plt.ylabel('Avg Test Accuracy')
        plt.title(rf'{plot_title}Standard vs Scalar Invex Regularisation')
        plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Accuracy/{experiment_name}_invex.jpg')
    
        plt.figure()
        plt.plot(epochs_to_plot, unreg_train_acc, 'k', ls=':')
        plt.plot(epochs_to_plot, l2_train_acc1, 'r')
        plt.plot(epochs_to_plot, l2_train_acc01, 'b')
        plt.plot(epochs_to_plot, l2_train_acc001, 'y')
        plt.plot(epochs_to_plot, l2_train_acc0001, 'm')
        plt.legend(['Unregularised', r'$\ell_2$ $\lambda=0.1$', r'$\ell_2$ $\lambda=0.01$', r'$\ell_2$ $\lambda=0.001$',
                    r'$\ell_2$ $\lambda=0.0001$'])
        plt.xlabel('Epochs')
        plt.ylabel('Avg Train Accuracy')
        plt.title(rf'{plot_title}$\ell_2$ Regularisation')
        plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Accuracy/{experiment_name}_l2.jpg')
    
        plt.figure()
        plt.plot(epochs_to_plot, unreg_test_acc, 'k', ls=':')
        plt.plot(epochs_to_plot, l2_test_acc1, 'r')
        plt.plot(epochs_to_plot, l2_test_acc01, 'b')
        plt.plot(epochs_to_plot, l2_test_acc001, 'y')
        plt.plot(epochs_to_plot, l2_test_acc0001, 'm')
        plt.legend(['Unregularised', r'$\ell_2$ $\lambda=0.1$', r'$\ell_2$ $\lambda=0.01$', r'$\ell_2$ $\lambda=0.001$',
                    r'$\ell_2$ $\lambda=0.0001$'])
        plt.xlabel('Epochs')
        plt.ylabel('Avg Test Accuracy')
        plt.title(rf'{plot_title}$\ell_2$ Regularisation')
        plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Accuracy/{experiment_name}_l2.jpg')
    
    # Infinity Gradient Norms
    plt.figure()
    plt.plot(epochs_to_plot, unreg_grad_norm_total, 'k', ls=':')
    plt.plot(epochs_to_plot, invex_grad_norm1_total, 'r')
    plt.plot(epochs_to_plot, invex_ones_grad_norm1_total, 'rx-')
    plt.plot(epochs_to_plot, invex_grad_norm01_total, 'b')
    plt.plot(epochs_to_plot, invex_ones_grad_norm01_total, 'bx-')
    plt.plot(epochs_to_plot, invex_grad_norm001_total, 'y')
    plt.plot(epochs_to_plot, invex_ones_grad_norm001_total, 'yx-')
    plt.plot(epochs_to_plot, invex_grad_norm0001_total, 'm')
    plt.plot(epochs_to_plot, invex_ones_grad_norm0001_total, 'mx-')
    plt.legend(['Unregularised', r'Invex $\lambda=0.1$', r'Invex Scalar $\lambda=0.1$', r'Invex $\lambda=0.01$',
                r'Invex Scalar $\lambda=0.01$', r'Invex $\lambda=0.001$', r'Invex Scalar $\lambda=0.001$',
                r'Invex $\lambda=0.0001$', r'Invex Scalar $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel(r'$\ell_\infty$ Gradient Norm')
    plt.title(rf'{plot_title}Standard vs Scalar Invex Regularisation')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_invex_total.jpg')

    plt.figure()
    plt.plot(epochs_to_plot, unreg_grad_norm_total, 'k', ls=':')
    plt.plot(epochs_to_plot, l2_grad_norm1_total, 'r')
    plt.plot(epochs_to_plot, l2_grad_norm01_total, 'b')
    plt.plot(epochs_to_plot, l2_grad_norm001_total, 'y')
    plt.plot(epochs_to_plot, l2_grad_norm0001_total, 'm')
    plt.legend(['Unregularised', r'$\ell_2$ $\lambda=0.1$', r'$\ell_2$ $\lambda=0.01$', r'$\ell_2$ $\lambda=0.001$',
                r'$\ell_2$ $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel(r'$\ell_\infty$ Gradient Norm')
    plt.title(rf'{plot_title}$\ell_2$ Regularisation')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_l2.jpg')

    # Theta and p separately for invex methods
    plt.figure()
    plt.plot(epochs_to_plot, invex_grad_norm1_theta, 'r')
    plt.plot(epochs_to_plot, invex_ones_grad_norm1_theta, 'rx-')
    plt.plot(epochs_to_plot, invex_grad_norm01_theta, 'b')
    plt.plot(epochs_to_plot, invex_ones_grad_norm01_theta, 'bx-')
    plt.plot(epochs_to_plot, invex_grad_norm001_theta, 'y')
    plt.plot(epochs_to_plot, invex_ones_grad_norm001_theta, 'yx-')
    plt.plot(epochs_to_plot, invex_grad_norm0001_theta, 'm')
    plt.plot(epochs_to_plot, invex_ones_grad_norm0001_theta, 'mx-')
    plt.legend([r'Invex ($\theta$) $\lambda=0.1$', r'Invex Scalar ($\theta$) $\lambda=0.1$',
                r'Invex ($\theta$) $\lambda=0.01$', r'Invex Scalar ($\theta$) $\lambda=0.01$',
                r'Invex ($\theta$) $\lambda=0.001$', r'Invex Scalar ($\theta$) $\lambda=0.001$',
                r'Invex ($\theta$) $\lambda=0.0001$', r'Invex Scalar ($\theta$) $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel(r'$\ell_\infty$ Gradient Norm')
    plt.title(rf'{plot_title}Standard vs Scalar Invex Regularisation')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_invex_theta.jpg')

    plt.figure()
    plt.plot(epochs_to_plot, invex_grad_norm1_p, 'r')
    plt.plot(epochs_to_plot, invex_ones_grad_norm1_p, 'rx-')
    plt.plot(epochs_to_plot, invex_grad_norm01_p, 'b')
    plt.plot(epochs_to_plot, invex_ones_grad_norm01_p, 'bx-')
    plt.plot(epochs_to_plot, invex_grad_norm001_p, 'y')
    plt.plot(epochs_to_plot, invex_ones_grad_norm001_p, 'yx-')
    plt.plot(epochs_to_plot, invex_grad_norm0001_p, 'm')
    plt.plot(epochs_to_plot, invex_ones_grad_norm0001_p, 'mx-')
    plt.legend([r'Invex ($p$) $\lambda=0.1$', r'Invex Scalar ($p$) $\lambda=0.1$', r'Invex ($p$) $\lambda=0.01$',
                r'Invex Scalar ($p$) $\lambda=0.01$', r'Invex ($p$) $\lambda=0.001$',
                r'Invex Scalar ($p$) $\lambda=0.001$', r'Invex ($p$) $\lambda=0.0001$',
                r'Invex Scalar ($p$) $\lambda=0.0001$'])
    plt.xlabel('Epochs')
    plt.ylabel(r'$\ell_\infty$ Gradient Norm')
    plt.title(rf'{plot_title}Standard vs Scalar Invex Regularisation')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}_invex_p.jpg')

    plt.show()
