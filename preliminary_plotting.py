"""
    Plotting/analysis code with **trained models**

    Comparing regularisation methods.

    Author: Adrian Rahul Kamal Rajkamal
"""
from matplotlib import rcParams
import matplotlib.pyplot as plt

import torch

from workflow import LOSS_METRICS_FOLDER, PLOTS_RESULTS_FOLDER

rcParams["text.usetex"] = True  # Allows LaTeX in titles/labels/legends

# What experiment we're plotting
model = "logistic_model"
lr = 0.1
lambda_1 = 2000.0
lambda_2 = 4000000.0
experiment_name = f"gd_lr{lr}_lambda{lambda_1}_l2lambda{lambda_2}_all_individual"

standard_lr = experiment_name.__contains__("lrst")

unreg_or_gd = "unregularised_" if experiment_name[0] == 's' and standard_lr else "with_"
zero_or_random = "zero_init_" if experiment_name.__contains__("zero") else ""
reg_or_gd = "" if experiment_name[0] == 's' else ("lbfgs_" if experiment_name[0] == 'l' else "gd_")
subset = "" if reg_or_gd else ""
lr_val = "" if standard_lr else f"lr{lr}_"

config = zero_or_random + reg_or_gd + subset + lr_val
unreg_config = unreg_or_gd + config

if "wie" in LOSS_METRICS_FOLDER or "rgp" in LOSS_METRICS_FOLDER:
    experiment_name = "full_" + experiment_name

model_title = "Multinomial Logistic Regression"
optim_title = " (GD) "
init_title = ""

plot_title = rf"{model_title}{init_title}{optim_title}(lr$\approx{lr}$, $\lambda_1={lambda_1}$, $\lambda_2={lambda_2}$)"

epochs_to_plot = torch.load(LOSS_METRICS_FOLDER + f'{model}/epochs_to_plot.pth').to('cpu')

# Train/Test Losses
unreg_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/{unreg_config}loss.pth').to('cpu')
invex_train_loss = torch.load(LOSS_METRICS_FOLDER +
                              f'{model}/Train/with_invex_{config}lambda{lambda_1}_loss.pth').to('cpu')
l2_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda{lambda_2}_loss.pth').to('cpu')
invex_l2_train_loss = torch.load(
    LOSS_METRICS_FOLDER + f'{model}/Train/with_invex_l2_{config}lambda{lambda_1}_l2lambda{lambda_2}_loss.pth').to('cpu')

with torch.no_grad():
    # Plot train/test losses for different models
    plt.figure()
    plt.semilogy(epochs_to_plot, unreg_train_loss, 'k', ls=(0, (5, 1)))  # densely dashed
    plt.semilogy(epochs_to_plot, invex_train_loss, 'r', ls=(0, (5, 5)))  # dashed
    plt.semilogy(epochs_to_plot, l2_train_loss, 'b', ls=(0, (5, 10)))  # loosely dashed
    plt.semilogy(epochs_to_plot, invex_l2_train_loss, 'y', ls=(0, (5, 15)))  # very loosely dashed
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', r'Invex $+ \ell_2$'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title(plot_title)
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Loss/{experiment_name}.pdf', format='pdf', dpi=1200)
    plt.show()
