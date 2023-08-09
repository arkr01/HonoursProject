"""
    Plotting/analysis code with **trained models**

    Author: Adrian Rahul Kamal Rajkamal
"""
import matplotlib
import matplotlib.pyplot as plt

import torch

from workflow import LOSS_METRICS_FOLDER, PLOTS_RESULTS_FOLDER, inf

matplotlib.rcParams["text.usetex"] = True  # Allows LaTeX in titles/labels/legends


# What experiment we're plotting
model = "resnet50_model"
plot_title = "ResNet50 Classification"
lr = 0.1
lambda_1 = 0.1
lambda_2 = 0.01

experiment_name = "sgd_lr0.1_lambda0.1_l2lambda0.01_both_unreg_invex_l2"
standard_lr = experiment_name.__contains__("lrst")

unreg_or_gd = "unregularised_" if experiment_name[0] == 's' and standard_lr else "with_"
reg_or_gd = "" if experiment_name[0] == 's' else "gd_"
lr_val = "" if standard_lr else f"lr{lr}_"
config = reg_or_gd + lr_val
unreg_config = unreg_or_gd + config

if "wie" in LOSS_METRICS_FOLDER or "rgp" in LOSS_METRICS_FOLDER:
    experiment_name = "full_" + experiment_name

epochs_to_plot = torch.load(LOSS_METRICS_FOLDER + f'{model}/epochs_to_plot.pth').to('cpu')

# Train/Test Losses
unreg_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/{unreg_config}loss.pth').to('cpu')
unreg_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/{unreg_config}loss.pth').to('cpu')
invex_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_invex_{config}lambda0.1_loss.pth').to('cpu')
invex_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.1_loss.pth').to('cpu')
l2_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.01_loss.pth').to('cpu')
l2_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.01_loss.pth').to('cpu')
invex_l2_train_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_invex_l2'
                                                       f'_{config}lambda0.1_l2lambda0.01_loss.pth').to('cpu')
invex_l2_test_loss = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_l2'
                                                      f'_{config}lambda0.1_l2lambda0.01_loss.pth').to('cpu')

# Train/Test Objectives
# unreg_train_obj = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/{unreg_config}objective.pth').to('cpu')
# unreg_test_obj = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/{unreg_config}objective.pth').to('cpu')
# invex_train_obj = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_invex'
#                                                    f'_{config}lambda0.1_objective.pth').to('cpu')
# invex_test_obj = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_{config}lambda0.1_objective.pth').to(
#     'cpu')
# l2_train_obj = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_l2_{config}l2lambda0.01_objective.pth').to(
#     'cpu')
# l2_test_obj = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_l2_{config}l2lambda0.01_objective.pth').to('cpu')
# invex_l2_train_obj = torch.load(LOSS_METRICS_FOLDER + f'{model}/Train/with_invex_l2'
#                                                       f'_{config}lambda0.1_l2lambda0.01_objective.pth').to('cpu')
# invex_l2_test_obj = torch.load(LOSS_METRICS_FOLDER + f'{model}/Test/with_invex_l2'
#                                                      f'_{config}lambda0.1_l2lambda0.01_objective.pth').to('cpu')

# L2 Gradient Norms
unreg_grad_norm = torch.load(LOSS_METRICS_FOLDER + f'{model}/{unreg_config}grad_norm.pth').to('cpu')
invex_grad_norm = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_invex_{config}lambda0.1_grad_norm.pth').to('cpu')
l2_grad_norm = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_l2_{config}l2lambda0.01_grad_norm.pth').to('cpu')
invex_l2_grad_norm = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_invex_l2'
                                                      f'_{config}lambda0.1_l2lambda0.01_grad_norm.pth').to('cpu')

# Model parameters (without p variables)
unregularised_params = torch.load(LOSS_METRICS_FOLDER + f'{model}/{unreg_config}parameters.pth')
invex_params = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_invex_{config}lambda0.1_parameters.pth')
l2_params = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_l2_{config}l2lambda0.01_parameters.pth')
both_params = torch.load(LOSS_METRICS_FOLDER + f'{model}/with_invex_l2_'
                                               f'{config}lambda0.1_l2lambda0.01_parameters.pth')

with torch.no_grad():
    # Plot train/test losses for different models
    plt.figure()
    plt.plot(epochs_to_plot, unreg_train_loss, 'k', ls=(0, (5, 1)))  # densely dashed
    plt.plot(epochs_to_plot, invex_train_loss, 'r', ls=(0, (5, 5)))  # dashed
    plt.plot(epochs_to_plot, l2_train_loss, 'b', ls=(0, (5, 10)))  # loosely dashed
    plt.plot(epochs_to_plot, invex_l2_train_loss, 'y', ls=(0, (5, 15)))  # very loosely dashed
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title(rf'{plot_title} (lr$\approx{lr}$, $\lambda_1={lambda_1}$, $\lambda_2={lambda_2}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Loss/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Loss/{experiment_name}.eps')

    plt.figure()
    plt.plot(epochs_to_plot, unreg_test_loss, 'k', ls=(0, (5, 1)))
    plt.plot(epochs_to_plot, invex_test_loss, 'r', ls=(0, (5, 5)))
    plt.plot(epochs_to_plot, l2_test_loss, 'b', ls=(0, (5, 10)))
    plt.plot(epochs_to_plot, invex_l2_test_loss, 'y', ls=(0, (5, 15)))
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Loss')
    plt.title(rf'{plot_title} (lr$\approx{lr}$, $\lambda_1={lambda_1}$, $\lambda_2={lambda_2}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Loss/{experiment_name}.jpg')
    # # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Loss/{experiment_name}.eps')

    # Plot train/test objectives for different models
    # plt.figure()
    # plt.plot(epochs_to_plot, unreg_train_obj, 'k', ls=(0, (1, 1)))  # densely dotted
    # plt.plot(epochs_to_plot, invex_train_obj, 'r', ls=(0, (1, 5)))  # dotted
    # plt.plot(epochs_to_plot, l2_train_obj, 'b', ls=(0, (1, 10)))  # loosely dotted
    # plt.plot(epochs_to_plot, invex_l2_train_obj, 'y', ls=(0, (1, 15)))  # very loosely dotted
    # plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Avg Train Objective')
    # plt.title(rf'{plot_title} (lr$\approx{lr}$, $\lambda_1={lambda_1}$, $\lambda_2={lambda_2}$)')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Objective/{experiment_name}.jpg')
    # # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Objective/{experiment_name}.eps')
    #
    # plt.figure()
    # plt.plot(epochs_to_plot, unreg_test_obj, 'k', ls=(0, (1, 1)))
    # plt.plot(epochs_to_plot, invex_test_obj, 'r', ls=(0, (1, 5)))
    # plt.plot(epochs_to_plot, l2_test_obj, 'b', ls=(0, (5, 10)))
    # plt.plot(epochs_to_plot, invex_l2_test_obj, 'y', ls=(0, (1, 15)))
    # plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Avg Test Objective')
    # plt.title(rf'{plot_title} (lr$\approx{lr}$, $\lambda_1={lambda_1}$, $\lambda_2={lambda_2}$)')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Objective/{experiment_name}.jpg')
    # # # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Objective/{experiment_name}.eps')
    #
    # plt.figure()
    # plt.plot(epochs_to_plot, unreg_train_loss, 'k', ls=(0, (5, 1)))
    # plt.plot(epochs_to_plot, unreg_train_obj, 'k', ls=(0, (1, 1)))
    # plt.plot(epochs_to_plot, invex_train_loss, 'r', ls=(0, (5, 5)))
    # plt.plot(epochs_to_plot, invex_train_obj, 'r', ls=(0, (1, 5)))
    # plt.plot(epochs_to_plot, l2_train_loss, 'b', ls=(0, (5, 10)))  # dense dot
    # plt.plot(epochs_to_plot, l2_train_obj, 'b', ls=(0, (1, 10)))
    # plt.plot(epochs_to_plot, invex_l2_train_loss, 'y', ls=(0, (5, 15)))
    # plt.plot(epochs_to_plot, invex_l2_train_obj, 'y', ls=(0, (1, 15)))
    # plt.legend(['Unregularised L', 'Unregularised O', 'Invex L', 'Invex O', r'$\ell_2$ L', r'$\ell_2$ O', 'Both L',
    #             'Both O'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Avg Train')
    # plt.title(rf'{plot_title} (lr$\approx{lr}$, $\lambda_1={lambda_1}$, $\lambda_2={lambda_2}$)')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Both/{experiment_name}.jpg')
    # # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Train/Both/{experiment_name}.eps')
    #
    # plt.figure()
    # plt.plot(epochs_to_plot, unreg_test_loss, 'k', ls=(0, (5, 1)))
    # plt.plot(epochs_to_plot, unreg_test_obj, 'k', ls=(0, (1, 1)))
    # plt.plot(epochs_to_plot, invex_test_loss, 'r', ls=(0, (5, 5)))
    # plt.plot(epochs_to_plot, invex_test_obj, 'r', ls=(0, (1, 5)))
    # plt.plot(epochs_to_plot, l2_test_loss, 'b', ls=(0, (5, 10)))
    # plt.plot(epochs_to_plot, l2_test_obj, 'b', ls=(0, (1, 10)))
    # plt.plot(epochs_to_plot, invex_l2_test_loss, 'y', ls=(0, (5, 15)))
    # plt.plot(epochs_to_plot, invex_l2_test_obj, 'y', ls=(0, (1, 15)))
    # plt.legend(['Unregularised L', 'Unregularised O', 'Invex L', 'Invex O', r'$\ell_2$ L', r'$\ell_2$ O', 'Both L',
    #             'Both O'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Avg Test')
    # plt.title(rf'{plot_title} (lr$\approx{lr}$, $\lambda_1={lambda_1}$, $\lambda_2={lambda_2}$)')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Both/{experiment_name}.jpg')
    # # # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Test/Both/{experiment_name}.eps')

    # Plot L2 gradient norm convergence for different models
    plt.figure()
    plt.semilogy(epochs_to_plot, unreg_grad_norm, 'k')
    plt.semilogy(epochs_to_plot, invex_grad_norm, 'r')
    plt.semilogy(epochs_to_plot, l2_grad_norm, 'b')
    plt.semilogy(epochs_to_plot, invex_l2_grad_norm, 'y')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel(r'$\ell_\infty$ Gradient Norm')
    plt.title(rf'{plot_title} (lr$\approx{lr}$, $\lambda_1={lambda_1}$, $\lambda_2={lambda_2}$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'{model}/Gradient Norm/{experiment_name}.eps')

    plt.show()

    # Infinity norms between all solutions - measure of similarity
    with open(f'{PLOTS_RESULTS_FOLDER}/{model}/InfNormDiffs/{experiment_name}.txt', 'w') as f:
        f.write(f'{experiment_name}:\n')
        f.write("||invex - unregularised||_inf = " +
                str(torch.linalg.vector_norm((invex_params - unregularised_params), ord=inf).item()) + "\n")
        f.write("||L2 - unregularised||_inf = " +
                str(torch.linalg.vector_norm((l2_params - unregularised_params), ord=inf).item()) + "\n")
        f.write("||invex - L2||_inf = " +
                str(torch.linalg.vector_norm((invex_params - l2_params), ord=inf).item()) + "\n")
        f.write("||invex - both||_inf = " +
                str(torch.linalg.vector_norm((invex_params - both_params), ord=inf).item()) + "\n")
        f.write("||both - L2||_inf = " +
                str(torch.linalg.vector_norm((both_params - l2_params), ord=inf).item()) + "\n")
        f.write("||both - unregularised||_inf = " +
                str(torch.linalg.vector_norm((both_params - unregularised_params), ord=inf).item()) + "\n")

        f.write('\n')

        f.write("||invex - unregularised||_2 = " +
                str(torch.linalg.vector_norm((invex_params - unregularised_params)).item()) + "\n")
        f.write("||L2 - unregularised||_2 = " +
                str(torch.linalg.vector_norm((l2_params - unregularised_params)).item()) + "\n")
        f.write("||invex - L2||_2 = " + str(torch.linalg.vector_norm((invex_params - l2_params)).item())
                + "\n")
        f.write("||invex - both||_2 = " + str(torch.linalg.vector_norm((invex_params - both_params)).item())
                + "\n")
        f.write("||both - L2||_2 = " + str(torch.linalg.vector_norm((both_params - l2_params)).item())
                + "\n")
        f.write("||both - unregularised||_2 = " +
                str(torch.linalg.vector_norm((both_params - unregularised_params)).item()) + "\n")
