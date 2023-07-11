"""
    Plotting/analysis code with **trained models**

    Author: Adrian Rahul Kamal Rajkamal
"""
import matplotlib
import matplotlib.pyplot as plt

import torch

from workflow import LOSS_METRICS_FOLDER, PLOTS_RESULTS_FOLDER, inf

matplotlib.rcParams["text.usetex"] = True  # Allows LaTeX in titles/labels/legends

epochs_to_plot = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/epochs_to_plot.pth').to('cpu')

# What experiment we're plotting
experiment_name = "gd_lr0.01_lambda0.001_both_reduced_1e6epochs_unreg_invex_l2"
if "wie" in LOSS_METRICS_FOLDER or "rgp" in LOSS_METRICS_FOLDER:
    experiment_name = "full_" + experiment_name
unreg_or_gd = "unregularised_" if experiment_name[0] == 's' else "with_gd_"
reg_or_gd = "" if experiment_name[0] == 's' else "gd_"

# Train/Test Losses
logistic_unreg_train_loss = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/Train/'
                                                             f'{unreg_or_gd}lr0.01_lambda0.001_loss.pth').to(
    'cpu')
logistic_unreg_test_loss = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/Test/{unreg_or_gd}lr0.01_lambda0.001_loss.pth').to('cpu')
logistic_invex_train_loss = torch.load(LOSS_METRICS_FOLDER +
                                       f'logistic_model/Train/with_invex_{reg_or_gd}lr0.01_lambda0.001_loss.pth').to('cpu')
logistic_invex_test_loss = torch.load(LOSS_METRICS_FOLDER +
                                      f'logistic_model/Test/with_invex_{reg_or_gd}lr0.01_lambda0.001_loss.pth').to('cpu')
logistic_l2_train_loss = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/Train/with_l2_{reg_or_gd}lr0.01_lambda0.001_loss.pth').to('cpu')
logistic_l2_test_loss = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/Test/with_l2_{reg_or_gd}lr0.01_lambda0.001_loss.pth').to('cpu')
logistic_invex_l2_train_loss = torch.load(LOSS_METRICS_FOLDER +
                                          f'logistic_model/Train/with_invex_l2_{reg_or_gd}lr0.01_lambda0.001_loss.pth').to('cpu')
logistic_invex_l2_test_loss = torch.load(LOSS_METRICS_FOLDER +
                                         f'logistic_model/Test/with_invex_l2_{reg_or_gd}lr0.01_lambda0.001_loss.pth').to('cpu')

# Train/Test Objectives
logistic_unreg_train_obj = torch.load(LOSS_METRICS_FOLDER +
                                      f'logistic_model/Train/{unreg_or_gd}lr0.01_lambda0.001_objective.pth').to('cpu')
logistic_unreg_test_obj = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/Test/{unreg_or_gd}lr0.01_lambda0.001_objective.pth').to('cpu')
logistic_invex_train_obj = torch.load(LOSS_METRICS_FOLDER +
                                      f'logistic_model/Train/with_invex_{reg_or_gd}lr0.01_lambda0.001_objective.pth').to('cpu')
logistic_invex_test_obj = torch.load(LOSS_METRICS_FOLDER +
                                     f'logistic_model/Test/with_invex_{reg_or_gd}lr0.01_lambda0.001_objective.pth').to('cpu')
logistic_l2_train_obj = torch.load(LOSS_METRICS_FOLDER +
                                   f'logistic_model/Train/with_l2_{reg_or_gd}lr0.01_lambda0.001_objective.pth').to('cpu')
logistic_l2_test_obj = torch.load(LOSS_METRICS_FOLDER +
                                  f'logistic_model/Test/with_l2_{reg_or_gd}lr0.01_lambda0.001_objective.pth').to('cpu')
logistic_invex_l2_train_obj = torch.load(LOSS_METRICS_FOLDER +
                                         f'logistic_model/Train/with_invex_l2_{reg_or_gd}lr0.01_lambda0.001_objective.pth').to('cpu')
logistic_invex_l2_test_obj = torch.load(LOSS_METRICS_FOLDER +
                                        f'logistic_model/Test/with_invex_l2_{reg_or_gd}lr0.01_lambda0.001_objective.pth').to('cpu')

# L2 Gradient Norms
logistic_unreg_grad_norm = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/{unreg_or_gd}lr0.01_lambda0.001_grad_norm.pth').to('cpu')
logistic_invex_grad_norm = torch.load(LOSS_METRICS_FOLDER +
                                      f'logistic_model/with_invex_{reg_or_gd}lr0.01_lambda0.001_grad_norm.pth').to('cpu')
logistic_l2_grad_norm = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/with_l2_{reg_or_gd}lr0.01_lambda0.001_grad_norm.pth').to('cpu')
logistic_invex_l2_grad_norm = torch.load(LOSS_METRICS_FOLDER +
                                         f'logistic_model/with_invex_l2_{reg_or_gd}lr0.01_lambda0.001_grad_norm.pth').to('cpu')

# Model parameters (without p variables)
unregularised_params = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/{unreg_or_gd}lr0.01_lambda0.001_parameters.pth')
invex_params = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/with_invex_{reg_or_gd}lr0.01_lambda0.001_parameters.pth')
l2_params = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/with_l2_{reg_or_gd}lr0.01_lambda0.001_parameters.pth')
both_params = torch.load(LOSS_METRICS_FOLDER + f'logistic_model/with_invex_l2_{reg_or_gd}lr0.01_lambda0.001_parameters.pth')

with torch.no_grad():
    # Plot train/test losses for different models
    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_train_loss, 'k')
    plt.plot(epochs_to_plot, logistic_invex_train_loss, 'r', ls='--')
    plt.plot(epochs_to_plot, logistic_l2_train_loss, 'b', ls='-.')
    plt.plot(epochs_to_plot, logistic_invex_l2_train_loss, 'y', ls=':')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title(r'Multinomial Logistic Regression (lr=$0.01$, $\lambda=0.001$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Train/Loss/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Train/Loss/{experiment_name}.eps')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_test_loss, 'k')
    plt.plot(epochs_to_plot, logistic_invex_test_loss, 'r', ls='--')
    plt.plot(epochs_to_plot, logistic_l2_test_loss, 'b', ls='-.')
    plt.plot(epochs_to_plot, logistic_invex_l2_test_loss, 'y', ls=':')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Loss')
    plt.title(r'Multinomial Logistic Regression (lr=$0.01$, $\lambda=0.001$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Test/Loss/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Test/Loss/{experiment_name}.eps')

    # Plot train/test accuracies for different models
    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_train_obj, 'k')
    plt.plot(epochs_to_plot, logistic_invex_train_obj, 'r', ls='--')
    plt.plot(epochs_to_plot, logistic_l2_train_obj, 'b', ls='-.')
    plt.plot(epochs_to_plot, logistic_invex_l2_train_obj, 'y', ls=':')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Objective')
    plt.title(r'Multinomial Logistic Regression (lr=$0.01$, $\lambda=0.001$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Train/Objective/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Train/Objective/{experiment_name}.eps')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_test_obj, 'k')
    plt.plot(epochs_to_plot, logistic_invex_test_obj, 'r', ls='--')
    plt.plot(epochs_to_plot, logistic_l2_test_obj, 'b', ls='-.')
    plt.plot(epochs_to_plot, logistic_invex_l2_test_obj, 'y', ls=':')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Objective')
    plt.title(r'Multinomial Logistic Regression (lr=$0.01$, $\lambda=0.001$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Test/Objective/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Test/Objective/{experiment_name}.eps')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_train_loss, 'k')
    plt.plot(epochs_to_plot, logistic_unreg_train_obj, 'r', ls=(0, (3, 1, 1, 1)))  # dense dash dot
    plt.plot(epochs_to_plot, logistic_invex_train_loss, 'b', ls='--')
    plt.plot(epochs_to_plot, logistic_invex_train_obj, 'c', ls=(0, (3, 5, 1, 5, 1, 5)))  # dash dot dot
    plt.plot(epochs_to_plot, logistic_l2_train_loss, 'm', ls=(0, (1, 1)))  # dense dot
    plt.plot(epochs_to_plot, logistic_l2_train_obj, 'y', ls='-.')
    plt.plot(epochs_to_plot, logistic_invex_l2_train_loss, 'y', ls=':')
    plt.plot(epochs_to_plot, logistic_invex_l2_train_obj, 'y', ls=(0, (3, 1, 1, 1, 1, 1)))  # dense dash dot dot
    plt.legend(['Unregularised L', 'Unregularised O', 'Invex L', 'Invex O', r'$\ell_2$ L', r'$\ell_2$ O', 'Both L',
                'Both O'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train')
    plt.title(r'Multinomial Logistic Regression (lr=$0.01$, $\lambda=0.001$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Train/Both/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Train/Both/{experiment_name}.eps')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_test_loss, 'k')
    plt.plot(epochs_to_plot, logistic_unreg_test_obj, 'r', ls=(0, (3, 1, 1, 1)))  # dense dash dot
    plt.plot(epochs_to_plot, logistic_invex_test_loss, 'b', ls='--')
    plt.plot(epochs_to_plot, logistic_invex_test_obj, 'c', ls=(0, (3, 5, 1, 5, 1, 5)))  # dash dot dot
    plt.plot(epochs_to_plot, logistic_l2_test_loss, 'm', ls=(0, (1, 1)))  # dense dot
    plt.plot(epochs_to_plot, logistic_l2_test_obj, 'y', ls='-.')
    plt.plot(epochs_to_plot, logistic_invex_l2_test_loss, 'y', ls=':')
    plt.plot(epochs_to_plot, logistic_invex_l2_test_obj, 'y', ls=(0, (3, 1, 1, 1, 1, 1)))  # dense dash dot dot
    plt.legend(['Unregularised L', 'Unregularised O', 'Invex L', 'Invex O', r'$\ell_2$ L', r'$\ell_2$ O', 'Both L',
                'Both O'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test')
    plt.title(r'Multinomial Logistic Regression (lr=$0.01$, $\lambda=0.001$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Test/Both/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Test/Both/{experiment_name}.eps')

    # Plot L2 gradient norm convergence for different models
    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_grad_norm, 'k')
    plt.plot(epochs_to_plot, logistic_invex_grad_norm, 'r', ls='--')
    plt.plot(epochs_to_plot, logistic_l2_grad_norm, 'b', ls='-.')
    plt.plot(epochs_to_plot, logistic_invex_l2_grad_norm, 'y', ls=':')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel(r'$\ell_\infty$ Gradient Norm')
    plt.title(r'Multinomial Logistic Regression (lr=$0.01$, $\lambda=0.001$)')
    plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Gradient Norm/{experiment_name}.jpg')
    # plt.savefig(PLOTS_RESULTS_FOLDER + f'logistic_model/Gradient Norm/{experiment_name}.eps')

    plt.show()

    # Infinity norms between all solutions - measure of similarity
    with open(f'{PLOTS_RESULTS_FOLDER}/logistic_model/InfNormDiffs/{experiment_name}.txt', 'w') as f:
        f.write(f'{experiment_name}:\n')
        f.write("||invex - unregularised||_inf = " + str(torch.linalg.vector_norm((invex_params - unregularised_params),
                                                                                  ord=inf).item()) + "\n")
        f.write("||L2 - unregularised||_inf = " + str(torch.linalg.vector_norm((l2_params - unregularised_params),
                                                                               ord=inf).item()) + "\n")
        f.write("||invex - L2||_inf = " + str(torch.linalg.vector_norm((invex_params - l2_params), ord=inf).item())
                + "\n")
        f.write("||invex - both||_inf = " + str(torch.linalg.vector_norm((invex_params - both_params), ord=inf).item())
                + "\n")
        f.write("||both - L2||_inf = " + str(torch.linalg.vector_norm((both_params - l2_params), ord=inf).item())
                + "\n")
        f.write("||both - unregularised||_inf = " + str(torch.linalg.vector_norm((both_params - unregularised_params),
                                                                                 ord=inf).item()) + "\n")
