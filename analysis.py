"""
    Plotting/analysis code with **trained models**

    Author: Adrian Rahul Kamal Rajkamal
"""
import matplotlib
import matplotlib.pyplot as plt

import torch

from workflow import LOSS_METRICS_FOLDER, PLOTS_FOLDER, inf

matplotlib.rcParams["text.usetex"] = True  # Allows LaTeX in titles/labels/legends

epochs_to_plot = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/epochs_to_plot.pth').to('cpu')

# Train/Test Losses
logistic_unreg_train_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_gd_loss.pth').to('cpu')
logistic_unreg_test_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_gd_loss.pth').to('cpu')
logistic_invex_train_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_invex_gd_loss.pth').to('cpu')
logistic_invex_test_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_invex_gd_loss.pth').to('cpu')
logistic_l2_train_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_l2_gd_loss.pth').to('cpu')
logistic_l2_test_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_l2_gd_loss.pth').to('cpu')
logistic_invex_l2_train_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_invex_l2_gd_loss.pth').to(
    'cpu')
logistic_invex_l2_test_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_invex_l2_gd_loss.pth').to(
    'cpu')

# Train/Test Objectives
logistic_unreg_train_obj = torch.load(LOSS_METRICS_FOLDER +
                                      'logistic_model/Train/with_gd_objective.pth').to('cpu')
logistic_unreg_test_obj = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_gd_objective.pth').to('cpu')
logistic_invex_train_obj = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_invex_gd_objective.pth').to('cpu'
                                                                                                                   )
logistic_invex_test_obj = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_invex_gd_objective.pth').to('cpu')
logistic_l2_train_obj = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_l2_gd_objective.pth').to('cpu')
logistic_l2_test_obj = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_l2_gd_objective.pth').to('cpu')
logistic_invex_l2_train_obj = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_invex_l2_gd_objective.pth').to(
    'cpu')
logistic_invex_l2_test_obj = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_invex_l2_gd_objective.pth').to(
    'cpu')

# L2 Gradient Norms
logistic_unreg_grad_norm = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_gd_grad_norm.pth').to('cpu')
logistic_invex_grad_norm = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_invex_gd_grad_norm.pth').to('cpu')
logistic_l2_grad_norm = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_l2_gd_grad_norm.pth').to('cpu')
logistic_invex_l2_grad_norm = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_invex_l2_gd_grad_norm.pth').to(
    'cpu')

unregularised_params = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_gd_parameters.pth')
invex_params = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_invex_gd_parameters.pth')
l2_params = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_l2_gd_parameters.pth')

with torch.no_grad():
    # Plot train/test losses for different models
    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_train_loss, 'k')
    plt.plot(epochs_to_plot, logistic_invex_train_loss, 'r', ls='--')
    plt.plot(epochs_to_plot, logistic_l2_train_loss, 'b')
    plt.plot(epochs_to_plot, logistic_invex_l2_train_loss, 'y', ls=':')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/Train/unreg_invex_l2_loss.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/Train/unreg_invex_l2_loss.eps')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_test_loss, 'k')
    plt.plot(epochs_to_plot, logistic_invex_test_loss, 'r', ls='--')
    plt.plot(epochs_to_plot, logistic_l2_test_loss, 'b')
    plt.plot(epochs_to_plot, logistic_invex_l2_test_loss, 'y', ls=':')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Loss')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/Test/unreg_invex_l2_loss.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/Test/unreg_invex_l2_loss.eps')

    # Plot train/test accuracies for different models
    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_train_obj, 'k')
    plt.plot(epochs_to_plot, logistic_invex_train_obj, 'r')
    plt.plot(epochs_to_plot, logistic_l2_train_obj, 'b')
    plt.plot(epochs_to_plot, logistic_invex_l2_train_obj, 'y')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel('Train Objective')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/Train/unreg_invex_l2_obj.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/Train/unreg_invex_l2_obj.eps')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_test_obj, 'k')
    plt.plot(epochs_to_plot, logistic_invex_test_obj, 'r')
    plt.plot(epochs_to_plot, logistic_l2_test_obj, 'b')
    plt.plot(epochs_to_plot, logistic_invex_l2_test_obj, 'y')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel('Test Objective')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/Test/unreg_invex_l2_obj.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/Test/unreg_invex_l2_obj.eps')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_train_loss, 'k')
    plt.plot(epochs_to_plot, logistic_invex_train_loss, 'r')
    plt.plot(epochs_to_plot, logistic_l2_train_loss, 'b')
    plt.plot(epochs_to_plot, logistic_invex_l2_train_loss, 'y', ls='--')
    plt.plot(epochs_to_plot, logistic_unreg_train_obj, 'c')
    plt.plot(epochs_to_plot, logistic_invex_train_obj, 'm')
    plt.plot(epochs_to_plot, logistic_l2_train_obj, 'y')
    plt.plot(epochs_to_plot, logistic_invex_l2_train_obj, 'y', ls=':')
    plt.legend(['Unregularised L', 'Invex L', r'$\ell_2$ L', 'Both L', 'Unregularised O', 'Invex O', r'$\ell_2$ O',
                'Both O'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/Test/unreg_invex_l2_both.jpg')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_test_loss, 'k')
    plt.plot(epochs_to_plot, logistic_invex_test_loss, 'r')
    plt.plot(epochs_to_plot, logistic_l2_test_loss, 'b')
    plt.plot(epochs_to_plot, logistic_invex_l2_test_loss, 'y', ls='--')
    plt.plot(epochs_to_plot, logistic_unreg_test_obj, 'c')
    plt.plot(epochs_to_plot, logistic_invex_test_obj, 'm')
    plt.plot(epochs_to_plot, logistic_l2_test_obj, 'y')
    plt.plot(epochs_to_plot, logistic_invex_l2_test_obj, 'y', ls=':')
    plt.legend(['Unregularised L', 'Invex L', r'$\ell_2$ L', 'Both L', 'Unregularised O', 'Invex O', r'$\ell_2$ O',
                'Both O'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/Test/unreg_invex_l2_both.jpg')

    # Plot L2 gradient norm convergence for different models
    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_grad_norm, 'k')
    plt.plot(epochs_to_plot, logistic_invex_grad_norm, 'r')
    plt.plot(epochs_to_plot, logistic_l2_grad_norm, 'b')
    plt.plot(epochs_to_plot, logistic_invex_l2_grad_norm, 'y')
    plt.legend(['Unregularised', 'Invex', r'$\ell_2$', 'Both'])
    plt.xlabel('Epochs')
    plt.ylabel(r'$\ell_\infty$ Gradient Norm')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/unreg_invex_l2_grad.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/unreg_invex_l2_grad.eps')

    plt.show()

    # Infinity norm between invex and L2-regularised solutions - measure of similarity
    with open(f'{LOSS_METRICS_FOLDER}/logistic_model/inf_norm_diffs.txt', 'w') as f:
        f.write("||invex - unregularised||_inf = " + str(torch.linalg.vector_norm((invex_params - unregularised_params),
                                                                                  ord=inf).item()) + "\n")
        f.write("||L2 - unregularised||_inf = " + str(torch.linalg.vector_norm((l2_params - unregularised_params),
                                                                               ord=inf).item()) + "\n")
        f.write("||invex - L2||_inf = " + str(torch.linalg.vector_norm((invex_params - l2_params), ord=inf).item())
                + "\n")
