"""
    Plotting/analysis code with **trained models**

    Author: Adrian Rahul Kamal Rajkamal
"""
import math
import matplotlib.pyplot as plt

import torch

from workflow import LOSS_METRICS_FOLDER, PLOTS_FOLDER

epochs_to_plot = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/epochs_to_plot.pth').to('cpu')

# Train/Test Losses
logistic_unreg_train_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/unregularised_loss.pth').to('cpu')
logistic_unreg_test_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/unregularised_loss.pth').to('cpu')
logistic_invex_train_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_invex_loss.pth').to('cpu')
logistic_invex_test_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_invex_loss.pth').to('cpu')
logistic_l2_train_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_l2_loss.pth').to('cpu')
logistic_l2_test_loss = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_l2_loss.pth').to('cpu')

# Train/Test Accuracies
logistic_unreg_train_acc = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/unregularised_accuracy.pth').to('cpu')
logistic_unreg_test_acc = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/unregularised_accuracy.pth').to('cpu')
logistic_invex_train_acc = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_invex_accuracy.pth').to('cpu')
logistic_invex_test_acc = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_invex_accuracy.pth').to('cpu')
logistic_l2_train_acc = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_l2_accuracy.pth').to('cpu')
logistic_l2_test_acc = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_l2_accuracy.pth').to('cpu')

# L2 Gradient Norms
logistic_unreg_grad_norm = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/unregularised_grad_norm.pth').to('cpu')
logistic_invex_grad_norm = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_invex_grad_norm.pth').to('cpu')
logistic_l2_grad_norm = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_l2_grad_norm.pth').to('cpu')

unregularised_params = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/unregularised_parameters.pth')
invex_params = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_invex_parameters.pth')
l2_params = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_l2_parameters.pth')

with torch.no_grad():
    # Plot train/test losses for different models
    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_train_loss, 'k')
    plt.plot(epochs_to_plot, logistic_invex_train_loss, 'r')
    plt.plot(epochs_to_plot, logistic_l2_train_loss, 'b')
    plt.legend(['Unregularised', 'Invex', 'L2'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/train_unreg_invex_l2_loss.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/train_unreg_invex_l2.eps')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_test_loss, 'k')
    plt.plot(epochs_to_plot, logistic_invex_test_loss, 'r')
    plt.plot(epochs_to_plot, logistic_l2_test_loss, 'b')
    plt.legend(['Unregularised', 'Invex', 'L2'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Loss')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/test_unreg_invex_l2_loss.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/test_unreg_invex_l2.eps')

    # Plot train/test accuracies for different models
    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_train_acc, 'k')
    plt.plot(epochs_to_plot, logistic_invex_train_acc, 'r')
    plt.plot(epochs_to_plot, logistic_l2_train_acc, 'b')
    plt.legend(['Unregularised', 'Invex', 'L2'])
    plt.xlabel('Epochs')
    plt.ylabel('Train Accuracy')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/train_unreg_invex_l2_acc.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/train_unreg_invex_l2.eps')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_test_acc, 'k')
    plt.plot(epochs_to_plot, logistic_invex_test_acc, 'r')
    plt.plot(epochs_to_plot, logistic_l2_test_acc, 'b')
    plt.legend(['Unregularised', 'Invex', 'L2'])
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/test_unreg_invex_l2_acc.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/test_unreg_invex_l2.eps')

    # Plot L2 gradient norm convergence for different models
    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_grad_norm, 'k')
    plt.plot(epochs_to_plot, logistic_invex_grad_norm, 'r')
    plt.plot(epochs_to_plot, logistic_l2_grad_norm, 'b')
    plt.legend(['Unregularised', 'Invex', 'L2'])
    plt.xlabel('Epochs')
    plt.ylabel('L2 Gradient Norm')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/train_unreg_invex_l2_grad.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/train_unreg_invex_l2.eps')

    plt.show()

    # Infinity norm between invex and L2-regularised solutions - measure of similarity
    with open(f'{LOSS_METRICS_FOLDER}/logistic_model/inf_norm_diffs.txt', 'w') as f:
        f.write("||invex - unregularised||_inf = " + str(torch.linalg.vector_norm((invex_params - unregularised_params),
                                                                                  ord=math.inf).item()) + "\n")
        f.write("||L2 - unregularised||_inf = " + str(torch.linalg.vector_norm((l2_params - unregularised_params),
                                                                               ord=math.inf).item()) + "\n")
        f.write("||invex - L2||_inf = " + str(torch.linalg.vector_norm((invex_params - l2_params), ord=math.inf).item())
                + "\n")
