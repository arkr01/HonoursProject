"""
    Plotting/analysis code with **trained models**

    Author: Adrian Rahul Kamal Rajkamal
"""
import math
import matplotlib.pyplot as plt

import torch

from workflow import LOSS_METRICS_FOLDER, PLOTS_FOLDER

epochs_to_plot = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/epochs_to_plot.pth').to('cpu')

logistic_unreg_train = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/unregularised_loss.pth').to('cpu')
logistic_unreg_test = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/unregularised_loss.pth').to('cpu')

logistic_invex_train = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_invex_loss.pth').to('cpu')
logistic_invex_test = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_invex_loss.pth').to('cpu')

logistic_l2_train = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Train/with_l2_loss.pth').to('cpu')
logistic_l2_test = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/Test/with_l2_loss.pth').to('cpu')

unregularised_params = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/unregularised_parameters.pth')
invex_params = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_invex_parameters.pth')
l2_params = torch.load(LOSS_METRICS_FOLDER + 'logistic_model/with_l2_parameters.pth')

with torch.no_grad():
    # Plot train/test losses for different models
    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_train, 'k')
    plt.plot(epochs_to_plot, logistic_invex_train, 'r')
    # plt.plot(epochs_to_plot, logistic_l2_train, 'b')
    plt.legend(['Unregularised', 'Invex', 'L2'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Train Loss')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/train_unreg_invex_lambda100.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/train_unreg_invex.eps')

    plt.figure()
    plt.plot(epochs_to_plot, logistic_unreg_test, 'k')
    plt.plot(epochs_to_plot, logistic_invex_test, 'r')
    # plt.plot(epochs_to_plot, logistic_l2_test, 'b')
    plt.legend(['Unregularised', 'Invex', 'L2'])
    plt.xlabel('Epochs')
    plt.ylabel('Avg Test Loss')
    plt.title('Multinomial Logistic Regression')
    plt.savefig(PLOTS_FOLDER + 'logistic_model/test_unreg_invex_lambda100.jpg')
    # plt.savefig(PLOTS_FOLDER + 'logistic_model/test_unreg_invex.eps')

    plt.show()

    # Infinity norm between invex and L2-regularised solution - measure of similarity
    with open(f'{LOSS_METRICS_FOLDER}/logistic_model/inf_norm_diffs.txt', 'w') as f:
        f.write("||invex - L2||_inf = " + str(torch.linalg.vector_norm((invex_params - l2_params), ord=math.inf).item())
                + "\n")
        f.write("||invex - unregularised||_inf = " + str(torch.linalg.vector_norm((invex_params - unregularised_params),
                                                                                  ord=math.inf).item()) + "\n")
        f.write("||L2 - unregularised||_inf = " + str(torch.linalg.vector_norm((l2_params - unregularised_params),
                                                                               ord=math.inf).item()) + "\n")
