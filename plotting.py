"""
    Plotting/analysis code with **trained models**

    Author: Adrian Rahul Kamal Rajkamal
"""
import math

import matplotlib.pyplot as plt

from train import *


def load_model(filename, model_type='logistic_reg', input_dim=28, num_labels=10, invex_lambda=0.0):
    """
    Create and load a trained model.

    :param filename: name of saved model file
    :param model_type: type of model to be loaded (i.e. multinomial logistic regression, [add others])
    :param input_dim: input dimension of data
    :param num_labels: number of classes/labels (for classification)
    :param invex_lambda: lambda parameter for invex regularisation
    :return: loaded model in evaluation mode
    """
    model = None
    if model_type == 'logistic_reg':
        model = MultinomialLogisticRegression(input_dim=input_dim, num_classes=num_labels)
    model = ModuleWrapper(model, lamda=invex_lambda)
    model.init_ps(train_dataloader=fashion_train_dataloader)
    model = model.to(device)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


# Load trained models
logistic_model_unregularised = load_model(filename='./Models/logistic_model_unregularised.pth')
logistic_model_with_invex = load_model(filename='./Models/logistic_model_with_invex.pth', invex_lambda=INVEX_VAL)
logistic_model_with_l2 = load_model(filename='./Models/logistic_model_with_l2.pth')
logistic_model_with_invex_l2 = load_model(filename='./Models/logistic_model_with_invex_l2.pth',
                                          invex_lambda=INVEX_VAL)

# Load training/test losses/accuracies
training_losses_unregularised = torch.load('./Losses_Metrics/Train/logistic_model_unregularised_loss.pth').to('cpu')
test_losses_unregularised = torch.load('./Losses_Metrics/Test/logistic_model_unregularised_loss.pth').to('cpu')
training_losses_with_invex = torch.load('./Losses_Metrics/Train/logistic_model_with_invex_loss.pth').to('cpu')
test_losses_with_invex = torch.load('./Losses_Metrics/Test/logistic_model_with_invex_loss.pth').to('cpu')
training_losses_with_l2 = torch.load('./Losses_Metrics/Train/logistic_model_with_l2_loss.pth').to('cpu')
test_losses_with_l2 = torch.load('./Losses_Metrics/Test/logistic_model_with_l2_loss.pth').to('cpu')

unregularised_params = torch.load('./Losses_Metrics/logistic_model_unregularised_parameters.pth')
invex_params = torch.load('./Losses_Metrics/logistic_model_with_invex_parameters.pth')
l2_params = torch.load('./Losses_Metrics/logistic_model_with_l2_parameters.pth')

# Generate (and save) plots
with torch.no_grad():
    # Plot train/test losses for different models
    plt.figure()
    plt.plot(epochs_to_plot, training_losses_unregularised, epochs_to_plot, training_losses_with_invex,
             epochs_to_plot, training_losses_with_l2)

    plt.figure()
    plt.plot(epochs_to_plot, test_losses_unregularised, epochs_to_plot, test_losses_with_invex,
             epochs_to_plot, test_losses_with_l2)

    # Compare train/test losses between Invex and L2 regularisation
    plt.figure()
    plt.plot(epochs_to_plot, abs(training_losses_with_invex - training_losses_with_l2))

    plt.figure()
    plt.plot(epochs_to_plot, abs(test_losses_with_invex - test_losses_with_l2))
    plt.show()

    # Infinity norm between invex and L2-regularised solution - measure of similarity
    with open(f'{LOSS_METRICS_FOLDER}inf_norm_diffs.txt', 'w') as f:
        f.write("||invex - L2||_inf = " + str(torch.linalg.vector_norm((invex_params - l2_params), ord=math.inf).item())
                + "\n")
        f.write("||invex - unregularised||_inf = " + str(torch.linalg.vector_norm((invex_params - unregularised_params),
                                                                                  ord=math.inf).item()) + "\n")
        f.write("||L2 - unregularised||_inf = " + str(torch.linalg.vector_norm((l2_params - unregularised_params),
                                                                               ord=math.inf).item()) + "\n")
