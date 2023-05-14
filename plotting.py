"""
    Plotting/analysis code with **trained models**

    Author: Adrian Rahul Kamal Rajkamal
"""
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
    model = NNClassifier()
    if model_type == 'logistic_reg':
        model = MultinomialLogisticRegression(input_dim=input_dim, num_classes=num_labels)
    model = ModuleWrapper(model, lamda=invex_lambda)
    model.init_ps(train_dataloader=train_dataloader)
    model = model.to(device)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


# Load trained models
logistic_model_unregularised = load_model(filename='logistic_model_unregularised.pth')
logistic_model_with_invex = load_model(filename='logistic_model_with_invex.pth', invex_lambda=INVEX_LAMBDA)
logistic_model_with_l2 = load_model(filename='logistic_model_with_l2.pth')
logistic_model_with_invex_l2 = load_model(filename='logistic_model_with_invex_l2.pth', invex_lambda=INVEX_LAMBDA)

# Load training/test losses/accuracies
training_losses_with_invex = torch.load('logistic_model_with_invex_training_loss.pth')
test_losses_with_invex = torch.load('logistic_model_with_invex_training_loss.pth')

# Generate (and save) plots
with torch.no_grad():
    plt.figure()
    plt.loglog(epochs_to_plot, training_losses_with_invex.to('cpu'))
    plt.figure()
    plt.loglog(epochs_to_plot, test_losses_with_invex.to('cpu'))
    plt.show()
