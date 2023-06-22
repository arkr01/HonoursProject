import sys
import os

# Handle import issues by referencing parent directory via absolute paths
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from workflow import *
from networks import *

if __name__ == '__main__':
    # Set up data loaders, set hyperparameters, etc.
    experiment = Workflow(fashion_training_data, fashion_test_data, num_epochs=30)

    # Define model and loss function/optimiser
    logistic_model = MultinomialLogisticRegression(input_dim=fashion_img_length, num_classes=num_fashion_classes)
    logistic_model_name = f"{logistic_model=}".split('=')[0]  # Gives name of model variable!
    print(logistic_model)

    logistic_model = ModuleWrapper(logistic_model, lamda=experiment.invex_param)
    logistic_model.init_ps(train_dataloader=experiment.training_loader)
    logistic_model = logistic_model.to(device)

    cross_entropy = nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(logistic_model.parameters(), lr=experiment.lr, weight_decay=experiment.l2_param)

    print("\nUsing", device, "\n")

    # Train/test until convergence or specified # epochs
    for epoch in range(experiment.num_epochs):
        converged = experiment.train(logistic_model, cross_entropy, sgd, epoch)
        experiment.test(logistic_model, cross_entropy, epoch)
        if converged:
            experiment.truncate_losses_to_plot()
            break

    # Save model and losses/metrics for further analysis and plotting
    experiment.save(logistic_model, logistic_model_name)
