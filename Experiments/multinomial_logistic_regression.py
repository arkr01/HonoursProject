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
    experiment = Workflow(fashion_training_subset, fashion_test_subset, sgd=False, grad_norm_tol=-1,
                          num_epochs=int(2e6))

    # Define model and loss function/optimiser
    logistic_model_old = MultinomialLogisticRegression(input_dim=fashion_img_length, num_classes=num_fashion_classes,
                                                       invex=experiment.compare_invex).to(dtype=torch.float64)
    logistic_model_old_name = f"{logistic_model_old=}".split('=')[0]  # Gives name of model variable!
    print(logistic_model_old)

    logistic_model_old = ModuleWrapper(logistic_model_old, lamda=experiment.invex_param, log_out=True)
    logistic_model_old.init_ps(train_dataloader=experiment.training_loader)
    logistic_model_old = logistic_model_old.to(device)

    cross_entropy = nn.NLLLoss()
    sgd = torch.optim.SGD(logistic_model_old.parameters(), lr=experiment.lr)

    print("\nUsing", device, "\n")

    # Train/test until convergence or specified # epochs
    for epoch in range(experiment.num_epochs):
        converged = experiment.train(logistic_model_old, cross_entropy, sgd, epoch)
        experiment.test(logistic_model_old, cross_entropy, epoch)
        if converged:
            experiment.truncate_metrics_to_plot()
            break

    # Save model and losses/metrics for further analysis and plotting
    experiment.save(logistic_model_old, logistic_model_old_name)
