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
    experiment = Workflow(fashion_training_two_class, fashion_test_two_class, sgd=False, binary_log_reg=True,
                          grad_norm_tol=-1)  # int(1e7) for synthetic

    # Define model and loss function/optimiser
    binary_logistic_model_fashion = MultinomialLogisticRegression(fashion_img_length, 1).to(dtype=torch.float64)
    binary_logistic_model_fashion_name = f"{binary_logistic_model_fashion=}".split('=')[0]  # Gives name of model variable!
    print(binary_logistic_model_fashion)

    binary_logistic_model_fashion = InvexRegulariser(binary_logistic_model_fashion, lamda=experiment.invex_param)
    binary_logistic_model_fashion.init_ps(train_dataloader=experiment.training_loader)
    binary_logistic_model_fashion = binary_logistic_model_fashion.to(device)

    bce = nn.BCELoss()
    sgd = torch.optim.SGD(binary_logistic_model_fashion.parameters(), lr=experiment.lr)

    print("\nUsing", device, "\n")

    # Train/test until convergence or specified # epochs
    for epoch in range(experiment.num_epochs):
        converged = experiment.train(binary_logistic_model_fashion, bce, sgd, epoch)
        # Synthetic data has no test set
        if not experiment.synthetic:
            experiment.test(binary_logistic_model_fashion, bce, epoch)
        if converged:
            experiment.truncate_metrics_to_plot()
            break

    # Save model and losses/metrics for further analysis and plotting
    experiment.save(binary_logistic_model_fashion, binary_logistic_model_fashion_name)
