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
    experiment = Workflow(synthetic_dataset, synthetic_dataset, sgd=False, least_sq=True, synthetic=True,
                          grad_norm_tol=-1, num_epochs=int(1e7))

    # Define model and loss function/optimiser
    least_squares_model = LinearLeastSquares(synthetic_data_A.size(1)).to(dtype=torch.float64)
    least_squares_model_name = f"{least_squares_model=}".split('=')[0]  # Gives name of model variable!
    print(least_squares_model)

    least_squares_model = InvexRegulariser(least_squares_model, lamda=experiment.invex_param)
    least_squares_model.init_ps(train_dataloader=experiment.training_loader)
    least_squares_model = least_squares_model.to(device)

    squared_l2 = nn.MSELoss(reduction='sum')
    sgd = torch.optim.SGD(least_squares_model.parameters(), lr=experiment.lr)

    print("\nUsing", device, "\n")

    # Train/test until convergence or specified # epochs
    for epoch in range(experiment.num_epochs):
        converged = experiment.train(least_squares_model, squared_l2, sgd, epoch)
        if converged:
            experiment.truncate_metrics_to_plot()
            break

    # Save model and losses/metrics for further analysis and plotting
    experiment.save(least_squares_model, least_squares_model_name)
