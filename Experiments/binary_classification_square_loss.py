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
    experiment = Workflow(synthetic_dataset_sigmoid, synthetic_dataset_sigmoid, sgd=False, least_sq=True,
                          synthetic=True, grad_norm_tol=-1, num_epochs=int(1e7))

    # Define model and loss function/optimiser
    binary_classification_model = BinaryClassifier(synthetic_data_A.size(1)).to(dtype=torch.float64)
    binary_classification_model_name = f"{binary_classification_model=}".split('=')[0]  # Gives name of model variable!
    print(binary_classification_model)

    binary_classification_model = ModuleWrapper(binary_classification_model, lamda=experiment.invex_param)
    binary_classification_model.init_ps(train_dataloader=experiment.training_loader)
    binary_classification_model = binary_classification_model.to(device)

    squared_l2 = nn.MSELoss(reduction='sum')
    sgd = torch.optim.SGD(binary_classification_model.parameters(), lr=experiment.lr)

    print("\nUsing", device, "\n")

    # Train/test until convergence or specified # epochs
    for epoch in range(experiment.num_epochs):
        converged = experiment.train(binary_classification_model, squared_l2, sgd, epoch)
        if converged:
            experiment.truncate_metrics_to_plot()
            break

    # Save model and losses/metrics for further analysis and plotting
    experiment.save(binary_classification_model, binary_classification_model_name)
