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
    experiment = Workflow(cifar10_training_subset, cifar10_test_subset, sgd=False, reconstruction=True)

    # Define model and loss function/optimiser
    vae_model = VAE(input_dim=cifar_img_shape[1], num_input_channels=cifar_img_shape[0]).to(dtype=torch.float64)
    vae_model_name = f"{vae_model=}".split('=')[0]  # Gives name of model variable!
    print(vae_model)

    vae_model = InvexRegulariser(vae_model, lamda=experiment.invex_param, multi_output=True)
    vae_model.init_ps(train_dataloader=experiment.training_loader)
    vae_model = vae_model.to(device)

    mse = nn.MSELoss()
    sgd = torch.optim.SGD(vae_model.parameters(), lr=experiment.lr, weight_decay=experiment.l2_param)

    print("\nUsing", device, "\n")

    # Train/test until convergence or specified # epochs
    for epoch in range(experiment.num_epochs):
        converged = experiment.train(vae_model, mse, sgd, epoch)
        experiment.test(vae_model, mse, epoch)
        if converged:
            experiment.truncate_metrics_to_plot()
            break

    # Save model and losses/metrics for further analysis and plotting
    experiment.save(vae_model, vae_model_name)
