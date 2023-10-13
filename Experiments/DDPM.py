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
    experiment = Workflow(cifar10_training_data, cifar10_test_data, lr=0.01, num_epochs=200, diffusion=True)
    print(experiment.lr)

    # Define model and loss function/optimiser
    diffusion_setup = DiffusionSetup(input_dim=cifar_img_shape[1], device=device)
    unet_model = UNet(input_dim=cifar_img_shape[1], device=device, compare_dropout=experiment.compare_dropout,
                      dropout_param=experiment.dropout_param,
                      compare_batch_norm=experiment.compare_batch_norm).to(dtype=torch.float64)
    print(unet_model)

    # Initialise parameters to 0 or 1 if needed
    if experiment.zero_init or experiment.one_init:
        init_val = int(experiment.one_init)
        for _, param in unet_model.named_parameters():
            param.detach().fill_(init_val)

    unet_model = InvexRegulariser(unet_model, lamda=experiment.invex_param, p_ones=experiment.invex_p_ones,
                                  diffusion=experiment.diffusion)
    unet_model.init_ps(train_dataloader=experiment.training_loader)
    unet_model = unet_model.to(device)

    diffusion_model = unet_model, diffusion_setup
    diffusion_model_name = f"{diffusion_model=}".split('=')[0]  # Gives name of model variable!

    mse = nn.MSELoss()
    if experiment.lbfgs:
        optimiser = torch.optim.LBFGS(unet_model.parameters(), lr=experiment.lr, history_size=20)
    else:
        optimiser = torch.optim.SGD(unet_model.parameters(), lr=experiment.lr)

    print("\nUsing", device, "\n")

    # Train/test until convergence or specified # epochs
    for epoch in range(experiment.num_epochs):
        converged = experiment.train(diffusion_model, mse, optimiser, epoch)
        experiment.test(diffusion_model, mse, epoch)
        if converged:
            experiment.truncate_metrics_to_plot()
            break

    # Save model and losses/metrics for further analysis and plotting
    experiment.save(diffusion_model, diffusion_model_name)
