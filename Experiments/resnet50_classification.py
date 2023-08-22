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
    experiment = Workflow(cifar10_training_data, cifar10_test_data, lr=0.01, grad_norm_tol=-1, num_epochs=200)
    print(experiment.lr)

    # Define model and loss function/optimiser
    resnet50_model = ResNet50(experiment.compare_batch_norm, experiment.compare_dropout, experiment.dropout_param,
                              num_classes=10).to(dtype=torch.float64)
    resnet50_model_name = f"{resnet50_model=}".split('=')[0]  # Gives name of model variable!
    resnet50_model_name += "_ones" if experiment.invex_p_ones else ""
    print(resnet50_model)

    resnet50_model = ModuleWrapper(resnet50_model, lamda=experiment.invex_param, p_ones=experiment.invex_p_ones)
    resnet50_model.init_ps(train_dataloader=experiment.training_loader)
    resnet50_model = resnet50_model.to(device)

    cross_entropy = nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(resnet50_model.parameters(), lr=experiment.lr)

    print("\nUsing", device, "\n")

    # Train/test until convergence or specified # epochs
    for epoch in range(experiment.num_epochs):
        converged = experiment.train(resnet50_model, cross_entropy, sgd, epoch)
        experiment.test(resnet50_model, cross_entropy, epoch)
        if converged:
            experiment.truncate_metrics_to_plot()
            break

    # Save model and losses/metrics for further analysis and plotting
    experiment.save(resnet50_model, resnet50_model_name)
