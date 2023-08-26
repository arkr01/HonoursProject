import sys
import os

from torchvision.models import resnet50, ResNet50_Weights

# Handle import issues by referencing parent directory via absolute paths
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from workflow import *
from networks import *

if __name__ == '__main__':
    # Set up data loaders, set hyperparameters, etc.
    experiment = Workflow(cifar10_training_data, cifar10_test_data, lr=0.01, grad_norm_tol=-1, num_epochs=200,
                          img_length=cifar_img_shape[1])
    num_classes = 10
    print(experiment.lr)

    # Define model and loss function/optimiser
    resnet50_last_layer = resnet50(weights=ResNet50_Weights.DEFAULT).to(dtype=torch.float64)

    # 'Freeze' every layer except for the final layer
    for name, param in resnet50_last_layer.named_parameters():
        if 'fc' not in name:  # (only) the last layer is named 'fc' (fully-connected)
            param.requires_grad = False

    # Get number of input features from final layer
    resnet_input_shape = 0
    for name, layer in resnet50_last_layer.named_modules():
        if isinstance(layer, nn.Linear):
            resnet_input_shape = resnet50_last_layer._modules[name].in_features
            break

    # Set number of classes
    resnet50_last_layer.fc = nn.Linear(resnet_input_shape, num_classes)

    # Define generalised model to allow for dropout and batch normalisation
    resnet50_last_layer_model = ResNet50LastLayer(resnet50_last_layer, num_classes, experiment.compare_batch_norm,
                                                  experiment.compare_dropout,
                                                  experiment.dropout_param).to(dtype=torch.float64)
    resnet50_last_layer_model_name = f"{resnet50_last_layer_model=}".split('=')[0]  # Gives name of model variable!
    print(resnet50_last_layer_model)

    resnet50_last_layer_model = ModuleWrapper(resnet50_last_layer_model, lamda=experiment.invex_param,
                                              p_ones=experiment.invex_p_ones)
    resnet50_last_layer_model.init_ps(train_dataloader=experiment.training_loader)
    resnet50_last_layer_model = resnet50_last_layer_model.to(device)

    cross_entropy = nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(resnet50_last_layer_model.parameters(), lr=experiment.lr)

    print("\nUsing", device, "\n")

    # Train/test until convergence or specified # epochs
    for epoch in range(experiment.num_epochs):
        converged = experiment.train(resnet50_last_layer_model, cross_entropy, sgd, epoch)
        experiment.test(resnet50_last_layer_model, cross_entropy, epoch)
        if converged:
            experiment.truncate_metrics_to_plot()
            break

    # Save model and losses/metrics for further analysis and plotting
    experiment.save(resnet50_last_layer_model, resnet50_last_layer_model_name)
