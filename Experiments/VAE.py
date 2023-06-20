from workflow import *
from networks import *

import matplotlib.pyplot as plt

if __name__ == '__main__':
    experiment = Workflow(cifar10_training_data, cifar10_test_data, num_epochs=10)
    vae_model = VAE(input_dim=cifar_img_shape[1], num_input_channels=cifar_img_shape[0])
    vae_model_name = f"{vae_model=}".split('=')[0]  # Gives name of model variable!
    print(vae_model)

    vae_model = ModuleWrapper(vae_model, lamda=experiment.invex_param, multi_output=True)
    vae_model.init_ps(train_dataloader=experiment.training_loader)
    vae_model = vae_model.to(device)

    mse = nn.MSELoss()
    sgd = torch.optim.SGD(vae_model.parameters(), lr=experiment.lr, weight_decay=experiment.l2_param)

    print("\nUsing", device, "\n")

    for epoch in range(experiment.num_epochs):
        converged = experiment.train(vae_model, mse, sgd, epoch, reconstruction=True)
        experiment.test(vae_model, mse, epoch, reconstruction=True)
        if converged:
            experiment.truncate_losses_to_plot()
            break
    experiment.save(vae_model, vae_model_name)

    with torch.no_grad():
        # Plot train/test losses for different models
        plt.figure()
        plt.plot(experiment.epochs_to_plot.to('cpu'), experiment.avg_training_losses_to_plot.to('cpu'))
        plt.xlabel('Epochs')
        plt.ylabel('Avg Train Loss')
        plt.title('VAE (with Invex)')

        # TODO Fix plot saving
        plt.savefig('vae_model_with_invex_train.jpg')
        plt.savefig('vae_model_with_invex_train.eps')

        plt.figure()
        plt.plot(experiment.epochs_to_plot.to('cpu'), experiment.avg_test_losses_to_plot.to('cpu'))
        plt.xlabel('Epochs')
        plt.ylabel('Avg Test Loss')
        plt.title('VAE (with Invex)')
        plt.savefig('vae_model_with_invex_test.jpg')
        plt.savefig('vae_model_with_invex_test.eps')

        plt.show()
