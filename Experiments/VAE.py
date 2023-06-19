from training import *
from modules import *

if __name__ == '__main__':
    cifar10_train_dataloader, _ = experiment_setup(cifar10_training_data, cifar10_test_data)
    vae_model = VAE(input_dim=cifar_img_shape[1], num_input_channels=cifar_img_shape[0])
    vae_model_name = f"{vae_model=}".split('=')[0]  # Gives name of model variable!
    print(vae_model)

    vae_model = ModuleWrapper(vae_model, lamda=INVEX_PARAM, reconstruction=True)
    vae_model.init_ps(train_dataloader=cifar10_train_dataloader)
    vae_model = vae_model.to(device)

    mse = nn.MSELoss()
    sgd = torch.optim.SGD(vae_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_PARAM)

    print("\nUsing", device, "\n")
    train(cifar10_train_dataloader, vae_model, mse, sgd, reconstruction=True)

    # Model and loss/metrics saving
    save(vae_model, vae_model_name)
