from workflow import *
from networks import *

if __name__ == '__main__':
    cifar10_train_dataloader, cifar10_test_dataloader = experiment_setup(cifar10_training_data, cifar10_test_data)
    vae_model = VAE(input_dim=cifar_img_shape[1], num_input_channels=cifar_img_shape[0])
    vae_model_name = f"{vae_model=}".split('=')[0]  # Gives name of model variable!
    print(vae_model)

    vae_model = ModuleWrapper(vae_model, lamda=INVEX_PARAM, multi_output=True)
    vae_model.init_ps(train_dataloader=cifar10_train_dataloader)
    vae_model = vae_model.to(device)

    mse = nn.MSELoss()
    sgd = torch.optim.SGD(vae_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_PARAM)

    print("\nUsing", device, "\n")

    for epoch in range(NUM_EPOCHS):
        converged = train(cifar10_train_dataloader, vae_model, mse, sgd, epoch, reconstruction=True)
        test(cifar10_test_dataloader, vae_model, mse, epoch, reconstruction=True)
        if converged:
            break
    save(vae_model, vae_model_name)
