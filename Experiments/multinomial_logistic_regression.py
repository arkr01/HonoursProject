from workflow import *
from networks import *

import matplotlib.pyplot as plt

if __name__ == '__main__':
    experiment = Workflow(fashion_training_data, fashion_test_data, num_epochs=30)
    logistic_model = MultinomialLogisticRegression(input_dim=fashion_img_length, num_classes=num_fashion_classes)
    logistic_model_name = f"{logistic_model=}".split('=')[0]  # Gives name of model variable!
    print(logistic_model)

    logistic_model = ModuleWrapper(logistic_model, lamda=experiment.invex_param)
    logistic_model.init_ps(train_dataloader=experiment.training_loader)
    logistic_model = logistic_model.to(device)

    cross_entropy = nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(logistic_model.parameters(), lr=experiment.lr, weight_decay=experiment.l2_param)

    print("\nUsing", device, "\n")

    for epoch in range(experiment.num_epochs):
        converged = experiment.train(logistic_model, cross_entropy, sgd, epoch)
        experiment.test(logistic_model, cross_entropy, epoch)
        if converged:
            experiment.truncate_losses_to_plot()
            break
    experiment.save(logistic_model, logistic_model_name)

    with torch.no_grad():
        # Plot train/test losses for different models
        plt.figure()
        plt.plot(experiment.epochs_to_plot.to('cpu'), experiment.avg_training_losses_to_plot.to('cpu'))
        plt.xlabel('Epochs')
        plt.ylabel('Avg Train Loss')
        plt.title('Multinomial Logistic Regression (with Invex)')

        # TODO Fix plot saving
        plt.savefig('logistic_model_with_invex_train.jpg')
        plt.savefig('logistic_model_with_invex_train.eps')

        plt.figure()
        plt.plot(experiment.epochs_to_plot.to('cpu'), experiment.avg_test_losses_to_plot.to('cpu'))
        plt.xlabel('Epochs')
        plt.ylabel('Avg Test Loss')
        plt.title('Multinomial Logistic Regression (with Invex)')
        plt.savefig('logistic_model_with_invex_test.jpg')
        plt.savefig('logistic_model_with_invex_test.eps')

        plt.show()
