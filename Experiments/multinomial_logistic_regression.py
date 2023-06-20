from workflow import *
from networks import *

if __name__ == '__main__':
    fashion_train_dataloader, fashion_test_dataloader = experiment_setup(fashion_training_data, fashion_test_data,
                                                                         num_epochs=30)
    logistic_model = MultinomialLogisticRegression(input_dim=fashion_img_length, num_classes=num_fashion_classes)
    logistic_model_name = f"{logistic_model=}".split('=')[0]  # Gives name of model variable!
    print(logistic_model)

    logistic_model = ModuleWrapper(logistic_model, lamda=INVEX_PARAM)
    logistic_model.init_ps(train_dataloader=fashion_train_dataloader)
    logistic_model = logistic_model.to(device)

    cross_entropy = nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(logistic_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_PARAM)

    print("\nUsing", device, "\n")

    for epoch in range(NUM_EPOCHS):
        converged = train(fashion_train_dataloader, logistic_model, cross_entropy, sgd, epoch)
        test(fashion_test_dataloader, logistic_model, cross_entropy, epoch)
        if converged:
            break
    save(logistic_model, logistic_model_name)
