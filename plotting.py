"""
    Plotting/analysis code with **trained models**

    Author: Adrian Rahul Kamal Rajkamal
"""
import matplotlib.pyplot as plt
from train import *


def load_model(filename, model_type='logistic_reg', input_dim=28, num_classes=10, invex_lambda=0.0):
    model = MultinomialLogisticRegression(input_dim=img_length, num_classes=num_classes)
    model = ModuleWrapper(model, lamda=invex_lambda)
    model.init_ps(train_dataloader=train_dataloader)
    model = model.to(device)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


logistic_model_unregularised = load_model(filename='logistic_model_unregularised.pth')
logistic_model_with_invex = load_model(filename='logistic_model_with_invex.pth', invex_lambda=INVEX_LAMBDA)
logistic_model_with_l2 = load_model(filename='logistic_model_with_l2.pth')
logistic_model_with_invex_l2 = load_model(filename='logistic_model_with_invex_l2.pth', invex_lambda=INVEX_LAMBDA)
x, y = test_data[0][0], test_data[0][1]

with torch.no_grad():
    x = x.to(device)

    pred = logistic_model_unregularised(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

    pred = logistic_model_with_invex(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

    pred = logistic_model_with_l2(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

    pred = logistic_model_with_invex_l2(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
