"""
    Plotting/analysis code with **trained models**

    Author: Adrian Rahul Kamal Rajkamal
"""
import matplotlib.pyplot as plt
from train import *

logistic_model = MultinomialLogisticRegression(input_dim=img_length, num_classes=num_classes)
logistic_model = ModuleWrapper(logistic_model, lamda=INVEX_LAMBDA)
logistic_model.init_ps(train_dataloader=train_dataloader)
logistic_model = logistic_model.to(device)
logistic_model.load_state_dict(torch.load('logistic_model_unregularised.pth'))

logistic_model.eval()
x, y = test_data[0][0], test_data[0][1]

with torch.no_grad():
    x = x.to(device)
    pred = logistic_model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
