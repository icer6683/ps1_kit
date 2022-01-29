import utils
import matplotlib.pyplot as plt
import numpy as np
from logistic_regression import LogisticRegression

# HINT: this is really all the imports you need

X_train, y_train, X_test, y_test = utils.load_all_train_test_data(
    "P4")

subsets_X, subsets_y = utils.load_learning_curve_data("P4/Train-subsets")
weight = None
train_errors = np.empty(len(subsets_X))
test_errors = np.empty(len(subsets_X))

for i, X in enumerate(subsets_X):
    y = subsets_y[i]
    # Train on X and y
    train = LogisticRegression(0)
    train.fit(X, y)
    weight = train.weight
    bias = train.bias
    ytrain = train.predict(X)
    train_errors[i] = utils.classification_error(ytrain, y)

    test = LogisticRegression(0)
    test.weight = weight
    test.bias = bias
    ytest = test.predict(X_test)
    test_errors[i] = utils.classification_error(ytest, y_test)
    print(i)

xaxis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.plot(xaxis, test_errors)
plt.xlabel("Percent Training Data")
plt.ylabel("Testing Data Classification Errors")
plt.show()
plt.savefig("testing_learning_curve.png")

plt.plot(xaxis, train_errors)
plt.xlabel("Percent Training Data")
plt.ylabel("Training Data Classification Errors")
plt.show()
plt.savefig("training_learning_curve.png")
