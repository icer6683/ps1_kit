import utils
import matplotlib.pyplot as plt
import numpy as np
from logistic_regression import LogisticRegression

# HINT: this is really all the imports you need

X_train, y_train, X_test, y_test = utils.load_all_train_test_data(
    "P4")
"""
subsets_X, subsets_y = utils.load_learning_curve_data("P4/Train-subsets")
weight = None
train_errors = np.empty(len(subsets_X))
test_errors = np.empty(len(subsets_X))

for i, X in enumerate(subsets_X):
    y = subsets_y[i]
    # Train on X and y
    train = LogisticRegression()
    train.fit(X, y)
    ytrain = train.predict(X)
    train_errors[i] = utils.classification_error(ytrain, y)

    ytest = train.predict(X_test)
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
"""

regularizer_values = [10**-7, 10**-6, 10**-
                      5, 10**-4, 10**-3, 10**-2, 10**-1, 1]
train_errors_reg = np.empty(len(regularizer_values))
test_errors_reg = np.empty(len(regularizer_values))
for j in range(len(regularizer_values)):
    train_reg = LogisticRegression(regularizer_values[j])
    train_reg.fit(X_train, y_train)
    ytrain = train_reg.predict(X_train)
    train_errors_reg[j] = utils.classification_error(ytrain, y_train)

    ytest = train_reg.predict(X_test)
    test_errors_reg[j] = utils.classification_error(ytest, y_test)
    print(j)

xaxis = [-7, -6, -5, -4, -3, -2, -1, 1]
plt.plot(xaxis, test_errors_reg)
plt.xlabel("Percent Training Data")
plt.ylabel("Testing Data Classification Errors")
plt.savefig("testing_learning_curve_reg.png")
plt.show()

plt.plot(xaxis, train_errors_reg)
plt.xlabel("Percent Training Data")
plt.ylabel("Training Data Classification Errors")
plt.savefig("training_learning_curve_reg.png")
plt.show()
