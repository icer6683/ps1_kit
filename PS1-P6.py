import utils
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import LeastSquareRegression

# Hint: this is all the imports you need
X_train, y_train, X_test, y_test = utils.load_all_train_test_data(
    "P6/Data-set-1")
subsets_X, subsets_y = utils.load_learning_curve_data(
    "P6/Data-set-1/Train-subsets")
train_errors = np.empty(len(subsets_X))
test_errors = np.empty(len(subsets_X))

for i, X in enumerate(subsets_X):
    y = subsets_y[i]
    # Train on X and y
    train = LeastSquareRegression(0)
    train.fit(X, y)
    ytrain = train.predict(X)
    train_errors[i] = utils.mean_squared_error(ytrain, y)

    ytest = train.predict(X_test)
    test_errors[i] = utils.mean_squared_error(ytest, y_test)
    print(i)

xaxis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.plot(xaxis, test_errors, label="Testing")
plt.plot(xaxis, train_errors, label="Training")
plt.xlabel("Percent Training Data")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.savefig("learning_curve6.png")
plt.show()

print("Weight vector: ", train.weight)
print("Bias: ", train.bias)
print("Training error: ", train_errors[9])
print("Testing error: ", test_errors[9])

plt.plot(X_test, ytest)
plt.scatter(X_test, y_test)
plt.xlabel("Test Instance")
plt.ylabel("Test Labels")
plt.savefig("learned_linear_function.png")
plt.show()
