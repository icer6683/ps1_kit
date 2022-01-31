import utils
import matplotlib.pyplot as plt
import numpy as np
from logistic_regression import LogisticRegression

# HINT: this is really all the imports you need
# Part A
X_train, y_train, X_test, y_test = utils.load_all_train_test_data(
    "P4")
subsets_X, subsets_y = utils.load_learning_curve_data("P4/Train-subsets")
train_errors = np.empty(len(subsets_X))
test_errors = np.empty(len(subsets_X))

for i, X in enumerate(subsets_X):
    y = subsets_y[i]
    # Train on X and y
    train = LogisticRegression(0)
    train.fit(X, y)
    ytrain = train.predict(X)
    train_errors[i] = utils.classification_error(ytrain, y)

    ytest = train.predict(X_test)
    test_errors[i] = utils.classification_error(ytest, y_test)
    print(i)

xaxis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.plot(xaxis, test_errors, label="Testing")
plt.plot(xaxis, train_errors, label="Training")
plt.xlabel("Percent Training Data")
plt.ylabel("Classification Error")
plt.title("Unregularized Learning Curve")
plt.legend()
plt.savefig("learning_curve4.png")
plt.show()

print("Training error: ", train_errors[9])
print("Testing error: ", test_errors[9])

# Part B
regularizer_values = np.array([10**-7, 10**-6, 10**-
                               5, 10**-4, 10**-3, 10**-2, 10**-1, 1])

train_errors_reg = np.empty(len(regularizer_values))
test_errors_reg = np.empty(len(regularizer_values))
fold = utils.load_all_cross_validation_data("P4/Cross-validation")

cross_error = np.empty((len(regularizer_values), 5))

for j in range(len(regularizer_values)):
    train_reg = LogisticRegression(regularizer_values[j])
    train_reg.fit(X_train, y_train)
    ytrain = train_reg.predict(X_train)
    train_errors_reg[j] = utils.classification_error(ytrain, y_train)

    ytest = train_reg.predict(X_test)
    test_errors_reg[j] = utils.classification_error(ytest, y_test)

    for k in range(5):
        test_cross, train_cross = utils.partition_cross_validation_fold(
            fold, k)
        train_reg_cross = LogisticRegression(regularizer_values[j])
        train_reg_cross.fit(train_cross[0], train_cross[1])
        ytest_cross = train_reg_cross.predict(test_cross[0])
        cross_error[j][k] = utils.classification_error(
            ytest_cross, test_cross[1])
        print(k)
    print(j)

xaxis2 = [-7, -6, -5, -4, -3, -2, -1, 1]
plt.plot(xaxis2, test_errors_reg, label="Testing")
plt.plot(xaxis2, train_errors_reg, label="Training")
plt.xlabel("Log of Lambda")
plt.ylabel("Classification Error")
plt.title("L2-Regularized Learning Curve")
plt.legend()
plt.savefig("learning_curve_reg4.png")
plt.show()

avg_reg_error = np.empty(len(regularizer_values))
for i in range(len(avg_reg_error)):
    avg_reg_error = np.mean(cross_error, axis=1)
cross_index = np.argmin(avg_reg_error)
best_lambda = regularizer_values[cross_index]

print("Set of all cross-validation errors are as follows: ", avg_reg_error)
print("The optimal lambda is: "+str(best_lambda))
print("The corresponding training error is: ", train_errors_reg[cross_index])
print("The corresponding testing error is: ", test_errors_reg[cross_index])
