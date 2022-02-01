import utils
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import LeastSquareRegression

# Hint: this is all the imports you need
# Part A.i
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
plt.title("Unregularized Learning Curve")
plt.legend()
plt.savefig("learning_curve6.png")
plt.show()

# Part A.ii
print("Weight vector: ", train.weight)
print("Bias: ", train.bias)
print("Training error: ", train_errors[9])
print("Testing error: ", test_errors[9])

plt.plot(X_test, ytest, color="orange", label="Predicted")
plt.scatter(X_test, y_test, label="Actual")
plt.xlabel("Test Instances")
plt.ylabel("Test Labels")
plt.title("Learned Linear Function")
plt.legend()
plt.savefig("learned_linear_function.png")
plt.show()

# Part B.i
subset_X10 = utils.load_data_from_txt_file(
    "P6/Data-set-2/Train-subsets/X_train_10%.txt")
subset_y10 = utils.load_data_from_txt_file(
    "P6/Data-set-2/Train-subsets/y_train_10%.txt", True)
subset_X100 = utils.load_data_from_txt_file(
    "P6/Data-set-2/Train-subsets/X_train_100%.txt")
subset_y100 = utils.load_data_from_txt_file(
    "P6/Data-set-2/Train-subsets/y_train_100%.txt", True)

X_train2, y_train2, X_test2, y_test2 = utils.load_all_train_test_data(
    "P6/Data-set-2")

train_errors2 = np.empty(2)
test_errors2 = np.empty(2)

train2 = LeastSquareRegression(0)
train2.fit(subset_X10, subset_y10)
ytrain10 = train2.predict(subset_X10)
train_errors2[0] = utils.mean_squared_error(ytrain10, subset_y10)
ytest10 = train2.predict(X_test2)
test_errors2[0] = utils.mean_squared_error(ytest10, y_test2)
print("10% Weight vector: ", train2.weight)
print("10% Bias: ", train2.bias)
print("10% Training error: ", train_errors2[0])
print("10% Testing error: ", test_errors2[0])

train2.fit(subset_X100, subset_y100)
ytrain100 = train2.predict(subset_X100)
train_errors2[1] = utils.mean_squared_error(ytrain100, subset_y100)
ytest100 = train2.predict(X_test2)
test_errors2[1] = utils.mean_squared_error(ytest100, y_test2)
print("100% Weight vector: ", train2.weight)
print("100% Bias: ", train2.bias)
print("100% Training error: ", train_errors2[1])
print("100% Testing error: ", test_errors2[1])

# Part B.ii
regularizer_values = [0.1, 1, 10, 100, 500, 1000]
train_errors2_reg10 = np.empty(6)
test_errors2_reg10 = np.empty(6)
train_errors2_reg100 = np.empty(6)
test_errors2_reg100 = np.empty(6)
fold10 = utils.load_all_cross_validation_data10(
    "P6/Data-set-2/Cross-validation")
fold100 = utils.load_all_cross_validation_data100(
    "P6/Data-set-2/Cross-validation")
cross_error10 = np.empty((6, 5))
cross_error100 = np.empty((6, 5))
for j in range(6):
    train2_reg = LeastSquareRegression(regularizer_values[j])
    train2_reg.fit(subset_X10, subset_y10)
    ytrain10_reg = train2_reg.predict(subset_X10)
    train_errors2_reg10[j] = utils.mean_squared_error(
        ytrain10_reg, subset_y10)

    ytest10_reg = train2_reg.predict(X_test2)
    test_errors2_reg10[j] = utils.mean_squared_error(ytest10_reg, y_test2)

    train2_reg.fit(subset_X100, subset_y100)
    ytrain100_reg = train2_reg.predict(subset_X100)
    train_errors2_reg100[j] = utils.mean_squared_error(
        ytrain100_reg, subset_y100)

    ytest100_reg = train2_reg.predict(X_test2)
    test_errors2_reg100[j] = utils.mean_squared_error(ytest100_reg, y_test2)

    for k in range(5):
        test_cross10, train_cross10 = utils.partition_cross_validation_fold(
            fold10, k)
        test_cross100, train_cross100 = utils.partition_cross_validation_fold(
            fold100, k)

        train2_reg.fit(train_cross10[0], train_cross10[1])
        ytrain10_cross = train2_reg.predict(test_cross10[0])
        cross_error10[j][k] = utils.mean_squared_error(
            ytrain10_cross, test_cross10[1])

        train2_reg.fit(train_cross100[0], train_cross100[1])
        ytrain100_cross = train2_reg.predict(test_cross100[0])
        cross_error100[j][k] = utils.mean_squared_error(
            ytrain100_cross, test_cross100[1])

avg_reg_error10 = np.empty(len(regularizer_values))
avg_reg_error100 = np.empty(len(regularizer_values))
for i in range(len(avg_reg_error10)):
    avg_reg_error10 = np.mean(cross_error10, axis=1)
    avg_reg_error100 = np.mean(cross_error100, axis=1)
cross_index10 = np.argmin(avg_reg_error10)
best_lambda10 = regularizer_values[cross_index10]
cross_index100 = np.argmin(avg_reg_error100)
best_lambda100 = regularizer_values[cross_index100]

train_final = LeastSquareRegression(regularizer_values[cross_index10])
train_final.fit(train_cross10[0], train_cross10[1])
print("10%: Set of all cross-validation errors are as follows: ", avg_reg_error10)
print("10%: The optimal lambda is: "+str(best_lambda10))
print("10%: The corresponding training error is: ",
      train_errors2_reg10[cross_index10])
print("10%: The corresponding testing error is: ",
      test_errors2_reg10[cross_index10])
print("10%: Weight vector: ", train_final.weight)
print("10%: Bias term: ", train_final.bias)

train_final = LeastSquareRegression(regularizer_values[cross_index100])
train_final.fit(train_cross100[0], train_cross100[1])
print("100%: Set of all cross-validation errors are as follows: ", avg_reg_error100)
print("100%: The optimal lambda is: "+str(best_lambda100))
print("100%: The corresponding training error is: ",
      train_errors2_reg100[cross_index100])
print("100%: The corresponding testing error is: ",
      test_errors2_reg100[cross_index100])
print("100%: Weight vector: ", train_final.weight)
print("100%: Bias term: ", train_final.bias)

plt.plot(regularizer_values, avg_reg_error10, label="Cross")
plt.plot(regularizer_values, test_errors2_reg10, label="Testing")
plt.plot(regularizer_values, train_errors2_reg10, label="Training")
plt.xscale("log")
plt.xlabel("Log of Lambda")
plt.ylabel("Mean Squared Error")
plt.title("Optimal Lambda at 10% Data")
plt.legend()
plt.savefig("learning_curve_reg6_10.png")
plt.show()

plt.plot(regularizer_values, avg_reg_error100, label="Cross")
plt.plot(regularizer_values, test_errors2_reg100, label="Testing")
plt.plot(regularizer_values, train_errors2_reg100, label="Training")
plt.xscale("log")
plt.xlabel("Log of Lambda")
plt.ylabel("Mean Squared Error")
plt.title("Optimal Lambda at 100% Data")
plt.legend()
plt.savefig("learning_curve_reg6_100.png")
plt.show()
