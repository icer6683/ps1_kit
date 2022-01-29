import utils
import matplotlib.pyplot as plt
import numpy as np
from logistic_regression import LogisticRegression

# HINT: this is really all the imports you need

X_train, y_train, X_test, y_test = utils.load_all_train_test_data(
    "P4")

subsets_X, subsets_y = utils.load_learning_curve_data("P4/Train-subsets")
for i, X in enumerate(subsets_X):
    y = subsets_y[i]
    # Train on X and y
    ten = LogisticRegression()
    ten.logistic_loss
