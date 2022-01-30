import numpy as np
import utils
from scipy.optimize import minimize
from scipy.special import logsumexp


class LogisticRegression():
    """ Logistic regression
    """

    def __init__(self, lam):
        """ Construct a least square regression object
        Don't modify this

        Args:
            lam (float):
                regularization constant
                Default = 0 (ordinary logistic regression)
                If lam > 0 (regularized logistic regression)
        """
        self.lam = lam

    def logistic_loss(self, w, X, y):
        """ Compute the logistic loss

        Args:
            w (numpy.array)
                Weight vector

            X (numpy.array)
                Input feature matrix

            y (numpy.array)
                Label vector

        Returns:
            Logistic loss

        """
        # Your solution goes here
        margin = np.empty(len(y))
        in_log = np.empty(len(y))
        regularizer = 0
        for i in range(len(y)):
            f = np.dot(w, X[i])
            margin[i] = np.dot(y[i], f)
            in_log[i] = 1 + np.exp(-margin[i])
        for j in range(len(X[0])):
            regularizer += w[j]**2
        return np.mean(np.log2(in_log)) + self.lam*regularizer

    def fit(self, X, y, w0=None):
        """ Learn the weight vector of the logistic regressor via
        iterative optimization using scipy.optimize.minimize

        Args:
            X (numpy.array)
                Input feature matrix with shape (N,d)z
            y (numpy.array)
                Label vector with shape (N,)
            w0 (numpy.array)
                (Optional) initial estimate of weight vector

        Returns:
            (w, b)
                w (numpy.array):
                    Learned weight vector with shape (d,)
                    where d is the number of features of data X
                b (float)
                    Learned bias term

        """
        (N, d) = X.shape
        assert(y.shape == (N,))

        # You should initialize w0 here
        if not w0:
            w0 = np.zeros(len(X[0])+1)

        res = minimize(self.logistic_loss, w0,
                       args=(utils.augment_bias(X), y))
        w = res.x[:len(X[0])]
        self.weight = w
        b = res.x[len(X[0])]
        self.bias = b
        return (w, b)

    def predict(self, X):
        """ Predict the label {+1, -1} of data points

        Args:
            X (numpy.array)
                Input feature matrix with shape (N, d)


        Returns:
            y (numpy.array)
                Predicted pylabel vector with shape (N,)
                Each entry is in {+1, -1}

        """
        # Your solution goes here
        weight = np.hstack((self.weight, self.bias))
        yhat = np.empty(len(X))
        for i in range(len(yhat)):
            yhat[i] = np.sign(
                np.dot(weight, np.transpose(utils.augment_bias(X)[i])))
            if yhat[i] == 0:
                yhat[i] = 1
        return yhat
