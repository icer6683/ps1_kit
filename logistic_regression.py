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
        f = np.dot(w.T, X)
        margin = np.empty(len(y))
        in_log = np.empty(len(y))
        for i in len(y):
            margin[i] = np.dot(y[i], f)
            in_log[i] = 1 + np.exp(-margin[i])
        return np.mean(np.log(in_log))

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
            w0 = np.zero(len(X[0])+1)

        ones = np.ones((len(X), 1))
        Xnew = np.hstack(X, ones)
        # Hint:
        res = minimize(self.logistic_loss, w0, args=(Xnew, y))
        w = res.x[:len(X[0])+1]
        self.weight = w
        b = w[len(X[0])+1]
        self.bias = b
        # The rest of your solution goes here
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
        weight = np.hstack(self.weight, self.bias)
        return np.sign(np.dot(np.transpose(weight), utils.augment_bias(X)) > 0)
