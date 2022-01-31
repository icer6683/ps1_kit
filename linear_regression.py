import numpy as np
import utils


class LeastSquareRegression():
    """ Least square regression
    """

    def __init__(self, lam=0):
        """ Construct a least square regression object
        Don't modify this

        Args: 
            lam (float):
                regularization constant
                Default = 0 (ordinary least square regression)
                If lam > 0 (regularized least square regression)
        """
        self.lam = lam

    def fit(self, X, y):
        """ Learn the weights of the linear regression model
        by solving for w in closed form

        Args:
            X (numpy.array)
                Input feature data with shape (N, d)

            y (numpy.array)
                Label vector with shape (N,)

        Returns:
            (w, b)
                w (numpy.array):
                    Learned weight vector with shape (d,)
                    where d is the number of features of data X
                b (float)
                    Learned bias term

        """
        N, d = X.shape
        assert(y.shape == (N,))

        # Your solution goes here
        X = utils.augment_bias(X)
        in_inv = np.dot(np.transpose(X), X)+self.lam * \
            len(y)*np.identity(len(X[0]))
        wb = np.dot(np.dot(np.linalg.inv(in_inv), np.transpose(X)), y)
        w = wb[:(len(X[0])-1)]
        self.weight = w
        b = wb[len(X[0])-1]
        self.bias = b
        return (w, b)

    def predict(self, X):
        """ Do prediction on given input feature matrix

        Args:
            X (numpy.array)
                Input feature matrix with shape (N, d)
        Returns:
            Predicted output vector with shape (N,)

        """

        # Your solution goes here
        weight = np.hstack((self.weight, self.bias))
        return np.dot(weight, np.transpose(utils.augment_bias(X)))
