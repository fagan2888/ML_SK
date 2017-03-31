import numpy as np
import math
from sklearn.metrics.pairwise import check_pairwise_arrays,manhattan_distances
from sklearn.utils import gen_even_slices
from sklearn.externals.joblib import delayed, Parallel


def LaplacianKernel(X, Y=None, gamma=None):
    return GeneralizedNormalKernel(X, Y=Y, gamma=gamma)

def GeneralizedNormalKernel(X, Y=None, gamma = None, beta = 1):
    """Compute the generalized normal kernel between X and Y.
    The generalized normal kernel is defined as::
        K(x, y) = exp(-gamma ||x-y||_1^beta)
    for each pair of rows x in X and y in Y.
    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
    Y : array of shape (n_samples_Y, n_features)
    gamma : float
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """

    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    if beta == 1:
        K = -gamma * manhattan_distances(X, Y)
    else:
        K = -gamma * manhattan_distances(X, Y) ** beta
    np.exp(K, K)    # exponentiate K in-place
    return K

def MaternKernel(X, Y=None, gamma = None, p = 0):
    """Compute the generalized normal kernel between X and Y.
    The generalized normal kernel is defined as::
        K(x, y) = exp(-gamma ||x-y||_1^beta)
    for each pair of rows x in X and y in Y.
    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
    Y : array of shape (n_samples_Y, n_features)
    gamma : float
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    assert(p == int(p))

    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    r = manhattan_distances(X, Y)
    if p == 0:
        K = -gamma * r
        np.exp(K, K)    # exponentiate K in-place
    if p == 1:
        K = -gamma * r * math.sqrt(3)
        np.exp(K, K)    # exponentiate K in-place
        K *= (1+gamma * r * math.sqrt(3))
    if p == 1:
        K = -gamma * r * math.sqrt(5)
        np.exp(K, K)    # exponentiate K in-place
        K *= (1+gamma * r * math.sqrt(5) + 5./3. * (r*gamma)**2)
    return K
