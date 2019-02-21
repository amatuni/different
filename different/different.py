import numpy as np
from scipy.special import gamma, digamma
import numpy.linalg as la
from scipy import stats
import multiprocessing as mp
from multiprocessing.managers import BaseManager, SyncManager
import itertools

import signal


def entropy(X, method='auto', **kwargs):

    if method == 'auto':
        d = X.shape[1]
        if d <= 5:
            return kde_entropy(X, **kwargs)
        else:
            return knn_entropy(X, **kwargs)
    elif method == 'knn':
        return knn_entropy(X, **kwargs)
    elif method == 'gauss_kde':
        return kde_entropy(X, **kwargs)


def knn_entropy(X, **kwargs):
    """
    Nearest neighbor entropy estimator

    :param X: (n_samples, n_dimension) ndarray of samples
    "param subsample: if provided, run estimation on a random N subsample of X
    :return: entropy of X
    """
    if 'subsample' in kwargs and kwargs['subsample'] > X.shape[0]:
        raise Exception("subsample size is larger than number of samples in X")

    X = resample(X, kwargs['subsample']) if 'subsample' in kwargs else X
    nth = kwargs['nth'] if 'nth' in kwargs else 1
    r = kwargs['r'] if 'r' in kwargs else np.inf
    trunc_fract = kwargs['trunc_fract'] if 'trunc_fract' in kwargs else 1.0
    trunc_n = kwargs['trunc_n'] if 'trunc_n' in kwargs else X.shape[0]
    parallel = kwargs['parallel'] if 'parallel' in kwargs else False

    d = X.shape[1]
    X = X[:trunc_n]
    X = X[:int(X.shape[0]*trunc_fract)]
    X = X[0::nth]

    X = bound(X, r)

    neighbors = brute_neighbors(X, parallel=parallel)

    d = X.shape[1]
    vol = np.pi**(d/2)/gamma(d/2 + 1)
    accum = 0
    N = neighbors.shape[0]

    C = np.log((vol * (N-1))/np.exp(digamma(1)))

    return np.sum(d * np.log(neighbors[:, 1]) + C) / N


def kde_entropy(X, **kwargs):
    """
    Entropy estimator using Gaussian kernel density estimation
    and Monte-Carlo integral estimation
    """
    if 'subsample' in kwargs and kwargs['subsample'] > X.shape[0]:
        raise Exception("subsample size is larger than number of samples in X")

    X = resample(X, kwargs['subsample']) if 'subsample' in kwargs else X
    N = kwargs['n'] if 'n' in kwargs else 10000
    r = kwargs['r'] if 'r' in kwargs else 10

    if 'r' in kwargs:
        r = kwargs['r']

    p = stats.gaussian_kde(X.T)

    def entropy_func(x):
        return p(x) * np.log(p(x)+0.000001)

    samples = np.random.uniform(-r, r, size=(N, p.d))
    vals = entropy_func(samples.T)
    H = -((2 * r) ** p.d * np.nanmean(vals))

    return H


def resample(X, n):
    if n > X.shape[0]:
        raise Exception("subsample size is larger than number of samples in X")

    samples = np.random.choice(
        X.shape[0], replace=False, size=n)
    X = X[samples, :]

    return X


def brute_neighbors(X, parallel=True, n=mp.cpu_count()):
    neighbors = np.ndarray(shape=(X.shape[0], 2))
    # neighbors = []

    if parallel:
        neighbors = parallel_neighbors(X, n)
        return neighbors

    # just need to brute force nearest neighbor search
    # because the dimensionality is so high.
    # KDTree is not significantly faster
    for i in range(X.shape[0]):
        nearest = (0, 1000000000)
        for j in range(X.shape[0]):
            if j == i:
                continue
            d = la.norm((X[i] - X[j]))
            if d < nearest[1]:
                if d == 0:
                    continue
                    # print()
                nearest = (j, d)
        # neighbors.append(nearest)
        neighbors[i] = nearest
    return neighbors


def parallel_neighbors(X, n=mp.cpu_count()):
    manager = SyncManager()
    manager.start(mgr_init)
    state = manager.dict()

    Xs = np.array_split(X, n)
    procs = []

    for i, x in enumerate(Xs):
        # print(x.shape)
        p = mp.Process(target=par_neighb, args=(x, X, i, state))
        procs.append(p)
        p.start()

    # print(len(procs))

    for p in procs:
        p.join()

    # print(np.vstack(state.values()).shape)
    # print(state)
    # print([x.shape for x in state.values()])
    return np.vstack(state.values())


def par_neighb(Xs, X, idx, state):
    # neighbors = []
    neighbors = np.ndarray(shape=(Xs.shape[0], 2))

    for i in range(Xs.shape[0]):
        nearest = (0, 1000000000)
        for j in range(X.shape[0]):
            if j == i:
                continue
            d = la.norm((Xs[i] - X[j]))
            if d < nearest[1]:
                if d == 0:
                    continue
                nearest = (j, d)
        # neighbors.append(nearest)
        neighbors[i] = nearest

    # print(neighbors[:4])
    # print()

    state[idx] = neighbors


def mgr_init():
    # signal.signal(signal.SIGINT, mgr_sig_handler)
    # <- OR do this to just ignore the signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def bound(X, r):
    """

    :param X: (n_samples, n_dimensions) array of samples
    :param r: the radius to bound by
    :return: a filtered X containing only those pionts
    within an r*r n-cube about the origin
    """
    d = X.shape[0]
    inbnd = np.all(np.logical_and(-r <= X, X <= r), axis=1)
    return X[inbnd]


def lddp(X, r=5, **kwargs):
    """
    Compute the limiting density of discrete points estimate
    of a distribution of empirical samples. m(x) is taken to be
    the uniform distribution over the support.

    :param X: (n_samples, n_dimensions) array of samples
    :param r: the radius to bound by
    :param kwargs:
    :return: the lddp estimate of the distribution
    """
    d = X.shape[1]
    H = entropy(X, r=r, **kwargs)
    c = np.log((1/(2*r))**d)
    return H - c


def support_bounds(X):
    bounds = [(np.min(x), np.max(x)) for x in X]
    b = np.max(np.abs(bounds))
    return b
