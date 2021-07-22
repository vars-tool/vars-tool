import numpy as np

from typing import Tuple


def symlhs(
    sp: int,
    params: int,
    seed: int=None,
    criterion: str='maximin',
    iterations: int=10
) -> np.ndarray:
    '''Generate symmetrical LHS of ``sp``
    datapoints in the ``params``-dimensional hypercube
    of [0,1]; developed based on [1].

    Parameters
    ----------
    sp : int
        the number of sampling points
    params : int
        the number of parameters/variables/factors
    seed : int or None
        the seed number for randomization, defaults to ``None``
    criterion : str, optional
        method for evaluation of the generated
        sampled array, options are ``'maximin'`` 
        and ``'correlation'``, defaults to ``'maximin'``
    iterations : int, optional
        number of iterations to get the optimal
        sampled array, defaults to ``10``

    Returns
    -------
    symlhs_sample : array_like
        the returned symmetrical LHS sampled array

    References
    ----------
    .. [1] K.Q. Ye, W. Li, A. Sudjianto
           Algorithmic construction of optimal symmetric Latin hypercube designs
           J. Stat. Plan. Infer., 90 (1) (2000), pp. 145-159, 
           doi: 10.1016/S0378-3758(00)00105-1

    '''

    # set the seed number
    if seed:
        np.random.seed(seed)

    # Check the inputs and raise appropriate exceptions
    msg_crt = ("'{}' is not defined; available options: 'maximin', 'correlation'")
    if criterion not in ['maximin', 'correlation']:
        raise ValueError(msg_crt.format(criterion))

    # Check the criterion
    if criterion == 'maximin':
        best_sample = _symlhs_sampled(sp, params)
        best_sample_cost = _get_min_distance(best_sample, k=3)

        for it in range(iterations):
            new_sample = _symlhs_sampled(sp, params)
            new_sample_cost = _get_min_distance(new_sample)

            # check the cost function value
            if new_sample_cost > best_sample_cost:
                best_sample = new_sample
                best_sample_cost = new_sample_cost

        symlhs_sample_maximin = best_sample

        return symlhs_sample_maximin

    elif criterion == 'correlation':
        best_sample = _symlhs_sampled(sp, params)
        best_sample_cost = _get_corr(best_sample)

        for it in range(iterations):
            new_sample = _symlhs_sampled(sp, params)
            new_sample_cost = _get_corr(new_sample)

            # check the cost function value
            if new_sample_cost < best_sample_cost:
                best_sample = new_sample
                best_sample_cost = new_sample_cost

        symlhs_sample_correl = best_sample

        return symlhs_sample_correl


def _symlhs_sampled(sp: int, params: int, seed=None) -> np.ndarray:
    '''returning a symmetrical LHS sample


    Parameters
    ----------
    sp : int
        the number of sampling points
    params : int
        the number of parameters/variables/factors

    Returns
    -------
    symlhs_sample : array_like
        the returned sample

    '''

    if seed:
        np.random.seed(seed)

    # preparing the array - probably python list is more efficient
    # while being propagated in each loop or within a comprehension
    # list

    symlhs = np.ones((sp, params))

    if sp % params == 0:
        start = 0
    else:
        start = 1
        symlhs[0, :] = (sp + 1) / 2

    for i in np.arange(start, sp, 2):
        symlhs[i, :] = _perm_intv(1, sp, params - 1)
        for c in range(symlhs.shape[1]):
            while np.unique(symlhs[0:i + 1, c]).size < (i + 1):
                symlhs[i, c] = _perm_intv(1, sp, 0)
        symlhs[i + 1, :] = sp + 1 - symlhs[i, :]

    symlhs_sample = np.random.uniform(low=symlhs - 1, high=symlhs) / sp

    return symlhs_sample


def _perm_intv(
    lb: int,
    ub: int,
    slices: int,
    seed: int=None
) -> np.ndarray:
    '''A simple random sampling given the lower and upper bounds,
    without permutation, and amongst the integers in the interval


    Parameters
    ----------
    lb : int
        lower bound of the sequence
    ub : int
        upper bound of the sequence
    slices : int
        the number of slices
    seed : int or None, optional
        the seed number of randomization, defaults to ``None``

    Returns
    -------
    perm : array_like
        the sampled np.array

    '''

    # define the randomization seed number
    if seed:
        np.random.seed(seed)

    # a simple sampling without permutation algorithm
    length = np.abs(ub - lb) + 1
    perm = np.arange(start=lb, stop=ub + 1, step=1)
    for k in range(2, length + 1):
        index1 = np.int(np.ceil(np.random.rand() * k))
        index2 = perm[k - 1]
        perm[k - 1] = perm[index1 - 1]
        perm[index1 - 1] = index2
    perm = perm[0:slices + 1]

    # DEBUG
    # print('perm is:')
    # print(perm)
    # END DEBUG

    return perm


def _knn(
    arr1: np.ndarray,
    arr2: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    '''A simple k-NN algorithm to find the minimum Euclidean distance


    Parameters
    ----------
    arr1 : array_like
        the first array of data, consisting of ``n`` rows 
        and ``d`` columns
    arr2 : array_like
        the second array of data, consisting of ``m`` rows
        and `d` columns
    k : int
        the number of neighbors

    Returns
    -------
    distances : array_like
        Euclidean distances between `arr1` and `arr2` points
    indices : array_like
        the indices of the distances between `arr1` and `arr2` 
        points

    '''

    # calculating the distance between points
    distances = -2 * arr1 @ arr2.T + np.sum(arr2**2, axis=1) + \
        np.sum(arr1**2, axis=1)[:, np.newaxis]

    # taking into account the floating point discrepancies
    distances[distances < 0] = 0
    distances = distances**.5
    indices = np.argsort(distances, 0)
    distances = np.sort(distances, 0)

    # reshaping the arrays
    indices = indices[0:k, :].T

    distances = distances[0:k, :].T.flatten().reshape(arr1.shape[0], k)

    return indices, distances


def _get_min_distance(arr: np.ndarray, k: int=3) -> float:
    '''calculateing the minimum Euclidean distance between sample points as a measure
    of sparsity of the sampling space

    Parameters
    ----------
    arr : array_like
        the input array of any size

    Returns
    -------
    min_distance : float
        the minimum distance calculated

    '''

    idx, distance = _knn(arr, arr, k)  # idx index start from 0
    min_distance = np.min(distance[:, 1])

    return min_distance


def _get_corr(arr: np.ndarray) -> float:
    '''calculateing the correlation between the sample columns and
    reports the sum of squared correlation values.

    Parameters
    ----------
    arr : array_like
        the input array of any size

    Returns
    -------
    sq_corr : float
        sum of the squared correlation values

    '''

    return sum(sum(np.triu(np.corrcoef(arr, rowvar=False)**2, k=1)))
