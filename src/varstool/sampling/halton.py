'''
The code for this sampling method is inspired from several sources,
listed as follows:
* https://scikit-optimize.github.io/stable/_modules/skopt/sampler/halton.html
* https://github.com/scipy/scipy/blob/e625869f112e797d32107a658cdd9519221e9fea/scipy/stats/_sobol.pyx
* https://github.com/scipy/scipy/blob/e625869f112e797d32107a658cdd9519221e9fea/scipy/stats/_qmc.py
A formal list of references is printed under the docstrings of each
corresponding algorithm.
'''

import numpy as np


def halton(sp: int, params: int, seed: int=None, scramble: bool=True, skip: int=1000, leap: int=101) -> np.ndarray:
    '''
    Description:
    ------------
    This function generates (scrambled) quasi-random halton sequence.
    In brief, it generalizes the Van der Corput's sequence for multiple
    dimensions. The Halton sequence uses the base-two Van der Corput
    sequence for the first dimension, base-three for its second and
    base-:math:`n` for its n-dimension.


    Arguments:
    ----------
    :param sp: the number of sampling points
    :type sp: {int, np.int32, or np.int64}
    :param params: the number of parameters
    :type params: {int, np.int32, np.int64}
    :param seed: seed number for randomization, defaults to ``None``
    :type seed: {None, int, np.int32, np.int64, `numpy.random.Generator`}, optional
    :param scramble: scrambling flag, defaults to ``False``
    :type scramble: bool, optional
    :param skip: the number of points to skip
    :type skip: {int, numpy.int32, numpy.int64}, optional
    :param leap: the interval of picking values
    :type leap: {int, numpy.int32, numpy.int64}, optional


    Returns:
    --------
    :return halton_seq: the halton sequence array
    :rtype halton_seq: numpy.ndarray


    References:
    -----------
    .. [1] https://github.com/scipy/scipy/scipy/stats/_qmc.py
    .. [2] https://github.com/scipy/scipy/scipy/stats/_sobol.pyx
    .. [3] Halton, J.H. On the efficiency of certain quasi-random sequences of 
           points in evaluating multi-dimensional integrals. Numer. Math. 2, 
           84–90 (1960). https://doi.org/10.1007/BF01386213
    .. [4] Owen, A.B. A randomized Halton algorithm in R (2017). arXiv:1706.02808v2


    Contributors:
    -------------
    Razavi, Saman, (2018): supervision, function call in MATLAB (c)
    Keshavarz, Kasra, (2021): code in Python 3
    '''

    # check the seed number
    if seed:
        np.random.seed(int(seed))

    # check int signs
    sign_msg = ("the sign of '{}' must be positive (>0).")
    if sp < 0:
        raise ValueError(sign_msg.format('sp'))
    if params < 0:
        raise ValueError(sign_msg.format('params'))

    # check dtypes
    dtype_msg = (
        "dtype of '{}' array must be 'int', 'numpy.int32' or 'numpy.int64'.")
    if type(sp) not in [int, np.int32, np.int64]:
        raise ValueError(dtype_msg.format('sp'))
    if type(params) not in [int, np.int32, np.int64]:
        raise ValueError(dtype_msg.format('params'))

    # Generate a sample using a Van der Corput sequence per dimension.
    # important to have ``type(bdim) == int`` for performance reason
    sample = [_van_der_corput(sp=sp * (leap + 1) + skip, base=int(bdim), start_index=0,
                              scramble=scramble,
                              seed=seed)
              for bdim in _n_primes(params)]

    halton_sample = np.array(sample).T.reshape(sp * (leap + 1) + skip, params)

    halton_sample = halton_sample[0 +
                                  skip:halton_sample.shape[0]:(leap + 1), :]

    return halton_sample


def _van_der_corput(sp: int, base: int=2, start_index: int=0, scramble: bool=True, seed: int=None) -> np.ndarray:
    '''
    Description:
    ------------
    Van der Corput sequence.
    Pseudo-random number generator based on a b-adic expansion.
    Scrambling uses permutations of the remainders (see [1]_ and [2]_).
    Multiple permutations are applied to construct a point. The sequence
    of permutations has to be the same for all points of the sequence.


    Arguments:
    ----------
    :param sp: number of elements in the sequence
    :type sp: {int, numpy.int32, numpy.int64}
    :param base: base of the sequence, defaults to 2.
    :type base: {int, numpy.int32, numpy.int64}, optional
    :param start_index: index to start the sequence from, defaults to 0.
    :type start_index: {int, numpy.int32, numpy.int64}, optional
    :param scramble: if ``True``, use Owen scrambling, defaults to ``True``.
    :type scarmble: bool, optional
    :param seed: seed number for randomization
    :type seed: {None, int, numpy.int32, numpy.int64}, optional


    Returns:
    --------
    :return sequence: Sequence of van der Corput
    :rtype sequence: list


    References:
    -----------
    .. [1] A. B. Owen. "A randomized Halton algorithm in R",
       arXiv:1706.02808, 2017.
    .. [2] scipy.stats.qmc, version: dev-1.7
    '''

    # check seed number
    if seed:
        np.random.seed(seed)

    sequence = np.zeros(sp)

    quotient = np.arange(start_index, start_index + sp)
    b2r = 1 / base

    while (1 - b2r) < 1:
        remainder = quotient % base

        if scramble:
            # permutation must be the same for all points of the sequence
            perm = np.random.permutation(base)
            remainder = perm[np.array(remainder).astype(int)]

        sequence += remainder * b2r
        b2r /= base
        quotient = (quotient - remainder) / base

    return sequence


def _n_primes(n: int) -> list:
    '''
    Description:
    ------------
    Generate ``n`` number of prime numbers, taken from [1]_

    Arguments:
    ----------
    :param n: the number of primes to be produced
    :type n: {int, numpy.int32, numpy.int64}


    Returns:
    --------
    :return primes: a list of primes with while ``len(primes)`` is `n`.
    :rtype primes: list


    Source:
    -------
    .. [1] `Scipy <https://github.com/scipy/scipy/stats/_qmc.py>`_.
    '''

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
              131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
              197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
              271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
              353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
              433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
              509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
              601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673,
              677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761,
              769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857,
              859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947,
              953, 967, 971, 977, 983, 991, 997][:n]

    if len(primes) < n:
        big_number = 2000
        while 'not enough primes':
            primes = _gen_primes(big_number)[:n]
            if len(primes) == n:
                break
            big_number += 1000

    return primes


def _gen_primes(threshold: int) -> list:
    """
    Description:
    ------------
    Generate prime values using sieve of Eratosthenes method
    between 2 and the ``threshold``.


    Arguments:
    ----------
    :param threshold: the upper bound for the size of the prime values.
                     the ``threshold`` is included in the sequence.
    :type threshold: {int, numpy.int32, numpy.int64}


    Returns:
    --------
    :return primes: all primes from 2 and up to ``threshold``.
    :rtype primes: list
    """

    if threshold == 2:
        return [2]
    elif threshold < 2:
        return []

    numbers = list(range(3, threshold + 1, 2))
    root_of_threshold = threshold ** 0.5
    half = int((threshold + 1) / 2 - 1)
    idx = 0
    counter = 3

    while counter <= root_of_threshold:
        if numbers[idx]:
            idy = int((counter * counter - 3) / 2)
            numbers[idy] = 0
            while idy < half:
                numbers[idy] = 0
                idy += counter
        idx += 1
        counter = 2 * idx + 3

    primes = np.array([2] + [number for number in numbers if number])

    return primes
