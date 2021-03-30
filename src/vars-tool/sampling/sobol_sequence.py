import numpy as np

import math

from typing import List
from scipy import stats

def sobol_sequence(sp:int, params:int, seed:int=None, scramble:bool=True, skip:int=1000, leap:int=101) -> np.ndarray:
    '''
    Description:
    ------------
    Sobol' sequences are low-discrepancy, quasi-random numbers. The
    code is taken from the `scipy dev-1.7` [1]_.
    
    
    Arguments:
    ----------
    :param sp: the number of sampling points
    :type sp: {int, numpy.int32, numpy.int64}
    :param params: the number of parameters/factors/variables
    :type params: {int, numpy.int32, numpy.int64}
    :param seed: randomization seed number
    :type seed: {int, numpy.int32, numpy.int64}, optional
    :param scramble: scrambling the produced array, defaults to ``True``
    :type scramble: bool, optional
    :param skip: the number of points to skip
    :type skip: {int, numpy.int32, numpy.int64}, optional
    :param leap: the interval of picking values
    :type leap: {int, numpy.int32, numpy.int64}, optional
    
    
    Returns:
    --------
    :return sobol_seq: the sobol sequence
    :rtype sobol_seq: numpy.ndarray
    
    
    lengthotes:
    ------
    There are many versions of Sobol' sequences depending on their
    "direction numbers". This code uses direction numbers from [4]_. Hence,
    the maximum number of dimension is 21201. The direction numbers have been
    precomputed with search criterion 6 and can be retrieved at
    https://web.maths.unsw.edu.au/~fkuo/sobol/
    
    
    References:
    -----------
    .. [1] https://github.com/scipy/scipy/stats/_qmc.py
    .. [2] https://scipy.github.io/devdocs/generated/scipy.stats.qmc.Sobol.html
    .. [3] I. M. Sobol. The distribution of points in a cube 
           and the accurate evaluation of integrals. Zh. Vychisl.
           Mat. i Mat. Phys., 7:784-802, 1967.
    .. [4] S. Joe and F. Y. Kuo. Constructing sobol sequences with better
           two-dimensional projections. SIAM Journal on Scientific Computing,
           30(5):2635-2654, 2008.
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
    dtype_msg = ("dtype of '{}' array must be 'int', 'numpy.int32' or 'numpy.int64'.")
    if type(sp) not in [int, np.int32, np.int64]:
        raise ValueError(dtype_msg.format('sp'))
    if type(params) not in [int, np.int32, np.int64]:
        raise ValueError(dtype_msg.format('params'))
        
    if scramble is True:
        return _scrambled_sobol_generate(sp, params, skip, leap)
    else:
        return _sobol_generate(sp, params, skip, leap)