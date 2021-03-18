import numpy as np

from . import slhs


def plhs(sp:int, params:int, slices:int, seed=None, _iter=10, criterion='maximin') -> Tuple[np.ndarray, np.ndarray]:
    '''
    Description:
    ------------
    This function created SLHS samples, based on [1] and [2]
    
    
    Arguments:
    ----------
    :param lb: lower bound of the sequence
    :type lb: one of int, np.int32, np.int64
    :param ub: upper bound of the sequence
    :type ub: one of int, np.int32, np.int64
    :param slices: the number of slices
    :type slices: one of int, np.int32, np.int64
    :param seed: seed number for randomization
    :type seed: int, np.int32, np.int64
    :param iter: maximum iteration number 
    :type iter: int, np.int32, np.int64, optional
    :param criterion: the criterion for assessing the quality of sample points
                      the available options are: 'maximin' and 'correlation',
                      defaults to 'maximin'
    :type criterion: str, optional
    
    
    Returns:
    --------
    :return slhs_sample_x: the final slhs sample array based on 'x' criterion
    :rtype slhs_sample_x: np.array
    
    
    References:
    -----------
    .. [1] Ba, S., Myers, W.R., Brenneman, W.A., 2015. Optimal sliced Latin
           hypercube designs. Technometrics 57 (4), 479e487.
           http://dx.doi.org/10.1080/00401706.2014.957867
    .. [2] Sheikholeslami, R., & Razavi, S. (2017). Progressive Latin Hypercube 
           Sampling: An efficient approach for robust sampling-based analysis of 
           environmental models. Environmental modelling & software, 93, 109-126
    
    
    Contributors:
    -------------
    Sheikholeslami, Razi, (2017): algorithm, code in MATLAB (c)
    Razavi, Saman, (2017): supervision
    Keshavarz, Kasra, (2021): code in Python 3
    Matott, Shawn, (2019): code in C/++
    '''
    slice_sp = sp // slices
    
    for it in range(_iter):
    
    # This does not make sense to me. User can decide what to do
    # regarding the number of iterations and choosing the best
    # sample!

    return plhs_sample


def _greedy_plhs(sp:int, slices:int, sample:np.array) -> Tuple[np.array, np.array, float, float]:
    '''
    Description:
    ------------
    Generates a Progressive Latin Hypercube Sampling (PLHS) from
    an optimal Sliced Lating Hypercube Sampling design (SLHS) 
    using a greedy algorithm; based on [1] and [2]


    Arguments:
    ----------
    :param sp: number of sample points
    :type sp: int, np.int32, or np.int64
    :param slices: number of slices
    :type slices: int, np.int32, or np.int64
    :param sample: the sampled matrix\array
    :type sample: np.array


    Returns:
    --------
    :return plhs: plhs sample array
    :rtype plhs: np.array
    :return plhs_slices: plhs sample slices (sub-samples)
    :rtype plhs_slices: np.array
    :return f_priori: objective function value before optimization
    :rtype f_priori: float
    :return f_posteriori: objective function value after optimization
    :rtype f_posteriori: float


    References:
    -----------
    .. [1] Ba, S., Myers, W.R., Brenneman, W.A., 2015. Optimal sliced Latin
           hypercube designs. Technometrics 57 (4), 479e487.
           http://dx.doi.org/10.1080/00401706.2014.957867
    .. [2] Sheikholeslami, R., & Razavi, S. (2017). Progressive Latin Hypercube 
           Sampling: An efficient approach for robust sampling-based analysis of 
           environmental models. Environmental modelling & software, 93, 109-126


    Contributors:
    -------------
    Sheikholeslami, Razi, (2017): algorithm, code in MATLAB (c)
    Razavi, Saman, (2017): algorithm, code in MATLAB (c), supervision
    Keshavarz, Kasra, (2021): code in Python 3
    Matott, Shawn, (2019): code in C/++

    '''
    # define the randomization seed number
    if seed:
        np.random.seed(seed)

    # check the dtype of input arguments
    msg = ("dtype of '{}' array must be 'int', 'numpy.int32' or 'numpy.int64'.")
    if type(sp) not in [int, np.int32, np.int64]:
        raise ValueError(msg.format('sp'))
    if type(slices) not in [int, np.int32, np.int64]:
        raise ValueError(msg.format('slices'))
    if type(sample) not in [numpy.ndarray]:
        raise ValueError(msg.format('sample'))

    # check the number of slices and sample points
    if (sp % slices) != 0:
        raise ValueError("sample points must be a multiplier of slices.")

    # check the sign of the input arguments
    sign_msg = ("the sign of '{}' must be positive (>0).")
    if sp < 0:
        raise ValueError(sign_msg.format('sp'))
    if slices < 0:
        raise ValueError(sign_msg.format('slices'))
        
    
    slice_sp = sp // slices
    # row-wise slicing - PLHS standard
    sub_sample = np.array(np.split(sample, slices, axis=0))
    # priori cost function value
    f_priori = np.mean([_lhd_cost(sl_agg) for sl_agg in
              [np.concatenate(sub_sample[0:t+1,...]) for t in range(slices)]])
    
    # let's find out the first two slices that results in the lowest
    # cost function and make the original code more efficient...
    # pay attention to axis=0, PLHS standard is row-wise...
    indices = list(range(sub_sample.shape[0]))
    least_cost = lambda idx: _lhd_cost(np.concatenate(np.take(sub_sample, idx, axis=0)))
    greedy_indices = list(min(combinations(indices,2), key=least_cost)) # 2: pair
    
    # find the next slices in a loop and add its indice to the 
    # `greedy_indices` list
    indices = list(set(indices) - set(greedy_indices))
    for _ in range(len(indices)):
        greedy_indices = list(min([greedy_indices+[idx] for idx in indices], key=least_cost))
        indices = list(set(indices) - set(greedy_indices)) # same as above...
    
    # check the `posteriori` cost function value
    # pay attention to axis=0, PLHS standard is row-wise...
    plhs_slices = np.take(sub_sample, greedy_indices, axis=0)
    plhs = np.concatenate(plhs_slices)
    f_posteriori = np.mean([_lhd_cost(sl_agg) for sl_agg in \
                            [np.concatenate(plhs_slices[0:t+1,...]) for t in range(slices)]])
    
    return (plhs, plhs_slices, f_priori, f_posteriori)
    

def _lhd_cost(arr:np.ndarray, axis:int=1) -> float:
    '''
    Description:
    ------------
    This is a simple cost function used in PLHS Greedy algorithm
    
    
    Arguments:
    ----------
    :param arr: the input array (nxd dimensions)
    :type arr: np.array
    :param axis: the axis along which the cost is calculated
    :type axis: int, defaults to 1 in PLHS
    
    
    Returns:
    --------
    :return f: the cost function
    :rtype f: int, np.int32, np.int64
    
    '''
    
    # get the bins equal to the number of rows
    # in PLHS, each row is a sample series, and each column
    # corresponds to a parameter/factor/variable
    edges = np.linspace(start=0, stop=1, num=arr.shape[0]+1)
    f = -np.sum(_bin_count(np.digitize(arr, edges), axis=axis))
    
    return f


def _bin_count(arr:np.ndarray, axis:int=0) -> np.ndarray:
    '''
    Description:
    ------------
    Calculates the number of unique values along the `axis` of the given
    `arr`. This function is used in PLHS algorithm to check LHS-conformity 
    of the generated random samples.
    
    
    Arguments:
    ----------
    :param arr: the input array of interest
    :type arr: np.array
    :param axis: the axis along which the unique values are counted
    :type arr: int, `0` for `rows` and `1` for `columns`, defaults to 0
    
    
    Returns:
    --------
    :return unique_count: the number of unique values along each axis
    :rtype unique_count: np.array
    
    
    Source/Credit:
    --------------
    .. [1] https://stackoverflow.com/questions/
           48473056/number-of-unique-elements-per-row-in-a-numpy-array
           (the most efficient method)
    '''
    if axis: # the method does operations row-wise...
        arr = arr.T
        
    n = arr.max()+1
    a_off = arr+(np.arange(arr.shape[0])[:,None])*n
    M = arr.shape[0]*n
    unique_count = (np.bincount(a_off.ravel(), minlength=M).reshape(-1,n)!=0).sum(1)
    return unique_count