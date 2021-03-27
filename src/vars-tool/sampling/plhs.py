import numpy as np
from typing import Tuple
from itertools import combinations

def plhs(sp:int, params:int, slices:int, seed:int=None, iterations:int=10, criterion:str='maximin') -> Tuple[np.ndarray, np.ndarray]:
    '''
    Description:
    ------------
    This function optimizes SLHS samples based on [1] and [2]
    
    
    Arguments:
    ----------
    :param sp: number of sampling points
    :type sp: int, np.int32, or np.int64
    :param params: number of parameters/factors/variables
    :type params: int, np.int32, or np.int64
    :param slices: the number of slices
    :type slices: int, np.int32, or np.int64
    :param seed: seed number for randomization
    :type seed: int, np.int32, np.int64
    :param iterations: number of iterations
    :type iterations: int, np.int32, or np.int64, optional
    :param criterion: the criterion for assessing the quality of sample points
                      the available options are: 'maximin' and 'correlation',
                      defaults to 'maximin'
    :type criterion: str, optional
    
    
    Returns:
    --------
    :return plhs_sample_x: the final slhs sample array based on 'x' criterion
    :rtype plhs_sample_x: np.array
    
    
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
    Razavi, Saman, (2017): algorithm code in MATLAB (c), supervision
    Keshavarz, Kasra, (2021): code in Python 3
    Matott, Shawn, (2019): code in C/++
    '''
    
    slice_sp = sp // slices
    
    # iterate given the number of iterations and choose the best sample
    slhs_list = []
    plhs_list = []
    
    for _iter in range(iterations):
        if seed:
            seed += seed
        slhs_list.append(slhs(sp, params, slices, seed, iterations, criterion)[0])
        plhs_list.append(_greedy_plhs(sp, slices, slhs_list[_iter]))
    
    cost_f_posteriori = [trial[-1] for trial in plhs_list]
    min_cost_idx = cost_f_posteriori.index(min(cost_f_posteriori))
    plhs_sample = plhs_list[min_cost_idx][0] # first returned item
    plhs_sample_slices = plhs_sample.reshape((slices, slice_sp, params))
    
    # This does not make sense to me. User can decide what to do
    # regarding the number of iterations and choosing the best
    # sample!

    return plhs_sample, plhs_sample_slices


def slhs(sp, params, slices, seed=None, iterations=20, criterion='maximin') -> Tuple[np.ndarray, np.ndarray]:
    '''
    Description:
    ------------
    This function created SLHS samples, based on [1] and [2]. In
    order to find optimal ordering of slices the KNN method is 
    utilized.
    
    
    Arguments:
    ----------
    :param sp: number of sample points
    :type sp: one of int, np.int32, np.int64
    :param params: number of parameters/variables/factors
    :type params: one of int, np.int32, np.int64
    :param slices: number of slices
    :type slices: one of int, np.int32, np.int64
    :param seed: seed number for randomization
    :type seed: int, np.int32, np.int64, optional
    :param _iter: maximum iteration number 
    :type _iter: int, np.int32, np.int64, optional
    :param criterion: the criterion for assessing the quality of sample points;
                      the available options are: 'maximin' and 'correlation',
                      defaults to 'maximin'
    :type criterion: str
    
    
    Returns:
    --------
    :return slhs_sample_x: the final slhs sample array based on 'x' criterion
    :rtype slhs_sample_x: np.array
    :return slhs_sample_x_slice: the final slhs sample array slices based on
                                 'x' criterion
    :rtype slhs_sample_x_slice: np.array
    
    
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
    Sheikholeslami, Razi, (2017): algorithm, code in MATLAB (c) vars-tool
    Razavi, Saman, (2017): supervision, vars-tool
    Keshavarz, Kasra, (2021): code in Python 3
    Matott, Shawn, (2019): code in C/++
    
    '''

    # define the seed number
    if seed:
        np.random.seed(seed)


    # Check the inputs and raise appropriate exceptions
    msg_crt = ("'{}' is not defined; available options: 'maximin', 'correlation'")
    if type(criterion) is not str:
        raise TypeError(msg_crt.format(str(criterion)))
    if criterion not in ['maximin', 'correlation']:
        raise ValueError(msg_crt.format(criterion))
    
    # calculate the number of slices
    slice_sp = sp // slices # to get int
    
    # Check the criterion
    if criterion == 'maximin':
        best_sample = _sampler(sp, params, slices)
        best_sub_sample = best_sample.reshape((slices, slice_sp, params))
        best_sample_cost = _get_min_distance(best_sample, k=3)
        best_sub_sample_cost = _get_min_distance_sub(best_sub_sample)
        cost_func = np.mean([best_sample_cost, best_sub_sample_cost])
        
        for it in range(iterations):
            new_sample = _sampler(sp, params, slices)
            new_sub_sample = new_sample.reshape((slices, slice_sp, params))
            new_sample_cost = _get_min_distance(new_sample)
            new_sub_sample_cost = _get_min_distance_sub(new_sub_sample)
            new_cost_func = np.mean([new_sample_cost, new_sub_sample_cost])
            
            # check the cost function value
            if new_cost_func > cost_func:
                best_sample = new_sample
                cost_func = new_cost_func
        
        slhs_sample_maximin = best_sample
        slhs_sample_maximin_slice = slhs_sample_maximin.reshape((slices, slice_sp, params))

        return slhs_sample_maximin, slhs_sample_maximin_slice

    elif criterion == 'correlation':
        best_sample = _sampler(sp, params, slices)
        best_sub_sample = best_sample.reshape((slices, slice_sp, params))
        best_sample_cost = _get_corr(best_sample)
        best_sub_sample_cost = _get_corr_sub(best_sub_sample)
        cost_func = np.mean([best_sample_cost, best_sub_sample_cost])
        
        for it in range(iterations):
            new_sample = _sampler(sp, params, slices)
            new_sub_sample = new_sample.reshape((slices, slice_sp, params))
            new_sample_cost = _get_corr(new_sample)
            new_sub_sample_cost = _get_corr_sub(new_sub_sample)
            new_cost_func = np.mean([new_sample_cost, new_sub_sample_cost])
            
            # check the cost function value
            if new_cost_func < cost_func:
                best_sample = new_sample
                cost_func = new_cost_func
        
        slhs_sample_correl = best_sample
        slhs_sample_correl_slice = slhs_sample_correl.reshape((slices, slice_sp, params))

        return slhs_sample_correl, slhs_sample_correl_slice    


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

    # check the dtype of input arguments
    msg = ("dtype of '{}' array must be 'int', 'numpy.int32' or 'numpy.int64'.")
    if type(sp) not in [int, np.int32, np.int64]:
        raise ValueError(msg.format('sp'))
    if type(slices) not in [int, np.int32, np.int64]:
        raise ValueError(msg.format('slices'))

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


def _sampler(sp:int, params:int, slices:int, seed:int=None) -> np.ndarray:
    '''
    Description:
    ------------
    A simple sampling algorithm to create lhs slices.


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


    Returns:
    --------
    :return sample_array: the final sample array
    :rtype sample_array: np.array


    References:
    -----------
    .. [1] Ba, S., Myers, W.R., Brenneman, W.A., 2015. Optimal sliced Latin
           hypercube designs. Technometrics 57 (4), 479e487.
           http://dx.doi.org/10.1080/00401706.2014.957867.
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
    
    # define the randomization seed number
    if seed:
        np.random.seed(seed)


    # check the dtype of input arguments
    msg = ("dtype of '{}' array must be 'int', 'numpy.int32' or 'numpy.int64'.")
    if type(sp) not in [int, np.int32, np.int64]:
        raise ValueError(msg.format('sp'))
    if type(params) not in [int, np.int32, np.int64]:
        raise ValueError(msg.format('params'))
    if type(slices) not in [int, np.int32, np.int64]:
        raise ValueError(msg.format('slices'))

    # check the number of slices and sample points
    if (sp % slices) != 0:
        raise ValueError("sample points must be a multiplier of slices.")

    # check the sign of the input arguments
    sign_msg = ("the sign of '{}' must be positive (>0).")
    if sp < 0:
        raise ValueError(sign_msg.format('sp'))
    if params < 0:
        raise ValueError(sign_msg.format('params'))
    if slices < 0:
        raise ValueError(sign_msg.format('slices'))


    # calculate the number of slices
    slice_sp = sp // slices # to get int

    # generate slices using sampling (int) without permutation
    rand_perm = lambda slice_sp, slices: np.concatenate([np.random.permutation(slice_sp)+1 for _j in range(slices)])
    sample_array = np.stack([rand_perm(slice_sp, slices) for _i in range(params)])
    
    # DEBUG
    # print('sample_array:')
    # print(sample_array)
    # END DEBUG

    # positional function definition
    slice_spec = lambda row, slice_sp: np.stack([(row==_j+1) for _j in range(slice_sp)])

    # row-wise assessment
    for _row in range(0, sample_array.shape[0]):
        position_array = slice_spec(sample_array[_row, :], slice_sp)
        for kk in range(0, slice_sp):
            lb = (kk*slices)+1
            ub = (kk+1)*slices
            perm = _perm_intv(lb, ub, slices, seed)
            try:
                sample_array[_row, position_array[kk, :]] = perm
            except: # sometimes a number might be missing due to randomness...
                raise RuntimeError("error! change the seed number and try again.")
    sample_array = np.random.uniform(sample_array-1, sample_array)
    sample_array /= sp

    return sample_array.T


def _perm_intv(lb:int, ub:int, slices:int, seed:int=None) -> np.ndarray:
    '''
    Description:
    ------------
    A simple random sampling given the lower and upper bounds,
    without permutation, and amongst the integers in the interval


    Arguments:
    ----------
    :param lb: lower bound of the sequence
    :type lb: one of int, np.int32, np.int64
    :param ub: upper bound of the sequence
    :type ub: one of int, np.int32, np.int64
    :param slices: the number of slices
    :type slices: one of int, np.int32, np.int64


    Returns:
    --------
    :return perm: the sampled np.array
    :type perm: np.array


    Contributors:
    -------------
    Sheikholeslami, Razi, (2017): algorithm, code in MATLAB (c)
    Razavi, Saman, (2017): supervision
    Keshavarz, Kasra, (2021): code in Python 3
    Matott, Shawn, (2019): code in C/++
    '''

    # define the randomization seed number
    if seed:
        np.random.seed(seed)

    # a simple sampling without permutation algorithm
    length = np.abs(ub-lb)+1
    perm   = np.arange(start=lb, stop=ub+1, step=1)
    for k in range(2, length+1):
        index1 = np.int(np.ceil(np.random.rand() * k))
        index2 = perm[k-1]
        perm[k-1] = perm[index1-1]
        perm[index1-1] = index2
    perm = perm[0:slices+1]
    
    # DEBUG
    # print('perm is:')
    # print(perm)
    # END DEBUG

    return perm


def _knn(arr1:np.ndarray, arr2:np.ndarray, k:int) -> Tuple[np.ndarray, np.ndarray]:
    
    '''
    Description:
    ------------
    A simple KNN ML algorithm to find the minimum Euclidean distance
    
    
    Arguments:
    ----------
    :param arr1: the first array of data
    :type arr1: np.array, `n` rows and `d` columns
    :param arr2: the second array of data
    :type arr2: np.array, `m` rows and `d` columns
    :param k: the number of neighbors
    :type k: int, np.int32, np.int64
    
    
    Returns:
    --------
    :return distances: Euclidean distances between `arr1` and `arr2` points
    :rtype distances: np.array
    :return indices: the indices of the distances between `arr1` and `arr2` 
                     points
    :rtype indices: np.array
    '''
    
    # calculating the distance between points
    distances = -2 * arr1@arr2.T + np.sum(arr2**2,axis=1) + \
                     np.sum(arr1**2,axis=1)[:, np.newaxis]
    
    # taking into account the floating point discrepancies 
    distances[distances < 0] = 0
    distances = distances**.5
    indices = np.argsort(distances, 0)
    distances = np.sort(distances,0)
    
    # reshaping the arrays
    indices = indices[0:k, : ].T
    
#     DEBUG
#     print(distances)
#     print(distances.shape)
#     END DEBUG
    
    distances = distances[0:k, : ].T.flatten().reshape(arr1.shape[0], k)
    
    return indices, distances


def _get_min_distance(arr:np.ndarray, k:int=3) -> float:
    '''
    Description:
    ------------
    Calculates the minimum Euclidean distance between sample points as a measure
    of sparsity of the sampling space
    
    
    Arguments:
    ----------
    :param arr: the input array of any size
    :type arr: np.array
    
    
    Returns:
    --------
    :return min_distance: the minimum distance calculated
    :rtype min_distance: np.float
    '''
    
    idx, distance = _knn(arr, arr, k) # idx index start from 0
    min_distance = np.min(distance[:, 1])
    
    return min_distance


def _get_corr(arr:np.ndarray) -> float:
    '''
    Description:
    ------------
    Calculates the correlation between the sample columns and
    reports the sum of squared correlation values.
    
    
    Arguments:
    ----------
    :param arr: the input array of any size
    :type arr: np.array
    
    
    Returns:
    --------
    :return sq_corr: sum of the squared correlation values
    :rtype sq_corr: np.float
    '''

    return sum(sum(np.triu(np.corrcoef(arr, rowvar=False)**2, k=1)))


def _get_corr_sub(arr:np.ndarray) -> float:
    '''
    Description:
    ------------
    Calculates the correlation between the sample columns and
    reports the sum of squared correlation values.
    
    
    Arguments:
    ----------
    :param arr: the input array of any size
    :type arr: np.array
    
    
    Returns:
    --------
    :return sq_corr: sum of the squared correlation values
    :rtype sq_corr: np.float
    '''

    return np.mean(np.array([_get_corr(x) for x in arr]))


def _get_min_distance_sub(arr:np.ndarray, k:int=3) -> float:
    '''
    Description:
    ------------
    Calculates the minimum Euclidean distance between sample points as a measure
    of sparsity of the sampling space in each slice. The returned value is aver-
    aged amongst the minimum value of the slices.
    
    
    Arguments:
    ----------
    :param arr: the input array of any size
    :type arr: np.array, n x m dimension
    :param k: the number of neightbors
    :type k: int, np.int32, np.int64
    
    
    Returns:
    --------
    :return min_distance: the minimum distance calculated
    :rtype min_distance: np.float
    '''
    
    return np.mean(np.array([_get_min_distance(x, k) for x in arr]))