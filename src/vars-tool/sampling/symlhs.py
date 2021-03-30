import numpy as np
from typing import Tuple

def symlhs(sp:int, params:int, seed:int=None, criterion:str='maximin', iterations:int=10) -> np.ndarray:
    '''
    Description:
    ------------
    This function generates symmetrical LHS of `sp`
    datapoints in the `params`-dimensional hypercube
    of [0,1] developed based on [1].
    
    
    Arguments:
    ----------
    :param sp: the number of sampling points
    :type sp: int, np.int32, np.int64
    :param params: the number of parameters/variables/factors
    :type params: int, np.int32, np.int64
    :param seed: the seed number for randomization
    :type seed: int, np.int32, np.int64, defaults to `None`
    :param criterion: method for evaluation of the generated
                      sampled array, options are `maximin` 
                      and `correlation`
    :type criterion: str, defaults to maximin
    :param iterations: number of iterations to get the optimal
                       sampled array
    :type iterations: int, np.int32, np.int64
    
    
    Returns:
    --------
    :return symlhs_sample: the returned symmetrical LHS sampled array
    :rtype symlhs_sample: np.ndarray
    
    
    Contributors:
    -------------
    Sheikholeslami, Razi, (2017): code in MATLAB(c)
    Razavi, Saman, (2017): supervision, code in MATLAB(c)
    Keshavarz, Kasra, (2021): code in Python 3
    Matott, Shawn, (2019): code in C/++
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


def _symlhs_sampled(sp:int, params:int, seed=None) -> np.ndarray:
    '''
    Description:
    ------------
    This function returns a symmetrical LHS sample
    
    
    Arguments:
    ----------
    :param sp: the number of sampling points
    :type sp: int, np.int32, np.int64
    :param params: the number of parameters/variables/factors
    :type params: int, np.int32, np.int64
    
    
    Returns:
    --------
    :return symlhs_sample: the returned sample
    :rtype symlhs_sample: np.ndarray
    
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
        symlhs[0,:] = (sp+1)/2


    for i in np.arange(start, sp, 2):
        symlhs[i,:] = _perm_intv(1, sp, params-1)
        for c in range(symlhs.shape[1]):
            while np.unique(symlhs[0:i+1,c]).size < (i+1):
                symlhs[i,c] = _perm_intv(1, sp, 0)
        symlhs[i+1,:] = sp+1-symlhs[i,:]
    
    symlhs_sample = np.random.uniform(low=symlhs-1, high=symlhs)/sp
    
#     DEBUG
#     print(symlhs)
#     print('-----')
#     print(symlhs_sample)
#     END DEBUG
    
    return symlhs_sample


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