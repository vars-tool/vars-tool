import numpy as np
import numpy.typing as npt
import pandas as pd

from decimal import Decimal
from typing import Dict


def star(star_centres:npt.ArrayLike, delta_h:float=0.1, parameters=[], rettype:str='dict', precision=10) -> np.ndarray:
    '''
    Description:
    ------------
    This function generates ``star_points`` based on [1]_ for each
    sample set (i.e., each row consisting of ``star_centres``).
    ``star_centres`` are the points along which in each direction
    the `star_points` are generated. The resolution of sampling is
    :math:`\Delta h` (``delta_h``). This appraoch is a structured
    sampling straregy; read more in [2]_ and [3]_.


    Arguments:
    ----------
    :param star_centres: the 2d array (n, m) containing sample sets
                    ``n`` is the number of sample sets and
                    ``m`` is the number of parameters/factors/
                    variables
    :type star_centres: np.typing.ArrayLike
    :param delta_h: sampling resolution, defaults to 0.1
    :type delta_h: float
    :param parameters: parameter names
    :type parameters: list
    :param rettype: the type of returned value
    :type rettype: str
    :param precision: the number of digits after the precision point, defaults to 0.1
    :type precision: int, optional


    Returns:
    --------
    :return star_points: np.array of star points, each element of this 4d
                         array is a 3d np.array with each 2d array containing
                         star points along each parameter/factor/variable.
    :rtype star_points: np.ndarray


    References:
    -----------
    .. [1] Razavi, S., Sheikholeslami, R., Gupta, H. V., &
           Haghnegahdar, A. (2019). VARS-TOOL: A toolbox for
           comprehensive, efficient, and robust sensitivity
           and uncertainty analysis. Environmental modelling
           & software, 112, 95-107.
           doi: 10.1016/j.envsoft.2018.10.005
    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework
           for comprehensive, robust, and efficient global sensitivity
           analysis: 1. Theory. Water Resources Research, 52(1), 423-439.
           doi: 10.1002/2015WR017558
    .. [3] Razavi, S., & Gupta, H. V. (2016). A new framework
           for comprehensive, robust, and efficient global sensitivity
           analysis: 2. Application. Water Resources Research, 52(1),
           423-439. doi: 10.1002/2015WR017559


    Contributors:
    -------------
    Razavi, Saman, (2016): algorithm, code in MATLAB (c)
    Gupta, Hoshin, (2016): algorithm, code in MATLAB (c)
    Keshavarz, Kasra, (2021): code in Python 3
    '''

    # check the type of star_cenres
    if not isinstance(star_centres, np.ndarray):
        raise TypeError("'star_centres' must be of type numpy.ndarray.")

    if (delta_h >= 1) ^ (delta_h <= 0):
        raise ValueError("'delta_h' must be a float between 0 and 1.")

    if rettype not in ['dict', 'DataFrame']:
        raise ValueError("'rettype' can be either 'dict' or 'DataFrame'.")

    # check `rettype` in inner if clauses
    if star_centres.ndim == 1:
        # star_points
        dict_of_points = _star_sampler_dict(star_centres.reshape(1, star_centres.size), delta_h, parameters)
        if rettype=='DataFrame':
            return pd.concat({key:pd.concat({k:pd.DataFrame(d) for k, d in value.items()}) for key,value in dict_of_points.items()})
        return dict_of_points

    elif star_centres.ndim == 2:
        # star_points
        dict_of_points = _star_sampler_dict(star_centres, delta_h, parameters)
        if rettype=='DataFrame':
            from pandas import concat # import pandas here to avoid overhead if rettype=='dict'
            return pd.concat({key:pd.concat({k:pd.DataFrame(d) for k, d in value.items()}) for key,value in dict_of_points.items()})
        return dict_of_points

    else:
        # cannot operate on more than 2 dimensional arrays at the moment
        raise ValueError('dimension mismatch: "star_centres" must be a 1- or 2-dimensional array')


def _star_sampler_dict(centres, resolution=0.1, parameters=[], precision=10):
    '''
    Description:
    ------------
    This function returns a dictionary of star samples


    Arguments:
    ----------
    :param centres: an array of `star centres`
    :type centres: np.ndarray
    :param resolution: also `delta_h`, the resolution of star sampling
    :type resolution: float, defaults to 0.1
    :param parameters: parameter names, optional
    :type parameters: list


    Returns:
    --------
    :return dict_of_points: a dictionary of star samples
    :rtype dict_of_points: dict
    '''
    if not parameters:
        parameters = list(range(centres.shape[1]))

    dict_of_points = {}
    points = {}

    # for each row of star centres
    for i in range(centres.shape[0]):
        row = centres[i,:]
        for k in range(row.size):
            idx_size = len(_range_vector(row[k], step=resolution))
            col_size = row.size

            # a bit high memory usage but nothing else comes to my minds for now (KK)
            temp_view = np.broadcast_to(row, (idx_size, col_size)).reshape(idx_size, col_size).copy()
            temp_view[:,k] = _range_vector(row[k], start=0, end=1, step=resolution, precision=precision)
            points[parameters[k]] = temp_view

        dict_of_points[i] = points.copy()

    return dict_of_points


def _range_vector(num, start=0, end=1, step=0.1, precision=10):
    '''
    Produces the ranges between 0 and 1 with
    incremental steps while including 0=<``num``=<1.
    '''

    first_part = [-i for i in rangef(-num, start, step, precision)]
    second_part = [i for i in rangef(num, end, step, precision)]

    return np.unique(np.array(first_part + second_part))


def rangef(start, stop, step, fround=10):
    """
    Description:
    ------------
    Yields sequence of numbers from start (inclusive) to stop (inclusive)
    by step (increment) with rounding set to n digits.


    Arguments:
    ----------
    :param start: start of sequence
    :type start: float or int
    :param stop: end of sequence
    :type stop: float or int
    :param step: int or float increment (e.g. 1 or 0.001)
    :type step: float or int
    :param fround: float rounding, n decimal places
    :type fround: int, defaults to 5


    Returns:
    --------
    :return: yielding the next value of the sequence
    :rtype: float or int


    Credit:
    -------
    The code is taken from the following link, thanks to Goran B.
    https://stackoverflow.com/a/49059292/5188208
    """
    try:
        i = 0
        while stop >= start and step > 0:
            if i==0:
                yield start
            elif start >= stop:
                yield stop
            elif start < stop:
                if start == 0:
                    yield 0
                if start != 0:
                    yield start
            i += 1
            start += step
            start = round(start, fround)
        else:
            pass
    except TypeError as e:
        yield "type-error({})".format(e)
    else:
        pass


def _cli_save(dict_of_points):
    # temp

    return pd.concat({key:pd.concat({k:pd.DataFrame(d) for k, d in value.items()}) for key,value in dict_of_points.items()})
