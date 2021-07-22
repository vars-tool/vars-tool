import numpy as np
import pandas as pd

from decimal import Decimal
from typing import Dict


def star(
    star_centres,
    delta_h: float=0.1,
    parameters: list=[],
    rettype: str='dict',
    precision: int=10
) -> np.ndarray:
    ''' STAR sampling algorithm

    This function generates ``star_points`` based on [1] for each
    sample set (i.e., each row consisting of ``star_centres``).
    ``star_centres`` are the points along which in each direction
    the `star_points` are generated. The resolution of sampling is
    :math:`\Delta h` (``delta_h``). This appraoch is a structured
    sampling straregy; read more in [2] and [3].

    Parameters
    ----------
    star_centres : array_like
        the 2d array (n, m) containing sample sets,
        ``n`` is the number of sample sets and 
        ``m`` is the number of parameters/factors/
        variables
    delta_h : float, optional
        sampling resolution, defaults to ``0.1``
    parameters : list
        parameter names
    rettype : str, optional
        ``'dict'`` or ``'dataframe'``, defaults to ``'dict'``
    precision : int, optional
        the number of digits after the precision point, defaults to ``10``

    Returns
    -------
    star_points : array_like
        np.array of star points, each element of this 4d
        array is a 3d np.array with each 2d array containing
        star points along each parameter/factor/variable.

    References
    ----------
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
        dict_of_points = _star_sampler_dict(star_centres.reshape(
            1, star_centres.size), delta_h, parameters)
        if rettype == 'DataFrame':
            return pd.concat({key: pd.concat({k: pd.DataFrame(d) for k, d in value.items()}) for key, value in dict_of_points.items()})
        return dict_of_points

    elif star_centres.ndim == 2:
        # star_points
        dict_of_points = _star_sampler_dict(star_centres, delta_h, parameters)
        if rettype == 'DataFrame':
            from pandas import concat  # import pandas here to avoid overhead if rettype=='dict'
            return pd.concat({key: pd.concat({k: pd.DataFrame(d) for k, d in value.items()}) for key, value in dict_of_points.items()})
        return dict_of_points

    else:
        # cannot operate on more than 2 dimensional arrays at the moment
        raise ValueError(
            'dimension mismatch: "star_centres" must be a 1- or 2-dimensional array')


def _star_sampler_dict(
        centres,
        resolution: float=0.1,
        parameters: list=[],
        precision: int=10
) -> dict:
    '''Returning a dictionary of star samples

    Parameters
    ----------
    centres : array_like
        an array of `star centres`
    resolution : float, optional
        a.k.a ``delta_h``, the resolution of star sampling, defaults to ``0.1``
    parameters : list, optional
        parameter names, defaults to an empty list
    precision : int, optional
        number of digits after the precision point, defaults to ``10``

    Returns
    -------
    dict_of_points : dict
        a dictionary of star samples

    '''

    if not parameters:
        parameters = list(range(centres.shape[1]))

    dict_of_points = {}
    points = {}

    # for each row of star centres
    for i in range(centres.shape[0]):
        row = centres[i, :]
        for k in range(row.size):
            idx_size = len(_range_vector(row[k], step=resolution))
            col_size = row.size

            # a bit high memory usage but nothing else comes to my minds for now (KK)
            temp_view = np.broadcast_to(row, (idx_size, col_size)).reshape(
                idx_size, col_size).copy()
            temp_view[:, k] = _range_vector(
                row[k], start=0, end=1, step=resolution, precision=precision)
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
    """Yields sequence of numbers from start (inclusive) to stop (inclusive)
    by step (increment) with rounding set to n digits.

    Parameters
    ----------
    start : float or int
        start of sequence
    stop : float or int
        end of sequence
    step : float or int
        int or float increment (e.g. 1 or 0.001)
    fround : int
        float rounding, n decimal places, defaults to 5

    Yields
    ------
    int
        yielding the next value of the `range` sequence

    Source
    ------
    The code is taken from the following link, thanks to Goran B.
    https://stackoverflow.com/a/49059292/5188208
    """
    try:
        i = 0
        while stop >= start and step > 0:
            if i == 0:
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


"""
    Contributors:
    -------------
    Razavi, Saman, (2016): algorithm, code in MATLAB (c)
    Gupta, Hoshin, (2016): algorithm, code in MATLAB (c)
    Keshavarz, Kasra, (2021): code in Python 3
"""
