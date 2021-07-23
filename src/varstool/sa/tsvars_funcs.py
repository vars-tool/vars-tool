# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from itertools import combinations

from typing import Tuple


# helper functions
def scale(
    df: pd.DataFrame,
    bounds: pd.DataFrame,
    axis: int=1
) -> pd.DataFrame:
    """Scales the sampled matrix ``df`` to the ``bounds``
    that is a defined via a dictionary with ``ub``, ``lb`` keys;
    the values of the dictionary are lists of the upper and lower
    bounds of the parameters/variables/factors. if (``axis = 1``)
    then each row of `df` is selected, otherwise columns.

    Parameters
    ----------
    df : array_like
        a dataframe of randomly sampled values
    bounds : dict
        a lower and upper bounds to scale the values
    axis : ``0`` for index, ``1`` for columns

    Returns
    -------
    df : array_like
        the returned dataframe scaled using bounds

    """

    # numpy equivalent for math operations
    bounds_np = {key: np.array(value) for key, value in bounds.items()}

    if axis:
        return df * (bounds_np['ub'] - bounds_np['lb']) + bounds_np['lb']
    else:
        return df.T * (bounds_np['ub'] - bounds_np['lb']) + bounds_np['lb']


def pairs_h(iterable) -> pd.DataFrame:
    """Gives the pairs of numbers considering their differences.

    Parameters
    ----------
    iterable : iterable
        an iterable object

    Returns
    -------
    pairs : array_like
        the returned dataframe of paired values

    """

    interval = range(min(iterable), max(iterable) - min(iterable))
    pairs = {key + 1: [j for j in combinations(iterable, 2) if np.abs(
        j[0] - j[1]) == key + 1] for key in interval}

    return pairs


def section_df(
    df: pd.DataFrame,
    delta_h: float
) -> pd.DataFrame:
    """Gets the paired values of each section based on index.

    Parameters
    ----------
    df : array_like
        a dataframe of star points
    delta_h : array_like
        resolution of star samples

    Returns
    -------
    sample : array_like
        the paired values for each section of star points

    """

    pairs = pairs_h(df.index.get_level_values(-1))
    df_values = df.to_numpy()
    sample = pd.concat({h * delta_h:
                        pd.DataFrame.from_dict({str(idx_tup): [df_values[idx_tup[0]].item(),
                                                               df_values[idx_tup[1]].item()]
                                                for idx_tup in idx},
                                               'index')
                        for h, idx in pairs.items()})

    return sample


# TSVARS core functions
def cov_section(pair_cols: pd.DataFrame, mu_star: pd.DataFrame) -> pd.Series:
    """Returns the sectional covariogram of the pairs of function evaluations
    that resulted from each star point. This function is specific for the time-series
    varying/aggregate of the VARS sensitivity analysis.

    Parameters
    ----------
    pair_cols : array_like
        a Pandas Dataframe of paired values function evaluations
    mu_star : array_like
        a Pandas DataFrame of mu star values that are calculated separately

    Returns
    -------
    cov_section_values : array_like
        the sectional covariogram dataframe

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    cov_section_values = (pair_cols.sub(mu_star, axis=0)[0] * pair_cols.sub(mu_star, axis=0)[1]).\
        groupby(level=['ts', 'centre', 'param', 'h']).mean()

    return cov_section_values


def variogram(pair_cols: pd.DataFrame) -> pd.Series:
    """Returns the variogram calculated from the pairs of function evaluations
    that each resulted from each star point. This function is specific for the time-series
    varying/aggregate of the VARS sensitivity analysis.


    Parameters
    ----------
    pair_cols : array_like
        a Pandas Dataframe of paired values function evaluations

    Returns
    -------
    variogram_values : array_like
        the variogram dataframe

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    variogram_values = 0.5 * \
        (pair_cols[0] - pair_cols[1]
         ).pow(2).groupby(level=['ts', 'param', 'h']).mean()

    return variogram_values


def morris_eq(
    pair_cols: pd.DataFrame
) -> Tuple[pd.Series, ...]:
    """Return the Morris Equivalent values derived from the pairs of 
    function evaluations that each resulted from each star point. This function
    is specific for the time-series varying/aggregate of the VARS sensitivity analysis.

    Parameters
    ----------
    pair_cols : array_like
        a Pandas Dataframe of paired values function evaluations

    Returns
    -------
    morris_eq_values : array_like
        the morris dataframe

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    morris_eq_values = ((pair_cols[1] - pair_cols[0]).abs().groupby(level=['ts', 'param', 'h']).mean(),
                        (pair_cols[1] - pair_cols[0]).groupby(level=['ts', 'param', 'h']).mean())

    return morris_eq_values


def covariogram(
    pair_cols: pd.DataFrame,
    mu_overall: pd.Series
) -> pd.Series:
    """Return the covariogram values derived from the pairs of 
    function evaluations that each resulted from each star point. This function
    is specific for the time-series varying/aggregate of the VARS sensitivity analysis.

    Parameters
    ----------
    pair_cols : array_like
        a Pandas Dataframe of paired values function evaluations
    mu_overall : array_like
        a Pandas Dataframe of overall mu calculated on all
        function evaluation values for each time-step

    Returns
    -------
    covariogram_values : array_like
        the covariogram dataframe

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    covariogram_values = (pair_cols[0].sub(mu_overall, level=0) * pair_cols[1].sub(mu_overall, level=0)) \
        .groupby(level=['ts', 'param', 'h']).mean()

    return covariogram_values


def e_covariogram(cov_section_all: pd.DataFrame) -> pd.Series:
    """Returns the Expected value of covariogram values derived from the pairs of 
    function evaluations that each resulted from each star point. This function
    is specific for the time-series varying/aggregate of the VARS sensitivity analysis.

    Parameters
    ----------
    cov_section_all : array_like
        a Pandas Dataframe of sectional covariograms

    Returns
    -------
    e_covariogram_values : array_like
        the covariogram dataframe

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    e_covariogram_values = cov_section_all.groupby(
        level=['ts', 'param', 'h']).mean()

    return e_covariogram_values


def sobol_eq(
    gamma: pd.DataFrame,
    ecov: pd.DataFrame,
    variance: pd.Series,
    delta_h: float
) -> pd.Series:
    """Returns the Sobol Equivalent values derived from the variogram (`gamma`),
    expected values of sectional covariograms (`ecov`), and overall variance (`variance`).
    This function is specific for the time-series varying/aggregate of the VARS sensitivity
    analysis.

    Parameters
    ----------
    gamma : array_like
        a Pandas Dataframe of variogram values for each time-step
    ecov : array_like
        a Pandas DataFrame of expected values of sectional covariograms
        for each time-step
    variance : array_like
        variance of function evaluations over all time-steps
    delta_h : float
        resolution of star samples

    Returns
    -------
    sobol_eq_values : array_like
        the Sobol Equivalent values

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    sobol_eq_values = (gamma + ecov).div(variance,
                                         level='ts').loc[:, :, delta_h]

    return sobol_eq_values


def ivars(
        variogram_array: pd.DataFrame,
        scale: float,
        delta_h: float
) -> pd.DataFrame:
    """Generates Integrated Variogram Across a Range of Scales (IVARS) by approximating 
    area using right trapezoids having width of `delta_h` and hights of variogram values.
    This function is specific for the time-series varying/aggregate of the VARS sensitivity
    analysis.

    Parameters
    ----------
    variogram_array : array_like
        a Pandas Dataframe of variogram values for each time-step
    scale : gloat
        the scale for the IVARS evaluations
    delta_h : float
        the resolution of star point generation

    Returns
    -------
    ivars_values : array_like
        the Sobol Equivalent values

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    num_h = len(variogram_array.index.levels[-1].to_list())
    x_bench = np.arange(start=0, stop=delta_h * (num_h + 1), step=delta_h)
    x_int = np.arange(start=0, stop=(scale * 10 + 1) / 10, step=delta_h)

    # calculate interpolated values for both x (h) and y (variogram)
    if x_int[-1] < scale:
        x_int.append(scale)
    y_bench = [0] + variogram_array.to_list()

    y_int = np.interp(x=x_int, xp=x_bench, fp=y_bench)

    # for loop for each step size to caluclate the area
    ivars_values = 0
    for i in range(len(x_int) - 1):
        ivars_values += 0.5 * \
            (y_int[i + 1] + y_int[i]) * (x_int[i + 1] - x_int[i])

    return ivars_values
