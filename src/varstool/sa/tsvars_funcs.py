# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from itertools import combinations


# helper functions
def scale(
        df: pd.DataFrame,
        bounds: pd.DataFrame,
        axis: int=1
) -> pd.DataFrame:
    """
    Description:
    ------------
    This function scales the sampled matrix `df` to the `bounds`
    that is a defined via a dictionary with `ub`, `lb` keys;
    the values of the dictionary are lists of the upper and lower
    bounds of the parameters/variables/factors. if (``axis = 1``)
    then each row of `df` is selected, otherwise columns.


    Parameters:
    -----------
    :param df: a dataframe of randomly sampled values
    :type df: pd.DataFrame
    :param bounds: a lower and upper bounds to scale the values
    :type bounds: dict
    :param axis: 0 for index, 1 for columns
    :type axis: int, optional


    Returns:
    --------
    :return df: the returned dataframe scaled using bounds
    :rtype df: pd.DataFrame


    Contributors:
    -------------
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
    """

    # numpy equivalent for math operations
    bounds_np = {key: np.array(value) for key, value in bounds.items()}

    if axis:
        return df * (bounds_np['ub'] - bounds_np['lb']) + bounds_np['lb']
    else:
        return df.T * (bounds_np['ub'] - bounds_np['lb']) + bounds_np['lb']


def pairs_h(iterable) -> pd.DataFrame:
    """
    Description:
    ------------
    This function gives the pairs of numbers considering their differences.


    Parameters:
    -----------
    :param iterable: an iterable object
    :type iterable: iterable 


    Returns:
    --------
    :return pairs: the returned dataframe of paired values
    :rtype pairs: pd.DataFrame


    Contributors:
    -------------
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
    """

    interval = range(min(iterable), max(iterable) - min(iterable))
    pairs = {key + 1: [j for j in combinations(iterable, 2) if np.abs(
        j[0] - j[1]) == key + 1] for key in interval}

    return pairs


def section_df(df: pd.DataFrame, delta_h: float) -> pd.DataFrame:
    """
    Description:
    ------------
    This function gets the paired values of each section based on index.


    Parameters:
    -----------
    :param df: a dataframe of star points
    :type df: pd.DataFrame
    :param delta_h: resolution of star samples
    :type delta_h: float


    Returns:
    --------
    :return sample: the paired values for each section of star points
    :rtype sample: pd.DataFrame


    Contributors:
    -------------
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
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
def cov_section(pair_cols: pd.DataFrame, mu_star: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
    ------------
    This function return the sectional covariogram of the pairs of function evaluations
    that resulted from each star point. This function is specific for the time-series
    varying/aggregate of the VARS sensitivity analysis.


    Parameters:
    -----------
    :param pair_cols: a Pandas Dataframe of paired values function evaluations
    :type pair_cols: pd.DataFrame
    :param mu_star: a Pandas DataFrame of mu star values that are calculated separately
    :type mu_star: pd.DataFrame


    Returns:
    --------
    :return cov_section_values: the sectional covariogram dataframe
    :rtype cov_section_values: pd.DataFrame


    References:
    -----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559


    Contributors:
    -------------
    Razavi, Saman, (2017): algorithm, code in MATLAB (c)
    Matott, Shawn, (2019): code in C/++
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
    """

    cov_section_values = (pair_cols.sub(mu_star, axis=0)[0] * pair_cols.sub(mu_star, axis=0)[1]).\
        groupby(level=['ts', 'centre', 'param', 'h']).mean()

    return cov_section_values


def variogram(pair_cols: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
    ------------
    This function return the variogram calculated from the pairs of function evaluations
    that each resulted from each star point. This function is specific for the time-series
    varying/aggregate of the VARS sensitivity analysis.


    Parameters:
    -----------
    :param pair_cols: a Pandas Dataframe of paired values function evaluations
    :type pair_cols: pd.DataFrame


    Returns:
    --------
    :return variogram_values: the variogram dataframe
    :rtype variogram_values: pd.DataFrame


    References:
    -----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559


    Contributors:
    -------------
    Razavi, Saman, (2017): algorithm, code in MATLAB (c)
    Matott, Shawn, (2019): code in C/++
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
    """

    variogram_values = 0.5 * \
        (pair_cols[0] - pair_cols[1]
         ).pow(2).groupby(level=['ts', 'param', 'h']).mean()

    return variogram_values


def morris_eq(pair_cols: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
    ------------
    This function return the Morris Equivalent values derived from the pairs of 
    function evaluations that each resulted from each star point. This function
    is specific for the time-series varying/aggregate of the VARS sensitivity analysis.


    Parameters:
    -----------
    :param pair_cols: a Pandas Dataframe of paired values function evaluations
    :type pair_cols: pd.DataFrame


    Returns:
    --------
    :return morris_eq_values: the morris dataframe
    :rtype morris_eq_values: pd.DataFrame


    References:
    -----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559


    Contributors:
    -------------
    Razavi, Saman, (2017): algorithm, code in MATLAB (c)
    Matott, Shawn, (2019): code in C/++
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
    """
    morris_eq_values = ((pair_cols[1] - pair_cols[0]).abs().groupby(level=['ts', 'param', 'h']).mean(),
                        (pair_cols[1] - pair_cols[0]).groupby(level=['ts', 'param', 'h']).mean())

    return morris_eq_values


def covariogram(pair_cols: pd.DataFrame, mu_overall: pd.Series) -> pd.DataFrame:
    """
    Description:
    ------------
    This function return the covariogram values derived from the pairs of 
    function evaluations that each resulted from each star point. This function
    is specific for the time-series varying/aggregate of the VARS sensitivity analysis.


    Parameters:
    -----------
    :param pair_cols: a Pandas Dataframe of paired values function evaluations
    :type pair_cols: pd.DataFrame
    :param mu_overall: a Pandas Dataframe of overall mu calculated on all
                                       function evaluation values for each time-step
    :type mu_overall: pd.DataFrame


    Returns:
    --------
    :return covariogram_values: the covariogram dataframe
    :rtype covariogram_values: pd.DataFrame


    References:
    -----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559


    Contributors:
    -------------
    Razavi, Saman, (2017): algorithm, code in MATLAB (c)
    Matott, Shawn, (2019): code in C/++
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
    """

    covariogram_values = (pair_cols[0].sub(mu_overall, level=0) * pair_cols[1].sub(mu_overall, level=0)) \
        .groupby(level=['ts', 'param', 'h']).mean()

    return covariogram_values


def e_covariogram(cov_section_all: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
    ------------
    This function return the Expected value of covariogram values derived from the pairs of 
    function evaluations that each resulted from each star point. This function
    is specific for the time-series varying/aggregate of the VARS sensitivity analysis.


    Parameters:
    -----------
    :param cov_section_all: a Pandas Dataframe of sectional covariograms
    :type cov_section_all: pd.DataFrame


    Returns:
    --------
    :return e_covariogram_values: the covariogram dataframe
    :rtype e_covariogram_values: pd.DataFrame


    References:
    -----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559


    Contributors:
    -------------
    Razavi, Saman, (2017): algorithm, code in MATLAB (c)
    Matott, Shawn, (2019): code in C/++
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
    """

    e_covariogram_values = cov_section_all.groupby(
        level=['ts', 'param', 'h']).mean()

    return e_covariogram_values


def sobol_eq(
        gamma: pd.DataFrame,
        ecov: pd.DataFrame,
        variance: pd.Series,
        delta_h: float
) -> pd.DataFrame:
    """
    Description:
    ------------
    This function return the Sobol Equivalent values derived from the variogram (`gamma`),
    expected values of sectional covariograms (`ecov`), and overall variance (`variance`).
    This function is specific for the time-series varying/aggregate of the VARS sensitivity
    analysis.


    Parameters:
    -----------
    :param gamma: a Pandas Dataframe of variogram values for each time-step
    :type gamma: pd.DataFrame
    :param ecov: a Pandas DataFrame of expected values of sectional covariograms
                             for each time-step
    :type ecov: pd.DataFrame
    :param variance: variance of function evaluations over all time-steps
    :type variance: pd.Series
    :param delta_h: resolution of star samples
    :type delta_h: float


    Returns:
    --------
    :return sobol_eq_values: the Sobol Equivalent values
    :rtype sobol_eq_values: pd.DataFrame


    References:
    -----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559


    Contributors:
    -------------
    Razavi, Saman, (2017): algorithm, code in MATLAB (c)
    Matott, Shawn, (2019): code in C/++
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
    """

    sobol_eq_values = (gamma + ecov).div(variance,
                                         level='ts').loc[:, :, delta_h]

    return sobol_eq_values


# ivars function
def ivars(
        variogram_array: pd.DataFrame,
        scale: float,
        delta_h: float
) -> pd.DataFrame:
    """
    Description:
    ------------
    Generates Integrated Variogram Across a Range of Scales (IVARS) by approximating 
    area using right trapezoids having width of `delta_h` and hights of variogram values.
    This function is specific for the time-series varying/aggregate of the VARS sensitivity
    analysis.


    Parameters:
    -----------
    :param variogram_array: a Pandas Dataframe of variogram values for each time-step
    :type variogram_array: pd.DataFrame
    :param scale: the scale for the IVARS evaluations
    :type scale: float
    :param delta_h: the resolution of star point generation
    :type delta_h: float


    Returns:
    --------
    :return ivars_values: the Sobol Equivalent values
    :rtype ivars_values: pd.DataFrame


    References:
    -----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559


    Contributors:
    -------------
    Razavi, Saman, (2017): algorithm, code in MATLAB (c)
    Matott, Shawn, (2019): code in C/++
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
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
