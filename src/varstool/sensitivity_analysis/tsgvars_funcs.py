# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def ivars(
    variogram_array: pd.DataFrame,
    scale: float, delta_h: float
) -> pd.DataFrame:
    """Generates Integrated Variogram Across a Range of Scales (IVARS) by approximating
    area using right trapezoids having width of `delta_h` and hights of variogram values.
    This function is specific for the time-series varying/aggregate of the VARS sensitivity
    analysis.

    Parameters
    ----------
    variogram_array : array_like
        a Pandas Dataframe of variogram values for each time-step
    scale : float
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

    x_bench = [0] + variogram_array.index.dropna().get_level_values(2).to_list()
    x_int = np.arange(start=0, stop=(scale * 10 + 1) / 10, step=delta_h)

    # calculate interpolated values for both x (h) and y (variogram)
    if x_int[-1] < scale:
        x_int = np.append(x_int, scale)
    y_bench = [0] + variogram_array.to_list()

    y_int = np.interp(x=x_int, xp=x_bench, fp=y_bench)

    # for loop for each step size to caluclate the area
    ivars_values = 0
    for i in range(len(x_int) - 1):
        ivars_values += 0.5 * (y_int[i + 1] + y_int[i]) * (x_int[i + 1] - x_int[i])

    return ivars_values


def find_boundaries(parameters):
    """
    finds maximum and minimum boundary of each parameter.

    Parameters
    ----------
    parameters : Dictionary
        dictionary containing parameters names and attributes

    Returns
    -------
    xmin : array_like
        the lower boundaries of each parameter
    xmax : array_like
        the upper boundaries of each parameter
    """

    # store the max and min values of each parameter in arrays
    xmin = []
    xmax = []
    for param in sorted(parameters.keys()):
        if parameters[param][3] == 'unif':
            xmin.append(parameters[param][0])  # lower bound
            xmax.append(parameters[param][1])  # upper bound
        elif parameters[param][3] == 'triangle':
            xmin.append(parameters[param][0]) # lower bound
            xmax.append(parameters[param][1])  # upper bound
        elif parameters[param][3] == 'norm':
            xmin.append(parameters[param][0] - 3 * parameters[param][1])
            xmax.append(parameters[param][0] + 3 * parameters[param][1])
        elif parameters[param][3] == 'lognorm':
            xmin.append(1)
            xmax.append(1.25)
        elif parameters[param][3] == 'expo':
            xmin.append(0)  # change this
            xmax.append(0)  # change this
        elif parameters[param][3] == 'gev':
            xmin.append(0)  # change this
            xmax.append(0)  # change this

    return xmin, xmax
