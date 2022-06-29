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
        x_int.append(scale)
    y_bench = [0] + variogram_array.to_list()

    y_int = np.interp(x=x_int, xp=x_bench, fp=y_bench)

    # for loop for each step size to caluclate the area
    ivars_values = 0
    for i in range(len(x_int) - 1):
        ivars_values += 0.5 * (y_int[i + 1] + y_int[i]) * (x_int[i + 1] - x_int[i])

    return ivars_values