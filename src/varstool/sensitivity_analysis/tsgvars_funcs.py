# -*- coding: utf-8 -*-
from itertools import combinations

import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from typing import (
    List,
    Tuple,
    Dict,
    Union, Iterable
)

def pairs_h(
    iterable: Iterable
) -> pd.DataFrame:
    """Give the pairs of numbers considering their differences.

    Parameters
    ----------
    iterable : iterable
        an iterable object

    Returns
    -------
    pairs : array_like
        the returned dataframe of paired values
    """

    # gives the pairs of numbers considering their differences
    interval = range(min(iterable), max(iterable) - min(iterable))
    pairs = {key + 1: [j for j in combinations(iterable, 2) if np.abs(
        j[0] - j[1]) == key + 1] for key in interval}

    return pairs

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


def find_boundaries(parameters, dist_sample_file):
    """
    finds maximum and minimum boundary of each parameter.

    Parameters
    ----------
    parameters : Dictionary
        dictionary containing parameters names and attributes
    dist_sample_file : str
        name of file containing distributions data

    Returns
    -------
    xmin : array_like
        the lower boundaries of each parameter
    xmax : array_like
        the upper boundaries of each parameter
    """

    # store distributions in dataframe
    distributions_df = pd.DataFrame()
    if dist_sample_file:
        distributions_df = pd.read_csv(dist_sample_file)

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
        elif parameters[param][3] == 'custom':
            xmin.append(distributions_df[param].min())
            xmax.append(distributions_df[param].max())

    return xmin, xmax

def reorder_pairs(pair_df: pd.DataFrame,
                  num_stars: int,
                  parameters: Dict[Union[str, int], Tuple[Union[float, str]]],
                  df: pd.DataFrame,
                  delta_h: float,
                  report_verbose: bool,
                  xmax: np.ndarray,
                  xmin: np.ndarray
                  ) -> pd.DataFrame:

    """
    Calculates the differences('h') between the pairings of the star points, and
    bins and reorders the pair dataframe according to the calculated 'h' values

    Parameters
    ----------
    pair_df : pd.DataFrame
        Pandas DataFrame containing the paired star points values with the model outputs
    num_stars : int
        number of star samples
    parameters : dictionary
        dictionary containing parameter names and their attributes
    df : pd.DataFrame
        Pandas DataFrame containing the star points
    delta_h : float
        resolution of star samples
    report_verbose : boolean
        if True will use a loading bar when generating stars, does nothing if False
    xmax : arraylike
        array containing max boundary of each parameter
    xmin : arraylike
        array containing min boundary of each parameter
    offline_mode : boolean
        if True GVARS analysis is in offline mode, if False it is in online mode

    Returns
    -------
    pair_df : array_like
        the returned dataframe of paired values
    """

    # for loading bar when calculating differences in values 'h'
    if report_verbose:
        star_centres = tqdm(range(0, num_stars), desc='calculating \'h\' values')
    else:
        star_centres = range(0, num_stars)

    # gather the actual 'h' differences between each star point value for every pair
    # possibly find a faster way to do this later
    dist_list = []
    param_names = sorted(list(parameters.keys()))
    for star_centre in star_centres:
        param_num=0
        for param in sorted(parameters.keys()):
            pairs = pairs_h(df.loc[star_centre, param][param_names[param_num]].index.get_level_values(-1))
            for ignore, idx in pairs.items():
                for idx_tup in idx:
                    dist_list.append(np.abs((df.loc[star_centre, param][param_names[param_num]][idx_tup[0]] -
                                             df.loc[star_centre, param][param_names[param_num]][idx_tup[1]]) / (
                                                xmax[param_num]-xmin[param_num])))
            param_num = param_num + 1

    # loading bar for binning and reording pairs based on new 'h' values
    if report_verbose:
        star_centres = tqdm(range(0, num_stars), desc='binning pairs based on \'h\' values')
    else:
        star_centres = range(0, num_stars)

    # add new distances to dataframe
    pair_df['actual h'] = dist_list

    # create bin ranges
    num_bins = int(1 / delta_h)  # the number of bins created by delta h
    bins = np.zeros(num_bins + 1)
    bins[1:] = np.arange(start=delta_h / 2, step=delta_h, stop=1)  # create middle bin ranges

    # create labels for the bin ranges which will be the actual delta h values
    labels = np.zeros(num_bins)
    labels[0] = delta_h / 4
    labels[1:] = np.arange(start=delta_h, step=delta_h, stop=1)

    # bin pair values according to their distances 'h' for each paramter at each star centre
    binned_pairs = []
    for star_centre in star_centres:
        for param in sorted(parameters.keys()):
            binned_pairs.append(
                pd.cut(pair_df.loc[star_centre, param, :]['actual h'], bins=bins, labels=labels).sort_values())

    # put binned pairs into a panda series
    binned_pairs = pd.concat(binned_pairs, ignore_index=False)

    return binned_pairs, dist_list
