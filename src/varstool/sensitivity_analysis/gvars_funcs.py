# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats as stat

from scipy.optimize import newton_krylov
from scipy.integrate import dblquad

from itertools import combinations

from collections.abc import (
    Iterable,
)

from typing import (
    List,
    Tuple,
    Dict,
    Union
)

from tqdm.auto import tqdm

'''
Common functions used in GVARS analysis
'''

def rx2rn(distpair_type: List,
          param1: List,
          param2: List,
          rxpair: float
          ) -> float:
    """
    transforms value rx in a correlation matrix to value rn

    Parameters
    ----------
    distpair_type : List
        a list containing parameter 1 and parameter 2's distribution types
    param1 : List
        a list containing statistical information about parameter 1
    param2 : List
        a list containing statistical information about parameter 2
    rxpair : float
        value containing rx from the correlation matrix

    Returns
    -------
    rn : float
        the transformed rx value


    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive,
           robust, and efficient global sensitivity analysis: 1. Theory. Water
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558
    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive,
           robust, and efficient global sensitivity analysis: 1. Application. Water
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559
    .. [3] Razavi, S., & Do, C. N. (2020). Correlation Effects? A Major but Often
           Neglected Component in Sensitivity and Uncertainty Analysis. Water Resources
           Research, 56(3). doi: /10.1029/2019WR025436
    """

    # getting the inverse cdf of distribution 1
    if (distpair_type[0] == 'unif'):
        mu1 = (param1[1] + param1[0]) / 2
        std1 = (param1[1] - param1[0]) / 12 ** 0.5
        inv_cdf1 = lambda x: param1[0] + (param1[1] - param1[0]) * x
    elif (distpair_type[0] == 'norm'):
        mu1 = param1[0]
        std1 = param1[1]
        inv_cdf1 = lambda x: stat.norm.ppf(x, mu1, std1)
    elif (distpair_type[0] == 'triangle'):
        mu1 = (param1[0] + param1[1] + param1[2]) / 3
        std1 = (np.sqrt(
            param1[0] ** 2 + param1[1] ** 2 + param1[2] ** 2 - param1[0] * param1[1] - param1[0] * param1[2] - param1[
                1] * param1[2])) / np.sqrt(18)
        loc1 = param1[0]
        scale1 = param1[1] - param1[0]
        c1 = (param1[2] - param1[0]) / (param1[1] - param1[0])
        inv_cdf1 = lambda x: stat.triang.ppf(q=x, c=c1, loc=loc1, scale=scale1)
    elif (distpair_type[0] == 'lognorm'):
        mu1 = param1[0]
        std1 = param1[1]
        # compute associated normal
        cv = std1 / mu1 ** 2
        m = np.log(mu1 / (np.sqrt(1 + cv)))
        v = np.sqrt(np.log(1 + cv))
        inv_cdf1 = lambda x: stat.lognorm.ppf(x, scale=np.exp(m), s=v, loc=0)
    elif (distpair_type[0] == 'expo'):
        lamda = param1[0]
        mu1 = 1 / lamda
        std1 = 1 / (lamda ** 2)
        inv_cdf1 = lambda x: stat.expon.ppf(x, scale=mu1)
    elif (distpair_type[0] == 'gev'):
        mu = param1[0]  # location
        sigma = param1[1]  # scale
        k1 = -1 * param1[2]  # shape
        inv_cdf1 = lambda x: stat.genextreme.ppf(x, c=k1, scale=sigma, loc=mu);
        [mu1, std1] = stat.genextreme.stats(k1, scale=sigma, loc=mu);

    # getting the inverse cdf of distribution 2
    if (distpair_type[1] == 'unif'):
        mu2 = (param2[1] + param2[0]) / 2
        std2 = (param2[1] - param2[0]) / 12 ** 0.5
        inv_cdf2 = lambda x: param2[0] + (param2[1] - param2[0]) * x
    elif (distpair_type[1] == 'norm'):
        mu2 = param2[0]
        std2 = param2[1]
        inv_cdf2 = lambda x: stat.norm.ppf(x, mu2, std2)
    elif (distpair_type[1] == 'triangle'):
        mu2 = (param2[0] + param2[1] + param2[2]) / 3
        std2 = (np.sqrt(
            param2[0] ** 2 + param2[1] ** 2 + param2[2] ** 2 - param2[0] * param2[1] - param2[0] * param2[2] - param2[
                1] * param2[2])) / np.sqrt(18)
        loc2 = param2[0]
        scale2 = param2[1] - param2[0]
        c2 = (param2[2] - param2[0]) / (param2[1] - param2[0])
        inv_cdf2 = lambda x: stat.triang.ppf(q=x, c=c2, loc=loc2, scale=scale2)
    elif (distpair_type[1] == 'lognorm'):
        mu2 = param2[0]
        std2 = param2[1]
        # compute associated normal
        cv = std2 / mu2 ** 2
        m = np.log(mu2 / (np.sqrt(1 + cv)))
        v = np.sqrt(np.log(1 + cv))
        inv_cdf2 = lambda x: stat.lognorm.ppf(x, scale=np.exp(m), s=v, loc=0)
    elif (distpair_type[1] == 'expo'):
        lamda = param2[0]
        mu2 = 1 / lamda
        std2 = 1 / (lamda ** 2)
        inv_cdf2 = lambda x: stat.expon.ppf(x, scale=mu2)
    elif (distpair_type[1] == 'gev'):
        mu = param2[0]  # location
        sigma = param2[1]  # scale
        k2 = -1 * param2[2]  # shape
        inv_cdf2 = lambda x: stat.genextreme.ppf(x, c=k2, scale=sigma, loc=mu)
        [mu2, std2] = stat.genextreme.stats(k2, scale=sigma, loc=mu)

    # bivariate standard normal distribution
    stdnorm2_pdf = lambda x1, x2: np.exp(-1 * (x1 ** 2 + x2 ** 2) / 2.0) / (2.0 * np.pi)

    # integral bound zmax=5.0, zmin = -5.0
    integrand = lambda x1, x2: inv_cdf1(stat.norm.cdf(x1 * np.sqrt(1 - rxpair ** 2) + rxpair * x2, 0, 1)) * inv_cdf2(
        stat.norm.cdf(x2, 0, 1)) * stdnorm2_pdf(x1, x2)
    # compute double integral of integrand with x1 ranging from -5.0 to 5.0 and x2 ranging from -5.0 to 5.0
    integral_val = dblquad(integrand, -5, 5, lambda x: -5, lambda x: 5, epsabs=1.49e-06, epsrel=1.49e-06)[0]
    rn = (integral_val - mu1 * mu2) / (std1 * std2)

    return rn


def rn2rx(distpair_type: List,
          param1: List,
          param2: List,
          rnpair: float
          ) -> float:
    """
    transforms value rn in a correlation matrix to values rx

    Parameters
    ----------
    distpair_type : List
        a list containing parameter 1 and parameter 2's distribution type
    param1 : List
        a list containing statistical information about parameter 1
    param2 : List
        a list containing statistical information about parameter 2
    rnpair : float
        value containing rn from the correlation matrix

    Returns
    -------
    rx : float
        the transformed rn value


    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive,
           robust, and efficient global sensitivity analysis: 1. Theory. Water
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558
    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive,
           robust, and efficient global sensitivity analysis: 1. Application. Water
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559
    .. [3] Razavi, S., & Do, C. N. (2020). Correlation Effects? A Major but Often
           Neglected Component in Sensitivity and Uncertainty Analysis. Water Resources
           Research, 56(3). doi: /10.1029/2019WR025436
    """

    fun = lambda r: (rnpair - rx2rn(distpair_type, param1, param2, r))
    # try to find point x where fun(x) = 0
    try:
        rx = newton_krylov(F=fun, xin=rnpair, x_tol=1e-5)
    except:
        print("Function could not converge, fictive matrix was not computed")
        rx = rnpair

    return rx


def map_2_cornorm(parameters: Dict[Union[str, int], Tuple[Union[float, str]]],
                  corr_mat: np.ndarray, progress: bool
                  ) -> np.ndarray:
    """
    Computes the fictive correlation matrix given a correlation matrix and its
    corresponding parameters using nataf transformation

    Parameters
    ----------
    parameters : dictionary
        a dictionary containing parameter names and their attributes
    corr_mat : np.ndarray
        correlation matrix
    progress : boolean
        true if loading bar is to be shown, false otherwise

    Returns
    -------
    corr_n : np.ndarray
        the fictive correlation matrix


    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive,
           robust, and efficient global sensitivity analysis: 1. Theory. Water
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558
    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive,
           robust, and efficient global sensitivity analysis: 1. Application. Water
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559
    .. [3] Razavi, S., & Do, C. N. (2020). Correlation Effects? A Major but Often
           Neglected Component in Sensitivity and Uncertainty Analysis. Water Resources
           Research, 56(3). doi: /10.1029/2019WR025436
    """

    # store parameter info in a list
    param_info = list(parameters.values())

    corr_n = np.eye(corr_mat.shape[0], corr_mat.shape[1])

    for i in tqdm(range(0, corr_mat.shape[0] - 1), desc='building fictive matrix', disable=not progress, dynamic_ncols=True):
        for j in range(i + 1, corr_mat.shape[0]):
            # input paramter info
            corr_n[i][j] = rn2rx([param_info[i][3], param_info[j][3]],
                                 [param_info[i][0], param_info[i][1], param_info[i][2]],
                                 [param_info[j][0], param_info[j][1], param_info[j][2]], corr_mat[i][j])
            # matrix is symmetrical
            corr_n[j][i] = corr_n[i][j]
    return corr_n


def n2x_transform(norm_vectors: np.ndarray,
                  param_info: List
                  ) -> np.ndarray:
    """
    transforms multivariate normal samples into parameters original distributions

    Parameters
    ----------
    norm_vectors : np.ndarray
        multivariate normal samples
    param_info : list
        a list containing parameter information (bounds, distributions, etc.)

    Returns
    -------
    x : np.ndarray
        the transformed vectors


    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive,
           robust, and efficient global sensitivity analysis: 1. Theory. Water
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558
    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive,
           robust, and efficient global sensitivity analysis: 1. Application. Water
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559
    .. [3] Razavi, S., & Do, C. N. (2020). Correlation Effects? A Major but Often
           Neglected Component in Sensitivity and Uncertainty Analysis. Water Resources
           Research, 56(3). doi: /10.1029/2019WR025436
    """

    # Transform from correlated standard normal to original distributions

    k = norm_vectors.shape[1]
    x = np.zeros(norm_vectors.shape)

    for i in range(0, k):
        if param_info[i][3] == 'unif':
            lb = param_info[i][0]
            ub = param_info[i][1]

            x[:, i] = lb + (ub - lb) * stat.norm.cdf(norm_vectors[:, i], 0, 1)
        elif param_info[i][3] == 'norm':
            mu = param_info[i][0]
            std = param_info[i][1]

            x[:, i] = stat.norm.ppf(stat.norm.cdf(norm_vectors[:, i], 0, 1), mu, std)
        elif param_info[i][3] == 'triangle':
            a = param_info[i][0]
            b = param_info[i][1]
            c = param_info[i][2]
            mid = (c - a) / (b - a)
            term1 = (b - a) * (c - a)
            term2 = (b - a) * (b - c)
            x_norm = stat.norm.cdf(norm_vectors[:, i], 0, 1)
            x[:, i] = (a + np.sqrt(term1) * np.sqrt(x_norm)) * ((x_norm >= 0).astype(int)) * (
                (x_norm < mid).astype(int)) + (b - np.sqrt(term2) * np.sqrt((1 - x_norm))) * (
                          (x_norm >= mid).astype(int)) * ((x_norm < 1).astype(int))
        elif param_info[i][3] == 'lognorm':
            mu = param_info[i][0]
            std = param_info[i][1]
            term1 = std / mu ** 2
            m = np.log(mu / (np.sqrt(1 + term1)))
            v = np.sqrt(np.log(1 + term1))
            x[:, i] = stat.lognorm.ppf(stat.norm.cdf(norm_vectors[:, i], 0, 1), scale=np.exp(mu), s=std, loc=0)
        elif param_info[i][3] == 'expo':
            mu = param_info[i][0]
            x[:, i] = np.expon.ppf(stat.norm.cdf(norm_vectors[:, i], 0, 1), scale=mu)
        elif param_info[i][3] == 'gev':
            mu = param_info[i][0]  # location
            sigma = param_info[i][1]  # scale
            k = -1 * param_info[i][2]  # shape
            x[:, i] = stat.genextreme.ppf(stat.norm.cdf(norm_vectors[:, i], 0, 1), c=k, scale=sigma, loc=mu)

    return x


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


def reorder_pairs(pair_df: pd.DataFrame,
                  num_stars: int,
                  parameters: Dict[Union[str, int], Tuple[Union[float, str]]],
                  df: pd.DataFrame,
                  delta_h: float,
                  report_verbose: bool,
                  xmax: np.ndarray,
                  xmin: np.ndarray,
                  offline_mode: bool
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
        Pandas DataFrame containing the star points, and model outputs
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
    param_names = list(parameters.keys())
    for star_centre in star_centres:
        param_num=0
        for param in parameters.keys():
            pairs = pairs_h(df.loc[star_centre, param][param_names[param_num]].index.get_level_values(-1))
            for ignore, idx in pairs.items():
                for idx_tup in idx:
                    dist_list.append(np.abs((df.loc[star_centre, param][param_names[param_num]][idx_tup[0]] -
                                             df.loc[star_centre, param][param_names[param_num]][idx_tup[1]]) / (
                                                xmax[param_num]-xmin[param_num])))
            param_num = param_num + 1

    # loading bar for binning and reording pairs based on new 'h' values
    if report_verbose:
        star_centres = tqdm(range(0, num_stars), desc='binning and reording pairs based on \'h\' values')
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
        for param in parameters.keys():
            binned_pairs.append(
                pd.cut(pair_df.loc[star_centre, param, :]['actual h'], bins=bins, labels=labels).sort_values())

    # put binned pairs into a panda series
    binned_pairs = pd.concat(binned_pairs, ignore_index=False)

    # re order pairs values according to the bins
    pair_df = pair_df.loc[binned_pairs.index]

    # add in new index h, according to bin ranges
    # ex.) h = 0.1 = [0-0.15], h = 0.2 = [0.15-0.25]
    h = list(binned_pairs.values)
    pair_df['h'] = h

    # format data frame so that it works properly with variogram analsysis functions
    pair_df.set_index('h', append=True, inplace=True)
    pair_df.set_index('actual h', append=True, inplace=True)

    pair_df = pair_df.reorder_levels(['centre', 'param', 'h', 'actual h', 'pair_ind'])

    return pair_df


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
    # store parameter info in a list
    param_info = list(parameters.values())

    # store the max and min values of each paramter in arrays
    xmin = np.zeros(len(parameters))
    xmax = np.zeros(len(parameters))
    for i in range(0, len(parameters)):
        if param_info[i][3] == 'unif':
            xmin[i] = param_info[i][0]  # lower bound
            xmax[i] = param_info[i][1]  # upper bound
        elif param_info[i][3] == 'triangle':
            xmin[i] = param_info[i][0]  # lower bound
            xmax[i] = param_info[i][1]  # upper bound
        elif param_info[i][3] == 'norm':
            xmin[i] = param_info[i][0] - 3 * param_info[i][1]
            xmax[i] = param_info[i][0] + 3 * param_info[i][1]
        elif param_info[i][3] == 'lognorm':
            xmin[i] = 1
            xmax[i] = 1.25
        elif param_info[i][3] == 'expo':
            xmin[i] = 0  # change this
            xmax[i] = 0  # change this
        elif param_info[i][3] == 'gev':
            xmin[i] = 0  # change this
            xmax[i] = 0  # change this

    return xmin, xmax