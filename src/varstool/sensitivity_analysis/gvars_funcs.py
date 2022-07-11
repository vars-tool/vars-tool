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
    Union, Optional
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
        print("It is recommended to switch fictive_mat_flag parameter to False to assume that correlation "
              "matrix is equal to fictive matrix")
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
                  parameters: Dict,
                  dist_sample_file: Optional[str] = None
                  ) -> np.ndarray:
    """
    transforms multivariate normal samples into parameters original distributions

    Parameters
    ----------
    norm_vectors : np.ndarray
        multivariate normal samples
    parameters : dict
        a dictionary containing parameter information (name: bounds, distributions, etc.)
    dist_sample_file : String
        name of file that contains custom distribution data, optional only for users with custom distributions

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

    x = np.zeros(norm_vectors.shape)
    i = 0

    for param in parameters.keys():
        if parameters[param][3] == 'unif':
            lb = parameters[param][0]
            ub = parameters[param][1]

            x[:, i] = lb + (ub - lb) * stat.norm.cdf(norm_vectors[:, i], 0, 1)
        elif parameters[param][3] == 'norm':
            mu = parameters[param][0]
            std = parameters[param][1]

            x[:, i] = stat.norm.ppf(stat.norm.cdf(norm_vectors[:, i], 0, 1), mu, std)
        elif parameters[param][3] == 'triangle':
            a = parameters[param][0]
            b = parameters[param][1]
            c = parameters[param][2]
            mid = (c - a) / (b - a)
            term1 = (b - a) * (c - a)
            term2 = (b - a) * (b - c)
            x_norm = stat.norm.cdf(norm_vectors[:, i], 0, 1)
            x[:, i] = (a + np.sqrt(term1) * np.sqrt(x_norm)) * ((x_norm >= 0).astype(int)) * (
                (x_norm < mid).astype(int)) + (b - np.sqrt(term2) * np.sqrt((1 - x_norm))) * (
                          (x_norm >= mid).astype(int)) * ((x_norm < 1).astype(int))
        elif parameters[param][3] == 'lognorm':
            mu = parameters[param][0]
            std = parameters[param][1]
            term1 = std / mu ** 2
            x[:, i] = np.lognorm.ppf(stat.norm.cdf(norm_vectors[:, i], 0, 1), scale=np.exp(mu), s=std, loc=0)
        elif parameters[param][3] == 'expo':
            mu = parameters[param][0]
            x[:, i] = np.expon.ppf(stat.norm.cdf(norm_vectors[:, i], 0, 1), scale=mu)
        elif parameters[param][3] == 'gev':
            mu = parameters[param][0]  # location
            sigma = parameters[param][1]  # scale
            k = -1 * parameters[param][2]  # shape
            x[:, i] = stat.genextreme.ppf(stat.norm.cdf(norm_vectors[:, i], 0, 1), c=k, scale=sigma, loc=mu)
        elif parameters[param][3] == 'custom':
            cdp = custom_distribution_probabilites(dist_sample_file, param)
            x[:, i] = np.interp(stat.norm.cdf(norm_vectors[:, i], 0, 1), cdp['Probabilities'], cdp[param])

        i += 1

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
    centres = pair_df.index.get_level_values(0).to_numpy()
    params = pair_df.index.get_level_values(1).to_numpy()
    bps = binned_pairs.index.to_numpy()
    new_index = pd.MultiIndex.from_arrays([centres, params, bps], names = ['centre', 'param', 'pair_ind'])
    pair_df = pair_df.reindex(new_index)

    # add in new index h, according to bin ranges
    # ex.) h = 0.1 = [0-0.15], h = 0.2 = [0.15-0.25]
    h = list(binned_pairs.values)
    pair_df['h'] = h

    # format data frame so that it works properly with variogram analsysis functions
    pair_df.set_index('h', append=True, inplace=True)
    pair_df.set_index('actual h', append=True, inplace=True)

    pair_df = pair_df.reorder_levels(['centre', 'param', 'h', 'actual h', 'pair_ind'])

    return pair_df


def find_boundaries(parameters, dist_sample_file: Optional[str] = None):
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

    # store the max and min values of each paramter in arrays
    xmin = np.zeros(len(parameters))
    xmax = np.zeros(len(parameters))
    index = 0
    for param in parameters.keys():
        if parameters[param][3] == 'unif':
            xmin[index] = parameters[param][0]  # lower bound
            xmax[index] = parameters[param][1]  # upper bound
        elif parameters[param][3] == 'triangle':
            xmin[index] = parameters[param][0]  # lower bound
            xmax[index] = parameters[param][1]  # upper bound
        elif parameters[param][3] == 'norm':
            xmin[index] = parameters[param][0] - 3 * parameters[param][1]
            xmax[index] = parameters[param][0] + 3 * parameters[param][1]
        elif parameters[param][3] == 'lognorm':
            xmin[index] = 1
            xmax[index] = 1.25
        elif parameters[param][3] == 'expo':
            xmin[index] = 0  # change this
            xmax[index] = 0  # change this
        elif parameters[param][3] == 'gev':
            xmin[index] = 0  # change this
            xmax[index] = 0  # change this
        elif parameters[param][3] == 'custom':
            # read in file and find min and max
            xmin[index] = distributions_df[param].min()
            xmax[index] = distributions_df[param].max()
        index += 1

    return xmin, xmax


def custom_distribution_probabilites(dist_sample_file: Optional[str], param):
    """
    finds empirical cdf for custom probability distribution and puts it in a dataframe.

    Parameters
    ----------
    dist_sample_file : str
        string name of .csv file containing custom distribution data
    param : String
        name of parameter

    Returns
    -------
    cdp : array_like
        df containing custom distributions and empirical cdf
    """

    cdp = pd.read_csv(dist_sample_file)

    # get just the singular parameter distribution
    cdp = cdp[param].to_frame().dropna(how='all').reset_index(drop=True)

    # sort data from smallest to largest
    cdp.sort_values(by=param, ignore_index=True, inplace=True)

    # find empirical cdf for all parameters
    cdp['Probabilities'] = (cdp.index + 1) / (cdp.shape[0] + 1)

    return cdp
