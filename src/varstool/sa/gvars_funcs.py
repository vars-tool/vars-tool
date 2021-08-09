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
    Callable,
    Optional,
    Any, Dict, Union
)

from tqdm.auto import tqdm
from time import sleep

'''
Common functions used in VARS analysis
'''


# Document all of these functions
def rx2rn(distpair_type, param1, param2, rxpair):
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
        mid1 = (param1[2] - param1[0]) / (param1[1] - param1[0])
        term11 = (param1[1] - param1[0]) * (param1[2] - param1[0])
        term21 = (param1[1] - param1[0]) * (param1[1] - param1[2])
        inv_cdf1 = lambda x: ((param1[0] + np.sqrt(term11) * np.sqrt(x / 1)) * ((x >= 0).astype(int)) * (
            (x < mid1).astype(int)) + (param1[1] - np.sqrt(term21) * np.sqrt(1 - x)) * ((x >= mid1).astype(int)) * (
                                  (x < 1).astype(int)))
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
        k1 = param1[2]  # shape
        inv_cdf1 = lambda x: stat.genextreme.ppf(x, c=k1, scale=sigma, loc=mu)
        [mu1, std1] = stat.genextreme.stats(k1, scale=sigma, loc=mu)

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
        mid2 = (param2[2] - param2[0]) / (param2[1] - param2[0])
        term12 = (param2[1] - param2[0]) * (param2[2] - param2[0])
        term22 = (param2[1] - param2[0]) * (param2[1] - param2[2])
        inv_cdf2 = lambda x: ((param2[0] + np.sqrt(term12) * np.sqrt(x / 1)) * ((x >= 0).astype(int)) * (
            (x < mid2).astype(int)) + (param2[1] - np.sqrt(term22) * np.sqrt(1 - x)) * ((x >= mid1).astype(int)) * (
                                  (x < 1).astype(int)))
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
        k2 = param2[2]  # shape
        inv_cdf2 = lambda x: stat.genextreme.ppf(x, c=k2, scale=sigma, loc=mu)
        [mu2, std2] = stat.genextreme.stats(k2, scale=sigma, loc=mu)

    # bivariate standard normal distribution
    stdnorm2_pdf = lambda x1, x2: np.exp(-1 * (x1 ** 2 + x2 ** 2) / 2.0) / (2.0 * np.pi)

    # integral bound zmax=5.0, zmin = -5.0
    integrand = lambda x1, x2: inv_cdf1(stat.norm.cdf(x1 * np.sqrt(1 - rxpair ** 2) + rxpair * x2, 0, 1)) * inv_cdf2(
        stat.norm.cdf(x2, 0, 1)) * stdnorm2_pdf(x1, x2)
    # compute double integral of integrand with x1 ranging from -5.0 to 5.0 and x2 ranging from -5.0 to 5.0
    rn = (dblquad(integrand, -5.0, 5.0, lambda x: -5.0, lambda x: 5.0) - mu1 * mu2) / (std1 * std2)

    return rn


def rn2rx(distpair_type, param1, param2, rnpair):
    fun = lambda r: (rnpair - rx2rn(distpair_type, param1, param2, r))
    # try to find point x where fun(x) = 0
    try:
        rx = newton_krylov(fun, rnpair, x_tol=1e-5)
    except:
        rx = rnpair

    return rx


def map_2_cornorm(parameters, corr_mat):
    # store parameter info in a list
    param_info = list(parameters.values())

    corr_n = np.eye(corr_mat.shape[0], corr_mat.shape[1])
    for i in range(0, corr_mat.shape[0] - 1):
        for j in range(i + 1, corr_mat.shape[0]):
            # input paramter info (lb, ub, ?, dist type)
            corr_n[i][j] = rn2rx([param_info[i][3], param_info[j][3]],
                                 [param_info[i][0], param_info[i][1], param_info[i][2]],
                                 [param_info[j][0], param_info[j][1], param_info[j][2]], corr_mat[i][j])
            # matrix is symmetrical
            corr_n[j][i] = corr_n[i][j]
    return corr_n


def n2x_transform(norm_vectors, parameters):
    # Transform from correlated standard normal to original distributions
    param_info = list(parameters.values())

    #
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
            x[:, i] = np.lognorm.ppf(stat.norm.cdf(norm_vectors[:, i], 0, 1), scale=np.exp(mu), s=std, loc=0)
        elif param_info[i][3] == 'expo':
            mu = param_info[i][0]
            x[:, i] = np.expon.ppf(stat.norm.cdf(norm_vectors[:, i], 0, 1), scale=mu)
        elif param_info[i][3] == 'gev':
            mu = param_info[i][0]  # location
            sigma = param_info[i][1]  # scale
            k = param_info[i][2]  # shape
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


def reorder_pairs(pair_df, num_stars, parameters, df, delta_h, report_verbose):

    # for loading bar when calculating differences in values 'h'
    if report_verbose:
        star_centres = tqdm(range(0, num_stars), desc='calculating \'h\' values')
    else:
        star_centres = range(0, num_stars)

    # gather the actual 'h' differences between each star point value for every pair
    # possibly find a faster way to do this later
    dist_list = []
    for star_centre in star_centres:
        param_num = 0
        for param in parameters.keys():
            pairs = pairs_h(df.loc[star_centre, param][param_num].index.get_level_values(-1))
            for ignore, idx in pairs.items():
                for idx_tup in idx:
                    dist_list.append(np.abs(
                        df.loc[star_centre, param][param_num][idx_tup[0]] - df.loc[star_centre, param][param_num][
                            idx_tup[1]]))

            param_num += 1

    # loading bar for binning and reording pairs based on new 'h' values
    if report_verbose:
        pairs_pbar = tqdm(desc='binning and reording pairs based on \'h\' values', total=2, dynamic_ncols=True)

    # drop old distance values
    pair_df = pair_df.droplevel('h')

    # add new distances to dataframe
    pair_df['h'] = dist_list

    # create bin ranges
    num_bins = int(1 / delta_h)  # the number of bins created by delta h
    bins = np.zeros(num_bins)  # create the array to hold the bin ranges
    bins[1:] = np.arange(start=delta_h / 2 + delta_h, step=delta_h, stop=1)  # create middle bin ranges

    # create labels for the bin ranges which will be the actual delta h values
    labels = np.arange(start=delta_h, step=delta_h, stop=1)

    # bin pair values according to their distances 'h' for each paramter at each star centre
    binned_pairs = []
    for star_centre in range(0, num_stars):
        for param in parameters.keys():
            binned_pairs.append(pd.cut(pair_df.loc[star_centre, param, :]['h'], bins=bins, labels=labels).sort_values())

    # put binned pairs into a panda series
    binned_pairs = pd.concat(binned_pairs, ignore_index=False)

    if report_verbose:
        pairs_pbar.update(1)

    # re order pairs values according to the bins
    pair_df = pair_df.loc[binned_pairs.index]

    # add in new index h, according to bin ranges
    # ex.) h = 0.1 = [0-0.15], h = 0.2 = [0.15-0.25]
    h = list(binned_pairs.values)

    # drop actual h values for new rounded ones
    pair_df.drop(columns='h')

    pair_df['h'] = h

    # format data frame so that it works properly with variogram analsysis functions
    pair_df.set_index('h', append=True, inplace=True)

    pair_df = pair_df.reorder_levels(['centre', 'param', 'h', 'pair_ind'])

    if report_verbose:
        sleep(0.1)
        pairs_pbar.update(1)

    return pair_df
