# -*- coding: utf-8 -*-

# dvars_sensitivity.py
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numba import jit
from tqdm import tqdm

"""
All code in this file was originally developed by Scott Shambaugh at https://github.com/scottshambaugh/monaco.

Modification Dates:
Cordell Blanchard 7/25/2022: made DVARS work with pandas dataframes instead of the monaco class.
"""


def calc_sensitivities(simulation_df: pd.DataFrame,
                       outvarname: str,
                       Hj: float = 0.5,
                       tol: float = 1e-6,
                       phi_max: float = 1e6,
                       phi0: float = 1.0,
                       correlation_func_type: str = 'Linear',
                       verbose: bool = False,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the global sensitivity indices and ratios for a specific output
    variable to each of a simulation's input variables.

    This implements the D-VARS algorithm as dscribed in Reference [1]_ to
    calculate global sensitivity indices from a set of given data. The
    implementation largely follows the Reference's supplementary notes, though
    Multi-SQP is replaced with scipy's minimize function implementing L-BFGS-B.
    See also Reference [2]_ for some theoretical background on the VARS and
    IVARS methods of calculating sensitivity indices.

    Parameters
    ----------
    simulation_df : pd.DataFrame
        The input simulation.
    outvarname : str
        The name of the output variable to calculate sensitivities for. Note
        that the output variable must be scalar.
    Hj : float, default: 1.0
        The fraction of the total parameter space to integrate over. Note that
        the linear correlation function only has one hyperparameter, so as the
        Reference notes it is unable to distinguish variogram effects at varying
        length scales. So, this should not be set to anything other than 1 in
        practice.
    tol : float, default 1e-6
        The convergence tolerance for scipy's minimize function acting on the
        negative log likelihood function.
    phi_max : float, default 1e6
        the upper bound value for the optimal phi value
    phi0 : float, default 1
        the value where we start our search for the optimal phi.
    correlation_func_type : str
        determines the type of covariance function to use
    verbose : bool, default False
        Whether to print diagnostic information.

    Returns
    -------
    sensitivities : numpy.ndarray
        The global sensitivity indices for the output variable for each of the
        sim's input variables.
    ratios : numpy.ndarray
        The global sensitivity ratios for the output variable for each of the
        sim's input variables, essentially the fraction of each input variable's
        ability to explain the output variance.

    References
    ----------
    .. [1] Sheikholeslami, Razi, and Saman Razavi. "A fresh look at variography:
           measuring dependence and possible sensitivities across geophysical
           systems from any given data." Geophysical Research Letters 47.20
           (2020): e2020GL089829.
    .. [2] Razavi, Saman, and Hoshin V. Gupta. "A new framework for
           comprehensive, robust, and efficient global sensitivity analysis:
           1. Theory." Water Resources Research 52.1 (2016): 423-439.
    """

    ninvars = simulation_df.shape[1] - 1  # the number of invars is equal to the number of columns - 1
    nobvs = simulation_df.shape[0] - 1  # number of observations is equal to the number of rows - 1

    # normalize input variables and output
    simulation_df = simulation_df.div(simulation_df.sum(axis=0), axis=1)

    # calculate optimal phi vaalues
    phi_opt = calc_phi_opt(simulation_df, ninvars, nobvs, outvarname, tol, phi_max, phi0, verbose)

    variance = np.var([simulation_df[outvarname]])

    sensitivities = np.zeros(ninvars)  # zeros the size of number of input variables
    for j in range(ninvars):  # iterate through number of input variables
        sensitivities[j] = calc_Gammaj(Hj, phi_opt[j], variance, correlation_func_type)
    ratios = sensitivities / sum(sensitivities)

    return sensitivities, ratios


def calc_phi_opt(simulation_df: pd.DataFrame,
                 ninvars: int,
                 nobvs: int,
                 outvarname: str,
                 tol: float = 1e-6,
                 phi_max: float = 1e6,
                 phi0: float = 1.0,
                 verbose: bool = False
                 ) -> np.ndarray:
    """
    Calculate the optimal hyperparameters for the covariance functions between
    the output variable and each of the input variables via maximum likelihood
    estimation (MLE). MLE works by minimizing a negative log-likelihood
    function.

    Parameters
    ----------
    simulation_df : pd.Dataframe
        The input simulation.
    ninvars : int
        number of input variables
    nobvs : int
        number of observations
    outvarname : str
        The name of the output variable to calculate sensitivities for. Note
        that the output variable must be scalar.
    tol : float, default 1e-6
        The convergence tolerance for scipy's minimize function acting on the
        negative log likelihood function.
    phi_max : float, default 1e6
        the upper bound value for the optimal phi value
    phi0 : float, default 1
        the value where we start our search for the optimal phi.
    verbose : bool, default False
        Whether to print diagnostic information.

    Returns
    -------
    phi_opt : numpy.ndarray
        The learned hyperparameters for the covariance functions.
    """
    phi_min = 0
    # replicate for # of dimensions
    phi0s = phi0 * np.ones(ninvars)
    bounds = [(phi_min, phi_max)] * ninvars

    # put input and outputs into seperate arrays for minimize function
    X = np.array(simulation_df.iloc[:, simulation_df.columns != outvarname])
    Y = np.reshape(np.array(simulation_df[outvarname]), (nobvs + 1, 1))

    if verbose:
        print('Calculating optimal hyperparameters Φ for ' +
                       f"'{outvarname}' covariances...")

    res = minimize(L_runner, phi0s, args=(X, Y, verbose), bounds=bounds,
                   tol=tol, method='L-BFGS-B', options={'disp': verbose})

    if verbose:
        print('Done calculating optimal hyperparameters')


    phi_opt = res.x

    return phi_opt


def calc_Gammaj(Hj: float,
                phij: float,
                variance: float,
                correlation_func_type: str='Linear',
                ) -> float:
    """
    Calculates the IVARS sensitivity index from a learned covariance function.

    This integrates the directional variogram for a specific input variable
    using trapezoidal integration.

    Parameters
    ----------
    Hj : float
        The fraction of the total parameter space to integrate over. Note that
        the linear correlation function only has one hyperparameter, so as the
        Reference notes it is unable to distinguish variogram effects at varying
        length scales. So, this should not be set to anything other than 1 in
        practice.
    phij : float
        The learned hyperparameter for the covariance function between the
        output and input variable.
    variance : float
        The variance of the output variable.
    correlation_func_type : str
        determines the type of covariance function to use

    Returns
    -------
    Gammaj : float
        The global sensitivity index for this output-input variable pair.
    """
    dh = 1e-3
    q = int(np.floor(Hj / dh))
    rjs = np.zeros(q + 1)
    for i in range(q + 1):
        rjs[i] = calc_rj(dh * i, phij, correlation_func_type)

    Gammaj = np.trapz(1 - rjs) * dh * variance
    return Gammaj


def L_runner(phi: np.ndarray,
             X: np.ndarray,
             Y: np.ndarray,
             verbose: bool
             ) -> float:
    """
    A wrapper function for calculating the negative log-likelihood cost.

    Parameters
    ----------
    phi : numpy.ndarray
        The hyperparameters for the covariance function.
    X : numpy.ndarray
        The state matrix for all the input variables percentiles.
    Y : numpy.ndarray
        The state matrix for the output variables nums.
    verbose: bool, default False
        whether to print diagnostic information or not

    Returns
    -------
    L : float
        The negative log-likelihood cost.
    """
    L = calc_L(phi, X, Y)

    if verbose:
        print(f'L = {L:0.4f}, Φ = {phi}')
    return L


@jit(nopython=True, cache=True)
def calc_L(phi: np.ndarray,
           X: np.ndarray,
           Y: np.ndarray
           ) -> float:
    """
    Calculate the negative log-likelihood cost. Note that this is just-in-time
    compiled by numba for increased speed.

    Parameters
    ----------
    phi : numpy.ndarray
        The hyperparameters for the covariance function.
    X : numpy.ndarray
        The state matrix for all the input variables percentiles.
    Y : numpy.ndarray
        The state matrix for the output variables nums.

    Returns
    -------
    L : float
        The negative log-likelihood cost.
    """
    m = len(Y)
    M = np.ones((m, 1))
    R = calc_R(phi, X)
    Rinv = np.linalg.inv(R)
    Rdet = max(np.linalg.det(R), 1e-12)  # Protect for poor conditioning

    mu = np.linalg.inv(M.T @ Rinv @ M) @ (M.T @ Rinv @ Y)

    L_inner = Y - M * mu
    L = np.log(Rdet) / m + m * np.log(L_inner.T @ Rinv @ L_inner)
    L = L[0][0]
    return L


@jit(nopython=True, cache=True)
def calc_R(phi: np.ndarray,
           X: np.ndarray,
           ) -> np.ndarray:
    """
    Calculate the correlation matrix between each of the input states.
    Note that this is just-in-time compiled by numba for increased speed.

    Parameters
    ----------
    phi : numpy.ndarray
        The hyperparameters for the covariance function.
    X : numpy.ndarray
        The state matrix for all the input variables percentiles.

    Returns
    -------
    R : float
        The correlation matrix.
    """
    m = X.shape[0]
    R = np.ones((m, m))
    for u in range(1, m):
        # do lower triangle only and duplicate across diag
        # diag will be all 1s
        for w in range(1, u):
            Ruw = calc_Ruw(phi, X[u, :], X[w, :])
            R[u, w] = Ruw
            R[w, u] = Ruw
    return R


@jit(nopython=True, cache=True)
def calc_Ruw(phi: np.ndarray,
             Xu: np.ndarray,
             Xw: np.ndarray
             ) -> float:
    """
    Calculate the correlation between two input states.
    Note that this is just-in-time compiled by numba for increased speed.

    Parameters
    ----------
    phi : numpy.ndarray
        The hyperparameters for the covariance function.
    Xu : numpy.ndarray
        The u'th input state.
    Xw : numpy.ndarray
        The w'th input state.

    Returns
    -------
    Ruw : float
        The correlation between the two states.
    """
    h = Xu - Xw
    Ruw = 1
    for j, hj in enumerate(h):
        Ruw = Ruw * calc_rj(hj, phi[j])
    return Ruw


@jit(nopython=True, cache=True)
def calc_rj(hj: float,
            phij: float,
            correlation_func_type: Optional[str] = 'linear',
            ) -> float:
    """
    The covariance function (also called a kernel). We currently use a linear
    kernel which has a single hyperparameter that must be learned.
    Note that this is just-in-time compiled by numba for increased speed.

    Parameters
    ----------
    hj : float
        The distance between two state elements.
    phij : float
        The hyperparameter for the function.
    correlation_func_type: str
        determines the type of covariance function to be used

    Returns
    -------
    rj : float
        The covariance.
    """

    rj = 0 # initialize rj

    if correlation_func_type == 'linear':
        rj = max(0, 1 - phij * abs(hj))  # linear covariance function
    elif correlation_func_type == 'exponential':
        rj = np.exp(-(abs(hj)/phij))  # exponential covariance function
    elif correlation_func_type == 'square':
        rj = np.exp(-(hj/phij)**2)  # squared exponential covariance function

    return rj

