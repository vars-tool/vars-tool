# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

'''
    A set of common functions used in the vars-tool package.
    These are sample functions to test the capabilities of
    varstool in sensitivity and uncertainty analysis.

'''

def ishigami(
        x,
        a: float=7,
        b: float=0.05
) -> float:
    """Ishigami test function.

    Parameters
    ----------
    x : array_like
        a list of parameters
    a : float, optional
    b : float, optional


    Returns
    -------
    model_val : float
        float value representing Ishigami calculation

    """
    # check whether the input x is an array-like

    if not isinstance(x, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray, list)):
        raise TypeError(
            '`x` must be of type pandas.DataFrame, numpy.ndarray, pd.Series, or list')

    if x.shape[0] > 3:
        raise ValueError('`x` must have only three arguments at a time')
    elif x.shape[0] < 3:
        raise ValueError(
            '`x` must have three arguments passed in an array-like object')

    return np.sin(x[0]) + a * (np.sin(x[1])**2) + b * (x[2]**4) * np.sin(x[0])


def linear_additive(x):
    """Linear additive test function accepting two parameters in an array_like object as ``x``

    Parameters
    ----------
    x : array_like
        a list of parameters


    Returns
    -------
    model_val : float
        float value representing linear additive calculation

    """
    term1 = 2*x[0]
    term2 = 3*x[1]
    return term1 + term2