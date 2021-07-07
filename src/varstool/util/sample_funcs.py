# -*- coding: utf-8 -*-

'''
    A set of common functions used in the vars-tool package.
    These are sample functions to test the capabilities of
    varstool in sensitivity and uncertainty analysis.

''' 
import numpy as np
import pandas as pd


def ishigami(x, a=7, b=0.05):
    '''Ishigami test function'''
    # check whether the input x is an array-like

    if not isinstance(x, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray, list)):
        raise TypeError('`x` must be of type pandas.DataFrame, numpy.ndarray, pd.Series, or list')

    if x.shape[0] > 3:
        raise ValueError('`x` must have only three arguments at a time')
    elif x.shape[0] <3:
        raise ValueError('`x` must have three arguments passed in an array-like object')

    return np.sin(x[0]) + a*(np.sin(x[1])**2) + b*(x[2]**4)*np.sin(x[0])
