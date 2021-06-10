'''
    A set of common functions used in the vars-tool package.
    If the function could be used as a decorator, it is mentioned in
    its docstring.

''' 

import numpy as np


def scale(df, bounds, axis=1, *args, **kwargs):
    '''
    scale the sampled matrix
    bounds is a dict with ['ub', 'lb'] keys
    the values are lists of the upper and lower bounds
    of the parameters/variables/factors
    '''

    # numpy equivalent for math operations
    bounds_np = {key:np.array(value) for key,value in bounds.items()}

    if axis:
        return df * (bounds_np['ub'] - bounds_np['lb']) + bounds_np['lb']
    else:
        return df.T * (bounds_np['ub'] - bounds_np['lb']) + bounds_np['lb']
