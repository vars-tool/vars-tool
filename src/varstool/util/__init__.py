'''
    A set of common functions used in the vars-tool package.
    If the function could be used as a decorator, it is mentioned in
    its docstring.

'''

import numpy as np

__all__ = ["scale", "ishigami"]


def scale(df, bounds, axis=1, *args, **kwargs):
    """
    Description:
    ------------
    This function scales the sampled matrix `df` to the `bounds`
    that is a defined via a dictionary with ['ub', 'lb'] keys;
    the values of the dictionary are lists of the upper and lower
    bounds of the parameters/variables/factors. if (``axis = 1``)
    then each row is selected, otherwise each column.


    Parameters:
    -----------
    :param df: a dataframe of randomly sampled values
    :type df: pd.DataFrame
    :param bounds: a lower and upper bounds to scale the values
    :type bounds: dict
    :param axis: 0 for index, 1 for columns
    :type axis: int


    Returns:
    --------
    :return df: the returned dataframe scaled using bounds
    :rtype df: pd.DataFrame


    Contributors:
    -------------
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
    """

    # numpy equivalent for math operations
    bounds_np = {key: np.array(value) for key, value in bounds.items()}

    if axis:
        return df * (bounds_np['ub'] - bounds_np['lb']) + bounds_np['lb']
    else:
        return df.T * (bounds_np['ub'] - bounds_np['lb']) + bounds_np['lb']
