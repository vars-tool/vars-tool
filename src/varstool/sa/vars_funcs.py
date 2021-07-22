# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats as stat
import scipy.cluster.hierarchy as hchy

from itertools import combinations, compress

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

'''
Common functions used in VARS analysis
'''


def apply_unique(
    func: Callable,
    df: pd.DataFrame,
    axis: int=1,
    progress: bool=False,
) -> pd.DataFrame:
    """Apply ``func`` to unique rows (``axis=1``) or columns (``axis=0``)
    of ``df`` in order to increase the efficiency of `func` evaluations. 

    Parameters
    ----------
    func : Callable
        the function of interest to be applied to df
    df : array_like
        the Pandas DataFrame of interest
    axis : int, optional
        ``0`` for `index`, ``1`` for `columns`, defaults to ``1``
    progress: bool, optional
        ``False`` for hiding the progress bar, ``True`` for otherwise,
        defaults to ``False``

    Returns
    -------
    applied_df : array_like
        the returned dataframe with the `func` evaluations

    """
    if progress:
        tqdm.pandas(desc='function evaluation', dynamic_ncols=True)
        applied_df = df.merge(df.drop_duplicates()
                              .assign(**{func.__name__: lambda x: x.progress_apply(func, axis=axis)}),
                              how='left')
    else:
        applied_df = df.merge(df.drop_duplicates()
                              .assign(**{func.__name__: lambda x: x.apply(func, axis=axis)}),
                              how='left')

    applied_df.index = df.index

    return applied_df


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


def scale(
    df: pd.DataFrame,
    bounds: pd.DataFrame,
    axis: int=1
) -> pd.DataFrame:
    """This function scales the sampled matrix ``df`` to the ```bounds```
    that is a defined via a dictionary with ``ub``, ``lb`` keys;
    the values of the dictionary are lists of the upper and lower
    bounds of the parameters/variables/factors. if (``axis = 1``)
    then each row of `df` is selected, otherwise columns.

    Parameters
    ----------
    df : array_like
        a dataframe of randomly sampled values
    bounds : dict
        a lower and upper bounds to scale the values
    axis : int, optional
        ``0`` for index, ``1`` for columns

    Returns
    -------
    df : array_like
        the returned dataframe scaled using bounds

    """

    # numpy equivalent for math operations
    bounds_np = {key: np.array(value) for key, value in bounds.items()}

    if axis:
        return df * (bounds_np['ub'] - bounds_np['lb']) + bounds_np['lb']
    else:
        return df.T * (bounds_np['ub'] - bounds_np['lb']) + bounds_np['lb']


def section_df(
    df: pd.DataFrame,
    delta_h: float,
) -> pd.DataFrame:
    """This function gets the paired values of each section based on index.

    Parameters
    ----------
    df : array_like
        a dataframe of star points
    delta_h : float
        resolution of star samples

    Returns
    -------
    sample : array_like
        the paired values for each section of star points

    """

    pairs = pairs_h(df.index.get_level_values(-1))
    df_values = df.to_numpy()
    sample = pd.concat({h * delta_h:  # realistic delta_h values are shown
                        pd.DataFrame.from_dict({str(idx_tup): [
                                               df_values[idx_tup[0]], df_values[idx_tup[1]]] for idx_tup in idx}, 'index')
                        for h, idx in pairs.items()})

    return sample


# VARS core functions
def cov_section(
    pair_cols: pd.DataFrame,
    mu_star: pd.DataFrame
) -> pd.DataFrame:
    """This function return the sectional covariogram of the pairs of function evaluations
    that resulted from each star point. This function is specific for the time-series
    varying/aggregate of the VARS sensitivity analysis.

    Parameters
    ----------
    pair_cols : array_like
        a Pandas Dataframe of paired values function evaluations
    mu_star : array_like
        a Pandas DataFrame of mu star values that are calculated separately

    Returns
    -------
    cov_section_values : array_like
        the sectional covariogram dataframe

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    cov_section_values = (pair_cols.sub(mu_star, axis=0)[0] * pair_cols.sub(mu_star, axis=0)[1]).\
        groupby(level=['centre', 'param', 'h']).mean()

    return cov_section_values


def variogram(
    pair_cols: pd.DataFrame
) -> pd.DataFrame:
    """This function return the variogram calculated from the pairs of function evaluations
    that each resulted from each star point. This function is specific for the time-series
    varying/aggregate of the VARS sensitivity analysis.

    Parameters
    ----------
    pair_cols : array_like
        a Pandas Dataframe of paired values function evaluations

    Returns
    -------
    variogram_values : array_like
        the variogram dataframe

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    variogram_values = 0.5 * \
        (pair_cols[0] - pair_cols[1]
         ).pow(2).groupby(level=['param', 'h']).mean()

    return variogram_values


def morris_eq(
    pair_cols: pd.DataFrame
) -> pd.DataFrame:
    """This function return the Morris Equivalent values derived from the pairs of 
    function evaluations that each resulted from each star point. This function
    is specific for the time-series varying/aggregate of the VARS sensitivity analysis.

    Parameters
    ----------
    pair_cols : array_like
        a Pandas Dataframe of paired values function evaluations

    Returns
    -------
    morris_eq_values : array_like
        the morris dataframe

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    morris_eq_values = ((pair_cols[1] - pair_cols[0]).abs().groupby(level=['param', 'h']).mean(),
                        (pair_cols[1] - pair_cols[0]).groupby(level=['param', 'h']).mean())

    return morris_eq_values


def covariogram(
    pair_cols: pd.DataFrame,
    mu_overall: pd.Series,
) -> pd.DataFrame:
    """This function return the covariogram values derived from the pairs of 
    function evaluations that each resulted from each star point. This function
    is specific for the time-series varying/aggregate of the VARS sensitivity analysis.

    Parameters
    ----------
    pair_cols : array_like
        a Pandas Dataframe of paired values function evaluations
    mu_overall : array_like
        a Pandas Dataframe of overall mu calculated on all
        function evaluation values for each time-step

    Returns
    -------
    covariogram_values : array_like
        the covariogram dataframe

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    covariogram_values = ((pair_cols - mu_overall)[0] * (
        pair_cols - mu_overall)[1]).groupby(level=['param', 'h']).mean()

    return covariogram_values


def e_covariogram(
    cov_section_all: pd.DataFrame
) -> pd.DataFrame:
    """This function return the Expected value of covariogram values derived from the pairs of 
    function evaluations that each resulted from each star point. This function
    is specific for the time-series varying/aggregate of the VARS sensitivity analysis.

    Parameters
    ----------
    cov_section_all : array_like
        a Pandas Dataframe of sectional covariograms

    Returns
    -------
    e_covariogram_values : array_like
        the covariogram dataframe

    References
    ----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558

    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive, 
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559

    """

    e_covariogram_values = cov_section_all.groupby(level=['param', 'h']).mean()

    return e_covariogram_values


def sobol_eq(
        gamma: pd.DataFrame,
        ecov: pd.DataFrame,
        variance: pd.Series,
        delta_h: float
) -> pd.DataFrame:
    """This function return the Sobol Equivalent values derived from the variogram (`gamma`),
    expected values of sectional covariograms (`ecov`), and overall variance (`variance`).
    This function is specific for the time-series varying/aggregate of the VARS sensitivity
    analysis.

    Parameters
    ----------
    gamma : array_like
        a Pandas Dataframe of variogram values for each time-step
    ecov : array_like
        a Pandas DataFrame of expected values of sectional covariograms
        for each time-step
    variance : array_like
        variance of function evaluations over all time-steps
    delta_h : float
        resolution of star samples

    Returns
    -------
    sobol_eq_values : array_like
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

    sobol_eq_values = ((gamma + ecov) / variance)[:, delta_h]  # to

    return sobol_eq_values


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

    num_h = len(variogram_array.index.levels[-1].to_list())
    x_bench = np.arange(start=0, stop=delta_h * (num_h + 1), step=delta_h)
    x_int = np.arange(start=0, stop=(scale * 10 + 1) / 10, step=delta_h)

    # calculate interpolated values for both x (h) and y (variogram)
    if x_int[-1] < scale:
        x_int.append(scale)
    y_bench = [0] + variogram_array.to_list()

    y_int = np.interp(x=x_int, xp=x_bench, fp=y_bench)

    # for loop for each step size to caluclate the area
    ivars_values = 0
    for i in range(len(x_int) - 1):
        ivars_values += 0.5 * \
            (y_int[i + 1] + y_int[i]) * (x_int[i + 1] - x_int[i])

    return ivars_values


def factor_ranking(factors):
    """Ranks factors based on their influence (how large or small results are)
    The lowest rank corresponds to the most influential (larger) factor

    Parameters
    ----------
    factors : array_like
        an array like object that contains factors/parameters/variables of the
        sensitivity analysis problem

    Returns
    -------
    ranks : array_like
        a numpy array containing the ranks of each factor in their corresponding index

    """

    # check the factors is array like
    if not isinstance(factors,
                      (pd.DataFrame, pd.Series, np.ndarray, List, Tuple)):
        raise TypeError(
            "factors must be an array-like object: "
            "pandas.Dataframe, pandas.Series, numpy.array, List, Tuple"
        )

    # gather indices for sorting factor in descending order
    temp = np.argsort(factors)[::-1]
    # create an array the same shape and type as temp
    ranks = np.empty_like(temp)
    # rank factors with highest value being the lowest rank
    ranks[temp] = np.arange(len(factors))

    return ranks


def factor_grouping(
        sens_idx: pd.DataFrame,
        num_grp: int=None
) -> pd.DataFrame:
    """

    KK: DOCSTRING IS MISSING - PLEASE FILL & REMOVE THIS LINE THEREAFTER

    """
    [m, n] = sens_idx.shape

    # make data 1d
    r = sens_idx.stack()
    # replacing zeros with a constant number due to numerical reasoning
    r[r == 0] = np.ones(len(r[r == 0]))

    # do a box-cox transformation
    [transdat, lam] = stat.boxcox(r)
    if lam <= 0.0099:
        transdat = np.log(r)

    indices = np.argwhere(np.isinf(transdat).tolist())
    if indices.shape == (2, 1):
        transdat[indices[0], indices[1]] = np.log(r[r > 0])

    # reshape data for the linkage calculation
    s = np.reshape(transdat.tolist(), [n, m])

    # Agglomerative hierarchical cluster
    z = hchy.linkage(s, method='ward', metric='euclidean')

    # Optimal group number
    clusters = []
    for i in range(2, n + 1):
        clusters.append(hchy.fcluster(z, criterion='maxclust', t=i))
    # if user gives the group number preform calculations
    if num_grp:
        rank_grp = hchy.fcluster(z, criterion='maxclust', t=num_grp)
        optm_num_grp = num_grp
        nn = 1
        id = len(z)
        while nn != optm_num_grp:
            cutoff = z[id - 1][2]
            rank_grp = hchy.fcluster(z, criterion='distance', t=cutoff)
            nn = np.amax(rank_grp)
            id = id - 1

    # if user does not give optimal group number use elbow method
    else:
        cutoff = elbow_method(z)
        rank_grp = hchy.fcluster(z, criterion='distance', t=cutoff)
        optm_num_grp = max(rank_grp)

    return optm_num_grp, rank_grp, clusters


def elbow_method(z):
    """
    KK: DOCSTRING IS MISSING - FILL & REMOVE THIS LINE THEREAFTER
    """
    # creating Q1 and Q2 for elbow method calculations
    q1 = np.array([1, z[0][2]])
    q2 = np.array([len(z), z[-1][2]])

    # Use elbow method to find the cutoff and color threshold for clustering
    d = []
    for i in range(0, len(z) - 2):
        p = [i + 1, z[i][2]]
        d.append(np.abs(np.linalg.det(
            np.array([[q2 - q1], [p - q1]]))) / np.linalg.norm(q2 - q1))

    id = d.index(max(d))
    cutoff = z[id][2]

    return cutoff


def grouping(
    result_bs_ivars_df: pd.DataFrame,
    result_bs_sobol: pd.DataFrame,
    result_bs_ivars_ranking: pd.DataFrame,
    result_bs_sobol_ranking: pd.DataFrame,
    num_grps: int,
    st_factor_ranking: pd.DataFrame,
    ivars_factor_ranking: pd.DataFrame,
    parameters: Dict[Union[str, int], Tuple[float, float]],
    bootstrap_size: int
) -> Tuple:
    """
    KK: DOCSTRING IS MISSING - PLEASE FILL & REMOVE THIS LINE THEREAFTER

    """
    # group the parameters
    num_grp_ivars50, ivars50_grp_array, clusters_ivars50 = factor_grouping(result_bs_ivars_df.loc[0.5],
                                                                           num_grp=num_grps)
    num_grp_sobol, sobol_grp_array, clusters_sobol = factor_grouping(result_bs_sobol,
                                                                     num_grp=num_grps)

    # calculate reliability estimates based on factor grouping
    cluster_sobol = []
    cluster_rank_sobol = []
    # associate group numbers with the parameters
    for g in range(0, num_grp_sobol):
        cluster_sobol.append(np.argwhere(sobol_grp_array == g + 1).flatten())
        cluster_rank_sobol.append(
            st_factor_ranking.to_numpy().flatten()[cluster_sobol[g]])
        cluster_rank_sobol[g] = np.sort(cluster_rank_sobol[g], axis=0)

    cluster_ivars50 = []
    cluster_rank_ivars50 = []
    for g in range(0, num_grp_ivars50):
        cluster_ivars50.append(np.argwhere(
            ivars50_grp_array == g + 1).flatten())
        cluster_rank_ivars50.append(
            ivars_factor_ranking.loc[0.5].to_numpy()[cluster_ivars50[g]])
        cluster_rank_ivars50[g] = np.sort(cluster_rank_ivars50[g], axis=0)

    # calculate the reliability estimates based on the factor groupings and their corresponding paramaters
    reli_sobol_grp_array = np.zeros(len(parameters.keys()))
    reli_ivars50_grp_array = np.zeros(len(parameters.keys()))
    for D in range(0, len(parameters.keys())):
        match = [np.argwhere(cluster_sobol[x] == D).flatten()
                 for x in range(0, len(cluster_sobol))]
        rank_range_sobol = [(match[x].size != 0) for x in range(0, len(match))]
        rank_sobol_benchmark = list(
            compress(cluster_rank_sobol, rank_range_sobol))
        rank_sobol_benchmark = rank_sobol_benchmark[0]

        match = [np.argwhere(cluster_ivars50[x] == D).flatten()
                 for x in range(0, len(cluster_ivars50))]
        rank_range_ivars50 = [(match[x].size != 0)
                              for x in range(0, len(match))]
        rank_ivars50_benchmark = list(
            compress(cluster_rank_ivars50, rank_range_ivars50))
        rank_ivars50_benchmark = rank_ivars50_benchmark[0]

        # calculate the reliability of parameter number D
        reli_sobol = 0
        reli_ivars50 = 0
        for i in range(0, bootstrap_size):
            reli_sobol += len(
                np.argwhere(result_bs_sobol_ranking.iloc[i, D] == rank_sobol_benchmark)) / bootstrap_size
            reli_ivars50 += len(np.argwhere(
                result_bs_ivars_ranking.loc[0.5].iloc[i, D] == rank_ivars50_benchmark)) / bootstrap_size

        reli_sobol_grp_array[D] = reli_sobol
        reli_ivars50_grp_array[D] = reli_ivars50

    reli_sobol_grp = pd.DataFrame(
        [reli_sobol_grp_array], columns=parameters.keys(), index=[''])
    reli_ivars50_grp = pd.DataFrame(
        [reli_ivars50_grp_array], columns=parameters.keys(), index=[0.5])

    # change numbering of groups to be consistent with matlab results
    for i in range(0, len(ivars50_grp_array)):
        ivars50_grp_array[i] = np.abs(ivars50_grp_array[i] - num_grps) + 1

    for i in range(0, len(sobol_grp_array)):
        sobol_grp_array[i] = np.abs(sobol_grp_array[i] - num_grps) + 1

    ivars50_grp = pd.DataFrame(
        [ivars50_grp_array], columns=parameters.keys(), index=[0.5])
    sobol_grp = pd.DataFrame(
        [sobol_grp_array], columns=parameters.keys(), index=[''])

    return ivars50_grp, sobol_grp, reli_sobol_grp, reli_ivars50_grp


def bootstrapping(
    pair_df: pd.DataFrame,
    df: pd.DataFrame,
    cov_section_all: pd.DataFrame,
    bootstrap_size: int,
    bootstrap_ci: float,
    func: Callable,
    delta_h: float,
    ivars_scales: Tuple[float, ...],
    parameters: Dict[Union[str, int], Tuple[float, float]],
    st_factor_ranking: pd.DataFrame,
    ivars_factor_ranking: pd.DataFrame,
    grouping_flag: bool,
    num_grps: int,
    progress: bool=False
) -> Tuple:
    """
    KK: DOCSTRING MISSING - FILL & REMOVE THIS LINE THEREAFTER 
    """
    # create result dataframes if bootstrapping is chosen to be done
    result_bs_variogram = pd.DataFrame()
    result_bs_sobol = pd.DataFrame()
    result_bs_ivars_df = pd.DataFrame()
    result_bs_sobol_ranking = pd.DataFrame()
    result_bs_ivars_ranking = pd.DataFrame()

    for i in tqdm(range(0, bootstrap_size), desc='bootstrapping', disable=not progress, dynamic_ncols=True):
        # bootstrapping to get CIs
        # specify random sequence by sampling with replacement
        bootstrap_rand = np.random.choice(
            list(range(0, 10)), size=len(range(0, 10)), replace=True).tolist()
        bootstrapped_pairdf = pd.concat(
            [pair_df.loc[pd.IndexSlice[i, :, :, :], :] for i in bootstrap_rand])
        bootstrapped_df = pd.concat(
            [df.loc[pd.IndexSlice[i, :, :], :] for i in bootstrap_rand])

        # calculating sectional covariograms
        bootstrapped_cov_section_all = pd.concat(
            [cov_section_all.loc[pd.IndexSlice[i, :]] for i in bootstrap_rand])

        # calculating variogram, ecovariogram, variance, mean, Sobol, and IVARS values
        bootstrapped_variogram = variogram(bootstrapped_pairdf)

        bootstrapped_ecovariogram = e_covariogram(bootstrapped_cov_section_all)

        bootstrapped_var = bootstrapped_df[func.__name__].unique().var(ddof=1)

        bootstrapped_sobol = sobol_eq(bootstrapped_variogram, bootstrapped_ecovariogram,
                                      bootstrapped_var, delta_h)

        bootstrapped_ivars_df = pd.DataFrame.from_dict(
            {scale: bootstrapped_variogram.groupby(level=0).apply(ivars, scale=scale,
                                                                  delta_h=delta_h)
             for scale in ivars_scales}, 'index')

        # calculating factor rankings for sobol and ivars
        bootstrapped_sobol_ranking = factor_ranking(bootstrapped_sobol)
        bootstrapped_sobol_ranking_df = pd.DataFrame(data=[bootstrapped_sobol_ranking],
                                                     columns=parameters.keys())

        # do factor ranking on IVARS results
        bootstrapped_ivars_factor_ranking_list = []
        for scale in ivars_scales:
            bootstrapped_ivars_factor_ranking_list.append(
                factor_ranking(bootstrapped_ivars_df.loc[scale]))
        # turn results into data frame
        bootstrapped_ivars_ranking_df = pd.DataFrame(data=bootstrapped_ivars_factor_ranking_list, columns=parameters.keys(),
                                                     index=ivars_scales)

        # unstack variogram so that results concat nicely
        bootstrapped_variogram_df = bootstrapped_variogram.unstack(level=0)

        # swap sobol results rows and columns so that results concat nicely
        bootstrapped_sobol_df = bootstrapped_sobol.to_frame().transpose()

        # attatch new results to previous results (order does not matter here)
        result_bs_variogram = pd.concat(
            [bootstrapped_variogram_df, result_bs_variogram])
        result_bs_sobol = pd.concat([bootstrapped_sobol_df, result_bs_sobol])
        result_bs_ivars_df = pd.concat(
            [bootstrapped_ivars_df, result_bs_ivars_df])
        result_bs_sobol_ranking = pd.concat(
            [bootstrapped_sobol_ranking_df, result_bs_sobol_ranking])
        result_bs_ivars_ranking = pd.concat(
            [bootstrapped_ivars_ranking_df, result_bs_ivars_ranking])

    # calculate upper and lower confidence interval limits for variogram results
    gammalb = pd.DataFrame()
    gammaub = pd.DataFrame()
    # iterate through each h value
    for h in np.unique(result_bs_variogram.index.values).tolist():
        # find all confidence interval limits for each h value
        gammalb = pd.concat(
            [gammalb,
             result_bs_variogram.loc[h].quantile((1 - bootstrap_ci) / 2).rename(h).to_frame()], axis=1)
        gammaub = pd.concat(
            [gammaub,
             result_bs_variogram.loc[h].quantile(1 - ((1 - bootstrap_ci) / 2)).rename(h).to_frame()],
            axis=1)

    # index value name is h?? not sure if this should be changed later
    gammalb.index.names = ['h']
    gammaub.index.names = ['h']

    # transpose to get into correct format
    gammalb = gammalb.transpose()
    gammaub = gammaub.transpose()

    # calculate upper and lower confidence interval limits for sobol results in a nice looking format
    stlb = result_bs_sobol.quantile(
        (1 - bootstrap_ci) / 2).rename('').to_frame().transpose()
    stub = result_bs_sobol.quantile(
        1 - ((1 - bootstrap_ci) / 2)).rename('').to_frame().transpose()

    # calculate upper and lower confidence interval limits of the ivars values
    ivarslb = pd.DataFrame()
    ivarsub = pd.DataFrame()
    # iterate through each IVARS scale
    for scale in ivars_scales:
        # find confidence interval limits for each scale
        ivarslb = pd.concat(
            [ivarslb,
             result_bs_ivars_df.loc[scale].quantile((1 - bootstrap_ci) / 2).rename(scale).to_frame()], axis=1)
        ivarsub = pd.concat(
            [ivarsub,
             result_bs_ivars_df.loc[scale].quantile(1 - ((1 - bootstrap_ci) / 2)).rename(scale).to_frame()],
            axis=1)

    # transpose the results to get them in the right format
    ivarslb = ivarslb.transpose()
    ivarsub = ivarsub.transpose()

    # calculate reliability estimates based on factor ranking of sobol result
    # calculate reliability estimates based on factor ranking of sobol result
    rel_sobol_results = []
    for param in parameters.keys():
        rel_sobol_results.append(
            result_bs_sobol_ranking[param].eq(st_factor_ranking[param][0]).sum() / bootstrap_size)

    rel_sobol_factor_ranking = pd.DataFrame(
        [rel_sobol_results], columns=parameters.keys(), index=[''])

    # calculate reliability estimates based on factor ranking of ivars results
    rel_ivars_results = []
    # iterate through each paramter
    for param in parameters.keys():
        rel_ivars_results_scale = []
        # iterate through each ivars scale
        for scale in ivars_scales:
            # ... to find the reliability estimate of the ivars rankings at each ivars scale
            rel_ivars_results_scale.append(
                result_bs_ivars_ranking.eq(ivars_factor_ranking)[param].loc[scale].sum() / bootstrap_size)
        rel_ivars_results.append(rel_ivars_results_scale)

    rel_ivars_factor_ranking = pd.DataFrame(rel_ivars_results, columns=ivars_scales,
                                            index=parameters.keys())
    # transpose to get data frame in correct format
    rel_ivars_factor_ranking = rel_ivars_factor_ranking.transpose()

    # grouping can only be done if bootstrapping has been done and 0.5 ivars was chosen as a scale
    if grouping_flag and (0.5 in result_bs_ivars_df.index):
        ivars50_grp, sobol_grp, reli_sobol_grp, reli_ivars50_grp = \
            grouping(result_bs_ivars_df, result_bs_sobol, result_bs_ivars_ranking, result_bs_sobol_ranking,
                     num_grps, st_factor_ranking, ivars_factor_ranking, parameters, bootstrap_size)

        return gammalb, gammaub, stlb, stub, ivarslb, ivarsub, rel_sobol_factor_ranking,\
            rel_ivars_factor_ranking, ivars50_grp, sobol_grp, reli_sobol_grp, reli_ivars50_grp
    # if grouping is not chosen to be done return only bootstrapping results
    else:
        return gammalb, gammaub, stlb, stub, ivarslb, ivarsub, rel_sobol_factor_ranking, \
            rel_ivars_factor_ranking
