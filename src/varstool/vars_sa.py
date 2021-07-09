import warnings
import decimal
import multiprocessing

import pandas as pd
import numpy  as np
import scipy.stats as stat
import scipy.cluster.hierarchy as hchy
#import numpy.typing as npt # let's further investigate this
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from itertools import compress


from .sampling import starvars
from .sa import vars_funcs
from .sa import tsvars_funcs

from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
    Callable,
    Any,
    List,
)

from collections.abc import (
    Iterable,
)


class Model():
    __doc__ = """A wrapper class for a function of interest"""

    def __init__(
        self,
        func: Callable = None,
        unknown_options: Dict[str, Any] = {},
    ) -> None:

        # check whether the input is a callable
        assert callable(func)
        self.func = func

        # unkown_options must be a dict
        assert isinstance(unknown_options, dict)
        self.unknown_options = {}

        if unknown_options:
            self.unknown_options = unknown_options

    def __repr__(self, ) -> str:
        """official representation"""
        return "wrapped function: "+self.func.__name__

    def __str__(self, ) -> str:
        """the name of the wrapper function"""
        return self.func.__name__

    def __call__(
        self,
        params,
        options: Dict = None,
    ) -> Union[Iterable, float, int]:

        # check if params is an array-like object
        assert isinstance(params,
            (pd.DataFrame, pd.Series, np.ndarray, List, Tuple))

        if options:
            self.unknown_options = options

        return self.func(params, **self.unknown_options)


class VARS(object):
    __doc__ = """VARS object"""

    #-------------------------------------------
    # Constructors

    def __init__(
        self,
        star_centres = [],  # sampled star centres (random numbers) used to create star points
        num_stars: int = 100, # default number of stars
        parameters: Dict[Union[str, int], Tuple[float, float]] = {}, # name and bounds
        delta_h: Optional[float] = 0.1, # delta_h for star sampling
        ivars_scales: Optional[Tuple[float, ...]] = (0.1, 0.3, 0.5), # ivars scales
        model: Model = None, # model (function) to run for each star point
        seed: Optional[int] = 123456789, # randomization state
        sampler: Optional[str] = None, # one of the default random samplers of varstool
        bootstrap_flag: Optional[bool] = False, # bootstrapping flag
        bootstrap_size: Optional[int]  = 1000, # bootstrapping size
        bootstrap_ci: Optional[int] = 0.9, # bootstrap confidence interval
        grouping_flag: Optional[bool] = False, # grouping flag
        num_grps: Optional[int] = None, # number of groups
        report_verbose: Optional[bool] = False, # reporting - using tqdm??
    ) -> None:

        # initialize values
        self.parameters = parameters
        self.num_stars = num_stars
        self.delta_h = delta_h
        self.ivars_scales = ivars_scales
        self.star_centres = star_centres
        self.star_points  = [] # an empty list works for now - but needs to be changed - really?
        self.seed = seed # seed number
        self.bootstrap_flag = bootstrap_flag
        self.bootstrap_size = bootstrap_size
        self.bootstrap_ci = bootstrap_ci
        self.grouping_flag = grouping_flag
        self.num_grps = num_grps
        self.report_verbose = report_verbose
        self.sampler = sampler # one of the default sampling algorithms or None
        # analysis stage is set to False before running anything
        self.run_status = False

        # Check input arguments
        # ***add error checking, and possibly default value for star centres

        # default value for bootstrap_flag
        if not bootstrap_flag:
            self.bootstrap_flag = False
        if not isinstance(bootstrap_flag, bool):
            warnings.warn(
                "`bootstrap_flag` must be either `True` or `False`."
                "default value of `False` will be considered.",
                UserWarning,
                stacklevel=1
            )

        # default value for grouping flag
        if not grouping_flag:
            self.grouping_flag = False
        if not isinstance(grouping_flag, bool):
            warnings.warn(
                "`grouping_flag` must be either `True` or `False`."
                "default value of `False` will be considered.",
                UserWarning,
                stacklevel=1
            )

        ## default value for the IVARS scales are 0.1, 0.3, and 0.5
        if not self.ivars_scales:
            warnings.warn(
                "IVARS scales are not valid, default values of (0.1, 0.3, 0.5) "
                "will be considered.",
                UserWarning,
                stacklevel=1
            )
            self.ivars_scales = (0.1, 0.3, 0.5)

        ## if there is any value in IVARS scales that is greater than 0.5
        if any(i for i in self.ivars_scales if i > 0.5):
            warnings.warn(
                "IVARS scales greater than 0.5 are not recommended.",
                UserWarning,
                stacklevel=1
            )

        ## if delta_h is not a factor of 1, NaNs or ZeroDivison might happen
        if (decimal.Decimal(str(1)) % decimal.Decimal(str(self.delta_h))) != 0:
            warnings.warn(
                "If delta_h is not a factor of 1, NaNs and ZeroDivisionError are probable. "
                "It is recommended to change `delta_h` to a divisible number by 1.",
                RuntimeWarning,
                stacklevel=1
            )

        ## if delta_h is not between 0 and 1
        if ((delta_h <= 0) or (delta_h >=1 )):
            raise ValueError(
                "`delta_h` must be greater than 0 and less than 1."
            )

        ## check seed dtype and sign
        if ((not isinstance(seed, int)) or (seed < 0)):
            warnings.warn(
                "`seed` must be an integer greater than zero."
                " value is set to default, i.e., 123456789"
            )
            self.seed = 123456789

        ## check bootstrap dtype and sign
        if ((not isinstance(bootstrap_size, int)) or (bootstrap_size < 1)):
            raise ValueError(
                "`bootstrap_size` must be an integer greater than one."
                " value is set to default, i.e., 1000"
            )

        ## check bootstrap ci dtype, value and sign
        if ((not isinstance(bootstrap_ci, float)) or \
            (bootstrap_ci <= 0) or \
            (bootstrap_ci >= 1)):
            warnings.warn(
                "bootstrap condifence interval (CI) must be a float"
                " within (0, 1) interval. Setting `boostrap_ci` value"
                " to default, i.e., 0.9"
            )
            self.bootstrap_ci = 0.9

        ## check the dtypes and instances
        ### `parameters`
        if not isinstance(self.parameters, dict):
            raise ValueError(
                "`parameters` must be of type `dict`; the keys must be"
                "their names, either strings or integers, and values must"
                "be the lower and upper bounds of their factor space."
            )

        if 0 < len(self.parameters) < 2:
            raise ValueError(
                "the number of parameters in a sensitivity analysis problem"
                "must be greater than 1"
            )

        ### `model`
        if model:
            if not isinstance(model, Model):
                raise TypeError(
                    "`model` must be of type varstool.Model"
                )
            self.model = model

        # check the sampling algorithms
        if self.sampler == 'rnd':
            self.star_centres = np.random.rand(self.num_stars, len(self.parameters))
        elif self.sampler == 'lhs':
            from .sampling import lhs
            self.star_centres = lhs(sp=self.num_stars,
                                      params=len(self.parameters),
                                      seed=self.seed,
                                )
        elif self.sampler == 'plhs':
            from .sampling import plhs
            self.star_centres = plhs(sp=self.num_stars,
                                       params=len(self.parameters),
                                       seed=self.seed,
                                )
        elif self.sampler == 'sobol_seq':
            from .sampling import sobol_sequence
            self.star_centres = sobol_sequence(sp=self.num_stars,
                                                 params=len(self.parameters),
                                                 seed=self.seed,
                                )
        elif self.sampler == 'halton_seq':
            from .sampling import halton
            self.star_centres = halton(sp=self.num_stars,
                                         params=len(self.parameters),
                                         seed=self.seed,
                                )
        elif self.sampler == 'symlhs':
            from .sampling import symlhs
            self.star_centres = symlhs(sp=self.num_stars,
                                         params=len(self.parameters),
                                         seed=self.seed,
                                )
        elif self.sampler == None:
            pass
        else:
            raise ValueError(
                "`sampler` must be either None, or one of the following:"
                "'rnd', 'lhs', 'plhs', 'halton_seq', 'sobol_seq', 'symlhs'"
            )

    @classmethod
    def from_dict(cls, input_dict):

        return cls()

    @classmethod
    def from_text(cls, input_text_file):

        return cls()


    #-------------------------------------------
    # Representators
    def __repr__(self, ) -> str:
        """show the status of VARS analysis"""

        status_star_centres = "Star Centres: " + ("Loaded" if len(self.star_centres) != 0 else "Not Loaded")
        status_star_points = "Star Points: " + ("Loaded" if len(self.star_points) != 0 else "Not Loaded")
        status_parameters = "Parameters: " + (str(len(self.parameters))+" paremeters set" if self.parameters else "None")
        status_delta_h = "Delta h: " + (str(self.delta_h)+"" if self.delta_h else "None")
        status_model = "Model: " + (str(self.model)+"" if self.model else "None")
        status_seed = "Seed Number: " + (str(self.seed)+"" if self.seed else "None")
        status_bstrap = "Bootstrap: " + ("On" if self.bootstrap_flag else "Off")
        status_bstrap_size = "Bootstrap Size: " + (str(self.bootstrap_size)+"" if self.bootstrap_flag else "N/A")
        status_bstrap_ci = "Bootstrap CI: " + (str(self.bootstrap_ci)+"" if self.bootstrap_flag else "N/A")
        status_grouping = "Grouping: " + ("On" if self.grouping_flag else "Off")
        status_num_grps = "Number of Groups: " + (str(self.num_grps)+"" if self.num_grps else "None")
        status_analysis = "VARS Analysis: " + ("Done" if self.run_status else "Not Done")

        status_report_list = [status_star_centres, status_star_points, status_parameters, \
                              status_delta_h, status_model, status_seed, status_bstrap, \
                              status_bstrap_size, status_bstrap_ci, status_grouping, status_num_grps, status_analysis]

        return "\n".join(status_report_list)


    def __str__(self, ) -> str:

        return self.__class__.__name__


    #-------------------------------------------
    # Core properties

    ## using dunder variables for avoiding confusion with
    ## D-/GVARS sublcasses.

    @property
    def star_centres(self, ):
        return self._star_centres

    @star_centres.setter
    def star_centres(self, new_centres):
        if not isinstance(new_centres,
              (pd.DataFrame, pd.Series, np.ndarray, List, Tuple)):
            raise TypeError(
                "new_centres must be an array-like object: "
                "pandas.Dataframe, pandas.Series, numpy.array, List, Tuple"
            )
        self._star_centres = new_centres

    @property
    def star_points(self, ):
        return self._star_points

    @star_points.setter
    def star_points(self, new_points):
        if not isinstance(new_points,
              (pd.DataFrame, pd.Series, np.ndarray, List, Tuple)):
            raise TypeError(
                "new_points must be an array-like object: "
                "pandas.Dataframe, pandas.Series, numpy.array, List, Tuple"
            )
        self._star_points = new_points


    #-------------------------------------------
    # Core functions
    @staticmethod
    def generate_star(star_centres, delta_h, param_names):

        # generate star points using star.py functions
        star_points = starvars.star(star_centres, delta_h=delta_h, parameters=param_names, rettype='DataFrame')

        # figure out way to return this?
        return star_points  # for now will just do this


    def _plot(self, ):

        # make similar plots as the matlab plots showing the important
        # SA results from analysis

        pass


    def run_online(self, ):

        # generate star points
        self.star_points = starvars.star(self.star_centres, # star centres
                                           delta_h=self.delta_h, # delta_h
                                           parameters=[*self.parameters], # parameters dictionary keys
                                           rettype='DataFrame',
                                       ) # return type is a dataframe

        self.star_points = vars_funcs.scale(df=self.star_points, # star points must be scaled
                                             bounds={ # bounds are created while scaling
                                             'lb':[val[0] for _, val in self.parameters.items()],
                                             'ub':[val[1] for _, val in self.parameters.items()],
                                             }
                                        )

        # apply model to the generated star points
        df = vars_funcs.apply_unique(self.model.func, self.star_points)
        df.index.names = ['centre', 'param', 'points']

        # get paired values for each section based on 'h'
        pair_df = df[str(self.model)].groupby(level=[0,1]).apply(vars_funcs.section_df, self.delta_h)
        pair_df.index.names = ['centre', 'param', 'h', 'pair_ind']

        # get mu_star value
        mu_star_df = df[str(self.model)].groupby(level=[0,1]).mean()
        mu_star_df.index.names = ['centre', 'param']

        # overall mean of the unique evaluated function value over all star points
        mu_overall = df[str(self.model)].unique().mean()

        # overall variance of the unique evaluated function over all star points
        var_overall = df[str(self.model)].unique().var(ddof=1)

        # sectional covariogram calculation
        cov_section_all = vars_funcs.cov_section(pair_df, mu_star_df)

        # variogram calculation
        self.variogram_value = vars_funcs.variogram(pair_df)

        # morris calculation
        self.morris_value = vars_funcs.morris_eq(pair_df)

        # overall covariogram calculation
        self.covariogram_value = vars_funcs.covariogram(pair_df, mu_overall)

        # expected value of the overall covariogram calculation
        self.e_covariogram_value = vars_funcs.e_covariogram(cov_section_all)

        # sobol calculation
        self.sobol_value = vars_funcs.sobol_eq(self.variogram_value, self.e_covariogram_value, var_overall, self.delta_h)

        # do factor ranking on sobol results
        sobol_factor_ranking_array = self._factor_ranking(self.sobol_value)
        # turn results into data frame
        self.sobol_factor_ranking = pd.DataFrame(data=[sobol_factor_ranking_array], columns=self.parameters.keys())

        # IVARS calculation
        self.ivars = pd.DataFrame.from_dict({scale: self.variogram_value.groupby(level=0).apply(vars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
             for scale in self.ivars_scales}, 'index')

        # do factor ranking on IVARS results
        ivars_factor_ranking_list = []
        for scale in self.ivars_scales:
            ivars_factor_ranking_list.append(self._factor_ranking(self.ivars.loc[scale]))
        # turn results into data frame
        self.ivars_factor_ranking = pd.DataFrame(data=ivars_factor_ranking_list, columns=self.parameters.keys(), index=self.ivars_scales)

        if self.bootstrap_flag and self.grouping_flag:
            self.variogram_low, self.variogram_upp, self.sobol_low, self.sobol_upp, self.ivars_low, self.ivars_upp, \
            self.rel_sobol_factor_ranking,self.rel_ivars_factor_ranking, self.ivars50_grp, self.sobol_grp, \
            self.reli_sobol_grp, self.reli_ivars50_grp = self._bootstrapping(pair_df, df, cov_section_all)
        else:
            self.variogram_low, self.variogram_upp, self.sobol_low, self.sobol_upp, self.ivars_low, self.ivars_upp, \
            self.rel_sobol_factor_ranking,self.rel_ivars_factor_ranking = self._bootstrapping(pair_df, df, cov_section_all)


        self.run_status = True


    def run_offline(star_points,):

        # do analysis on offline formatted star points

        # figure out a way to return results

        return


    def _factor_ranking(self, factors):
        """ Ranks factors based on their influence (how large or small results are)
            The lowest rank corresponds to the most influential (larger) factor

            parameters:
            factors: an array like input that contains factors that are to be ranked

            returns:
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

    def _factor_grouping(self, sens_idx, num_grp=None):
        [m, n] = sens_idx.shape

        # make data 1d
        R = sens_idx.stack()
        # replacing zeros with a constant number due to numerical reasoning
        R[R == 0] = np.ones(len(R[R == 0]))

        # do a box-cox transformation
        [TRANSDAT, LAMBDA] = stat.boxcox(R)
        if LAMBDA <= 0.0099:
            TRANSDAT = np.log(R)

        indices = np.argwhere(np.isinf(TRANSDAT).tolist())
        if indices.shape == (2, 1):
            TRANSDAT[indices[0], indices[1]] = np.log(R[R > 0])

        # reshape data for the linkage calculation
        S = np.reshape(TRANSDAT.tolist(), [n, m])

        # Agglomerative hierarchical cluster
        Z = hchy.linkage(S, method='ward', metric='euclidean')

        # Optimal group number
        Clusters = []
        for i in range(2, n + 1):
            Clusters.append(hchy.fcluster(Z, criterion='maxclust', t=i))
        # if user gives the group number preform calculations
        if num_grp:
            rank_grp = hchy.fcluster(Z, criterion='maxclust', t=num_grp)
            optm_num_grp = num_grp
            nn = 1
            id = len(Z)
            while nn != optm_num_grp:
                cutoff = Z[id - 1][2]
                rank_grp = hchy.fcluster(Z, criterion='distance', t=cutoff)
                nn = np.amax(rank_grp)
                id = id - 1

            clrThrshl = 0.5 * (Z[id][2] + Z[id + 1][2])
        # if user does not give optimal group number use elbow method
        else:
            cutoff, clrThrshl = self._elbow_method(Z)
            rank_grp = hchy.fcluster(Z, criterion='distance', t=cutoff)
            optm_num_grp = max(rank_grp)

        # *** this part can be edited once we start working on plots
        # fig = plt.figure(figsize=(25,10))
        # dn = hchy.dendrogram(Z)
        # plt.show()

        return optm_num_grp, rank_grp, Clusters


    def _elbow_method(self, Z):
        # creating Q1 and Q2 for elbow method calculations
        Q1 = np.array([1, Z[0][2]])
        Q2 = np.array([len(Z), Z[-1][2]])

        # Use elbow method to find the cutoff and color threshold for clustering
        d = []
        for i in range(0, len(Z) - 2):
            P = [i + 1, Z[i][2]]
            d.append(np.abs(np.linalg.det(np.array([[Q2 - Q1], [P - Q1]]))) / np.linalg.norm(Q2 - Q1))

        id = d.index(max(d))
        cutoff = Z[id][2]
        clrThrshl = 0.5 * (Z[id][2] + Z[id + 1][2])

        return cutoff, clrThrshl

    def _grouping(self, result_bs_ivars_df, result_bs_sobol, result_bs_ivars_ranking, result_bs_sobol_ranking):
        # group the parameters
        num_grp_ivars50, ivars50_grp_array, ClustersIvars50 = self._factor_grouping(result_bs_ivars_df.loc[0.5],
                                                                                    num_grp=self.num_grps)
        num_grp_sobol, sobol_grp_array, ClustersSobol = self._factor_grouping(result_bs_sobol,
                                                                              num_grp=self.num_grps)

        ivars50_grp = pd.DataFrame([ivars50_grp_array], columns=self.parameters.keys())
        sobol_grp = pd.DataFrame([sobol_grp_array], columns=self.parameters.keys())

        # calculate reliability estimates based on factor grouping
        cluster_sobol = []
        cluster_rank_sobol = []
        # associate group numbers with the parameters
        for g in range(0, num_grp_sobol):
            cluster_sobol.append(np.argwhere(sobol_grp_array == g + 1).flatten())
            cluster_rank_sobol.append(self.sobol_factor_ranking.to_numpy().flatten()[cluster_sobol[g]])
            cluster_rank_sobol[g] = np.sort(cluster_rank_sobol[g], axis=0)

        cluster_ivars50 = []
        cluster_rank_ivars50 = []
        for g in range(0, num_grp_ivars50):
            cluster_ivars50.append(np.argwhere(ivars50_grp_array == g + 1).flatten())
            cluster_rank_ivars50.append(self.ivars_factor_ranking.loc[0.5].to_numpy()[cluster_ivars50[g]])
            cluster_rank_ivars50[g] = np.sort(cluster_rank_ivars50[g], axis=0)

        # calculate the reliability estimates based on the factor groupings and their corresponding paramaters
        reli_sobol_grp_array = np.zeros(len(self.parameters.keys()))
        reli_ivars50_grp_array = np.zeros(len(self.parameters.keys()))
        for D in range(0, len(self.parameters.keys())):
            match = [np.argwhere(cluster_sobol[x] == D).flatten() for x in range(0, len(cluster_sobol))]
            rank_range_sobol = [(match[x].size != 0) for x in range(0, len(match))]
            rank_sobol_benchmark = list(compress(cluster_rank_sobol, rank_range_sobol))
            rank_sobol_benchmark = rank_sobol_benchmark[0]

            match = [np.argwhere(cluster_ivars50[x] == D).flatten() for x in range(0, len(cluster_ivars50))]
            rank_range_ivars50 = [(match[x].size != 0) for x in range(0, len(match))]
            rank_ivars50_benchmark = list(compress(cluster_rank_ivars50, rank_range_ivars50))
            rank_ivars50_benchmark = rank_ivars50_benchmark[0]

            # calculate the reliability of parameter number D
            reli_sobol = 0
            reli_ivars50 = 0
            for i in range(0, self.bootstrap_size):
                reli_sobol += len(
                    np.argwhere(result_bs_sobol_ranking.iloc[i, D] == rank_sobol_benchmark)) / self.bootstrap_size
                reli_ivars50 += len(np.argwhere(
                    result_bs_ivars_ranking.loc[0.5].iloc[i, D] == rank_ivars50_benchmark)) / self.bootstrap_size

            reli_sobol_grp_array[D] = reli_sobol
            reli_ivars50_grp_array[D] = reli_ivars50

        reli_sobol_grp = pd.DataFrame([reli_sobol_grp_array], columns=self.parameters.keys())
        reli_ivars50_grp = pd.DataFrame([reli_ivars50_grp_array], columns=self.parameters.keys())

        return ivars50_grp, sobol_grp, reli_sobol_grp, reli_ivars50_grp


    def _bootstrapping(self, pair_df, df, cov_section_all):
        # create result dataframes if bootstrapping is chosen to be done
        result_bs_variogram = pd.DataFrame()
        result_bs_sobol = pd.DataFrame()
        result_bs_ivars_df = pd.DataFrame()
        result_bs_sobol_ranking = pd.DataFrame()
        result_bs_ivars_ranking = pd.DataFrame()

        for i in range(0, self.bootstrap_size):
            # bootstrapping to get CIs
            # specify random sequence by sampling with replacement
            bootstrap_rand = np.random.choice(list(range(0, 10)), size=len(range(0, 10)), replace=True).tolist()
            bootstrapped_pairdf = pd.concat([pair_df.loc[pd.IndexSlice[i, :, :, :], :] for i in bootstrap_rand])
            bootstrapped_df = pd.concat([df.loc[pd.IndexSlice[i, :, :], :] for i in bootstrap_rand])

            # calculating sectional covariograms
            bootstrapped_cov_section_all = pd.concat([cov_section_all.loc[pd.IndexSlice[i, :]] for i in bootstrap_rand])

            # calculating variogram, ecovariogram, variance, mean, Sobol, and IVARS values
            bootstrapped_variogram = vars_funcs.variogram(bootstrapped_pairdf)

            bootstrapped_ecovariogram = vars_funcs.e_covariogram(bootstrapped_cov_section_all)

            bootstrapped_var = bootstrapped_df[self.model.func.__name__].unique().var(ddof=1)

            bootstrapped_sobol = vars_funcs.sobol_eq(bootstrapped_variogram, bootstrapped_ecovariogram,
                                                     bootstrapped_var, self.delta_h)

            bootstrapped_ivars_df = pd.DataFrame.from_dict(
                {scale: bootstrapped_variogram.groupby(level=0).apply(vars_funcs.ivars, scale=scale,
                                                                      delta_h=self.delta_h) \
                 for scale in self.ivars_scales}, 'index')

            # calculating factor rankings for sobol and ivars
            bootstrapped_sobol_ranking = self._factor_ranking(bootstrapped_sobol)
            bootstrapped_sobol_ranking_df = pd.DataFrame(data=[bootstrapped_sobol_ranking],
                                                         columns=self.parameters.keys())

            # do factor ranking on IVARS results
            bootstrapped_ivars_factor_ranking_list = []
            for scale in self.ivars_scales:
                bootstrapped_ivars_factor_ranking_list.append(self._factor_ranking(bootstrapped_ivars_df.loc[scale]))
            # turn results into data frame
            bootstrapped_ivars_ranking_df = pd.DataFrame(data=bootstrapped_ivars_factor_ranking_list, columns=self.parameters.keys(),
                                                     index=self.ivars_scales)

            # unstack variogram so that results concat nicely
            bootstrapped_variogram_df = bootstrapped_variogram.unstack(level=0)

            # swap sobol results rows and columns so that results concat nicely
            bootstrapped_sobol_df = bootstrapped_sobol.to_frame().transpose()

            # attatch new results to previous results (order does not matter here)
            result_bs_variogram = pd.concat([bootstrapped_variogram_df, result_bs_variogram])
            result_bs_sobol = pd.concat([bootstrapped_sobol_df, result_bs_sobol])
            result_bs_ivars_df = pd.concat([bootstrapped_ivars_df, result_bs_ivars_df])
            result_bs_sobol_ranking = pd.concat([bootstrapped_sobol_ranking_df, result_bs_sobol_ranking])
            result_bs_ivars_ranking = pd.concat([bootstrapped_ivars_ranking_df, result_bs_ivars_ranking])

        # calculate upper and lower confidence interval limits for variogram results
        variogram_low = pd.DataFrame()
        variogram_upp = pd.DataFrame()
        # iterate through each h value
        for h in np.unique(result_bs_variogram.index.values).tolist():
            # find all confidence interval limits for each h value
            variogram_low = pd.concat(
                [variogram_low,
                 result_bs_variogram.loc[h].quantile((1 - self.bootstrap_ci) / 2).rename(h).to_frame()], axis=1)
            variogram_upp = pd.concat(
                [variogram_upp,
                 result_bs_variogram.loc[h].quantile(1 - ((1 - self.bootstrap_ci) / 2)).rename(h).to_frame()],
                axis=1)

        # index value name is h?? not sure if this should be changed later
        variogram_low.index.names = ['h']
        variogram_upp.index.names = ['h']

        # transpose to get into correct format
        variogram_low = variogram_low.transpose()
        variogram_upp = variogram_upp.transpose()

        # calculate upper and lower confidence interval limits for sobol results in a nice looking format
        sobol_low = result_bs_sobol.quantile((1 - self.bootstrap_ci) / 2).rename('').to_frame().transpose()
        sobol_upp = result_bs_sobol.quantile(1 - ((1 - self.bootstrap_ci) / 2)).rename('').to_frame().transpose()

        # calculate upper and lower confidence interval limits of the ivars values
        ivars_low = pd.DataFrame()
        ivars_upp = pd.DataFrame()
        # iterate through each IVARS scale
        for scale in self.ivars_scales:
            # find confidence interval limits for each scale
            ivars_low = pd.concat(
                [ivars_low,
                 result_bs_ivars_df.loc[scale].quantile((1 - self.bootstrap_ci) / 2).rename(scale).to_frame()], axis=1)
            ivars_upp = pd.concat(
                [ivars_upp,
                 result_bs_ivars_df.loc[scale].quantile(1 - ((1 - self.bootstrap_ci) / 2)).rename(scale).to_frame()],
                axis=1)

        # transpose the results to get them in the right format
        ivars_low = ivars_low.transpose()
        ivars_upp = ivars_upp.transpose()

        # calculate reliability estimates based on factor ranking of sobol result
        rel_sobol_results = []
        for param in self.parameters.keys():
            rel_sobol_results.append(
                result_bs_sobol_ranking.eq(self.sobol_factor_ranking)[param].sum() / self.bootstrap_size)

        rel_sobol_factor_ranking = pd.DataFrame([rel_sobol_results], columns=self.parameters.keys())

        # calculate reliability estimates based on factor ranking of ivars results
        rel_ivars_results = []
        # iterate through each paramter
        for param in self.parameters.keys():
            rel_ivars_results_scale = []
            # iterate through each ivars scale
            for scale in self.ivars_scales:
                # ... to find the reliability estimate of the ivars rankings at each ivars scale
                rel_ivars_results_scale.append(
                    result_bs_ivars_ranking.eq(self.ivars_factor_ranking)[param].loc[scale].sum() / self.bootstrap_size)
            rel_ivars_results.append(rel_ivars_results_scale)

        rel_ivars_factor_ranking = pd.DataFrame(rel_ivars_results, columns=self.ivars_scales,
                                                     index=self.parameters.keys())
        # transpose to get data frame in correct format
        rel_ivars_factor_ranking = rel_ivars_factor_ranking.transpose()

        # grouping can only be done if bootstrapping has been done and 0.5 ivars was chosen as a scale
        if self.grouping_flag and (0.5 in result_bs_ivars_df.index):
            ivars50_grp, sobol_grp, reli_sobol_grp, reli_ivars50_grp = \
                self._grouping(result_bs_ivars_df, result_bs_sobol, result_bs_ivars_ranking, result_bs_sobol_ranking)

            return variogram_low, variogram_upp, sobol_low, sobol_upp, ivars_low, ivars_upp, rel_sobol_factor_ranking,\
                rel_ivars_factor_ranking, ivars50_grp, sobol_grp, reli_sobol_grp, reli_ivars50_grp
        # if grouping is not chosen to be done return only bootstrapping results
        else:
            return variogram_low, variogram_upp, sobol_low, sobol_upp, ivars_low, ivars_upp, rel_sobol_factor_ranking, \
               rel_ivars_factor_ranking


class GVARS(VARS):
    __doc__ = """GVARS object"""

    #-------------------------------------------
    # Constructors

    def __init__(self,
                 param_dist_types,  # distribution types of the model parameters
                 corr_mat,  # correlation matrix
                 num_direct_samples: int = 50, # number of directional samples
                 num_stars: int = 2000, # number of star samples
                 ):

        # initialize values
        super().__init__() # initialize all values from VARS super method
        self.num_direct_samples = num_direct_samples
        self.num_stars = num_stars
        self.param_dist_types = param_dist_types
        self.corr_mat = corr_mat

        ## default value for the number of directional samples
        if not self.num_direct_samples:
            warnings.warn(
                "Number of directional samples are not valid, default value of 50 "
                "will be considered.",
                UserWarning,
                stacklevel=1
            )
        self.num_direct_samples = 50


        ## default value for the number of star samples
        if not self.num_stars:
            warnings.warn(
                "Number of star samples are not valid, default value of 2000 "
                "will be considered.",
                UserWarning,
                stacklevel=1
            )
        self.num_stars = 2000

        # number of parameters in users model
        self.num_factors = len(self.parameters)


#-------------------------------------------
    # Representators
    def __repr__(self, ) -> str:

        pass


    def _repr_html(self, ):

        pass


    def __str__(self, ) -> str:

        pass

    #-------------------------------------------
    # Core properties

    #-------------------------------------------
    # Core functions

    def run(self):

        n_var = self.corr_mat.shape[1]
        cov_mat = np.array([[1, 0.6], [0.6, 1]])  # make funcion for this for now just using matlab results

        # Generate independent standard normal samples
        # the amount of samples is the same as the amount of stars
        U = np.random.multivariate_normal(np.zeros(n_var), np.eye(n_var), self.num_stars)

        # Generate correlated standard normal samples
        # the amount of samples is the same as the amount of stars
        cholU = np.linalg.cholesky(cov_mat)
        cholU = cholU.transpose()  # to get in correct format for matrix multiplication
        Z = np.matmul(U, cholU)  # transform samples to standard normal distribution

        # Generate Nstar actual multivariate samples X (not done yet)
        X = 0

        # define index matrix of complement subset
        compsub = np.empty([n_var, n_var - 1])
        for i in range(0, n_var):
            temp = np.arange(n_var)
            compsub[i] = np.delete(temp, i)
        compsub = compsub.astype(int)

        # computer coditional variance and conditional expectation for each star center
        chol_cond_std = []
        std_cond_norm = []
        mui_on_noti = np.zeros((len(Z), n_var))
        for i in range(0, n_var):
            noti = compsub[i]
            # 2 dimensional or greater matrix case
            if (cov_mat[noti, noti].ndim >= 2):
                cond_std = cov_mat[i][i] - cov_mat[i, noti] * np.linalg.inv(cov_mat[noti, noti]) * cov_mat[noti, i]
                chol_cond_std.append(np.linalg.cholesky(cond_std))
                std_cond_norm.append(cond_std)
                for j in range(0, len(Z)):
                    mui_on_noti[j][i] = cov_mat[i, noti] * np.linalg.inv(cov_mat[noti, noti]) * Z[j, noti]
            # less then 2 dimenional matrix case
            else:
                cond_std = cov_mat[i][i] - cov_mat[i, noti] * cov_mat[noti, noti] * cov_mat[noti, i]
                chol_cond_std.append(np.linalg.cholesky([[cond_std]]).flatten())
                std_cond_norm.append(cond_std)
                for j in range(0, len(Z)):
                    mui_on_noti[j][i] = cov_mat[i, noti] * cov_mat[noti, noti] * Z[j, noti]

        # Generate directional sample:
        # Create samples in correlated standard normal space
        all_section_condZ = []
        condZ = []
        for j in range(0, self.num_direct_samples):
            stnrm_base = np.random.multivariate_normal(np.zeros(n_var), np.eye(n_var), self.num_stars)
            for i in range(0, n_var):
                condZ.append(stnrm_base[:, i] * chol_cond_std[i] + mui_on_noti[:, i])
            all_section_condZ.append(condZ.copy())
            condZ.clear()

        # define the Xmax,Xmin along directional sample of all stars

        # collect directional samples (for distance in variogram)
        # sectionXi contains all section samples. each sell corresponds to
        # each input factor: row= dirStar, column=nStar

        # calculate variogram

        # calculate ivars

        # calculate sobol results

        # bootstrapping

        # calculate confidence intervals

        # collect IVARS 50 ??

        pass


    def __map_to_cor_norm(self, corr_mat, dist_types, factors):
        """
        ***(doc string will probably need to be cleaned up)

        This function is based on Kucherenko et al. 2012:
        Kucherenko S., Tarantola S. and Annoni p. 2012 "Estimation of
        global sensitivity indices for models with dependent variables" Computer
        Physics Communications, doi:10.1016/j.cpc.2011.12.020

        The code is modified from GSA_CORRELATED_MIXED_DISTRIBUTIONs code
        (S. Kucherenko,  A. Klimenko, S. Tarantola), obtained from SAMO2018

        1st update 20/10/2018
        2nd update 20/12/2018

        The code has been further modified to run in python on 18/06/2021:
        Contributors:
        Kasra Keshavarz
        Cordell Blanchard

        Parameters
        factors: user model parameters (add more detail)

        Returns
        A fictive correlation matrix based on inputted parameters
        """

        pass

    def __NtoX_transform(self, norm_vectors, dist_types, parameters):
        """Transform variables from standard normal to original distributions"""

        # can have the parameters in a data frame with the indices
        # being their distribution type and use pd.loc to gather
        # the parameters with a specific distribution type
        # then preform the transformation on them to avoid for loop

        # will have to inform use that only unif, norm, triangle, lognorm, expo, and gev
        # are the only ones that can be transformed

        pass


class TSVARS(VARS):
    __doc__ = "TSVARS Documentation"

    def __init__(
        self, #itself
        star_centres = [],  # sampled star centres (random numbers) used to create star points
        num_stars: int = 100, # default number of stars
        parameters: Dict[Union[str, int], Tuple[float, float]] = {}, # name and bounds
        delta_h: Optional[float] = 0.1, # delta_h for star sampling
        ivars_scales: Optional[Tuple[float, ...]] = (0.1, 0.3, 0.5), # ivars scales
        model: Model = None, # model (function) to run for each star point
        seed: Optional[int] = 123456789, # randomization state
        sampler: Optional[str] = None, # one of the default random samplers of varstool
        bootstrap_flag: Optional[bool] = False, # bootstrapping flag
        bootstrap_size: Optional[int]  = 1000, # bootstrapping size
        bootstrap_ci: Optional[int] = 0.9, # bootstrap confidence interval
        grouping_flag: Optional[bool] = False, # grouping flag
        num_grps: Optional[int] = None, # number of groups
        report_verbose: Optional[bool] = False, # reporting - using tqdm??
        func_eval_method: Optional[str] = 'serial', # the method to evaluate the model or function
        vars_eval_method: Optional[str] = 'serial', # the method to make pair_df dataframe
        vars_chunk_size: Optional[int] = None, # the chunk size to make pair_dfs to save memory
    ) -> None:

        super().__init__(
            star_centres,
            num_stars,
            parameters,
            delta_h,
            ivars_scales,
            model,
            seed,
            sampler,
            bootstrap_flag,
            bootstrap_size,
            bootstrap_ci,
            grouping_flag,
            num_grps,
            report_verbose,
        ) # main instance variables are just the same as VARS

        # defining the TSVARS specific instance variables
        self.func_eval_method = func_eval_method
        self.vars_eval_method = vars_eval_method
        self.vars_chunk_size = vars_chunk_size

        # writing some errors that might happen here
        if self.vars_eval_method not in ('serial', 'parallel'):
            raise ValueError(
                "`vars_eval_method` must be either 'parallel' or 'serial'"
            )

        if self.func_eval_method not in ('serial', 'parallel'):
            raise ValueError(
                "`func_eval_method` must be either 'parallel' or 'serial'"
            )


    #-------------------------------------------
    # Representators
    def __repr__(self) -> str:
        """method shows the status of VARS analysis"""

        status_star_centres = "Star Centres: " + ("Loaded" if len(self.star_centres) != 0 else "Not Loaded")
        status_star_points = "Star Points: " + ("Loaded" if len(self.star_points) != 0 else "Not Loaded")
        status_parameters = "Parameters: " + (str(len(self.parameters))+" paremeters set" if self.parameters else "None")
        status_delta_h = "Delta h: " + (str(self.delta_h)+"" if self.delta_h else "None")
        status_model = "Model: " + (str(self.model)+"" if self.model else "None")
        status_seed = "Seed Number: " + (str(self.seed)+"" if self.seed else "None")
        status_bstrap = "Bootstrap: " + ("On" if self.bootstrap_flag else "Off")
        status_bstrap_size = "Bootstrap Size: " + (str(self.bootstrap_size)+"" if self.bootstrap_flag else "N/A")
        status_bstrap_ci = "Bootstrap CI: " + (str(self.bootstrap_ci)+"" if self.bootstrap_flag else "N/A")
        status_grouping = "Grouping: " + ("On" if self.grouping_flag else "Off")
        status_num_grps = "Number of Groups: " + (str(self.num_grps)+"" if self.num_grps else "None")
        status_analysis = "TSVARS Analysis: " + ("Done" if self.run_status else "Not Done")
        status_func_eval_method = "Function Evaluation Method: " + self.func_eval_method
        status_vars_eval_method = "TSVARS Evaluation Method: " + self.vars_eval_method
        status_vars_chunk_size  = "TSVARS Chunk Size: " + (str(self.vars_chunk_size) if self.vars_chunk_size else "N/A")

        status_report_list = [
                              status_star_centres, status_star_points, status_parameters, \
                              status_delta_h, status_model, status_seed, status_bstrap, \
                              status_bstrap_size, status_bstrap_ci, status_grouping, \
                              status_num_grps, status_analysis, status_func_eval_method, \
                              status_vars_eval_method, status_vars_chunk_size
                              ]

        return "\n".join(status_report_list) # join lines together and show them all


    def __str__(self) -> str:

        return self.__class__.__name__


    #-------------------------------------------
    # Core functions

    def run_online(self, ):

        self.star_points = starvars.star(self.star_centres, # star centres
                                           delta_h=self.delta_h, # delta_h
                                           parameters=[*self.parameters], # parameters dictionary keys
                                           rettype='DataFrame',
                                       ) # return type must be a dataframe
        self.star_points.columns = [*self.parameters]

        self.star_points = tsvars_funcs.scale(df=self.star_points, # star points must be scaled
                                             bounds={ # bounds are created while scaling
                                             'lb':[val[0] for _, val in self.parameters.items()],
                                             'ub':[val[1] for _, val in self.parameters.items()],
                                             }
                                        )

        # doing function evaluations either on serial or parallel mode
        if self.func_eval_method == 'serial':
            df = self.star_points.apply(self.model, axis=1, result_type='expand')
            df.index.names = ['centre', 'param', 'point']

        elif self.func_eval_method == 'parallel':
            warnings.warn(
                "Evaluating function in parallel is not stable yet! "
                "varstool currently uses `mapply` to parallelize function "
                "evaluations, see https://github.com/ddelange/mapply",
                UserWarning,
                stacklevel=1
            )

            #import mapply inside this if clause to avoid unnecessary overhead
            import mapply
            import psutil

            mapply.init(
                n_workers=-1,
                chunk_size=1,
                max_chunks_per_worker=int(df.shape[0]//psutil.psutil.cpu_count(logical=False)),
                progressbar=True,
            )
            df = self.star_points.mapply(self.model.func, axis=1, result_type='expand')
            df.index.names = ['centre', 'param', 'point']

        # defining a lambda function to do the pairing for each time-step
        ts_pair = lambda ts: ts.groupby(level=['centre', 'param']).apply(tsvars_funcs.section_df, self.delta_h)

        # VARS evaluations can be done in two modes: serial and parallel
        ## if serial mode is chosen, two engines are available:
        ## 1. pandas dataframe `apply` method & 2. simple for loop over time-steps
        if self.vars_eval_method == 'serial':

            if self.vars_chunk_size: # if chunk size is provided by the user

                self.pair_df = pd.DataFrame()
                self.sec_covariogram = pd.DataFrame()
                self.variogram = pd.DataFrame()
                self.mu_star = pd.DataFrame()
                self.mu_overall = pd.DataFrame()
                self.var_overall = pd.DataFrame()
                self.sec_covariogram = pd.DataFrame()
                self.morris = pd.DataFrame()
                self.covariogram = pd.DataFrame()
                self.e_covariogram = pd.DataFrame()
                self.sobol = pd.DataFrame()
                self.ivars = pd.DataFrame()

                for chunk in range(int(df.shape[1]//self.vars_chunk_size)+1): # total number of chunks

                    # make a chunk of the main df (result of func eval)
                    df_temp = df.iloc[:, chunk*self.vars_chunk_size:min((chunk+1)*self.vars_chunk_size, df.shape[1]-1)]

                    # make pairs for each chunk
                    temp_pair_df = df_temp.groupby(level=0, axis=1).apply(ts_pair)
                    temp_pair_df.index.names = ['centre', 'param', 'h', 'pair_ind']
                    temp_pair_df.columns.names = ['ts', None]
                    temp_pair_df.stack(level='ts').reorder_levels(['ts','centre','param','h','pair_ind']).sort_index()
                    self.pair_df = pd.concat([self.pair_df, temp_pair_df])

                    # mu star
                    temp_mu_star = df_temp.groupby(level=['centre','param']).mean().stack().reorder_levels(order=[2,0,1]).sort_index()
                    temp_mu_star.index.names = ['ts', 'centre', 'param']
                    self.mu_star = pd.concat([self.mu_star, temp_mu_star])

                    # mu overall
                    temp_mu_overall = df_temp.apply(lambda x: np.mean(list(np.unique(x))))
                    self.mu_overall = pd.concat([self.mu_overall, temp_mu_overall])

                    #var overall
                    temp_var_overall = df_temp.apply(lambda x: np.var(list(np.unique(x)), ddof=1))
                    self.var_overall = pd.concat([self.var_overall, temp_var_overall])

                    #variogram
                    temp_variogram = tsvars_funcs.variogram(temp_pair_df)
                    self.variogram = pd.concat([self.variogram, temp_variogram])

                    #sectional variogram
                    temp_sec_covariogram = tsvars_funcs.cov_section(temp_pair_df, temp_mu_star)
                    self.sec_covariogram = pd.concat([self.sec_covariogram, temp_sec_covariogram])

                    #morris
                    temp_morris_values = tsvars_funcs.morris_eq(temp_pair_df)
                    self.morris = pd.concat([self.morris, temp_morris_values])

                    #covariogram
                    temp_covariogram = tsvars_funcs.covariogram(temp_pair_df, temp_mu_overall)
                    self.covariogram = pd.concat([self.covariogram, temp_covariogram])

                    #e_covariogram
                    temp_e_covariogram = tsvars_funcs.e_covariogram(temp_sec_covariogram)
                    self.e_covariogram = pd.concat([self.e_covariogram, temp_e_covariogram])

                    #sobol
                    temp_sobol_values = tsvars_funcs.sobol_eq(self.variogram_value, self.e_covariogram_value, self.var_overall, self.delta_h)
                    self.sobol = pd.concat([self.sobol, temp_sobol_values])

                    #ivars
                    temp_ivars_values = pd.DataFrame.from_dict({scale: temp_variogram_values.groupby(level=['ts', 'param']).apply(tsvars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
                      for scale in self.ivars_scales}, 'index')
                    self.ivars = pd.concat([self.ivars_values, temp_ivars_values])

                    self.run_status = True

            else:
                # pair_df is built serially - other functions are the same as parallel

                self.pair_df = df.groupby(level=0, axis=1).apply(ts_pair)
                self.pair_df.index.names = ['centre', 'param', 'h', 'pair_ind']
                self.pair_df.columns.names = ['ts', None]
                self.pair_df = self.pair_df.stack(level='ts').reorder_levels([-1,0,1,2,3]).sort_index()

                self.mu_star_df = df.groupby(level=['centre','param']).mean().stack().reorder_levels(order=[2,0,1]).sort_index()
                self.mu_star_df.index.names = ['ts', 'centre', 'param']

                self.mu_overall = df.apply(lambda x: np.mean(list(np.unique(x))))
                self.var_overall = df.apply(lambda x: np.var(list(np.unique(x)), ddof=1))
                self.variogram = tsvars_funcs.variogram(self.pair_df)
                self.sec_covariogram = tsvars_funcs.cov_section(self.pair_df, self.mu_star_df)
                self.morris = tsvars_funcs.morris_eq(self.pair_df)
                self.covariogram = tsvars_funcs.covariogram(self.pair_df, self.mu_overall)
                self.e_covariogram = tsvars_funcs.e_covariogram(self.sec_covariogram)
                self.sobol_value = tsvars_funcs.sobol_eq(self.variogram, self.e_covariogram, self.var_overall, self.delta_h)
                self.ivars = pd.DataFrame.from_dict({scale: self.variogram.groupby(level=['ts', 'param']).apply(tsvars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
                      for scale in self.ivars_scales}, 'index')

                self.run_status = True


        elif self.vars_eval_method == 'parallel':

            if self.vars_chunk_size: # if chunk size is provided by the user

                self.pair_df = pd.DataFrame()
                self.sec_covariogram = pd.DataFrame()
                self.variogram = pd.DataFrame()
                self.mu_star = pd.DataFrame()
                self.mu_overall = pd.DataFrame()
                self.var_overall = pd.DataFrame()
                self.sec_covariogram = pd.DataFrame()
                self.morris = pd.DataFrame()
                self.covariogram = pd.DataFrame()
                self.e_covariogram = pd.DataFrame()
                self.sobol = pd.DataFrame()
                self.ivars = pd.DataFrame()

                for chunk in range(int(df.shape[1]//self.vars_chunk_size)+1): # total number of chunks

                    # make a chunk of the main df (result of func eval)
                    df_temp = df.iloc[:, chunk*self.vars_chunk_size:min((chunk+1)*self.vars_chunk_size, df.shape[1]-1)]

                    # make pairs for each chunk
                    temp_pair_df = applyParallel(a.groupby(level=0, axis=1), ts_pair)
                    temp_pair_df.index.names = ['ts', 'centre', 'param', 'h', 'pair_ind']
                    self.pair_df = pd.concat([self.pair_df, temp_pair_df])

                    # mu star
                    temp_mu_star = df_temp.groupby(level=['centre','param']).mean().stack().reorder_levels(order=[2,0,1]).sort_index()
                    temp_mu_star.index.names = ['ts', 'centre', 'param']
                    self.mu_star = pd.concat([self.mu_star, temp_mu_star])

                    # mu overall
                    temp_mu_overall = df_temp.apply(lambda x: np.mean(list(np.unique(x))))
                    self.mu_overall = pd.concat([self.mu_overall, temp_mu_overall])

                    #var overall
                    temp_var_overall = df_temp.apply(lambda x: np.var(list(np.unique(x)), ddof=1))
                    self.var_overall = pd.concat([self.var_overall, temp_var_overall])

                    #variogram
                    temp_variogram = tsvars_funcs.variogram(temp_pair_df)
                    self.variogram = pd.concat([self.variogram, temp_variogram])

                    #sectional variogram
                    temp_sec_covariogram = tsvars_funcs.cov_section(temp_pair_df, temp_mu_star)
                    self.sec_covariogram = pd.concat([self.sec_covariogram, temp_sec_covariogram])

                    #morris
                    temp_morris_values = tsvars_funcs.morris_eq(temp_pair_df)
                    self.morris = pd.concat([self.morris, temp_morris_values])

                    #covariogram
                    temp_covariogram = tsvars_funcs.covariogram(temp_pair_df, temp_mu_overall)
                    self.covariogram = pd.concat([self.covariogram, temp_covariogram])

                    #e_covariogram
                    temp_e_covariogram = tsvars_funcs.e_covariogram(temp_sec_covariogram)
                    self.e_covariogram = pd.concat([self.e_covariogram, temp_e_covariogram])

                    #sobol
                    temp_sobol_values = tsvars_funcs.sobol_eq(self.variogram_value, self.e_covariogram_value, self.var_overall, self.delta_h)
                    self.sobol = pd.concat([self.sobol, temp_sobol_values])

                    #ivars
                    temp_ivars_values = pd.DataFrame.from_dict({scale: temp_variogram_values.groupby(level=['ts', 'param']).apply(tsvars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
                      for scale in self.ivars_scales}, 'index')
                    self.ivars = pd.concat([self.ivars_values, temp_ivars_values])

                    self.run_status = True

            else:
                # pair_df is built serially - other functions are the same as parallel
                self.pair_df = applyParallel(a.groupby(level=0, axis=1), ts_pair)
                self.pair_df.index.names = ['ts', 'centre', 'param', 'h', 'pair_ind']

                self.mu_star_df = df.groupby(level=['centre','param']).mean().stack().reorder_levels(order=[2,0,1]).sort_index()
                self.mu_star_df.index.names = ['ts', 'centre', 'param']

                self.mu_overall = df.apply(lambda x: np.mean(list(np.unique(x))))
                self.var_overall = df.apply(lambda x: np.var(list(np.unique(x)), ddof=1))
                self.variogram = tsvars_funcs.variogram(pair_df)
                self.sec_covariogram = tsvars_funcs.cov_section(pair_df, mu_star_df)
                self.morris = tsvars_funcs.morris_eq(pair_df)
                self.covariogram = tsvars_funcs.covariogram(pair_df, mu_overall)
                self.sobol_value = tsvars_funcs.sobol_eq(variogram_value, e_covariogram_value, var_overall)
                self.ivars = pd.DataFrame.from_dict({scale: self.variogram.groupby(level=['ts', 'param']).apply(tsvars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
                      for scale in self.ivars_scales}, 'index')

                self.run_status = True


    def temp_func(func, name, group):
        return func(group), name

    def applyParallel(dfGrouped, func):
        retLst, top_index = zip(*Parallel(n_jobs=multiprocessing.cpu_count())\
                                    (delayed(temp_func)(func, name, group)\
                                for name, group in dfGrouped))
        return pd.concat(retLst, keys=top_index)


class DVARS(VARS):
    def __init__(self, ):
        super().__init__()
