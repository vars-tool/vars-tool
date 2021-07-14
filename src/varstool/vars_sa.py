"""
VARS Sensitivity Anlaysis Framework
-----------------------------------

The Variogram Analysis of Response Surfaces (VARS) is a 
powerful sensitivity analysis (SA) method first applied
to  Earth and Environmental System models. 

"""


import warnings
import decimal
import multiprocessing

import pandas as pd
import numpy  as np
import scipy.stats as stat
import scipy.cluster.hierarchy as hchy
import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange


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


class Model(object):
    """
    Description:
    ------------
    A wrapper class to contain various models and functions
    to be fed into VARS and its variations. The models can be
    called by simply calling the wrapper class itself.


    Parameters:
    -----------
    :param func: function of interest 
    :type func: Callable
    :param unknown_options: a dictionary of options with keys as
                            parameters and values of parameter
                            values.
    :type unknown_options: dict
    """
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
    """
    Description:
    ------------
    The Python implementation of the Variogram Analysis of Response
    Surfaces (VARS) first introduced in Razavi and Gupta (2015) (see
    [1]_ and [2]_).


    Parameters:
    -----------
    :param star_centres: contains star centres of the analysis
    :type star_centres: numpy.array
    :param num_stars: number of stars to generate
    :type num_stars: int, numpy.int32, numpy.int64, defaults to 100
    :param parameters: the parameters of the model including lower and
                       upper bounds
    :type parameters: dict
    :param delta_h: the resolution of star samples
    :type delta_h: float, defaults to 0.1
    :param ivars_scales: the IVARS scales
    :type ivars_scales: tuple, defaults to (0.1, 0.3, 0.5)
    :param model: the model used in the sensitivity analysis
    :type model: varstool.Model
    :param seed: the seed number used in generating star centres
    :type seed: int, numpy.int32, numpy.int64
    :param bootstrap_flag: flag to bootstrap the sensitivity analysis results
    :type bootstrap_flag: bool, defaults to False
    :param bootstrap_size: the size of bootstrapping experiment
    :type bootstrap_size: int, defaults to 1000
    :param bootstrap_ci: the condifence interval of boostrapping
    :type bootstrap_ci: float, defaults to 0.9
    :param grouping_flag: flag to conduct grouping of sensitive parameters
    :type grouping_flag: bool, defaults to False
    :param num_grps: the number of groups to categorize parameters
    :type num_grps: int, defaults to None
    :param report_verbose: flag to show the sensitvity analysis progress
    :type report_verbose: bool, False


    References:
    -----------
    .. [1] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive,
           robust, and efficient global sensitivity analysis: 1. Theory. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017558
    .. [2] Razavi, S., & Gupta, H. V. (2016). A new framework for comprehensive,
           robust, and efficient global sensitivity analysis: 1. Application. Water 
           Resources Research, 52(1), 423-439. doi: 10.1002/2015WR017559


    Contributors:
    -------------
    Razavi, Saman, (2015): algorithm, code in MATALB (c)
    Gupta, Hoshin, (2015): algorithm, code in MATLAB (c)
    Mattot, Shawn, (2019): code in C/++
    Keshavarz, Kasra, (2021): code in Python 3
    Blanchard, Cordell, (2021): code in Python 3
    """

    #-------------------------------------------
    # Constructors

    def __init__(
        self,
        star_centres: np.ndarray = np.array([]),  # sampled star centres (random numbers) used to create star points
        num_stars: Optional[int] = 100, #  number of stars
        parameters: Dict[Union[str, int], Tuple[float, float]] = {}, # name and bounds
        delta_h: Optional[float] = 0.1, # delta_h for star sampling
        ivars_scales: Optional[Tuple[float, ...]] = (0.1, 0.3, 0.5), # ivars scales
        model: Model = None, # model (function) to run for each star point
        seed: Optional[int] = np.random.randint(1, 123456789), # randomization state
        sampler: Optional[str] = None, # one of the default random samplers of varstool
        bootstrap_flag: Optional[bool] = False, # bootstrapping flag
        bootstrap_size: Optional[int]  = 1000, # bootstrapping size
        bootstrap_ci: Optional[float] = 0.9, # bootstrap confidence interval
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
        self.star_points  = pd.DataFrame([]) # an empty list works for now - but needs to be changed - really?
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
                " value is set to default, i.e., randomized integer between 1 and 123456789"
            )
            self.seed = np.random.randint(1, 123456790)

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
            np.random.seed(self.seed)
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


    #-------------------------------------------
    # Representators
    def __repr__(self) -> str:
        """shows the status of VARS analysis"""

        status_star_centres = "Star Centres: " + (str(self.star_centres.shape[0])+ " Centers Loaded" if len(self.star_centres) != 0 else "Not Loaded")
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
        status_verbose  = "Verbose: " + ("On" if self.report_verbose else "Off")
        status_analysis = "VARS Analysis: " + ("Done" if self.run_status else "Not Done")

        status_report_list = [status_star_centres, status_star_points, status_parameters, \
                              status_delta_h, status_model, status_seed, status_bstrap, \
                              status_bstrap_size, status_bstrap_ci, status_grouping, \
                              status_num_grps, status_verbose, status_analysis]

        return "\n".join(status_report_list)

    def __str__(self) -> str:
        """shows the instance name of the VARS analysis experiment"""

        return self.__class__.__name__


    #-------------------------------------------
    # Core properties

    ## using dunder variables for avoiding confusion with
    ## D-/GVARS sublcasses.

    @property
    def star_centres(self):
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
    def star_points(self):
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
    def generate_star(self) -> pd.DataFrame:

        # generate star points
        star_points = starvars.star(self.star_centres, # star centres
                                           delta_h=self.delta_h, # delta_h
                                           parameters=[*self.parameters], # parameters dictionary keys
                                           rettype='DataFrame',
                                       ) # return type is a dataframe

        star_points = vars_funcs.scale(df=star_points, # star points must be scaled
                                             bounds={ # bounds are created while scaling
                                             'lb':[val[0] for _, val in self.parameters.items()],
                                             'ub':[val[1] for _, val in self.parameters.items()],
                                             }
                                        )

        star_points.index.names = ['centre', 'param', 'points']

        return star_points

    def _plot(self):

        # make similar plots as the matlab plots showing the important
        # SA results from analysis

        pass

    def run_online(self):

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
        df = vars_funcs.apply_unique(func=self.model.func, 
                                     df=self.star_points,
                                     axis=1,
                                     progress=self.report_verbose,
                            )
        df.index.names = ['centre', 'param', 'points']

        # get paired values for each section based on 'h' - considering the progress bar if report_verbose is True
        if self.report_verbose:
            tqdm.pandas(desc='building pairs')
            self.pair_df = df[str(self.model)].groupby(level=[0,1]).progress_apply(vars_funcs.section_df, 
                                                                                   delta_h=self.delta_h)
        else:
            self.pair_df = df[str(self.model)].groupby(level=[0,1]).apply(vars_funcs.section_df, 
                                                                          delta_h=self.delta_h)
        self.pair_df.index.names = ['centre', 'param', 'h', 'pair_ind']

        # progress bar for vars analysis
        if self.report_verbose:
            vars_pbar = tqdm(desc='VARS analysis', total=10) # 10 steps for different components

        # get mu_star value
        self.mu_star_df = df[str(self.model)].groupby(level=[0,1]).mean()
        self.mu_star_df.index.names = ['centre', 'param']
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Averages of function evaluations (`mu_star`) calculated - access via .mu_star_df')

        # overall mean of the unique evaluated function value over all star points
        self.mu_overall = df[str(self.model)].unique().mean()
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Overall expected value of function evaluations (`mu_overall`) calculated - access via .mu_overall')

        # overall variance of the unique evaluated function over all star points
        self.var_overall = df[str(self.model)].unique().var(ddof=1)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Overall variance of function evaluations (`var_overall`) calculated - access via .var_overall')

        # sectional covariogram calculation
        self.cov_section_all = vars_funcs.cov_section(self.pair_df, self.mu_star_df)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Sectional covariogram `cov_section_all` calculated - access via .cov_section_all')

        # variogram calculation
        # MATLAB: Gamma
        self.gamma = vars_funcs.variogram(self.pair_df)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Variogram (`gamma`) calculated - access via .gamma')

        # morris calculation
        morris_value = vars_funcs.morris_eq(self.pair_df)
        self.maee = morris_value[0] # MATLAB: MAEE
        self.mee  = morris_value[1] # MATLAB: MEE
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Morris MAEE and MEE (`maee` and `mee`) calculated - access via .maee and .mee')

        # overall covariogram calculation
        # MATLAB: COV
        self.cov = vars_funcs.covariogram(self.pair_df, self.mu_overall)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Covariogram (`cov`) calculated - access via .cov')

        # expected value of the overall covariogram calculation
        # MATLAB: ECOV
        self.ecov = vars_funcs.e_covariogram(self.cov_section_all)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Expected value of covariogram (`ecov`) calculated - access via .ecov')

        # sobol calculation
        # MATLAB: ST
        self.st = vars_funcs.sobol_eq(self.gamma, self.ecov, self.var_overall, self.delta_h)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Sobol ST (`st`) calculated - access via .st')

        # IVARS calculation
        self.ivars = pd.DataFrame.from_dict({scale: self.gamma.groupby(level=0).apply(vars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
             for scale in self.ivars_scales}, 'index')
        if self.report_verbose:
            vars_pbar.update(1)
            vars_pbar.close()

        # progress bar for factor ranking
        if self.report_verbose:
            factor_rank_pbar = tqdm(desc='factor ranking', total=2) # only two components

        # do factor ranking on sobol results
        sobol_factor_ranking_array = vars_funcs.factor_ranking(self.st)
        # turn results into data frame
        self.st_factor_ranking = pd.DataFrame(data=[sobol_factor_ranking_array], columns=self.parameters.keys(), index=[''])
        if self.report_verbose:
            factor_rank_pbar.update(1)

        # do factor ranking on IVARS results
        ivars_factor_ranking_list = []
        for scale in self.ivars_scales:
            ivars_factor_ranking_list.append(vars_funcs.factor_ranking(self.ivars.loc[scale]))
        # turn results into data frame
        self.ivars_factor_ranking = pd.DataFrame(data=ivars_factor_ranking_list, columns=self.parameters.keys(), index=self.ivars_scales)
        if self.report_verbose:
            factor_rank_pbar.update(1)
            factor_rank_pbar.close()

        if self.bootstrap_flag and self.grouping_flag:
            self.gammalb, self.gammaub, self.stlb, self.stub, self.ivarslb, self.ivarsub, \
            self.rel_st_factor_ranking, self.rel_ivars_factor_ranking, self.ivars50_grp, self.st_grp, \
            self.reli_st_grp, self.reli_ivars50_grp = vars_funcs.bootstrapping(self.pair_df, df, self.cov_section_all,
                                                                               self.bootstrap_size, self.bootstrap_ci,
                                                                               self.model.func, self.delta_h, self.ivars_scales,
                                                                               self.parameters, self.st_factor_ranking,
                                                                               self.ivars_factor_ranking, self.grouping_flag,
                                                                               self.num_grps, self.report_verbose)
        else:
            self.gammalb, self.gammaub, self.stlb, self.stub, self.ivarslb, self.ivarsub, \
            self.rel_st_factor_ranking, self.rel_ivars_factor_ranking = vars_funcs.bootstrapping(self.pair_df, df, self.cov_section_all,
                                                                               self.bootstrap_size, self.bootstrap_ci,
                                                                               self.model.func, self.delta_h, self.ivars_scales,
                                                                               self.parameters, self.st_factor_ranking,
                                                                               self.ivars_factor_ranking, self.grouping_flag,
                                                                               self.num_grps, self.report_verbose)

        # for status update
        self.run_status = True

        # output dictionary
        self.output = {
            'Gamma':self.gamma,
            'MAEE':self.maee,
            'MEE':self.mee,
            'COV':self.cov,
            'ECOV':self.ecov,
            'IVARS':self.ivars,
            'IVARSid':self.ivars_scales,
            'rnkST':self.st_factor_ranking,
            'rnkIVARS':self.ivars_factor_ranking,
            'Gammalb':self.gammalb if self.bootstrap_flag is True else None,
            'Gammaub':self.gammaub if self.bootstrap_flag is True else None,
            'STlb':self.stlb if self.bootstrap_flag is True else None,
            'STub':self.stub if self.bootstrap_flag is True else None,
            'IVARSlb':self.ivarslb if self.bootstrap_flag is True else None,
            'IVARSub':self.ivarsub if self.bootstrap_flag is True else None,
            'relST':self.rel_st_factor_ranking if self.bootstrap_flag is True else None,
            'relIVARS':self.rel_ivars_factor_ranking if self.bootstrap_flag is True else None,
            'Groups': [self.ivars50_grp, self.st_grp] if self.grouping_flag is True else None,
            'relGrp': [self.reli_st_grp, self.reli_ivars50_grp] if self.grouping_flag is True else None,
        }

        return


    def run_offline(self, df):

        # get paired values for each section based on 'h' - considering the progress bar if report_verbose is True
        if self.report_verbose:
            tqdm.pandas(desc='building pairs')
            self.pair_df = df[str(self.model)].groupby(level=[0,1]).progress_apply(vars_funcs.section_df,
                                                                                   delta_h=self.delta_h)
        else:
            self.pair_df = df[str(self.model)].groupby(level=[0,1]).apply(vars_funcs.section_df,
                                                                          delta_h=self.delta_h)
        self.pair_df.index.names = ['centre', 'param', 'h', 'pair_ind']

        # progress bar for vars analysis
        if self.report_verbose:
            vars_pbar = tqdm(desc='VARS Analysis', total=10) # 10 steps for different components

        # get mu_star value
        self.mu_star_df = df[str(self.model)].groupby(level=[0,1]).mean()
        self.mu_star_df.index.names = ['centre', 'param']
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Averages of function evaluations (`mu_star`) calculated - access via .mu_star_df')

        # overall mean of the unique evaluated function value over all star points
        self.mu_overall = df[str(self.model)].unique().mean()
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Overall expected value of function evaluations (`mu_overall`) calculated - access via .mu_overall')

        # overall variance of the unique evaluated function over all star points
        self.var_overall = df[str(self.model)].unique().var(ddof=1)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Overall variance of function evaluations (`var_overall`) calculated - access via .var_overall')

        # sectional covariogram calculation
        self.cov_section_all = vars_funcs.cov_section(self.pair_df, self.mu_star_df)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Sectional covariogram `cov_section_all` calculated - access via .cov_section_all')

        # variogram calculation
        # MATLAB: Gamma
        self.gamma = vars_funcs.variogram(self.pair_df)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Variogram (`gamma`) calculated - access via .gamma')

        # morris calculation
        morris_value = vars_funcs.morris_eq(self.pair_df)
        self.maee = morris_value[0] # MATLAB: MAEE
        self.mee  = morris_value[1] # MATLAB: MEE
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Morris MAEE and MEE (`maee` and `mee`) calculated - access via .maee and .mee')

        # overall covariogram calculation
        # MATLAB: COV
        self.cov = vars_funcs.covariogram(self.pair_df, self.mu_overall)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Covariogram (`cov`) calculated - access via .cov')

        # expected value of the overall covariogram calculation
        # MATLAB: ECOV
        self.ecov = vars_funcs.e_covariogram(self.cov_section_all)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Expected value of covariogram (`ecov`) calculated - access via .ecov')

        # sobol calculation
        # MATLAB: ST
        self.st = vars_funcs.sobol_eq(self.gamma, self.ecov, self.var_overall, self.delta_h)
        if self.report_verbose:
            vars_pbar.update(1)
            # vars_pbar.write('Sobol ST (`st`) calculated - access via .st')

        # IVARS calculation
        self.ivars = pd.DataFrame.from_dict({scale: self.gamma.groupby(level=0).apply(vars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
             for scale in self.ivars_scales}, 'index')
        if self.report_verbose:
            vars_pbar.update(1)
            vars_pbar.close()

        # progress bar for factor ranking
        if self.report_verbose:
            factor_rank_pbar = tqdm(desc='factor ranking', total=2)

        # do factor ranking on sobol results
        sobol_factor_ranking_array = vars_funcs.factor_ranking(self.st)
        # turn results into data frame
        self.st_factor_ranking = pd.DataFrame(data=[sobol_factor_ranking_array], columns=self.parameters.keys(), index=[''])
        if self.report_verbose:
            factor_rank_pbar.update(1)

        # do factor ranking on IVARS results
        ivars_factor_ranking_list = []
        for scale in self.ivars_scales:
            ivars_factor_ranking_list.append(vars_funcs.factor_ranking(self.ivars.loc[scale]))
        # turn results into data frame
        self.ivars_factor_ranking = pd.DataFrame(data=ivars_factor_ranking_list, columns=self.parameters.keys(), index=self.ivars_scales)
        if self.report_verbose:
            factor_rank_pbar.update(1)
            factor_rank_pbar.close()

        if self.bootstrap_flag and self.grouping_flag:
            self.gammalb, self.gammaub, self.stlb, self.stub, self.ivarslb, self.ivarsub, \
            self.rel_st_factor_ranking, self.rel_ivars_factor_ranking, self.ivars50_grp, self.st_grp, \
            self.reli_st_grp, self.reli_ivars50_grp = vars_funcs.bootstrapping(self.pair_df, df, self.cov_section_all,
                                                                               self.bootstrap_size, self.bootstrap_ci,
                                                                               self.model.func, self.delta_h, self.ivars_scales,
                                                                               self.parameters, self.st_factor_ranking,
                                                                               self.ivars_factor_ranking, self.grouping_flag,
                                                                               self.num_grps, self.report_verbose)
        else:
            self.gammalb, self.gammaub, self.stlb, self.stub, self.ivarslb, self.ivarsub, \
            self.rel_st_factor_ranking, self.rel_ivars_factor_ranking = vars_funcs.bootstrapping(self.pair_df, df, self.cov_section_all,
                                                                               self.bootstrap_size, self.bootstrap_ci,
                                                                               self.model.func, self.delta_h, self.ivars_scales,
                                                                               self.parameters, self.st_factor_ranking,
                                                                               self.ivars_factor_ranking, self.grouping_flag,
                                                                               self.num_grps, self.report_verbose)

        # for status update
        self.run_status = True

        # output dictionary
        self.output = {
            'Gamma':self.gamma,
            'MAEE':self.maee,
            'MEE':self.mee,
            'COV':self.cov,
            'ECOV':self.ecov,
            'IVARS':self.ivars,
            'IVARSid':self.ivars_scales,
            'rnkST':self.st_factor_ranking,
            'rnkIVARS':self.ivars_factor_ranking,
            'Gammalb':self.gammalb if self.bootstrap_flag is True else None,
            'Gammaub':self.gammaub if self.bootstrap_flag is True else None,
            'STlb':self.stlb if self.bootstrap_flag is True else None,
            'STub':self.stub if self.bootstrap_flag is True else None,
            'IVARSlb':self.ivarslb if self.bootstrap_flag is True else None,
            'IVARSub':self.ivarsub if self.bootstrap_flag is True else None,
            'relST':self.rel_st_factor_ranking if self.bootstrap_flag is True else None,
            'relIVARS':self.rel_ivars_factor_ranking if self.bootstrap_flag is True else None,
            'Groups': [self.ivars50_grp, self.st_grp] if self.grouping_flag is True else None,
            'relGrp': [self.reli_st_grp, self.reli_ivars50_grp] if self.grouping_flag is True else None,
        }

        return


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
        star_centres = np.array([]),  # sampled star centres (random numbers) used to create star points
        num_stars: int = 100, # default number of stars
        parameters: Dict[Union[str, int], Tuple[float, float]] = {}, # name and bounds
        delta_h: Optional[float] = 0.1, # delta_h for star sampling
        ivars_scales: Optional[Tuple[float, ...]] = (0.1, 0.3, 0.5), # ivars scales
        model: Model = None, # model (function) to run for each star point
        seed: Optional[int] = np.random.randint(1, 123456789), # randomization state
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
        status_func_eval_method = "Function Evaluation Method: " + self.func_eval_method
        status_vars_eval_method = "TSVARS Evaluation Method: " + self.vars_eval_method
        status_vars_chunk_size  = "TSVARS Chunk Size: " + (str(self.vars_chunk_size) if self.vars_chunk_size else "N/A")
        
        status_verbose  = "Verbose: " + ("On" if self.report_verbose else "Off")
        status_analysis = "TSVARS Analysis: " + ("Done" if self.run_status else "Not Done")

        status_report_list = [
                                status_star_centres, status_star_points, status_parameters, \
                                status_delta_h, status_model, status_seed, status_bstrap, \
                                status_bstrap_size, status_bstrap_ci, status_grouping, \
                                status_num_grps, status_func_eval_method, status_vars_eval_method, \
                                status_vars_chunk_size, status_verbose, status_analysis
                            ]

        return "\n".join(status_report_list) # join lines together and show them all


    def __str__(self) -> str:

        return self.__class__.__name__


    #-------------------------------------------
    # Core functions

    def run_online(self):

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
            if self.report_verbose:
                tqdm.pandas(desc='function evaluation')
                self.star_points_eval = self.star_points.progress_apply(self.model, axis=1, result_type='expand')
            else:
                self.star_points_eval = self.star_points.apply(self.model, axis=1, result_type='expand')
            self.star_points_eval.index.names = ['centre', 'param', 'point']

        elif self.func_eval_method == 'parallel':
            warnings.warn(
                "Evaluating function in parallel mode is not stable yet. "
                "varstool currently uses `mapply` to parallelize function "
                "evaluations, see https://github.com/ddelange/mapply",
                UserWarning,
                stacklevel=1
            )

            #importing `mapply` inside this if clause to avoid unnecessary overhead
            import mapply
            import psutil

            mapply.init(
                n_workers=-1, # -1 indicates max_chunks_per_worker makes the decision on parallelization
                chunk_size=1, # 1 indicates max_chunks_per_worker makes the decision on parallelization
                max_chunks_per_worker=int(self.star_points.shape[0]//psutil.cpu_count(logical=False)),
                progressbar=True if self.report_verbose else False,
            )
            self.star_points_eval = self.star_points.mapply(self.model, axis=1, result_type='expand')
            self.star_points_eval.index.names = ['centre', 'param', 'point']

        # defining a lambda function to build pairs for each time-step
        ts_pair = lambda ts: ts.groupby(level=['centre', 'param']).apply(tsvars_funcs.section_df, self.delta_h)

        # VARS evaluations can be done in two modes: serial and parallel
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
                    temp_sobol_values = tsvars_funcs.sobol_eq(self.gamma, self.ecov, self.var_overall, self.delta_h)
                    self.sobol = pd.concat([self.sobol, temp_sobol_values])

                    #ivars
                    temp_ivars_values = pd.DataFrame.from_dict({scale: temp_variogram_values.groupby(level=['ts', 'param']).apply(tsvars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
                      for scale in self.ivars_scales}, 'index')
                    self.ivars = pd.concat([self.ivars_values, temp_ivars_values])

                    self.run_status = True

            else:
                # pair_df is built serially - other functions are the same as parallel
                if self.report_verbose: # making a progress bar
                    tqdm.pandas(desc='building pairs')
                    self.pair_df = self.star_points_eval.groupby(level=0, axis=1).progress_apply(ts_pair)
                else:
                    self.pair_df = self.star_points_eval.groupby(level=0, axis=1).apply(ts_pair)
                self.pair_df.index.names = ['centre', 'param', 'h', 'pair_ind']
                self.pair_df.columns.names = ['ts', None]
                self.pair_df = self.pair_df.stack(level='ts').reorder_levels([-1,0,1,2,3]).sort_index()

                vars_pbar = tqdm(desc='TSVARS analysis', total=10)
                self.mu_star_df = self.star_points_eval.groupby(level=['centre','param']).mean().stack().reorder_levels(order=[2,0,1]).sort_index()
                self.mu_star_df.index.names = ['ts', 'centre', 'param']
                if self.report_verbose:
                    vars_pbar.update(1)

                self.mu_overall = self.star_points_eval.apply(lambda x: np.mean(list(np.unique(x))))
                if self.report_verbose:
                    vars_pbar.update(1)

                self.var_overall = self.star_points_eval.apply(lambda x: np.var(list(np.unique(x)), ddof=1))
                if self.report_verbose:
                    vars_pbar.update(1)

                self.gamma = tsvars_funcs.variogram(self.pair_df)
                if self.report_verbose:
                    vars_pbar.update(1)

                self.sec_covariogram = tsvars_funcs.cov_section(self.pair_df, self.mu_star_df)
                if self.report_verbose:
                    vars_pbar.update(1)

                self.morris = tsvars_funcs.morris_eq(self.pair_df)
                self.maee = self.morris[0]
                self.mee  = self.morris[1]
                if self.report_verbose:
                    vars_pbar.update(1)

                self.cov = tsvars_funcs.covariogram(self.pair_df, self.mu_overall)
                if self.report_verbose:
                    vars_pbar.update(1)

                self.ecov = tsvars_funcs.e_covariogram(self.sec_covariogram)
                if self.report_verbose:
                    vars_pbar.update(1)

                self.st = tsvars_funcs.sobol_eq(self.gamma, self.ecov, self.var_overall, self.delta_h)
                if self.report_verbose:
                    vars_pbar.update(1)

                self.ivars = pd.DataFrame.from_dict({scale: self.gamma.groupby(level=['ts', 'param']).apply(tsvars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
                      for scale in self.ivars_scales}, 'index')
                if self.report_verbose:
                    vars_pbar.update(1)

                self.run_status = True

                # output dictionary
                self.output = {
                    'Gamma':self.gamma,
                    'MAEE':self.maee,
                    'MEE':self.mee,
                    'COV':self.cov,
                    'ECOV':self.ecov,
                    'IVARS':self.ivars,
                    'IVARSid':self.ivars_scales,
                    # 'rnkST':self.st_factor_ranking,
                    # 'rnkIVARS':self.ivars_factor_ranking,
                    # 'Gammalb':self.gammalb if self.bootstrap_flag is True else None,
                    # 'Gammaub':self.gammaub if self.bootstrap_flag is True else None,
                    # 'STlb':self.stlb if self.bootstrap_flag is True else None,
                    # 'STub':self.stub if self.bootstrap_flag is True else None,
                    # 'IVARSlb':self.ivarslb if self.bootstrap_flag is True else None,
                    # 'IVARSub':self.ivarsub if self.bootstrap_flag is True else None,
                    # 'relST':self.rel_st_factor_ranking if self.bootstrap_flag is True else None,
                    # 'relIVARS':self.rel_ivars_factor_ranking if self.bootstrap_flag is True else None,
                    # 'Groups': [self.ivars50_grp, self.st_grp] if self.grouping_flag is True else None,
                    # 'relGrp': [self.reli_st_grp, self.reli_ivars50_grp] if self.grouping_flag is True else None,
                }


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
                    temp_sobol_values = tsvars_funcs.sobol_eq(self.gamma, self.ecov, self.var_overall, self.delta_h)
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
                self.sobol_value = tsvars_funcs.sobol_eq(gamma, ecov, var_overall)
                self.ivars = pd.DataFrame.from_dict({scale: self.variogram.groupby(level=['ts', 'param']).apply(tsvars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
                      for scale in self.ivars_scales}, 'index')

                self.run_status = True

    def _temp_func(func, name, group):
        return func(group), name

    def _applyParallel(dfGrouped, func):
        retLst, top_index = zip(*Parallel(n_jobs=multiprocessing.cpu_count())\
                                    (delayed(temp_func)(func, name, group)\
                                for name, group in dfGrouped))
        return pd.concat(retLst, keys=top_index)


class DVARS(VARS):
    def __init__(self, ):
        super().__init__()
